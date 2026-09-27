"""Check every configured proxy from inside a worker container.

This is a deliberately low-impact diagnostic: each proxy gets one egress-IP
lookup and one GET of the public BLS login page. It never submits credentials,
CAPTCHAs, or forms, and it never retries a request.

Run inside a worker pod:
    python scripts/check_proxy_ips.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = PROJECT_ROOT / "proxy_inventory.json"
sys.path.insert(0, str(PROJECT_ROOT))

from scraper.proxy import configured_proxy_urls, playwright_proxy_config  # noqa: E402


BLS_LOGIN_URL = "https://iran.blsspainglobal.com/Global/account/login"
EGRESS_IP_URL = "https://checkip.amazonaws.com/"
KNOWN_BLOCK_BODY_SHA256_16 = "8b73b6ccd7091d6d"


@dataclass
class ProbeResult:
    proxy_endpoint: str
    route_name: str | None = None
    provider: str | None = None
    expected_egress_ip: str | None = None
    egress_matches_expected: bool | None = None
    egress_ip: str | None = None
    egress_error: str | None = None
    http_status: int | None = None
    server: str | None = None
    body_length: int | None = None
    body_sha256_16: str | None = None
    known_403_fingerprint: bool = False
    verdict: str = "inconclusive"
    target_error: str | None = None


def safe_endpoint(proxy_url: str) -> str:
    """Return a proxy endpoint without embedded credentials."""
    parsed = urlsplit(proxy_url)
    host = parsed.hostname or "unknown"
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, "", "", ""))


def load_proxy_inventory(path: Path = INVENTORY_PATH) -> tuple[dict[str, str], ...]:
    """Load the credential-free, named proxy-route inventory."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    routes = tuple(payload.get("routes", ()))
    required = {
        "name",
        "provider",
        "endpoint",
        "expected_egress_ip",
        "url_env",
        "role",
    }
    names: set[str] = set()
    for route in routes:
        missing = required - route.keys()
        if missing:
            raise ValueError(
                f"Proxy route is missing fields {sorted(missing)}: {route!r}"
            )
        if route["name"] in names:
            raise ValueError(f"Duplicate proxy route name: {route['name']}")
        names.add(route["name"])
    return routes


def configured_inventory_routes(
    route_names: list[str],
) -> list[tuple[dict[str, str], str]]:
    """Resolve selected names to authenticated URLs stored in secrets."""
    inventory = {route["name"]: route for route in load_proxy_inventory()}
    unknown = [name for name in route_names if name not in inventory]
    if unknown:
        raise ValueError(f"Unknown proxy route(s): {', '.join(unknown)}")

    configured: list[tuple[dict[str, str], str]] = []
    for name in dict.fromkeys(route_names):
        route = inventory[name]
        proxy_url = os.getenv(route["url_env"], "").strip()
        if not proxy_url:
            raise ValueError(
                f"{name} requires secret environment variable {route['url_env']}."
            )
        if safe_endpoint(proxy_url) != route["endpoint"]:
            raise ValueError(
                f"{route['url_env']} points to {safe_endpoint(proxy_url)}, "
                f"but {name} is recorded as {route['endpoint']}."
            )
        configured.append((route, proxy_url))
    return configured


def print_inventory(*, as_json: bool = False) -> None:
    """Show route identities and secret readiness without revealing credentials."""
    for route in load_proxy_inventory():
        configured = bool(os.getenv(route["url_env"], "").strip())
        if as_json:
            print(json.dumps({
                **route,
                "credentials_configured": configured,
            }, sort_keys=True))
        else:
            state = "configured" if configured else "missing-secret"
            print(
                f"{route['name']} | {route['provider']} | "
                f"egress={route['expected_egress_ip']} | endpoint={route['endpoint']} | "
                f"role={route['role']} | credentials={state} ({route['url_env']})"
            )


def classify_response(status: int, server: str, body_sha256_16: str) -> tuple[str, bool]:
    known_fingerprint = (
        status == 403
        and server.lower() == "awselb/2.0"
        and body_sha256_16 == KNOWN_BLOCK_BODY_SHA256_16
    )
    if status == 403:
        return "blocked", known_fingerprint
    if 200 <= status < 400:
        return "accessible", False
    return f"http_{status}", False


def probe_proxy(playwright, proxy_url: str, timeout_ms: int) -> ProbeResult:
    result = ProbeResult(proxy_endpoint=safe_endpoint(proxy_url))
    launch_options: dict[str, object] = {
        "headless": True,
        "proxy": playwright_proxy_config(proxy_url),
    }
    executable_path = os.getenv("PLAYWRIGHT_EXECUTABLE_PATH", "").strip()
    if executable_path:
        launch_options["executable_path"] = executable_path

    browser = None
    context = None
    try:
        browser = playwright.chromium.launch(**launch_options)
        context = browser.new_context()

        try:
            response = context.request.get(EGRESS_IP_URL, timeout=timeout_ms)
            if response.ok:
                result.egress_ip = response.text().strip() or None
            else:
                result.egress_error = f"http_{response.status}"
        except PlaywrightTimeoutError:
            result.egress_error = "timeout"
        except Exception as error:
            result.egress_error = type(error).__name__

        page = context.new_page()
        try:
            response = page.goto(
                BLS_LOGIN_URL,
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            if response is None:
                result.target_error = "no_response"
                return result

            body = response.body()
            result.http_status = response.status
            result.server = response.headers.get("server")
            result.body_length = len(body)
            result.body_sha256_16 = hashlib.sha256(body).hexdigest()[:16]
            result.verdict, result.known_403_fingerprint = classify_response(
                response.status,
                result.server or "",
                result.body_sha256_16,
            )
        except PlaywrightTimeoutError:
            result.target_error = "timeout"
        except Exception as error:
            result.target_error = type(error).__name__
    except PlaywrightTimeoutError:
        result.target_error = "browser_launch_timeout"
    except Exception as error:
        result.target_error = f"browser_launch_{type(error).__name__}"
    finally:
        if context is not None:
            context.close()
        if browser is not None:
            browser.close()
    return result


def print_human(result: ProbeResult) -> None:
    fields = [
        result.verdict.upper(),
        result.proxy_endpoint,
        f"egress={result.egress_ip or result.egress_error or 'unknown'}",
        f"status={result.http_status if result.http_status is not None else 'none'}",
    ]
    if result.server:
        fields.append(f"server={result.server}")
    if result.body_sha256_16:
        fields.append(f"body_sha256_16={result.body_sha256_16}")
    if result.known_403_fingerprint:
        fields.append("known_403_fingerprint=yes")
    if result.target_error:
        fields.append(f"error={result.target_error}")
    print(" | ".join(fields), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test each SCRAPER_PROXY_URLS endpoint once from this worker.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Per-request timeout; requests are never retried (default: 30).",
    )
    parser.add_argument("--json", action="store_true", help="Print one JSON object per proxy.")
    parser.add_argument(
        "--list-routes",
        action="store_true",
        help="List named routes and whether their secret URL is configured.",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="Test one named inventory route; repeat to test several routes.",
    )
    args = parser.parse_args()
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be at least 1")

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.secrets")
    if args.list_routes:
        print_inventory(as_json=args.json)
        return 0

    try:
        if args.route:
            targets = configured_inventory_routes(args.route)
        else:
            targets = [(None, url) for url in configured_proxy_urls()]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    if not targets:
        print("SCRAPER_PROXY_URLS contains no proxy endpoints.", file=sys.stderr)
        return 2

    results: list[ProbeResult] = []
    with sync_playwright() as playwright:
        for route, proxy_url in targets:
            result = probe_proxy(playwright, proxy_url, args.timeout_seconds * 1000)
            if route is not None:
                result.route_name = route["name"]
                result.provider = route["provider"]
                result.expected_egress_ip = route["expected_egress_ip"]
                if result.egress_ip is not None:
                    result.egress_matches_expected = (
                        result.egress_ip == result.expected_egress_ip
                    )
                    if not result.egress_matches_expected:
                        result.verdict = "egress_mismatch"
            results.append(result)
            if args.json:
                print(json.dumps(asdict(result), sort_keys=True), flush=True)
            else:
                print_human(result)

    blocked = sum(result.verdict == "blocked" for result in results)
    accessible = sum(result.verdict == "accessible" for result in results)
    inconclusive = len(results) - blocked - accessible
    if args.json:
        print(json.dumps({
            "summary": {
                "tested": len(results),
                "accessible": accessible,
                "blocked": blocked,
                "inconclusive": inconclusive,
            }
        }, sort_keys=True), flush=True)
    else:
        print(
            f"Summary: tested={len(results)} accessible={accessible} "
            f"blocked={blocked} inconclusive={inconclusive}",
            flush=True,
        )
    if blocked:
        return 1
    if inconclusive:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
