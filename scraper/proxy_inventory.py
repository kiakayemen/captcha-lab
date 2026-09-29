from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


INVENTORY_PATH = Path(__file__).resolve().parents[1] / "proxy_inventory.json"


def safe_proxy_endpoint(proxy_url: str) -> str:
    """Return a normalized proxy endpoint without embedded credentials."""
    parsed = urlsplit(proxy_url)
    host = parsed.hostname or "unknown"
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, "", "", ""))


@lru_cache(maxsize=1)
def load_proxy_inventory(path: Path = INVENTORY_PATH) -> tuple[dict[str, str], ...]:
    """Load named, credential-free identities for every owned proxy IP."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    routes = tuple(payload.get("routes", ()))
    required = {
        "name",
        "label",
        "provider",
        "resource",
        "endpoint",
        "expected_egress_ip",
        "url_env",
        "role",
    }
    names: set[str] = set()
    addresses: set[str] = set()
    for route in routes:
        missing = required - route.keys()
        if missing:
            raise ValueError(
                f"Proxy route is missing fields {sorted(missing)}: {route!r}"
            )
        if route["name"] in names:
            raise ValueError(f"Duplicate proxy route name: {route['name']}")
        if route["expected_egress_ip"] in addresses:
            raise ValueError(
                f"Duplicate owned IP: {route['expected_egress_ip']}"
            )
        names.add(route["name"])
        addresses.add(route["expected_egress_ip"])
    return routes


def proxy_identity_for_ip(address: str | None) -> dict[str, str] | None:
    if not address:
        return None
    return next(
        (
            route
            for route in load_proxy_inventory()
            if route["expected_egress_ip"] == address
        ),
        None,
    )


def proxy_identity_for_endpoint(endpoint: str | None) -> dict[str, str] | None:
    if not endpoint:
        return None
    normalized = safe_proxy_endpoint(endpoint)
    return next(
        (
            route
            for route in load_proxy_inventory()
            if route["endpoint"]
            and safe_proxy_endpoint(route["endpoint"]) == normalized
        ),
        None,
    )


def proxy_label_for_ip(address: str | None) -> str:
    identity = proxy_identity_for_ip(address)
    return identity["label"] if identity else (address or "Unknown IP")


def proxy_label_for_endpoint(endpoint: str | None) -> str:
    identity = proxy_identity_for_endpoint(endpoint)
    return identity["label"] if identity else (endpoint or "Unknown proxy")
