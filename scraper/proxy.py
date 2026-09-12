from __future__ import annotations

import os
import random
from urllib.parse import unquote, urlsplit, urlunsplit


SUPPORTED_PROXY_SCHEMES = {
    "http",
    "https",
}


def configured_proxy_urls() -> tuple[str, ...]:
    """Return the proxy endpoints configured for browser attempts."""
    raw_value = os.getenv(
        "SCRAPER_PROXY_URLS",
        "",
    )

    return tuple(
        value.strip()
        for value in raw_value.replace(
            "\n",
            ",",
        ).split(",")
        if value.strip()
    )


def playwright_proxy_config(
    proxy_url: str,
) -> dict[str, str]:
    """Convert one proxy URL into Playwright's launch configuration."""
    parsed = urlsplit(proxy_url)

    if parsed.scheme not in SUPPORTED_PROXY_SCHEMES:
        raise ValueError(
            "Proxy URLs must start with http:// or https://."
        )

    if parsed.hostname is None:
        raise ValueError(
            f"Proxy URL has no hostname: {proxy_url!r}"
        )

    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(
            "Proxy URLs must contain only scheme, hostname, port, "
            "and optional credentials."
        )

    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"

    if parsed.port is not None:
        host = f"{host}:{parsed.port}"

    result = {
        "server": urlunsplit(
            (
                parsed.scheme,
                host,
                "",
                "",
                "",
            )
        )
    }

    if parsed.username is not None:
        result["username"] = unquote(
            parsed.username
        )

    if parsed.password is not None:
        result["password"] = unquote(
            parsed.password
        )

    return result


def choose_playwright_proxy() -> dict[str, str] | None:
    """Randomly choose one configured proxy for a fresh browser."""
    proxy_urls = configured_proxy_urls()

    if not proxy_urls:
        return None

    return playwright_proxy_config(
        random.choice(proxy_urls)
    )
