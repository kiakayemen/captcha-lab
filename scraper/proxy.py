from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from urllib.parse import unquote, urlsplit, urlunsplit


SUPPORTED_PROXY_SCHEMES = {
    "http",
    "https",
}
MINIMUM_PROXY_COUNT = 2


class ProxyConfigurationError(RuntimeError):
    pass


def configured_proxy_urls() -> tuple[str, ...]:
    """Return the proxy endpoints configured for browser attempts."""
    raw_value = os.getenv(
        "SCRAPER_PROXY_URLS",
        "",
    )

    values = (
        value.strip()
        for value in raw_value.replace(
            "\n",
            ",",
        ).split(",")
    )

    # Repeating an endpoint in the environment must not give that IP
    # additional weight in the rotation.
    return tuple(dict.fromkeys(
        value.rstrip("/") for value in values if value
    ))


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


@dataclass
class PlaywrightProxyRotator:
    """Choose every configured proxy once per randomly shuffled cycle."""

    proxy_urls: tuple[str, ...] = field(
        default_factory=configured_proxy_urls
    )
    _remaining: list[str] = field(
        default_factory=list,
        init=False,
    )
    _last_url: str | None = field(
        default=None,
        init=False,
    )

    def validate_required_pool(self, *, allow_single_proxy: bool = False) -> None:
        minimum = 1 if allow_single_proxy else MINIMUM_PROXY_COUNT
        if len(self.proxy_urls) < minimum:
            raise ProxyConfigurationError(
                f"SCRAPER_PROXY_URLS must contain at least {minimum} distinct proxy "
                f"endpoints; found {len(self.proxy_urls)}. Direct egress is disabled."
            )

    def choose(self) -> dict[str, str] | None:
        if not self.proxy_urls:
            return None

        if not self._remaining:
            self._remaining = list(self.proxy_urls)
            random.shuffle(self._remaining)

            # A cycle boundary must not immediately reuse the proxy that
            # ended the previous cycle when another endpoint is available.
            if (
                len(self._remaining) > 1
                and self._remaining[0] == self._last_url
            ):
                self._remaining[0], self._remaining[1] = (
                    self._remaining[1],
                    self._remaining[0],
                )

        selected_url = self._remaining.pop(0)
        self._last_url = selected_url
        return playwright_proxy_config(selected_url)


def choose_playwright_proxy() -> dict[str, str] | None:
    """Choose a proxy for callers that need a single browser."""
    return PlaywrightProxyRotator().choose()
