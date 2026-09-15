from __future__ import annotations

import re

from playwright.sync_api import Page


HTTP_FORBIDDEN_PATTERN = re.compile(r"\b403\s+Forbidden\b", re.IGNORECASE)
HTTP_FORBIDDEN_MESSAGE = (
    "The target site returned HTTP 403 Forbidden. "
    "Stopping immediately; this failure is not retryable."
)


class HTTP403Forbidden(RuntimeError):
    pass


class ServerUnavailable(RuntimeError):
    """The target returned a server-side failure that must not be retried."""

    pass


def http_forbidden_page_visible(page: Page | None) -> bool:
    if page is None:
        return False
    response_state = getattr(
        page,
        "_captcha_lab_http_403_response",
        None,
    )
    if isinstance(response_state, dict):
        return True
    try:
        heading = page.get_by_role(
            "heading",
            name=HTTP_FORBIDDEN_PATTERN,
        ).first
        if heading.is_visible() is True:
            return True
    except Exception:
        pass
    try:
        body_text = page.locator("body").inner_text(timeout=1_000)
        return bool(HTTP_FORBIDDEN_PATTERN.search(body_text))
    except Exception:
        return False


def raise_for_http_forbidden(page: Page | None) -> None:
    if http_forbidden_page_visible(page):
        raise HTTP403Forbidden(HTTP_FORBIDDEN_MESSAGE)


def track_http_forbidden_responses(page: Page) -> None:
    """Remember workflow-level 403 responses for the next state check."""

    def remember(response) -> None:
        try:
            resource_type = response.request.resource_type
            if response.status == 403 and resource_type in {
                "document",
                "xhr",
                "fetch",
            }:
                page._captcha_lab_http_403_response = {
                    "url": response.url,
                    "resource_type": resource_type,
                    "response": response,
                    "diagnostics_recorded": False,
                }
        except Exception:
            return

    page.on("response", remember)


def http_forbidden_response_state(page: Page | None) -> dict | None:
    if page is None:
        return None
    state = getattr(page, "_captcha_lab_http_403_response", None)
    return state if isinstance(state, dict) else None
