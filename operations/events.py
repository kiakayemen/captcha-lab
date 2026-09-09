from __future__ import annotations

import re
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .models import ScraperEvent, ScraperRun


_event_context: ContextVar[tuple[str, uuid.UUID] | None] = ContextVar(
    "scraper_event_context",
    default=None,
)
_attempt_context: ContextVar[tuple[str, int] | None] = ContextVar(
    "scraper_attempt_context",
    default=None,
)

# Playwright's synchronous bridge can switch greenlet execution contexts while
# remaining on the same OS thread.  ContextVars do not necessarily follow
# that switch, so retain a thread-local fallback for this synchronous scraper
# execution path.  Celery's normal worker isolation keeps separate runs from
# sharing this state.
_thread_context = threading.local()


def _current_event_context() -> tuple[str, uuid.UUID] | None:
    return _event_context.get() or getattr(
        _thread_context,
        "event_context",
        None,
    )


def _current_attempt_context() -> tuple[str, int] | None:
    return _attempt_context.get() or getattr(
        _thread_context,
        "attempt_context",
        None,
    )


@contextmanager
def bind_scraper_event_context(
    run: ScraperRun,
) -> Iterator[uuid.UUID]:
    execution_id = uuid.uuid4()
    token = _event_context.set((str(run.pk), execution_id))
    previous_event_context = getattr(
        _thread_context,
        "event_context",
        None,
    )
    previous_attempt_context = getattr(
        _thread_context,
        "attempt_context",
        None,
    )
    _thread_context.event_context = (str(run.pk), execution_id)
    _thread_context.attempt_context = None
    try:
        yield execution_id
    finally:
        _event_context.reset(token)
        _thread_context.event_context = previous_event_context
        _thread_context.attempt_context = previous_attempt_context


def record_scraper_event(
    event_type: str,
    *,
    visa_sub_type: str = "",
    attempt_number: int | None = None,
    status: str = "",
    reason_code: str = "",
    duration_ms: int | None = None,
    message: str = "",
    data: dict | None = None,
) -> ScraperEvent | None:
    context = _current_event_context()
    if context is None:
        return None

    run_id, execution_id = context
    current_attempt = _current_attempt_context()
    if not visa_sub_type and current_attempt is not None:
        visa_sub_type = current_attempt[0]
    if attempt_number is None and current_attempt is not None:
        attempt_number = current_attempt[1]
    try:
        return ScraperEvent.objects.create(
            run_id=run_id,
            execution_id=execution_id,
            event_type=event_type,
            visa_sub_type=visa_sub_type,
            attempt_number=attempt_number,
            status=status,
            reason_code=reason_code,
            duration_ms=duration_ms,
            message=message,
            data=data or {},
        )
    except Exception:
        # Observability must never make the scraper fail.
        return None


def _first_int(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else None


def record_event_from_log(
    message: str,
) -> None:
    """Convert stable existing lifecycle messages into structured events.

    This intentionally acts as a compatibility bridge: existing text logs
    remain unchanged while the important facts become queryable immediately.
    """
    if message.startswith("Scraper run started."):
        return

    if message.startswith("Scraper run finished."):
        return

    if message.startswith("Scraper run crashed unexpectedly."):
        return

    if message.startswith("Starting fresh browser attempt."):
        subtype_match = re.search(r"Visa subtype=([^|]+)", message)
        attempt_match = re.search(r"Attempt=(\d+)/(\d+)", message)
        if subtype_match and attempt_match:
            attempt_context = (
                subtype_match.group(1).strip(),
                int(attempt_match.group(1)),
            )
            _attempt_context.set(attempt_context)
            _thread_context.attempt_context = attempt_context
        record_scraper_event(
            ScraperEvent.EventType.SUBTYPE_STARTED,
            visa_sub_type=subtype_match.group(1).strip() if subtype_match else "",
            attempt_number=int(attempt_match.group(1)) if attempt_match else None,
            data={
                "max_attempts": int(attempt_match.group(2))
                if attempt_match
                else None,
            },
            message=message,
        )
        return

    if message.startswith("Subtype attempt failed;"):
        subtype_match = re.search(r"Visa subtype=([^|]+)", message)
        attempt_match = re.search(r"Failed attempt=(\d+)/(\d+)", message)
        error_match = re.search(r"Error=([\s\S]+)", message)
        error_text = error_match.group(1).strip() if error_match else ""
        if "Login CAPTCHA was rejected" in error_text:
            reason_code = "LOGIN_CAPTCHA_REJECTED"
        elif "Login CAPTCHA outcome was unclear" in error_text:
            reason_code = "LOGIN_CAPTCHA_UNCLEAR"
        elif "Second CAPTCHA was not verified" in error_text:
            reason_code = "SECOND_CAPTCHA_NOT_VERIFIED"
        elif "Missing X server" in error_text:
            reason_code = "BROWSER_LAUNCH_NO_XSERVER"
        elif (
            "Locator expected to be visible" in error_text
            or "new-app-active" in error_text
        ):
            reason_code = "SITE_ERROR_NAVIGATION"
        else:
            reason_code = "SUBTYPE_ATTEMPT_FAILED"
        record_scraper_event(
            ScraperEvent.EventType.SUBTYPE_RETRY,
            visa_sub_type=subtype_match.group(1).strip() if subtype_match else "",
            attempt_number=int(attempt_match.group(1)) if attempt_match else None,
            reason_code=reason_code,
            message=message,
            data={"error": error_text},
        )
        return

    if message.startswith("Independent subtype check complete."):
        subtype_match = re.search(r"Visa subtype=([^|]+)", message)
        result_match = re.search(r"Result=([^|]+)", message)
        record_scraper_event(
            ScraperEvent.EventType.SUBTYPE_FINISHED,
            visa_sub_type=subtype_match.group(1).strip() if subtype_match else "",
            status=result_match.group(1).strip() if result_match else "",
            message=message,
        )
        return

    if message.startswith("Chromium browser launched."):
        record_scraper_event(
            ScraperEvent.EventType.BROWSER_STARTED,
            message=message,
        )
        return

    if message.startswith("Fresh browser attempt timed out."):
        record_scraper_event(
            ScraperEvent.EventType.BROWSER_FAILED,
            reason_code="PLAYWRIGHT_TIMEOUT",
            message=message,
        )
        return

    if message.startswith("Fresh browser attempt failed."):
        record_scraper_event(
            ScraperEvent.EventType.BROWSER_FAILED,
            reason_code="PLAYWRIGHT_ERROR",
            message=message,
        )
        return

    captcha_stage = None
    if message.startswith("Login CAPTCHA"):
        captcha_stage = "login"
    elif message.startswith("Second CAPTCHA"):
        captcha_stage = "second"

    if captcha_stage is not None:
        duration_match = re.search(r"(?:solved in|solve_time=)([0-9.]+)s", message)
        if "single-attempt mode" in message:
            event_type = ScraperEvent.EventType.CAPTCHA_STARTED
        elif "discovered" in message:
            return
        elif "verification succeeded" in message or 'returned "Verified!"' in message:
            event_type = ScraperEvent.EventType.CAPTCHA_FINISHED
        elif "solved in" in message or "decision:" in message:
            event_type = ScraperEvent.EventType.CAPTCHA_DECISION
        else:
            return
        record_scraper_event(
            event_type,
            duration_ms=(
                round(float(duration_match.group(1)) * 1000)
                if duration_match
                else None
            ),
            message=message,
            data={"stage": captcha_stage},
        )
