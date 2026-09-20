from __future__ import annotations

import json
import hashlib
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    expect,
    sync_playwright,
)

from captcha_solver import (
    print_decision,
    solve_captcha_image,
    write_outputs,
)
from config import (
    BLS_EMAIL,
    BLS_PASSWORD,
    LOGIN_URL,
)
from flows.appointment_flow import (
    appointment_available_dialog_visible,
    fill_appointment_form,
    no_appointments_dialog_visible,
)
from flows.captcha_flow import (
    appointment_form_ready,
    blocking_overlay_visible,
    captcha_instruction_present,
    click_background_submit,
    click_nav_book_new_appointment,
    click_ok_dialog,
    click_selected_captcha_tiles,
    click_submit_selection,
    click_verify_selection,
    disclaimer_dialog_visible,
    find_true_captcha_label,
    find_true_captcha_label_in_scope,
    get_verify_selection_frame,
    login_captcha_invalid,
    login_captcha_succeeded,
    save_captcha_crop,
    site_error_page_visible,
    wait_for_captcha_tiles_ready,
)
from flows.login_flow import (
    fill_visible_password,
    submit_email,
)
from flows.errors import (
    HTTP403Forbidden,
    HTTP_FORBIDDEN_MESSAGE,
    http_forbidden_page_visible,
    http_forbidden_response_state,
    raise_for_http_forbidden,
    track_http_forbidden_responses,
)
from flows.selectors import (
    BACKGROUND_SUBMIT_BUTTON_SELECTOR,
    CAPTCHA_INSTRUCTION_PATTERN,
    CAPTCHA_TILE_SELECTOR,
)
from notifications import (
    log_no_appointment,
    notify_admin,
)
from operations.events import record_scraper_event
from operations.models import ScraperEvent
from ocr import get_reader

from scraper.models import (
    ScraperConfig,
    ScraperResult,
    ScraperStatus,
)
from scraper.http_diagnostics import (
    resolve_egress_ip,
    response_diagnostics,
)
from scraper.proxy import PlaywrightProxyRotator, ProxyConfigurationError


logger = logging.getLogger(
    "captcha_lab"
)


# A subtype gets this many completely fresh browser attempts.
#
# We deliberately do NOT retry forever. If BLS changes or is down,
# an infinite unattended browser loop would be dangerous.
MAX_SUBTYPE_ATTEMPTS = 5
SUBTYPE_RETRY_BACKOFF_SECONDS = (
    30,
    60,
    120,
    180,
)
LOGIN_CAPTCHA_OUTCOME_TIMEOUT_SECONDS = 15
MAX_LOGIN_CAPTCHA_ATTEMPTS = 3
MAX_SECOND_CAPTCHA_ATTEMPTS = 3
SECOND_CAPTCHA_RETRY_SETTLE_MS = 3_000
SECOND_CAPTCHA_VERIFICATION_TIMEOUT_SECONDS = 25
SECOND_CAPTCHA_LOADING_GRACE_SECONDS = 15
SECOND_CAPTCHA_REGENERATION_TIMEOUT_SECONDS = 10
RECOVERABLE_UNCLEAR_LOGIN_PATHS = {
    "/",
    "/global",
    "/global/",
    "/global/home/index",
    "/global/newcaptcha/logincaptchasubmit",
}


class ScraperStopRequested(RuntimeError):
    pass


class SecondCaptchaUnconfirmed(RuntimeError):
    """Submission did not produce a conclusive response or a fresh challenge."""


class AppointmentResultUnconfirmed(RuntimeError):
    """The form returned neither an explicit available nor unavailable result."""


def check_stop_requested(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise ScraperStopRequested("Scraper stop was requested by an operator.")


def interruptible_cooldown(
    seconds: int,
    should_stop: Callable[[], bool] | None,
) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check_stop_requested(should_stop)
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def subtype_retry_delay_seconds(attempt_number: int) -> int:
    """Return an increasing cooldown with bounded random jitter."""
    if attempt_number < 1 or attempt_number >= MAX_SUBTYPE_ATTEMPTS:
        return 0
    base_delay = SUBTYPE_RETRY_BACKOFF_SECONDS[attempt_number - 1]
    return random.randint(
        round(base_delay * 0.8),
        round(base_delay * 1.2),
    )


def inspect_page_state(page) -> dict[str, bool]:
    """Check known terminal and valid UI states before retrying."""
    if page is None:
        return {}
    checks = {
        "http_403": lambda: http_forbidden_page_visible(page),
        "server_error": lambda: site_error_page_visible(page),
        "no_appointment": lambda: no_appointments_dialog_visible(page),
        "appointment_form": lambda: appointment_form_ready(page),
        "verified": lambda: page.get_by_text("Verified!", exact=True).first.is_visible(),
        "second_captcha_popup": lambda: page.locator(
            "div.k-widget.k-window"
        ).filter(has_text="Verify Selection").first.is_visible(),
        "disclaimer_ok": lambda: disclaimer_dialog_visible(page),
        "background_submit": lambda: page.locator(
            BACKGROUND_SUBMIT_BUTTON_SELECTOR
        ).last.is_visible(),
    }
    state: dict[str, bool] = {}
    for name, check in checks.items():
        try:
            state[name] = check() is True
        except Exception:
            state[name] = False
    return state


def wait_for_form_result(
    page,
    *,
    timeout_seconds: int = 30,
    should_stop: Callable[[], bool] | None = None,
) -> ScraperStatus:
    """Keep an inconclusive form response distinct from confirmed availability."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        check_stop_requested(should_stop)
        raise_for_http_forbidden(page)
        if site_error_page_visible(page):
            raise RuntimeError(
                "Target site returned its temporary processing-error page."
            )
        if "/account/login" in str(page.url).lower():
            raise RuntimeError(
                "Form submission returned to login before an appointment result."
            )
        if no_appointments_dialog_visible(page):
            return ScraperStatus.NO_APPOINTMENT
        if appointment_available_dialog_visible(page):
            return ScraperStatus.APPOINTMENT_FOUND
        page.wait_for_timeout(250)
    check_stop_requested(should_stop)
    raise_for_http_forbidden(page)
    if site_error_page_visible(page):
        raise RuntimeError(
            "Target site returned its temporary processing-error page."
        )
    if "/account/login" in str(page.url).lower():
        raise RuntimeError(
            "Form submission returned to login before an appointment result."
        )
    if no_appointments_dialog_visible(page):
        return ScraperStatus.NO_APPOINTMENT
    if appointment_available_dialog_visible(page):
        return ScraperStatus.APPOINTMENT_FOUND
    raise AppointmentResultUnconfirmed(
        "Appointment form returned neither an explicit available nor "
        "unavailable result within 30 seconds."
    )


def record_detected_http_403(
    *,
    page,
    context,
    egress_ip: str | None,
    egress_ip_hash: str | None,
    egress_lookup_error: str | None,
    proxy_endpoint: str | None,
    login_requested_at: float | None,
) -> None:
    state = http_forbidden_response_state(page)
    if not state or state.get("diagnostics_recorded"):
        return
    response = state.get("response")
    if response is None or context is None:
        return
    data = response_diagnostics(
        response=response,
        context=context,
        account=BLS_EMAIL,
        egress_ip=egress_ip,
        egress_ip_hash=egress_ip_hash,
        egress_lookup_error=egress_lookup_error,
        proxy_endpoint=proxy_endpoint,
        seconds_since_previous_login=(
            time.monotonic() - login_requested_at
            if login_requested_at is not None
            else None
        ),
    )
    (
        failure_egress_ip,
        failure_egress_ip_hash,
        failure_egress_lookup_error,
    ) = resolve_egress_ip(context)
    data["egress_ip_at_failure"] = failure_egress_ip
    data["egress_ip_at_failure_observation"] = "independent_ipify_probe"
    data["egress_ip_hash_at_failure"] = failure_egress_ip_hash
    data["egress_lookup_error_at_failure"] = failure_egress_lookup_error
    data["response_url"] = state.get("url")
    data["response_observed_at"] = state.get("observed_at")
    data["subsequent_403s"] = state.get("subsequent_403s", [])
    data["resource_type"] = state.get("resource_type")
    record_scraper_event(
        ScraperEvent.EventType.LOGIN_RESPONSE,
        status="403",
        reason_code="HTTP_403_WORKFLOW",
        data=data,
    )
    logger.error(
        "Workflow HTTP 403 response diagnostics=%s",
        json.dumps(data, sort_keys=True),
    )
    state["diagnostics_recorded"] = True


def failure_record(
    result: ScraperResult,
    *,
    visa_sub_type: str,
    attempt_number: int,
) -> dict[str, object]:
    return {
        "visa_sub_type": visa_sub_type,
        "attempt_number": attempt_number,
        "status": result.status.value,
        "error_type": result.error_type or "",
        "error_message": result.error_message or "",
        "page_url": result.page_url or "",
        "failure_screenshot": (
            str(result.failure_screenshot) if result.failure_screenshot else ""
        ),
        "occurred_at": result.finished_at.isoformat(),
    }


def wait_for_login_captcha_outcome(page) -> str:
    """Wait for rejection or successful navigation after CAPTCHA submit."""
    deadline = time.monotonic() + LOGIN_CAPTCHA_OUTCOME_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        if http_forbidden_page_visible(page):
            raise HTTP403Forbidden(HTTP_FORBIDDEN_MESSAGE)
        if site_error_page_visible(page):
            raise RuntimeError(
                "Target site returned its temporary processing-error page."
            )
        if login_captcha_invalid(page):
            return "rejected"
        if login_captcha_succeeded(page):
            return "succeeded"
        page.wait_for_timeout(250)

    if captcha_instruction_present(page):
        return "instruction_present"
    return "unclear"


def log_captcha_decision(stage: str, decision) -> None:
    """Temporary detailed CAPTCHA diagnostics for live scraper runs."""
    record_scraper_event(
        ScraperEvent.EventType.CAPTCHA_DECISION,
        status=str(decision.status),
        data={
            "stage": stage,
            "target": decision.target,
            "selected_tiles": list(decision.selected_tiles),
            "uncertain_tiles": list(decision.uncertain_tiles),
            "tiles": [
                {
                    "tile": tile.tile,
                    "prediction": tile.prediction or "",
                    "score": float(tile.score),
                    "votes": tile.votes,
                    "matches_target": tile.matches_target,
                    "attempts": [
                        {
                            "variant": attempt["variant"],
                            "prediction": attempt["prediction"] or "",
                            "confidence": float(attempt["confidence"]),
                        }
                        for attempt in tile.attempts
                    ],
                }
                for tile in decision.tiles
            ],
        },
    )
    logger.info(
        "%s decision: status=%s target=%s selected=%s uncertain=%s",
        stage,
        decision.status,
        decision.target,
        list(decision.selected_tiles),
        list(decision.uncertain_tiles),
    )
    for tile in decision.tiles:
        attempts = ", ".join(
            f"{attempt['variant']}={attempt['prediction'] or '-'}"
            f"({float(attempt['confidence']):.3f})"
            for attempt in tile.attempts
        )
        logger.info(
            "%s tile=%s final=%s score=%.3f votes=%s match=%s | %s",
            stage,
            tile.tile,
            tile.prediction or "-",
            tile.score,
            tile.votes,
            tile.matches_target,
            attempts,
        )


def record_captcha_stage(
    *,
    captcha: str,
    stage: str,
    attempt_number: int,
    started_at: datetime,
    duration_ms: int,
    status: str = "completed",
    finished_at: datetime | None = None,
) -> None:
    finished_at = finished_at or datetime.now(timezone.utc)
    data = {
        "captcha": captcha,
        "stage": stage,
        "attempt_number": attempt_number,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
    }
    record_scraper_event(
        ScraperEvent.EventType.CAPTCHA_STAGE,
        attempt_number=attempt_number,
        status=status,
        duration_ms=max(0, duration_ms),
        data=data,
    )
    logger.info(
        "CAPTCHA telemetry: captcha=%s stage=%s attempt=%s "
        "started_at=%s finished_at=%s duration_ms=%s status=%s",
        captcha,
        stage,
        attempt_number,
        data["started_at"],
        data["finished_at"],
        max(0, duration_ms),
        status,
    )
def run_login_step(page) -> None:
    logger.info(
        "Submitting login email."
    )

    submit_email(
        page,
        BLS_EMAIL,
    )

    page.wait_for_load_state(
        "domcontentloaded"
    )

    logger.info(
        "Login email submitted successfully."
    )


def save_live_attempt_bundle(
    *,
    output_dir: Path,
    step_name: str,
    attempt_number: int,
    page_url: str,
    target: str,
    decision,
    captcha_image,
    debug_image,
    tiles,
) -> Path:
    attempt_dir = (
        output_dir
        / step_name
        / f"attempt_{attempt_number:02d}"
    )

    write_outputs(
        attempt_dir,
        captcha_image,
        debug_image,
        tiles,
        decision,
    )

    metadata = {
        "step": step_name,
        "attempt": attempt_number,
        "page_url": page_url,
        "target": target,
        "selected_tiles": list(
            decision.selected_tiles
        ),
        "uncertain_tiles": list(
            decision.uncertain_tiles
        ),
        "status": decision.status,
    }

    metadata_path = (
        attempt_dir
        / "live_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved CAPTCHA attempt artifacts. "
        "Step=%s | Attempt=%s | Directory=%s",
        step_name,
        attempt_number,
        attempt_dir,
    )

    return attempt_dir


def run_captcha_step(
    page,
    *,
    gpu: bool,
    output_dir: Path,
    reader,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Retry rejected login CAPTCHAs without discarding the session."""
    for attempt_number in range(1, MAX_LOGIN_CAPTCHA_ATTEMPTS + 1):
        check_stop_requested(should_stop)
        logger.info(
            "Login CAPTCHA attempt %s/%s.",
            attempt_number,
            MAX_LOGIN_CAPTCHA_ATTEMPTS,
        )
        outcome = _run_login_captcha_attempt(
            page,
            gpu=gpu,
            output_dir=output_dir,
            reader=reader,
            attempt_number=attempt_number,
        )
        if outcome == "succeeded":
            return
        if outcome == "unclear":
            if attempt_number == MAX_LOGIN_CAPTCHA_ATTEMPTS:
                raise RuntimeError(
                    "Login CAPTCHA remained unclear after "
                    f"{MAX_LOGIN_CAPTCHA_ATTEMPTS} same-session attempts."
                )
            if not restart_unclear_login_captcha(page):
                raise RuntimeError(
                    "Login CAPTCHA outcome was unclear on an unknown page state."
                )
        if attempt_number < MAX_LOGIN_CAPTCHA_ATTEMPTS:
            logger.warning(
                "Login CAPTCHA was rejected or regenerated; re-solving it "
                "in the current browser session."
            )
            page.wait_for_timeout(SECOND_CAPTCHA_RETRY_SETTLE_MS)

    raise RuntimeError(
        "Login CAPTCHA was rejected "
        f"{MAX_LOGIN_CAPTCHA_ATTEMPTS} times in the same browser session."
    )


def restart_unclear_login_captcha(page) -> bool:
    """Restart login inside the existing browser for known unclear pages."""
    path = urlparse(str(page.url)).path.lower()
    if path not in RECOVERABLE_UNCLEAR_LOGIN_PATHS:
        return False
    if http_forbidden_page_visible(page):
        raise HTTP403Forbidden(HTTP_FORBIDDEN_MESSAGE)
    if site_error_page_visible(page):
        raise RuntimeError(
            "Target site returned its temporary processing-error page."
        )

    logger.warning(
        "Login CAPTCHA ended on known unclear endpoint %s; restarting the "
        "login challenge in the current browser session.",
        path,
    )
    page.goto(
        LOGIN_URL,
        wait_until="domcontentloaded",
        timeout=60_000,
    )
    run_login_step(page)
    return True


def _run_login_captcha_attempt(
    page,
    *,
    gpu: bool,
    output_dir: Path,
    reader,
    attempt_number: int,
) -> str:
    """Solve one rendered login CAPTCHA and classify its resulting state."""

    screenshot_path = Path(
        "captcha_page.png"
    )

    fill_password(
        page,
        target_password=BLS_PASSWORD,
    )

    logger.info(
        "Waiting for login CAPTCHA instruction."
    )

    (
        _true_label,
        true_label_id,
        target,
    ) = find_true_captcha_label(
        page
    )

    true_label_by_id = page.locator(
        f"#{true_label_id}"
    )

    expect(
        true_label_by_id
    ).to_have_count(
        1
    )

    expect(
        true_label_by_id
    ).to_be_visible()

    confirmed_text = (
        true_label_by_id
        .inner_text()
        .strip()
    )

    confirmed_match = (
        CAPTCHA_INSTRUCTION_PATTERN
        .fullmatch(
            confirmed_text
        )
    )

    if (
        confirmed_match is None
        or confirmed_match.group(1)
        != target
    ):
        raise RuntimeError(
            "The discovered CAPTCHA label "
            "changed before solving."
        )

    logger.info(
        "Login CAPTCHA discovered. "
        "Label ID=%s | Target=%s",
        true_label_id,
        target,
    )

    #
    # The DOM tile resolver already requires exactly nine
    # visible, deduplicated CAPTCHA tiles.
    #
    # Calling it before the screenshot prevents us from
    # snapshotting a partially rendered grid.
    #
    logger.info(
        "Waiting for all 9 login CAPTCHA tiles."
    )

    wait_for_captcha_tiles_ready(
        page,
        page,
    )

    #
    # Give Chromium a tiny amount of time to paint the completed
    # grid after the ninth tile becomes available.
    #
    page.wait_for_timeout(
        500
    )

    logger.info(
        "Login CAPTCHA grid ready. "
        "Taking solver screenshot."
    )

    acquisition_started_at = datetime.now(timezone.utc)
    acquisition_started = time.perf_counter()
    captcha_image = save_captcha_crop(
        page,
        screenshot_path,
    )
    record_captcha_stage(
        captcha="login",
        stage="image_acquisition",
        attempt_number=attempt_number,
        started_at=acquisition_started_at,
        duration_ms=round((time.perf_counter() - acquisition_started) * 1000),
    )

    logger.info(
        "Saved cropped login CAPTCHA image: %s",
        screenshot_path,
    )

    solve_start = (
        time.perf_counter()
    )

    solver_timings: dict[str, object] = {}
    (
        decision,
        tiles,
        _boxes,
        debug,
    ) = solve_captcha_image(
        captcha_image,
        target=target,
        reader=reader,
        timings=solver_timings,
    )

    solve_seconds = (
        time.perf_counter()
        - solve_start
    )
    record_captcha_stage(
        captcha="login",
        stage="ocr_inference",
        attempt_number=attempt_number,
        started_at=datetime.fromisoformat(str(solver_timings["ocr_started_at"])),
        finished_at=datetime.fromisoformat(str(solver_timings["ocr_finished_at"])),
        duration_ms=int(solver_timings["ocr_duration_ms"]),
    )

    logger.info(
        "Login CAPTCHA image shape=%s | extracted tiles=%s | solve_time=%.3fs",
        getattr(captcha_image, "shape", None),
        len(tiles),
        solve_seconds,
    )
    log_captcha_decision("Login CAPTCHA", decision)

    logger.info(
        "Login CAPTCHA solved in %.3fs. "
        "Status=%s | Selected tiles=%s | "
        "Uncertain tiles=%s",
        solve_seconds,
        decision.status,
        list(
            decision.selected_tiles
        ),
        list(
            decision.uncertain_tiles
        ),
    )

    print_decision(
        decision
    )

    selection_started_at = datetime.now(timezone.utc)
    selection_started = time.perf_counter()
    click_selected_captcha_tiles(
        page,
        decision.selected_tiles,
    )
    record_captcha_stage(
        captcha="login",
        stage="box_selection",
        attempt_number=attempt_number,
        started_at=selection_started_at,
        duration_ms=round((time.perf_counter() - selection_started) * 1000),
    )

    logger.info(
        "Selected login CAPTCHA tiles=%s",
        list(
            decision.selected_tiles
        ),
    )

    submission_started_at = datetime.now(timezone.utc)
    submission_started = time.perf_counter()
    click_verify_selection(
        page
    )
    record_captcha_stage(
        captcha="login",
        stage="submission",
        attempt_number=attempt_number,
        started_at=submission_started_at,
        duration_ms=round((time.perf_counter() - submission_started) * 1000),
    )

    logger.info(
        "Submitted login CAPTCHA selection."
    )

    save_live_attempt_bundle(
        output_dir=output_dir,
        step_name="login_captcha",
        attempt_number=attempt_number,
        page_url=page.url,
        target=target,
        decision=decision,
        captcha_image=captcha_image,
        debug_image=debug,
        tiles=tiles,
    )

    verification_started_at = datetime.now(timezone.utc)
    verification_started = time.perf_counter()
    outcome = wait_for_login_captcha_outcome(page)
    record_captcha_stage(
        captcha="login",
        stage="verification",
        attempt_number=attempt_number,
        started_at=verification_started_at,
        duration_ms=round((time.perf_counter() - verification_started) * 1000),
        status=outcome,
    )

    if outcome == "rejected":
        return outcome

    if outcome == "succeeded":
        logger.info(
            "Login CAPTCHA verification succeeded."
        )

        write_outputs(
            output_dir,
            captcha_image,
            debug,
            tiles,
            decision,
        )

        return outcome

    if outcome == "instruction_present":
        return outcome

    return outcome


def run_second_captcha_step(
    page,
    *,
    gpu: bool,
    output_dir: Path,
    reader,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Solve a regenerated Verify Selection CAPTCHA in the same session."""
    for attempt_number in range(1, MAX_SECOND_CAPTCHA_ATTEMPTS + 1):
        check_stop_requested(should_stop)
        logger.info(
            "Second CAPTCHA attempt %s/%s.",
            attempt_number,
            MAX_SECOND_CAPTCHA_ATTEMPTS,
        )

        try:
            verified = _run_second_captcha_attempt(
                page,
                gpu=gpu,
                output_dir=output_dir,
                reader=reader,
                attempt_number=attempt_number,
            )
        except HTTP403Forbidden:
            raise
        except SecondCaptchaUnconfirmed:
            page_state = inspect_page_state(page)
            if page_state.get("http_403"):
                raise HTTP403Forbidden(HTTP_FORBIDDEN_MESSAGE)
            if not page_state.get("second_captcha_popup") and any(page_state.get(name) for name in (
                "verified", "disclaimer_ok", "appointment_form"
            )):
                logger.info(
                    "Second CAPTCHA advanced after its verification deadline. "
                    "State=%s", page_state,
                )
                return
            raise
        except (PlaywrightTimeoutError, AssertionError, RuntimeError) as error:
            page_state = inspect_page_state(page)
            if page_state.get("http_403"):
                raise HTTP403Forbidden(HTTP_FORBIDDEN_MESSAGE) from error
            if page_state.get("server_error"):
                raise
            if not page_state.get("second_captcha_popup") and any(
                page_state.get(name)
                for name in (
                    "verified",
                    "disclaimer_ok",
                    "appointment_form",
                )
            ):
                logger.info(
                    "Second CAPTCHA click reached a valid downstream state; "
                    "continuing without a browser retry. State=%s",
                    page_state,
                )
                return
            if attempt_number == MAX_SECOND_CAPTCHA_ATTEMPTS:
                raise RuntimeError(
                    "Second CAPTCHA could not be prepared after "
                    f"{MAX_SECOND_CAPTCHA_ATTEMPTS} same-session attempts."
                ) from error
            logger.warning(
                "Second CAPTCHA UI was incomplete; reloading the appointment "
                "verification page and retrying in the same session. Error=%s",
                error,
            )
            page.reload(wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_timeout(SECOND_CAPTCHA_RETRY_SETTLE_MS)
            click_verify_selection(page)
            continue
        if verified:
            return

        if attempt_number == MAX_SECOND_CAPTCHA_ATTEMPTS:
            break

        logger.warning(
            "Second CAPTCHA was rejected and regenerated; "
            "re-solving it in the current browser session."
        )
        page.wait_for_timeout(SECOND_CAPTCHA_RETRY_SETTLE_MS)

    raise RuntimeError(
        "Second CAPTCHA was rejected "
        f"{MAX_SECOND_CAPTCHA_ATTEMPTS} times in the same browser session."
    )


def classify_second_captcha_state(page, verified_label, invalid_label, popup) -> str:
    """Prioritize HTTP failure and explicit site feedback over popup visibility."""
    if http_forbidden_page_visible(page):
        raise HTTP403Forbidden(HTTP_FORBIDDEN_MESSAGE)
    if site_error_page_visible(page):
        raise RuntimeError(
            "Target site returned its temporary processing-error page."
        )
    if verified_label.is_visible():
        return "verified"
    if invalid_label.is_visible():
        return "explicitly_rejected"
    if blocking_overlay_visible(page):
        return "loading"
    if not popup.is_visible():
        page_state = inspect_page_state(page)
        if any(page_state.get(name) for name in (
            "disclaimer_ok", "appointment_form"
        )):
            return "advanced"
    return "pending"


def second_captcha_challenge_signature(frame) -> str | None:
    """Fingerprint the rendered grid without storing the CAPTCHA images."""
    try:
        sources = frame.locator(CAPTCHA_TILE_SELECTOR).evaluate_all(
            "images => images.map(image => image.currentSrc || image.src || '')"
        )
        if len(sources) != 9 or not all(sources):
            return None
        return hashlib.sha256(
            json.dumps(sources, separators=(",", ":")).encode()
        ).hexdigest()[:16]
    except Exception:
        return None


def wait_for_regenerated_second_captcha(
    page, frame, old_signature: str | None, invalid_label=None
) -> bool:
    """Retry in-session only when a different complete grid is visible."""
    if old_signature is None:
        raise SecondCaptchaUnconfirmed(
            "Explicit CAPTCHA rejection was shown, but the original grid "
            "could not be fingerprinted."
        )
    deadline = time.monotonic() + SECOND_CAPTCHA_REGENERATION_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if http_forbidden_page_visible(page):
            raise HTTP403Forbidden(HTTP_FORBIDDEN_MESSAGE)
        if site_error_page_visible(page):
            raise RuntimeError(
                "Target site returned its temporary processing-error page."
            )
        signature = second_captcha_challenge_signature(frame)
        invalid_cleared = invalid_label is None or invalid_label.is_visible() is False
        if signature is not None and signature != old_signature and invalid_cleared:
            logger.info("A new complete second-CAPTCHA grid is visible; retrying in-session.")
            return True
        page.wait_for_timeout(250)
    raise SecondCaptchaUnconfirmed(
        "Explicit CAPTCHA rejection was shown, but no new complete grid "
        "appeared within 10 seconds."
    )


def _run_second_captcha_attempt(
    page,
    *,
    gpu: bool,
    output_dir: Path,
    reader,
    attempt_number: int,
) -> bool:
    """Solve one rendered Verify Selection challenge."""

    frame = get_verify_selection_frame(
        page
    )

    screenshot_path = Path(
        "captcha_page_2.png"
    )

    popup = (
        page.locator(
            "div.k-widget.k-window"
        )
        .filter(
            has_text="Verify Selection"
        )
        .first
    )

    logger.info(
        "Waiting for second CAPTCHA instruction."
    )

    (
        _true_label,
        true_label_id,
        target,
    ) = find_true_captcha_label_in_scope(
        frame
    )

    true_label_by_id = (
        frame.locator(
            f"#{true_label_id}"
        )
    )

    expect(
        true_label_by_id
    ).to_have_count(
        1
    )

    expect(
        true_label_by_id
    ).to_be_visible()

    confirmed_text = (
        true_label_by_id
        .inner_text()
        .strip()
    )

    confirmed_match = (
        CAPTCHA_INSTRUCTION_PATTERN
        .fullmatch(
            confirmed_text
        )
    )

    if (
        confirmed_match is None
        or confirmed_match.group(1)
        != target
    ):
        raise RuntimeError(
            "The discovered second CAPTCHA "
            "label changed before solving."
        )

    logger.info(
        "Second CAPTCHA discovered. "
        "Label ID=%s | Target=%s",
        true_label_id,
        target,
    )

    logger.info(
        "Waiting for all 9 second CAPTCHA tiles."
    )

    tiles_in_frame = wait_for_captcha_tiles_ready(
        page,
        frame,
    )

    page.wait_for_timeout(
        500
    )

    logger.info(
        "Second CAPTCHA grid ready. "
        "Taking solver screenshot."
    )

    acquisition_started_at = datetime.now(timezone.utc)
    acquisition_started = time.perf_counter()
    captcha_image = save_captcha_crop(
        page,
        screenshot_path,
    )
    record_captcha_stage(
        captcha="second",
        stage="image_acquisition",
        attempt_number=attempt_number,
        started_at=acquisition_started_at,
        duration_ms=round((time.perf_counter() - acquisition_started) * 1000),
    )

    logger.info(
        "Saved cropped second CAPTCHA image: %s",
        screenshot_path,
    )

    solve_start = (
        time.perf_counter()
    )

    solver_timings: dict[str, object] = {}
    (
        decision,
        tiles,
        _boxes,
        debug,
    ) = solve_captcha_image(
        captcha_image,
        target=target,
        reader=reader,
        timings=solver_timings,
    )

    solve_seconds = (
        time.perf_counter()
        - solve_start
    )
    record_captcha_stage(
        captcha="second",
        stage="ocr_inference",
        attempt_number=attempt_number,
        started_at=datetime.fromisoformat(str(solver_timings["ocr_started_at"])),
        finished_at=datetime.fromisoformat(str(solver_timings["ocr_finished_at"])),
        duration_ms=int(solver_timings["ocr_duration_ms"]),
    )

    logger.info(
        "Second CAPTCHA image shape=%s | extracted tiles=%s | solve_time=%.3fs",
        getattr(captcha_image, "shape", None),
        len(tiles),
        solve_seconds,
    )
    log_captcha_decision("Second CAPTCHA", decision)

    logger.info(
        "Second CAPTCHA solved in %.3fs. "
        "Status=%s | Selected tiles=%s | "
        "Uncertain tiles=%s",
        solve_seconds,
        decision.status,
        list(
            decision.selected_tiles
        ),
        list(
            decision.uncertain_tiles
        ),
    )

    print_decision(
        decision
    )

    submitted_challenge_signature = second_captcha_challenge_signature(frame)
    selection_started_at = datetime.now(timezone.utc)
    selection_started = time.perf_counter()
    click_selected_captcha_tiles(
        page,
        decision.selected_tiles,
        tiles=tiles_in_frame,
    )
    record_captcha_stage(
        captcha="second",
        stage="box_selection",
        attempt_number=attempt_number,
        started_at=selection_started_at,
        duration_ms=round((time.perf_counter() - selection_started) * 1000),
    )

    submission_started_at = datetime.now(timezone.utc)
    submission_started = time.perf_counter()
    try:
        click_submit_selection(
            frame
        )
    except PlaywrightTimeoutError:
        if http_forbidden_page_visible(page):
            raise HTTP403Forbidden(HTTP_FORBIDDEN_MESSAGE)
        logger.warning(
            "Second CAPTCHA submit click timed out while its result was "
            "settling; inspecting verification state before retrying."
        )
    record_captcha_stage(
        captcha="second",
        stage="submission",
        attempt_number=attempt_number,
        started_at=submission_started_at,
        duration_ms=round((time.perf_counter() - submission_started) * 1000),
    )

    logger.info(
        "Submitted second CAPTCHA selection."
    )

    save_live_attempt_bundle(
        output_dir=output_dir,
        step_name="second_captcha",
        attempt_number=attempt_number,
        page_url=page.url,
        target=target,
        decision=decision,
        captcha_image=captcha_image,
        debug_image=debug,
        tiles=tiles,
    )

    verified_label = frame.get_by_text("Verified!", exact=True).first
    invalid_label = frame.get_by_text(
        re.compile(r"invalid captcha selection|incorrect selection", re.IGNORECASE)
    ).first

    verification_started_at = datetime.now(timezone.utc)
    verification_started = time.perf_counter()
    deadline = time.monotonic() + SECOND_CAPTCHA_VERIFICATION_TIMEOUT_SECONDS
    loading_grace_used = False
    while time.monotonic() < deadline:
        outcome = classify_second_captcha_state(
            page, verified_label, invalid_label, popup
        )
        if outcome == "verified":
            break
        if outcome == "explicitly_rejected":
            record_captcha_stage(
                captcha="second",
                stage="verification",
                attempt_number=attempt_number,
                started_at=verification_started_at,
                duration_ms=round((time.perf_counter() - verification_started) * 1000),
                status="explicitly_rejected",
            )
            logger.warning(
                "Second CAPTCHA attempt %s received an explicit invalid-selection response.",
                attempt_number,
            )
            wait_for_regenerated_second_captcha(
                page, frame, submitted_challenge_signature, invalid_label
            )
            return False
        if outcome == "advanced":
            page_state = inspect_page_state(page)
            record_captcha_stage(
                captcha="second",
                stage="verification",
                attempt_number=attempt_number,
                started_at=verification_started_at,
                duration_ms=round(
                    (time.perf_counter() - verification_started) * 1000
                ),
                status="advanced",
            )
            logger.info(
                "Second CAPTCHA advanced directly to a valid downstream "
                "state. State=%s",
                page_state,
            )
            return True
        if outcome == "loading" and not loading_grace_used:
            if deadline - time.monotonic() <= 1:
                deadline += SECOND_CAPTCHA_LOADING_GRACE_SECONDS
                loading_grace_used = True
                logger.info(
                    "Second CAPTCHA still has a loading overlay; allowing %ss more.",
                    SECOND_CAPTCHA_LOADING_GRACE_SECONDS,
                )
        page.wait_for_timeout(250)
    else:
        record_captcha_stage(
            captcha="second",
            stage="verification",
            attempt_number=attempt_number,
            started_at=verification_started_at,
            duration_ms=round((time.perf_counter() - verification_started) * 1000),
            status="loading_timeout" if outcome == "loading" else "unconfirmed",
        )
        raise SecondCaptchaUnconfirmed(
            "Second CAPTCHA was not confirmed after "
            f"{SECOND_CAPTCHA_VERIFICATION_TIMEOUT_SECONDS}s; "
            f"popup_visible={popup.is_visible()}. "
            "No new puzzle will be requested without an explicit rejection."
        )

    record_captcha_stage(
        captcha="second",
        stage="verification",
        attempt_number=attempt_number,
        started_at=verification_started_at,
        duration_ms=round((time.perf_counter() - verification_started) * 1000),
        status="verified",
    )

    logger.info(
        'Second CAPTCHA returned "Verified!".'
    )

    try:
        expect(popup).to_be_hidden(timeout=10_000)
    except Exception as error:
        if http_forbidden_page_visible(page):
            raise HTTP403Forbidden(HTTP_FORBIDDEN_MESSAGE) from error
        raise SecondCaptchaUnconfirmed(
            "Second CAPTCHA showed Verified but its popup remained open."
        ) from error

    logger.info(
        "Second CAPTCHA verification succeeded."
    )

    write_outputs(
        output_dir,
        captcha_image,
        debug,
        tiles,
        decision,
    )
    return True


def fill_password(
    page,
    *,
    target_password: str,
) -> None:
    logger.info(
        "Filling password field."
    )

    fill_visible_password(
        page,
        target_password,
    )


def submit_password(
    page
) -> None:
    logger.info(
        "Submitting password."
    )

    page.get_by_role(
        "button",
        name="Submit",
    ).click(
        timeout=10_000
    )


def run_post_login_step(
    page,
    *,
    target_password: str,
) -> None:
    fill_password(
        page,
        target_password=target_password,
    )

    submit_password(
        page
    )


def _run_single_subtype_attempt(
    *,
    config: ScraperConfig,
    visa_sub_type: str,
    attempt_number: int,
    reader,
    proxy_config: dict[str, str] | None,
    login_request_state: dict[str, float | None],
    should_stop: Callable[[], bool] | None,
) -> ScraperResult:
    """
    One completely fresh browser attempt for exactly one visa subtype.

    Success means the appointment form for this subtype was submitted
    and produced either:
        - NO_APPOINTMENT
        - APPOINTMENT_FOUND

    Any infrastructure/CAPTCHA/navigation failure returns FAILED.
    """

    started_at = datetime.now(
        timezone.utc
    )

    logger.info(
        "=================================================="
    )

    logger.info(
        "Starting fresh browser attempt. "
        "Visa subtype=%s | Attempt=%s/%s",
        visa_sub_type,
        attempt_number,
        MAX_SUBTYPE_ATTEMPTS,
    )

    logger.info(
        "=================================================="
    )

    with sync_playwright() as playwright:
        browser = None
        context = None
        page = None
        egress_ip = None
        egress_ip_hash = None
        egress_lookup_error = None
        proxy_endpoint = (
            proxy_config.get("server") if proxy_config is not None else None
        )
        login_requested_at = None

        try:
            check_stop_requested(should_stop)
            browser_options = {
                "headless": config.headless,
            }
            executable_path = os.getenv(
                "PLAYWRIGHT_EXECUTABLE_PATH",
                "",
            ).strip()
            if executable_path:
                browser_options["executable_path"] = executable_path

            if proxy_config is not None:
                browser_options["proxy"] = proxy_config
                logger.info(
                    "Using outbound proxy for browser attempt: %s",
                    proxy_config["server"],
                )
            else:
                logger.info(
                    "No outbound proxy configured; using direct network access."
                )

            browser = (
                playwright.chromium.launch(
                    **browser_options,
                )
            )

            logger.info(
                "Chromium browser launched. "
                "Visa subtype=%s",
                visa_sub_type,
            )

            context = (
                browser.new_context(
                    viewport={
                        "width": 1440,
                        "height": 1000,
                    },
                )
            )

            egress_ip, egress_ip_hash, egress_lookup_error = resolve_egress_ip(context)

            page = context.new_page()
            track_http_forbidden_responses(page)

            logger.info(
                "Opening login page: %s",
                LOGIN_URL,
            )

            login_requested_at = time.monotonic()
            previous_login_requested_at = login_request_state.get(
                "previous_request_monotonic"
            )
            login_request_state["previous_request_monotonic"] = login_requested_at
            response = page.goto(
                LOGIN_URL,
                wait_until="domcontentloaded",
                timeout=60_000,
            )
            check_stop_requested(should_stop)

            if response is not None:
                login_response_data = response_diagnostics(
                    response=response,
                    context=context,
                    account=BLS_EMAIL,
                    egress_ip=egress_ip,
                    egress_ip_hash=egress_ip_hash,
                    egress_lookup_error=egress_lookup_error,
                    proxy_endpoint=proxy_endpoint,
                    seconds_since_previous_login=(
                        login_requested_at - previous_login_requested_at
                        if previous_login_requested_at is not None
                        else None
                    ),
                )
                record_scraper_event(
                    ScraperEvent.EventType.LOGIN_RESPONSE,
                    status=str(response.status),
                    reason_code=(
                        "HTTP_403" if response.status == 403 else ""
                    ),
                    data=login_response_data,
                )
                logger.info(
                    "Initial HTTP response diagnostics=%s",
                    json.dumps(login_response_data, sort_keys=True),
                )

                if response.status == 403:
                    message = (
                        "The target site returned HTTP 403 Forbidden. "
                        "Stopping immediately; this failure is not retryable."
                    )
                    logger.error(
                        "%s Visa subtype=%s | URL=%s",
                        message,
                        visa_sub_type,
                        page.url,
                    )
                    return ScraperResult(
                        status=ScraperStatus.FAILED,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                        page_url=page.url,
                        visa_sub_type=visa_sub_type,
                        error_type="HTTP403Forbidden",
                        error_message=message,
                    )

            #
            # CAPTCHA 1
            #
            logger.info(
                "Running login step."
            )

            run_login_step(
                page
            )

            logger.info(
                "Running login CAPTCHA step."
            )

            run_captcha_step(
                page,
                gpu=config.gpu,
                output_dir=(
                    config.output_dir
                    / visa_sub_type
                    / f"browser_attempt_{attempt_number:02d}"
                ),
                reader=reader,
                should_stop=should_stop,
            )
            check_stop_requested(should_stop)

            #
            # CAPTCHA 2
            #
            logger.info(
                "Login CAPTCHA complete. "
                "Opening new appointment workflow."
            )

            click_nav_book_new_appointment(
                page
            )

            logger.info(
                "Opening Verify Selection CAPTCHA."
            )

            click_verify_selection(
                page
            )

            run_second_captcha_step(
                page,
                gpu=config.gpu,
                output_dir=(
                    config.output_dir
                    / visa_sub_type
                    / f"browser_attempt_{attempt_number:02d}"
                ),
                reader=reader,
                should_stop=should_stop,
            )
            check_stop_requested(should_stop)

            logger.info(
                "Second CAPTCHA complete."
            )

            click_background_submit(
                page
            )

            logger.info(
                "Submitted background appointment step."
            )

            click_ok_dialog(
                page
            )
            check_stop_requested(should_stop)

            logger.info(
                "Visa type disclaimer accepted."
            )

            #
            # ONE form only.
            #
            logger.info(
                "Filling appointment form. "
                "Visa subtype=%s",
                visa_sub_type,
            )

            fill_appointment_form(
                page,
                visa_sub_type=visa_sub_type,
            )
            check_stop_requested(should_stop)

            logger.info(
                "Appointment form filled. "
                "Visa subtype=%s",
                visa_sub_type,
            )

            submit_button = (
                page.get_by_role(
                    "button",
                    name="Submit",
                )
                .first
            )

            expect(
                submit_button
            ).to_be_visible(
                timeout=30_000
            )

            expect(
                submit_button
            ).to_be_enabled(
                timeout=30_000
            )

            logger.info(
                "Clicking appointment Submit. "
                "Visa subtype=%s",
                visa_sub_type,
            )

            submit_button.click(
                timeout=10_000
            )

            # A delayed no-appointments response must not become a false
            # confirmed-appointment result merely because it was absent at 3s.
            form_result = wait_for_form_result(page, should_stop=should_stop)

            #
            # Form check successfully completed.
            #
            if form_result is ScraperStatus.NO_APPOINTMENT:
                logger.info(
                    "Successful form check: "
                    "NO APPOINTMENT. "
                    "Visa subtype=%s",
                    visa_sub_type,
                )

                log_no_appointment(
                    page_url=page.url,
                    visa_sub_type=visa_sub_type,
                )

                return ScraperResult(
                    status=(
                        ScraperStatus
                        .NO_APPOINTMENT
                    ),
                    started_at=started_at,
                    finished_at=datetime.now(
                        timezone.utc
                    ),
                    page_url=page.url,
                    visa_sub_type=visa_sub_type,
                )

            #
            # Only an explicit availability response reaches this branch.
            logger.warning(
                "Form check showed an explicit appointment-available result. "
                "Visa subtype=%s | URL=%s",
                visa_sub_type,
                page.url,
            )
            evidence_path = (
                config.output_dir
                / visa_sub_type
                / f"browser_attempt_{attempt_number:02d}"
                / "appointment_found.png"
            )
            try:
                evidence_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(evidence_path), full_page=True)
                logger.info("Saved appointment page evidence: %s", evidence_path)
            except Exception:
                logger.exception("Could not save possible-appointment screenshot.")

            result = ScraperResult(
                status=(
                    ScraperStatus
                    .APPOINTMENT_FOUND
                ),
                started_at=started_at,
                finished_at=datetime.now(
                    timezone.utc
                ),
                page_url=page.url,
                visa_sub_type=visa_sub_type,
            )

            # Notify immediately when this form check finds an appointment.
            # The overall run may continue checking other subtypes, but the
            # alert must not wait for final run bookkeeping.
            if result.status is ScraperStatus.APPOINTMENT_FOUND:
                notify_admin(
                    "Appointment availability was explicitly shown by the site.",
                    page_url=result.page_url or page.url,
                    visa_sub_type=result.visa_sub_type,
                    confirmed=True,
                )

            return result

        except ScraperStopRequested as error:
            logger.warning(
                "Stopping scraper safely and closing browser. Visa subtype=%s",
                visa_sub_type,
            )
            return ScraperResult(
                status=ScraperStatus.STOPPED,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                page_url=page.url if page is not None else None,
                visa_sub_type=visa_sub_type,
                error_type=type(error).__name__,
                error_message=str(error),
            )

        except PlaywrightTimeoutError as error:
            page_state = inspect_page_state(page)
            if page_state.get("http_403"):
                record_detected_http_403(
                    page=page,
                    context=context,
                    egress_ip=egress_ip,
                    egress_ip_hash=egress_ip_hash,
                    egress_lookup_error=egress_lookup_error,
                    proxy_endpoint=proxy_endpoint,
                    login_requested_at=login_requested_at,
                )
            screenshot_path = (
                config.output_dir
                / visa_sub_type
                / (
                    f"browser_attempt_"
                    f"{attempt_number:02d}"
                )
                / "playwright_timeout.png"
            )

            if page is not None:
                try:
                    screenshot_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    page.screenshot(
                        path=str(
                            screenshot_path
                        ),
                        full_page=True,
                    )
                    (screenshot_path.parent / "page_state.json").write_text(
                        json.dumps(page_state, indent=2),
                        encoding="utf-8",
                    )

                    logger.error(
                        "Saved timeout screenshot: %s",
                        screenshot_path,
                    )

                except Exception:
                    logger.exception(
                        "Failed to save timeout screenshot."
                    )

            logger.exception(
                "Fresh browser attempt timed out. "
                "Visa subtype=%s | Attempt=%s/%s",
                visa_sub_type,
                attempt_number,
                MAX_SUBTYPE_ATTEMPTS,
            )

            detected_status = (
                ScraperStatus.SERVER_ERROR
                if page_state.get("server_error")
                else ScraperStatus.NO_APPOINTMENT
                if page_state.get("no_appointment")
                else ScraperStatus.FAILED
            )
            error_type = (
                "HTTP403Forbidden"
                if page_state.get("http_403")
                else type(error).__name__
            )
            error_message = (
                HTTP_FORBIDDEN_MESSAGE
                if page_state.get("http_403")
                else str(error)
            )
            return ScraperResult(
                status=(
                    detected_status
                ),
                started_at=started_at,
                finished_at=datetime.now(
                    timezone.utc
                ),
                page_url=(
                    page.url
                    if page is not None
                    else None
                ),
                visa_sub_type=visa_sub_type,
                error_type=error_type,
                error_message=error_message,
                failure_screenshot=(
                    screenshot_path
                ),
            )

        except (
            PlaywrightError,
            RuntimeError,
            ValueError,
            OSError,
            AssertionError,
        ) as error:
            page_state = inspect_page_state(page)
            if page_state.get("http_403"):
                record_detected_http_403(
                    page=page,
                    context=context,
                    egress_ip=egress_ip,
                    egress_ip_hash=egress_ip_hash,
                    egress_lookup_error=egress_lookup_error,
                    proxy_endpoint=proxy_endpoint,
                    login_requested_at=login_requested_at,
                )
            screenshot_path = (
                config.output_dir
                / visa_sub_type
                / (
                    f"browser_attempt_"
                    f"{attempt_number:02d}"
                )
                / "playwright_error.png"
            )

            if page is not None:
                try:
                    screenshot_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    page.screenshot(
                        path=str(
                            screenshot_path
                        ),
                        full_page=True,
                    )
                    (screenshot_path.parent / "page_state.json").write_text(
                        json.dumps(page_state, indent=2),
                        encoding="utf-8",
                    )

                    logger.error(
                        "Saved failure screenshot: %s",
                        screenshot_path,
                    )

                except Exception:
                    logger.exception(
                        "Failed to save failure screenshot."
                    )

            logger.exception(
                "Fresh browser attempt failed. "
                "Visa subtype=%s | Attempt=%s/%s",
                visa_sub_type,
                attempt_number,
                MAX_SUBTYPE_ATTEMPTS,
            )

            detected_status = (
                ScraperStatus.SERVER_ERROR
                if page_state.get("server_error")
                else ScraperStatus.NO_APPOINTMENT
                if page_state.get("no_appointment")
                else ScraperStatus.FAILED
            )
            error_type = (
                "HTTP403Forbidden"
                if page_state.get("http_403")
                else type(error).__name__
            )
            error_message = (
                HTTP_FORBIDDEN_MESSAGE
                if page_state.get("http_403")
                else str(error)
            )
            return ScraperResult(
                status=(
                    detected_status
                ),
                started_at=started_at,
                finished_at=datetime.now(
                    timezone.utc
                ),
                page_url=(
                    page.url
                    if page is not None
                    else None
                ),
                visa_sub_type=visa_sub_type,
                error_type=error_type,
                error_message=error_message,
                failure_screenshot=(
                    screenshot_path
                ),
            )

        finally:
            logger.info(
                "Closing fresh Playwright browser. "
                "Visa subtype=%s | Attempt=%s",
                visa_sub_type,
                attempt_number,
            )

            if context is not None:
                try:
                    context.close()
                except Exception:
                    logger.exception(
                        "Failed to close browser context cleanly."
                    )

            if browser is not None:
                try:
                    browser.close()
                except Exception:
                    logger.exception(
                        "Failed to close browser cleanly."
                    )

            logger.info(
                "Fresh browser closed. "
                "Visa subtype=%s | Attempt=%s",
                visa_sub_type,
                attempt_number,
            )


def run_scraper(
    config: ScraperConfig,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> ScraperResult:
    """
    Complete one scraper job.

    Each configured visa subtype is checked independently.

    For every subtype:
        fresh browser
        -> login CAPTCHA
        -> Verify Selection CAPTCHA
        -> submit exactly one appointment form

    A failed attempt is discarded completely and retried using
    another fresh browser.

    A subtype is considered complete only once its form produces
    either APPOINTMENT_FOUND or NO_APPOINTMENT.

    The overall scraper succeeds only once EVERY configured subtype
    has been successfully checked.
    """

    overall_started_at = (
        datetime.now(
            timezone.utc
        )
    )

    logger.info(
        "Starting independent-subtype scraper. "
        "Headless=%s | GPU=%s | Visa subtypes=%s | "
        "Max fresh attempts per subtype=%s",
        config.headless,
        config.gpu,
        ", ".join(
            config.visa_sub_types
        ),
        MAX_SUBTYPE_ATTEMPTS,
    )

    try:
        check_stop_requested(should_stop)
    except ScraperStopRequested as error:
        return ScraperResult(
            status=ScraperStatus.STOPPED,
            started_at=overall_started_at,
            finished_at=datetime.now(timezone.utc),
            error_type=type(error).__name__,
            error_message=str(error),
        )

    proxy_rotator = PlaywrightProxyRotator()
    try:
        proxy_rotator.validate_required_pool()
    except ProxyConfigurationError as error:
        logger.error("Proxy configuration rejected: %s", error)
        return ScraperResult(
            status=ScraperStatus.FAILED,
            started_at=overall_started_at,
            finished_at=datetime.now(timezone.utc),
            error_type=type(error).__name__,
            error_message=str(error),
        )

    logger.info(
        "Validated fail-closed proxy pool with %s endpoints; "
        "direct browser egress is disabled.",
        len(proxy_rotator.proxy_urls),
    )
    logger.info("Getting PARSeq-tiny reader for this worker. GPU=%s", config.gpu)
    reader = get_reader(gpu=config.gpu)
    login_request_state: dict[str, float | None] = {
        "previous_request_monotonic": None,
    }

    successful_results: list[
        ScraperResult
    ] = []

    appointment_results: list[
        ScraperResult
    ] = []
    attempt_failures: list[dict[str, object]] = []

    for visa_sub_type in (
        config.visa_sub_types
    ):
        try:
            check_stop_requested(should_stop)
        except ScraperStopRequested as error:
            return ScraperResult(
                status=ScraperStatus.STOPPED,
                started_at=overall_started_at,
                finished_at=datetime.now(timezone.utc),
                error_type=type(error).__name__,
                error_message=str(error),
                first_failure=(attempt_failures[0] if attempt_failures else None),
                attempt_failures=tuple(attempt_failures),
            )
        logger.info(
            "--------------------------------------------------"
        )

        logger.info(
            "Beginning independent check. "
            "Visa subtype=%s",
            visa_sub_type,
        )

        subtype_result: (
            ScraperResult
            | None
        ) = None

        last_failure: (
            ScraperResult
            | None
        ) = None

        for attempt_number in range(
            1,
            MAX_SUBTYPE_ATTEMPTS + 1,
        ):
            result = (
                _run_single_subtype_attempt(
                    config=config,
                    visa_sub_type=visa_sub_type,
                    attempt_number=attempt_number,
                    reader=reader,
                    proxy_config=proxy_rotator.choose(),
                    login_request_state=login_request_state,
                    should_stop=should_stop,
                )
            )

            if result.succeeded:
                subtype_result = result
                break

            if result.status is ScraperStatus.STOPPED:
                return ScraperResult(
                    status=ScraperStatus.STOPPED,
                    started_at=overall_started_at,
                    finished_at=datetime.now(timezone.utc),
                    page_url=result.page_url,
                    visa_sub_type=visa_sub_type,
                    error_type=result.error_type,
                    error_message=result.error_message,
                    first_failure=(attempt_failures[0] if attempt_failures else None),
                    attempt_failures=tuple(attempt_failures),
                )

            last_failure = result
            current_failure = failure_record(
                result,
                visa_sub_type=visa_sub_type,
                attempt_number=attempt_number,
            )
            attempt_failures.append(current_failure)

            if result.error_type == "HTTP403Forbidden":
                logger.error(
                    "HTTP 403 is terminal; stopping without another attempt."
                )
                return ScraperResult(
                    status=ScraperStatus.FAILED,
                    started_at=overall_started_at,
                    finished_at=datetime.now(timezone.utc),
                    page_url=result.page_url,
                    visa_sub_type=visa_sub_type,
                    error_type=result.error_type,
                    error_message=result.error_message,
                    failure_screenshot=result.failure_screenshot,
                    first_failure=attempt_failures[0],
                    attempt_failures=tuple(attempt_failures),
                    terminal_failure=current_failure,
                )

            logger.warning(
                "Subtype attempt failed; "
                "discarding browser state and "
                "starting completely fresh. "
                "Visa subtype=%s | "
                "Failed attempt=%s/%s | "
                "Error=%s: %s",
                visa_sub_type,
                attempt_number,
                MAX_SUBTYPE_ATTEMPTS,
                result.error_type,
                result.error_message,
            )

            retry_delay = subtype_retry_delay_seconds(attempt_number)
            if retry_delay:
                logger.info(
                    "Cooling down before the next fresh browser attempt. "
                    "Visa subtype=%s | Next attempt=%s/%s | Delay=%ss",
                    visa_sub_type,
                    attempt_number + 1,
                    MAX_SUBTYPE_ATTEMPTS,
                    retry_delay,
                )
                try:
                    interruptible_cooldown(retry_delay, should_stop)
                except ScraperStopRequested as error:
                    return ScraperResult(
                        status=ScraperStatus.STOPPED,
                        started_at=overall_started_at,
                        finished_at=datetime.now(timezone.utc),
                        visa_sub_type=visa_sub_type,
                        error_type=type(error).__name__,
                        error_message=str(error),
                        first_failure=attempt_failures[0],
                        attempt_failures=tuple(attempt_failures),
                    )

        if subtype_result is None:
            logger.error(
                "Visa subtype could not be checked "
                "after %s fresh browser attempts. "
                "Visa subtype=%s",
                MAX_SUBTYPE_ATTEMPTS,
                visa_sub_type,
            )

            return ScraperResult(
                status=(
                    ScraperStatus.FAILED
                ),
                started_at=(
                    overall_started_at
                ),
                finished_at=datetime.now(
                    timezone.utc
                ),
                page_url=(
                    last_failure.page_url
                    if last_failure
                    else None
                ),
                visa_sub_type=visa_sub_type,
                error_type=(
                    "SubtypeRetryExhausted"
                ),
                error_message=(
                    f"Could not successfully check "
                    f"{visa_sub_type!r} after "
                    f"{MAX_SUBTYPE_ATTEMPTS} fresh "
                    "browser attempts. "
                    f"Last error: "
                    f"{last_failure.error_type if last_failure else 'unknown'}: "
                    f"{last_failure.error_message if last_failure else 'unknown'}"
                ),
                failure_screenshot=(
                    last_failure.failure_screenshot
                    if last_failure
                    else None
                ),
                first_failure=(attempt_failures[0] if attempt_failures else None),
                attempt_failures=tuple(attempt_failures),
                terminal_failure={
                    "visa_sub_type": visa_sub_type,
                    "attempt_number": MAX_SUBTYPE_ATTEMPTS,
                    "status": ScraperStatus.FAILED.value,
                    "error_type": "SubtypeRetryExhausted",
                    "error_message": "Fresh browser retry limit exhausted.",
                    "page_url": last_failure.page_url if last_failure else "",
                    "failure_screenshot": (
                        str(last_failure.failure_screenshot)
                        if last_failure and last_failure.failure_screenshot
                        else ""
                    ),
                    "occurred_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        successful_results.append(
            subtype_result
        )

        if subtype_result.status in (
            ScraperStatus.APPOINTMENT_FOUND,
            ScraperStatus.POSSIBLE_APPOINTMENT,
        ):
            appointment_results.append(
                subtype_result
            )

        logger.info(
            "Independent subtype check complete. "
            "Visa subtype=%s | Result=%s",
            visa_sub_type,
            subtype_result.status.value,
        )

        #
        # IMPORTANT:
        # Even if the first subtype has an appointment, we continue.
        # Your definition of a completed task is that BOTH configured
        # forms have been successfully checked.
        #

    logger.info(
        "All configured visa subtypes "
        "were successfully checked. "
        "Completed forms=%s/%s",
        len(
            successful_results
        ),
        len(
            config.visa_sub_types
        ),
    )

    if appointment_results:
        appointment_subtypes = (
            ", ".join(
                result.visa_sub_type
                for result
                in appointment_results
                if result.visa_sub_type
            )
        )

        overall_appointment_status = (
            ScraperStatus.APPOINTMENT_FOUND
            if any(result.status is ScraperStatus.APPOINTMENT_FOUND
                   for result in appointment_results)
            else ScraperStatus.POSSIBLE_APPOINTMENT
        )
        logger.warning(
            "Overall result: %s. Subtypes=%s",
            overall_appointment_status.value,
            appointment_subtypes,
        )

        return ScraperResult(
            status=overall_appointment_status,
            started_at=(
                overall_started_at
            ),
            finished_at=datetime.now(
                timezone.utc
            ),
            page_url=(
                appointment_results[
                    0
                ].page_url
            ),
            visa_sub_type=(
                appointment_subtypes
            ),
            first_failure=(attempt_failures[0] if attempt_failures else None),
            attempt_failures=tuple(attempt_failures),
        )

    logger.info(
        "Overall result: NO_APPOINTMENT. "
        "All %s configured forms were "
        "successfully checked.",
        len(
            successful_results
        ),
    )

    last_result = (
        successful_results[-1]
        if successful_results
        else None
    )

    return ScraperResult(
        status=(
            ScraperStatus
            .NO_APPOINTMENT
        ),
        started_at=(
            overall_started_at
        ),
        finished_at=datetime.now(
            timezone.utc
        ),
        page_url=(
            last_result.page_url
            if last_result
            else None
        ),
        first_failure=(attempt_failures[0] if attempt_failures else None),
        attempt_failures=tuple(attempt_failures),
    )
