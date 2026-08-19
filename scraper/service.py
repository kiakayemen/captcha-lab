from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

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
    fill_appointment_form,
    no_appointments_dialog_visible,
)
from flows.captcha_flow import (
    captcha_instruction_present,
    click_background_submit,
    click_nav_book_new_appointment,
    click_ok_dialog,
    click_selected_captcha_tiles,
    click_submit_selection,
    click_verify_selection,
    find_true_captcha_label,
    find_true_captcha_label_in_scope,
    get_captcha_tiles_in_scope,
    get_verify_selection_frame,
    login_captcha_invalid,
    login_captcha_succeeded,
    save_captcha_crop,
)
from flows.login_flow import (
    fill_visible_password,
    submit_email,
)
from flows.selectors import (
    CAPTCHA_INSTRUCTION_PATTERN,
)
from notifications import (
    log_no_appointment,
    notify_admin,
)
from ocr import build_reader

from scraper.models import (
    ScraperConfig,
    ScraperResult,
    ScraperStatus,
)


logger = logging.getLogger(
    "captcha_lab"
)


# A subtype gets this many completely fresh browser attempts.
#
# We deliberately do NOT retry forever. If BLS changes or is down,
# an infinite unattended browser loop would be dangerous.
MAX_SUBTYPE_ATTEMPTS = 5


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
) -> None:
    """
    Solve the login CAPTCHA once.

    IMPORTANT:
    We intentionally do not retry the CAPTCHA inside the same
    browser session anymore.

    If this CAPTCHA fails, this function raises and the entire
    subtype attempt is abandoned. The caller then starts a
    completely fresh browser session.
    """

    screenshot_path = Path(
        "captcha_page.png"
    )

    logger.info(
        "Login CAPTCHA: single-attempt mode."
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

    get_captcha_tiles_in_scope(
        page
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

    captcha_image = save_captcha_crop(
        page,
        screenshot_path,
    )

    logger.info(
        "Saved cropped login CAPTCHA image: %s",
        screenshot_path,
    )

    logger.info(
        "Loading EasyOCR. GPU=%s",
        gpu,
    )

    solve_start = (
        time.perf_counter()
    )

    reader = build_reader(
        gpu=gpu
    )

    (
        decision,
        tiles,
        _boxes,
        debug,
    ) = solve_captcha_image(
        captcha_image,
        target=target,
        reader=reader,
    )

    solve_seconds = (
        time.perf_counter()
        - solve_start
    )

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

    click_selected_captcha_tiles(
        page,
        decision.selected_tiles,
    )

    logger.info(
        "Selected login CAPTCHA tiles=%s",
        list(
            decision.selected_tiles
        ),
    )

    click_verify_selection(
        page
    )

    logger.info(
        "Submitted login CAPTCHA selection."
    )

    page.wait_for_timeout(
        1_000
    )

    save_live_attempt_bundle(
        output_dir=output_dir,
        step_name="login_captcha",
        attempt_number=1,
        page_url=page.url,
        target=target,
        decision=decision,
        captcha_image=captcha_image,
        debug_image=debug,
        tiles=tiles,
    )

    if login_captcha_invalid(
        page
    ):
        raise RuntimeError(
            "Login CAPTCHA was rejected."
        )

    if login_captcha_succeeded(
        page
    ):
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

        return

    if captcha_instruction_present(
        page
    ):
        raise RuntimeError(
            "Login CAPTCHA instruction remained "
            "present after verification."
        )

    raise RuntimeError(
        "Login CAPTCHA outcome was unclear."
    )


def run_second_captcha_step(
    page,
    *,
    gpu: bool,
    output_dir: Path,
) -> None:
    """
    Solve Verify Selection CAPTCHA once.

    Any failure aborts the current browser attempt.
    """

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
        "Second CAPTCHA: single-attempt mode."
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

    tiles_in_frame = (
        get_captcha_tiles_in_scope(
            frame
        )
    )

    page.wait_for_timeout(
        500
    )

    logger.info(
        "Second CAPTCHA grid ready. "
        "Taking solver screenshot."
    )

    captcha_image = save_captcha_crop(
        page,
        screenshot_path,
    )

    logger.info(
        "Saved cropped second CAPTCHA image: %s",
        screenshot_path,
    )

    logger.info(
        "Loading EasyOCR for second CAPTCHA. "
        "GPU=%s",
        gpu,
    )

    solve_start = (
        time.perf_counter()
    )

    reader = build_reader(
        gpu=gpu
    )

    (
        decision,
        tiles,
        _boxes,
        debug,
    ) = solve_captcha_image(
        captcha_image,
        target=target,
        reader=reader,
    )

    solve_seconds = (
        time.perf_counter()
        - solve_start
    )

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

    for tile_number in (
        decision.selected_tiles
    ):
        if (
            tile_number < 1
            or tile_number
            > len(
                tiles_in_frame
            )
        ):
            raise ValueError(
                f"Selected tile "
                f"{tile_number} "
                "is outside the "
                f"1..{len(tiles_in_frame)} "
                "range"
            )

        tile = tiles_in_frame[
            tile_number - 1
        ]

        tile.scroll_into_view_if_needed(
            timeout=10_000
        )

        tile.click(
            timeout=10_000
        )

        logger.info(
            "Clicked second CAPTCHA tile=%s",
            tile_number,
        )

    click_submit_selection(
        frame
    )

    logger.info(
        "Submitted second CAPTCHA selection."
    )

    save_live_attempt_bundle(
        output_dir=output_dir,
        step_name="second_captcha",
        attempt_number=1,
        page_url=page.url,
        target=target,
        decision=decision,
        captcha_image=captcha_image,
        debug_image=debug,
        tiles=tiles,
    )

    verified_label = (
        frame.locator(
            "text=Verified!"
        )
    )

    try:
        expect(
            verified_label
        ).to_be_visible(
            timeout=15_000
        )

    except Exception as exc:
        raise RuntimeError(
            "Second CAPTCHA was not verified."
        ) from exc

    logger.info(
        'Second CAPTCHA returned "Verified!".'
    )

    page.wait_for_timeout(
        2_000
    )

    if popup.is_visible():
        raise RuntimeError(
            "Second CAPTCHA popup remained open "
            "after verification."
        )

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

        try:
            browser = (
                playwright.chromium.launch(
                    headless=config.headless,
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

            page = context.new_page()

            logger.info(
                "Opening login page: %s",
                LOGIN_URL,
            )

            response = page.goto(
                LOGIN_URL,
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            if response is not None:
                logger.info(
                    "Initial HTTP status=%s",
                    response.status,
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
            )

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
            )

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

            page.wait_for_timeout(
                3_000
            )

            #
            # Form check successfully completed.
            #
            if no_appointments_dialog_visible(
                page
            ):
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
            # Existing behavior:
            # absence of the No Appointments modal means possible
            # appointment availability.
            #
            logger.warning(
                "Successful form check: possible "
                "APPOINTMENT AVAILABLE. "
                "Visa subtype=%s | URL=%s",
                visa_sub_type,
                page.url,
            )

            notify_admin(
                (
                    "Appointment availability detected. "
                    "Manual booking is required."
                ),
                page_url=page.url,
                visa_sub_type=visa_sub_type,
            )

            return ScraperResult(
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

        except PlaywrightTimeoutError as error:
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

            return ScraperResult(
                status=(
                    ScraperStatus.FAILED
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
                error_type=type(
                    error
                ).__name__,
                error_message=str(
                    error
                ),
                failure_screenshot=(
                    screenshot_path
                ),
            )

        except (
            PlaywrightError,
            RuntimeError,
            ValueError,
            OSError,
        ) as error:
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

            return ScraperResult(
                status=(
                    ScraperStatus.FAILED
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
                error_type=type(
                    error
                ).__name__,
                error_message=str(
                    error
                ),
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

    successful_results: list[
        ScraperResult
    ] = []

    appointment_results: list[
        ScraperResult
    ] = []

    for visa_sub_type in (
        config.visa_sub_types
    ):
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
                )
            )

            if result.succeeded:
                subtype_result = result
                break

            last_failure = result

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
            )

        successful_results.append(
            subtype_result
        )

        if (
            subtype_result.status
            is ScraperStatus.APPOINTMENT_FOUND
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

        logger.warning(
            "Overall result: "
            "APPOINTMENT_FOUND. "
            "Subtypes=%s",
            appointment_subtypes,
        )

        return ScraperResult(
            status=(
                ScraperStatus
                .APPOINTMENT_FOUND
            ),
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
    )
