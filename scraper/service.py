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
from config import BLS_EMAIL, BLS_PASSWORD, LOGIN_URL
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
from flows.selectors import CAPTCHA_INSTRUCTION_PATTERN
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


logger = logging.getLogger("captcha_lab")


def run_login_step(page) -> None:
    logger.info("Submitting login email.")

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
        attempt_dir / "live_metadata.json"
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
    screenshot_path = Path(
        "captcha_page.png"
    )

    max_attempts = 3

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        logger.info(
            "Login CAPTCHA attempt %s/%s",
            attempt,
            max_attempts,
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
        ) = find_true_captcha_label(page)

        true_label_by_id = page.locator(
            f"#{true_label_id}"
        )

        expect(
            true_label_by_id
        ).to_have_count(1)

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
            .fullmatch(confirmed_text)
        )

        if (
            confirmed_match is None
            or confirmed_match.group(1) != target
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

        solve_start = time.perf_counter()

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
            list(decision.selected_tiles),
            list(decision.uncertain_tiles),
        )

        print_decision(decision)

        click_selected_captcha_tiles(
            page,
            decision.selected_tiles,
        )

        logger.info(
            "Selected login CAPTCHA tiles=%s",
            list(decision.selected_tiles),
        )

        click_verify_selection(page)

        logger.info(
            "Submitted login CAPTCHA selection."
        )

        page.wait_for_timeout(
            1_000
        )

        save_live_attempt_bundle(
            output_dir=output_dir,
            step_name="login_captcha",
            attempt_number=attempt,
            page_url=page.url,
            target=target,
            decision=decision,
            captcha_image=captcha_image,
            debug_image=debug,
            tiles=tiles,
        )

        if login_captcha_invalid(page):
            logger.warning(
                "Login CAPTCHA was rejected. Retrying."
            )
            continue

        if login_captcha_succeeded(page):
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

        if captcha_instruction_present(page):
            logger.warning(
                "Login CAPTCHA instruction is still "
                "present after verification. Retrying."
            )
            continue

        logger.warning(
            "Login CAPTCHA outcome is unclear. Retrying."
        )

    raise RuntimeError(
        "Login CAPTCHA verification failed "
        f"after {max_attempts} attempts."
    )


def run_second_captcha_step(
    page,
    *,
    gpu: bool,
    output_dir: Path,
) -> None:
    frame = get_verify_selection_frame(
        page
    )

    screenshot_path = Path(
        "captcha_page_2.png"
    )

    max_attempts = 3

    popup = (
        page.locator(
            "div.k-widget.k-window"
        )
        .filter(
            has_text="Verify Selection"
        )
        .first
    )

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        logger.info(
            "Second CAPTCHA attempt %s/%s",
            attempt,
            max_attempts,
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

        true_label_by_id = frame.locator(
            f"#{true_label_id}"
        )

        expect(
            true_label_by_id
        ).to_have_count(1)

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
            .fullmatch(confirmed_text)
        )

        if (
            confirmed_match is None
            or confirmed_match.group(1) != target
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

        solve_start = time.perf_counter()

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
            list(decision.selected_tiles),
            list(decision.uncertain_tiles),
        )

        print_decision(decision)

        tiles_in_frame = (
            get_captcha_tiles_in_scope(
                frame
            )
        )

        for tile_number in (
            decision.selected_tiles
        ):
            if (
                tile_number < 1
                or tile_number > len(tiles_in_frame)
            ):
                raise ValueError(
                    f"Selected tile {tile_number} "
                    "is outside the "
                    f"1..{len(tiles_in_frame)} range"
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
            attempt_number=attempt,
            page_url=page.url,
            target=target,
            decision=decision,
            captcha_image=captcha_image,
            debug_image=debug,
            tiles=tiles,
        )

        verified_label = frame.locator(
            "text=Verified!"
        )

        try:
            expect(
                verified_label
            ).to_be_visible(
                timeout=15_000
            )

            logger.info(
                'Second CAPTCHA returned "Verified!".'
            )

        except Exception:
            logger.warning(
                "Second CAPTCHA did not return "
                "Verified. Retrying."
            )
            continue

        page.wait_for_timeout(
            5_000
        )

        if popup.is_visible():
            logger.warning(
                "Second CAPTCHA popup remained open "
                "after verification. Retrying."
            )
            continue

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

        return

    raise RuntimeError(
        "Second CAPTCHA failed after "
        f"{max_attempts} attempts."
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


def submit_password(page) -> None:
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


def run_scraper(
    config: ScraperConfig,
) -> ScraperResult:
    started_at = datetime.now(
        timezone.utc
    )

    logger.info(
        "Starting Playwright scraper. "
        "Headless=%s | GPU=%s | Output=%s",
        config.headless,
        config.gpu,
        config.output_dir,
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=config.headless,
        )

        logger.info(
            "Chromium browser launched."
        )

        context = browser.new_context(
            viewport={
                "width": 1440,
                "height": 1000,
            },
        )

        page = context.new_page()

        try:
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
                output_dir=config.output_dir,
            )

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
                output_dir=config.output_dir,
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

            for visa_sub_type in (
                config.visa_sub_types
            ):
                logger.info(
                    "Trying appointment form. "
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
                    "Clicking appointment Submit button. "
                    "Visa subtype=%s",
                    visa_sub_type,
                )

                submit_button.click(
                    timeout=10_000
                )

                page.wait_for_timeout(
                    3_000
                )

                if no_appointments_dialog_visible(
                    page
                ):
                    logger.info(
                        "No appointments available. "
                        "Visa subtype=%s",
                        visa_sub_type,
                    )

                    log_no_appointment(
                        page_url=page.url,
                        visa_sub_type=visa_sub_type,
                    )

                    click_ok_dialog(
                        page
                    )

                    continue

                logger.warning(
                    "No 'No Appointments Available' "
                    "dialog was found. Treating this as "
                    "possible appointment availability. "
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

                logger.warning(
                    "Appointment availability detected "
                    "and admin notification sent. "
                    "Visa subtype=%s",
                    visa_sub_type,
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

            logger.info(
                "No appointments found for any "
                "configured visa subtype."
            )

            log_no_appointment(
                page_url=page.url,
                visa_sub_type=None,
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
            )

        except PlaywrightTimeoutError as error:
            screenshot_path = Path(
                "playwright_timeout.png"
            )

            try:
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
                "Playwright timed out. URL=%s",
                page.url,
            )

            return ScraperResult(
                status=ScraperStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(
                    timezone.utc
                ),
                page_url=page.url,
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
            screenshot_path = Path(
                "playwright_error.png"
            )

            try:
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
                "Automation failed. URL=%s",
                page.url,
            )

            return ScraperResult(
                status=ScraperStatus.FAILED,
                started_at=started_at,
                finished_at=datetime.now(
                    timezone.utc
                ),
                page_url=page.url,
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
                "Closing Playwright browser context."
            )

            context.close()
            browser.close()

            logger.info(
                "Playwright browser closed."
            )
