from __future__ import annotations

import logging
from pathlib import Path
import random
import re
import time

import cv2
import numpy as np
from playwright.sync_api import (
    FrameLocator,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    expect,
)
from extract_tiles import (
    bounding_rectangle,
    crop_box,
    expand_box,
    find_square_candidates,
    select_grid_boxes,
)
from .selectors import (
    CAPTCHA_INSTRUCTION_PATTERN,
    BOOK_NOW_SELECTOR,
    BACKGROUND_SUBMIT_BUTTON_SELECTOR,
    CAPTCHA_LABEL_SELECTOR,
    CAPTCHA_TILE_SELECTOR,
    LOGIN_FORM_SELECTOR,
    NAV_BOOK_NEW_APPOINTMENT_SELECTOR,
    OK_DIALOG_BUTTON_SELECTOR,
    SECOND_CAPTCHA_SUBMIT_SELECTOR,
    VERIFY_BUTTON_SELECTOR,
)
from .errors import raise_for_http_forbidden


logger = logging.getLogger("captcha_lab")
PRELOADER_SELECTOR = "div.preloader"
SITE_ERROR_PATTERN = re.compile(
    r"An error occurr?ed while processing your request",
    re.IGNORECASE,
)
CAPTCHA_PRE_CLICK_SETTLE_MS = 1_500
CAPTCHA_INTER_TILE_DELAY_MIN_MS = 350
CAPTCHA_INTER_TILE_DELAY_MAX_MS = 750
CAPTCHA_SELECTION_STATE_TIMEOUT_MS = 1_500
CAPTCHA_POST_SELECTION_SETTLE_MS = 5_000
POST_SECOND_CAPTCHA_SETTLE_MS = 3_000
BLOCKING_OVERLAY_SELECTOR = (
    "div.preloader:visible, "
    "div.global-overlay:visible, "
    "#global-overlay:visible, "
    ".global-overlay-loader:visible"
)
DISCLAIMER_MODAL_SELECTOR = "#disclaimarModal:visible"


def wait_for_preloader_to_clear(page: Page, timeout: int = 60_000) -> None:
    preloader = page.locator(PRELOADER_SELECTOR)
    try:
        expect(preloader).to_be_hidden(timeout=timeout)
    except Exception as exc:
        raise_for_http_forbidden(page)
        if preloader.is_visible():
            raise RuntimeError(
                "Loading overlay remained visible and continued blocking the page."
            ) from exc
        logger.info("Preloader node remained attached but is no longer visible.")


def site_error_page_visible(page: Page) -> bool:
    try:
        return page.get_by_text(SITE_ERROR_PATTERN).first.is_visible()
    except Exception:
        return False


def appointment_form_visible(page: Page) -> bool:
    try:
        return (
            page.locator('label.form-label:has-text("Jurisdiction")')
            .first
            .is_visible()
            is True
        )
    except Exception:
        return False


def disclaimer_dialog_visible(page: Page) -> bool:
    try:
        return page.locator(DISCLAIMER_MODAL_SELECTOR).first.is_visible() is True
    except Exception:
        return False


def appointment_form_ready(page: Page) -> bool:
    return (
        appointment_form_visible(page)
        and not disclaimer_dialog_visible(page)
        and not blocking_overlay_visible(page)
    )


def blocking_overlay_visible(page: Page) -> bool:
    try:
        overlays = page.locator(BLOCKING_OVERLAY_SELECTOR)
        return any(
            overlays.nth(index).is_visible() is True
            for index in range(overlays.count())
        )
    except Exception:
        return False


def post_captcha_destination_visible(page: Page) -> bool:
    if blocking_overlay_visible(page):
        return False
    return disclaimer_dialog_visible(page) or appointment_form_ready(page)


def background_submit_ready(page: Page) -> bool:
    if blocking_overlay_visible(page) or disclaimer_dialog_visible(page):
        return False
    try:
        button = page.locator(BACKGROUND_SUBMIT_BUTTON_SELECTOR).last
        return button.is_visible() is True and button.is_enabled() is True
    except Exception:
        return False


def wait_for_post_captcha_page_ready(
    page: Page,
    *,
    timeout_ms: int = 30_000,
) -> str:
    """Wait for overlays to clear while watching for valid destinations."""
    deadline = time.monotonic() + timeout_ms / 1_000
    while time.monotonic() < deadline:
        raise_for_http_forbidden(page)
        if site_error_page_visible(page):
            raise RuntimeError(
                "Target site returned its temporary processing-error page."
            )
        if not blocking_overlay_visible(page):
            if post_captcha_destination_visible(page):
                return "destination"
            if background_submit_ready(page):
                return "ready"
        page.wait_for_timeout(250)
    raise RuntimeError(
        "Neither a usable background Submit nor a post-CAPTCHA destination "
        "appeared after the loading overlay cleared."
    )


def find_true_captcha_label(page: Page) -> tuple[Locator, str, str]:
    return find_true_captcha_label_in_scope(page)


def find_true_captcha_label_in_scope(scope) -> tuple[Locator, str, str]:
    labels = scope.locator(CAPTCHA_LABEL_SELECTOR)
    expect(labels).not_to_have_count(0, timeout=30_000)

    candidates: list[dict[str, object]] = []
    for index in range(labels.count()):
        label = labels.nth(index)
        text = label.inner_text().strip()
        match = CAPTCHA_INSTRUCTION_PATTERN.fullmatch(text)
        if match is None:
            continue
        element_id = (label.get_attribute("id") or "").strip()
        if not element_id:
            continue
        render_data = label.evaluate(
            """element => {
                const style = window.getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                const parsedZ = Number.parseInt(style.zIndex, 10);
                return {
                    zIndex: Number.isNaN(parsedZ) ? 0 : parsedZ,
                    display: style.display,
                    visibility: style.visibility,
                    opacity: Number.parseFloat(style.opacity || "1"),
                    width: rect.width,
                    height: rect.height,
                    x: rect.x,
                    y: rect.y
                };
            }"""
        )
        if (
            render_data["display"] == "none"
            or render_data["visibility"] == "hidden"
            or float(render_data["opacity"]) <= 0
            or float(render_data["width"]) <= 0
            or float(render_data["height"]) <= 0
        ):
            continue
        candidates.append(
            {
                "locator": label,
                "index": index,
                "id": element_id,
                "text": text,
                "target": match.group(1),
                "z_index": int(render_data["zIndex"]),
                "x": float(render_data["x"]),
                "y": float(render_data["y"]),
                "width": float(render_data["width"]),
                "height": float(render_data["height"]),
            }
        )
    if not candidates:
        raise RuntimeError(
            "No rendered CAPTCHA instruction candidates matched the expected "
            "three-digit instruction format."
        )
    highest_z = max(int(candidate["z_index"]) for candidate in candidates)
    top_candidates = [
        candidate for candidate in candidates if int(candidate["z_index"]) == highest_z
    ]
    if len(top_candidates) != 1:
        raise RuntimeError(
            f"Could not identify one unique top CAPTCHA label. Highest z-index was {highest_z}."
        )

    winner = top_candidates[0]
    logger.info(
        "Resolved true CAPTCHA label. id=%s target=%s z_index=%s",
        winner["id"],
        winner["target"],
        winner["z_index"],
    )
    return winner["locator"], str(winner["id"]), str(winner["target"])


def save_captcha_crop(page: Page, output_path: Path) -> np.ndarray:
    screenshot_bytes = page.screenshot(full_page=True)
    screenshot = cv2.imdecode(
        np.frombuffer(screenshot_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if screenshot is None:
        raise RuntimeError("Playwright returned an unreadable screenshot")
    candidates, _edges = find_square_candidates(screenshot)
    grid_boxes = select_grid_boxes(candidates)
    if len(grid_boxes) != 9:
        raise RuntimeError(f"Expected 9 CAPTCHA grid boxes, found {len(grid_boxes)}")
    grid_box = bounding_rectangle(grid_boxes)
    captcha_box = expand_box(
        grid_box,
        screenshot.shape,
        left_ratio=0.04,
        top_ratio=0.23,
        right_ratio=0.04,
        bottom_ratio=0.04,
    )
    captcha = crop_box(screenshot, captcha_box)
    if captcha is None or captcha.size == 0:
        raise RuntimeError("The CAPTCHA crop is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), captcha):
        raise OSError(f"Could not write CAPTCHA image: {output_path}")

    logger.info("Saved CAPTCHA crop: %s", output_path)
    return captcha


def get_captcha_tiles(page: Page) -> list[Locator]:
    return get_captcha_tiles_in_scope(page)


def get_captcha_tiles_in_scope(scope) -> list[Locator]:
    tiles = scope.locator(CAPTCHA_TILE_SELECTOR)
    expect(tiles).not_to_have_count(0, timeout=30_000)
    candidates = tiles.evaluate_all(
        """elements => elements.map((element, index) => {
            const parent = element.parentElement;
            const parentStyle = parent ? window.getComputedStyle(parent) : null;
            const style = window.getComputedStyle(element);
            const parentRect = parent ? parent.getBoundingClientRect() : element.getBoundingClientRect();
            const parsedZ = Number.parseInt(style.zIndex, 10);
            const parsedParentZ = parentStyle ? Number.parseInt(parentStyle.zIndex, 10) : NaN;
            return {
                index,
                id: parent ? (parent.id || "") : "",
                display: style.display,
                visibility: style.visibility,
                opacity: Number.parseFloat(style.opacity || "1"),
                z_index: Number.isNaN(parsedZ) ? 0 : parsedZ,
                parent_z_index: Number.isNaN(parsedParentZ) ? 0 : parsedParentZ,
                left: parentRect.left,
                top: parentRect.top,
                width: parentRect.width,
                height: parentRect.height
            };
        })"""
    )
    visible_candidates = [
        candidate
        for candidate in candidates
        if candidate["display"] != "none"
        and candidate["visibility"] != "hidden"
        and float(candidate["opacity"]) > 0
        and float(candidate["width"]) > 0
        and float(candidate["height"]) > 0
    ]
    if not visible_candidates:
        raise RuntimeError("No visible CAPTCHA tile candidates were found.")
    grouped: dict[tuple[int, int, int, int], dict[str, object]] = {}
    for candidate in visible_candidates:
        key = (
            int(round(float(candidate["left"]))),
            int(round(float(candidate["top"]))),
            int(round(float(candidate["width"]))),
            int(round(float(candidate["height"]))),
        )
        score = int(candidate["z_index"]) + int(candidate["parent_z_index"])
        previous = grouped.get(key)
        if previous is None or score > int(previous["score"]):
            grouped[key] = {"score": score, "candidate": candidate}
    chosen = [item["candidate"] for item in grouped.values()]
    chosen.sort(key=lambda item: (round(float(item["top"]), 2), round(float(item["left"]), 2)))
    if len(chosen) != 9:
        raise RuntimeError(f"Expected 9 visible CAPTCHA tiles, found {len(chosen)}.")
    result: list[Locator] = []
    for candidate in chosen:
        tile_id = str(candidate["id"])
        if tile_id:
            result.append(scope.locator(f"#{tile_id}"))
    if len(result) != 9:
        raise RuntimeError(f"Resolved {len(result)} clickable tiles after deduping, expected 9.")
    logger.info("Resolved 9 visible CAPTCHA tiles.")
    return result


def wait_for_captcha_tiles_ready(
    page: Page,
    scope,
    *,
    expected_count: int = 9,
    timeout: int = 30_000,
    settle_ms: int = 500,
) -> list[Locator]:
    """
    Wait until the CAPTCHA grid is genuinely ready to screenshot.

    Readiness means:
    - exactly `expected_count` visible/clickable tiles resolve;
    - any tile backed by an <img> has finished loading;
    - every loaded image has non-zero natural dimensions.

    A short settle delay is added after readiness so the browser has
    time to finish painting the completed grid before the screenshot.
    """

    logger.info(
        "Waiting for CAPTCHA tiles to load. "
        "Expected count=%s",
        expected_count,
    )

    deadline = (
        time.monotonic()
        + timeout / 1000
    )

    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            tiles = get_captcha_tiles_in_scope(
                scope
            )

            if len(tiles) != expected_count:
                raise RuntimeError(
                    "Expected "
                    f"{expected_count} CAPTCHA tiles, "
                    f"resolved {len(tiles)}."
                )

            all_images_ready = True

            for tile in tiles:
                image = tile.locator("img").first

                if image.count() == 0:
                    # CAPTCHA_TILE_SELECTOR itself may already
                    # point at the image rather than its parent.
                    candidate = tile

                else:
                    candidate = image

                ready = candidate.evaluate(
                    """element => {
                        if (
                            element instanceof HTMLImageElement
                        ) {
                            return (
                                element.complete &&
                                element.naturalWidth > 0 &&
                                element.naturalHeight > 0
                            );
                        }

                        const img = element.querySelector("img");

                        if (!img) {
                            return true;
                        }

                        return (
                            img.complete &&
                            img.naturalWidth > 0 &&
                            img.naturalHeight > 0
                        );
                    }"""
                )

                if not ready:
                    all_images_ready = False
                    break

            if not all_images_ready:
                raise RuntimeError(
                    "All 9 CAPTCHA elements exist, "
                    "but one or more tile images "
                    "are still loading."
                )

            logger.info(
                "CAPTCHA grid ready. "
                "Resolved %s fully loaded tiles.",
                len(tiles),
            )

            if settle_ms > 0:
                page.wait_for_timeout(
                    settle_ms
                )

            return tiles

        except Exception as error:
            last_error = error

            page.wait_for_timeout(
                250
            )

    raise RuntimeError(
        "CAPTCHA grid did not become ready "
        f"within {timeout / 1000:.1f} seconds. "
        f"Last state: {last_error}"
    ) from last_error


def click_selected_captcha_tiles(
    page: Page,
    selected_tiles: tuple[int, ...],
    *,
    tiles: list[Locator] | None = None,
) -> None:
    if tiles is None:
        tiles = get_captcha_tiles(page)

    for tile_number in selected_tiles:
        if tile_number < 1 or tile_number > len(tiles):
            raise ValueError(f"Selected tile {tile_number} is outside the 1..{len(tiles)} range")

    logger.info(
        "Waiting %.1fs before CAPTCHA tile interaction.",
        CAPTCHA_PRE_CLICK_SETTLE_MS / 1_000,
    )
    page.wait_for_timeout(CAPTCHA_PRE_CLICK_SETTLE_MS)

    for index, tile_number in enumerate(selected_tiles):
        click_captcha_tile(page, tiles[tile_number - 1], tile_number)

        if index < len(selected_tiles) - 1:
            delay_ms = random.randint(
                CAPTCHA_INTER_TILE_DELAY_MIN_MS,
                CAPTCHA_INTER_TILE_DELAY_MAX_MS,
            )
            logger.info(
                "Waiting %.3fs before the next CAPTCHA tile.",
                delay_ms / 1_000,
            )
            page.wait_for_timeout(delay_ms)

    logger.info(
        "Waiting %.1fs after CAPTCHA tile selection before submit.",
        CAPTCHA_POST_SELECTION_SETTLE_MS / 1_000,
    )
    page.wait_for_timeout(CAPTCHA_POST_SELECTION_SETTLE_MS)


def _captcha_tile_state(tile: Locator) -> str:
    image = tile.locator("img.captcha-img").first
    target = image if image.count() else tile
    return target.evaluate(
        """element => {
            const parent = element.parentElement;
            const state = node => node ? {
                className: String(node.className || ""),
                style: node.getAttribute("style") || "",
                ariaPressed: node.getAttribute("aria-pressed") || "",
                ariaSelected: node.getAttribute("aria-selected") || "",
                dataSelected: node.getAttribute("data-selected") || ""
            } : null;
            return JSON.stringify({element: state(element), parent: state(parent)});
        }"""
    )


def click_captcha_tile(page: Page, tile: Locator, tile_number: int) -> None:
    """Click one tile and allow its selected state to settle."""
    image = tile.locator("img.captcha-img").first
    target = image if image.count() else tile
    before_state = _captcha_tile_state(tile)
    target.scroll_into_view_if_needed(timeout=10_000)
    target.click(timeout=10_000)
    logger.info("Clicked tile %s", tile_number)

    deadline = time.monotonic() + CAPTCHA_SELECTION_STATE_TIMEOUT_MS / 1_000
    while time.monotonic() < deadline:
        if _captcha_tile_state(tile) != before_state:
            logger.info("Confirmed visible state change for tile %s", tile_number)
            return
        page.wait_for_timeout(100)

    logger.warning(
        "Tile %s did not expose a visible selected-state change; "
        "proceeding without a second click.",
        tile_number,
    )


def click_verify_selection(page: Page) -> None:
    raise_for_http_forbidden(page)
    verify_button = page.locator(VERIFY_BUTTON_SELECTOR)
    expect(verify_button).to_be_visible(timeout=30_000)
    expect(verify_button).to_be_enabled(timeout=30_000)
    verify_button.scroll_into_view_if_needed(timeout=10_000)
    try:
        verify_button.click(timeout=10_000)
    except PlaywrightTimeoutError:
        raise_for_http_forbidden(page)
        logger.warning(
            "Verify Selection click timed out while navigation was settling; "
            "continuing with post-click state detection."
        )
        return
    logger.info("Clicked Verify Selection")


def captcha_verification_succeeded(page: Page) -> bool:
    book_now = page.locator(BOOK_NOW_SELECTOR).filter(has_text="Book Now").first
    try:
        return book_now.is_visible()
    except Exception:
        return False


def wait_for_book_now(page: Page) -> None:
    book_now = page.locator(BOOK_NOW_SELECTOR).filter(has_text="Book Now").first
    expect(book_now).to_be_visible(timeout=60_000)
    logger.info("Book New Appointment is visible")


def click_nav_book_new_appointment(page: Page) -> None:
    wait_for_preloader_to_clear(page, timeout=90_000)
    nav_link = page.locator(NAV_BOOK_NEW_APPOINTMENT_SELECTOR)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        raise_for_http_forbidden(page)
        if site_error_page_visible(page):
            raise RuntimeError(
                "Target site returned its temporary processing-error page."
            )
        if nav_link.is_visible():
            break
        page.wait_for_timeout(250)
    else:
        raise RuntimeError(
            "Book New Appointment link did not become visible within 60 seconds."
        )
    expect(nav_link).to_be_enabled(timeout=60_000)
    try:
        nav_link.click(timeout=30_000)
    except PlaywrightTimeoutError:
        raise_for_http_forbidden(page)
        logger.warning(
            "Book New Appointment click timed out while navigation was "
            "settling; continuing with destination-state detection."
        )
        return
    logger.info("Clicked navbar Book New Appointment")


def captcha_instruction_present(page: Page) -> bool:
    return page.locator(CAPTCHA_LABEL_SELECTOR).count() > 0


def login_captcha_invalid(page: Page) -> bool:
    try:
        return page.get_by_text("Invalid captcha selection").first.is_visible()
    except Exception:
        return False


def login_captcha_succeeded(page: Page) -> bool:
    try:
        if "logincaptcha" in page.url.lower():
            return False
        return page.locator(NAV_BOOK_NEW_APPOINTMENT_SELECTOR).first.is_visible()
    except Exception:
        return False


def click_book_now(page: Page) -> None:
    wait_for_preloader_to_clear(page, timeout=90_000)
    book_now = page.locator(BOOK_NOW_SELECTOR).filter(
        has_text="Book Now"
    ).first
    expect(book_now).to_be_visible(timeout=30_000)
    expect(book_now).to_be_enabled(timeout=30_000)
    book_now.evaluate("(element) => element.click()")
    logger.info("Clicked Book Now")


def click_ok_dialog(page: Page) -> bool:
    modal = page.locator("#disclaimarModal").first
    ok_button = modal.locator('button:has-text("Ok"):visible').first
    deadline = time.monotonic() + 30

    while time.monotonic() < deadline:
        raise_for_http_forbidden(page)
        if site_error_page_visible(page):
            raise RuntimeError(
                "Target site returned its temporary processing-error page."
            )
        if disclaimer_dialog_visible(page):
            if ok_button.is_visible() and ok_button.is_enabled():
                break
        elif appointment_form_ready(page):
            logger.info(
                "Appointment form is already visible; no disclaimer dialog is required."
            )
            return False
        page.wait_for_timeout(250)
    else:
        raise RuntimeError(
            "Neither a usable disclaimer OK button nor an unobstructed "
            "appointment form appeared within 30 seconds."
        )

    ok_button.scroll_into_view_if_needed(timeout=10_000)

    try:
        ok_button.click(timeout=10_000)
    except Exception:
        raise_for_http_forbidden(page)
        logger.warning("Normal OK click failed; falling back to DOM click.")
        ok_button.evaluate("(element) => element.click()")
    try:
        expect(modal).to_be_hidden(timeout=30_000)
    except Exception as error:
        raise_for_http_forbidden(page)
        raise RuntimeError("Disclaimer dialog remained open after OK.") from error
    if not appointment_form_ready(page):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            raise_for_http_forbidden(page)
            if site_error_page_visible(page):
                raise RuntimeError(
                    "Target site returned its temporary processing-error page."
                )
            if appointment_form_ready(page):
                break
            page.wait_for_timeout(250)
        else:
            raise RuntimeError(
                "Appointment form did not become usable after disclaimer OK."
            )
    logger.info("Clicked OK dialog button")
    return True


def click_submit_selection(page: Page) -> None:
    submit_selection = page.locator(SECOND_CAPTCHA_SUBMIT_SELECTOR)
    expect(submit_selection).to_be_visible(timeout=30_000)
    expect(submit_selection).to_be_enabled(timeout=30_000)
    submit_selection.click(timeout=10_000)
    logger.info("Clicked Submit Selection")


def click_background_submit(page: Page) -> None:
    background_submit = page.locator(BACKGROUND_SUBMIT_BUTTON_SELECTOR).last
    raise_for_http_forbidden(page)
    if post_captcha_destination_visible(page):
        logger.info(
            "Post-CAPTCHA destination is already visible; background Submit is complete."
        )
        return
    if site_error_page_visible(page):
        raise RuntimeError(
            "Target site returned its temporary processing-error page."
        )
    expect(background_submit).to_be_visible(timeout=30_000)
    expect(background_submit).to_be_enabled(timeout=30_000)
    logger.info(
        "Waiting %.1fs for the verified state to settle before background Submit.",
        POST_SECOND_CAPTCHA_SETTLE_MS / 1_000,
    )
    page.wait_for_timeout(POST_SECOND_CAPTCHA_SETTLE_MS)
    if wait_for_post_captcha_page_ready(page) == "destination":
        logger.info(
            "Post-CAPTCHA destination appeared while waiting for the overlay."
        )
        return
    try:
        background_submit.click(timeout=10_000)
    except Exception as first_error:
        raise_for_http_forbidden(page)
        state = wait_for_post_captcha_page_ready(page)
        if state == "destination":
            logger.info(
                "Background Submit reached its destination while the click "
                "was still settling."
            )
            return
        logger.warning(
            "Background Submit click failed after the overlay cleared and no "
            "destination appeared; retrying it once. Error=%r",
            first_error,
        )
        background_submit.click(timeout=10_000)
    logger.info("Clicked background Submit")

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        raise_for_http_forbidden(page)
        if post_captcha_destination_visible(page):
            return
        if site_error_page_visible(page):
            raise RuntimeError(
                "Target site returned its temporary processing-error page."
            )
        page.wait_for_timeout(250)

    if background_submit.is_visible() and background_submit.is_enabled():
        logger.warning(
            "Background Submit produced no visible transition; waiting for "
            "any overlay to clear before one final state check."
        )
        state = wait_for_post_captcha_page_ready(page)
        if state == "destination":
            return
        logger.warning(
            "No valid destination appeared after the overlay cleared; "
            "retrying background Submit once."
        )
        background_submit.click(timeout=10_000)
        logger.info("Retried background Submit once")
        if wait_for_post_captcha_page_ready(page) == "destination":
            return
    raise RuntimeError(
        "Background Submit did not reach an unobstructed disclaimer or form."
    )


def get_verify_selection_frame(page: Page) -> FrameLocator:
    frame = page.frame_locator('iframe[title="Verify Selection"]')
    expect(frame.locator("body")).to_be_visible(timeout=60_000)
    logger.info("Verify Selection iframe is visible.")
    return frame
