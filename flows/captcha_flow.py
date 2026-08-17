from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import FrameLocator, Locator, Page, expect
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


logger = logging.getLogger("captcha_lab")
PRELOADER_SELECTOR = "div.preloader"


def wait_for_preloader_to_clear(page: Page, timeout: int = 60_000) -> None:
    preloader = page.locator(PRELOADER_SELECTOR)
    try:
        expect(preloader).to_be_hidden(timeout=timeout)
    except Exception:
        logger.warning("Preloader did not clear cleanly; waiting an extra second.")
        page.wait_for_timeout(1_000)


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


def click_selected_captcha_tiles(page: Page, selected_tiles: tuple[int, ...]) -> None:
    tiles = get_captcha_tiles(page)
    for tile_number in selected_tiles:
        if tile_number < 1 or tile_number > len(tiles):
            raise ValueError(f"Selected tile {tile_number} is outside the 1..{len(tiles)} range")
        tile = tiles[tile_number - 1]
        tile.scroll_into_view_if_needed(timeout=10_000)
        tile.click(timeout=10_000)
        logger.info("Clicked tile %s", tile_number)


def click_verify_selection(page: Page) -> None:
    verify_button = page.locator(VERIFY_BUTTON_SELECTOR)
    expect(verify_button).to_be_visible(timeout=30_000)
    expect(verify_button).to_be_enabled(timeout=30_000)
    verify_button.scroll_into_view_if_needed(timeout=10_000)
    verify_button.click(timeout=10_000)
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
    expect(nav_link).to_be_visible(timeout=60_000)
    expect(nav_link).to_be_enabled(timeout=60_000)
    nav_link.click(timeout=30_000)
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
        if "logincaptcha" in page.url:
            return False
        return page.locator(LOGIN_FORM_SELECTOR).count() == 0
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


def click_ok_dialog(page: Page) -> None:
    ok_button = page.locator('button:has-text("Ok"):visible').first
    expect(ok_button).to_be_visible(timeout=30_000)
    expect(ok_button).to_be_enabled(timeout=30_000)
    ok_button.scroll_into_view_if_needed(timeout=10_000)

    try:
        ok_button.click(timeout=10_000)
    except Exception:
        logger.warning("Normal OK click failed; falling back to DOM click.")
        ok_button.evaluate("(element) => element.click()")
    modal = page.locator('div[role="dialog"]:visible').first
    try:
        expect(modal).to_be_hidden(timeout=30_000)
    except Exception:
        logger.warning("Dialog did not report hidden cleanly; waiting an extra second.")
        page.wait_for_timeout(1_000)
    logger.info("Clicked OK dialog button")


def click_submit_selection(page: Page) -> None:
    submit_selection = page.locator(SECOND_CAPTCHA_SUBMIT_SELECTOR)
    expect(submit_selection).to_be_visible(timeout=30_000)
    expect(submit_selection).to_be_enabled(timeout=30_000)
    submit_selection.click(timeout=10_000)
    logger.info("Clicked Submit Selection")


def click_background_submit(page: Page) -> None:
    background_submit = page.locator(BACKGROUND_SUBMIT_BUTTON_SELECTOR).last
    expect(background_submit).to_be_visible(timeout=30_000)
    expect(background_submit).to_be_enabled(timeout=30_000)
    background_submit.click(timeout=10_000)
    logger.info("Clicked background Submit")


def get_verify_selection_frame(page: Page) -> FrameLocator:
    frame = page.frame_locator('iframe[title="Verify Selection"]')
    expect(frame.locator("body")).to_be_visible(timeout=60_000)
    logger.info("Verify Selection iframe is visible.")
    return frame
