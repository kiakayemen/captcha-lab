from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from captcha_solver import print_decision, solve_captcha_image, write_outputs
from config import BLS_EMAIL, LOGIN_URL
from extract_tiles import (
    bounding_rectangle,
    crop_box,
    expand_box,
    find_square_candidates,
    select_grid_boxes,
)
from ocr import build_reader
from playwright.sync_api import (
    Error as PlaywrightError,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    expect,
    sync_playwright,
)


LOGIN_FORM_SELECTOR = 'form[action="/Global/account/LoginSubmit"]'
EMAIL_INPUT_SELECTOR = f"{LOGIN_FORM_SELECTOR} input.entry-disabled:visible"
VERIFY_BUTTON_SELECTOR = f"{LOGIN_FORM_SELECTOR} #btnVerify"

CAPTCHA_LABEL_SELECTOR = "div.col-12.box-label"
CAPTCHA_INSTRUCTION_PATTERN = re.compile(
    r"^\s*Please\s+select\s+all\s+boxes\s+with\s+number\s+(\d{3})\s*$",
    re.IGNORECASE,
)
CAPTCHA_TILE_SELECTOR = "#captcha-main-div img.captcha-img"


def find_true_captcha_label(page: Page) -> tuple[Locator, str, str]:
    """
    Find the rendered CAPTCHA instruction from a stack of overlapping decoys.

    The site keeps every instruction element "visible" according to Playwright.
    They occupy the same coordinates, and CSS z-index determines which label is
    actually painted on top. Therefore `Locator.is_visible()` cannot identify
    the real label; the candidate with the highest computed z-index is used.
    """
    labels = page.locator(CAPTCHA_LABEL_SELECTOR)
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
        candidate
        for candidate in candidates
        if int(candidate["z_index"]) == highest_z
    ]

    if len(top_candidates) != 1:
        diagnostics = [
            {
                "index": candidate["index"],
                "id": candidate["id"],
                "text": candidate["text"],
                "z_index": candidate["z_index"],
                "rect": (
                    candidate["x"],
                    candidate["y"],
                    candidate["width"],
                    candidate["height"],
                ),
            }
            for candidate in sorted(
                candidates,
                key=lambda item: int(item["z_index"]),
                reverse=True,
            )
        ]
        raise RuntimeError(
            "Could not identify one unique top CAPTCHA label. "
            f"Highest z-index was {highest_z} with "
            f"{len(top_candidates)} candidates. Candidates: {diagnostics}"
        )

    winner = top_candidates[0]
    return (
        winner["locator"],
        str(winner["id"]),
        str(winner["target"]),
    )


def save_captcha_crop(page: Page, output_path: Path) -> np.ndarray:
    """Save and return the instruction plus 3x3 grid."""
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
        raise RuntimeError(
            f"Expected 9 CAPTCHA grid boxes, found {len(grid_boxes)}"
        )

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

    return captcha


def get_captcha_tiles(page: Page) -> list[Locator]:
    """
    Return the nine visible CAPTCHA tiles in top-to-bottom, left-to-right order.

    The page duplicates the CAPTCHA markup multiple times. Each visual position
    can have several stacked copies, so we keep only the top-most copy for each
    screen position by comparing computed z-index and element geometry.
    """
    tiles = page.locator(CAPTCHA_TILE_SELECTOR)
    expect(tiles).not_to_have_count(0, timeout=30_000)

    candidates = tiles.evaluate_all(
        """elements => elements.map((element, index) => {
            const parent = element.parentElement;
            const parentStyle = parent ? window.getComputedStyle(parent) : null;
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();
            const parentRect = parent ? parent.getBoundingClientRect() : rect;
            const parsedZ = Number.parseInt(style.zIndex, 10);
            const parsedParentZ = parentStyle
                ? Number.parseInt(parentStyle.zIndex, 10)
                : NaN;

            return {
                index,
                id: parent ? (parent.id || "") : "",
                onclick: element.getAttribute("onclick") || "",
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
            grouped[key] = {
                "score": score,
                "candidate": candidate,
            }

    chosen = [
        item["candidate"]
        for item in grouped.values()
    ]

    chosen.sort(
        key=lambda item: (
            round(float(item["top"]), 2),
            round(float(item["left"]), 2),
        )
    )

    if len(chosen) != 9:
        diagnostics = [
            {
                "id": candidate["id"],
                "onclick": candidate["onclick"],
                "z_index": candidate["z_index"],
                "parent_z_index": candidate["parent_z_index"],
                "rect": (
                    candidate["left"],
                    candidate["top"],
                    candidate["width"],
                    candidate["height"],
                ),
            }
            for candidate in sorted(
                visible_candidates,
                key=lambda item: (
                    int(item["z_index"]) + int(item["parent_z_index"]),
                ),
                reverse=True,
            )[:15]
        ]
        raise RuntimeError(
            f"Expected 9 visible CAPTCHA tiles, found {len(chosen)}. "
            f"Candidates: {diagnostics}"
        )

    result: list[Locator] = []
    for candidate in chosen:
        tile_id = str(candidate["id"])
        if not tile_id:
            continue
        result.append(page.locator(f"#{tile_id}"))

    if len(result) != 9:
        raise RuntimeError(
            f"Resolved {len(result)} clickable tiles after deduping, expected 9."
        )

    return result


def click_selected_captcha_tiles(
    page: Page,
    selected_tiles: tuple[int, ...],
) -> None:
    tiles = get_captcha_tiles(page)

    for tile_number in selected_tiles:
        if tile_number < 1 or tile_number > len(tiles):
            raise ValueError(
                f"Selected tile {tile_number} is outside the 1..{len(tiles)} range"
            )

        tile = tiles[tile_number - 1]
        tile.scroll_into_view_if_needed(timeout=10_000)
        tile.click(timeout=10_000)
        print(f"Clicked tile {tile_number}")


def submit_email(page: Page, email: str) -> None:
    print("Waiting for the login form...")

    form = page.locator(LOGIN_FORM_SELECTOR)
    expect(form).to_be_visible(timeout=60_000)

    print("Waiting for the visible email field...")

    email_input = page.locator(EMAIL_INPUT_SELECTOR)

    # The page contains multiple decoy inputs. Exactly one should be visible.
    expect(email_input).to_have_count(1, timeout=30_000)
    expect(email_input).to_be_visible(timeout=30_000)
    expect(email_input).to_be_editable(timeout=30_000)

    visible_id = email_input.get_attribute("id")
    visible_name = email_input.get_attribute("name")

    print(f"Visible email input ID: {visible_id}")
    print(f"Visible email input name: {visible_name}")

    email_input.fill(email)
    expect(email_input).to_have_value(email)

    print("Email entered successfully.")

    verify_button = page.locator(VERIFY_BUTTON_SELECTOR)
    expect(verify_button).to_be_visible(timeout=30_000)
    expect(verify_button).to_be_enabled(timeout=30_000)

    print("Clicking Verify...")

    previous_url = page.url
    verify_button.click()

    try:
        page.wait_for_url(
            lambda url: url != previous_url,
            timeout=60_000,
            wait_until="domcontentloaded",
        )
    except PlaywrightTimeoutError:
        page.screenshot(
            path="login_submit_timeout.png",
            full_page=True,
        )
        raise RuntimeError(
            "The URL did not change after clicking Verify. "
            "Saved login_submit_timeout.png for inspection."
        )

    print(f"Redirected to: {page.url}")
    print(f"Page title: {page.title()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enter the BLS account email, find the true CAPTCHA instruction, "
            "extract its target, crop the CAPTCHA, and solve its nine tiles."
        )
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying the browser.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Ask EasyOCR to use a supported GPU.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/live_solver"),
        help="Solver output directory.",
    )
    args = parser.parse_args()

    screenshot_path = Path("captcha_page.png")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=args.headless,
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
        )
        page = context.new_page()

        try:
            print(f"Opening login page: {LOGIN_URL}")

            response = page.goto(
                LOGIN_URL,
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            if response is not None:
                print(f"Initial HTTP status: {response.status}")

            submit_email(page, BLS_EMAIL)

            page.wait_for_load_state("domcontentloaded")

            print("Waiting for the true CAPTCHA instruction...")
            true_label, true_label_id, target = find_true_captcha_label(page)

            # Re-address the exact element by its ID after discovery. This
            # protects the rest of this page-load flow from the 25 decoys.
            true_label_by_id = page.locator(f"#{true_label_id}")
            expect(true_label_by_id).to_have_count(1)
            expect(true_label_by_id).to_be_visible()

            # Confirm the ID still points to the same target text.
            confirmed_text = true_label_by_id.inner_text().strip()
            confirmed_match = CAPTCHA_INSTRUCTION_PATTERN.fullmatch(
                confirmed_text
            )
            if confirmed_match is None or confirmed_match.group(1) != target:
                raise RuntimeError(
                    "The discovered CAPTCHA label changed before solving."
                )

            print(f"True CAPTCHA label ID: {true_label_id}")
            print(f"Target extracted from DOM: {target}")

            captcha_image = save_captcha_crop(page, screenshot_path)
            print(f"Cropped CAPTCHA image saved: {screenshot_path}")

            print("Loading EasyOCR...")
            reader = build_reader(gpu=args.gpu)

            decision, tiles, _boxes, debug = solve_captcha_image(
                captcha_image,
                target=target,
                reader=reader,
            )

            print_decision(decision)
            click_selected_captcha_tiles(page, decision.selected_tiles)
            write_outputs(
                args.output,
                captcha_image,
                debug,
                tiles,
                decision,
            )

            print(f"Decision file: {args.output / 'decision.json'}")
            print("Browser is paused on the CAPTCHA page.")
            input("Press Enter to close the browser...")

        except PlaywrightTimeoutError as error:
            page.screenshot(
                path="playwright_timeout.png",
                full_page=True,
            )
            print(f"Playwright timed out: {error}")
            print(f"Current URL: {page.url}")
            print("Saved playwright_timeout.png")

        except (PlaywrightError, RuntimeError, ValueError, OSError) as error:
            page.screenshot(
                path="playwright_error.png",
                full_page=True,
            )
            print(f"Automation failed: {error}")
            print(f"Current URL: {page.url}")
            print("Saved playwright_error.png")

        finally:
            context.close()
            browser.close()


if __name__ == "__main__":
    main()
