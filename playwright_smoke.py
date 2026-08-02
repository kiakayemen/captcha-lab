from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from config import BLS_EMAIL, LOGIN_URL
from extract_tiles import (
    bounding_rectangle,
    crop_box,
    expand_box,
    find_square_candidates,
    select_grid_boxes,
)
from playwright.sync_api import (
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    expect,
    sync_playwright,
)


LOGIN_FORM_SELECTOR = 'form[action="/Global/account/LoginSubmit"]'
EMAIL_INPUT_SELECTOR = f"{LOGIN_FORM_SELECTOR} input.entry-disabled:visible"
VERIFY_BUTTON_SELECTOR = f"{LOGIN_FORM_SELECTOR} #btnVerify"


def save_captcha_crop(page: Page, output_path: Path) -> None:
    """Save the instruction and 3x3 grid, excluding the rest of the page."""
    screenshot_bytes = page.screenshot(full_page=True)
    screenshot = cv2.imdecode(
        np.frombuffer(screenshot_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if screenshot is None:
        raise RuntimeError("Playwright returned an unreadable screenshot")

    candidates, _edges = find_square_candidates(screenshot)
    grid_boxes = select_grid_boxes(candidates)
    grid_box = bounding_rectangle(grid_boxes)

    # Match the CAPTCHA-only crop used by the OCR pipeline: retain the prompt
    # above the tiles and a small border on the remaining three sides.
    captcha_box = expand_box(
        grid_box,
        screenshot.shape,
        left_ratio=0.04,
        top_ratio=0.23,
        right_ratio=0.04,
        bottom_ratio=0.04,
    )
    captcha = crop_box(screenshot, captcha_box)

    if not cv2.imwrite(str(output_path), captcha):
        raise OSError(f"Could not write CAPTCHA image: {output_path}")


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

    # Verify Playwright actually populated the field.
    expect(email_input).to_have_value(email)

    print("Email entered successfully.")

    verify_button = page.locator(VERIFY_BUTTON_SELECTOR)
    expect(verify_button).to_be_visible(timeout=30_000)
    expect(verify_button).to_be_enabled(timeout=30_000)

    print("Clicking Verify...")

    previous_url = page.url

    verify_button.click()

    # The form posts to LoginSubmit and should redirect afterward.
    try:
        page.wait_for_url(
            lambda url: url != previous_url,
            timeout=60_000,
            wait_until="domcontentloaded",
        )
    except PlaywrightTimeoutError:
        # Some navigation flows may update the page without an immediate URL
        # change, so collect diagnostics before failing.
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
        description="Enter the BLS account email and navigate to the CAPTCHA page."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying the browser.",
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

            # Wait for the redirected page to finish rendering.
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(2_000)

            save_captcha_crop(page, screenshot_path)

            print(f"Cropped CAPTCHA image saved: {screenshot_path}")
            print("Browser is paused on the page after email submission.")

            input("Press Enter to close the browser...")

        except PlaywrightTimeoutError as error:
            page.screenshot(
                path="playwright_timeout.png",
                full_page=True,
            )
            print(f"Playwright timed out: {error}")
            print(f"Current URL: {page.url}")
            print("Saved playwright_timeout.png")

        except (PlaywrightError, RuntimeError) as error:
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
