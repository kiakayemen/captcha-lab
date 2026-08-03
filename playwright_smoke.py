from __future__ import annotations

import argparse
from pathlib import Path

from captcha_solver import print_decision, solve_captcha_image, write_outputs
from config import BLS_EMAIL, LOGIN_URL, BLS_PASSWORD
from captcha_flow import (
    click_selected_captcha_tiles,
    find_true_captcha_label,
    save_captcha_crop,
)
from login_flow import find_visible_password_input, submit_email
from ocr import build_reader
from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    expect,
    sync_playwright,
)
from selectors import CAPTCHA_INSTRUCTION_PATTERN


def run_phase_one(page, *, target_password: str, gpu: bool, output_dir: Path) -> None:
    screenshot_path = Path("captcha_page.png")

    submit_email(page, BLS_EMAIL)
    page.wait_for_load_state("domcontentloaded")

    print("Waiting for the true CAPTCHA instruction...")
    _true_label, true_label_id, target = find_true_captcha_label(page)

    true_label_by_id = page.locator(f"#{true_label_id}")
    expect(true_label_by_id).to_have_count(1)
    expect(true_label_by_id).to_be_visible()

    confirmed_text = true_label_by_id.inner_text().strip()
    confirmed_match = CAPTCHA_INSTRUCTION_PATTERN.fullmatch(confirmed_text)
    if confirmed_match is None or confirmed_match.group(1) != target:
        raise RuntimeError("The discovered CAPTCHA label changed before solving.")

    print(f"True CAPTCHA label ID: {true_label_id}")
    print(f"Target extracted from DOM: {target}")

    captcha_image = save_captcha_crop(page, screenshot_path)
    print(f"Cropped CAPTCHA image saved: {screenshot_path}")

    print("Loading EasyOCR...")
    reader = build_reader(gpu=gpu)

    decision, tiles, _boxes, debug = solve_captcha_image(
        captcha_image,
        target=target,
        reader=reader,
    )

    print_decision(decision)
    click_selected_captcha_tiles(page, decision.selected_tiles)

    visible_input = find_visible_password_input(page)
    visible_input.fill(target_password)

    page.get_by_role("button", name="Submit").click(timeout=10_000)

    write_outputs(
        output_dir,
        captcha_image,
        debug,
        tiles,
        decision,
    )

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

            run_phase_one(
                page,
                target_password=BLS_PASSWORD,
                gpu=args.gpu,
                output_dir=args.output,
            )

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
