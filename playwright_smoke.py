from __future__ import annotations

import argparse
import time
from pathlib import Path

from captcha_solver import print_decision, solve_captcha_image, write_outputs
from config import BLS_EMAIL, LOGIN_URL, BLS_PASSWORD
from flows.captcha_flow import (
    captcha_instruction_present,
    captcha_verification_succeeded,
    click_book_now,
    click_ok_dialog,
    click_selected_captcha_tiles,
    click_verify_selection,
    find_true_captcha_label,
    save_captcha_crop,
)
from flows.login_flow import find_visible_password_input, submit_email
from ocr import build_reader
from playwright.sync_api import (
    Error as PlaywrightError,
    TimeoutError as PlaywrightTimeoutError,
    expect,
    sync_playwright,
)
from flows.selectors import CAPTCHA_INSTRUCTION_PATTERN


def run_login_step(page) -> None:
    submit_email(page, BLS_EMAIL)
    page.wait_for_load_state("domcontentloaded")


def run_captcha_step(page, *, gpu: bool, output_dir: Path) -> None:
    screenshot_path = Path("captcha_page.png")
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        print(f"Captcha attempt {attempt}/{max_attempts}")

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
        solve_start = time.perf_counter()
        reader = build_reader(gpu=gpu)

        decision, tiles, _boxes, debug = solve_captcha_image(
            captcha_image,
            target=target,
            reader=reader,
        )
        solve_seconds = time.perf_counter() - solve_start
        print(f"Captcha solve time: {solve_seconds:.3f}s")

        print_decision(decision)
        click_selected_captcha_tiles(page, decision.selected_tiles)
        click_verify_selection(page)

        if captcha_verification_succeeded(page):
            print("Captcha verification succeeded.")
            write_outputs(
                output_dir,
                captcha_image,
                debug,
                tiles,
                decision,
            )
            return

        if not captcha_instruction_present(page):
            print("Captcha instruction disappeared after Verify Selection; treating this as a transition.")
            return

        print("Captcha verification did not advance the page; retrying.")

    raise RuntimeError(f"Captcha verification failed after {max_attempts} attempts.")


def fill_password(page, *, target_password: str) -> None:
    visible_input = find_visible_password_input(page)
    visible_input.fill(target_password)


def submit_password(page) -> None:
    page.get_by_role("button", name="Submit").click(timeout=10_000)


def run_post_login_step(page, *, target_password: str) -> None:
    fill_password(page, target_password=target_password)
    submit_password(page)


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

            run_login_step(page)
            fill_password(page, target_password=BLS_PASSWORD)
            run_captcha_step(page, gpu=args.gpu, output_dir=args.output)
            click_book_now(page)
            click_ok_dialog(page)
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
