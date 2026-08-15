from __future__ import annotations
from flows.appointment_flow import fill_appointment_form

import argparse
import time
from pathlib import Path

from captcha_solver import print_decision, solve_captcha_image, write_outputs
from config import BLS_EMAIL, LOGIN_URL, BLS_PASSWORD
from flows.captcha_flow import (
    captcha_instruction_present,
    captcha_verification_succeeded,
    click_background_submit,
    click_nav_book_new_appointment,
    click_ok_dialog,
    click_selected_captcha_tiles,
    click_submit_selection,
    click_verify_selection,
    find_true_captcha_label_in_scope,
    get_captcha_tiles_in_scope,
    get_verify_selection_frame,
    find_true_captcha_label,
    save_captcha_crop,
)
from flows.login_flow import fill_visible_password, submit_email
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
        fill_password(page, target_password=BLS_PASSWORD)

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

def run_second_captcha_step(page, *, gpu: bool, output_dir: Path) -> None:
    frame = get_verify_selection_frame(page)
    screenshot_path = Path("captcha_page_2.png")
    max_attempts = 3
    popup = page.locator('div.k-widget.k-window').filter(has_text="Verify Selection").first

    for attempt in range(1, max_attempts + 1):
        print(f"Second captcha attempt {attempt}/{max_attempts}")

        print("Waiting for the second CAPTCHA instruction...")
        _true_label, true_label_id, target = find_true_captcha_label_in_scope(frame)

        true_label_by_id = frame.locator(f"#{true_label_id}")
        expect(true_label_by_id).to_have_count(1)
        expect(true_label_by_id).to_be_visible()

        confirmed_text = true_label_by_id.inner_text().strip()
        confirmed_match = CAPTCHA_INSTRUCTION_PATTERN.fullmatch(confirmed_text)
        if confirmed_match is None or confirmed_match.group(1) != target:
            raise RuntimeError("The discovered second CAPTCHA label changed before solving.")

        print(f"Second CAPTCHA label ID: {true_label_id}")
        print(f"Second target extracted from DOM: {target}")

        captcha_image = save_captcha_crop(page, screenshot_path)
        print(f"Cropped second CAPTCHA image saved: {screenshot_path}")

        print("Loading EasyOCR for second CAPTCHA...")
        solve_start = time.perf_counter()
        reader = build_reader(gpu=gpu)
        decision, tiles, _boxes, debug = solve_captcha_image(
            captcha_image,
            target=target,
            reader=reader,
        )
        solve_seconds = time.perf_counter() - solve_start
        print(f"Second captcha solve time: {solve_seconds:.3f}s")

        print_decision(decision)
        tiles_in_frame = get_captcha_tiles_in_scope(frame)
        for tile_number in decision.selected_tiles:
            if tile_number < 1 or tile_number > len(tiles_in_frame):
                raise ValueError(
                    f"Selected tile {tile_number} is outside the 1..{len(tiles_in_frame)} range"
                )
            tile = tiles_in_frame[tile_number - 1]
            tile.scroll_into_view_if_needed(timeout=10_000)
            tile.click(timeout=10_000)
            print(f"Clicked second-captcha tile {tile_number}")
        click_submit_selection(frame)

        verified_label = frame.locator("text=Verified!")
        try:
            expect(verified_label).to_be_visible(timeout=15_000)
            print('Second captcha returned "Verified!"')
        except Exception:
            print("Second captcha did not return Verified; retrying.")
            continue

        page.wait_for_timeout(5_000)
        if popup.is_visible():
            print("Second captcha popup stayed open after 5s; retrying.")
            continue

        print("Second captcha verification succeeded.")
        write_outputs(
            output_dir,
            captcha_image,
            debug,
            tiles,
            decision,
        )
        return

        print("Second captcha did not advance the page; retrying.")

    raise RuntimeError(f"Second captcha failed after {max_attempts} attempts.")


def fill_password(page, *, target_password: str) -> None:
    fill_visible_password(page, target_password)


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

    parser.add_argument(
        "--visa-sub-type",
        choices=[
            "Student Visa",
            "Non-Working Residence Visa",
        ],
        default="Student Visa",
        help=(
            "Visa subtype to use when filling the appointment form."
        ),
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
            run_captcha_step(page, gpu=args.gpu, output_dir=args.output)
            click_nav_book_new_appointment(page)
            click_verify_selection(page)
            run_second_captcha_step(
                page,
                gpu=args.gpu,
                output_dir=args.output,
            )
            click_background_submit(page)
            # Visa Type page automatically displays its disclaimer modal.
            click_ok_dialog(page)
            fill_appointment_form(
                page,
                visa_sub_type=args.visa_sub_type,
            )
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
