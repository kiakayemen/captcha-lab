from __future__ import annotations

import argparse
from pathlib import Path

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
        "--url",
        required=True,
        help="Exact BLS login URL.",
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Account email address.",
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
            print(f"Opening login page: {args.url}")

            response = page.goto(
                args.url,
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            if response is not None:
                print(f"Initial HTTP status: {response.status}")

            submit_email(page, args.email)

            # Wait for the redirected page to finish rendering.
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(2_000)

            page.screenshot(
                path=str(screenshot_path),
                full_page=True,
            )

            print(f"Redirected-page screenshot saved: {screenshot_path}")
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