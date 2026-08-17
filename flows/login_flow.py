from __future__ import annotations

import logging

from playwright.sync_api import Locator, Page, TimeoutError as PlaywrightTimeoutError, expect

from config import BLS_EMAIL
from .selectors import EMAIL_INPUT_SELECTOR, LOGIN_FORM_SELECTOR, VERIFY_BUTTON_SELECTOR


logger = logging.getLogger("captcha_lab")


def submit_email(page: Page, email: str = BLS_EMAIL) -> None:
    logger.info("Waiting for the login form...")

    form = page.locator(LOGIN_FORM_SELECTOR)
    expect(form).to_be_visible(timeout=60_000)

    logger.info("Waiting for the visible email field...")
    email_input = page.locator(EMAIL_INPUT_SELECTOR)
    expect(email_input).to_have_count(1, timeout=30_000)
    expect(email_input).to_be_visible(timeout=30_000)
    expect(email_input).to_be_editable(timeout=30_000)

    visible_id = email_input.get_attribute("id")
    visible_name = email_input.get_attribute("name")
    logger.info("Visible email input ID: %s", visible_id)
    logger.info("Visible email input name: %s", visible_name)
    email_input.fill(email)
    expect(email_input).to_have_value(email)
    logger.info("Email entered successfully.")

    verify_button = page.locator(VERIFY_BUTTON_SELECTOR)
    expect(verify_button).to_be_visible(timeout=30_000)
    expect(verify_button).to_be_enabled(timeout=30_000)

    logger.info("Clicking Verify...")
    previous_url = page.url
    verify_button.click()
    try:
        page.wait_for_url(
            lambda url: url != previous_url,
            timeout=60_000,
            wait_until="domcontentloaded",
        )
    except PlaywrightTimeoutError:
        page.screenshot(path="login_submit_timeout.png", full_page=True)
        logger.exception(
            "The URL did not change after clicking Verify. "
            "Saved login_submit_timeout.png for inspection."
        )
        raise RuntimeError(
            "The URL did not change after clicking Verify. "
            "Saved login_submit_timeout.png for inspection."
        )
    logger.info("Redirected to: %s", page.url)
    logger.info("Page title: %s", page.title())


def find_visible_password_input(page: Page) -> Locator:
    visible_input = page.locator(
        'input.entry-disabled[type="password"]:visible'
    )
    expect(visible_input).to_have_count(1)
    input_id = visible_input.first.get_attribute("id")
    logger.info("Visible password input ID: %s", input_id)
    return visible_input.first


def fill_visible_password(page: Page, password: str) -> None:
    password_input = find_visible_password_input(page)
    password_input.scroll_into_view_if_needed(timeout=10_000)
    expect(password_input).to_be_editable(timeout=30_000)
    password_input.fill(password)
    expect(password_input).to_have_value(password)
    logger.info("Password entered successfully.")
