from __future__ import annotations

import json
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
    captcha_verification_succeeded,
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
from flows.login_flow import fill_visible_password, submit_email
from flows.selectors import CAPTCHA_INSTRUCTION_PATTERN
from notifications import log_no_appointment, notify_admin
from ocr import build_reader

from scraper.models import (
    ScraperConfig,
    ScraperResult,
    ScraperStatus,
)
