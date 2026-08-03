from __future__ import annotations

import re

LOGIN_FORM_SELECTOR = 'form[action="/Global/account/LoginSubmit"]'
EMAIL_INPUT_SELECTOR = f"{LOGIN_FORM_SELECTOR} input.entry-disabled:visible"
VERIFY_BUTTON_SELECTOR = "#btnVerify"
BOOK_NOW_SELECTOR = 'a[href="/Global/blsappointment/manageappointment"]'
OK_DIALOG_BUTTON_SELECTOR = '#alertModal button.btn.btn-success.btn-block'

CAPTCHA_LABEL_SELECTOR = "div.col-12.box-label"
CAPTCHA_INSTRUCTION_PATTERN = re.compile(
    r"^\s*Please\s+select\s+all\s+boxes\s+with\s+number\s+(\d{3})\s*$",
    re.IGNORECASE,
)
CAPTCHA_TILE_SELECTOR = "#captcha-main-div img.captcha-img"
