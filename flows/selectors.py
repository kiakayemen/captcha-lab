from __future__ import annotations

import re

LOGIN_FORM_SELECTOR = 'form[action="/Global/account/LoginSubmit"]'
EMAIL_INPUT_SELECTOR = f"{LOGIN_FORM_SELECTOR} input.entry-disabled:visible"
VERIFY_BUTTON_SELECTOR = "#btnVerify"
BOOK_NOW_SELECTOR = 'a[href="/Global/blsappointment/manageappointment"]'
NO_APPOINTMENTS_MODAL_SELECTOR = 'div.modal-dialog.modal-dialog-centered:has(#commonModalLabel)'
NO_APPOINTMENTS_HEADER_SELECTOR = "#commonModalHeader"
NO_APPOINTMENTS_BODY_SELECTOR = "#commonModalBody"
NAV_BOOK_NEW_APPOINTMENT_SELECTOR = 'a.nav-link.new-app-active[href="/Global/bls/visatypeverification"]'
OK_DIALOG_BUTTON_SELECTOR = (
    'div.modal-dialog.modal-dialog-centered:has(#commonModalLabel) '
    'button[data-bs-dismiss="modal"], '
    'div.modal-dialog.modal-dialog-centered:has(#commonModalLabel) '
    'button:has-text("Ok")'
)
SUBMIT_SELECTION_BUTTON_SELECTOR = 'button:has-text("Submit Selection")'
SECOND_CAPTCHA_SUBMIT_SELECTOR = 'div.img-action-div[onclick="onSubmit();"]'
BACKGROUND_SUBMIT_BUTTON_SELECTOR = 'button:has-text("Submit")'

CAPTCHA_LABEL_SELECTOR = "div.col-12.box-label"
CAPTCHA_INSTRUCTION_PATTERN = re.compile(
    r"^\s*Please\s+select\s+all\s+boxes\s+with\s+number\s+(\d{3})\s*$",
    re.IGNORECASE,
)
CAPTCHA_TILE_SELECTOR = "#captcha-main-div img.captcha-img"
