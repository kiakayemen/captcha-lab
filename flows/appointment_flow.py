from __future__ import annotations

import logging
import re

from playwright.sync_api import Locator, Page, expect

from .selectors import (
    NO_APPOINTMENTS_BODY_SELECTOR,
    NO_APPOINTMENTS_HEADER_SELECTOR,
    NO_APPOINTMENTS_MODAL_SELECTOR,
)


logger = logging.getLogger("captcha_lab")

JURISDICTION = "Tehran"
LOCATION = "Tehran"
VISA_TYPE = "National Visa/ Long Term Visa"
APPOINTMENT_CATEGORY = "Normal"

ALLOWED_VISA_SUB_TYPES = {
    "Student Visa",
    "Non-Working Residence Visa",
}


def _log(message: str) -> None:
    logger.info("[appointment] %s", message)


def _describe_container(container: Locator, label_text: str) -> None:
    try:
        label = container.locator("label.form-label").first
        label_text_value = label.inner_text().strip() if label.count() else "<missing>"
        hidden_input = container.locator('input[data-role="dropdownlist"]').first
        widget = container.locator("span.k-widget.k-dropdown:visible").first
        input_id = hidden_input.get_attribute("id") if hidden_input.count() else None
        widget_text = ""
        if widget.count():
            widget_text = widget.locator("span.k-input").inner_text().strip()
        _log(
            f'container for "{label_text}": label={label_text_value!r}, '
            f"id={input_id!r}, widget_text={widget_text!r}"
        )
    except Exception as error:
        _log(f'failed to describe container for "{label_text}": {error!r}')


def _find_visible_dropdown_container(
    page: Page,
    label_text: str,
) -> Locator:
    containers = page.locator("div.mb-3:visible")

    matches = []

    pattern = re.compile(
        rf"^\s*{re.escape(label_text)}\s*\*?\s*$",
        re.IGNORECASE,
    )

    for i in range(containers.count()):
        container = containers.nth(i)
        label = container.locator("label.form-label")

        if label.count() == 0:
            continue
        if not pattern.match(label.first.inner_text().strip()):
            continue

        hidden_input = container.locator(
            'input[data-role="dropdownlist"]'
        )

        visible_widget = container.locator(
            "span.k-widget.k-dropdown:visible"
        )

        if hidden_input.count() == 1 and visible_widget.count() == 1:
            matches.append(container)
            _describe_container(container, label_text)
    if len(matches) != 1:
        _log(
            f'expected one visible "{label_text}" dropdown, found {len(matches)}'
        )
        raise RuntimeError(
            f'Expected one visible "{label_text}" dropdown, '
            f"found {len(matches)}"
        )

    return matches[0]


def _get_visible_dropdown_id(
    page: Page,
    label_text: str,
) -> str:
    container = _find_visible_dropdown_container(
        page,
        label_text,
    )
    input_id = container.locator(
        'input[data-role="dropdownlist"]'
    ).get_attribute("id")

    if not input_id:
        raise RuntimeError(
            f'Could not resolve ID for "{label_text}"'
        )

    return input_id


def _wait_for_kendo_data(
    page: Page,
    label_text: str,
    *,
    expected_text: str | None = None,
) -> None:
    input_id = _get_visible_dropdown_id(
        page,
        label_text,
    )
    _log(f'waiting for Kendo data for "{label_text}" (id={input_id})')

    page.wait_for_function(
        """
        ({ id, expected }) => {
            const el = document.getElementById(id);

            if (!el || !window.jQuery) {
                return false;
            }

            const ddl = window.jQuery(el).data("kendoDropDownList");

            if (!ddl || !ddl.dataSource) {
                return false;
            }

            const items = ddl.dataSource.view();
            if (!items || items.length === 0) {
                return false;
            }

            if (!expected) {
                return true;
            }

            return items.some(item => {
                const text =
                    item.Name ||
                    item.Text ||
                    item.text ||
                    "";
                return text.trim() === expected;
            });
        }
        """,
        arg={
            "id": input_id,
            "expected": expected_text,
        },
        timeout=30_000,
    )
    _log(f'Kendo data ready for "{label_text}"')


def _select_kendo_option(
    page: Page,
    label_text: str,
    option_text: str,
) -> None:
    container = _find_visible_dropdown_container(
        page,
        label_text,
    )
    hidden_input = container.locator(
        'input[data-role="dropdownlist"]'
    )

    input_id = hidden_input.get_attribute("id")

    if not input_id:
        raise RuntimeError(
            f'Visible "{label_text}" dropdown has no ID'
        )

    widget = container.locator(
        "span.k-widget.k-dropdown:visible"
    )
    expect(widget).to_be_visible(timeout=30_000)
    popup = page.locator(f"#{input_id}-list")
    option_pattern = re.compile(
        rf"^\s*{re.escape(option_text)}\s*$",
        re.IGNORECASE,
    )

    max_attempts = 5
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        _log(
            f'opening "{label_text}" dropdown attempt {attempt}/{max_attempts} '
            f"(id={input_id}, option={option_text!r})"
        )
        try:
            widget.scroll_into_view_if_needed(timeout=10_000)
            _log(f'clicking "{label_text}" dropdown handle')
            widget.click(timeout=10_000)
            page.wait_for_timeout(250)
            if not popup.is_visible():
                _log(
                    f"popup #{input_id}-list still hidden after click, "
                    f'retrying click for "{label_text}"'
                )
                widget.click(timeout=10_000)
                page.wait_for_timeout(250)
            _log(f"waiting for popup #{input_id}-list to become visible")
            expect(popup).to_be_visible(timeout=10_000)
            page.wait_for_timeout(750)
            _log(f"popup #{input_id}-list visible; waiting briefly before reading options")

            option = popup.locator("li.k-item").filter(
                has_text=option_pattern
            )
            option_count = option.count()
            _log(
                f'"{label_text}" popup option count for '
                f"{option_text!r}: {option_count}"
            )

            if option_count != 1:
                available = popup.locator("li.k-item").all_inner_texts()
                _log(f'available options for "{label_text}": {available}')
                raise RuntimeError(
                    f'Option {option_text!r} not visible yet in "{label_text}"'
                )
            option.wait_for(state="visible", timeout=10_000)
            _log(f'clicking option {option_text!r} for "{label_text}"')
            option.click(timeout=10_000)

            expect(popup).to_be_hidden(timeout=10_000)

            selected_text = widget.locator("span.k-input")

            expect(selected_text).to_have_text(
                option_pattern,
                timeout=10_000,
            )
            _log(f'"{label_text}" selected text confirmed as {option_text!r}')
            page.wait_for_function(
                """
                ({ id, expected }) => {
                    const el = document.getElementById(id);

                    if (!el || !window.jQuery) {
                        return false;
                    }

                    const ddl = window.jQuery(el).data("kendoDropDownList");

                    if (!ddl) {
                        return false;
                    }

                    const item = ddl.dataItem();
                    if (!item) {
                        return false;
                    }

                    const text =
                        item.Name ||
                        item.Text ||
                        item.text ||
                        "";
                    return text.trim() === expected;
                }
                """,
                arg={
                    "id": input_id,
                    "expected": option_text,
                },
                timeout=10_000,
            )
            return
        except Exception as error:
            last_error = error
            _log(
                f'attempt {attempt}/{max_attempts} failed for "{label_text}": '
                f"{error!r}"
            )
            if attempt < max_attempts:
                page.wait_for_timeout(500)

    raise RuntimeError(
        f'Failed to select "{option_text}" for "{label_text}" after '
        f"{max_attempts} attempts"
    ) from last_error


def _fill_label_driven_dropdown(
    page: Page,
    label_text: str,
    option_text: str,
) -> None:
    _log(f'finding label "{label_text}"')
    _select_kendo_option(
        page,
        label_text,
        option_text,
    )


def fill_appointment_form(
    page: Page,
    *,
    visa_sub_type: str,
) -> None:
    if visa_sub_type not in ALLOWED_VISA_SUB_TYPES:
        raise ValueError(
            f"Unsupported visa subtype: {visa_sub_type}"
        )

    _log("starting appointment form fill")

    _log('skipping "Appointment For" because it must remain at its default value')

    _fill_label_driven_dropdown(
        page,
        "Appointment Category",
        APPOINTMENT_CATEGORY,
    )

    _wait_for_kendo_data(
        page,
        "Jurisdiction",
        expected_text=JURISDICTION,
    )

    _fill_label_driven_dropdown(
        page,
        "Jurisdiction",
        JURISDICTION,
    )

    _fill_label_driven_dropdown(
        page,
        "Location",
        LOCATION,
    )

    _fill_label_driven_dropdown(
        page,
        "Visa Type",
        VISA_TYPE,
    )

    _fill_label_driven_dropdown(
        page,
        "Visa Sub Type",
        visa_sub_type,
    )

    _log("appointment form filled successfully")


def no_appointments_dialog_visible(page: Page) -> bool:
    try:
        modal = page.locator(NO_APPOINTMENTS_MODAL_SELECTOR).first
        if not modal.is_visible():
            return False

        header = page.locator(NO_APPOINTMENTS_HEADER_SELECTOR).first
        body = page.locator(NO_APPOINTMENTS_BODY_SELECTOR).first

        header_text = header.inner_text().strip() if header.count() else ""
        body_text = body.inner_text().strip() if body.count() else ""
        return "No Appointments Available" in header_text or bool(body_text)
    except Exception:
        logger.exception("[appointment] Failed while checking no-appointments dialog.")
        return False
