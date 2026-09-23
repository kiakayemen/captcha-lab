from datetime import datetime, timezone
from contextlib import ExitStack
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from config import BLS_EMAIL, BLS_PASSWORD, configured_bls_accounts

from flows.captcha_flow import (
    CAPTCHA_POST_SELECTION_SETTLE_MS,
    CAPTCHA_PRE_CLICK_SETTLE_MS,
    click_background_submit,
    click_ok_dialog,
    click_selected_captcha_tiles,
    click_verify_selection,
    wait_for_post_captcha_page_ready,
    wait_for_post_captcha_destination,
    background_submit_ready,
    appointment_form_ready,
    post_disclaimer_state,
    post_captcha_destination_visible,
    login_captcha_succeeded,
    SITE_ERROR_PATTERN,
)
from flows.appointment_flow import (
    _select_kendo_option,
    appointment_available_dialog_visible,
    no_appointments_dialog_visible,
)
from flows.errors import (
    HTTP403Forbidden,
    http_forbidden_page_visible,
    track_http_forbidden_responses,
)
from scraper.service import (
    SECOND_CAPTCHA_RETRY_SETTLE_MS,
    SecondCaptchaUnconfirmed,
    classify_second_captcha_state,
    second_captcha_challenge_signature,
    wait_for_regenerated_second_captcha,
    inspect_page_state,
    run_captcha_step,
    run_second_captcha_step,
    record_captcha_stage,
    wait_for_form_result,
    run_scraper,
    subtype_retry_delay_seconds,
    wait_for_login_captcha_outcome,
    restart_unclear_login_captcha,
    run_authenticated_appointment_cycle,
    reach_appointment_form,
    submit_appointment_form_and_wait,
    _run_login_captcha_attempt,
    _run_second_captcha_attempt,
    AppointmentResultUnconfirmed,
)
from scraper.http_diagnostics import EGRESS_IP_URL, resolve_egress_ip, response_diagnostics
from scraper.models import ScraperConfig, ScraperResult, ScraperStatus

from scraper.proxy import (
    PlaywrightProxyRotator,
    ProxyConfigurationError,
    choose_playwright_proxy,
    configured_proxy_urls,
    playwright_proxy_config,
)


class AccountConfigurationTests(TestCase):
    @patch.dict("os.environ", {"BLS_EMAIL_2": "second@example.com", "BLS_PASSWORD_2": "second-pass"})
    def test_loads_both_account_pairs(self):
        accounts = configured_bls_accounts()
        self.assertEqual(accounts[0].email, BLS_EMAIL)
        self.assertEqual(accounts[0].password, BLS_PASSWORD)
        self.assertEqual(accounts[1].email, "second@example.com")
        self.assertEqual(accounts[1].password, "second-pass")

    @patch.dict("os.environ", {"BLS_EMAIL_2": "second@example.com", "BLS_PASSWORD_2": ""})
    def test_rejects_incomplete_second_account(self):
        with self.assertRaisesRegex(RuntimeError, "must both be set"):
            configured_bls_accounts()


class CaptchaAttemptRegressionTests(TestCase):
    def test_login_attempt_reaches_tile_selection_without_second_captcha_frame(self):
        class ReachedTileSelection(Exception):
            pass

        page = MagicMock()
        page.locator.return_value.inner_text.return_value = "Please select all boxes with number 123"
        decision = MagicMock()
        decision.selected_tiles = (1,)
        decision.uncertain_tiles = ()

        def solve(_image, *, target, reader, timings):
            now = datetime.now(timezone.utc).isoformat()
            timings.update(ocr_started_at=now, ocr_finished_at=now, ocr_duration_ms=0)
            return decision, [MagicMock()] * 9, None, MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch("scraper.service.fill_password"))
            stack.enter_context(patch("scraper.service.find_true_captcha_label", return_value=(None, "label", "123")))
            stack.enter_context(patch("scraper.service.expect"))
            stack.enter_context(patch("scraper.service.wait_for_captcha_tiles_ready"))
            stack.enter_context(patch("scraper.service.save_captcha_crop", return_value=MagicMock()))
            stack.enter_context(patch("scraper.service.record_captcha_stage"))
            stack.enter_context(patch("scraper.service.solve_captcha_image", side_effect=solve))
            stack.enter_context(patch("scraper.service.log_captcha_decision"))
            stack.enter_context(patch("scraper.service.print_decision"))
            stack.enter_context(patch("scraper.service.click_selected_captcha_tiles", side_effect=ReachedTileSelection))

            with self.assertRaises(ReachedTileSelection):
                _run_login_captcha_attempt(
                    page, gpu=False, output_dir=MagicMock(), reader=MagicMock(), attempt_number=1
                )

    def test_second_attempt_snapshots_its_own_grid_before_tile_selection(self):
        class ReachedTileSelection(Exception):
            pass

        page = MagicMock()
        frame = MagicMock()
        frame.locator.return_value.inner_text.return_value = (
            "Please select all boxes with number 123"
        )
        decision = MagicMock()
        decision.selected_tiles = (1,)
        decision.uncertain_tiles = ()

        def solve(_image, *, target, reader, timings):
            now = datetime.now(timezone.utc).isoformat()
            timings.update(ocr_started_at=now, ocr_finished_at=now, ocr_duration_ms=0)
            return decision, [MagicMock()] * 9, None, MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch("scraper.service.get_verify_selection_frame", return_value=frame))
            stack.enter_context(patch("scraper.service.find_true_captcha_label_in_scope", return_value=(None, "label", "123")))
            stack.enter_context(patch("scraper.service.expect"))
            stack.enter_context(patch("scraper.service.wait_for_captcha_tiles_ready", return_value=[MagicMock()] * 9))
            stack.enter_context(patch("scraper.service.save_captcha_crop", return_value=MagicMock()))
            stack.enter_context(patch("scraper.service.record_captcha_stage"))
            stack.enter_context(patch("scraper.service.solve_captcha_image", side_effect=solve))
            stack.enter_context(patch("scraper.service.log_captcha_decision"))
            stack.enter_context(patch("scraper.service.print_decision"))
            signature = stack.enter_context(patch("scraper.service.second_captcha_challenge_signature", return_value="original"))
            stack.enter_context(patch("scraper.service.click_selected_captcha_tiles", side_effect=ReachedTileSelection))

            with self.assertRaises(ReachedTileSelection):
                _run_second_captcha_attempt(
                    page, gpu=False, output_dir=MagicMock(), reader=MagicMock(), attempt_number=1
                )

            signature.assert_called_once_with(frame)


class ProxyConfigurationTests(TestCase):
    @patch.dict(
        "os.environ",
        {
            "SCRAPER_PROXY_URLS": (
                "http://proxy-1.internal:8888,\n"
                "http://proxy-2.internal:8888"
            )
        },
        clear=False,
    )
    def test_configured_proxy_urls_accepts_commas_and_newlines(self):
        self.assertEqual(
            configured_proxy_urls(),
            (
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
            ),
        )

    @patch.dict(
        "os.environ",
        {
            "SCRAPER_PROXY_URLS": (
                "http://proxy-1.internal:8888,"
                "http://proxy-1.internal:8888"
            )
        },
        clear=False,
    )
    def test_configured_proxy_urls_removes_duplicates(self):
        self.assertEqual(
            configured_proxy_urls(),
            ("http://proxy-1.internal:8888",),
        )

    @patch.dict(
        "os.environ",
        {
            "SCRAPER_PROXY_URLS": (
                "http://captcha:secret@proxy-1.internal:8888,"
                "http://captcha:secret@proxy-1.internal:8888/"
            )
        },
        clear=False,
    )
    def test_trailing_slash_cannot_disguise_one_proxy_as_two(self):
        self.assertEqual(
            configured_proxy_urls(),
            ("http://captcha:secret@proxy-1.internal:8888",),
        )
        with self.assertRaises(ProxyConfigurationError):
            PlaywrightProxyRotator().validate_required_pool()

    def test_playwright_proxy_config_separates_credentials(self):
        self.assertEqual(
            playwright_proxy_config(
                "http://proxy-user:proxy-pass@proxy.internal:8888"
            ),
            {
                "server": "http://proxy.internal:8888",
                "username": "proxy-user",
                "password": "proxy-pass",
            },
        )

    def test_playwright_proxy_config_decodes_special_characters(self):
        self.assertEqual(
            playwright_proxy_config(
                "http://root%40proxy:p%21%40%5B%5B@proxy.internal:8888"
            ),
            {
                "server": "http://proxy.internal:8888",
                "username": "root@proxy",
                "password": "p!@[[",
            },
        )

    @patch.dict(
        "os.environ",
        {
            "SCRAPER_PROXY_URLS": (
                "http://proxy-1.internal:8888,"
                "http://proxy-2.internal:8888"
            )
        },
        clear=False,
    )
    def test_choose_playwright_proxy_selects_one_endpoint(
        self,
    ):
        self.assertIn(
            choose_playwright_proxy(),
            (
                {"server": "http://proxy-1.internal:8888"},
                {"server": "http://proxy-2.internal:8888"},
            ),
        )

    @patch("scraper.proxy.random.shuffle")
    def test_rotator_uses_all_proxies_before_repeating(self, _shuffle):
        rotator = PlaywrightProxyRotator(
            proxy_urls=(
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
                "http://proxy-3.internal:8888",
            )
        )

        selected = [
            rotator.choose()["server"]
            for _ in range(3)
        ]

        self.assertEqual(
            selected,
            [
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
                "http://proxy-3.internal:8888",
            ],
        )

    @patch("scraper.proxy.random.shuffle")
    def test_rotator_avoids_repeat_at_cycle_boundary(self, _shuffle):
        rotator = PlaywrightProxyRotator(
            proxy_urls=(
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
            )
        )

        selected = [
            rotator.choose()["server"]
            for _ in range(4)
        ]

        self.assertEqual(
            selected,
            [
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
            ],
        )

    def test_worker_proxy_pool_requires_at_least_two_endpoints(self):
        for urls in ((), ("http://one:8888",)):
            with self.subTest(urls=urls), self.assertRaises(ProxyConfigurationError):
                PlaywrightProxyRotator(proxy_urls=urls).validate_required_pool()

        PlaywrightProxyRotator(
            proxy_urls=("http://one:8888", "http://two:8888")
        ).validate_required_pool()
        PlaywrightProxyRotator(
            proxy_urls=(
                "http://one:8888",
                "http://two:8888",
                "http://three:8888",
            )
        ).validate_required_pool()

        PlaywrightProxyRotator(
            proxy_urls=("http://one:8888",)
        ).validate_required_pool(allow_single_proxy=True)
        with self.assertRaises(ProxyConfigurationError):
            PlaywrightProxyRotator(proxy_urls=()).validate_required_pool(
                allow_single_proxy=True
            )

    @patch.dict(
        "os.environ",
        {
            "SCRAPER_PROXY_URLS": "",
        },
        clear=False,
    )
    def test_choose_playwright_proxy_allows_direct_mode(self):
        self.assertIsNone(
            choose_playwright_proxy()
        )

    @patch.dict(
        "os.environ",
        {"SCRAPER_PROXY_URLS": "http://one:8888,http://two:8888"},
    )
    @patch("scraper.service.get_reader")
    @patch("scraper.service._run_single_subtype_attempt")
    def test_explicit_direct_run_ignores_configured_proxies(
        self, run_attempt, _reader
    ):
        now = datetime.now(timezone.utc)
        run_attempt.return_value = ScraperResult(
            status=ScraperStatus.NO_APPOINTMENT,
            started_at=now,
            finished_at=now,
        )

        result = run_scraper(ScraperConfig(
            visa_sub_types=("Student Visa",),
            direct_connection=True,
        ))

        self.assertEqual(result.status, ScraperStatus.NO_APPOINTMENT)
        self.assertIsNone(run_attempt.call_args.kwargs["proxy_config"])

    @patch.dict("os.environ", {"SCRAPER_PROXY_URLS": ""})
    @patch("scraper.service.get_reader")
    def test_default_run_still_requires_proxy_pool(self, get_reader):
        result = run_scraper(ScraperConfig(
            visa_sub_types=("Student Visa",),
        ))

        self.assertEqual(result.error_type, "ProxyConfigurationError")
        get_reader.assert_not_called()

    def test_playwright_proxy_config_rejects_invalid_scheme(self):
        with self.assertRaises(ValueError):
            playwright_proxy_config(
                "ftp://proxy.internal:8888"
            )


class CaptchaPacingTests(TestCase):
    def test_exact_bls_processing_error_is_classified_as_server_error(self):
        message = (
            "An error occured while processing your request. "
            "Please try again after some time."
        )

        self.assertIsNotNone(SITE_ERROR_PATTERN.search(message))

    @patch(
        "flows.appointment_flow._find_visible_dropdown_container",
        return_value=None,
    )
    def test_missing_optional_appointment_category_is_skipped(self, find_container):
        page = MagicMock()

        selected = _select_kendo_option(
            page,
            "Appointment Category",
            "Normal",
            optional=True,
        )

        self.assertFalse(selected)
        find_container.assert_called_once_with(
            page,
            "Appointment Category",
            timeout_seconds=2,
            required=False,
        )

    @patch("flows.appointment_flow.expect", side_effect=AssertionError("unusable"))
    @patch("flows.appointment_flow._find_visible_dropdown_container")
    def test_unusable_optional_appointment_category_is_skipped(
        self,
        find_container,
        _expect,
    ):
        page = MagicMock()
        container = MagicMock()
        container.locator.return_value.get_attribute.return_value = "category-id"
        find_container.return_value = container

        selected = _select_kendo_option(
            page,
            "Appointment Category",
            "Normal",
            optional=True,
        )

        self.assertFalse(selected)

    @patch(
        "flows.captcha_flow.blocking_overlay_visible",
        side_effect=(True, True, False, False),
    )
    @patch(
        "flows.captcha_flow.post_captcha_destination_visible",
        return_value=False,
    )
    @patch("flows.captcha_flow.site_error_page_visible", return_value=False)
    @patch("flows.captcha_flow.raise_for_http_forbidden")
    def test_post_captcha_waits_for_overlay_to_clear(
        self,
        _forbidden,
        _server_error,
        _destination,
        overlay_visible,
    ):
        page = MagicMock()
        page.locator.return_value.last.is_visible.return_value = True
        page.locator.return_value.last.is_enabled.return_value = True

        self.assertEqual(wait_for_post_captcha_page_ready(page), "ready")

        self.assertEqual(overlay_visible.call_count, 4)
        self.assertEqual(page.wait_for_timeout.call_count, 2)

    @patch(
        "flows.captcha_flow.blocking_overlay_visible",
        side_effect=(True, False),
    )
    @patch(
        "flows.captcha_flow.post_captcha_destination_visible",
        return_value=True,
    )
    @patch("flows.captcha_flow.site_error_page_visible", return_value=False)
    @patch("flows.captcha_flow.raise_for_http_forbidden")
    def test_post_captcha_destination_waits_for_overlay_to_clear(
        self,
        _forbidden,
        _server_error,
        _destination,
        _overlay,
    ):
        page = MagicMock()

        self.assertEqual(
            wait_for_post_captcha_page_ready(page),
            "destination",
        )
        self.assertEqual(page.wait_for_timeout.call_count, 1)

    @patch("flows.captcha_flow.blocking_overlay_visible", return_value=False)
    @patch("flows.captcha_flow.disclaimer_dialog_visible", return_value=False)
    @patch("flows.captcha_flow.post_captcha_destination_visible", return_value=False)
    @patch("flows.captcha_flow.site_error_page_visible", return_value=False)
    @patch("flows.captcha_flow.raise_for_http_forbidden")
    def test_clear_overlay_is_not_ready_without_usable_submit(
        self, _forbidden, _error, _destination, _disclaimer, _overlay
    ):
        page = MagicMock()
        page.locator.return_value.last.is_visible.return_value = False

        self.assertFalse(background_submit_ready(page))
        with self.assertRaisesRegex(RuntimeError, "Neither a usable background Submit"):
            wait_for_post_captcha_page_ready(page, timeout_ms=0)

    @patch("flows.captcha_flow.post_captcha_destination_visible", side_effect=(False, True))
    @patch("flows.captcha_flow.site_error_page_visible", return_value=False)
    @patch("flows.captcha_flow.raise_for_http_forbidden")
    def test_submit_waits_for_destination_even_while_button_remains_visible(
        self, _forbidden, _error, _destination
    ):
        page = MagicMock()
        page.locator.return_value.last.is_visible.return_value = True

        wait_for_post_captcha_destination(page)

        page.wait_for_timeout.assert_called_once_with(250)


class HttpDiagnosticsTests(TestCase):
    def test_egress_probe_reads_aws_plain_text_response(self):
        context = MagicMock()
        context.request.get.return_value.ok = True
        context.request.get.return_value.text.return_value = "94.184.43.30\n"

        address, address_hash, error = resolve_egress_ip(context)

        context.request.get.assert_called_once_with(EGRESS_IP_URL, timeout=10_000)
        self.assertEqual(address, "94.184.43.30")
        self.assertEqual(len(address_hash), 16)
        self.assertIsNone(error)

    def test_egress_probe_rejects_non_ip_response(self):
        context = MagicMock()
        context.request.get.return_value.ok = True
        context.request.get.return_value.text.return_value = "upstream error"

        address, address_hash, error = resolve_egress_ip(context)

        self.assertIsNone(address)
        self.assertIsNone(address_hash)
        self.assertEqual(error, "ValueError")

    def test_visible_403_heading_is_detected(self):
        page = MagicMock()
        page.get_by_role.return_value.first.is_visible.return_value = True

        self.assertTrue(http_forbidden_page_visible(page))

    def test_workflow_403_response_is_remembered(self):
        page = MagicMock()
        track_http_forbidden_responses(page)
        response_handler = page.on.call_args.args[1]
        response = MagicMock()
        response.status = 403
        response.url = "https://example.test/visatypeverification"
        response.request.resource_type = "document"

        response_handler(response)

        self.assertTrue(http_forbidden_page_visible(page))
        self.assertEqual(
            page._captcha_lab_http_403_response["url"],
            response.url,
        )

    def test_first_403_is_preserved_when_later_requests_also_fail(self):
        page = MagicMock()
        track_http_forbidden_responses(page)
        response_handler = page.on.call_args.args[1]
        first = MagicMock(status=403, url="https://example.test/Verify")
        first.request.resource_type = "xhr"
        second = MagicMock(status=403, url="https://example.test/GenerateCaptcha")
        second.request.resource_type = "xhr"

        response_handler(first)
        response_handler(second)

        state = page._captcha_lab_http_403_response
        self.assertIs(state["response"], first)
        self.assertEqual(state["subsequent_403s"][0]["url"], second.url)

    def test_login_captcha_outcome_terminates_on_403(self):
        page = MagicMock()
        page.get_by_role.return_value.first.is_visible.return_value = True

        with self.assertRaises(HTTP403Forbidden):
            wait_for_login_captcha_outcome(page)

    @patch("scraper.service.run_login_step")
    @patch("scraper.service.site_error_page_visible", return_value=False)
    @patch("scraper.service.http_forbidden_page_visible", return_value=False)
    def test_known_unclear_login_endpoint_restarts_in_same_browser(
        self,
        _forbidden,
        _server_error,
        run_login,
    ):
        page = MagicMock()
        page.url = (
            "https://iran.blsspainglobal.com/"
            "Global/NewCaptcha/LoginCaptchaSubmit"
        )

        self.assertTrue(restart_unclear_login_captcha(page))

        page.goto.assert_called_once()
        run_login.assert_called_once_with(page, account=None)

    @patch("flows.captcha_flow.raise_for_http_forbidden")
    @patch("flows.captcha_flow.expect")
    def test_verify_click_navigation_timeout_continues_to_state_detection(
        self,
        _expect,
        _raise_forbidden,
    ):
        page = MagicMock()
        verify_button = page.locator.return_value
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

        verify_button.click.side_effect = PlaywrightTimeoutError("navigation")

        click_verify_selection(page)

        verify_button.click.assert_called_once_with(timeout=10_000)

    def test_login_captcha_url_check_is_case_insensitive(self):
        page = MagicMock()
        page.url = "https://example.test/Global/NewCaptcha/LoginCaptcha?data=x"

        self.assertFalse(login_captcha_succeeded(page))

    def test_login_success_requires_visible_appointment_navigation(self):
        page = MagicMock()
        page.url = "https://example.test/Global/account/login"
        page.locator.return_value.first.is_visible.return_value = False

        self.assertFalse(login_captcha_succeeded(page))

    def test_login_success_accepts_visible_book_link_without_logout(self):
        page = MagicMock()
        page.url = "https://example.test/Global/home/index"
        page.locator.return_value.first.is_visible.return_value = True

        self.assertTrue(login_captcha_succeeded(page))

    @patch(
        "flows.captcha_flow.wait_for_post_captcha_page_ready",
        side_effect=("ready", "destination"),
    )
    @patch("flows.captcha_flow.post_captcha_destination_visible", return_value=False)
    @patch("flows.captcha_flow.site_error_page_visible", return_value=False)
    @patch("flows.captcha_flow.expect")
    def test_background_submit_accepts_modal_that_appears_during_click(
        self,
        _expect,
        _server_error,
        _destination,
        _page_ready,
    ):
        page = MagicMock()
        background_submit = MagicMock()
        page.locator.return_value = background_submit
        background_submit.last = background_submit
        background_submit.is_visible.return_value = True
        background_submit.is_enabled.return_value = True
        background_submit.click.side_effect = RuntimeError("intercepted")

        click_background_submit(page)

        background_submit.click.assert_called_once()

    @patch.dict("os.environ", {"HOSTNAME": "worker-abc"})
    def test_403_diagnostics_include_requested_evidence_without_cookies(self):
        response = MagicMock()
        response.status = 403
        response.body.return_value = b"blocked response"
        response.all_headers.return_value = {
            "cf-ray": "request-123",
            "content-type": "text/html",
            "set-cookie": "secret-cookie",
        }
        context = MagicMock()
        context.cookies.return_value = [
            {"name": "session", "domain": "example.test", "value": "secret"}
        ]

        data = response_diagnostics(
            response=response,
            context=context,
            account="person@example.test",
            egress_ip="203.0.113.42",
            egress_ip_hash="egress-hash",
            egress_lookup_error=None,
            proxy_endpoint="http://proxy-1.internal:8888",
            seconds_since_previous_login=31.2349,
        )

        self.assertEqual(data["http_status"], 403)
        self.assertEqual(data["egress_ip"], "203.0.113.42")
        self.assertEqual(data["egress_ip_hash"], "egress-hash")
        self.assertEqual(data["proxy_endpoint"], "http://proxy-1.internal:8888")
        self.assertEqual(data["worker_container_id"], "worker-abc")
        self.assertEqual(data["server_request_id"], "request-123")
        self.assertEqual(data["seconds_since_previous_login_request"], 31.235)
        self.assertEqual(data["body_length"], 16)
        self.assertNotIn("set-cookie", data["response_headers"])
        self.assertNotIn("person@example.test", str(data))
        self.assertNotIn("secret", str(data))

    @patch("scraper.service.record_scraper_event")
    def test_captcha_stage_telemetry_has_timestamps_and_duration(self, record_event):
        started_at = datetime.now(timezone.utc)

        record_captcha_stage(
            captcha="second",
            stage="verification",
            attempt_number=2,
            started_at=started_at,
            duration_ms=15321,
            status="rejected",
        )

        payload = record_event.call_args.kwargs
        self.assertEqual(payload["duration_ms"], 15321)
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["data"]["captcha"], "second")
        self.assertEqual(payload["data"]["stage"], "verification")
        self.assertEqual(payload["data"]["attempt_number"], 2)
        self.assertEqual(payload["data"]["started_at"], started_at.isoformat())
        self.assertIn("finished_at", payload["data"])


@patch.dict(
    "os.environ",
    {"SCRAPER_PROXY_URLS": "http://proxy-1:8888,http://proxy-2:8888"},
)
class FailureChainTests(TestCase):
    @patch("scraper.service.get_reader")
    def test_stop_request_exits_before_loading_browser_dependencies(self, get_reader):
        result = run_scraper(
            ScraperConfig(visa_sub_types=("Student Visa",)),
            should_stop=lambda: True,
        )

        self.assertEqual(result.status, ScraperStatus.STOPPED)
        get_reader.assert_not_called()

    @patch("scraper.service.get_reader")
    @patch("scraper.service._run_single_subtype_attempt")
    def test_terminal_403_is_stored_separately(self, run_attempt, _reader):
        now = datetime.now(timezone.utc)
        run_attempt.return_value = ScraperResult(
            status=ScraperStatus.FAILED,
            started_at=now,
            finished_at=now,
            error_type="HTTP403Forbidden",
            error_message="blocked",
        )

        result = run_scraper(
            ScraperConfig(visa_sub_types=("Student Visa",))
        )

        self.assertEqual(len(result.attempt_failures), 1)
        self.assertEqual(result.first_failure, result.attempt_failures[0])
        self.assertEqual(result.terminal_failure, result.attempt_failures[0])
        self.assertEqual(result.terminal_failure["error_type"], "HTTP403Forbidden")

    @patch.dict("os.environ", {"BLS_EMAIL_2": "second@example.com", "BLS_PASSWORD_2": "second-pass"})
    @patch("scraper.service.interruptible_cooldown")
    @patch("scraper.service.get_reader")
    @patch("scraper.service._run_single_subtype_attempt")
    def test_temporary_server_error_retries_with_fresh_browser(
        self,
        run_attempt,
        _reader,
        _cooldown,
    ):
        now = datetime.now(timezone.utc)
        run_attempt.side_effect = (
            ScraperResult(
                status=ScraperStatus.SERVER_ERROR,
                started_at=now,
                finished_at=now,
                error_type="RuntimeError",
                error_message="Temporary processing-error page.",
            ),
            ScraperResult(
                status=ScraperStatus.NO_APPOINTMENT,
                started_at=now,
                finished_at=now,
            ),
        )

        result = run_scraper(
            ScraperConfig(visa_sub_types=("Student Visa",))
        )

        self.assertEqual(result.status, ScraperStatus.NO_APPOINTMENT)
        self.assertEqual(run_attempt.call_count, 2)
        selected_accounts = [entry.kwargs["account"].email for entry in run_attempt.call_args_list]
        self.assertEqual(len(set(selected_accounts)), 2)
        self.assertEqual(len(result.attempt_failures), 1)
        self.assertEqual(result.first_failure["status"], "server_error")
        self.assertIsNone(result.terminal_failure)

    @patch("scraper.service.interruptible_cooldown")
    @patch("scraper.service.get_reader")
    @patch("scraper.service._run_single_subtype_attempt")
    def test_recovered_run_preserves_all_attempt_failures(
        self,
        run_attempt,
        _reader,
        _cooldown,
    ):
        now = datetime.now(timezone.utc)
        run_attempt.side_effect = (
            ScraperResult(
                status=ScraperStatus.FAILED,
                started_at=now,
                finished_at=now,
                error_type="CaptchaRejected",
                error_message="first",
            ),
            ScraperResult(
                status=ScraperStatus.FAILED,
                started_at=now,
                finished_at=now,
                error_type="MissingElement",
                error_message="second",
            ),
            ScraperResult(
                status=ScraperStatus.NO_APPOINTMENT,
                started_at=now,
                finished_at=now,
            ),
        )

        result = run_scraper(
            ScraperConfig(visa_sub_types=("Student Visa",))
        )

        self.assertEqual(len(result.attempt_failures), 2)
        self.assertEqual(result.first_failure["error_message"], "first")
        self.assertEqual(result.attempt_failures[1]["error_message"], "second")
        self.assertIsNone(result.terminal_failure)

    @patch("flows.captcha_flow.appointment_form_visible", return_value=True)
    def test_background_submit_skips_when_form_is_already_visible(self, _form_visible):
        page = MagicMock()

        click_background_submit(page)

        page.wait_for_timeout.assert_not_called()
        page.locator.return_value.last.click.assert_not_called()

    @patch("scraper.service.site_error_page_visible", return_value=True)
    @patch("scraper.service.http_forbidden_page_visible", return_value=False)
    @patch("scraper.service.no_appointments_dialog_visible", return_value=False)
    def test_page_state_checks_server_error_before_retry(
        self,
        _no_slots,
        _forbidden,
        _error,
    ):
        state = inspect_page_state(MagicMock())

        self.assertTrue(state["server_error"])
        self.assertFalse(state["no_appointment"])

    @patch(
        "scraper.service._run_second_captcha_attempt",
        side_effect=(False, True),
    )
    def test_rejected_second_captcha_is_retried_in_same_page(self, solve_attempt):
        page = MagicMock()
        output_dir = MagicMock()
        reader = MagicMock()

        run_second_captcha_step(
            page,
            gpu=False,
            output_dir=output_dir,
            reader=reader,
        )

        self.assertEqual(solve_attempt.call_count, 2)
        self.assertIs(
            solve_attempt.call_args_list[0].args[0],
            page,
        )
        self.assertIs(
            solve_attempt.call_args_list[1].args[0],
            page,
        )
        self.assertEqual(
            [
                item.kwargs["attempt_number"]
                for item in solve_attempt.call_args_list
            ],
            [1, 2],
        )
        page.wait_for_timeout.assert_called_once_with(
            SECOND_CAPTCHA_RETRY_SETTLE_MS
        )

    @patch(
        "scraper.service._run_login_captcha_attempt",
        side_effect=("rejected", "succeeded"),
    )
    def test_rejected_login_captcha_is_retried_in_same_page(self, solve_attempt):
        page = MagicMock()

        run_captcha_step(
            page,
            gpu=False,
            output_dir=MagicMock(),
            reader=MagicMock(),
        )

        self.assertEqual(solve_attempt.call_count, 2)
        self.assertIs(solve_attempt.call_args_list[0].args[0], page)
        self.assertIs(solve_attempt.call_args_list[1].args[0], page)
        self.assertEqual(
            [item.kwargs["attempt_number"] for item in solve_attempt.call_args_list],
            [1, 2],
        )
        page.wait_for_timeout.assert_called_once_with(
            SECOND_CAPTCHA_RETRY_SETTLE_MS
        )

    @patch("scraper.service.restart_unclear_login_captcha", return_value=True)
    @patch(
        "scraper.service._run_login_captcha_attempt",
        side_effect=("unclear", "succeeded"),
    )
    def test_unclear_login_captcha_is_restarted_in_same_page(
        self,
        solve_attempt,
        restart_login,
    ):
        page = MagicMock()

        run_captcha_step(
            page,
            gpu=False,
            output_dir=MagicMock(),
            reader=MagicMock(),
        )

        self.assertEqual(solve_attempt.call_count, 2)
        self.assertIs(solve_attempt.call_args_list[0].args[0], page)
        self.assertIs(solve_attempt.call_args_list[1].args[0], page)
        restart_login.assert_called_once_with(page, account=None)

    @patch(
        "scraper.service._run_login_captcha_attempt",
        return_value="rejected",
    )
    def test_repeated_login_captcha_rejections_escalate(self, solve_attempt):
        page = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "rejected 3 times"):
            run_captcha_step(
                page,
                gpu=False,
                output_dir=MagicMock(),
                reader=MagicMock(),
            )

        self.assertEqual(solve_attempt.call_count, 3)
        self.assertEqual(page.wait_for_timeout.call_count, 2)

    @patch("scraper.service.click_verify_selection")
    @patch("scraper.service.site_error_page_visible", return_value=False)
    @patch(
        "scraper.service._run_second_captcha_attempt",
        side_effect=(AssertionError("missing labels"), True),
    )
    def test_incomplete_second_captcha_reloads_without_new_login(
        self,
        solve_attempt,
        _server_error,
        reopen_captcha,
    ):
        page = MagicMock()

        run_second_captcha_step(
            page,
            gpu=False,
            output_dir=MagicMock(),
            reader=MagicMock(),
        )

        self.assertEqual(solve_attempt.call_count, 2)
        page.reload.assert_called_once_with(
            wait_until="domcontentloaded",
            timeout=60_000,
        )
        reopen_captcha.assert_called_once_with(page)

    @patch(
        "scraper.service.inspect_page_state",
        return_value={"appointment_form": True},
    )
    @patch(
        "scraper.service._run_second_captcha_attempt",
        side_effect=AssertionError("verification label disappeared"),
    )
    def test_second_captcha_timeout_accepts_valid_downstream_state(
        self,
        solve_attempt,
        _page_state,
    ):
        page = MagicMock()

        run_second_captcha_step(
            page,
            gpu=False,
            output_dir=MagicMock(),
            reader=MagicMock(),
        )

        solve_attempt.assert_called_once()
        page.reload.assert_not_called()

    @patch(
        "scraper.service._run_second_captcha_attempt",
        side_effect=HTTP403Forbidden("blocked"),
    )
    def test_second_captcha_403_never_reloads_or_retries(self, solve_attempt):
        page = MagicMock()

        with self.assertRaises(HTTP403Forbidden):
            run_second_captcha_step(
                page,
                gpu=False,
                output_dir=MagicMock(),
                reader=MagicMock(),
            )

        solve_attempt.assert_called_once()
        page.reload.assert_not_called()

    @patch("scraper.service.http_forbidden_page_visible", return_value=True)
    def test_second_captcha_403_precedes_popup_based_classification(
        self, _forbidden
    ):
        with self.assertRaises(HTTP403Forbidden):
            classify_second_captcha_state(
                MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )

    @patch("scraper.service.site_error_page_visible", return_value=False)
    @patch("scraper.service.http_forbidden_page_visible", return_value=False)
    def test_open_popup_without_site_rejection_is_only_pending(
        self, _forbidden, _server_error
    ):
        verified = MagicMock()
        verified.is_visible.return_value = False
        invalid = MagicMock()
        invalid.is_visible.return_value = False
        popup = MagicMock()
        popup.is_visible.return_value = True

        self.assertEqual(
            classify_second_captcha_state(MagicMock(), verified, invalid, popup),
            "pending",
        )
        invalid.is_visible.return_value = True
        self.assertEqual(
            classify_second_captcha_state(MagicMock(), verified, invalid, popup),
            "explicitly_rejected",
        )

    def test_second_captcha_signature_changes_with_rendered_images(self):
        frame = MagicMock()
        images = [f"data:image/png;base64,{number}" for number in range(9)]
        frame.locator.return_value.evaluate_all.return_value = images
        first = second_captcha_challenge_signature(frame)
        frame.locator.return_value.evaluate_all.return_value = images[:-1] + ["new"]

        self.assertIsNotNone(first)
        self.assertNotEqual(first, second_captcha_challenge_signature(frame))

    @patch("scraper.service.site_error_page_visible", return_value=False)
    @patch("scraper.service.http_forbidden_page_visible", return_value=False)
    @patch(
        "scraper.service.second_captcha_challenge_signature",
        side_effect=("old", "new"),
    )
    def test_explicit_rejection_waits_for_different_grid(
        self, _signature, _forbidden, _server_error
    ):
        page = MagicMock()
        self.assertTrue(wait_for_regenerated_second_captcha(page, MagicMock(), "old"))
        page.wait_for_timeout.assert_called_once_with(250)

    @patch("scraper.service.SECOND_CAPTCHA_REGENERATION_TIMEOUT_SECONDS", 0)
    def test_explicit_rejection_without_new_grid_is_inconclusive(self):
        with self.assertRaises(SecondCaptchaUnconfirmed):
            wait_for_regenerated_second_captcha(MagicMock(), MagicMock(), "old")

    @patch("scraper.service.inspect_page_state", return_value={})
    @patch(
        "scraper.service._run_second_captcha_attempt",
        side_effect=SecondCaptchaUnconfirmed("no confirmation"),
    )
    def test_unconfirmed_second_captcha_does_not_request_another_puzzle(
        self, solve_attempt, _page_state
    ):
        page = MagicMock()

        with self.assertRaises(SecondCaptchaUnconfirmed):
            run_second_captcha_step(
                page, gpu=False, output_dir=MagicMock(), reader=MagicMock()
            )

        solve_attempt.assert_called_once()
        page.reload.assert_not_called()

    @patch(
        "scraper.service._run_second_captcha_attempt",
        return_value=False,
    )
    def test_repeated_second_captcha_rejections_escalate(self, solve_attempt):
        page = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "rejected 3 times"):
            run_second_captcha_step(
                page,
                gpu=False,
                output_dir=MagicMock(),
                reader=MagicMock(),
            )

        self.assertEqual(solve_attempt.call_count, 3)
        self.assertEqual(page.wait_for_timeout.call_count, 2)

    @patch("flows.captcha_flow.random.randint", side_effect=(400, 700))
    @patch("flows.captcha_flow.click_captcha_tile")
    @patch("flows.captcha_flow.get_captcha_tiles")
    def test_selected_tiles_are_paced_before_submit(
        self,
        get_tiles,
        click_tile,
        _random_delay,
    ):
        page = MagicMock()
        tiles = [MagicMock() for _ in range(9)]
        get_tiles.return_value = tiles

        click_selected_captcha_tiles(page, (1, 3, 9))

        self.assertEqual(
            click_tile.call_args_list,
            [
                call(page, tiles[0], 1),
                call(page, tiles[2], 3),
                call(page, tiles[8], 9),
            ],
        )
        self.assertEqual(
            page.wait_for_timeout.call_args_list,
            [
                call(CAPTCHA_PRE_CLICK_SETTLE_MS),
                call(400),
                call(700),
                call(CAPTCHA_POST_SELECTION_SETTLE_MS),
            ],
        )

    @patch("flows.captcha_flow.click_captcha_tile")
    def test_invalid_tile_is_rejected_before_any_click(self, click_tile):
        page = MagicMock()
        tiles = [MagicMock() for _ in range(9)]

        with self.assertRaises(ValueError):
            click_selected_captcha_tiles(
                page,
                (1, 10),
                tiles=tiles,
            )

        click_tile.assert_not_called()
        page.wait_for_timeout.assert_not_called()

    @patch("flows.captcha_flow.appointment_form_ready", return_value=True)
    @patch("flows.captcha_flow.disclaimer_dialog_visible", return_value=False)
    @patch("flows.captcha_flow.site_error_page_visible", return_value=False)
    def test_missing_ok_is_accepted_when_form_already_visible(
        self, _error, _dialog_visible, _form_ready
    ):
        page = MagicMock()

        self.assertFalse(click_ok_dialog(page))
        page.wait_for_timeout.assert_not_called()

    @patch("flows.captcha_flow.appointment_form_ready", return_value=True)
    @patch("flows.captcha_flow.disclaimer_dialog_visible", return_value=True)
    @patch("flows.captcha_flow.site_error_page_visible", return_value=False)
    @patch("flows.captcha_flow.expect")
    def test_visible_form_does_not_skip_disclaimer(
        self, _expect, _error, _dialog_visible, _form_ready
    ):
        page = MagicMock()

        self.assertTrue(click_ok_dialog(page))
        page.locator.assert_any_call("#disclaimarModal")

    @patch("flows.captcha_flow.blocking_overlay_visible", return_value=False)
    @patch("flows.captcha_flow.disclaimer_dialog_visible", return_value=True)
    @patch("flows.captcha_flow.appointment_form_visible", return_value=True)
    def test_form_behind_disclaimer_is_not_ready(
        self, _form_visible, _dialog_visible, _overlay
    ):
        self.assertFalse(appointment_form_ready(MagicMock()))
        self.assertTrue(post_captcha_destination_visible(MagicMock()))

    @patch("flows.captcha_flow.blocking_overlay_visible", return_value=True)
    @patch("flows.captcha_flow.disclaimer_dialog_visible", return_value=False)
    @patch("flows.captcha_flow.appointment_form_visible", return_value=True)
    def test_actionable_form_is_ready_despite_unrelated_visible_overlay(
        self, _form_visible, _disclaimer, _overlay
    ):
        page = MagicMock()

        self.assertTrue(appointment_form_ready(page))
        page.locator.return_value.first.click.assert_called_once_with(
            trial=True, timeout=1_000
        )

    @patch("flows.captcha_flow.disclaimer_dialog_visible", return_value=False)
    @patch("flows.captcha_flow.appointment_form_visible", return_value=True)
    def test_form_label_is_not_ready_when_dropdown_click_is_intercepted(
        self, _form_visible, _disclaimer
    ):
        page = MagicMock()
        page.locator.return_value.first.click.side_effect = RuntimeError("intercepted")

        self.assertFalse(appointment_form_ready(page))

    def test_post_disclaimer_snapshot_omits_query_and_passwords(self):
        page = MagicMock()
        page.url = "https://example.test/Global/bls/visatype?data=secret"
        page.locator.return_value.all_inner_texts.return_value = [
            "Jurisdiction", "Location", "Visa Type"
        ]
        page.locator.return_value.first.is_visible.return_value = False
        page.locator.return_value.count.return_value = 0

        state = post_disclaimer_state(page)

        self.assertEqual(state["path"], "/Global/bls/visatype")
        self.assertEqual(
            state["visible_form_labels"],
            ["Jurisdiction", "Location", "Visa Type"],
        )
        self.assertNotIn("secret", str(state))

    @patch("scraper.service.no_appointments_dialog_visible", return_value=False)
    @patch("scraper.service.site_error_page_visible", return_value=False)
    def test_missing_form_result_is_unconfirmed_not_an_appointment(
        self, _error, _no_appointments
    ):
        with patch("scraper.service.appointment_available_dialog_visible", return_value=False):
            with self.assertRaises(AppointmentResultUnconfirmed):
                wait_for_form_result(MagicMock(), timeout_seconds=0)

    @patch("scraper.service.no_appointments_dialog_visible", return_value=True)
    @patch("scraper.service.site_error_page_visible", return_value=False)
    def test_explicit_no_appointments_result_wins_at_deadline(
        self, _error, _no_appointments
    ):
        self.assertIs(
            wait_for_form_result(MagicMock(), timeout_seconds=0),
            ScraperStatus.NO_APPOINTMENT,
        )

    def test_other_modal_text_is_not_no_appointments(self):
        page = MagicMock()
        page.locator.return_value.first.is_visible.return_value = True
        page.locator.return_value.first.count.return_value = 1
        page.locator.return_value.first.inner_text.return_value = "Please try again later"

        self.assertFalse(no_appointments_dialog_visible(page))

    def test_explicit_no_appointments_text_is_recognized(self):
        page = MagicMock()
        page.locator.return_value.first.is_visible.return_value = True
        page.locator.return_value.first.count.return_value = 1
        page.locator.return_value.first.inner_text.return_value = (
            "No Appointments Available"
        )

        self.assertTrue(no_appointments_dialog_visible(page))

    def test_explicit_available_appointment_text_is_recognized(self):
        page = MagicMock()
        page.locator.return_value.first.is_visible.return_value = True
        page.locator.return_value.first.count.return_value = 1
        page.locator.return_value.first.inner_text.return_value = "Appointments Available"

        self.assertTrue(appointment_available_dialog_visible(page))

    def test_no_available_appointments_is_not_positive(self):
        page = MagicMock()
        page.locator.return_value.first.is_visible.return_value = True
        page.locator.return_value.first.count.return_value = 1
        page.locator.return_value.first.inner_text.return_value = "No available appointments"

        self.assertFalse(appointment_available_dialog_visible(page))

    @patch("scraper.service.appointment_available_dialog_visible", return_value=True)
    @patch("scraper.service.no_appointments_dialog_visible", return_value=False)
    @patch("scraper.service.site_error_page_visible", return_value=False)
    def test_explicit_available_result_is_confirmed(
        self, _error, _no_appointments, _available
    ):
        self.assertIs(
            wait_for_form_result(MagicMock(), timeout_seconds=0),
            ScraperStatus.APPOINTMENT_FOUND,
        )

    @patch("scraper.service.random.randint", side_effect=(31, 63, 125, 175))
    def test_retry_cooldown_grows_with_jitter(self, randint):
        self.assertEqual(
            [subtype_retry_delay_seconds(attempt) for attempt in range(1, 6)],
            [31, 63, 125, 175, 0],
        )
        self.assertEqual(
            randint.call_args_list,
            [
                call(24, 36),
                call(48, 72),
                call(96, 144),
                call(144, 216),
            ],
        )


class AppointmentCycleTests(TestCase):
    @patch("scraper.service.site_error_page_visible", return_value=False)
    @patch("scraper.service.appointment_form_ready", return_value=True)
    @patch("scraper.service.solve_visible_appointment_captcha")
    def test_direct_form_branch_does_not_force_captcha(
        self, solve_captcha, _form_ready, _site_error
    ):
        reach_appointment_form(
            MagicMock(),
            config=ScraperConfig(),
            reader=MagicMock(),
            output_dir=MagicMock(),
            should_stop=None,
        )

        solve_captcha.assert_not_called()

    @patch(
        "scraper.service.appointment_form_ready",
        side_effect=(False, True),
    )
    @patch("scraper.service.background_submit_ready", return_value=False)
    @patch("scraper.service.appointment_captcha_visible", return_value=True)
    @patch("scraper.service.solve_visible_appointment_captcha")
    @patch("scraper.service.disclaimer_dialog_visible", return_value=False)
    @patch("scraper.service.site_error_page_visible", return_value=False)
    def test_captcha_first_branch_reaches_form(
        self,
        _site_error,
        _disclaimer,
        solve_captcha,
        _captcha_visible,
        _background_submit,
        _form_ready,
    ):
        reach_appointment_form(
            MagicMock(),
            config=ScraperConfig(),
            reader=MagicMock(),
            output_dir=MagicMock(),
            should_stop=None,
        )

        solve_captcha.assert_called_once()

    @patch(
        "scraper.service.no_appointments_dialog_visible",
        side_effect=(False, True),
    )
    @patch("scraper.service.appointment_available_dialog_visible", return_value=False)
    @patch("scraper.service.appointment_captcha_visible", return_value=True)
    @patch("scraper.service.solve_visible_appointment_captcha")
    @patch("scraper.service.site_error_page_visible", return_value=False)
    @patch("scraper.service.expect")
    def test_post_form_captcha_is_solved_before_result(
        self,
        _expect,
        _site_error,
        solve_captcha,
        _captcha_visible,
        _available,
        _no_appointment,
    ):
        page = MagicMock()

        result = submit_appointment_form_and_wait(
            page,
            config=ScraperConfig(),
            reader=MagicMock(),
            output_dir=MagicMock(),
            should_stop=None,
        )

        self.assertIs(result, ScraperStatus.NO_APPOINTMENT)
        solve_captcha.assert_called_once()

    @patch("scraper.service.log_no_appointment")
    @patch("scraper.service.click_try_again")
    @patch(
        "scraper.service.submit_appointment_form_and_wait",
        side_effect=(ScraperStatus.NO_APPOINTMENT, ScraperStatus.NO_APPOINTMENT),
    )
    @patch("scraper.service.fill_appointment_form")
    @patch("scraper.service.reach_appointment_form")
    @patch("scraper.service.click_nav_book_new_appointment")
    def test_both_subtypes_share_one_authenticated_session(
        self,
        open_appointment,
        reach_form,
        fill_form,
        submit_form,
        try_again,
        _log_no_appointment,
    ):
        page = MagicMock(url="https://example.test/Global/bls/visatype")
        config = ScraperConfig()

        result = run_authenticated_appointment_cycle(
            page,
            config=config,
            reader=MagicMock(),
            attempt_number=1,
            should_stop=None,
        )

        self.assertIs(result.status, ScraperStatus.NO_APPOINTMENT)
        open_appointment.assert_called_once_with(page)
        self.assertEqual(reach_form.call_count, 2)
        self.assertEqual(
            [call.kwargs["visa_sub_type"] for call in fill_form.call_args_list],
            ["Student Visa", "Non-Working Residence Visa"],
        )
        self.assertEqual(submit_form.call_count, 2)
        try_again.assert_called_once_with(page)

    @patch("scraper.service.notify_admin")
    @patch("scraper.service.click_try_again")
    @patch(
        "scraper.service.submit_appointment_form_and_wait",
        return_value=ScraperStatus.APPOINTMENT_FOUND,
    )
    @patch("scraper.service.fill_appointment_form")
    @patch("scraper.service.reach_appointment_form")
    @patch("scraper.service.click_nav_book_new_appointment")
    def test_first_appointment_stops_before_second_subtype(
        self,
        _open_appointment,
        reach_form,
        fill_form,
        submit_form,
        try_again,
        notify,
    ):
        page = MagicMock(url="https://example.test/result")

        result = run_authenticated_appointment_cycle(
            page,
            config=ScraperConfig(),
            reader=MagicMock(),
            attempt_number=1,
            should_stop=None,
        )

        self.assertIs(result.status, ScraperStatus.APPOINTMENT_FOUND)
        self.assertEqual(reach_form.call_count, 1)
        self.assertEqual(fill_form.call_count, 1)
        self.assertEqual(submit_form.call_count, 1)
        try_again.assert_not_called()
        notify.assert_called_once()
