from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from flows.captcha_flow import (
    CAPTCHA_POST_SELECTION_SETTLE_MS,
    CAPTCHA_PRE_CLICK_SETTLE_MS,
    click_background_submit,
    click_ok_dialog,
    click_selected_captcha_tiles,
    click_verify_selection,
    login_captcha_succeeded,
    SITE_ERROR_PATTERN,
)
from flows.appointment_flow import _select_kendo_option
from flows.errors import (
    HTTP403Forbidden,
    http_forbidden_page_visible,
    track_http_forbidden_responses,
)
from scraper.service import (
    SECOND_CAPTCHA_RETRY_SETTLE_MS,
    inspect_page_state,
    run_captcha_step,
    run_second_captcha_step,
    record_captcha_stage,
    run_scraper,
    subtype_retry_delay_seconds,
    wait_for_login_captcha_outcome,
)
from scraper.http_diagnostics import response_diagnostics
from scraper.models import ScraperConfig, ScraperResult, ScraperStatus

from scraper.proxy import (
    PlaywrightProxyRotator,
    ProxyConfigurationError,
    choose_playwright_proxy,
    configured_proxy_urls,
    playwright_proxy_config,
)


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


class HttpDiagnosticsTests(TestCase):
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

    def test_login_captcha_outcome_terminates_on_403(self):
        page = MagicMock()
        page.get_by_role.return_value.first.is_visible.return_value = True

        with self.assertRaises(HTTP403Forbidden):
            wait_for_login_captcha_outcome(page)

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

    @patch("flows.captcha_flow.appointment_form_visible", return_value=False)
    @patch("flows.captcha_flow.site_error_page_visible", return_value=False)
    @patch("flows.captcha_flow.expect")
    def test_background_submit_accepts_modal_that_appears_during_click(
        self,
        _expect,
        _server_error,
        _form_visible,
    ):
        page = MagicMock()
        background_submit = MagicMock()
        ok_button = MagicMock()
        page.locator.side_effect = [background_submit, ok_button]
        background_submit.last = background_submit
        background_submit.is_visible.return_value = True
        background_submit.is_enabled.return_value = True
        background_submit.click.side_effect = RuntimeError("intercepted")
        ok_button.first = ok_button
        ok_button.is_visible.side_effect = (False, True)

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
            egress_ip_hash="egress-hash",
            egress_lookup_error=None,
            seconds_since_previous_login=31.2349,
        )

        self.assertEqual(data["http_status"], 403)
        self.assertEqual(data["egress_ip_hash"], "egress-hash")
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

    @patch("flows.captcha_flow.appointment_form_visible", return_value=True)
    def test_missing_ok_is_accepted_when_form_already_visible(self, _form_visible):
        page = MagicMock()

        self.assertFalse(click_ok_dialog(page))
        page.wait_for_timeout.assert_not_called()

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
