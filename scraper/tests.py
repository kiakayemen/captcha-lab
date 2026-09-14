from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from flows.captcha_flow import (
    CAPTCHA_POST_SELECTION_SETTLE_MS,
    CAPTCHA_PRE_CLICK_SETTLE_MS,
    click_background_submit,
    click_ok_dialog,
    click_selected_captcha_tiles,
    SITE_ERROR_PATTERN,
)
from scraper.service import (
    SECOND_CAPTCHA_RETRY_SETTLE_MS,
    inspect_page_state,
    run_second_captcha_step,
    record_captcha_stage,
    run_scraper,
    subtype_retry_delay_seconds,
)
from scraper.http_diagnostics import response_diagnostics
from scraper.models import ScraperConfig, ScraperResult, ScraperStatus

from scraper.proxy import (
    PlaywrightProxyRotator,
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


class HttpDiagnosticsTests(TestCase):
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


class FailureChainTests(TestCase):
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

    @patch("scraper.service.time.sleep")
    @patch("scraper.service.get_reader")
    @patch("scraper.service._run_single_subtype_attempt")
    def test_recovered_run_preserves_all_attempt_failures(
        self,
        run_attempt,
        _reader,
        _sleep,
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
    @patch("scraper.service.no_appointments_dialog_visible", return_value=False)
    def test_page_state_checks_server_error_before_retry(self, _no_slots, _error):
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
