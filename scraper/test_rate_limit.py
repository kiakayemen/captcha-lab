from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import MagicMock, patch

from flows.errors import HTTP429RateLimited, raise_for_http_forbidden, track_http_forbidden_responses
from config import BLSAccount
from scraper.models import ScraperConfig, ScraperResult, ScraperStatus
from scraper.service import run_scraper, _run_single_subtype_attempt


class RateLimitResponseTests(TestCase):
    def test_429_document_stops_workflow_and_calls_persistence_once(self):
        page = MagicMock()
        page._captcha_lab_http_429_response = None
        persist = MagicMock()
        track_http_forbidden_responses(page, persist)
        handler = page.on.call_args.args[1]
        response = MagicMock(status=429, url="https://iran.blsspainglobal.com/Global/account/login")
        response.request.resource_type = "document"
        handler(response)
        handler(response)
        persist.assert_called_once_with(response)
        with self.assertRaises(HTTP429RateLimited):
            raise_for_http_forbidden(page)

    @patch("scraper.service.record_scraper_event")
    @patch("scraper.service.response_diagnostics", return_value={})
    @patch("scraper.service.resolve_egress_ip", return_value=(None, None, None))
    @patch("scraper.service.note_rate_limit", return_value=datetime.now(timezone.utc))
    @patch("scraper.service.sync_playwright")
    def test_initial_429_persists_and_stops_before_login(self, playwright, persist, _ip, _diag, _event):
        page = playwright.return_value.__enter__.return_value.chromium.launch.return_value.new_context.return_value.new_page.return_value
        page._captcha_lab_http_429_response = None
        response = MagicMock(status=429, url="https://iran.blsspainglobal.com/Global/account/login")
        response.request.resource_type = "document"
        response.headers = {"retry-after": "600"}
        def navigate(*args, **kwargs):
            page.on.call_args.args[1](response)
            return response
        page.goto.side_effect = navigate
        with patch("scraper.service.run_login_step") as login:
            result = _run_single_subtype_attempt(account=BLSAccount(email="x@example.com", password="x"),
                config=ScraperConfig(), visa_sub_type="Student Visa", attempt_number=1,
                reader=None, proxy_config=None, login_request_state={}, should_stop=None)
        self.assertEqual(result.error_type, "HTTP429RateLimited")
        persist.assert_called_once_with("600")
        login.assert_not_called()

    @patch.dict("os.environ", {"BLS_EMAIL": "test@example.com", "BLS_PASSWORD": "test", "SCRAPER_PROXY_URLS": "http://one:8888,http://two:8888"})
    @patch("scraper.service.get_reader")
    @patch("scraper.service._run_single_subtype_attempt")
    def test_rate_limit_stops_run_without_switching_proxy(self, attempt, _reader):
        now = datetime.now(timezone.utc)
        attempt.return_value = ScraperResult(status=ScraperStatus.FAILED, started_at=now,
                                             finished_at=now, error_type="HTTP429RateLimited")
        result = run_scraper(ScraperConfig(visa_sub_types=("Student Visa",)))
        self.assertEqual(result.error_type, "HTTP429RateLimited")
        attempt.assert_called_once()
