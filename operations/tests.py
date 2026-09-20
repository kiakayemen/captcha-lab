import asyncio
import contextvars
import logging
import threading
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

from django.core.exceptions import SynchronousOnlyOperation
from django.core.management import call_command
from django.test import TestCase, TransactionTestCase, override_settings
from django.test import RequestFactory
from django.contrib.auth import get_user_model
from django.contrib.staticfiles import finders
from django.utils import timezone

from scraper.models import ScraperConfig

from .events import bind_scraper_event_context, record_event_from_log, record_scraper_event
from .database_writer import run_database_write
from .models import ScraperEvent, ScraperRun, ScraperRunLog
from .admin import LogDateRangeForm, ScraperRunAdmin
from .jalali import parse_jalali_date
from .run_logging import ScraperRunDatabaseHandler
from .services import (
    ScraperRunAlreadyStarted,
    active_scraper_runs,
    execute_scraper_run,
    recover_stale_scraper_runs,
    request_scraper_stop,
)


class ScraperEventTests(TestCase):
    def setUp(self):
        self.run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.SCHEDULED,
            visa_sub_types=["Student Visa"],
        )

    def test_existing_lifecycle_logs_become_structured_events(self):
        with bind_scraper_event_context(self.run):
            record_event_from_log(
                "Starting fresh browser attempt. "
                "Visa subtype=Student Visa | Attempt=2/5"
            )
            record_event_from_log(
                "Subtype attempt failed; discarding browser state and "
                "starting completely fresh. Visa subtype=Student Visa | "
                "Failed attempt=2/5 | Error=RuntimeError: "
                "Second CAPTCHA was not verified."
            )

        events = list(
            ScraperEvent.objects.order_by("created_at", "id")
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(
            events[0].event_type,
            ScraperEvent.EventType.SUBTYPE_STARTED,
        )
        self.assertEqual(events[0].visa_sub_type, "Student Visa")
        self.assertEqual(events[0].attempt_number, 2)
        self.assertEqual(
            events[1].reason_code,
            "SECOND_CAPTCHA_NOT_VERIFIED",
        )
        self.assertEqual(events[1].execution_id, events[0].execution_id)

    def test_admin_shows_independent_subtype_outcomes(self):
        self.run.visa_sub_types = ["Student Visa", "Non-Working Residence Visa"]
        self.run.status = ScraperRun.Status.FAILED
        self.run.terminal_failure = {"visa_sub_type": "Non-Working Residence Visa"}
        self.run.save()
        ScraperEvent.objects.create(
            run=self.run,
            event_type=ScraperEvent.EventType.SUBTYPE_FINISHED,
            visa_sub_type="Student Visa",
            status=ScraperRun.Status.NO_APPOINTMENT,
        )
        admin = ScraperRunAdmin(ScraperRun, None)
        self.assertEqual(admin.student_visa_result(self.run), "No appointment")
        self.assertEqual(admin.non_working_residence_result(self.run), "Failed")

    def test_admin_distinguishes_possible_appointment(self):
        self.run.status = ScraperRun.Status.POSSIBLE_APPOINTMENT
        self.run.save(update_fields=["status"])
        ScraperEvent.objects.create(
            run=self.run,
            event_type=ScraperEvent.EventType.SUBTYPE_FINISHED,
            visa_sub_type="Student Visa",
            status=ScraperRun.Status.POSSIBLE_APPOINTMENT,
        )

        admin = ScraperRunAdmin(ScraperRun, None)
        self.assertEqual(admin.student_visa_result(self.run), "Possible appointment")

    def test_proxy_403_report_groups_by_route_and_endpoint(self):
        for url in (
            "https://example.test/Global/NewCaptcha/GenerateCaptcha?one=1",
            "https://example.test/Global/NewCaptcha/GenerateCaptcha?two=2",
        ):
            ScraperEvent.objects.create(
                run=self.run,
                event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
                status="403",
                data={"proxy_endpoint": "http://proxy-a:8888", "response_url": url},
            )
        for status in ("403", "200"):
            ScraperEvent.objects.create(
                run=self.run,
                event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
                status=status,
                data={
                    "proxy_endpoint": "http://proxy-a:8888",
                    "response_request_url": "https://example.test/Global/account/login",
                },
            )
        output = StringIO()

        call_command("proxy_403_report", "--hours", "24", "--json", stdout=output)

        import json
        self.assertEqual(
            json.loads(output.getvalue())["rows"],
            [{
                "proxy_endpoint": "http://proxy-a:8888",
                "request_path": "/global/newcaptcha/generatecaptcha",
                "responses": 2,
                "runs": 1,
                "total_responses": None,
                "forbidden_percent": None,
                "independent_egress_probes": [],
            }, {
                "proxy_endpoint": "http://proxy-a:8888",
                "request_path": "/global/account/login",
                "responses": 1,
                "runs": 1,
                "total_responses": 2,
                "forbidden_percent": 50.0,
                "independent_egress_probes": [],
            }],
        )

    def test_duplicate_execution_is_rejected(self):
        self.run.status = ScraperRun.Status.RUNNING
        self.run.started_at = timezone.now()
        self.run.save(update_fields=["status", "started_at"])

        with self.assertRaises(ScraperRunAlreadyStarted):
            execute_scraper_run(
                config=ScraperConfig(
                    headless=True,
                    visa_sub_types=("Student Visa",),
                ),
                db_run=self.run,
            )

    def test_pending_run_stops_immediately(self):
        request_scraper_stop(self.run)
        self.run.refresh_from_db()

        self.assertEqual(self.run.status, ScraperRun.Status.STOPPED)
        self.assertIsNotNone(self.run.stop_requested_at)
        self.assertIsNotNone(self.run.stopped_at)
        self.assertFalse(active_scraper_runs().filter(pk=self.run.pk).exists())

    def test_running_run_becomes_stop_requested(self):
        self.run.status = ScraperRun.Status.RUNNING
        self.run.started_at = timezone.now()
        self.run.save(update_fields=["status", "started_at"])

        request_scraper_stop(self.run)
        self.run.refresh_from_db()

        self.assertEqual(self.run.status, ScraperRun.Status.STOP_REQUESTED)
        self.assertIsNotNone(self.run.stop_requested_at)
        self.assertTrue(active_scraper_runs().filter(pk=self.run.pk).exists())


class ScraperRunLoggingTests(TestCase):
    def setUp(self):
        self.run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.SCHEDULED,
            visa_sub_types=["Student Visa"],
        )

    def test_database_handler_keeps_run_id_without_contextvar(self):
        handler = ScraperRunDatabaseHandler(str(self.run.pk))
        handler.setFormatter(logging.Formatter("%(message)s"))

        try:
            handler.emit(
                logging.LogRecord(
                    name="captcha_lab",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=1,
                    msg="Chromium browser launched.",
                    args=(),
                    exc_info=None,
                )
            )
        finally:
            handler.close()

        log = ScraperRunLog.objects.get()
        self.assertEqual(log.run_id, self.run.pk)
        self.assertEqual(log.message, "Chromium browser launched.")

    def test_log_export_date_range_uses_local_calendar_days(self):
        before = ScraperRunLog.objects.create(run=self.run, message="before")
        inside = ScraperRunLog.objects.create(run=self.run, message="inside")
        after = ScraperRunLog.objects.create(run=self.run, message="after")
        for log, instant in (
            (before, datetime(2026, 9, 15, 20, 29, tzinfo=ZoneInfo("UTC"))),
            (inside, datetime(2026, 9, 15, 20, 31, tzinfo=ZoneInfo("UTC"))),
            (after, datetime(2026, 9, 16, 20, 31, tzinfo=ZoneInfo("UTC"))),
        ):
            ScraperRunLog.objects.filter(pk=log.pk).update(created_at=instant)
        admin = ScraperRunAdmin(ScraperRun, None)
        request = RequestFactory().get("/download-logs/", {
            "start_date": "1405/06/25", "end_date": "1405/06/25",
        })
        with timezone.override("Asia/Tehran"):
            response = admin.download_all_logs_view(request)
        content = response.content.decode()
        self.assertNotIn("before", content)
        self.assertIn("inside", content)
        self.assertNotIn("after", content)

    def test_log_export_rejects_reversed_dates(self):
        form = LogDateRangeForm({"start_date": "1405/06/26", "end_date": "1405/06/25"})
        self.assertFalse(form.is_valid())

    def test_jalali_log_dates_accept_persian_digits_and_validate_month_length(self):
        self.assertEqual(parse_jalali_date("۱۴۰۵/۰۶/۲۸"), datetime(2026, 9, 19).date())
        self.assertEqual(parse_jalali_date("1405/01/01"), datetime(2026, 3, 21).date())
        self.assertFalse(LogDateRangeForm({"start_date": "1405/07/31"}).is_valid())

    @override_settings(MIDDLEWARE=[
        "django.contrib.sessions.middleware.SessionMiddleware",
        "django.contrib.auth.middleware.AuthenticationMiddleware",
        "django.contrib.messages.middleware.MessageMiddleware",
    ])
    def test_jalali_display_script_is_present_on_admin_pages(self):
        user = get_user_model().objects.create_superuser(
            username="admin-date-test", email="admin@example.com", password="test-password"
        )
        self.client.force_login(user)
        for url in ("/admin/", "/admin/operations/scraperrun/"):
            response = self.client.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, "en-US-u-ca-persian")

        response = self.client.get("/admin/operations/scraperrun/")
        self.assertContains(response, "new PardisDatepicker")
        self.assertIsNotNone(finders.find(
            "operations/vendor/pardis-jalali-datepicker/1.0.2/pardis-jalali-datepicker.js"
        ))

    def test_database_handler_heartbeats_after_stop_request(self):
        old_heartbeat = timezone.now() - timezone.timedelta(minutes=10)
        self.run.status = ScraperRun.Status.STOP_REQUESTED
        self.run.heartbeat_at = old_heartbeat
        self.run.save(update_fields=["status", "heartbeat_at"])

        handler = ScraperRunDatabaseHandler(str(self.run.pk))
        handler._save_log_and_heartbeat(level="INFO", message="Stopping.")

        self.run.refresh_from_db()
        self.assertGreater(self.run.heartbeat_at, old_heartbeat)

    def test_structured_event_keeps_context_across_contextvar_switch(self):
        with bind_scraper_event_context(self.run):
            contextvars.Context().run(
                record_scraper_event,
                ScraperEvent.EventType.BROWSER_STARTED,
            )

        event = ScraperEvent.objects.get()
        self.assertEqual(event.run_id, self.run.pk)


class StaleStopRecoveryTests(TestCase):
    def test_stale_stop_request_is_closed_and_no_longer_active(self):
        old_time = timezone.now() - timezone.timedelta(minutes=6)
        run = ScraperRun.objects.create(
            status=ScraperRun.Status.STOP_REQUESTED,
            trigger=ScraperRun.Trigger.MANUAL,
            started_at=old_time,
            heartbeat_at=old_time,
            stop_requested_at=old_time,
        )

        self.assertEqual(recover_stale_scraper_runs(), 1)

        run.refresh_from_db()
        self.assertEqual(run.status, ScraperRun.Status.STOPPED)
        self.assertIsNotNone(run.finished_at)
        self.assertIsNotNone(run.stopped_at)
        self.assertFalse(active_scraper_runs().filter(pk=run.pk).exists())

    def test_live_stop_request_remains_active(self):
        run = ScraperRun.objects.create(
            status=ScraperRun.Status.STOP_REQUESTED,
            trigger=ScraperRun.Trigger.MANUAL,
            started_at=timezone.now() - timezone.timedelta(minutes=6),
            heartbeat_at=timezone.now(),
            stop_requested_at=timezone.now() - timezone.timedelta(minutes=6),
        )

        self.assertEqual(recover_stale_scraper_runs(), 0)

        run.refresh_from_db()
        self.assertEqual(run.status, ScraperRun.Status.STOP_REQUESTED)

class AsyncSafeScraperObservabilityTests(TransactionTestCase):
    def setUp(self):
        self.run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.SCHEDULED,
            visa_sub_types=["Student Visa"],
        )

    def test_database_write_moves_out_of_async_context(self):
        caller_thread = threading.get_ident()

        def operation():
            if threading.get_ident() == caller_thread:
                raise SynchronousOnlyOperation("async context")
            return threading.get_ident()

        writer_thread = run_database_write(operation)

        self.assertNotEqual(writer_thread, caller_thread)

    def test_database_handler_persists_log_from_async_context(self):
        handler = ScraperRunDatabaseHandler(str(self.run.pk))
        handler.setFormatter(logging.Formatter("%(message)s"))

        async def emit_log():
            handler.emit(
                logging.LogRecord(
                    name="captcha_lab",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=1,
                    msg="CAPTCHA telemetry: stage=verification",
                    args=(),
                    exc_info=None,
                )
            )

        try:
            asyncio.run(emit_log())
        finally:
            handler.close()

        log = ScraperRunLog.objects.get()
        self.assertEqual(
            log.message,
            "CAPTCHA telemetry: stage=verification",
        )

    def test_structured_event_persists_from_async_context(self):
        async def create_event():
            with bind_scraper_event_context(self.run):
                return record_scraper_event(
                    ScraperEvent.EventType.CAPTCHA_STAGE,
                    status="succeeded",
                    data={"stage": "verification"},
                )

        event = asyncio.run(create_event())

        self.assertIsNotNone(event)
        saved = ScraperEvent.objects.get()
        self.assertEqual(saved.run_id, self.run.pk)
        self.assertEqual(saved.data["stage"], "verification")
