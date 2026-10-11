import asyncio
import contextvars
import csv
import logging
import threading
import time
from io import StringIO
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from django.core.exceptions import SynchronousOnlyOperation
from django.core.management import call_command
from django.test import TestCase, TransactionTestCase, override_settings
from django.test import RequestFactory
from django.contrib.auth import get_user_model
from django.contrib.staticfiles import finders
from django.utils import timezone

from scraper.models import ScraperConfig
from notifications import notify_consecutive_403

from .events import bind_scraper_event_context, record_event_from_log, record_scraper_event
from .database_writer import run_database_write
from .models import (
    ProxyEndpointHealth,
    ScraperEvent,
    ScraperRun,
    ScraperRunLog,
    ScraperSchedule,
)
from .admin import LogDateRangeForm, ScraperRunAdmin
from .jalali import parse_jalali_date
from .run_logging import ScraperRunDatabaseHandler
from .proxy_health import select_proxy_pool, update_proxy_health_from_run
from .tasks import run_scraper_task, run_scheduled_scraper_task, schedule_phase, update_scheduled_403_streak
from .services import (
    ScraperRunAlreadyStarted,
    active_scraper_runs,
    execute_scraper_run,
    recover_stale_scraper_runs,
    request_scraper_stop,
    monitor_stop_requests,
    serialize_scraper_config,
)


class StopMonitorTests(TransactionTestCase):
    def test_cooperative_stop_does_not_exit_worker(self):
        run = ScraperRun.objects.create(status=ScraperRun.Status.RUNNING)
        with patch("operations.services.STOP_GRACE_SECONDS", 0.05), patch(
            "operations.services.os._exit"
        ) as exit_worker:
            with monitor_stop_requests(run.pk) as should_stop:
                request_scraper_stop(run)
                deadline = time.monotonic() + 3
                while not should_stop() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(should_stop())
            exit_worker.assert_not_called()

    def test_stuck_scraper_exits_after_explicit_stop(self):
        run = ScraperRun.objects.create(status=ScraperRun.Status.RUNNING)
        exit_called = threading.Event()
        with patch("operations.services.STOP_GRACE_SECONDS", 0.05), patch(
            "operations.services.os._exit", side_effect=lambda _code: exit_called.set()
        ) as exit_worker:
            with monitor_stop_requests(run.pk) as should_stop:
                request_scraper_stop(run)
                self.assertTrue(exit_called.wait(3))
                self.assertTrue(should_stop())
            exit_worker.assert_called_once_with(1)


@patch.dict("os.environ", {"SCRAPER_PROXY_URLS": "http://proxy-1.internal:8888,http://proxy-2.internal:8888"})
class BackToBackScheduleTests(TestCase):
    def setUp(self):
        ScraperSchedule.objects.create(enabled=True, interval_minutes=30)

    @staticmethod
    def at(hour, minute):
        return timezone.make_aware(
            datetime(2026, 10, 12, hour, minute), ZoneInfo("Asia/Tehran")
        )

    def test_window_uses_tehran_time_and_ends_at_eleven(self):
        self.assertEqual(schedule_phase(self.at(9, 49)), "interval")
        self.assertEqual(schedule_phase(self.at(9, 50)), "prepare")
        self.assertEqual(schedule_phase(self.at(10, 0)), "back_to_back")
        self.assertEqual(schedule_phase(self.at(10, 59)), "back_to_back")
        self.assertEqual(schedule_phase(self.at(11, 0)), "interval")

    @patch("operations.tasks.run_scraper_task.delay")
    def test_preparation_requests_stop_and_launches_nothing(self, enqueue_run):
        run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.SCHEDULED,
            status=ScraperRun.Status.RUNNING,
            started_at=self.at(9, 40),
            heartbeat_at=self.at(9, 49),
        )
        with patch("operations.tasks.timezone.now", return_value=self.at(9, 50)):
            result = run_scheduled_scraper_task.run()
        run.refresh_from_db()
        self.assertEqual(result, "preparing")
        self.assertEqual(run.status, ScraperRun.Status.STOP_REQUESTED)
        enqueue_run.assert_not_called()

    @patch("operations.tasks.run_scraper_task.delay")
    def test_ten_oclock_launches_one_burst_run(self, enqueue_run):
        with patch("operations.tasks.timezone.now", return_value=self.at(10, 0)):
            with self.captureOnCommitCallbacks(execute=True):
                first = run_scheduled_scraper_task.run()
            second = run_scheduled_scraper_task.run()
        self.assertEqual(ScraperRun.objects.get(pk=first).trigger, ScraperRun.Trigger.BACK_TO_BACK)
        self.assertEqual(second, f"active:{first}")
        enqueue_run.assert_called_once()

    @patch("operations.tasks.run_scraper_task.delay")
    def test_last_burst_finish_anchors_regular_interval(self, enqueue_run):
        schedule = ScraperSchedule.load()
        schedule.last_dispatched_at = self.at(9, 30)
        schedule.last_back_to_back_finished_at = self.at(11, 5)
        schedule.save(update_fields=["last_dispatched_at", "last_back_to_back_finished_at"])
        with patch("operations.tasks.timezone.now", return_value=self.at(11, 34)):
            self.assertEqual(run_scheduled_scraper_task.run(), "not_due")
        with patch("operations.tasks.timezone.now", return_value=self.at(11, 35)):
            with self.captureOnCommitCallbacks(execute=True):
                run_id = run_scheduled_scraper_task.run()
        self.assertEqual(ScraperRun.objects.get(pk=run_id).trigger, ScraperRun.Trigger.SCHEDULED)
        enqueue_run.assert_called_once()

    @patch("operations.tasks.run_scraper_task.delay")
    def test_run_finishing_before_eleven_starts_another_burst(self, enqueue_run):
        with patch("operations.tasks.timezone.now", return_value=self.at(10, 59)):
            with self.captureOnCommitCallbacks(execute=True):
                run_id = run_scheduled_scraper_task.run()
        self.assertEqual(ScraperRun.objects.get(pk=run_id).trigger, ScraperRun.Trigger.BACK_TO_BACK)
        enqueue_run.assert_called_once()

    @patch("operations.tasks.run_scraper_task.delay")
    def test_active_final_burst_is_left_running_after_eleven(self, enqueue_run):
        run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.BACK_TO_BACK,
            status=ScraperRun.Status.RUNNING,
            started_at=self.at(10, 59),
            heartbeat_at=self.at(11, 0),
        )
        with patch("operations.tasks.timezone.now", return_value=self.at(11, 0)):
            result = run_scheduled_scraper_task.run()
        run.refresh_from_db()
        self.assertEqual(result, f"active:{run.pk}")
        self.assertEqual(run.status, ScraperRun.Status.RUNNING)
        enqueue_run.assert_not_called()

    @patch("operations.tasks.run_scheduled_scraper_task.delay")
    @patch("operations.tasks.update_scheduled_403_streak")
    @patch("operations.tasks.update_proxy_health_from_run")
    @patch("operations.tasks.execute_scraper_run")
    def test_burst_completion_records_finish_and_requests_next_check(
        self, execute_run, _health, _streak, next_check
    ):
        run = ScraperRun.objects.create(trigger=ScraperRun.Trigger.BACK_TO_BACK)
        finished = self.at(11, 5)

        def complete(**kwargs):
            db_run = kwargs["db_run"]
            db_run.status = ScraperRun.Status.NO_APPOINTMENT
            db_run.finished_at = finished
            db_run.save(update_fields=["status", "finished_at"])

        execute_run.side_effect = complete
        run_scraper_task.run(str(run.pk), serialize_scraper_config(ScraperConfig()))
        schedule = ScraperSchedule.load()
        self.assertEqual(schedule.last_back_to_back_finished_at, finished)
        next_check.assert_called_once()


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

    def test_admin_counts_distinct_observed_egress_ips(self):
        for egress_ip in (
            "203.0.113.10",
            "203.0.113.11",
            "203.0.113.10",
        ):
            ScraperEvent.objects.create(
                run=self.run,
                event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
                data={"egress_ip": egress_ip},
            )
        ScraperEvent.objects.create(
            run=self.run,
            event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
            data={"egress_lookup_error": "probe failed"},
        )

        admin = ScraperRunAdmin(ScraperRun, None)
        request = RequestFactory().get("/admin/operations/scraperrun/")
        run = admin.get_queryset(request).get(pk=self.run.pk)

        self.assertEqual(admin.ips_used(run), 2)
        with self.assertNumQueries(0):
            self.assertEqual(admin.ips_used(run), 2)

    def test_admin_shows_owned_ip_names_for_a_run(self):
        for egress_ip in ("185.239.3.80", "94.184.43.110"):
            ScraperEvent.objects.create(
                run=self.run,
                event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
                data={"egress_ip": egress_ip},
            )

        admin = ScraperRunAdmin(ScraperRun, None)
        request = RequestFactory().get("/admin/operations/scraperrun/")
        run = admin.get_queryset(request).get(pk=self.run.pk)

        self.assertEqual(
            admin.ip_names(run),
            "ParsPack proxy 2 primary — active, "
            "ParsPack proxy 1 reserved — active",
        )
        with self.assertNumQueries(0):
            admin.ip_names(run)

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


@patch.dict(
    "os.environ",
    {
        "SCRAPER_PROXY_URLS": (
            "http://proxy-1.internal:8888,http://proxy-2.internal:8888"
        )
    },
)
class ProxyCircuitBreakerTests(TestCase):
    @patch("operations.tasks.notify_consecutive_403")
    def test_tenth_scheduled_403_disables_schedule_and_notifies_once(self, notify):
        schedule = ScraperSchedule.load()
        schedule.enabled = True
        schedule.consecutive_403_runs = 9
        schedule.save(update_fields=["enabled", "consecutive_403_runs", "updated_at"])
        run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.SCHEDULED,
            visa_sub_types=["Student Visa"],
            error_type="HTTP403Forbidden",
        )

        with self.captureOnCommitCallbacks(execute=True):
            count = update_scheduled_403_streak(run)

        schedule.refresh_from_db()
        self.assertEqual(count, 10)
        self.assertEqual(schedule.consecutive_403_runs, 10)
        self.assertFalse(schedule.enabled)
        notify.assert_called_once_with(run_id=str(run.pk), count=10)

    @patch("operations.tasks.notify_consecutive_403")
    def test_non_403_run_resets_scheduled_streak(self, notify):
        schedule = ScraperSchedule.load()
        schedule.consecutive_403_runs = 9
        schedule.save(update_fields=["consecutive_403_runs", "updated_at"])
        run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.SCHEDULED,
            visa_sub_types=["Student Visa"],
            error_type="",
        )

        count = update_scheduled_403_streak(run)

        schedule.refresh_from_db()
        self.assertEqual(count, 0)
        self.assertEqual(schedule.consecutive_403_runs, 0)
        self.assertTrue(schedule.enabled)
        notify.assert_not_called()

    @patch("notifications._append_event_log")
    @patch("notifications._send_email")
    def test_403_alert_uses_only_requested_recipient(self, send_email, _event_log):
        notify_consecutive_403(run_id="run-123", count=10)

        self.assertEqual(
            send_email.call_args.kwargs["recipients_override"],
            ["ikiapad00@gmail.com"],
        )

    def test_healthy_pool_excludes_quarantined_endpoint(self):
        ProxyEndpointHealth.objects.create(
            endpoint="http://proxy-1.internal:8888",
            quarantined_until=timezone.now() + timezone.timedelta(minutes=40),
        )

        selection = select_proxy_pool()

        self.assertEqual(
            selection.proxy_urls,
            ("http://proxy-2.internal:8888",),
        )
        self.assertFalse(selection.recovery_probe)

    def test_all_quarantined_endpoints_wait_until_probe_is_due(self):
        future = timezone.now() + timezone.timedelta(minutes=40)
        for endpoint in (
            "http://proxy-1.internal:8888",
            "http://proxy-2.internal:8888",
        ):
            ProxyEndpointHealth.objects.create(
                endpoint=endpoint,
                quarantined_until=future,
            )

        selection = select_proxy_pool()

        self.assertEqual(selection.proxy_urls, ())
        self.assertFalse(selection.recovery_probe)

    def test_expired_quarantines_restore_all_endpoints(self):
        now = timezone.now()
        ProxyEndpointHealth.objects.create(
            endpoint="http://proxy-1.internal:8888",
            quarantined_until=now - timezone.timedelta(minutes=1),
            last_checked_at=now - timezone.timedelta(minutes=80),
        )
        ProxyEndpointHealth.objects.create(
            endpoint="http://proxy-2.internal:8888",
            quarantined_until=now - timezone.timedelta(minutes=1),
            last_checked_at=now - timezone.timedelta(minutes=40),
        )

        selection = select_proxy_pool(now)

        self.assertEqual(
            selection.proxy_urls,
            (
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
            ),
        )
        self.assertTrue(selection.recovery_probe)
        self.assertFalse(
            ProxyEndpointHealth.objects.filter(
                quarantined_until__isnull=False
            ).exists()
        )

    def test_expired_quarantine_returns_even_when_an_ip_is_already_healthy(self):
        now = timezone.now()
        ProxyEndpointHealth.objects.create(
            endpoint="http://proxy-1.internal:8888",
            quarantined_until=now - timezone.timedelta(minutes=1),
        )

        selection = select_proxy_pool(now)

        self.assertEqual(
            selection.proxy_urls,
            (
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
            ),
        )
        self.assertTrue(selection.recovery_probe)

    def test_run_events_quarantine_403_and_clear_on_later_200(self):
        run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.SCHEDULED,
            visa_sub_types=["Student Visa"],
        )
        endpoint = "http://proxy-1.internal:8888"
        ScraperEvent.objects.create(
            run=run,
            event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
            status="403",
            data={"proxy_endpoint": endpoint},
        )

        update_proxy_health_from_run(run, cooldown_minutes=40)
        health = ProxyEndpointHealth.objects.get(endpoint=endpoint)
        self.assertEqual(health.consecutive_403, 1)
        self.assertIsNotNone(health.quarantined_until)

        recovery_run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.SCHEDULED,
            visa_sub_types=["Student Visa"],
        )
        ScraperEvent.objects.create(
            run=recovery_run,
            event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
            status="200",
            data={"proxy_endpoint": endpoint},
        )
        update_proxy_health_from_run(recovery_run, cooldown_minutes=40)
        health.refresh_from_db()
        self.assertEqual(health.consecutive_403, 0)
        self.assertIsNone(health.quarantined_until)
        self.assertIsNotNone(health.last_success_at)

    @patch("operations.tasks.schedule_phase", return_value="interval")
    @patch("operations.tasks.run_scraper_task.delay")
    def test_scheduler_continues_with_remaining_healthy_endpoint(
        self,
        enqueue_run,
        _phase,
    ):
        ProxyEndpointHealth.objects.create(
            endpoint="http://proxy-1.internal:8888",
            quarantined_until=timezone.now() + timezone.timedelta(minutes=40),
        )
        ScraperSchedule.objects.create(enabled=True, interval_minutes=40)

        with self.captureOnCommitCallbacks(execute=True):
            result = run_scheduled_scraper_task.run()

        self.assertTrue(result)
        payload = enqueue_run.call_args.args[1]
        self.assertEqual(payload["proxy_urls"], ["http://proxy-2.internal:8888"])
        self.assertTrue(payload["allow_single_proxy"])

    @patch("operations.tasks.schedule_phase", return_value="interval")
    @patch("operations.tasks.run_scraper_task.delay")
    def test_scheduler_does_not_launch_full_run_while_all_probes_cooling_down(
        self,
        enqueue_run,
        _phase,
    ):
        future = timezone.now() + timezone.timedelta(minutes=40)
        for endpoint in (
            "http://proxy-1.internal:8888",
            "http://proxy-2.internal:8888",
        ):
            ProxyEndpointHealth.objects.create(
                endpoint=endpoint,
                quarantined_until=future,
            )
        ScraperSchedule.objects.create(enabled=True, interval_minutes=40)

        result = run_scheduled_scraper_task.run()

        self.assertEqual(result, "circuit_open")
        enqueue_run.assert_not_called()
        self.assertFalse(ScraperRun.objects.exists())

    @patch("operations.tasks.schedule_phase", return_value="interval")
    @patch("operations.tasks.run_scraper_task.delay")
    def test_scheduler_restores_all_due_recovery_endpoints(
        self,
        enqueue_run,
        _phase,
    ):
        now = timezone.now()
        for index, endpoint in enumerate((
            "http://proxy-1.internal:8888",
            "http://proxy-2.internal:8888",
        )):
            ProxyEndpointHealth.objects.create(
                endpoint=endpoint,
                quarantined_until=now - timezone.timedelta(minutes=1),
                last_checked_at=now - timezone.timedelta(minutes=80 - index * 40),
            )
        ScraperSchedule.objects.create(enabled=True, interval_minutes=40)

        with self.captureOnCommitCallbacks(execute=True):
            run_scheduled_scraper_task.run()

        config = enqueue_run.call_args.args[1]
        self.assertEqual(
            config["proxy_urls"],
            [
                "http://proxy-1.internal:8888",
                "http://proxy-2.internal:8888",
            ],
        )
        self.assertFalse(config["allow_single_proxy"])


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
        self.assertTrue(response.streaming)
        self.assertEqual(
            response["Content-Disposition"],
            'attachment; filename="scraper_run_logs_2026-09-16_to_2026-09-16.csv"',
        )
        content = b"".join(response.streaming_content).decode()
        self.assertNotIn("before", content)
        self.assertIn("inside", content)
        self.assertNotIn("after", content)

    def test_log_export_streams_valid_csv_without_query_cache(self):
        ScraperRunLog.objects.create(
            run=self.run,
            level="WARNING",
            message='large, quoted "message"',
        )
        admin = ScraperRunAdmin(ScraperRun, None)

        response = admin.download_all_logs_view(
            RequestFactory().get("/download-logs/")
        )

        self.assertTrue(response.streaming)
        self.assertEqual(
            response["Content-Disposition"],
            (
                'attachment; filename="scraper_run_logs_'
                f'{timezone.localdate()}_to_{timezone.localdate()}.csv"'
            ),
        )
        content = b"".join(response.streaming_content).decode()
        rows = list(csv.DictReader(StringIO(content)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["level"], "WARNING")
        self.assertEqual(rows[0]["message"], 'large, quoted "message"')

    @override_settings(MIDDLEWARE=[
        "django.contrib.sessions.middleware.SessionMiddleware",
        "django.contrib.auth.middleware.AuthenticationMiddleware",
        "django.contrib.messages.middleware.MessageMiddleware",
    ])
    def test_selected_run_log_action_redirects_to_get_csv_for_selected_runs(self):
        selected_run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.MANUAL,
        )
        unselected_run = ScraperRun.objects.create(
            trigger=ScraperRun.Trigger.MANUAL,
        )
        ScraperRunLog.objects.create(run=self.run, message="selected first")
        ScraperRunLog.objects.create(
            run=selected_run,
            message="selected second",
        )
        ScraperRunLog.objects.create(
            run=unselected_run,
            message="not selected",
        )
        user = get_user_model().objects.create_superuser(
            username="selected-log-export-admin",
            email="selected-export@example.com",
            password="test-password",
        )
        self.client.force_login(user)

        action_response = self.client.post(
            "/admin/operations/scraperrun/",
            {
                "action": "download_selected_logs",
                "_selected_action": [str(self.run.pk), str(selected_run.pk)],
                "index": "0",
            },
        )
        self.assertEqual(action_response.status_code, 302)
        self.assertIn(
            "/admin/operations/scraperrun/download-selected-logs/",
            action_response["Location"],
        )

        response = self.client.get(action_response["Location"])
        self.assertTrue(response.streaming)
        self.assertEqual(
            response["Content-Disposition"],
            (
                'attachment; filename="selected_scraper_run_logs_'
                f'{timezone.localdate()}_to_{timezone.localdate()}.csv"'
            ),
        )
        rows = list(
            csv.DictReader(
                StringIO(b"".join(response.streaming_content).decode())
            )
        )
        self.assertEqual(
            {row["message"] for row in rows},
            {"selected first", "selected second"},
        )

    def test_single_run_log_filename_includes_actual_date_range(self):
        first = ScraperRunLog.objects.create(run=self.run, message="first")
        last = ScraperRunLog.objects.create(run=self.run, message="last")
        ScraperRunLog.objects.filter(pk=first.pk).update(
            created_at=datetime(2026, 9, 15, 20, 31, tzinfo=ZoneInfo("UTC"))
        )
        ScraperRunLog.objects.filter(pk=last.pk).update(
            created_at=datetime(2026, 9, 17, 20, 31, tzinfo=ZoneInfo("UTC"))
        )
        admin = ScraperRunAdmin(ScraperRun, None)

        with timezone.override("Asia/Tehran"):
            response = admin.download_run_logs_view(
                RequestFactory().get("/download-logs/"),
                self.run.pk,
            )

        self.assertEqual(
            response["Content-Disposition"],
            (
                f'attachment; filename="scraper_run_{self.run.pk}_logs_'
                '2026-09-16_to_2026-09-18.csv"'
            ),
        )

    def test_log_filename_uses_exported_dates_not_wider_filter(self):
        first = ScraperRunLog.objects.create(run=self.run, message="first")
        last = ScraperRunLog.objects.create(run=self.run, message="last")
        ScraperRunLog.objects.filter(pk=first.pk).update(
            created_at=datetime(2026, 9, 22, 20, 31, tzinfo=ZoneInfo("UTC"))
        )
        ScraperRunLog.objects.filter(pk=last.pk).update(
            created_at=datetime(2026, 9, 24, 20, 31, tzinfo=ZoneInfo("UTC"))
        )
        admin = ScraperRunAdmin(ScraperRun, None)
        request = RequestFactory().get(
            "/download-logs/",
            {"start_date": "1405/07/01", "end_date": "1405/07/04"},
        )

        with timezone.override("Asia/Tehran"):
            response = admin.download_all_logs_view(request)

        self.assertEqual(
            response["Content-Disposition"],
            (
                'attachment; filename="scraper_run_logs_'
                '2026-09-23_to_2026-09-25.csv"'
            ),
        )

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
