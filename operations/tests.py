import contextvars
import logging

from django.test import TestCase
from django.utils import timezone

from scraper.models import ScraperConfig

from .events import bind_scraper_event_context, record_event_from_log, record_scraper_event
from .models import ScraperEvent, ScraperRun, ScraperRunLog
from .run_logging import ScraperRunDatabaseHandler
from .services import ScraperRunAlreadyStarted, execute_scraper_run


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

    def test_structured_event_keeps_context_across_contextvar_switch(self):
        with bind_scraper_event_context(self.run):
            contextvars.Context().run(
                record_scraper_event,
                ScraperEvent.EventType.BROWSER_STARTED,
            )

        event = ScraperEvent.objects.get()
        self.assertEqual(event.run_id, self.run.pk)
