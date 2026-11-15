from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone

from .models import ScraperRun, ScraperSchedule
from .rate_limit import note_rate_limit, retry_after_seconds
from .services import execute_scraper_run
from .tasks import run_scheduled_scraper_task
from scraper.models import ScraperConfig


class RateLimitTests(TestCase):
    def setUp(self):
        self.now = timezone.now()
        ScraperSchedule.objects.create(enabled=True, interval_minutes=45, last_dispatched_at=self.now)

    @patch("operations.rate_limit.random.randint", return_value=10)
    def test_cooldown_persists_and_escalates_after_repeat(self, _jitter):
        until = note_rate_limit(now=self.now)
        self.assertEqual(until, self.now + timedelta(seconds=250))
        schedule = ScraperSchedule.objects.get(pk=1)
        self.assertTrue(schedule.rate_limit_retry_pending)
        self.assertEqual(schedule.rate_limit_streak, 1)
        until2 = note_rate_limit(now=until + timedelta(seconds=1))
        self.assertEqual(until2, until + timedelta(seconds=491))

    @patch("operations.rate_limit.random.randint", return_value=0)
    def test_retry_after_is_a_minimum_and_never_shortens_existing_cooldown(self, _jitter):
        until = note_rate_limit("900", now=self.now)
        self.assertEqual(until, self.now + timedelta(seconds=900))
        self.assertEqual(note_rate_limit("1", now=self.now + timedelta(seconds=1)), until)
        self.assertEqual(retry_after_seconds("invalid", self.now), 0)
        self.assertEqual(retry_after_seconds("Sun, 15 Nov 2026 09:10:00 GMT", self.now.replace(year=2026, month=11, day=15, hour=9, minute=0, second=0, microsecond=0)), 600)

    @patch("operations.tasks.schedule_phase", return_value="interval")
    @patch("operations.tasks.run_scraper_task.delay")
    @patch.dict("os.environ", {"SCRAPER_PROXY_URLS": "http://one:8888,http://two:8888"})
    def test_scheduler_waits_then_dispatches_without_waiting_full_interval(self, enqueue, _phase):
        note_rate_limit(now=self.now)
        self.assertEqual(run_scheduled_scraper_task.run(), "rate_limited")
        enqueue.assert_not_called()
        ScraperSchedule.objects.filter(pk=1).update(rate_limited_until=self.now - timedelta(seconds=1))
        with self.captureOnCommitCallbacks(execute=True):
            run_scheduled_scraper_task.run()
        enqueue.assert_called_once()
        self.assertFalse(ScraperSchedule.objects.get(pk=1).rate_limit_retry_pending)

    def test_manual_or_queued_run_does_not_start_during_cooldown(self):
        note_rate_limit(now=self.now)
        run = ScraperRun.objects.create()
        result = execute_scraper_run(config=ScraperConfig(), db_run=run)
        self.assertEqual(result.error_type, "HTTP429Cooldown")
        self.assertEqual(result.status, ScraperRun.Status.FAILED)
        self.assertIsNone(result.started_at)
