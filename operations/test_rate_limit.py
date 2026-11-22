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

    @patch("operations.rate_limit.random.randint", return_value=0)
    def test_later_longer_header_extends_active_cooldown(self, _jitter):
        self.assertEqual(note_rate_limit("240", now=self.now), self.now + timedelta(seconds=240))
        later = self.now + timedelta(seconds=10)
        self.assertEqual(note_rate_limit("3600", now=later), later + timedelta(seconds=3600))

    @patch("operations.tasks.schedule_phase", return_value="interval")
    @patch("operations.tasks.run_scraper_task.delay")
    @patch.dict("os.environ", {"SCRAPER_PROXY_URLS": "http://one:8888,http://two:8888"})
    def test_scheduler_waits_for_normal_interval_after_cooldown(self, enqueue, _phase):
        note_rate_limit(now=self.now)
        self.assertEqual(run_scheduled_scraper_task.run(), "not_due")
        enqueue.assert_not_called()
        ScraperSchedule.objects.filter(pk=1).update(rate_limited_until=self.now - timedelta(seconds=1))
        self.assertEqual(run_scheduled_scraper_task.run(), 'not_due')
        enqueue.assert_not_called()

    def test_finished_run_blocks_new_execution_for_twenty_minutes(self):
        ScraperRun.objects.create(status=ScraperRun.Status.NO_APPOINTMENT,
            started_at=self.now - timedelta(minutes=5), finished_at=self.now)
        with patch('operations.services.timezone.now', return_value=self.now + timedelta(minutes=19)):
            run = execute_scraper_run(config=ScraperConfig())
        self.assertEqual(run.error_type, 'RunAdmissionBlocked')
        self.assertIsNone(run.started_at)

    def test_active_run_blocks_new_execution(self):
        ScraperRun.objects.create(status=ScraperRun.Status.RUNNING, started_at=self.now)
        run = execute_scraper_run(config=ScraperConfig())
        self.assertEqual(run.error_type, 'RunAdmissionBlocked')
        self.assertIsNone(run.started_at)

    @patch('scraper.service.run_scraper')
    def test_failed_cooldown_write_keeps_durable_run_blocked(self, scrape):
        from flows.errors import RateLimitPersistenceError
        scrape.side_effect = RateLimitPersistenceError('database unavailable')
        run = ScraperRun.objects.create()
        with self.assertRaises(RateLimitPersistenceError):
            execute_scraper_run(config=ScraperConfig(), db_run=run)
        run.refresh_from_db()
        self.assertEqual(run.status, ScraperRun.Status.RUNNING)
        self.assertIsNone(run.finished_at)
        next_run = execute_scraper_run(config=ScraperConfig())
        self.assertEqual(next_run.error_type, 'RunAdmissionBlocked')

    def test_stale_running_heartbeat_does_not_allow_a_new_run(self):
        from .services import recover_stale_scraper_runs
        old = self.now - timedelta(hours=2)
        active = ScraperRun.objects.create(status=ScraperRun.Status.RUNNING,
            started_at=old, heartbeat_at=old)
        recover_stale_scraper_runs()
        active.refresh_from_db()
        self.assertEqual(active.status, ScraperRun.Status.RUNNING)
        self.assertEqual(execute_scraper_run(config=ScraperConfig()).error_type,
                         'RunAdmissionBlocked')

    def test_manual_or_queued_run_does_not_start_during_cooldown(self):
        note_rate_limit(now=self.now)
        run = ScraperRun.objects.create()
        result = execute_scraper_run(config=ScraperConfig(direct_connection=True), db_run=run)
        self.assertEqual(result.error_type, "HTTP429Cooldown")
        self.assertEqual(result.status, ScraperRun.Status.FAILED)
        self.assertIsNone(result.started_at)

class PerProxyRateLimitTests(TestCase):
    @patch('operations.rate_limit.random.randint', return_value=0)
    def test_cooldowns_are_independent_and_never_shortened(self, _jitter):
        from .models import ProxyEndpointHealth
        now = timezone.now()
        one = 'http://one:8888'
        two = 'http://two:8888'
        self.assertEqual(note_rate_limit('600', now=now, endpoint=one), now + timedelta(seconds=600))
        self.assertEqual(note_rate_limit('1', now=now + timedelta(seconds=1), endpoint=one), now + timedelta(seconds=600))
        self.assertFalse(ProxyEndpointHealth.objects.filter(endpoint=two).exists())
        self.assertFalse(ScraperSchedule.objects.exists())

    @patch('operations.rate_limit.random.randint', return_value=0)
    def test_pool_excludes_only_rate_limited_proxy_and_restores_it(self, _jitter):
        from .proxy_health import select_proxy_pool
        now = timezone.now()
        proxies = ('http://one:8888', 'http://two:8888')
        note_rate_limit(now=now, endpoint=proxies[0])
        self.assertEqual(select_proxy_pool(now, proxy_urls=proxies).proxy_urls, (proxies[1],))
        self.assertEqual(select_proxy_pool(now + timedelta(seconds=241), proxy_urls=proxies).proxy_urls, proxies)
        note_rate_limit(now=now, endpoint=proxies[1])
        self.assertEqual(select_proxy_pool(now, proxy_urls=proxies).proxy_urls, ())

    @patch('operations.rate_limit.random.randint', return_value=0)
    @patch('operations.tasks.run_scraper_task.delay')
    @patch.dict('os.environ', {'SCRAPER_PROXY_URLS': 'http://one:8888,http://two:8888'})
    def test_scheduler_waits_when_every_proxy_is_rate_limited(self, enqueue, _jitter):
        now = timezone.now()
        ScraperSchedule.objects.create(enabled=True, interval_minutes=45,
                                       last_dispatched_at=now - timedelta(hours=1))
        note_rate_limit(now=now, endpoint='http://one:8888')
        note_rate_limit(now=now, endpoint='http://two:8888')
        with patch('operations.tasks.timezone.now', return_value=now), \
             patch('operations.tasks.schedule_phase', return_value='interval'):
            self.assertEqual(run_scheduled_scraper_task.run(), 'circuit_open')
        enqueue.assert_not_called()

    @patch('scraper.service.run_scraper')
    @patch('operations.rate_limit.random.randint', return_value=0)
    def test_manual_run_filters_rate_limited_endpoint(self, _jitter, scrape):
        from scraper.models import ScraperResult, ScraperStatus
        now = timezone.now()
        one, two = 'http://one:8888', 'http://two:8888'
        note_rate_limit(now=now, endpoint=one)
        scrape.return_value = ScraperResult(status=ScraperStatus.NO_APPOINTMENT,
                                            started_at=now, finished_at=now)
        result = execute_scraper_run(config=ScraperConfig(proxy_urls=(one, two)))
        self.assertEqual(result.status, ScraperRun.Status.NO_APPOINTMENT)
        used = scrape.call_args.args[0]
        self.assertEqual(used.proxy_urls, (two,))
        self.assertTrue(used.allow_single_proxy)
