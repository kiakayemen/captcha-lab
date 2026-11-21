"""PostgreSQL-only concurrency check for scraper admission."""
import threading
from unittest import skipUnless
from unittest.mock import patch
from django.db import close_old_connections, connection, connections
from django.test import TransactionTestCase
from django.utils import timezone
from scraper.models import ScraperConfig, ScraperResult, ScraperStatus
from .models import ScraperRun, ScraperSchedule
from .services import execute_scraper_run


@skipUnless(connection.vendor == 'postgresql', 'PostgreSQL row locks required')
class ConcurrentAdmissionTests(TransactionTestCase):
    def test_only_one_of_two_concurrent_executions_enters_scraper(self):
        ScraperSchedule.objects.create(pk=1)
        runs = [ScraperRun.objects.create() for _ in range(2)]
        start = threading.Barrier(2)
        entered = threading.Event()
        release = threading.Event()
        errors = []
        def fake_scraper(*args, **kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError('second worker did not finish')
            now = timezone.now()
            return ScraperResult(ScraperStatus.NO_APPOINTMENT, now, now)
        def worker(pk):
            close_old_connections()
            try:
                start.wait(timeout=10)
                execute_scraper_run(config=ScraperConfig(), db_run=ScraperRun.objects.get(pk=pk))
            except Exception as exc:
                errors.append(repr(exc))
            finally:
                connections.close_all()
        with patch('scraper.service.run_scraper', side_effect=fake_scraper):
            threads = [threading.Thread(target=worker, args=(run.pk,)) for run in runs]
            for thread in threads: thread.start()
            self.assertTrue(entered.wait(10))
            for thread in threads: thread.join(timeout=2)
            release.set()
            for thread in threads: thread.join(timeout=15)
        self.assertFalse(errors, errors)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(ScraperRun.objects.filter(started_at__isnull=False).count(), 1)
        self.assertEqual(ScraperRun.objects.filter(error_type='RunAdmissionBlocked').count(), 1)
