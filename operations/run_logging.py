from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .database_writer import run_database_write
from .models import (
    ScraperRun,
    ScraperRunLog,
)
from .events import (
    bind_scraper_event_context,
    record_event_from_log,
)


class ScraperRunDatabaseHandler(
    logging.Handler
):
    def __init__(
        self,
        run_id: str,
    ) -> None:
        super().__init__()
        self.run_id = str(run_id)

    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        message = self.format(record)

        try:
            run_database_write(
                lambda: self._save_log_and_heartbeat(
                    level=record.levelname,
                    message=message,
                )
            )
        except Exception:
            #
            # Logging must never be capable
            # of crashing the scraper.
            #
            pass

        try:
            # Keep lifecycle context changes on the scraper thread.  Any ORM
            # write created by this parser uses the same safe writer path.
            record_event_from_log(message)
        except Exception:
            pass

    def _save_log_and_heartbeat(
        self,
        *,
        level: str,
        message: str,
    ) -> None:
        ScraperRunLog.objects.create(
            run_id=self.run_id,
            level=level,
            message=message,
        )

        # Every meaningful scraper log acts as a heartbeat.  If the worker
        # disappears completely, this timestamp stops advancing.
        ScraperRun.objects.filter(
            pk=self.run_id,
            status__in=(
                ScraperRun.Status.RUNNING,
                ScraperRun.Status.STOP_REQUESTED,
            ),
        ).update(
            heartbeat_at=timezone.now()
        )


class ScraperRunFileHandler(
    logging.FileHandler
):
    """Write a complete, separate transcript for one scraper run."""

    def __init__(
        self,
        run: ScraperRun,
    ) -> None:
        logs_dir = Path(settings.BASE_DIR) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        started_at = run.started_at or run.created_at
        timestamp = timezone.localtime(started_at).strftime(
            "%Y%m%d_%H%M%S"
        )
        path = logs_dir / f"scraper_{timestamp}_{run.pk}.log"

        super().__init__(
            filename=path,
            mode="a",
            encoding="utf-8",
        )


@contextmanager
def bind_scraper_run_logging(
    run: ScraperRun,
) -> Iterator[None]:
    logger = logging.getLogger(
        "captcha_lab"
    )

    handler = (
        ScraperRunDatabaseHandler(
            run_id=str(run.pk),
        )
    )

    handler.setLevel(
        logging.DEBUG
    )

    handler.setFormatter(
        logging.Formatter(
            "%(message)s"
        )
    )

    file_handler = ScraperRunFileHandler(run)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(
        handler
    )
    logger.addHandler(file_handler)

    with bind_scraper_event_context(run):
        try:
            yield

        finally:
            logger.removeHandler(
                handler
            )
            logger.removeHandler(file_handler)

            handler.close()
            file_handler.close()
