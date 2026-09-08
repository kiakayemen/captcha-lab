from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from django.conf import settings
from django.utils import timezone

from .models import (
    ScraperRun,
    ScraperRunLog,
)
from .events import (
    bind_scraper_event_context,
    record_event_from_log,
)


_current_run_id: ContextVar[
    str | None
] = ContextVar(
    "scraper_run_id",
    default=None,
)


class ScraperRunDatabaseHandler(
    logging.Handler
):
    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        run_id = (
            _current_run_id.get()
        )

        if run_id is None:
            return

        try:
            message = (
                self.format(
                    record
                )
            )

            ScraperRunLog.objects.create(
                run_id=run_id,
                level=record.levelname,
                message=message,
            )

            record_event_from_log(message)

            #
            # Every meaningful scraper log acts as a heartbeat.
            #
            # If the worker/process disappears completely,
            # this timestamp stops advancing.
            #
            ScraperRun.objects.filter(
                pk=run_id,
                status=(
                    ScraperRun
                    .Status
                    .RUNNING
                ),
            ).update(
                heartbeat_at=(
                    timezone.now()
                )
            )

        except Exception:
            #
            # Logging must never be capable
            # of crashing the scraper.
            #
            pass


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
        ScraperRunDatabaseHandler()
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
        token = (
            _current_run_id.set(
                str(
                    run.pk
                )
            )
        )

        try:
            yield

        finally:
            logger.removeHandler(
                handler
            )
            logger.removeHandler(file_handler)

            handler.close()
            file_handler.close()

            _current_run_id.reset(
                token
            )
