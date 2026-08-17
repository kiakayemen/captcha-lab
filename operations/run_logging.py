from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator

from .models import ScraperRun, ScraperRunLog


_current_run_id: ContextVar[str | None] = ContextVar(
    "scraper_run_id",
    default=None,
)


class ScraperRunDatabaseHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        run_id = _current_run_id.get()

        if run_id is None:
            return

        try:
            message = self.format(record)

            ScraperRunLog.objects.create(
                run_id=run_id,
                level=record.levelname,
                message=message,
            )

        except Exception:
            # Logging must never be capable of crashing the scraper.
            pass


@contextmanager
def bind_scraper_run_logging(
    run: ScraperRun,
) -> Iterator[None]:
    logger = logging.getLogger("captcha_lab")

    token = _current_run_id.set(str(run.pk))

    handler = ScraperRunDatabaseHandler()
    handler.setLevel(logging.DEBUG)

    handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    logger.addHandler(handler)

    try:
        yield

    finally:
        logger.removeHandler(handler)
        handler.close()

        _current_run_id.reset(token)
