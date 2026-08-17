from __future__ import annotations

import logging
from contextlib import redirect_stderr, redirect_stdout

from django.utils import timezone

from scraper.models import ScraperConfig, ScraperStatus
from scraper.service import run_scraper

from .models import ScraperRun
from .run_logging import bind_scraper_run_logging


logger = logging.getLogger("captcha_lab")


class _LoggerWriter:
    """
    Redirect print()/stdout/stderr output into the scraper logger.

    This lets old print statements in nested scraper modules appear
    in the persistent ScraperRun log without rewriting all of them yet.
    """

    def __init__(
        self,
        target_logger: logging.Logger,
        level: int,
    ) -> None:
        self.logger = target_logger
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        if not message:
            return 0

        self._buffer += message

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)

        return len(message)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""

    def _emit(self, line: str) -> None:
        line = line.rstrip("\r")

        if not line.strip():
            return

        self.logger.log(
            self.level,
            "%s",
            line,
        )


def execute_scraper_run(
    *,
    config: ScraperConfig,
    trigger: str = ScraperRun.Trigger.MANUAL,
) -> ScraperRun:
    db_run = ScraperRun.objects.create(
        status=ScraperRun.Status.PENDING,
        trigger=trigger,
        visa_sub_types=list(config.visa_sub_types),
    )

    db_run.status = ScraperRun.Status.RUNNING
    db_run.started_at = timezone.now()

    db_run.save(
        update_fields=[
            "status",
            "started_at",
        ]
    )

    stdout_writer = _LoggerWriter(
        logger,
        logging.INFO,
    )

    stderr_writer = _LoggerWriter(
        logger,
        logging.ERROR,
    )

    with bind_scraper_run_logging(db_run):
        try:
            logger.info(
                "Scraper run started. "
                "Trigger=%s | Headless=%s | GPU=%s | Visa subtypes=%s",
                trigger,
                config.headless,
                config.gpu,
                ", ".join(config.visa_sub_types),
            )

            with (
                redirect_stdout(stdout_writer),
                redirect_stderr(stderr_writer),
            ):
                result = run_scraper(config)

            stdout_writer.flush()
            stderr_writer.flush()

            logger.info(
                "Scraper execution returned status=%s",
                result.status.value,
            )

            status_map = {
                ScraperStatus.APPOINTMENT_FOUND:
                    ScraperRun.Status.APPOINTMENT_FOUND,
                ScraperStatus.NO_APPOINTMENT:
                    ScraperRun.Status.NO_APPOINTMENT,
                ScraperStatus.FAILED:
                    ScraperRun.Status.FAILED,
            }

            db_run.status = status_map[result.status]
            db_run.finished_at = timezone.now()
            db_run.page_url = result.page_url or ""
            db_run.appointment_visa_sub_type = (
                result.visa_sub_type or ""
            )
            db_run.error_type = result.error_type or ""
            db_run.error_message = result.error_message or ""

            db_run.failure_screenshot = (
                str(result.failure_screenshot)
                if result.failure_screenshot
                else ""
            )

            db_run.duration_seconds = result.duration_seconds

            db_run.save(
                update_fields=[
                    "status",
                    "finished_at",
                    "page_url",
                    "appointment_visa_sub_type",
                    "error_type",
                    "error_message",
                    "failure_screenshot",
                    "duration_seconds",
                ]
            )

            logger.info(
                "Scraper run finished. Status=%s | Duration=%.2fs",
                db_run.status,
                result.duration_seconds,
            )

            if result.visa_sub_type:
                logger.info(
                    "Appointment availability detected for visa subtype=%s",
                    result.visa_sub_type,
                )

            if result.error_message:
                logger.error(
                    "Scraper reported failure. %s: %s",
                    result.error_type,
                    result.error_message,
                )

            return db_run

        except Exception as exc:
            stdout_writer.flush()
            stderr_writer.flush()

            db_run.status = ScraperRun.Status.FAILED
            db_run.finished_at = timezone.now()
            db_run.error_type = type(exc).__name__
            db_run.error_message = str(exc)

            if db_run.started_at is not None:
                db_run.duration_seconds = (
                    db_run.finished_at - db_run.started_at
                ).total_seconds()

            db_run.save(
                update_fields=[
                    "status",
                    "finished_at",
                    "error_type",
                    "error_message",
                    "duration_seconds",
                ]
            )

            logger.exception(
                "Scraper run crashed unexpectedly."
            )

            raise
