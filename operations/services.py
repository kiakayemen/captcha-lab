from __future__ import annotations

import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from django.utils import timezone
import os

from scraper.models import ScraperConfig, ScraperStatus
from scraper.service import run_scraper
from datetime import timedelta

from .models import ScraperRun
from .run_logging import bind_scraper_run_logging


logger = logging.getLogger("captcha_lab")

PENDING_STALE_AFTER = timedelta(
    minutes=10
)

RUNNING_STALE_AFTER = timedelta(
    minutes=60
)


def _env_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(
        name,
        str(default),
    )

    return (
        value.strip().lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )

class _LoggerWriter:
    """
    Redirect print()/stdout/stderr output into the scraper logger.

    This preserves the current logging bridge while Celery moves
    execution out of the Django HTTP request.
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
            line, self._buffer = self._buffer.split(
                "\n",
                1,
            )
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


def build_default_scraper_config() -> ScraperConfig:
    output_dir = Path(
        os.getenv(
            "SCRAPER_OUTPUT_DIR",
            "output/live_solver",
        )
    )

    return ScraperConfig(
        headless=_env_bool(
            "SCRAPER_HEADLESS",
            True,
        ),
        gpu=_env_bool(
            "SCRAPER_GPU",
            False,
        ),
        output_dir=output_dir,
        visa_sub_types=(
            "Student Visa",
            "Non-Working Residence Visa",
        ),
    )

def serialize_scraper_config(
    config: ScraperConfig,
) -> dict:
    """
    Convert ScraperConfig into JSON-safe data suitable for Celery.
    """
    return {
        "headless": config.headless,
        "gpu": config.gpu,
        "output_dir": str(
            config.output_dir
        ),
        "visa_sub_types": list(
            config.visa_sub_types
        ),
    }


def deserialize_scraper_config(
    config_data: dict,
) -> ScraperConfig:
    """
    Rebuild ScraperConfig from the JSON-safe Celery payload.
    """
    return ScraperConfig(
        headless=bool(
            config_data["headless"]
        ),
        gpu=bool(
            config_data["gpu"]
        ),
        output_dir=Path(
            config_data["output_dir"]
        ),
        visa_sub_types=tuple(
            config_data["visa_sub_types"]
        ),
    )


def create_scraper_run(
    *,
    config: ScraperConfig,
    trigger: str = ScraperRun.Trigger.MANUAL,
) -> ScraperRun:
    return ScraperRun.objects.create(
        status=ScraperRun.Status.PENDING,
        trigger=trigger,
        visa_sub_types=list(
            config.visa_sub_types
        ),
    )


def execute_scraper_run(
    *,
    config: ScraperConfig,
    trigger: str = ScraperRun.Trigger.MANUAL,
    db_run: ScraperRun | None = None,
) -> ScraperRun:
    if db_run is None:
        db_run = create_scraper_run(
            config=config,
            trigger=trigger,
        )

    now = timezone.now()


    db_run.status = (
        ScraperRun.Status.RUNNING
    )

    db_run.started_at = now

    db_run.heartbeat_at = now

    db_run.finished_at = None

    db_run.error_type = ""

    db_run.error_message = ""

    db_run.save(
        update_fields=[
            "status",
            "started_at",
            "heartbeat_at",
            "finished_at",
            "error_type",
            "error_message",
        ]
    )

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
                ", ".join(
                    config.visa_sub_types
                ),
            )

            with (
                redirect_stdout(
                    stdout_writer
                ),
                redirect_stderr(
                    stderr_writer
                ),
            ):
                result = run_scraper(
                    config
                )

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

            db_run.status = status_map[
                result.status
            ]

            db_run.finished_at = (
                timezone.now()
            )

            db_run.page_url = (
                result.page_url or ""
            )

            db_run.appointment_visa_sub_type = (
                result.visa_sub_type or ""
            )

            db_run.error_type = (
                result.error_type or ""
            )

            db_run.error_message = (
                result.error_message or ""
            )

            db_run.failure_screenshot = (
                str(
                    result.failure_screenshot
                )
                if result.failure_screenshot
                else ""
            )

            db_run.duration_seconds = (
                result.duration_seconds
            )

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
                "Scraper run finished. "
                "Status=%s | Duration=%.2fs",
                db_run.status,
                result.duration_seconds,
            )

            if result.visa_sub_type:
                logger.info(
                    "Appointment availability detected "
                    "for visa subtype=%s",
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

            db_run.status = (
                ScraperRun.Status.FAILED
            )

            db_run.finished_at = (
                timezone.now()
            )

            db_run.error_type = (
                type(exc).__name__
            )

            db_run.error_message = (
                str(exc)
            )

            if db_run.started_at is not None:
                db_run.duration_seconds = (
                    db_run.finished_at
                    - db_run.started_at
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


def recover_stale_scraper_runs() -> int:
    """
    Mark abandoned PENDING/RUNNING jobs as FAILED.

    This handles cases such as:
    - Celery worker dies;
    - Python/Metal process crashes;
    - machine restarts;
    - a queued task is never picked up.

    Returns the number of runs recovered.
    """

    now = timezone.now()

    recovered = 0

    #
    # PENDING:
    # created but apparently never picked up.
    #
    pending_cutoff = (
        now
        - PENDING_STALE_AFTER
    )

    stale_pending = (
        ScraperRun.objects
        .filter(
            status=(
                ScraperRun
                .Status
                .PENDING
            ),
            created_at__lt=(
                pending_cutoff
            ),
        )
    )

    for run in stale_pending:
        run.status = (
            ScraperRun.Status.FAILED
        )

        run.finished_at = now

        run.error_type = (
            "StalePendingRun"
        )

        run.error_message = (
            "The scraper run remained "
            "pending for more than "
            f"{PENDING_STALE_AFTER} "
            "and was automatically "
            "marked failed."
        )

        run.save(
            update_fields=[
                "status",
                "finished_at",
                "error_type",
                "error_message",
            ]
        )

        recovered += 1

        logger.warning(
            "Recovered stale PENDING "
            "scraper run. Run ID=%s",
            run.pk,
        )

    #
    # RUNNING:
    # use heartbeat first, then started_at
    # as the fallback for older rows.
    #
    running_cutoff = (
        now
        - RUNNING_STALE_AFTER
    )

    running_runs = (
        ScraperRun.objects
        .filter(
            status=(
                ScraperRun
                .Status
                .RUNNING
            )
        )
    )

    for run in running_runs:
        last_alive_at = (
            run.heartbeat_at
            or run.started_at
        )

        if last_alive_at is None:
            continue

        if (
            last_alive_at
            >= running_cutoff
        ):
            continue

        run.status = (
            ScraperRun.Status.FAILED
        )

        run.finished_at = now

        run.error_type = (
            "StaleRunningRun"
        )

        run.error_message = (
            "The scraper stopped "
            "producing heartbeats for "
            f"more than "
            f"{RUNNING_STALE_AFTER}. "
            "The Celery worker or scraper "
            "process may have terminated."
        )

        if run.started_at:
            run.duration_seconds = (
                now
                - run.started_at
            ).total_seconds()

        run.save(
            update_fields=[
                "status",
                "finished_at",
                "error_type",
                "error_message",
                "duration_seconds",
            ]
        )

        recovered += 1

        logger.error(
            "Recovered stale RUNNING "
            "scraper run. "
            "Run ID=%s | "
            "Last heartbeat=%s",
            run.pk,
            last_alive_at,
        )

    return recovered
