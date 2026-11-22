from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from django.utils import timezone
from django.db import close_old_connections, connections, transaction

from scraper.models import ScraperConfig, ScraperStatus
from datetime import timedelta

from .models import ScraperRun, ScraperSchedule
from .events import record_scraper_event
from .models import ScraperEvent
from .run_logging import bind_scraper_run_logging


logger = logging.getLogger("captcha_lab")

PENDING_STALE_AFTER = timedelta(
    minutes=10
)

RUNNING_STALE_AFTER = timedelta(
    minutes=60
)

STOP_REQUESTED_STALE_AFTER = timedelta(
    minutes=5
)

# A stop is normally cooperative. If Playwright never returns from a browser
# call, the solo Celery worker cannot run another task or observe that stop.
STOP_GRACE_SECONDS = 120


class ScraperRunAlreadyStarted(RuntimeError):
    """Raised when duplicate task delivery targets one execution row."""


ACTIVE_RUN_STATUSES = (
    ScraperRun.Status.PENDING,
    ScraperRun.Status.RUNNING,
    ScraperRun.Status.STOP_REQUESTED,
)


def active_scraper_runs():
    return ScraperRun.objects.filter(status__in=ACTIVE_RUN_STATUSES).order_by(
        "-created_at"
    )


def request_scraper_stop(run: ScraperRun) -> ScraperRun:
    """Request cooperative cancellation without killing the worker."""
    now = timezone.now()
    if run.status == ScraperRun.Status.PENDING:
        run.status = ScraperRun.Status.STOPPED
        run.stop_requested_at = now
        run.stopped_at = now
        run.finished_at = now
        run.save(
            update_fields=[
                "status",
                "stop_requested_at",
                "stopped_at",
                "finished_at",
            ]
        )
    elif run.status in {
        ScraperRun.Status.RUNNING,
        ScraperRun.Status.STOP_REQUESTED,
    }:
        run.status = ScraperRun.Status.STOP_REQUESTED
        run.stop_requested_at = run.stop_requested_at or now
        run.save(update_fields=["status", "stop_requested_at"])
    return run


@contextmanager
def monitor_stop_requests(run_id):
    """Poll for cancellation and bound a stuck solo worker after a stop."""
    stop_event = threading.Event()
    monitor_finished = threading.Event()

    def monitor() -> None:
        close_old_connections()
        try:
            while not monitor_finished.is_set():
                try:
                    requested = ScraperRun.objects.filter(
                        pk=run_id,
                        status=ScraperRun.Status.STOP_REQUESTED,
                    ).exists()
                except Exception:
                    logger.exception("Could not poll scraper stop request. Run ID=%s", run_id)
                    close_old_connections()
                    monitor_finished.wait(1)
                    continue
                if requested:
                    stop_event.set()
                    deadline = time.monotonic() + STOP_GRACE_SECONDS
                    while not monitor_finished.is_set():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        monitor_finished.wait(min(1, remaining))
                    if monitor_finished.is_set():
                        return
                    # The scrape is still blocked after an explicit operator
                    # request. The worker runs one task in a solo process; an
                    # immediate exit lets the supervisor restart it. The
                    # scheduled recovery task then finalizes the run row.
                    logger.critical(
                        "Scraper did not stop within %ss; exiting stuck worker. "
                        "Run ID=%s", STOP_GRACE_SECONDS, run_id,
                    )
                    os._exit(1)
                monitor_finished.wait(1)
        finally:
            connections.close_all()

    thread = threading.Thread(
        target=monitor,
        name=f"scraper-stop-monitor-{run_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield stop_event.is_set
    finally:
        monitor_finished.set()
        thread.join(timeout=2)


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
        "allow_single_proxy": config.allow_single_proxy,
        "direct_connection": config.direct_connection,
        "proxy_urls": list(config.proxy_urls) if config.proxy_urls is not None else None,
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
        allow_single_proxy=bool(config_data.get("allow_single_proxy", False)),
        direct_connection=bool(config_data.get("direct_connection", False)),
        proxy_urls=(
            tuple(config_data["proxy_urls"])
            if config_data.get("proxy_urls") is not None
            else None
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

    if (
        db_run.status != ScraperRun.Status.PENDING
        or db_run.started_at is not None
    ):
        raise ScraperRunAlreadyStarted(
            "ScraperRun has already started and cannot be executed again. "
            f"Run ID={db_run.pk} | Status={db_run.status}"
        )

    # Serialize the final admission decision across manual and scheduled workers.
    # A database error prevents admission rather than allowing an unsafe start.
    with transaction.atomic():
        schedule, _ = ScraperSchedule.objects.select_for_update().get_or_create(pk=1)
        db_run = ScraperRun.objects.select_for_update().get(pk=db_run.pk)
        if db_run.status != ScraperRun.Status.PENDING or db_run.started_at is not None:
            raise ScraperRunAlreadyStarted(f"Run already started: {db_run.pk}")
        now = timezone.now()
        other = ScraperRun.objects.exclude(pk=db_run.pk).filter(
            status__in=(ScraperRun.Status.RUNNING, ScraperRun.Status.STOP_REQUESTED),
        ).exists()
        last_finish = (ScraperRun.objects.exclude(pk=db_run.pk)
                       .filter(started_at__isnull=False, finished_at__isnull=False)
                       .order_by('-finished_at').values_list('finished_at', flat=True).first())
        until = max((v for v in (schedule.rate_limited_until if config.direct_connection else None,
                    last_finish + timedelta(minutes=20) if last_finish else None) if v), default=None)
        if other or (until and now < until):
            db_run.status = ScraperRun.Status.FAILED
            db_run.finished_at = now
            db_run.error_type = ('HTTP429Cooldown' if config.direct_connection and not other and schedule.rate_limited_until
                                 and now < schedule.rate_limited_until else 'RunAdmissionBlocked')
            db_run.error_message = 'Another run is active.' if other else f'Run blocked until {until.isoformat()}.'
            db_run.save(update_fields=['status', 'finished_at', 'error_type', 'error_message'])
            return db_run
        if not config.direct_connection:
            from dataclasses import replace
            from scraper.proxy import configured_proxy_urls
            from operations.proxy_health import select_proxy_pool
            candidates = config.proxy_urls if config.proxy_urls is not None else configured_proxy_urls()
            if candidates:
                selected = select_proxy_pool(now, proxy_urls=candidates)
                if len(selected.proxy_urls) < 2:
                    db_run.status = ScraperRun.Status.FAILED
                    db_run.finished_at = now
                    db_run.error_type = "ProxyPoolUnavailable"
                    db_run.error_message = "At least two eligible proxy endpoints are required to start a run."
                    db_run.save(update_fields=["status", "finished_at", "error_type", "error_message"])
                    return db_run
                config = replace(config, proxy_urls=selected.proxy_urls, allow_single_proxy=False)
        db_run.status = ScraperRun.Status.RUNNING
        db_run.started_at = db_run.heartbeat_at = now
        db_run.finished_at = None
        db_run.error_type = db_run.error_message = ''
        db_run.first_failure = db_run.terminal_failure = None
        db_run.attempt_failures = []
        db_run.stop_requested_at = db_run.stopped_at = None
        db_run.save(update_fields=['status', 'started_at', 'heartbeat_at', 'finished_at',
            'error_type', 'error_message', 'first_failure', 'attempt_failures',
            'terminal_failure', 'stop_requested_at', 'stopped_at'])

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

            record_scraper_event(
                ScraperEvent.EventType.RUN_STARTED,
                status=db_run.status,
                data={
                    "trigger": trigger,
                    "headless": config.headless,
                    "gpu": config.gpu,
                    "visa_sub_types": list(config.visa_sub_types),
                },
            )

            with monitor_stop_requests(db_run.pk) as should_stop:
                with (
                    redirect_stdout(
                        stdout_writer
                    ),
                    redirect_stderr(
                        stderr_writer
                    ),
                ):
                    # Only the scraper worker needs browser and OCR imports.
                    # Keep the schedule controller small enough to share Beat's pod.
                    from scraper.service import run_scraper

                    result = run_scraper(config, should_stop=should_stop)

            stdout_writer.flush()
            stderr_writer.flush()

            logger.info(
                "Scraper execution returned status=%s",
                result.status.value,
            )

            status_map = {
                ScraperStatus.APPOINTMENT_FOUND:
                    ScraperRun.Status.APPOINTMENT_FOUND,
                ScraperStatus.POSSIBLE_APPOINTMENT:
                    ScraperRun.Status.POSSIBLE_APPOINTMENT,
                ScraperStatus.NO_APPOINTMENT:
                    ScraperRun.Status.NO_APPOINTMENT,
                ScraperStatus.SERVER_ERROR:
                    ScraperRun.Status.SERVER_ERROR,
                ScraperStatus.STOPPED:
                    ScraperRun.Status.STOPPED,
                ScraperStatus.FAILED:
                    ScraperRun.Status.FAILED,
            }

            db_run.status = status_map[
                result.status
            ]

            db_run.finished_at = (
                timezone.now()
            )
            if result.status is ScraperStatus.STOPPED:
                db_run.stopped_at = db_run.finished_at

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
            db_run.first_failure = result.first_failure
            db_run.attempt_failures = list(result.attempt_failures)
            db_run.terminal_failure = result.terminal_failure

            db_run.duration_seconds = (
                result.duration_seconds
            )

            db_run.save(
                update_fields=[
                    "status",
                    "finished_at",
                    "stopped_at",
                    "page_url",
                    "appointment_visa_sub_type",
                    "error_type",
                    "error_message",
                    "failure_screenshot",
                    "first_failure",
                    "attempt_failures",
                    "terminal_failure",
                    "duration_seconds",
                ]
            )

            logger.info(
                "Scraper run finished. "
                "Status=%s | Duration=%.2fs",
                db_run.status,
                result.duration_seconds,
            )

            if (
                db_run.status in (
                    ScraperRun.Status.APPOINTMENT_FOUND,
                    ScraperRun.Status.POSSIBLE_APPOINTMENT,
                )
                and result.visa_sub_type
            ):
                record_scraper_event(
                    ScraperEvent.EventType.APPOINTMENT_DETECTED,
                    visa_sub_type=result.visa_sub_type,
                    status=db_run.status,
                    duration_ms=round(
                        result.duration_seconds * 1000
                    ),
                    data={
                        "page_url": result.page_url or "",
                        "certainty": (
                            "confirmed"
                            if db_run.status == ScraperRun.Status.APPOINTMENT_FOUND
                            else "possible"
                        ),
                    },
                )

                logger.info(
                    "Appointment availability status=%s "
                    "for visa subtype=%s",
                    db_run.status,
                    result.visa_sub_type,
                )

            record_scraper_event(
                ScraperEvent.EventType.RUN_FINISHED,
                status=db_run.status,
                duration_ms=round(
                    result.duration_seconds * 1000
                ),
                data={
                    "page_url": result.page_url or "",
                    "error_type": result.error_type or "",
                },
            )

            if (
                result.error_message
                and result.status is not ScraperStatus.STOPPED
            ):
                logger.error(
                    "Scraper reported failure. %s: %s",
                    result.error_type,
                    result.error_message,
                )

            return db_run

        except Exception as exc:
            stdout_writer.flush()
            stderr_writer.flush()

            from flows.errors import RateLimitPersistenceError
            if isinstance(exc, RateLimitPersistenceError):
                # The durable RUNNING/STOP_REQUESTED row is the safety latch.
                # Do not finalize it when a server-requested wait may be lost.
                logger.critical("429 cooldown could not be persisted; run remains blocked for operator recovery.")
                raise

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

            record_scraper_event(
                ScraperEvent.EventType.RUN_FAILED,
                status=db_run.status,
                reason_code=type(exc).__name__.upper(),
                duration_ms=round(
                    db_run.duration_seconds * 1000
                )
                if db_run.duration_seconds is not None
                else None,
                message=str(exc),
            )

            raise


def recover_stale_scraper_runs() -> int:
    """
    Recover abandoned PENDING jobs.

    This handles cases such as:
    Running and stop-requested jobs stay active until an operator confirms
    their worker has stopped; a stale heartbeat alone cannot establish that.

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

    # A missed heartbeat cannot prove the browser stopped. Keep RUNNING and
    # STOP_REQUESTED rows active until an operator confirms termination.

    return recovered
