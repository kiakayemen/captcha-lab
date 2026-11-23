from __future__ import annotations

import logging
from dataclasses import replace
from datetime import timedelta
from zoneinfo import ZoneInfo

from celery import shared_task
from django.db import transaction
from django.utils import timezone

from notifications import notify_consecutive_403
from .models import (
    ScraperRun,
    ScraperSchedule,
)
from .proxy_health import select_proxy_pool, update_proxy_health_from_run
from .services import (
    ACTIVE_RUN_STATUSES,
    build_default_scraper_config,
    create_scraper_run,
    deserialize_scraper_config,
    execute_scraper_run,
    recover_stale_scraper_runs,
    request_scraper_stop,
    serialize_scraper_config,
)


logger = logging.getLogger(
    "captcha_lab"
)

CONSECUTIVE_403_LIMIT = 10
TEHRAN = ZoneInfo("Asia/Tehran")


def schedule_phase(now, *, back_to_back_enabled=True):
    """Select the daily schedule using Tehran wall-clock time."""
    if not back_to_back_enabled:
        return "interval"
    local = timezone.localtime(now, TEHRAN)
    minute_of_day = local.hour * 60 + local.minute
    if 9 * 60 + 50 <= minute_of_day < 10 * 60:
        return "prepare"
    if 10 * 60 <= minute_of_day < 11 * 60:
        return "back_to_back"
    return "interval"


def update_scheduled_403_streak(run: ScraperRun) -> int:
    """Persist the scheduled-run 403 streak and trip the safety stop at 10."""
    with transaction.atomic():
        schedule = ScraperSchedule.objects.select_for_update().get(pk=1)
        if run.error_type == "HTTP403Forbidden":
            schedule.consecutive_403_runs += 1
        else:
            schedule.consecutive_403_runs = 0

        threshold_reached = (
            schedule.consecutive_403_runs >= CONSECUTIVE_403_LIMIT
            and schedule.enabled
        )
        if threshold_reached:
            schedule.enabled = False

        schedule.save(
            update_fields=["consecutive_403_runs", "enabled", "updated_at"]
        )
        count = schedule.consecutive_403_runs

        if threshold_reached:
            transaction.on_commit(
                lambda: notify_consecutive_403(run_id=str(run.pk), count=count)
            )

    return count


@shared_task(
    name="operations.run_scraper",
)
def run_scraper_task(
    run_id: str,
    config_data: dict,
) -> str:
    """Execute one queued manual, interval, or back-to-back scraper run."""

    db_run = ScraperRun.objects.get(
        pk=run_id,
    )
    if db_run.status in {
        ScraperRun.Status.STOPPED,
        ScraperRun.Status.STOP_REQUESTED,
    }:
        if db_run.status == ScraperRun.Status.STOP_REQUESTED:
            now = timezone.now()
            db_run.status = ScraperRun.Status.STOPPED
            db_run.stopped_at = now
            db_run.finished_at = now
            db_run.save(update_fields=["status", "stopped_at", "finished_at"])
        return str(db_run.pk)

    config = deserialize_scraper_config(
        config_data
    )

    execute_scraper_run(
        config=config,
        trigger=db_run.trigger,
        db_run=db_run,
    )
    db_run.refresh_from_db()
    if db_run.started_at and db_run.status in {ScraperRun.Status.NO_APPOINTMENT, ScraperRun.Status.APPOINTMENT_FOUND,
                         ScraperRun.Status.POSSIBLE_APPOINTMENT}:
        ScraperSchedule.objects.filter(pk=1, rate_limit_last_seen_at__lte=db_run.started_at).update(
            rate_limit_streak=0
        )
    if db_run.trigger == ScraperRun.Trigger.BACK_TO_BACK:
        # Save the finish before optional health checks so the next regular
        # interval stays correct even if a post-run check fails.
        ScraperSchedule.objects.filter(pk=1).update(
            last_back_to_back_finished_at=db_run.finished_at,
        )
    update_proxy_health_from_run(db_run)

    if db_run.trigger in {
        ScraperRun.Trigger.SCHEDULED,
        ScraperRun.Trigger.BACK_TO_BACK,
    }:
        db_run.refresh_from_db(fields=["error_type", "finished_at"])
        update_scheduled_403_streak(db_run)

    if db_run.trigger == ScraperRun.Trigger.BACK_TO_BACK:
        run_scheduled_scraper_task.delay()

    return str(
        db_run.pk
    )


@shared_task(
    name="operations.run_scheduled_scraper",
)
def run_scheduled_scraper_task() -> str:
    """
    Run on the separate control worker so a busy browser cannot block the
    clock or a stop request. The scraper itself runs on the scraper queue.
    """

    recovered = (
        recover_stale_scraper_runs()
    )

    if recovered:
        logger.warning(
            "Recovered %s stale scraper "
            "run(s) before scheduled check.",
            recovered,
        )

    now = timezone.now()

    with transaction.atomic():
        schedule, _created = (
            ScraperSchedule.objects
            .select_for_update()
            .get_or_create(
                pk=1,
                defaults={
                    "enabled": True,
                    "interval_minutes": 30,
                },
            )
        )

        if not schedule.enabled:
            logger.debug(
                "Scheduled scraper check: disabled."
            )

            return "disabled"

        phase = schedule_phase(
            now, back_to_back_enabled=schedule.back_to_back_enabled
        )
        active_run = (
            ScraperRun.objects
            .filter(
                status__in=ACTIVE_RUN_STATUSES
            )
            .order_by(
                "-created_at"
            )
            .first()
        )

        if phase == "prepare":
            if active_run is not None and active_run.trigger != ScraperRun.Trigger.BACK_TO_BACK:
                request_scraper_stop(active_run)
                logger.warning(
                    "Requested stop before 10:00 back-to-back window. Run ID=%s",
                    active_run.pk,
                )
            return "preparing"

        if active_run is not None:
            logger.info(
                "Scheduled scraper is due, "
                "but another scraper run is active. "
                "Run ID=%s | Status=%s",
                active_run.pk,
                active_run.status,
            )

            return (
                "active:"
                f"{active_run.pk}"
            )

        last_finish = (ScraperRun.objects.filter(started_at__isnull=False, finished_at__isnull=False)
                       .order_by('-finished_at').values_list('finished_at', flat=True).first())
        if last_finish and now < last_finish + timedelta(minutes=20):
            return 'minimum_run_interval'

        if phase == "back_to_back" and not schedule.rate_limit_retry_pending:
            last_back_to_back = (
                ScraperRun.objects.filter(
                    trigger=ScraperRun.Trigger.BACK_TO_BACK,
                    finished_at__isnull=False,
                )
                .order_by("-finished_at")
                .first()
            )
            if (
                last_back_to_back is not None
                and last_back_to_back.status in {
                    ScraperRun.Status.FAILED,
                    ScraperRun.Status.SERVER_ERROR,
                    ScraperRun.Status.STOPPED,
                }
                and now < last_back_to_back.finished_at + timedelta(
                    minutes=schedule.interval_minutes
                )
            ):
                logger.warning(
                    "Back-to-back scraper paused after unsuccessful run. "
                    "Run ID=%s | Cooldown=%s minute(s)",
                    last_back_to_back.pk,
                    schedule.interval_minutes,
                )
                return "back_to_back_cooldown"

        if phase == "interval":
            anchor = max(
                (value for value in (
                    schedule.last_dispatched_at,
                    schedule.last_back_to_back_finished_at,
                ) if value is not None),
                default=None,
            )
            if anchor is not None and now < anchor + timedelta(
                minutes=schedule.interval_minutes
            ):
                return "not_due"

        config = build_default_scraper_config()
        proxy_selection = select_proxy_pool(now)
        if len(proxy_selection.proxy_urls) < 2:
            logger.warning(
                "Scheduled scraper circuit is open; no proxy endpoint is "
                "ready; at least two are required."
            )
            return "circuit_open"

        config = replace(
            config,
            proxy_urls=proxy_selection.proxy_urls,
            allow_single_proxy=False,
        )

        # Mark the dispatch before launching the scraper.
        #
        # This prevents two scheduler checks from launching
        # duplicate runs at approximately the same time.
        if phase == "interval":
            schedule.last_dispatched_at = now
        schedule.rate_limit_retry_pending = False
        schedule.save(update_fields=["last_dispatched_at", "rate_limit_retry_pending", "updated_at"])

        db_run = create_scraper_run(
            config=config,
            trigger=(
                ScraperRun.Trigger.BACK_TO_BACK
                if phase == "back_to_back"
                else ScraperRun.Trigger.SCHEDULED
            ),
        )

        payload = serialize_scraper_config(config)
        transaction.on_commit(
            lambda: run_scraper_task.delay(str(db_run.pk), payload)
        )

    logger.info(
        "Scheduled scraper run created. Run ID=%s | Interval=%s minute(s) | "
        "Proxy endpoints=%s | Recovery probe=%s",
        db_run.pk,
        schedule.interval_minutes,
        len(proxy_selection.proxy_urls),
        proxy_selection.recovery_probe,
    )

    return str(
        db_run.pk
    )
