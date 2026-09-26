from __future__ import annotations

import logging
from dataclasses import replace
from datetime import timedelta

from celery import shared_task
from django.db import transaction
from django.utils import timezone

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
)


logger = logging.getLogger(
    "captcha_lab"
)


@shared_task(
    name="operations.run_scraper",
)
def run_scraper_task(
    run_id: str,
    config_data: dict,
) -> str:
    """
    Execute a scraper run that has already been created.

    This is currently used by the manual Django Admin action.
    """

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
    update_proxy_health_from_run(
        db_run,
        cooldown_minutes=ScraperSchedule.load().interval_minutes,
    )

    return str(
        db_run.pk
    )


@shared_task(
    name="operations.run_scheduled_scraper",
)
def run_scheduled_scraper_task() -> str:
    """
    Celery Beat calls this task once per minute.

    This task decides whether the configured scraper interval
    has actually elapsed.

    It also prevents overlapping Playwright scraper runs.
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

        if (
            schedule.last_dispatched_at
            is not None
        ):
            next_run_at = (
                schedule.last_dispatched_at
                + timedelta(
                    minutes=(
                        schedule.interval_minutes
                    )
                )
            )

            if now < next_run_at:
                logger.debug(
                    "Scheduled scraper check: "
                    "not due yet. "
                    "Next run=%s",
                    next_run_at,
                )

                return "not_due"

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

        config = build_default_scraper_config()
        proxy_selection = select_proxy_pool(now)
        if not proxy_selection.proxy_urls:
            logger.warning(
                "Scheduled scraper circuit is open; no proxy endpoint is "
                "ready for a controlled recovery check."
            )
            return "circuit_open"

        config = replace(
            config,
            proxy_urls=proxy_selection.proxy_urls,
            allow_single_proxy=(len(proxy_selection.proxy_urls) == 1),
        )

        # Mark the dispatch before launching the scraper.
        #
        # This prevents two scheduler checks from launching
        # duplicate runs at approximately the same time.
        schedule.last_dispatched_at = now

        schedule.save(
            update_fields=[
                "last_dispatched_at",
                "updated_at",
            ]
        )

        db_run = create_scraper_run(
            config=config,
            trigger=(
                ScraperRun.Trigger.SCHEDULED
            ),
        )

    logger.info(
        "Scheduled scraper run created. Run ID=%s | Interval=%s minute(s) | "
        "Proxy endpoints=%s | Recovery probe=%s",
        db_run.pk,
        schedule.interval_minutes,
        len(proxy_selection.proxy_urls),
        proxy_selection.recovery_probe,
    )

    execute_scraper_run(
        config=config,
        trigger=(
            ScraperRun.Trigger.SCHEDULED
        ),
        db_run=db_run,
    )
    update_proxy_health_from_run(
        db_run,
        cooldown_minutes=schedule.interval_minutes,
    )

    return str(
        db_run.pk
    )
