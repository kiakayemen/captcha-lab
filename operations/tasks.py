from __future__ import annotations

import logging

from celery import shared_task

from .models import ScraperRun
from .services import (
    build_default_scraper_config,
    create_scraper_run,
    deserialize_scraper_config,
    execute_scraper_run,
)


logger = logging.getLogger("captcha_lab")


@shared_task(
    name="operations.run_scraper",
)
def run_scraper_task(
    run_id: str,
    config_data: dict,
) -> str:
    db_run = ScraperRun.objects.get(
        pk=run_id,
    )

    config = deserialize_scraper_config(
        config_data
    )

    execute_scraper_run(
        config=config,
        trigger=db_run.trigger,
        db_run=db_run,
    )

    return str(db_run.pk)


@shared_task(
    name="operations.run_scheduled_scraper",
)
def run_scheduled_scraper_task() -> str:
    """
    Entry point used by Celery Beat.

    A scheduled run is skipped if another scraper run is already
    pending or running. This prevents two Playwright instances from
    sharing the same output directory and BLS session workflow.
    """

    active_run = (
        ScraperRun.objects
        .filter(
            status__in=[
                ScraperRun.Status.PENDING,
                ScraperRun.Status.RUNNING,
            ]
        )
        .order_by("-created_at")
        .first()
    )

    if active_run is not None:
        logger.warning(
            "Scheduled scraper skipped because "
            "another run is active. "
            "Run ID=%s | Status=%s",
            active_run.pk,
            active_run.status,
        )

        return (
            "skipped:"
            f"{active_run.pk}"
        )

    config = build_default_scraper_config()

    db_run = create_scraper_run(
        config=config,
        trigger=(
            ScraperRun.Trigger.SCHEDULED
        ),
    )

    logger.info(
        "Scheduled scraper run created. "
        "Run ID=%s",
        db_run.pk,
    )

    execute_scraper_run(
        config=config,
        trigger=(
            ScraperRun.Trigger.SCHEDULED
        ),
        db_run=db_run,
    )

    return str(db_run.pk)
