from __future__ import annotations

from django.utils import timezone

from scraper.models import ScraperConfig, ScraperStatus
from scraper.service import run_scraper

from .models import ScraperRun


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

    try:
        result = run_scraper(config)

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

        return db_run

    except Exception as exc:
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

        raise
