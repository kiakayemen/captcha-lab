from __future__ import annotations

from pathlib import Path

from celery import shared_task

from scraper.models import ScraperConfig

from .models import ScraperRun
from .services import execute_scraper_run


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

    config = ScraperConfig(
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

    execute_scraper_run(
        config=config,
        trigger=db_run.trigger,
        db_run=db_run,
    )

    return str(db_run.pk)
