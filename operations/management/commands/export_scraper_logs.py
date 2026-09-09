from __future__ import annotations

import csv
from pathlib import Path

from django.core.management.base import BaseCommand

from operations.models import ScraperRunLog


class Command(BaseCommand):
    help = "Export scraper run logs and run metadata to a CSV file."

    def add_arguments(self, parser):
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("scraper_run_logs.csv"),
            help="CSV destination (default: scraper_run_logs.csv).",
        )
        parser.add_argument(
            "--run-id",
            help="Export only the logs belonging to this ScraperRun UUID.",
        )

    def handle(self, *args, **options):
        output: Path = options["output"]
        run_id = options["run_id"]

        logs = ScraperRunLog.objects.select_related("run").order_by(
            "run__created_at",
            "id",
        )
        if run_id:
            logs = logs.filter(run_id=run_id)

        output.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "log_id",
            "log_created_at",
            "level",
            "message",
            "run_id",
            "run_status",
            "run_trigger",
            "run_created_at",
            "run_started_at",
            "run_finished_at",
            "run_duration_seconds",
            "appointment_visa_sub_type",
            "run_page_url",
            "error_type",
            "error_message",
            "failure_screenshot",
        ]

        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            count = 0
            for log in logs.iterator():
                run = log.run
                writer.writerow(
                    {
                        "log_id": log.id,
                        "log_created_at": log.created_at.isoformat(),
                        "level": log.level,
                        "message": log.message,
                        "run_id": run.id,
                        "run_status": run.status,
                        "run_trigger": run.trigger,
                        "run_created_at": run.created_at.isoformat(),
                        "run_started_at": (
                            run.started_at.isoformat()
                            if run.started_at
                            else ""
                        ),
                        "run_finished_at": (
                            run.finished_at.isoformat()
                            if run.finished_at
                            else ""
                        ),
                        "run_duration_seconds": (
                            run.duration_seconds
                            if run.duration_seconds is not None
                            else ""
                        ),
                        "appointment_visa_sub_type": run.appointment_visa_sub_type,
                        "run_page_url": run.page_url,
                        "error_type": run.error_type,
                        "error_message": run.error_message,
                        "failure_screenshot": run.failure_screenshot,
                    }
                )
                count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Exported {count} log rows to {output}"
            )
        )
