from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from operations.services import execute_scraper_run
from scraper.models import ScraperConfig


class Command(BaseCommand):
    help = "Run the scraper and persist the result in ScraperRun."

    def add_arguments(self, parser):
        parser.add_argument(
            "--headless",
            action="store_true",
            help="Run without displaying the browser.",
        )

        parser.add_argument(
            "--gpu",
            action="store_true",
            help="Ask EasyOCR to use a supported GPU.",
        )

        parser.add_argument(
            "--output",
            type=Path,
            default=Path("output/live_solver"),
            help="Solver output directory.",
        )

        parser.add_argument(
            "--visa-sub-types",
            nargs="+",
            choices=[
                "Student Visa",
                "Non-Working Residence Visa",
            ],
            default=[
                "Student Visa",
                "Non-Working Residence Visa",
            ],
        )

    def handle(self, *args, **options):
        config = ScraperConfig(
            headless=options["headless"],
            gpu=options["gpu"],
            output_dir=options["output"],
            visa_sub_types=tuple(
                options["visa_sub_types"]
            ),
        )

        self.stdout.write(
            self.style.WARNING("Starting scraper run...")
        )

        db_run = execute_scraper_run(
            config=config,
        )

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Run finished: {db_run.id}"
            )
        )

        self.stdout.write(
            f"Status: {db_run.status}"
        )

        if db_run.duration_seconds is not None:
            self.stdout.write(
                f"Duration: "
                f"{db_run.duration_seconds:.2f}s"
            )

        if db_run.appointment_visa_sub_type:
            self.stdout.write(
                "Appointment visa subtype: "
                f"{db_run.appointment_visa_sub_type}"
            )

        if db_run.error_message:
            self.stdout.write(
                self.style.ERROR(
                    f"{db_run.error_type}: "
                    f"{db_run.error_message}"
                )
            )
