from django.core.management.base import BaseCommand, CommandError

from operations.models import ScraperRun
from operations.services import active_scraper_runs, request_scraper_stop


class Command(BaseCommand):
    help = "Safely stop one active scraper run, or all active runs."

    def add_arguments(self, parser):
        parser.add_argument("run_id", nargs="?")
        parser.add_argument("--all", action="store_true", dest="stop_all")

    def handle(self, *args, **options):
        if options["stop_all"]:
            runs = list(active_scraper_runs())
        elif options["run_id"]:
            try:
                run = ScraperRun.objects.get(pk=options["run_id"])
            except (ScraperRun.DoesNotExist, ValueError) as error:
                raise CommandError("Scraper run was not found.") from error
            runs = [run]
        else:
            runs = list(active_scraper_runs())
            if len(runs) > 1:
                raise CommandError("Multiple runs are active; pass a run ID or --all.")

        if not runs:
            self.stdout.write("No scraper run is active.")
            return

        for run in runs:
            request_scraper_stop(run)
            self.stdout.write(self.style.WARNING(f"Stop requested: {run.pk}"))
