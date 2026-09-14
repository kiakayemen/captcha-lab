from django.core.management.base import BaseCommand

from operations.services import active_scraper_runs, recover_stale_scraper_runs


class Command(BaseCommand):
    help = "Show pending, running, and stop-requested scraper runs."

    def handle(self, *args, **options):
        recover_stale_scraper_runs()
        runs = list(active_scraper_runs())
        if not runs:
            self.stdout.write("No scraper run is active.")
            return
        for run in runs:
            self.stdout.write(
                f"{run.pk} status={run.status} trigger={run.trigger} "
                f"started_at={run.started_at or '-'} "
                f"heartbeat_at={run.heartbeat_at or '-'}"
            )
