"""Summarize recent 403 responses by proxy route and target endpoint."""

import json
from collections import defaultdict
from datetime import timedelta
from urllib.parse import urlsplit

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from operations.models import ScraperEvent


class Command(BaseCommand):
    help = "Show recent HTTP 403 counts by proxy endpoint and BLS request path."

    def add_arguments(self, parser):
        parser.add_argument("--hours", type=int, default=24)
        parser.add_argument("--json", action="store_true", dest="as_json")

    def handle(self, *args, **options):
        hours = options["hours"]
        if hours < 1:
            raise CommandError("--hours must be at least 1")
        since = timezone.now() - timedelta(hours=hours)
        grouped = defaultdict(lambda: {"count": 0, "runs": set()})
        events = ScraperEvent.objects.filter(
            event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
            status="403",
            created_at__gte=since,
        ).values("run_id", "data")
        for event in events.iterator():
            data = event["data"] or {}
            proxy = data.get("proxy_endpoint") or "unknown proxy"
            url = data.get("response_url") or data.get("response_request_url") or ""
            path = urlsplit(url).path.lower() or "unknown path"
            bucket = grouped[(proxy, path)]
            bucket["count"] += 1
            bucket["runs"].add(str(event["run_id"]))

        rows = [
            {"proxy_endpoint": proxy, "request_path": path,
             "responses": values["count"], "runs": len(values["runs"])}
            for (proxy, path), values in grouped.items()
        ]
        rows.sort(key=lambda row: (-row["responses"], row["proxy_endpoint"], row["request_path"]))
        if options["as_json"]:
            self.stdout.write(json.dumps({"hours": hours, "rows": rows}))
            return
        self.stdout.write(f"HTTP 403 responses in the last {hours} hours:")
        if not rows:
            self.stdout.write("None recorded.")
            return
        for row in rows:
            self.stdout.write(
                f"{row['responses']:>4} responses / {row['runs']:>3} runs  "
                f"{row['proxy_endpoint']}  {row['request_path']}"
            )
        self.stdout.write(
            "IP probes in the diagnostic log are separate requests; "
            "this report does not identify the BLS request's source IP."
        )
