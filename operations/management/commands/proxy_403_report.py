"""Compare 403 responses with all observed responses by route and endpoint."""

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
        grouped = defaultdict(lambda: {
            "count": 0, "total": 0, "runs": set(), "egress_probes": set(),
        })
        events = ScraperEvent.objects.filter(
            event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
            created_at__gte=since,
        ).values("run_id", "status", "data")
        for event in events.iterator():
            data = event["data"] or {}
            if not event["status"].isdigit():
                continue
            proxy = data.get("proxy_endpoint") or "unknown proxy"
            url = data.get("response_url") or data.get("response_request_url") or ""
            path = urlsplit(url).path.lower() or "unknown path"
            bucket = grouped[(proxy, path)]
            bucket["total"] += 1
            if event["status"] == "403":
                bucket["count"] += 1
                bucket["runs"].add(str(event["run_id"]))
                probe = data.get("egress_ip_at_failure") or data.get("egress_ip")
                if probe:
                    bucket["egress_probes"].add(str(probe))

        rows = []
        for (proxy, path), values in grouped.items():
            if not values["count"]:
                continue
            # Initial login navigation records all statuses. Workflow events
            # currently record 403s only, so a workflow "rate" would lie.
            full_coverage = path == "/global/account/login"
            rows.append({
                "proxy_endpoint": proxy,
                "request_path": path,
                "responses": values["count"],
                "runs": len(values["runs"]),
                "total_responses": values["total"] if full_coverage else None,
                "forbidden_percent": (
                    round(100 * values["count"] / values["total"], 1)
                    if full_coverage else None
                ),
                "independent_egress_probes": sorted(values["egress_probes"]),
            })
        rows.sort(key=lambda row: (-row["responses"], row["proxy_endpoint"], row["request_path"]))
        if options["as_json"]:
            self.stdout.write(json.dumps({"hours": hours, "rows": rows}))
            return
        self.stdout.write(f"HTTP 403 responses in the last {hours} hours:")
        if not rows:
            self.stdout.write("None recorded.")
            return
        for row in rows:
            rate = (
                f"{row['responses']}/{row['total_responses']} "
                f"({row['forbidden_percent']:.1f}% 403)"
                if row["forbidden_percent"] is not None
                else f"{row['responses']} 403s (workflow successes not recorded)"
            )
            self.stdout.write(
                f"{rate} / {row['runs']:>3} runs  "
                f"{row['proxy_endpoint']}  {row['request_path']}"
            )
        self.stdout.write(
            "Egress probes are independent requests and cannot identify the "
            "blocked BLS request's source IP; proxy/provider NAT logs are needed."
        )
