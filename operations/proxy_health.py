from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from scraper.proxy import configured_proxy_urls, playwright_proxy_config

from .models import ProxyEndpointHealth, ScraperEvent, ScraperRun

QUARANTINE_DURATION = timedelta(hours=1)


@dataclass(frozen=True)
class ProxyPoolSelection:
    proxy_urls: tuple[str, ...]
    recovery_probe: bool = False


def select_proxy_pool(now=None, *, proxy_urls=None) -> ProxyPoolSelection:
    """Exclude proxy endpoints under either a 403 or 429 cooldown."""
    now = now or timezone.now()
    configured = configured_proxy_urls() if proxy_urls is None else proxy_urls
    if not configured:
        return ProxyPoolSelection(())

    urls_by_endpoint = {
        playwright_proxy_config(url)["server"]: url for url in configured
    }
    health_by_endpoint = {
        health.endpoint: health
        for health in ProxyEndpointHealth.objects.filter(
            endpoint__in=urls_by_endpoint
        )
    }
    for endpoint in urls_by_endpoint:
        if endpoint not in health_by_endpoint:
            health_by_endpoint[endpoint] = ProxyEndpointHealth.objects.create(
                endpoint=endpoint
            )

    expired_quarantines = [
        health
        for health in health_by_endpoint.values()
        if health.quarantined_until is not None
        and health.quarantined_until <= now
    ]
    for health in expired_quarantines:
        health.quarantined_until = None
        health.save(update_fields=["quarantined_until", "updated_at"])

    healthy_urls = tuple(
        url
        for endpoint, url in urls_by_endpoint.items()
        if health_by_endpoint[endpoint].quarantined_until is None
        and (health_by_endpoint[endpoint].rate_limited_until is None
             or health_by_endpoint[endpoint].rate_limited_until <= now)
    )
    if healthy_urls:
        return ProxyPoolSelection(
            healthy_urls,
            recovery_probe=bool(expired_quarantines),
        )

    return ProxyPoolSelection(())


@transaction.atomic
def update_proxy_health_from_run(run: ScraperRun) -> None:
    """Persist the last observed HTTP state for every endpoint used by a run."""
    latest_by_endpoint: dict[str, ScraperEvent] = {}
    events = run.events.filter(
        event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
    ).order_by("created_at", "id")
    for event in events:
        endpoint = event.data.get("proxy_endpoint")
        if endpoint and event.status in {"200", "403"}:
            previous = latest_by_endpoint.get(endpoint)
            # A later success in this run cannot erase an observed 403.
            # Probe IPs do not prove which address BLS saw.
            if previous is None or event.status == "403" or previous.status != "403":
                latest_by_endpoint[endpoint] = event

    for endpoint, event in latest_by_endpoint.items():
        health, _created = ProxyEndpointHealth.objects.select_for_update().get_or_create(
            endpoint=endpoint
        )
        health.last_checked_at = event.created_at
        if event.status == "403":
            health.consecutive_403 += 1
            health.last_403_at = event.created_at
            health.quarantined_until = event.created_at + QUARANTINE_DURATION
            update_fields = [
                "consecutive_403",
                "last_checked_at",
                "last_403_at",
                "quarantined_until",
                "updated_at",
            ]
        else:
            health.consecutive_403 = 0
            health.last_success_at = event.created_at
            health.quarantined_until = None
            update_fields = [
                "consecutive_403",
                "last_checked_at",
                "last_success_at",
                "quarantined_until",
                "updated_at",
            ]
        health.save(update_fields=update_fields)
