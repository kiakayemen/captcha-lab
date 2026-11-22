"""Persist BLS cooldowns by proxy endpoint across workers and restarts."""
from datetime import timedelta
from email.utils import parsedate_to_datetime
import random

from django.db import transaction
from django.utils import timezone

from .database_writer import run_database_write
from .models import ProxyEndpointHealth, ScraperSchedule

BASE_SECONDS = 240
MAX_BACKOFF_SECONDS = 3600
STREAK_RESET_AFTER = timedelta(hours=24)


def retry_after_seconds(value, now):
    if not value:
        return 0
    value = str(value).strip()
    if value.isdecimal():
        return int(value)
    try:
        target = parsedate_to_datetime(value)
        if timezone.is_naive(target):
            return 0
        return max(0, (target - now).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 0


def note_rate_limit(retry_after=None, now=None, *, endpoint=None):
    """Extend an endpoint cooldown, or the direct-connection cooldown."""
    now = now or timezone.now()

    def update():
        with transaction.atomic():
            schedule, _ = (ProxyEndpointHealth.objects.select_for_update().get_or_create(endpoint=endpoint)
                           if endpoint else ScraperSchedule.objects.select_for_update().get_or_create(pk=1))
            recent = (schedule.rate_limit_last_seen_at is not None and
                      now - schedule.rate_limit_last_seen_at < STREAK_RESET_AFTER)
            # Multiple 429 responses in the same cooldown are one incident.
            if not schedule.rate_limited_until or schedule.rate_limited_until <= now:
                schedule.rate_limit_streak = schedule.rate_limit_streak + 1 if recent else 1
            base = min(MAX_BACKOFF_SECONDS, BASE_SECONDS * 2 ** min(schedule.rate_limit_streak - 1, 4))
            delay = max(base, retry_after_seconds(retry_after, now)) + random.randint(0, 30)
            until = now + timedelta(seconds=delay)
            schedule.rate_limited_until = max(filter(None, (schedule.rate_limited_until, until)))
            schedule.rate_limit_last_seen_at = now
            fields = ["rate_limited_until", "rate_limit_last_seen_at", "rate_limit_streak", "updated_at"]
            if endpoint is None:
                schedule.rate_limit_retry_pending = True
                fields.append("rate_limit_retry_pending")
            schedule.save(update_fields=fields)
            return schedule.rate_limited_until

    return run_database_write(update)
