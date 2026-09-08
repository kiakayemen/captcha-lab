from __future__ import annotations

import uuid
from datetime import timedelta

from django.core.validators import MinValueValidator
from django.db import models
from django.utils import timezone


class ScraperRun(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        APPOINTMENT_FOUND = (
            "appointment_found",
            "Appointment Found",
        )
        NO_APPOINTMENT = (
            "no_appointment",
            "No Appointment",
        )
        FAILED = "failed", "Failed"

    class Trigger(models.TextChoices):
        MANUAL = "manual", "Manual"
        SCHEDULED = "scheduled", "Scheduled"

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )

    status = models.CharField(
        max_length=32,
        choices=Status.choices,
        default=Status.PENDING,
        db_index=True,
    )

    trigger = models.CharField(
        max_length=16,
        choices=Trigger.choices,
        default=Trigger.MANUAL,
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
    )

    started_at = models.DateTimeField(
        null=True,
        blank=True,
    )

    heartbeat_at = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
    )

    finished_at = models.DateTimeField(
        null=True,
        blank=True,
    )

    visa_sub_types = models.JSONField(
        default=list,
        blank=True,
    )

    appointment_visa_sub_type = models.CharField(
        max_length=255,
        blank=True,
    )

    page_url = models.URLField(
        max_length=2000,
        blank=True,
    )

    error_type = models.CharField(
        max_length=255,
        blank=True,
    )

    error_message = models.TextField(
        blank=True,
    )

    failure_screenshot = models.CharField(
        max_length=1000,
        blank=True,
    )

    duration_seconds = models.FloatField(
        null=True,
        blank=True,
    )

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return (
            f"{self.created_at:%Y-%m-%d %H:%M:%S} "
            f"— {self.status}"
        )


class ScraperRunLog(models.Model):
    run = models.ForeignKey(
        ScraperRun,
        on_delete=models.CASCADE,
        related_name="logs",
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
    )

    level = models.CharField(
        max_length=16,
        default="INFO",
    )

    message = models.TextField()

    class Meta:
        ordering = ["id"]

    def __str__(self) -> str:
        return (
            f"{self.created_at:%H:%M:%S} "
            f"[{self.level}] "
            f"{self.message}"
        )


class ScraperEvent(models.Model):
    """Durable, queryable facts emitted during one scraper execution."""

    class EventType(models.TextChoices):
        RUN_STARTED = "run_started", "Run started"
        RUN_FINISHED = "run_finished", "Run finished"
        RUN_FAILED = "run_failed", "Run failed"
        RUN_RECOVERED = "run_recovered", "Run recovered"
        SUBTYPE_STARTED = "subtype_started", "Subtype started"
        SUBTYPE_FINISHED = "subtype_finished", "Subtype finished"
        SUBTYPE_RETRY = "subtype_retry", "Subtype retry"
        CAPTCHA_STARTED = "captcha_started", "CAPTCHA started"
        CAPTCHA_DECISION = "captcha_decision", "CAPTCHA decision"
        CAPTCHA_FINISHED = "captcha_finished", "CAPTCHA finished"
        BROWSER_STARTED = "browser_started", "Browser started"
        BROWSER_FAILED = "browser_failed", "Browser failed"
        NAVIGATION_FAILED = "navigation_failed", "Navigation failed"
        APPOINTMENT_DETECTED = "appointment_detected", "Appointment detected"
        NOTIFICATION_SENT = "notification_sent", "Notification sent"
        NOTIFICATION_FAILED = "notification_failed", "Notification failed"

    run = models.ForeignKey(
        ScraperRun,
        on_delete=models.CASCADE,
        related_name="events",
    )

    # One ScraperRun can be accidentally invoked more than once by a
    # duplicated task delivery. This UUID distinguishes those executions.
    execution_id = models.UUIDField(
        default=uuid.uuid4,
        editable=False,
        db_index=True,
    )

    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
    )

    event_type = models.CharField(
        max_length=64,
        choices=EventType.choices,
        db_index=True,
    )

    visa_sub_type = models.CharField(
        max_length=255,
        blank=True,
        db_index=True,
    )

    attempt_number = models.PositiveIntegerField(
        null=True,
        blank=True,
    )

    status = models.CharField(
        max_length=64,
        blank=True,
    )

    reason_code = models.CharField(
        max_length=128,
        blank=True,
        db_index=True,
    )

    duration_ms = models.PositiveIntegerField(
        null=True,
        blank=True,
    )

    message = models.TextField(
        blank=True,
    )

    data = models.JSONField(
        default=dict,
        blank=True,
    )

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [
            models.Index(
                fields=["run", "created_at"],
                name="scraper_event_run_time_idx",
            ),
            models.Index(
                fields=["run", "event_type"],
                name="scraper_event_run_type_idx",
            ),
        ]


class ScraperSchedule(models.Model):
    """
    Singleton database configuration for automatic scraper runs.

    Celery Beat checks this configuration once per minute.
    The scraper itself only runs when the configured interval
    has elapsed.
    """

    enabled = models.BooleanField(
        default=True,
        help_text=(
            "Enable or disable automatic scheduled scraper runs."
        ),
    )

    interval_minutes = models.PositiveIntegerField(
        default=30,
        validators=[
            MinValueValidator(1),
        ],
        help_text=(
            "Minimum number of minutes between automatic runs."
        ),
    )

    last_dispatched_at = models.DateTimeField(
        null=True,
        blank=True,
        editable=False,
        help_text=(
            "Last time an automatic scraper run was dispatched."
        ),
    )

    updated_at = models.DateTimeField(
        auto_now=True,
    )

    class Meta:
        verbose_name = "Scraper schedule"
        verbose_name_plural = "Scraper schedule"

    def __str__(self) -> str:
        state = (
            "Enabled"
            if self.enabled
            else "Disabled"
        )

        return (
            f"{state} — every "
            f"{self.interval_minutes} minute(s)"
        )

    @property
    def next_run_at(self):
        """Estimated next dispatch time used by the admin UI."""
        if not self.enabled:
            return None
        if self.last_dispatched_at is None:
            return timezone.now()
        return self.last_dispatched_at + timedelta(
            minutes=self.interval_minutes
        )

    def save(
        self,
        *args,
        **kwargs,
    ) -> None:
        # This model is intentionally a singleton.
        self.pk = 1

        super().save(
            *args,
            **kwargs,
        )

    @classmethod
    def load(cls) -> "ScraperSchedule":
        schedule, _created = (
            cls.objects.get_or_create(
                pk=1,
                defaults={
                    "enabled": True,
                    "interval_minutes": 30,
                },
            )
        )

        return schedule
