from __future__ import annotations

import uuid

from django.core.validators import MinValueValidator
from django.db import models


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
