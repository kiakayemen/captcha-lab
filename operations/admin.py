from __future__ import annotations

from pathlib import Path

from django.contrib import admin, messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.urls import path, reverse

from scraper.models import ScraperConfig

from .models import ScraperRun
from .services import execute_scraper_run


@admin.register(ScraperRun)
class ScraperRunAdmin(admin.ModelAdmin):
    change_list_template = "admin/operations/scraperrun/change_list.html"

    list_display = (
        "created_at",
        "status",
        "trigger",
        "appointment_visa_sub_type",
        "duration_seconds",
    )

    list_filter = (
        "status",
        "trigger",
        "created_at",
    )

    search_fields = (
        "error_message",
        "error_type",
        "appointment_visa_sub_type",
        "page_url",
    )

    readonly_fields = (
        "id",
        "status",
        "trigger",
        "created_at",
        "started_at",
        "finished_at",
        "visa_sub_types",
        "appointment_visa_sub_type",
        "page_url",
        "error_type",
        "error_message",
        "failure_screenshot",
        "duration_seconds",
    )

    ordering = (
        "-created_at",
    )

    def has_add_permission(
        self,
        request: HttpRequest,
    ) -> bool:
        return False

    def get_urls(self):
        urls = super().get_urls()

        custom_urls = [
            path(
                "run-now/",
                self.admin_site.admin_view(
                    self.run_now_view
                ),
                name="operations_scraperrun_run_now",
            ),
        ]

        return custom_urls + urls

    def run_now_view(
        self,
        request: HttpRequest,
    ) -> HttpResponse:
        if request.method != "POST":
            return redirect(
                reverse(
                    "admin:operations_scraperrun_changelist"
                )
            )
        # change for production
        config = ScraperConfig(
            headless=False,
            gpu=True,
            output_dir=Path("output/live_solver"),
            visa_sub_types=(
                "Student Visa",
                "Non-Working Residence Visa",
            ),
        )

        try:
            db_run = execute_scraper_run(
                config=config,
                trigger=ScraperRun.Trigger.MANUAL,
            )

        except Exception as exc:
            self.message_user(
                request,
                (
                    "Scraper execution failed unexpectedly: "
                    f"{type(exc).__name__}: {exc}"
                ),
                level=messages.ERROR,
            )

            return redirect(
                reverse(
                    "admin:operations_scraperrun_changelist"
                )
            )

        if db_run.status == ScraperRun.Status.APPOINTMENT_FOUND:
            self.message_user(
                request,
                (
                    "Scraper completed: appointment "
                    "availability detected."
                ),
                level=messages.SUCCESS,
            )

        elif db_run.status == ScraperRun.Status.NO_APPOINTMENT:
            self.message_user(
                request,
                "Scraper completed: no appointments found.",
                level=messages.INFO,
            )

        elif db_run.status == ScraperRun.Status.FAILED:
            self.message_user(
                request,
                (
                    "Scraper finished with an error. "
                    "Open the run for details."
                ),
                level=messages.ERROR,
            )

        else:
            self.message_user(
                request,
                f"Scraper finished with status: {db_run.status}",
                level=messages.WARNING,
            )

        return redirect(
            reverse(
                "admin:operations_scraperrun_change",
                args=[db_run.pk],
            )
        )
