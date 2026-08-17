from __future__ import annotations

from pathlib import Path

from django.contrib import admin, messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect
from django.urls import path, reverse
from django.utils.html import format_html

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
        "log_count",
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
        "run_log",
    )

    fieldsets = (
        (
            "Run",
            {
                "fields": (
                    "id",
                    "status",
                    "trigger",
                    "created_at",
                    "started_at",
                    "finished_at",
                    "duration_seconds",
                )
            },
        ),
        (
            "Scraper configuration / result",
            {
                "fields": (
                    "visa_sub_types",
                    "appointment_visa_sub_type",
                    "page_url",
                )
            },
        ),
        (
            "Failure",
            {
                "fields": (
                    "error_type",
                    "error_message",
                    "failure_screenshot",
                )
            },
        ),
        (
            "Run log",
            {
                "fields": ("run_log",),
            },
        ),
    )

    ordering = ("-created_at",)

    @admin.display(description="Logs")
    def log_count(self, obj: ScraperRun) -> int:
        if not obj.pk:
            return 0
        return obj.logs.count()

    @admin.display(description="Run log")
    def run_log(self, obj: ScraperRun):
        if not obj.pk:
            return "No logs yet."

        entries = list(obj.logs.all())
        if not entries:
            return "No logs recorded for this run."

        lines = []
        for entry in entries:
            timestamp = entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(
                f"{timestamp} | {entry.level:<8} | {entry.message}"
            )

        output = "\n".join(lines)

        return format_html(
            '<pre style="background:#111827;color:#e5e7eb;padding:16px;'
            'border-radius:6px;overflow-x:auto;max-height:700px;'
            'overflow-y:auto;white-space:pre-wrap;word-break:break-word;'
            'font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,'
            'monospace;font-size:12px;line-height:1.55;margin:0;">{}</pre>',
            output,
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

        # Current local debugging configuration.
        # We can switch headless back on once UI logging is trusted.
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
                "Scraper completed: appointment availability detected.",
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
                "Scraper finished with an error. Open the run for details.",
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
