from __future__ import annotations

import csv

from django.contrib import (
    admin,
    messages,
)
from django.http import (
    HttpRequest,
    HttpResponse,
    JsonResponse,
)
from django.shortcuts import (
    get_object_or_404,
    redirect,
)
from django.urls import (
    path,
    reverse,
)
from django.utils import timezone
from django.utils.html import format_html

from .models import (
    ScraperEvent,
    ScraperRun,
    ScraperRunLog,
    ScraperSchedule,
)
from .services import (
    build_default_scraper_config,
    create_scraper_run,
    serialize_scraper_config,
    recover_stale_scraper_runs
)
from .tasks import run_scraper_task


@admin.register(ScraperEvent)
class ScraperEventAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "event_type",
        "visa_sub_type",
        "attempt_number",
        "status",
        "reason_code",
        "duration_ms",
        "run",
        "execution_id",
    )

    list_filter = (
        "event_type",
        "status",
        "reason_code",
        "created_at",
    )

    search_fields = (
        "run__id",
        "execution_id",
        "visa_sub_type",
        "message",
        "reason_code",
    )

    readonly_fields = (
        "id",
        "run",
        "execution_id",
        "created_at",
        "event_type",
        "visa_sub_type",
        "attempt_number",
        "status",
        "reason_code",
        "duration_ms",
        "message",
        "data",
    )


@admin.register(ScraperSchedule)
class ScraperScheduleAdmin(
    admin.ModelAdmin
):
    list_display = (
        "enabled",
        "interval_minutes",
        "last_dispatched_at",
        "next_run",
        "updated_at",
    )

    fields = (
        "enabled",
        "interval_minutes",
        "last_dispatched_at",
        "next_run",
        "updated_at",
    )

    readonly_fields = (
        "last_dispatched_at",
        "next_run",
        "updated_at",
    )

    @admin.display(description="Next scraper run")
    def next_run(self, obj: ScraperSchedule):
        next_run_at = obj.next_run_at
        if next_run_at is None:
            return "Disabled"
        return next_run_at

    def has_add_permission(
        self,
        request: HttpRequest,
    ) -> bool:
        return not (
            ScraperSchedule.objects.exists()
        )

    def has_delete_permission(
        self,
        request: HttpRequest,
        obj=None,
    ) -> bool:
        return False


@admin.register(ScraperRun)
class ScraperRunAdmin(
    admin.ModelAdmin
):
    change_list_template = (
        "admin/operations/"
        "scraperrun/change_list.html"
    )

    change_form_template = (
        "admin/operations/"
        "scraperrun/change_form.html"
    )

    list_display = (
        "created_at",
        "status",
        "trigger",
        "appointment_visa_sub_type",
        "duration_seconds",
        "log_count",
        "download_logs",
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
        "heartbeat_at",
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
                    "heartbeat_at",
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
                "fields": (
                    "run_log",
                ),
            },
        ),
    )

    ordering = (
        "-created_at",
    )

    @admin.display(
        description="Logs"
    )
    def log_count(
        self,
        obj: ScraperRun,
    ) -> int:
        if not obj.pk:
            return 0

        return obj.logs.count()

    @admin.display(description="Download")
    def download_logs(
        self,
        obj: ScraperRun,
    ):
        url = reverse(
            "admin:operations_scraperrun_download_logs",
            args=[obj.pk],
        )
        return format_html(
            '<a class="button" href="{}">Download CSV</a>',
            url,
        )

    @admin.display(
        description="Run log"
    )
    def run_log(
        self,
        obj: ScraperRun,
    ):
        if not obj.pk:
            return "No logs yet."

        entries = list(
            obj.logs.all()
        )

        output = "\n".join(
            entry.message
            for entry in entries
        )

        return format_html(
            '<pre id="live-run-log" '
            'data-last-log-id="{}" '
            'style="background:#111827;'
            'color:#e5e7eb;'
            'padding:16px;'
            'border-radius:6px;'
            'overflow-x:auto;'
            'max-height:700px;'
            'overflow-y:auto;'
            'white-space:pre-wrap;'
            'word-break:break-word;'
            'font-family:ui-monospace,'
            'SFMono-Regular,Menlo,Monaco,'
            'Consolas,monospace;'
            'font-size:12px;'
            'line-height:1.55;'
            'margin:0;">{}</pre>',
            (
                entries[-1].pk
                if entries
                else 0
            ),
            (
                output
                or "Waiting for output..."
            ),
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
                name=(
                    "operations_"
                    "scraperrun_run_now"
                ),
            ),
            path(
                (
                    "<uuid:run_id>/"
                    "live-state/"
                ),
                self.admin_site.admin_view(
                    self.live_state_view
                ),
                name=(
                    "operations_"
                    "scraperrun_live_state"
                ),
            ),
            path(
                "download-logs/",
                self.admin_site.admin_view(
                    self.download_all_logs_view
                ),
                name=(
                    "operations_"
                    "scraperrun_download_all_logs"
                ),
            ),
            path(
                (
                    "<uuid:run_id>/"
                    "download-logs/"
                ),
                self.admin_site.admin_view(
                    self.download_run_logs_view
                ),
                name=(
                    "operations_"
                    "scraperrun_download_logs"
                ),
            ),
        ]

        return (
            custom_urls
            + urls
        )

    def live_state_view(
        self,
        request: HttpRequest,
        run_id,
    ) -> JsonResponse:
        run = get_object_or_404(
            ScraperRun,
            pk=run_id,
        )

        try:
            after_id = int(
                request.GET.get(
                    "after",
                    "0",
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            after_id = 0

        entries = list(
            run.logs
            .filter(
                pk__gt=after_id
            )
            .order_by(
                "pk"
            )
        )

        return JsonResponse(
            {
                "status": (
                    run.status
                ),
                "status_label": (
                    run.get_status_display()
                ),
                "finished": (
                    run.status
                    in {
                        ScraperRun.Status.APPOINTMENT_FOUND,
                        ScraperRun.Status.NO_APPOINTMENT,
                        ScraperRun.Status.FAILED,
                    }
                ),
                "last_log_id": (
                    entries[-1].pk
                    if entries
                    else after_id
                ),
                "logs": [
                    {
                        "id": entry.pk,
                        "message": (
                            entry.message
                        ),
                    }
                    for entry
                    in entries
                ],
            }
        )

    def download_all_logs_view(
        self,
        request: HttpRequest,
    ) -> HttpResponse:
        logs = ScraperRunLog.objects.select_related("run").order_by(
            "run__created_at",
            "id",
        )
        return self._logs_csv_response(logs, "scraper_run_logs.csv")

    def download_run_logs_view(
        self,
        request: HttpRequest,
        run_id,
    ) -> HttpResponse:
        run = get_object_or_404(ScraperRun, pk=run_id)
        logs = run.logs.select_related("run").order_by("id")
        return self._logs_csv_response(
            logs,
            f"scraper_run_{run.pk}_logs.csv",
        )

    @staticmethod
    def _logs_csv_response(logs, filename: str) -> HttpResponse:
        response = HttpResponse(content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = (
            f'attachment; filename="{filename}"'
        )

        fieldnames = [
            "log_id",
            "log_created_at",
            "level",
            "message",
            "run_id",
            "run_status",
            "run_trigger",
            "run_created_at",
            "run_started_at",
            "run_finished_at",
            "run_duration_seconds",
            "appointment_visa_sub_type",
            "run_page_url",
            "error_type",
            "error_message",
            "failure_screenshot",
        ]
        writer = csv.DictWriter(response, fieldnames=fieldnames)
        writer.writeheader()

        for log in logs.iterator():
            run = log.run
            writer.writerow(
                {
                    "log_id": log.id,
                    "log_created_at": log.created_at.isoformat(),
                    "level": log.level,
                    "message": log.message,
                    "run_id": run.id,
                    "run_status": run.status,
                    "run_trigger": run.trigger,
                    "run_created_at": run.created_at.isoformat(),
                    "run_started_at": (
                        run.started_at.isoformat()
                        if run.started_at
                        else ""
                    ),
                    "run_finished_at": (
                        run.finished_at.isoformat()
                        if run.finished_at
                        else ""
                    ),
                    "run_duration_seconds": (
                        run.duration_seconds
                        if run.duration_seconds is not None
                        else ""
                    ),
                    "appointment_visa_sub_type": run.appointment_visa_sub_type,
                    "run_page_url": run.page_url,
                    "error_type": run.error_type,
                    "error_message": run.error_message,
                    "failure_screenshot": run.failure_screenshot,
                }
            )

        return response

    def run_now_view(
        self,
        request: HttpRequest,
    ) -> HttpResponse:
        if request.method != "POST":
            return redirect(
                reverse(
                    "admin:"
                    "operations_"
                    "scraperrun_changelist"
                )
            )

        recover_stale_scraper_runs()

        active_run = (
            ScraperRun.objects
            .filter(
                status__in=[
                    ScraperRun.Status.PENDING,
                    ScraperRun.Status.RUNNING,
                ]
            )
            .order_by(
                "-created_at"
            )
            .first()
        )

        if active_run is not None:
            self.message_user(
                request,
                (
                    "A scraper run is already "
                    "pending or running. "
                    f"Run ID: {active_run.pk}"
                ),
                level=messages.WARNING,
            )

            return redirect(
                reverse(
                    "admin:"
                    "operations_scraperrun_change",
                    args=[
                        active_run.pk
                    ],
                )
            )

        config = (
            build_default_scraper_config()
        )

        db_run = create_scraper_run(
            config=config,
            trigger=(
                ScraperRun.Trigger.MANUAL
            ),
        )

        config_data = (
            serialize_scraper_config(
                config
            )
        )

        try:
            run_scraper_task.delay(
                str(
                    db_run.pk
                ),
                config_data,
            )

        except Exception as exc:
            db_run.status = (
                ScraperRun.Status.FAILED
            )

            db_run.finished_at = (
                timezone.now()
            )

            db_run.error_type = (
                type(exc).__name__
            )

            db_run.error_message = (
                "Could not queue "
                "Celery task: "
                f"{exc}"
            )

            db_run.save(
                update_fields=[
                    "status",
                    "finished_at",
                    "error_type",
                    "error_message",
                ]
            )

            self.message_user(
                request,
                (
                    "Could not queue "
                    "scraper task. "
                    "Check Redis and "
                    "the Celery worker."
                ),
                level=messages.ERROR,
            )

        else:
            self.message_user(
                request,
                (
                    "Scraper queued. "
                    "Live output will appear "
                    "on this page."
                ),
                level=messages.SUCCESS,
            )

        return redirect(
            reverse(
                (
                    "admin:"
                    "operations_"
                    "scraperrun_change"
                ),
                args=[
                    db_run.pk
                ],
            )
        )
