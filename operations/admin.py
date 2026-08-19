from __future__ import annotations

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
    ScraperRun,
    ScraperSchedule,
)
from .services import (
    build_default_scraper_config,
    create_scraper_run,
    serialize_scraper_config,
    recover_stale_scraper_runs
)
from .tasks import run_scraper_task


@admin.register(ScraperSchedule)
class ScraperScheduleAdmin(
    admin.ModelAdmin
):
    list_display = (
        "enabled",
        "interval_minutes",
        "last_dispatched_at",
        "updated_at",
    )

    fields = (
        "enabled",
        "interval_minutes",
        "last_dispatched_at",
        "updated_at",
    )

    readonly_fields = (
        "last_dispatched_at",
        "updated_at",
    )

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
