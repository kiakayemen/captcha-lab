from __future__ import annotations

import csv
import json
from datetime import datetime, time, timedelta

from django.contrib import (
    admin,
    messages,
)
from django import forms
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    JsonResponse,
    StreamingHttpResponse,
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
from django.db.models import Max, Min, Prefetch

from .models import (
    ScraperEvent,
    ScraperRun,
    ScraperRunLog,
    ScraperSchedule,
)
from .jalali import parse_jalali_date
from .services import (
    active_scraper_runs,
    build_default_scraper_config,
    create_scraper_run,
    recover_stale_scraper_runs,
    request_scraper_stop,
    serialize_scraper_config,
)
from .tasks import run_scraper_task


class JalaliDateField(forms.Field):
    def to_python(self, value):
        if value in self.empty_values:
            return None
        try:
            return parse_jalali_date(str(value))
        except ValueError as error:
            raise forms.ValidationError(str(error)) from error


class LogDateRangeForm(forms.Form):
    start_date = JalaliDateField(required=False)
    end_date = JalaliDateField(required=False)

    def clean(self):
        cleaned = super().clean()
        start, end = cleaned.get("start_date"), cleaned.get("end_date")
        if start and end and start > end:
            raise forms.ValidationError("Start date must be on or before end date.")
        return cleaned


class CSVBuffer:
    """Minimal file-like object that lets csv write one row at a time."""

    @staticmethod
    def write(value: str) -> str:
        return value


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
        "student_visa_result",
        "non_working_residence_result",
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
        "stop_requested_at",
        "stopped_at",
        "visa_sub_types",
        "appointment_visa_sub_type",
        "page_url",
        "error_type",
        "error_message",
        "failure_screenshot",
        "first_failure",
        "attempt_failures",
        "terminal_failure",
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
                    "stop_requested_at",
                    "stopped_at",
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
                    "first_failure",
                    "attempt_failures",
                    "terminal_failure",
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

    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related(
            Prefetch(
                "events",
                queryset=ScraperEvent.objects.filter(
                    event_type__in=(
                        ScraperEvent.EventType.SUBTYPE_STARTED,
                        ScraperEvent.EventType.SUBTYPE_FINISHED,
                    )
                ).order_by("created_at", "id"),
                to_attr="subtype_outcome_events",
            )
        )

    @staticmethod
    def _subtype_result(run: ScraperRun, subtype: str) -> str:
        if subtype not in run.visa_sub_types:
            return "Not configured"

        events = getattr(run, "subtype_outcome_events", None)
        if events is None:
            events = run.events.filter(
                event_type__in=(
                    ScraperEvent.EventType.SUBTYPE_STARTED,
                    ScraperEvent.EventType.SUBTYPE_FINISHED,
                )
            ).order_by("created_at", "id")

        started = False
        result = None
        for event in events:
            if event.visa_sub_type != subtype:
                continue
            if event.event_type == ScraperEvent.EventType.SUBTYPE_STARTED:
                started = True
            elif event.event_type == ScraperEvent.EventType.SUBTYPE_FINISHED:
                result = event.status

        if result == ScraperRun.Status.APPOINTMENT_FOUND:
            return "Appointment found"
        if result == ScraperRun.Status.POSSIBLE_APPOINTMENT:
            return "Possible appointment"
        if result == ScraperRun.Status.NO_APPOINTMENT:
            return "No appointment"
        if result:
            return result.replace("_", " ").capitalize()

        # Older runs may predate structured subtype events.
        if run.status in (
            ScraperRun.Status.APPOINTMENT_FOUND,
            ScraperRun.Status.POSSIBLE_APPOINTMENT,
            ScraperRun.Status.NO_APPOINTMENT,
        ):
            if run.status == ScraperRun.Status.NO_APPOINTMENT:
                return "No appointment"
            if subtype in (run.appointment_visa_sub_type or "").split(", "):
                return (
                    "Appointment found"
                    if run.status == ScraperRun.Status.APPOINTMENT_FOUND
                    else "Possible appointment"
                )
            return "No appointment"

        failure_subtype = (run.terminal_failure or {}).get("visa_sub_type")
        if not failure_subtype and run.status in (
            ScraperRun.Status.FAILED,
            ScraperRun.Status.SERVER_ERROR,
        ):
            failure_subtype = run.appointment_visa_sub_type
        if failure_subtype == subtype:
            return "Server error" if run.status == ScraperRun.Status.SERVER_ERROR else "Failed"
        if started:
            if run.status in (ScraperRun.Status.RUNNING, ScraperRun.Status.STOP_REQUESTED):
                return "Running" if run.status == ScraperRun.Status.RUNNING else "Stopping"
            return "Stopped" if run.status == ScraperRun.Status.STOPPED else "Failed"
        return "Pending" if run.status == ScraperRun.Status.PENDING else "Not checked"

    @admin.display(description="Student visa")
    def student_visa_result(self, obj: ScraperRun) -> str:
        return self._subtype_result(obj, "Student Visa")

    @admin.display(description="Non-working residence visa")
    def non_working_residence_result(self, obj: ScraperRun) -> str:
        return self._subtype_result(obj, "Non-Working Residence Visa")

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
                "<uuid:run_id>/stop/",
                self.admin_site.admin_view(self.stop_run_view),
                name="operations_scraperrun_stop",
            ),
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
                        ScraperRun.Status.POSSIBLE_APPOINTMENT,
                        ScraperRun.Status.NO_APPOINTMENT,
                        ScraperRun.Status.SERVER_ERROR,
                        ScraperRun.Status.STOPPED,
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

    def stop_run_view(self, request: HttpRequest, run_id) -> HttpResponse:
        run = get_object_or_404(ScraperRun, pk=run_id)
        if request.method == "POST":
            request_scraper_stop(run)
            self.message_user(
                request,
                f"Stop requested for scraper run {run.pk}.",
                level=messages.WARNING,
            )
        return redirect(
            reverse("admin:operations_scraperrun_change", args=[run.pk])
        )

    def download_all_logs_view(
        self,
        request: HttpRequest,
    ) -> HttpResponse:
        form = LogDateRangeForm(request.GET)
        if not form.is_valid():
            return HttpResponseBadRequest("Invalid Jalali log date range. Use YYYY/MM/DD and start before end.")
        logs = self._logs_for_export(
            ScraperRunLog.objects.order_by("run__created_at", "id")
        )
        logs = self._filter_logs_by_date(logs, form)
        filename = self._logs_filename(logs, form, "scraper_run_logs")
        return self._logs_csv_response(logs, filename)

    def download_run_logs_view(
        self,
        request: HttpRequest,
        run_id,
    ) -> HttpResponse:
        run = get_object_or_404(ScraperRun, pk=run_id)
        form = LogDateRangeForm(request.GET)
        if not form.is_valid():
            return HttpResponseBadRequest("Invalid Jalali log date range. Use YYYY/MM/DD and start before end.")
        logs = self._logs_for_export(run.logs.order_by("id"))
        logs = self._filter_logs_by_date(logs, form)
        filename = self._logs_filename(
            logs,
            form,
            f"scraper_run_{run.pk}_logs",
        )
        return self._logs_csv_response(
            logs,
            filename,
        )

    @staticmethod
    def _filter_logs_by_date(logs, form: LogDateRangeForm):
        start = form.cleaned_data.get("start_date")
        end = form.cleaned_data.get("end_date")
        if start:
            logs = logs.filter(created_at__gte=timezone.make_aware(datetime.combine(start, time.min)))
        if end:
            next_day = end + timedelta(days=1)
            logs = logs.filter(created_at__lt=timezone.make_aware(datetime.combine(next_day, time.min)))
        return logs

    @staticmethod
    def _logs_filename(logs, form: LogDateRangeForm, prefix: str) -> str:
        """Include the actual exported date range in every filename."""
        bounds = logs.aggregate(
            first_log_at=Min("created_at"),
            last_log_at=Max("created_at"),
        )
        if bounds["first_log_at"] is not None:
            start = timezone.localdate(bounds["first_log_at"])
            end = timezone.localdate(bounds["last_log_at"])
        else:
            # An empty export has no actual range, so retain the requested
            # bounds to make the resulting filename as informative as possible.
            start = form.cleaned_data.get("start_date")
            end = form.cleaned_data.get("end_date")

        if start is None and end is None:
            range_label = "no-logs"
        elif start is None:
            range_label = f"through_{end.isoformat()}"
        elif end is None:
            range_label = f"from_{start.isoformat()}"
        else:
            range_label = f"{start.isoformat()}_to_{end.isoformat()}"

        return f"{prefix}_{range_label}.csv"

    @staticmethod
    def _logs_for_export(logs):
        """Fetch only CSV fields while retaining one joined query per batch."""
        return logs.select_related("run").only(
            "id",
            "created_at",
            "level",
            "message",
            "run__id",
            "run__status",
            "run__trigger",
            "run__created_at",
            "run__started_at",
            "run__finished_at",
            "run__duration_seconds",
            "run__appointment_visa_sub_type",
            "run__page_url",
            "run__error_type",
            "run__error_message",
            "run__failure_screenshot",
            "run__first_failure",
            "run__attempt_failures",
            "run__terminal_failure",
        )

    @staticmethod
    def _logs_csv_response(logs, filename: str) -> StreamingHttpResponse:
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
            "first_failure",
            "attempt_failures",
            "terminal_failure",
        ]

        def rows():
            writer = csv.DictWriter(CSVBuffer(), fieldnames=fieldnames)
            yield writer.writeheader()

            # A bounded iterator prevents Django from caching the full queryset.
            for log in logs.iterator(chunk_size=500):
                run = log.run
                yield writer.writerow(
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
                        "first_failure": json.dumps(run.first_failure or {}),
                        "attempt_failures": json.dumps(run.attempt_failures or []),
                        "terminal_failure": json.dumps(run.terminal_failure or {}),
                    }
                )

        response = StreamingHttpResponse(
            rows(),
            content_type="text/csv; charset=utf-8",
        )
        response["Content-Disposition"] = (
            f'attachment; filename="{filename}"'
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

        active_run = active_scraper_runs().first()

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
