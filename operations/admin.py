from __future__ import annotations

import csv
import json
import secrets
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
    HttpResponseNotAllowed,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import (
    get_object_or_404,
    redirect,
)
from django.core.exceptions import PermissionDenied
from django.urls import (
    path,
    reverse,
)
from django.utils import timezone
from django.utils.html import format_html
from django.db.models import Max, Min, Prefetch

from scraper.proxy_inventory import (
    load_proxy_inventory,
    proxy_label_for_endpoint,
    proxy_label_for_ip,
)

from .models import (
    ProxyEndpointHealth,
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
from .tasks import run_scraper_task, schedule_phase


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
    change_form_template = "admin/operations/scraperschedule/change_form.html"

    list_display = (
        "enabled",
        "back_to_back_enabled",
        "interval_minutes",
        "last_dispatched_at",
        "last_back_to_back_finished_at",
        "next_run",
        "updated_at",
    )

    fields = (
        "enabled",
        "back_to_back_status",
        "interval_minutes",
        "last_dispatched_at",
        "last_back_to_back_finished_at",
        "next_run",
        "updated_at",
    )

    readonly_fields = (
        "back_to_back_status",
        "last_dispatched_at",
        "last_back_to_back_finished_at",
        "next_run",
        "updated_at",
    )

    @admin.display(description="Back-to-back schedule")
    def back_to_back_status(self, obj: ScraperSchedule) -> str:
        return "Active" if obj.back_to_back_enabled else "Inactive"

    def get_urls(self):
        return [
            path(
                "<int:object_id>/set-back-to-back/",
                self.admin_site.admin_view(self.set_back_to_back_view),
                name="operations_scraperschedule_set_back_to_back",
            ),
        ] + super().get_urls()

    def set_back_to_back_view(self, request: HttpRequest, object_id: int) -> HttpResponse:
        if not self.has_change_permission(request):
            raise PermissionDenied
        if request.method != "POST":
            return HttpResponseNotAllowed(["POST"])
        desired = request.POST.get("enabled")
        if desired not in {"0", "1"}:
            return HttpResponseBadRequest("Invalid back-to-back state.")
        schedule = get_object_or_404(ScraperSchedule, pk=object_id)
        schedule.back_to_back_enabled = desired == "1"
        schedule.save(update_fields=["back_to_back_enabled", "updated_at"])
        self.message_user(
            request,
            "Back-to-back schedule activated."
            if schedule.back_to_back_enabled
            else "Back-to-back schedule deactivated.",
            level=messages.SUCCESS,
        )
        return redirect(
            reverse("admin:operations_scraperschedule_change", args=[schedule.pk])
        )

    @admin.display(description="Next scraper run")
    def next_run(self, obj: ScraperSchedule):
        if obj.enabled and obj.back_to_back_enabled:
            phase = schedule_phase(timezone.now(), back_to_back_enabled=True)
            if phase == "prepare":
                return "Back-to-back starts at 10:00 Tehran time"
            if phase == "back_to_back":
                return "Back-to-back runs until 11:00 Tehran time"
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


@admin.register(ProxyEndpointHealth)
class ProxyEndpointHealthAdmin(admin.ModelAdmin):
    list_display = (
        "proxy_name",
        "endpoint",
        "is_quarantined",
        "quarantined_until",
        "consecutive_403",
        "last_checked_at",
        "last_success_at",
    )
    readonly_fields = (
        "proxy_name",
        "endpoint",
        "quarantined_until",
        "consecutive_403",
        "last_checked_at",
        "last_403_at",
        "last_success_at",
        "updated_at",
    )

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    @admin.display(description="Name")
    def proxy_name(self, obj: ProxyEndpointHealth) -> str:
        return proxy_label_for_endpoint(obj.endpoint)

    def has_delete_permission(self, request: HttpRequest, obj=None) -> bool:
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
        "ips_used",
        "ip_names",
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

    actions = (
        "download_selected_logs",
    )

    def changelist_view(self, request, extra_context=None):
        from scraper.proxy import configured_proxy_urls, playwright_proxy_config

        now = timezone.now()
        configured = {
            playwright_proxy_config(url)["server"] for url in configured_proxy_urls()
        }
        health = {item.endpoint: item for item in ProxyEndpointHealth.objects.all()}
        endpoints = []
        inventoried = {route["endpoint"] for route in load_proxy_inventory() if route["endpoint"]}
        for endpoint in sorted(configured | set(health) | inventoried):
            item = health.get(endpoint)
            endpoints.append({
                "endpoint": endpoint,
                "name": proxy_label_for_endpoint(endpoint),
                "configured": endpoint in configured,
                "quarantine_until": item.quarantined_until if item and item.quarantined_until and item.quarantined_until > now else None,
                "cooldown_until": item.rate_limited_until if item and item.rate_limited_until and item.rate_limited_until > now else None,
            })
        last_finish = ScraperRun.objects.filter(
            started_at__isnull=False, finished_at__isnull=False,
        ).order_by("-finished_at").values_list("finished_at", flat=True).first()
        next_manual = last_finish + timedelta(minutes=20) if last_finish else None
        schedule = ScraperSchedule.objects.filter(pk=1).first()
        phase = schedule_phase(now, back_to_back_enabled=schedule.back_to_back_enabled) if schedule else None
        interval_due = schedule.next_run_at if schedule and schedule.enabled and phase == "interval" else None
        back_to_back_until = None
        if schedule and schedule.enabled and phase == "back_to_back" and not schedule.rate_limit_retry_pending:
            last_back_to_back = ScraperRun.objects.filter(
                trigger=ScraperRun.Trigger.BACK_TO_BACK, finished_at__isnull=False,
            ).order_by("-finished_at").first()
            if last_back_to_back and last_back_to_back.status in {
                ScraperRun.Status.FAILED, ScraperRun.Status.SERVER_ERROR, ScraperRun.Status.STOPPED,
            }:
                deadline = last_back_to_back.finished_at + timedelta(minutes=schedule.interval_minutes)
                if deadline > now:
                    back_to_back_until = deadline
        context = {
            **(extra_context or {}),
            "proxy_status_rows": endpoints,
            "proxy_status_now": now,
            "proxy_eligible_count": sum(
                row["configured"] and not row["quarantine_until"] and not row["cooldown_until"]
                for row in endpoints
            ),
            "proxy_configured_count": len(configured),
            "run_gap_until": next_manual if next_manual and next_manual > now else None,
            "run_is_active": ScraperRun.objects.filter(status__in=(ScraperRun.Status.RUNNING, ScraperRun.Status.STOP_REQUESTED)).exists(),
            "scheduler_config": schedule,
            "schedule_phase": phase,
            "interval_due_at": interval_due if interval_due and interval_due > now else None,
            "back_to_back_cooldown_until": back_to_back_until,
            "direct_cooldown_until": schedule.rate_limited_until if schedule and schedule.rate_limited_until and schedule.rate_limited_until > now else None,
        }
        return super().changelist_view(request, extra_context=context)

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
            ),
            Prefetch(
                "events",
                queryset=ScraperEvent.objects.filter(
                    event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
                ).only("run_id", "data"),
                to_attr="ip_usage_events",
            ),
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

    @admin.display(description="IPs used")
    def ips_used(self, obj: ScraperRun) -> int:
        events = getattr(obj, "ip_usage_events", None)
        if events is None:
            events = obj.events.filter(
                event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
            ).only("data")

        return len({
            event.data.get("egress_ip")
            for event in events
            if event.data.get("egress_ip")
        })

    @admin.display(description="IP names")
    def ip_names(self, obj: ScraperRun) -> str:
        events = getattr(obj, "ip_usage_events", None)
        if events is None:
            events = obj.events.filter(
                event_type=ScraperEvent.EventType.LOGIN_RESPONSE,
            ).only("data")

        addresses = dict.fromkeys(
            event.data.get("egress_ip")
            for event in events
            if event.data.get("egress_ip")
        )
        return (
            ", ".join(proxy_label_for_ip(address) for address in addresses)
            or "—"
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
                "download-selected-logs/",
                self.admin_site.admin_view(
                    self.download_selected_logs_view
                ),
                name=(
                    "operations_"
                    "scraperrun_download_selected_logs"
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

    @admin.action(description="Download logs for selected scraper runs")
    def download_selected_logs(
        self,
        request: HttpRequest,
        queryset,
    ) -> HttpResponse:
        # Admin actions are submitted with POST. Some browser download
        # managers retry an attachment URL with GET, which loses the selected
        # IDs and downloads the changelist HTML instead. Save the selection in
        # this admin session and redirect to a stable GET download endpoint,
        # matching the calendar export's behavior.
        selection_token = secrets.token_urlsafe(16)
        request.session["scraper_run_log_export"] = {
            "token": selection_token,
            "run_ids": [
                str(run_id)
                for run_id in queryset.values_list("pk", flat=True)
            ],
        }
        download_url = reverse(
            "admin:operations_scraperrun_download_selected_logs"
        )
        return redirect(f"{download_url}?selection={selection_token}")

    def download_selected_logs_view(
        self,
        request: HttpRequest,
    ) -> HttpResponse:
        selection = request.session.get("scraper_run_log_export", {})
        if (
            not request.GET.get("selection")
            or request.GET["selection"] != selection.get("token")
        ):
            return HttpResponseBadRequest("Invalid or expired scraper run selection.")

        logs = self._logs_for_export(
            ScraperRunLog.objects.filter(
                run_id__in=selection.get("run_ids", [])
            ).order_by(
                "run__created_at",
                "id",
            )
        )
        form = LogDateRangeForm({})
        form.is_valid()
        filename = self._logs_filename(
            logs,
            form,
            "selected_scraper_run_logs",
        )
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

        schedule = ScraperSchedule.load()
        if schedule.enabled and schedule_phase(
            timezone.now(), back_to_back_enabled=schedule.back_to_back_enabled
        ) in {"prepare", "back_to_back"}:
            self.message_user(
                request,
                "Manual runs are paused from 09:50 to 11:00 Tehran time "
                "so the 10:00 back-to-back schedule can start on time.",
                level=messages.WARNING,
            )
            return redirect(reverse("admin:operations_scraperrun_changelist"))

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
