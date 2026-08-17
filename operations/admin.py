from django.contrib import admin

from .models import ScraperRun


@admin.register(ScraperRun)
class ScraperRunAdmin(admin.ModelAdmin):
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
        "created_at",
        "started_at",
        "finished_at",
        "duration_seconds",
    )

    ordering = (
        "-created_at",
    )
