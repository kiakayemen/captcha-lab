from __future__ import annotations

import os

from celery import Celery


os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "control_panel.settings",
)

app = Celery("control_panel")

app.config_from_object(
    "django.conf:settings",
    namespace="CELERY",
)

app.conf.broker_url = os.getenv(
    "CELERY_BROKER_URL",
    "redis://127.0.0.1:6379/0",
)

app.conf.task_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.result_serializer = "json"
app.conf.timezone = "UTC"
app.conf.enable_utc = True

app.autodiscover_tasks()
