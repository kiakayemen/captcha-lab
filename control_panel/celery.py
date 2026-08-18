from __future__ import annotations

import os

from celery import Celery


os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "control_panel.settings",
)


app = Celery(
    "control_panel"
)


app.config_from_object(
    "django.conf:settings",
    namespace="CELERY",
)


app.autodiscover_tasks()
