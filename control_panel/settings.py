from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


def env_bool(
    name: str,
    default: bool = False,
) -> bool:
    value = os.getenv(
        name,
        str(default),
    )

    return (
        value.strip().lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )


def env_list(
    name: str,
    default: str = "",
) -> list[str]:
    raw = os.getenv(
        name,
        default,
    )

    return [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]


# ------------------------------------------------------------------
# Core Django
# ------------------------------------------------------------------

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "development-only-insecure-key",
)


DEBUG = env_bool(
    "DJANGO_DEBUG",
    True,
)


ALLOWED_HOSTS = env_list(
    "DJANGO_ALLOWED_HOSTS",
    "127.0.0.1,localhost",
)


CSRF_TRUSTED_ORIGINS = env_list(
    "DJANGO_CSRF_TRUSTED_ORIGINS",
)


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    "operations",
]


MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


ROOT_URLCONF = (
    "control_panel.urls"
)


TEMPLATES = [
    {
        "BACKEND": (
            "django.template.backends."
            "django.DjangoTemplates"
        ),
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                (
                    "django.template."
                    "context_processors.request"
                ),
                (
                    "django.contrib.auth."
                    "context_processors.auth"
                ),
                (
                    "django.contrib.messages."
                    "context_processors.messages"
                ),
            ],
        },
    },
]


WSGI_APPLICATION = (
    "control_panel.wsgi.application"
)


# ------------------------------------------------------------------
# Database
# ------------------------------------------------------------------
#
# Keep SQLite as the development default.
#
# Once you tell me the deployment target we'll decide whether this
# stays SQLite or moves to PostgreSQL.
#

DATABASES = {
    "default": {
        "ENGINE": (
            "django.db.backends.sqlite3"
        ),
        "NAME": os.getenv(
            "DJANGO_SQLITE_PATH",
            str(
                BASE_DIR
                / "db.sqlite3"
            ),
        ),
    }
}


# ------------------------------------------------------------------
# Password validation
# ------------------------------------------------------------------

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": (
            "django.contrib.auth."
            "password_validation."
            "UserAttributeSimilarityValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth."
            "password_validation."
            "MinimumLengthValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth."
            "password_validation."
            "CommonPasswordValidator"
        ),
    },
    {
        "NAME": (
            "django.contrib.auth."
            "password_validation."
            "NumericPasswordValidator"
        ),
    },
]


LANGUAGE_CODE = (
    "en-us"
)


TIME_ZONE = os.getenv(
    "DJANGO_TIME_ZONE",
    "UTC",
)


USE_I18N = True

USE_TZ = True


# ------------------------------------------------------------------
# Static files
# ------------------------------------------------------------------

STATIC_URL = (
    "/static/"
)


STATIC_ROOT = Path(
    os.getenv(
        "DJANGO_STATIC_ROOT",
        str(
            BASE_DIR
            / "staticfiles"
        ),
    )
)


DEFAULT_AUTO_FIELD = (
    "django.db.models.BigAutoField"
)


# ------------------------------------------------------------------
# Security
# ------------------------------------------------------------------

SECURE_SSL_REDIRECT = env_bool(
    "DJANGO_SECURE_SSL_REDIRECT",
    False,
)


SESSION_COOKIE_SECURE = env_bool(
    "DJANGO_SESSION_COOKIE_SECURE",
    False,
)


CSRF_COOKIE_SECURE = env_bool(
    "DJANGO_CSRF_COOKIE_SECURE",
    False,
)


SECURE_PROXY_SSL_HEADER = (
    (
        "HTTP_X_FORWARDED_PROTO",
        "https",
    )
    if env_bool(
        "DJANGO_TRUST_PROXY_SSL",
        False,
    )
    else None
)


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,

    "formatters": {
        "scraper": {
            "format": (
                "%(asctime)s | "
                "%(levelname)-8s | "
                "%(message)s"
            ),
            "datefmt": (
                "%Y-%m-%d %H:%M:%S"
            ),
        },
    },

    "handlers": {
        "console": {
            "class": (
                "logging.StreamHandler"
            ),
            "formatter": (
                "scraper"
            ),
        },
    },

    "loggers": {
        "captcha_lab": {
            "handlers": [
                "console"
            ],
            "level": os.getenv(
                "CAPTCHA_LOG_LEVEL",
                "INFO",
            ),
            "propagate": False,
        },
    },
}


# ------------------------------------------------------------------
# Celery / Redis
# ------------------------------------------------------------------

CELERY_BROKER_URL = os.getenv(
    "CELERY_BROKER_URL",
    "redis://127.0.0.1:6379/0",
)


CELERY_RESULT_BACKEND = os.getenv(
    "CELERY_RESULT_BACKEND",
    CELERY_BROKER_URL,
)


CELERY_TASK_SERIALIZER = (
    "json"
)


CELERY_ACCEPT_CONTENT = [
    "json"
]


CELERY_RESULT_SERIALIZER = (
    "json"
)


CELERY_TIMEZONE = os.getenv(
    "CELERY_TIMEZONE",
    "UTC",
)


CELERY_ENABLE_UTC = True


CELERY_TASK_ACKS_LATE = True


CELERY_WORKER_PREFETCH_MULTIPLIER = 1


# ------------------------------------------------------------------
# Beat
# ------------------------------------------------------------------
#
# Beat only performs the scheduler check.
# The actual interval remains controlled by ScraperSchedule in DB.
#

CELERY_BEAT_SCHEDULE = {
    "check-scheduled-scraper": {
        "task": (
            "operations."
            "run_scheduled_scraper"
        ),
        "schedule": timedelta(
            minutes=1
        ),
    },
}
