#!/bin/sh
set -eu

case "${1:-web}" in
  web)
    python manage.py migrate --noinput
    python manage.py collectstatic --noinput
    exec gunicorn control_panel.wsgi:application \
      --bind "0.0.0.0:${PORT:-8000}" \
      --workers "${WEB_CONCURRENCY:-2}" \
      --timeout "${WEB_TIMEOUT:-120}"
    ;;
  worker)
    exec celery -A control_panel worker -l "${CELERY_LOG_LEVEL:-info}" \
      --pool="${CELERY_POOL:-solo}" \
      --concurrency="${CELERY_CONCURRENCY:-1}"
    ;;
  beat)
    exec celery -A control_panel beat -l "${CELERY_LOG_LEVEL:-info}"
    ;;
  migrate)
    exec python manage.py migrate --noinput
    ;;
  *)
    exec "$@"
    ;;
esac
