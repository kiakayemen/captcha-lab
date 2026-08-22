#!/usr/bin/env bash

set -e

cleanup() {
    echo
    echo "Stopping development services..."

    kill "$DJANGO_PID" 2>/dev/null || true
    kill "$WORKER_PID" 2>/dev/null || true
    kill "$BEAT_PID" 2>/dev/null || true
    kill "$REDIS_PID" 2>/dev/null || true

    wait 2>/dev/null || true

    echo "Done."
}

trap cleanup EXIT INT TERM


echo "Starting Redis..."
redis-server &
REDIS_PID=$!


echo "Starting Django..."
python manage.py runserver &
DJANGO_PID=$!


echo "Starting Celery worker..."
celery -A control_panel worker \
    -l info \
    --pool=solo &
WORKER_PID=$!


echo "Starting Celery Beat..."
celery -A control_panel beat \
    -l info &
BEAT_PID=$!


echo
echo "======================================"
echo " Development stack is running"
echo "======================================"
echo
echo " Django:  http://127.0.0.1:8000"
echo " Redis PID:  $REDIS_PID"
echo " Django PID: $DJANGO_PID"
echo " Worker PID: $WORKER_PID"
echo " Beat PID:   $BEAT_PID"
echo
echo "Press Ctrl+C to stop everything."
echo

wait