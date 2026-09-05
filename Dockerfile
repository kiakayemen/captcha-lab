FROM python:3.12-slim

ARG GIT_COMMIT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_COMMIT=$GIT_COMMIT \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PLAYWRIGHT_EXECUTABLE_PATH=/usr/bin/chromium

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get update \
    && apt-get install -y --no-install-recommends chromium \
    && rm -rf /var/lib/apt/lists/*

COPY . .
COPY docker/entrypoint.sh /usr/local/bin/captcha-lab-entrypoint
RUN chmod +x /usr/local/bin/captcha-lab-entrypoint \
    && mkdir -p /app/staticfiles /app/media /app/data \
    && python -m compileall -q . \
    && python -c "import django, celery, cv2, joblib, numpy, pandas, sklearn, torch; import playwright.sync_api; import timm; import lightning"

EXPOSE 8000
ENTRYPOINT ["captcha-lab-entrypoint"]
CMD ["web"]
