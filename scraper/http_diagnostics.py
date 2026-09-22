from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import socket
from datetime import datetime, timezone
from typing import Any

from django.conf import settings


SAFE_RESPONSE_HEADERS = (
    "content-type",
    "content-length",
    "date",
    "server",
    "via",
    "cf-ray",
    "cf-cache-status",
    "x-request-id",
    "request-id",
    "x-correlation-id",
    "traceparent",
)
REQUEST_ID_HEADERS = (
    "cf-ray",
    "x-request-id",
    "request-id",
    "x-correlation-id",
    "traceparent",
)
EGRESS_IP_URL = "https://checkip.amazonaws.com/"


def diagnostic_hash(value: str) -> str:
    """Return a stable, non-reversible identifier for sensitive data."""
    return hmac.new(
        str(settings.SECRET_KEY).encode(),
        value.encode(),
        hashlib.sha256,
    ).hexdigest()[:16]


def worker_id() -> str:
    return next(
        (
            value
            for value in (
                os.getenv("POD_NAME"),
                os.getenv("HOSTNAME"),
                os.getenv("DYNO"),
                socket.gethostname(),
            )
            if value
        ),
        "unknown",
    )


def resolve_egress_ip(context) -> tuple[str | None, str | None, str | None]:
    """Observe the IP AWS checkip sees through the browser's proxy."""
    try:
        response = context.request.get(EGRESS_IP_URL, timeout=10_000)
        if not response.ok:
            return None, None, f"http_{response.status}"
        address = response.text().strip()
        if not address:
            return None, None, "missing_ip"
        address = str(ipaddress.ip_address(address))
        return address, diagnostic_hash(address), None
    except Exception as error:
        return None, None, type(error).__name__


def response_diagnostics(
    *,
    response,
    context,
    account: str,
    egress_ip: str | None,
    egress_ip_hash: str | None,
    egress_lookup_error: str | None,
    proxy_endpoint: str | None,
    seconds_since_previous_login: float | None,
) -> dict[str, Any]:
    try:
        headers = {key.lower(): value for key, value in response.all_headers().items()}
    except Exception:
        headers = {}
    safe_headers = {
        key: headers[key]
        for key in SAFE_RESPONSE_HEADERS
        if key in headers
    }
    request_id = next(
        (headers[key] for key in REQUEST_ID_HEADERS if headers.get(key)),
        None,
    )
    try:
        body = response.body()
        body_fingerprint = hashlib.sha256(body).hexdigest()[:16]
        body_length = len(body)
    except Exception:
        body_fingerprint = None
        body_length = None
    try:
        cookies = sorted(
            (
                cookie.get("name", ""),
                cookie.get("domain", ""),
                cookie.get("value", ""),
            )
            for cookie in context.cookies()
        )
        session_hash = diagnostic_hash(json.dumps(cookies, separators=(",", ":")))
    except Exception:
        session_hash = None

    return {
        "http_status": response.status,
        "response_observed_at": datetime.now(timezone.utc).isoformat(),
        "response_request_url": getattr(response, "url", None),
        "response_request_method": getattr(
            getattr(response, "request", None), "method", None
        ),
        "response_headers": safe_headers,
        "body_sha256_16": body_fingerprint,
        "body_length": body_length,
        "server_request_id": request_id,
        "egress_ip": egress_ip,
        "egress_ip_observation": "independent_aws_checkip_probe",
        "egress_ip_hash": egress_ip_hash,
        "egress_lookup_error": egress_lookup_error,
        "proxy_endpoint": proxy_endpoint,
        "worker_container_id": worker_id(),
        "account_hash": diagnostic_hash(account.strip().lower()),
        "session_hash": session_hash,
        "seconds_since_previous_login_request": (
            round(seconds_since_previous_login, 3)
            if seconds_since_previous_login is not None
            else None
        ),
    }
