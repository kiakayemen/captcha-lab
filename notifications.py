from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import smtplib
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


logger = logging.getLogger("captcha_lab")


OUTPUT_DIR = Path("output")

EVENT_LOG_PATH = (
    OUTPUT_DIR
    / "appointment_events.jsonl"
)

OUTBOX_DIR = (
    OUTPUT_DIR
    / "notification_outbox"
)

DELIVERED_DIR = (
    OUTPUT_DIR
    / "notification_delivered"
)


@dataclass(frozen=True)
class AppointmentEvent:
    event_id: str
    event_type: str
    message: str
    page_url: str
    visa_sub_type: str | None
    created_at: str


def _utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _create_event_id(
    *,
    event_type: str,
    page_url: str,
    visa_sub_type: str | None,
    created_at: str,
) -> str:
    raw = "|".join(
        [
            event_type,
            visa_sub_type or "",
            page_url,
            created_at,
        ]
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]


def _build_event(
    *,
    event_type: str,
    message: str,
    page_url: str,
    visa_sub_type: str | None,
) -> AppointmentEvent:
    created_at = _utc_now_iso()

    return AppointmentEvent(
        event_id=_create_event_id(
            event_type=event_type,
            page_url=page_url,
            visa_sub_type=visa_sub_type,
            created_at=created_at,
        ),
        event_type=event_type,
        message=message,
        page_url=page_url,
        visa_sub_type=visa_sub_type,
        created_at=created_at,
    )


def _fsync_file(handle) -> None:
    handle.flush()
    os.fsync(
        handle.fileno()
    )


def _append_event_log(
    event: AppointmentEvent,
) -> None:
    """
    Durable local audit record.

    This happens before any external notification attempt.
    """

    EVENT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = asdict(event)

    with EVENT_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
            )
        )

        handle.write("\n")

        _fsync_file(
            handle
        )


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

        _fsync_file(
            handle
        )

    temporary_path.replace(
        path
    )


def _create_outbox_item(
    event: AppointmentEvent,
) -> Path:
    """
    Put the alert into a durable outbox BEFORE external delivery.

    If Python, SMTP, or the network dies afterward, the event still
    exists on disk and can be retried.
    """

    path = (
        OUTBOX_DIR
        / f"{event.event_id}.json"
    )

    payload = {
        "event": asdict(
            event
        ),
        "created_at": (
            _utc_now_iso()
        ),
        "delivery_attempts": 0,
        "last_attempt_at": None,
        "last_error": None,
    }

    _atomic_write_json(
        path,
        payload,
    )

    return path


def _load_outbox_item(
    path: Path,
) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _save_outbox_item(
    path: Path,
    payload: dict[str, Any],
) -> None:
    _atomic_write_json(
        path,
        payload,
    )


def _split_recipients(
    raw: str,
) -> list[str]:
    return [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]


def _email_configured() -> bool:
    return all(
        [
            os.getenv(
                "APPOINTMENT_SMTP_HOST"
            ),
            os.getenv(
                "APPOINTMENT_EMAIL_TO"
            ),
        ]
    )


def _webhook_configured() -> bool:
    return bool(
        os.getenv(
            "APPOINTMENT_WEBHOOK_URL"
        )
    )


def _email_body(
    event: AppointmentEvent,
) -> str:
    return "\n".join(
        [
            "BLS APPOINTMENT ALERT",
            "",
            "Appointment availability may have been detected.",
            "",
            (
                "Visa subtype: "
                f"{event.visa_sub_type or 'Unknown'}"
            ),
            (
                "Detected at: "
                f"{event.created_at}"
            ),
            (
                "Page URL: "
                f"{event.page_url}"
            ),
            "",
            f"Message: {event.message}",
            "",
            (
                "Event ID: "
                f"{event.event_id}"
            ),
            "",
            "Manual verification / booking is required.",
        ]
    )


def _send_email(
    event: AppointmentEvent,
) -> None:
    if not _email_configured():
        raise RuntimeError(
            "Email notification is not configured."
        )

    host = os.environ[
        "APPOINTMENT_SMTP_HOST"
    ]

    port = int(
        os.getenv(
            "APPOINTMENT_SMTP_PORT",
            "587",
        )
    )

    username = os.getenv(
        "APPOINTMENT_SMTP_USERNAME",
        "",
    )

    password = os.getenv(
        "APPOINTMENT_SMTP_PASSWORD",
        "",
    )

    recipients = _split_recipients(
        os.environ[
            "APPOINTMENT_EMAIL_TO"
        ]
    )

    sender = os.getenv(
        "APPOINTMENT_EMAIL_FROM",
        username,
    )

    if not sender:
        raise RuntimeError(
            "APPOINTMENT_EMAIL_FROM is not configured."
        )

    if not recipients:
        raise RuntimeError(
            "APPOINTMENT_EMAIL_TO contains no recipients."
        )

    mode = os.getenv(
        "APPOINTMENT_SMTP_SECURITY",
        "starttls",
    ).strip().lower()

    email = EmailMessage()

    email["Subject"] = (
        "🚨 BLS APPOINTMENT AVAILABLE"
        + (
            f" — {event.visa_sub_type}"
            if event.visa_sub_type
            else ""
        )
    )

    email["From"] = sender
    email["To"] = ", ".join(
        recipients
    )

    email.set_content(
        _email_body(
            event
        )
    )

    timeout = float(
        os.getenv(
            "APPOINTMENT_SMTP_TIMEOUT_SECONDS",
            "15",
        )
    )

    # Some hosting providers use one hostname for the SMTP endpoint and a
    # different, certificate-covered hostname for TLS verification.
    tls_server_name = os.getenv(
        "APPOINTMENT_SMTP_TLS_SERVER_NAME",
        host,
    ).strip() or host

    ssl_context = (
        ssl.create_default_context()
    )

    if mode == "ssl":
        with smtplib.SMTP_SSL(
            timeout=timeout,
            context=ssl_context,
        ) as smtp:
            # SMTP_SSL performs its handshake during connect(). Set the
            # certificate name first, while still connecting to the endpoint.
            smtp._host = tls_server_name
            smtp.connect(host, port)

            if username:
                smtp.login(
                    username,
                    password,
                )

            smtp.send_message(
                email
            )

        return

    with smtplib.SMTP(
        host,
        port,
        timeout=timeout,
    ) as smtp:
        smtp.ehlo()

        if mode == "starttls":
            if tls_server_name != host:
                # smtplib uses _host as the TLS SNI and verification name.
                smtp._host = tls_server_name

            try:
                smtp.starttls(
                    context=ssl_context
                )
            except ssl.CertificateError as exc:
                raise RuntimeError(
                    "SMTP TLS certificate does not match "
                    f"{tls_server_name!r}. Set "
                    "APPOINTMENT_SMTP_TLS_SERVER_NAME to the "
                    "hostname listed in the mail provider's certificate, "
                    "or use the provider's correct SMTP hostname."
                ) from exc

            smtp.ehlo()

        elif mode not in {
            "none",
            "plain",
        }:
            raise RuntimeError(
                "Unsupported "
                "APPOINTMENT_SMTP_SECURITY="
                f"{mode!r}. "
                "Use starttls, ssl, or none."
            )

        if username:
            smtp.login(
                username,
                password,
            )

        smtp.send_message(
            email
        )


def _send_webhook(
    event: AppointmentEvent,
) -> None:
    url = os.getenv(
        "APPOINTMENT_WEBHOOK_URL"
    )

    if not url:
        raise RuntimeError(
            "Webhook notification is not configured."
        )

    payload = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "message": event.message,
        "visa_sub_type": (
            event.visa_sub_type
        ),
        "page_url": event.page_url,
        "created_at": event.created_at,
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(
            payload
        ).encode(
            "utf-8"
        ),
        headers={
            "Content-Type": (
                "application/json"
            ),
            "User-Agent": (
                "captcha-lab-notifier/1.0"
            ),
        },
        method="POST",
    )

    timeout = float(
        os.getenv(
            "APPOINTMENT_WEBHOOK_TIMEOUT_SECONDS",
            "15",
        )
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            status = (
                response.status
            )

            if not (
                200
                <= status
                < 300
            ):
                raise RuntimeError(
                    "Webhook returned HTTP "
                    f"{status}"
                )

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "Webhook returned HTTP "
            f"{exc.code}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Webhook connection failed: "
            f"{exc.reason}"
        ) from exc


def _configured_channels() -> list[str]:
    channels: list[str] = []

    if _email_configured():
        channels.append(
            "email"
        )

    if _webhook_configured():
        channels.append(
            "webhook"
        )

    return channels


def _deliver_channel(
    *,
    channel: str,
    event: AppointmentEvent,
) -> None:
    if channel == "email":
        _send_email(
            event
        )

        return

    if channel == "webhook":
        _send_webhook(
            event
        )

        return

    raise RuntimeError(
        f"Unknown notification channel: {channel}"
    )


def _deliver_with_retries(
    *,
    event: AppointmentEvent,
    channels: list[str],
) -> tuple[
    set[str],
    dict[str, str],
]:
    """
    Retry each configured external channel independently.

    One broken channel does not prevent another channel from working.
    """

    max_attempts = max(
        1,
        int(
            os.getenv(
                "APPOINTMENT_NOTIFICATION_ATTEMPTS",
                "3",
            )
        ),
    )

    base_delay = max(
        0.0,
        float(
            os.getenv(
                "APPOINTMENT_NOTIFICATION_RETRY_SECONDS",
                "2",
            )
        ),
    )

    delivered: set[str] = set()

    errors: dict[
        str,
        str,
    ] = {}

    for channel in channels:
        for attempt in range(
            1,
            max_attempts + 1,
        ):
            try:
                logger.warning(
                    "Sending appointment notification. "
                    "Channel=%s | Attempt=%s/%s | Event=%s",
                    channel,
                    attempt,
                    max_attempts,
                    event.event_id,
                )

                _deliver_channel(
                    channel=channel,
                    event=event,
                )

            except Exception as exc:
                errors[
                    channel
                ] = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                logger.exception(
                    "Appointment notification attempt failed. "
                    "Channel=%s | Attempt=%s/%s | Event=%s",
                    channel,
                    attempt,
                    max_attempts,
                    event.event_id,
                )

                if attempt < max_attempts:
                    time.sleep(
                        base_delay
                        * attempt
                    )

                continue

            delivered.add(
                channel
            )

            errors.pop(
                channel,
                None,
            )

            logger.warning(
                "Appointment notification delivered. "
                "Channel=%s | Event=%s",
                channel,
                event.event_id,
            )

            break

    return (
        delivered,
        errors,
    )


def _mark_delivered(
    *,
    outbox_path: Path,
    payload: dict[str, Any],
) -> None:
    DELIVERED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    delivered_path = (
        DELIVERED_DIR
        / outbox_path.name
    )

    payload[
        "delivered_at"
    ] = _utc_now_iso()

    _atomic_write_json(
        delivered_path,
        payload,
    )

    try:
        outbox_path.unlink()

    except FileNotFoundError:
        pass


def _attempt_outbox_delivery(
    path: Path,
) -> bool:
    payload = _load_outbox_item(
        path
    )

    event_data = payload[
        "event"
    ]

    event = AppointmentEvent(
        **event_data
    )

    channels = (
        _configured_channels()
    )

    if not channels:
        payload[
            "last_attempt_at"
        ] = _utc_now_iso()

        payload[
            "last_error"
        ] = (
            "No external notification channels "
            "are configured."
        )

        _save_outbox_item(
            path,
            payload,
        )

        logger.error(
            "APPOINTMENT DETECTED, but no external "
            "notification channel is configured. "
            "Event=%s",
            event.event_id,
        )

        return False

    payload[
        "delivery_attempts"
    ] = (
        int(
            payload.get(
                "delivery_attempts",
                0,
            )
        )
        + 1
    )

    payload[
        "last_attempt_at"
    ] = _utc_now_iso()

    delivered, errors = (
        _deliver_with_retries(
            event=event,
            channels=channels,
        )
    )

    payload[
        "delivered_channels"
    ] = sorted(
        delivered
    )

    payload[
        "channel_errors"
    ] = errors

    #
    # Reliability rule:
    #
    # Keep the item pending until EVERY configured channel succeeds.
    #
    # This means email can succeed while a webhook is down, and the
    # pending outbox item will retain the failed webhook for later.
    #
    all_delivered = (
        set(
            channels
        )
        <= delivered
    )

    if all_delivered:
        payload[
            "last_error"
        ] = None

        _mark_delivered(
            outbox_path=path,
            payload=payload,
        )

        return True

    payload[
        "last_error"
    ] = (
        "; ".join(
            f"{channel}: {error}"
            for channel, error
            in errors.items()
        )
        or "Some configured notification channels failed."
    )

    _save_outbox_item(
        path,
        payload,
    )

    return False


def flush_pending_notifications() -> int:
    """
    Retry anything that survived in the durable outbox.

    Safe to call frequently.
    """

    OUTBOX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending = sorted(
        OUTBOX_DIR.glob(
            "*.json"
        )
    )

    delivered_count = 0

    for path in pending:
        try:
            if _attempt_outbox_delivery(
                path
            ):
                delivered_count += 1

        except Exception:
            #
            # Absolutely nothing in notification processing is
            # permitted to crash the scraper.
            #
            logger.exception(
                "Unexpected failure while retrying "
                "notification outbox item=%s",
                path,
            )

    return delivered_count


def notify_admin(
    message: str,
    *,
    page_url: str,
    visa_sub_type: str | None = None,
) -> None:
    """
    Public appointment-found API used by the scraper.

    IMPORTANT:
    This function never intentionally raises into the scraper.
    """

    try:
        event = _build_event(
            event_type=(
                "appointment_available"
            ),
            message=message,
            page_url=page_url,
            visa_sub_type=visa_sub_type,
        )

        #
        # 1. Permanent audit log first.
        #
        _append_event_log(
            event
        )

        logger.critical(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
            "APPOINTMENT AVAILABLE "
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        )

        logger.critical(
            "Appointment detected. "
            "Visa subtype=%s | URL=%s | Event=%s",
            visa_sub_type,
            page_url,
            event.event_id,
        )

        #
        # 2. Durable outbox before network I/O.
        #
        outbox_path = (
            _create_outbox_item(
                event
            )
        )

        logger.warning(
            "Appointment alert stored in durable outbox. "
            "Event=%s | File=%s",
            event.event_id,
            outbox_path,
        )

        #
        # 3. Immediate delivery attempt.
        #
        delivered = (
            _attempt_outbox_delivery(
                outbox_path
            )
        )

        if delivered:
            logger.critical(
                "Appointment alert successfully delivered "
                "through every configured channel. "
                "Event=%s",
                event.event_id,
            )

        else:
            logger.critical(
                "APPOINTMENT ALERT STILL PENDING DELIVERY. "
                "It remains stored in the notification outbox. "
                "Event=%s",
                event.event_id,
            )

    except Exception:
        #
        # Last line of defence.
        #
        # A bug in notification infrastructure must never turn a
        # real appointment result into ScraperStatus.FAILED.
        #
        logger.exception(
            "CRITICAL: notification subsystem encountered "
            "an unexpected failure after appointment detection. "
            "The scraper will continue."
        )


def log_no_appointment(
    *,
    page_url: str,
    visa_sub_type: str | None = None,
) -> None:
    try:
        event = _build_event(
            event_type="no_appointment",
            message=(
                "No appointment found during this form check."
            ),
            page_url=page_url,
            visa_sub_type=visa_sub_type,
        )

        _append_event_log(
            event
        )

        logger.info(
            "No appointment found. "
            "Visa subtype=%s | Event=%s",
            visa_sub_type,
            event.event_id,
        )

        #
        # Every successful scraper check is also an opportunity
        # to retry an old appointment alert whose delivery failed.
        #
        flush_pending_notifications()

    except Exception:
        logger.exception(
            "Failed to record no-appointment event. "
            "Continuing scraper."
        )


def send_test_notification() -> None:
    """
    Exercise the exact real appointment-delivery path without
    requiring an actual BLS appointment.
    """

    logger.warning(
        "Sending TEST appointment notification."
    )

    notify_admin(
        (
            "TEST notification generated manually. "
            "This does NOT mean an appointment exists."
        ),
        page_url=(
            "https://example.invalid/"
            "bls-notification-test"
        ),
        visa_sub_type=(
            "TEST VISA SUBTYPE"
        ),
    )


def _configure_cli_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(message)s"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test and manage appointment notifications."
        )
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Send a fake appointment notification "
            "through the real notification pipeline."
        ),
    )

    parser.add_argument(
        "--flush",
        action="store_true",
        help=(
            "Retry every pending appointment notification."
        ),
    )

    args = parser.parse_args()

    _configure_cli_logging()

    if args.test:
        send_test_notification()
        return

    if args.flush:
        delivered = (
            flush_pending_notifications()
        )

        logger.info(
            "Outbox flush finished. "
            "Fully delivered=%s",
            delivered,
        )

        return

    parser.print_help()


if __name__ == "__main__":
    main()
