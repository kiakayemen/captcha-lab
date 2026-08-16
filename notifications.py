from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json


DEFAULT_EVENT_LOG = Path("output") / "appointment_events.jsonl"


@dataclass(frozen=True)
class AppointmentEvent:
    event_type: str
    message: str
    page_url: str
    visa_sub_type: str | None = None


def append_event(
    event: AppointmentEvent,
    *,
    log_path: Path = DEFAULT_EVENT_LOG,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **asdict(event),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def notify_admin(
    message: str,
    *,
    page_url: str,
    visa_sub_type: str | None = None,
    log_path: Path = DEFAULT_EVENT_LOG,
) -> None:
    event = AppointmentEvent(
        event_type="appointment_available",
        message=message,
        page_url=page_url,
        visa_sub_type=visa_sub_type,
    )
    append_event(event, log_path=log_path)
    print(f"[admin] {message}")
    print(f"[admin] recorded to {log_path}")


def log_no_appointment(
    *,
    page_url: str,
    visa_sub_type: str | None = None,
    log_path: Path = DEFAULT_EVENT_LOG,
) -> None:
    event = AppointmentEvent(
        event_type="no_appointment",
        message="No appointment found during this run.",
        page_url=page_url,
        visa_sub_type=visa_sub_type,
    )
    append_event(event, log_path=log_path)
    print(f"[appointment] no appointment found; recorded to {log_path}")
