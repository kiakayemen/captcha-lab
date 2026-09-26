from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ScraperStatus(str, Enum):
    APPOINTMENT_FOUND = "appointment_found"
    POSSIBLE_APPOINTMENT = "possible_appointment"
    NO_APPOINTMENT = "no_appointment"
    SERVER_ERROR = "server_error"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class ScraperConfig:
    headless: bool = False
    gpu: bool = False
    output_dir: Path = Path("output/live_solver")
    visa_sub_types: tuple[str, ...] = (
        "Student Visa",
        "Non-Working Residence Visa",
    )
    allow_single_proxy: bool = False
    direct_connection: bool = False
    proxy_urls: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ScraperResult:
    status: ScraperStatus
    started_at: datetime
    finished_at: datetime

    page_url: str | None = None
    visa_sub_type: str | None = None

    error_type: str | None = None
    error_message: str | None = None
    failure_screenshot: Path | None = None
    first_failure: dict[str, Any] | None = None
    attempt_failures: tuple[dict[str, Any], ...] = ()
    terminal_failure: dict[str, Any] | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {
            ScraperStatus.APPOINTMENT_FOUND,
            ScraperStatus.POSSIBLE_APPOINTMENT,
            ScraperStatus.NO_APPOINTMENT,
        }

    @property
    def appointment_found(self) -> bool:
        return self.status is ScraperStatus.APPOINTMENT_FOUND

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()
