from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class ScraperStatus(str, Enum):
    APPOINTMENT_FOUND = "appointment_found"
    NO_APPOINTMENT = "no_appointment"
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

    @property
    def succeeded(self) -> bool:
        return self.status is not ScraperStatus.FAILED

    @property
    def appointment_found(self) -> bool:
        return self.status is ScraperStatus.APPOINTMENT_FOUND

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()
