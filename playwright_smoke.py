from __future__ import annotations

import argparse
from pathlib import Path

from scraper.models import ScraperConfig
from scraper.service import run_scraper


VISA_SUB_TYPES = [
    "Student Visa",
    "Non-Working Residence Visa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check configured BLS visa subtypes "
            "for appointment availability."
        )
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying the browser.",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Ask EasyOCR to use a supported GPU.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/live_solver"),
        help="Solver output directory.",
    )

    parser.add_argument(
        "--visa-sub-types",
        nargs="+",
        choices=VISA_SUB_TYPES,
        default=VISA_SUB_TYPES,
        help="Visa subtypes to try.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ScraperConfig(
        headless=args.headless,
        gpu=args.gpu,
        output_dir=args.output,
        visa_sub_types=tuple(args.visa_sub_types),
    )

    result = run_scraper(config)

    print()
    print("=== Scraper run finished ===")
    print(f"Status: {result.status.value}")
    print(f"Duration: {result.duration_seconds:.2f}s")

    if result.page_url:
        print(f"Final URL: {result.page_url}")

    if result.visa_sub_type:
        print(f"Visa subtype: {result.visa_sub_type}")

    if result.error_message:
        print(
            f"Error: {result.error_type}: "
            f"{result.error_message}"
        )

    if not result.succeeded:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
