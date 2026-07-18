from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_RESULTS_PATH = Path("preprocessing_results.csv")
DEFAULT_VARIANT_LEADERBOARD_PATH = Path("preprocessing_leaderboard.csv")
DEFAULT_FUSION_RESULTS_PATH = Path("fusion_results.csv")
DEFAULT_FUSION_LEADERBOARD_PATH = Path("fusion_leaderboard.csv")

# Added once for every extra agreeing variant after the first.
# Example: three agreeing variants receive 2 * AGREEMENT_BONUS.
AGREEMENT_BONUS = 0.25


@dataclass(frozen=True)
class Prediction:
    variant: str
    value: str
    confidence: float


@dataclass(frozen=True)
class TileGroup:
    image: str
    tile: int
    tile_path: str
    ground_truth: str
    predictions: tuple[Prediction, ...]


@dataclass(frozen=True)
class FusionDecision:
    prediction: str
    score: float
    votes: int
    variants: str


FusionStrategy = Callable[
    [tuple[Prediction, ...], dict[str, float]],
    FusionDecision,
]


def is_valid_prediction(value: str) -> bool:
    """A valid OCR result must be exactly three decimal digits."""
    return len(value) == 3 and value.isdigit()


def safe_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def candidate_groups(
    predictions: Iterable[Prediction],
) -> dict[str, list[Prediction]]:
    grouped: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        if is_valid_prediction(prediction.value):
            grouped[prediction.value].append(prediction)
    return dict(grouped)


def empty_decision() -> FusionDecision:
    return FusionDecision("", 0.0, 0, "")


def make_decision(
    candidate: str,
    score: float,
    grouped: dict[str, list[Prediction]],
) -> FusionDecision:
    matches = grouped[candidate]
    variants = ",".join(sorted(item.variant for item in matches))
    return FusionDecision(
        prediction=candidate,
        score=score,
        votes=len(matches),
        variants=variants,
    )


def choose_best(
    grouped: dict[str, list[Prediction]],
    score_fn: Callable[[str, list[Prediction]], float],
) -> FusionDecision:
    """
    Pick the candidate with the highest strategy score.

    Tie-breakers:
    1. More agreeing variants
    2. Higher maximum OCR confidence
    3. Higher summed OCR confidence
    4. Lexicographically smaller prediction for deterministic output
    """
    if not grouped:
        return empty_decision()

    def rank(item: tuple[str, list[Prediction]]) -> tuple[float, int, float, float, str]:
        candidate, matches = item
        confidences = [match.confidence for match in matches]
        return (
            score_fn(candidate, matches),
            len(matches),
            max(confidences, default=0.0),
            sum(confidences),
            # max() is used, so invert the final lexical preference.
            "".join(chr(255 - ord(char)) for char in candidate),
        )

    candidate, matches = max(grouped.items(), key=rank)
    return make_decision(candidate, score_fn(candidate, matches), grouped)


def highest_confidence(
    predictions: tuple[Prediction, ...],
    variant_accuracies: dict[str, float],
) -> FusionDecision:
    del variant_accuracies
    grouped = candidate_groups(predictions)
    return choose_best(
        grouped,
        lambda _candidate, matches: max(
            match.confidence for match in matches
        ),
    )


def majority_vote(
    predictions: tuple[Prediction, ...],
    variant_accuracies: dict[str, float],
) -> FusionDecision:
    del variant_accuracies
    grouped = candidate_groups(predictions)
    return choose_best(
        grouped,
        lambda _candidate, matches: float(len(matches)),
    )


def benchmark_weighted_vote(
    predictions: tuple[Prediction, ...],
    variant_accuracies: dict[str, float],
) -> FusionDecision:
    grouped = candidate_groups(predictions)
    return choose_best(
        grouped,
        lambda _candidate, matches: sum(
            variant_accuracies.get(match.variant, 0.0)
            for match in matches
        ),
    )


def confidence_weighted_vote(
    predictions: tuple[Prediction, ...],
    variant_accuracies: dict[str, float],
) -> FusionDecision:
    del variant_accuracies
    grouped = candidate_groups(predictions)
    return choose_best(
        grouped,
        lambda _candidate, matches: sum(
            match.confidence for match in matches
        ),
    )


def agreement_bonus_vote(
    predictions: tuple[Prediction, ...],
    variant_accuracies: dict[str, float],
) -> FusionDecision:
    del variant_accuracies
    grouped = candidate_groups(predictions)

    def score(_candidate: str, matches: list[Prediction]) -> float:
        confidence_sum = sum(match.confidence for match in matches)
        bonus = AGREEMENT_BONUS * max(0, len(matches) - 1)
        return confidence_sum + bonus

    return choose_best(grouped, score)


STRATEGIES: list[tuple[str, FusionStrategy]] = [
    ("highest_confidence", highest_confidence),
    ("majority_vote", majority_vote),
    ("benchmark_weighted_vote", benchmark_weighted_vote),
    ("confidence_weighted_vote", confidence_weighted_vote),
    ("agreement_bonus_vote", agreement_bonus_vote),
]


def load_variant_accuracies(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing variant leaderboard: {path}\n"
            "Run benchmark.py first."
        )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"variant", "accuracy"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        accuracies: dict[str, float] = {}
        for row in reader:
            variant = (row.get("variant") or "").strip()
            if variant:
                accuracies[variant] = safe_float(row.get("accuracy"))

    if not accuracies:
        raise RuntimeError(f"No variant accuracies found in {path}")

    return accuracies


def load_tile_groups(path: Path) -> list[TileGroup]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing benchmark results: {path}\n"
            "Run benchmark.py first."
        )

    grouped: dict[
        tuple[str, int, str, str],
        list[Prediction],
    ] = defaultdict(list)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image",
            "tile",
            "tile_path",
            "variant",
            "prediction",
            "confidence",
            "ground_truth",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}"
            )

        for row_number, row in enumerate(reader, start=2):
            image = (row.get("image") or "").strip()
            tile_text = (row.get("tile") or "").strip()
            tile_path = (row.get("tile_path") or "").strip()
            ground_truth = (row.get("ground_truth") or "").strip()
            variant = (row.get("variant") or "").strip()
            raw_prediction = (row.get("prediction") or "").strip()

            if not image or not tile_text or not variant or not ground_truth:
                raise ValueError(
                    f"Invalid required value in {path}, row {row_number}"
                )

            try:
                tile = int(tile_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid tile number in {path}, row {row_number}: "
                    f"{tile_text!r}"
                ) from error

            # Register every tile before filtering predictions. This keeps
            # tiles with zero valid three-digit predictions in the evaluation,
            # where they correctly count as no-prediction errors.
            key = (image, tile, tile_path, ground_truth)
            grouped[key]

            if is_valid_prediction(raw_prediction):
                grouped[key].append(
                    Prediction(
                        variant=variant,
                        value=raw_prediction,
                        confidence=safe_float(row.get("confidence")),
                    )
                )

    tile_groups = [
        TileGroup(
            image=key[0],
            tile=key[1],
            tile_path=key[2],
            ground_truth=key[3],
            predictions=tuple(predictions),
        )
        for key, predictions in grouped.items()
    ]
    tile_groups.sort(key=lambda item: (item.image, item.tile))

    if not tile_groups:
        raise RuntimeError(f"No benchmark tiles were found in {path}")

    return tile_groups


def write_fusion_results(
    path: Path,
    tile_groups: list[TileGroup],
    variant_accuracies: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for tile_group in tile_groups:
        valid_prediction_count = len(tile_group.predictions)
        distinct_prediction_count = len(
            {prediction.value for prediction in tile_group.predictions}
        )

        for strategy_name, strategy in STRATEGIES:
            decision = strategy(
                tile_group.predictions,
                variant_accuracies,
            )
            rows.append(
                {
                    "image": tile_group.image,
                    "tile": tile_group.tile,
                    "tile_path": tile_group.tile_path,
                    "ground_truth": tile_group.ground_truth,
                    "strategy": strategy_name,
                    "fusion_prediction": decision.prediction,
                    "fusion_score": f"{decision.score:.6f}",
                    "vote_count": decision.votes,
                    "supporting_variants": decision.variants,
                    "valid_prediction_count": valid_prediction_count,
                    "distinct_prediction_count": distinct_prediction_count,
                    "correct": decision.prediction == tile_group.ground_truth,
                }
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "tile",
                "tile_path",
                "ground_truth",
                "strategy",
                "fusion_prediction",
                "fusion_score",
                "vote_count",
                "supporting_variants",
                "valid_prediction_count",
                "distinct_prediction_count",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def write_fusion_leaderboard(
    path: Path,
    result_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    no_prediction: Counter[str] = Counter()

    for row in result_rows:
        strategy = str(row["strategy"])
        totals[strategy] += 1
        correct[strategy] += int(bool(row["correct"]))
        no_prediction[strategy] += int(not bool(row["fusion_prediction"]))

    leaderboard: list[dict[str, object]] = []
    for order, (strategy_name, _strategy) in enumerate(STRATEGIES):
        total = totals[strategy_name]
        correct_count = correct[strategy_name]
        leaderboard.append(
            {
                "strategy": strategy_name,
                "correct": correct_count,
                "total": total,
                "accuracy": correct_count / total if total else 0.0,
                "wrong": total - correct_count,
                "no_prediction": no_prediction[strategy_name],
                "test_order": order + 1,
            }
        )

    leaderboard.sort(
        key=lambda row: (
            -float(row["accuracy"]),
            int(row["wrong"]),
            int(row["test_order"]),
        )
    )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "strategy",
                "correct",
                "total",
                "accuracy",
                "wrong",
                "no_prediction",
                "test_order",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(leaderboard, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    **row,
                    "accuracy": f"{float(row['accuracy']):.8f}",
                }
            )

    return leaderboard


def print_summary(
    tile_groups: list[TileGroup],
    leaderboard: list[dict[str, object]],
) -> None:
    print(f"Tiles evaluated: {len(tile_groups)}")
    print(f"Agreement bonus: {AGREEMENT_BONUS:.2f}")
    print()

    for rank, row in enumerate(leaderboard, start=1):
        print(
            f"{rank}. {row['strategy']:<26} "
            f"{row['correct']}/{row['total']} "
            f"({float(row['accuracy']):.2%})"
        )

    if leaderboard:
        best = leaderboard[0]
        print()
        print(
            f"Best measured strategy: {best['strategy']} "
            f"({float(best['accuracy']):.2%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark deterministic OCR fusion strategies using the output "
            "from benchmark.py."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Benchmark detail CSV (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--variant-leaderboard",
        type=Path,
        default=DEFAULT_VARIANT_LEADERBOARD_PATH,
        help=(
            "Variant accuracy CSV "
            f"(default: {DEFAULT_VARIANT_LEADERBOARD_PATH})"
        ),
    )
    parser.add_argument(
        "--fusion-results",
        type=Path,
        default=DEFAULT_FUSION_RESULTS_PATH,
        help=f"Per-tile output CSV (default: {DEFAULT_FUSION_RESULTS_PATH})",
    )
    parser.add_argument(
        "--fusion-leaderboard",
        type=Path,
        default=DEFAULT_FUSION_LEADERBOARD_PATH,
        help=(
            "Strategy leaderboard CSV "
            f"(default: {DEFAULT_FUSION_LEADERBOARD_PATH})"
        ),
    )
    args = parser.parse_args()

    variant_accuracies = load_variant_accuracies(
        args.variant_leaderboard
    )
    tile_groups = load_tile_groups(args.results)

    result_rows = write_fusion_results(
        args.fusion_results,
        tile_groups,
        variant_accuracies,
    )
    leaderboard = write_fusion_leaderboard(
        args.fusion_leaderboard,
        result_rows,
    )

    print_summary(tile_groups, leaderboard)
    print()
    print(f"Detailed results: {args.fusion_results}")
    print(f"Leaderboard: {args.fusion_leaderboard}")


if __name__ == "__main__":
    main()
