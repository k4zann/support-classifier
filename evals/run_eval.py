"""Classify every eval row with the service's classifier, print the metrics and
save the run to evals/results/<YYYY-MM-DD_HH-MM-SS>.json.

Usage: uv run python -m evals.run_eval [--dataset PATH] [--prompt PATH]
       [--model NAME] [--temperature FLOAT]
"""

import argparse
import asyncio
import hashlib
import itertools
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from app.classifier import Classifier, build_client
from app.config import Settings, load_prompt
from app.schemas import Category, ClassifyResponse
from evals.validate_dataset import DATASET_PATH, load_rows, validate

CATEGORIES = [c.value for c in Category]
RESULTS_DIR = Path("evals/results")
# Confusion-matrix column for rows whose classify call failed.
ERROR_COLUMN = "error"


class SupportsClassify(Protocol):
    async def classify(self, text: str) -> ClassifyResponse: ...


@dataclass
class RowResult:
    """One row's outcome; predicted labels are None when the call failed."""

    id: int
    text: str
    hard: bool
    expected_category: str
    expected_urgency: str
    predicted_category: str | None
    predicted_urgency: str | None
    summary: str | None
    error: str | None
    latency_s: float
    # Model calls for this row, retries included; None if the classifier
    # does not count them (test fakes).
    attempts: int | None


def _describe(exc: Exception) -> str:
    # ClassifierError's own message is generic; the chained cause says why
    # (e.g. a 429 quota error), which is what the student needs to see here.
    message = f"{type(exc).__name__}: {exc}"
    if exc.__cause__ is not None:
        message += f" ({type(exc.__cause__).__name__}: {exc.__cause__})"
    return message


async def evaluate(rows: list[dict], classifier: SupportsClassify) -> list[RowResult]:
    """Classify rows one at a time; a failing row is recorded and the run goes on."""
    results = []
    for row in rows:
        predicted: ClassifyResponse | None = None
        error = None
        start = time.perf_counter()
        try:
            predicted = await classifier.classify(row["text"])
        except Exception as exc:  # noqa: BLE001 - ClassifierError or anything else
            error = _describe(exc)
        latency = time.perf_counter() - start
        results.append(
            RowResult(
                id=row["id"],
                text=row["text"],
                hard=row.get("hard") is True,
                expected_category=row["category"],
                expected_urgency=row["urgency"],
                predicted_category=predicted.category.value if predicted else None,
                predicted_urgency=predicted.urgency.value if predicted else None,
                summary=predicted.summary if predicted else None,
                error=error,
                latency_s=latency,
                attempts=getattr(classifier, "last_attempts", None),
            )
        )
    return results


def _accuracy(pairs: list[tuple[str, str | None]]) -> dict:
    correct = sum(expected == predicted for expected, predicted in pairs)
    total = len(pairs)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else None,
    }


def compute_metrics(results: list[RowResult]) -> dict:
    """Accuracies, confusion matrix and error list. Failed rows count as wrong."""
    hard = [r for r in results if r.hard]
    matrix = {
        expected: dict.fromkeys([*CATEGORIES, ERROR_COLUMN], 0)
        for expected in CATEGORIES
    }
    for r in results:
        matrix[r.expected_category][r.predicted_category or ERROR_COLUMN] += 1
    errors = [
        {
            "id": r.id,
            "text": r.text,
            "expected_category": r.expected_category,
            "expected_urgency": r.expected_urgency,
            "predicted_category": r.predicted_category,
            "predicted_urgency": r.predicted_urgency,
            "error": r.error,
        }
        for r in results
        if r.error
        or (r.predicted_category, r.predicted_urgency)
        != (r.expected_category, r.expected_urgency)
    ]
    return {
        "category_accuracy": _accuracy(
            [(r.expected_category, r.predicted_category) for r in results]
        ),
        "urgency_accuracy": _accuracy(
            [(r.expected_urgency, r.predicted_urgency) for r in results]
        ),
        "hard_category_accuracy": _accuracy(
            [(r.expected_category, r.predicted_category) for r in hard]
        ),
        "hard_urgency_accuracy": _accuracy(
            [(r.expected_urgency, r.predicted_urgency) for r in hard]
        ),
        "confusion_matrix": matrix,
        "errors": errors,
    }


def print_report(metrics: dict) -> None:
    for name in ("category", "urgency", "hard category", "hard urgency"):
        acc = metrics[f"{name.replace(' ', '_')}_accuracy"]
        percent = f"{acc['accuracy']:.1%}" if acc["accuracy"] is not None else "n/a"
        print(
            f"{name.capitalize()} accuracy: {acc['correct']}/{acc['total']} ({percent})"
        )

    columns = [*CATEGORIES, ERROR_COLUMN]
    width = max(len(c) for c in columns) + 1
    print("\nConfusion matrix (rows: expected, columns: predicted)")
    print("".rjust(width) + "".join(c.rjust(width) for c in columns))
    for expected, cells in metrics["confusion_matrix"].items():
        print(
            expected.rjust(width) + "".join(str(cells[c]).rjust(width) for c in columns)
        )

    print(f"\nErrors ({len(metrics['errors'])}):")
    for e in metrics["errors"]:
        got = e["error"] or f"{e['predicted_category']}/{e['predicted_urgency']}"
        print(
            f"- id {e['id']}: expected {e['expected_category']}/{e['expected_urgency']}, got {got}"
        )
        print(f"  text: {e['text']}")


class TrimCounter(logging.Handler):
    """Counts the classifier's "summary trimmed" warnings during a run."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("summary trimmed"):
            self.count += 1


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_settings(args: argparse.Namespace) -> Settings:
    """Settings from .env, with the command-line overrides applied for this run."""
    overrides = {
        "prompt_path": args.prompt,
        "gemini_model": args.model,
        "temperature": args.temperature,
    }
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def build_classifier(settings: Settings) -> Classifier:
    """The real classifier, the same one the service uses."""
    prompt = load_prompt(settings.prompt_path)
    return Classifier(build_client(settings), prompt, settings)


def save_result(results_dir: Path, started: datetime, result: dict) -> Path:
    """Write the run to <results_dir>/<YYYY-MM-DD_HH-MM-SS>.json.

    An existing file is never overwritten: a second run started in the same
    second gets <...>_2.json, then _3 and so on. Written to a .tmp file first,
    so an interrupted write leaves no half-file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=results_dir, suffix=".tmp")
    tmp = Path(tmp_name)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2))
    stem = f"{started:%Y-%m-%d_%H-%M-%S}"
    try:
        for n in itertools.count(1):
            path = results_dir / (f"{stem}.json" if n == 1 else f"{stem}_{n}.json")
            try:
                # Unlike rename, link fails if the name is taken, even between
                # two processes racing for the same name.
                os.link(tmp, path)
                return path
            except FileExistsError:
                continue
    finally:
        tmp.unlink()


def main(
    argv: list[str] | None = None,
    make_classifier: Callable[[Settings], SupportsClassify] = build_classifier,
    results_dir: Path = RESULTS_DIR,
) -> int:
    parser = argparse.ArgumentParser(
        description="Run the eval set through the classifier."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--prompt", help="prompt file (default: from Settings)")
    parser.add_argument("--model", help="Gemini model (default: from Settings)")
    parser.add_argument("--temperature", type=float, help="default: from Settings")
    args = parser.parse_args(argv)

    # Validate before building the classifier, so a bad dataset never costs a call.
    violations = validate(args.dataset)
    if violations:
        print(f"{args.dataset} is invalid, not running:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    rows = [row for _, row in load_rows(args.dataset)[0]]
    settings = build_settings(args)
    classifier = make_classifier(settings)
    # Hashed before classifying, so an edit during the run can't change them.
    prompt_hash = sha256_of(Path(settings.prompt_path))
    dataset_hash = sha256_of(args.dataset)
    started = datetime.now()
    trims = TrimCounter()
    classifier_logger = logging.getLogger("app.classifier")
    classifier_logger.addHandler(trims)
    try:
        results = asyncio.run(evaluate(rows, classifier))
    finally:
        classifier_logger.removeHandler(trims)
    metrics = compute_metrics(results)
    print_report(metrics)

    total_latency = sum(r.latency_s for r in results)
    path = save_result(
        results_dir,
        started,
        {
            "timestamp": started.isoformat(timespec="seconds"),
            "model": settings.gemini_model,
            "temperature": settings.temperature,
            "prompt_path": settings.prompt_path,
            "prompt_sha256": prompt_hash,
            "dataset_path": str(args.dataset),
            "dataset_sha256": dataset_hash,
            "dataset_rows": len(rows),
            # Run conditions: latency includes retry backoff, see per-row attempts.
            "max_retries": settings.max_retries,
            "batch_concurrency": settings.batch_concurrency,
            "request_timeout_s": settings.request_timeout_s,
            **metrics,
            "latency_s": {"total": total_latency, "mean": total_latency / len(rows)},
            # Only the real Classifier counts tokens; test fakes may not.
            "token_usage": getattr(classifier, "token_usage", None),
            "summaries_trimmed": trims.count,
            "predictions": [asdict(r) for r in results],
        },
    )
    print(
        f"\nSaved {path} (model {settings.gemini_model}, "
        f"prompt {prompt_hash[:12]}, dataset {dataset_hash[:12]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
