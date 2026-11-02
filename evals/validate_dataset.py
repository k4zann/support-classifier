"""Check the hand-written eval set before any model run.

Usage: uv run python -m evals.validate_dataset [--dataset PATH] [--few-shot PATH]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from app.schemas import Category, Urgency

DATASET_PATH = Path("evals/dataset.jsonl")
# v3's few-shot examples must not leak into the eval set. Each example text is
# one line in the prompt file that starts with "Text:".
FEW_SHOT_PATH = Path("prompts/versions/v3.md")
FEW_SHOT_PREFIX = "Text:"

MIN_ROWS = 30
MIN_PER_CATEGORY = 4
MIN_HARD = 8

CATEGORIES = [c.value for c in Category]
URGENCIES = [u.value for u in Urgency]


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


def load_rows(path: Path) -> tuple[list[tuple[int, dict]], list[str]]:
    """Return (line number, row) for every JSON object line, plus parse violations."""
    rows: list[tuple[int, dict]] = []
    violations: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            violations.append(f"line {line_no}: not valid JSON")
            continue
        if not isinstance(row, dict):
            violations.append(f"line {line_no}: expected a JSON object")
            continue
        rows.append((line_no, row))
    return rows, violations


def _few_shot_texts(path: Path) -> list[str]:
    return [
        line.strip()[len(FEW_SHOT_PREFIX) :].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(FEW_SHOT_PREFIX)
    ]


def _row_violations(where: str, row: dict) -> list[str]:
    violations = []
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        violations.append(f"{where}: text must be a non-empty string")
    for field, allowed in (("category", CATEGORIES), ("urgency", URGENCIES)):
        if row.get(field) not in allowed:
            violations.append(
                f"{where}: {field} {row.get(field)!r} is not one of {', '.join(allowed)}"
            )
    if not isinstance(row.get("hard", False), bool):
        violations.append(f"{where}: hard must be true or false")
    return violations


def validate(path: Path = DATASET_PATH, few_shot_path: Path = FEW_SHOT_PATH) -> list[str]:
    """Return every violation in the dataset; an empty list means it is valid."""
    if not path.is_file():
        return [f"dataset file not found: {path}"]

    rows, violations = load_rows(path)

    first_line_by_id: dict[int, int] = {}
    for line_no, row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, int) or isinstance(row_id, bool):
            violations.append(f"line {line_no}: id must be an integer")
            where = f"line {line_no}"
        elif row_id in first_line_by_id:
            violations.append(
                f"line {line_no}: duplicate id {row_id} "
                f"(first on line {first_line_by_id[row_id]})"
            )
            where = f"line {line_no}"
        else:
            first_line_by_id[row_id] = line_no
            where = f"row id {row_id}"
        violations.extend(_row_violations(where, row))

    if len(rows) < MIN_ROWS:
        violations.append(f"dataset has {len(rows)} rows, need at least {MIN_ROWS}")
    per_category = Counter(row.get("category") for _, row in rows)
    for category in CATEGORIES:
        if per_category[category] < MIN_PER_CATEGORY:
            violations.append(
                f"category {category!r} has {per_category[category]} rows, "
                f"need at least {MIN_PER_CATEGORY}"
            )
    hard = sum(row.get("hard") is True for _, row in rows)
    if hard < MIN_HARD:
        violations.append(f"only {hard} hard rows, need at least {MIN_HARD}")

    # v3 does not exist until its prompt version is written; skip the check until then.
    if few_shot_path.is_file():
        few_shot = {_normalise(text) for text in _few_shot_texts(few_shot_path)}
        if not few_shot:
            violations.append(
                f"no few-shot examples ('{FEW_SHOT_PREFIX} ...' lines) "
                f"found in {few_shot_path}"
            )
        for _, row in rows:
            text = row.get("text")
            if isinstance(text, str) and _normalise(text) in few_shot:
                violations.append(
                    f"row id {row.get('id')}: text matches a few-shot example "
                    f"in {few_shot_path}"
                )

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the eval dataset.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--few-shot", type=Path, default=FEW_SHOT_PATH)
    args = parser.parse_args(argv)

    violations = validate(args.dataset, args.few_shot)
    if violations:
        print(f"{args.dataset} is invalid:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    rows = [row for _, row in load_rows(args.dataset)[0]]
    per_category = Counter(row["category"] for row in rows)
    per_urgency = Counter(row["urgency"] for row in rows)
    print(f"{args.dataset} is valid")
    print(f"rows: {len(rows)}")
    print(f"hard: {sum(row.get('hard') is True for row in rows)}")
    print("category:")
    for category in CATEGORIES:
        print(f"  {category}: {per_category[category]}")
    print("urgency:")
    for urgency in URGENCIES:
        print(f"  {urgency}: {per_urgency[urgency]}")
    if not args.few_shot.is_file():
        print(f"few-shot overlap check skipped: {args.few_shot} not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
