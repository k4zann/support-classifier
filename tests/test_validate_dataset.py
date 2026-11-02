import json
from pathlib import Path

import pytest

from evals.validate_dataset import main, validate

CATEGORIES = ["auth", "payment", "delivery", "technical", "other"]
URGENCIES = ["high", "medium", "low"]


def make_rows() -> list[dict]:
    # 30 rows: 6 per category, the first 8 marked hard. Placeholder texts only.
    return [
        {
            "id": i,
            "text": f"тестовое обращение номер {i}",
            "category": CATEGORIES[i % 5],
            "urgency": URGENCIES[i % 3],
            **({"hard": True} if i < 8 else {}),
        }
        for i in range(30)
    ]


def write_dataset(
    tmp_path: Path, rows: list[dict], extra_lines: tuple[str, ...] = ()
) -> Path:
    path = tmp_path / "dataset.jsonl"
    lines = [json.dumps(row, ensure_ascii=False) for row in rows] + list(extra_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def no_few_shot(tmp_path: Path) -> Path:
    return tmp_path / "missing_v3.md"


def test_valid_dataset_has_no_violations(tmp_path: Path, no_few_shot: Path) -> None:
    path = write_dataset(tmp_path, make_rows())

    assert validate(path, no_few_shot) == []


def test_cli_reports_counts_for_valid_dataset(
    tmp_path: Path, no_few_shot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_dataset(tmp_path, make_rows())

    exit_code = main(["--dataset", str(path), "--few-shot", str(no_few_shot)])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "rows: 30" in out
    assert "hard: 8" in out
    for category in CATEGORIES:
        assert f"{category}: 6" in out
    for urgency in URGENCIES:
        assert f"{urgency}: 10" in out


def test_too_few_rows(tmp_path: Path, no_few_shot: Path) -> None:
    path = write_dataset(tmp_path, make_rows()[:29])

    assert "dataset has 29 rows, need at least 30" in validate(path, no_few_shot)


def test_category_under_represented(tmp_path: Path, no_few_shot: Path) -> None:
    rows = make_rows()
    other_rows = [row for row in rows if row["category"] == "other"]
    for row in other_rows[:3]:
        row["category"] = "auth"
    path = write_dataset(tmp_path, rows)

    assert "category 'other' has 3 rows, need at least 4" in validate(path, no_few_shot)


def test_too_few_hard_rows(tmp_path: Path, no_few_shot: Path) -> None:
    rows = make_rows()
    rows[0]["hard"] = False
    path = write_dataset(tmp_path, rows)

    assert "only 7 hard rows, need at least 8" in validate(path, no_few_shot)


def test_duplicate_id(tmp_path: Path, no_few_shot: Path) -> None:
    rows = make_rows()
    rows[5]["id"] = 2
    path = write_dataset(tmp_path, rows)

    assert "line 6: duplicate id 2 (first on line 3)" in validate(path, no_few_shot)


def test_bad_json_line(tmp_path: Path, no_few_shot: Path) -> None:
    path = write_dataset(tmp_path, make_rows(), extra_lines=['{"id": 99, "text": '])

    assert "line 31: not valid JSON" in validate(path, no_few_shot)


@pytest.mark.parametrize(
    ("field", "value"), [("category", "billing"), ("urgency", "critical")]
)
def test_bad_label_names_row_id_and_value(
    tmp_path: Path, no_few_shot: Path, field: str, value: str
) -> None:
    rows = make_rows()
    rows[4][field] = value
    path = write_dataset(tmp_path, rows)

    violations = validate(path, no_few_shot)

    assert any(v.startswith(f"row id 4: {field} '{value}'") for v in violations)


def test_every_violation_is_listed(tmp_path: Path, no_few_shot: Path) -> None:
    rows = make_rows()
    rows[5]["id"] = 2
    rows[4]["category"] = "billing"
    path = write_dataset(tmp_path, rows, extra_lines=["not json"])

    violations = validate(path, no_few_shot)

    assert any("duplicate id 2" in v for v in violations)
    assert any("'billing'" in v for v in violations)
    assert any("not valid JSON" in v for v in violations)


def test_few_shot_text_in_dataset(tmp_path: Path) -> None:
    rows = make_rows()
    rows[7]["text"] = "Не могу  войти в аккаунт,\nкод не приходит"
    path = write_dataset(tmp_path, rows)
    v3 = tmp_path / "v3.md"
    v3.write_text(
        "Examples:\n"
        "Text: не могу войти в аккаунт, код не приходит\n"
        '{"category": "auth", "urgency": "high", "summary": "..."}\n'
        "Text: другой пример, которого нет в наборе\n",
        encoding="utf-8",
    )

    assert f"row id 7: text matches a few-shot example in {v3}" in validate(path, v3)


def test_few_shot_file_without_examples(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, make_rows())
    v3 = tmp_path / "v3.md"
    v3.write_text("Classify the request.\n", encoding="utf-8")

    assert f"no few-shot examples ('Text: ...' lines) found in {v3}" in validate(
        path, v3
    )


def test_missing_dataset_file(
    tmp_path: Path, no_few_shot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "dataset.jsonl"

    exit_code = main(["--dataset", str(path), "--few-shot", str(no_few_shot)])

    assert validate(path, no_few_shot) == [f"dataset file not found: {path}"]
    assert exit_code == 1
    assert f"dataset file not found: {path}" in capsys.readouterr().out


def test_cli_lists_violations_and_fails(
    tmp_path: Path, no_few_shot: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_dataset(tmp_path, make_rows()[:29])

    exit_code = main(["--dataset", str(path), "--few-shot", str(no_few_shot)])

    assert exit_code == 1
    assert "dataset has 29 rows, need at least 30" in capsys.readouterr().out
