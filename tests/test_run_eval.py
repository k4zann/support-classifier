import asyncio
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors

from app.classifier import Classifier, ClassifierError
from app.config import Settings
from app.schemas import ClassifyResponse, ModelOutput
from evals.run_eval import compute_metrics, evaluate, main, save_result
from tests.fakes import api_error

CATEGORIES = ["auth", "payment", "delivery", "technical", "other"]
URGENCIES = ["high", "medium", "low"]


class FakeClassifier:
    """Stands in for app.classifier.Classifier; records every text it is given.

    `outcomes` maps a text to the (category, urgency) to return, or to an
    exception to raise.
    """

    def __init__(self, outcomes: dict[str, tuple[str, str] | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    async def classify(self, text: str) -> ClassifyResponse:
        self.calls.append(text)
        outcome = self.outcomes[text]
        if isinstance(outcome, Exception):
            raise outcome
        category, urgency = outcome
        return ClassifyResponse(category=category, urgency=urgency, summary="кратко")


def row(row_id: int, category: str, urgency: str, hard: bool = False) -> dict:
    return {
        "id": row_id,
        "text": f"обращение {row_id}",
        "category": category,
        "urgency": urgency,
        "hard": hard,
    }


# Six rows covering every outcome: right, half right, wrong, and two failures.
ROWS = [
    row(1, "auth", "high", hard=True),
    row(2, "payment", "medium"),
    row(3, "delivery", "low", hard=True),
    row(4, "technical", "high", hard=True),
    row(5, "other", "low"),
    row(6, "other", "medium"),
]
FAKE = {
    "обращение 1": ("auth", "high"),
    "обращение 2": ("payment", "low"),
    "обращение 3": ("auth", "low"),
    "обращение 4": ClassifierError("model call failed"),
    "обращение 5": RuntimeError("unexpected"),
    "обращение 6": ("other", "medium"),
}


@pytest.fixture
def metrics() -> dict:
    return compute_metrics(asyncio.run(evaluate(ROWS, FakeClassifier(FAKE))))


def test_every_row_is_classified_in_order_despite_failures() -> None:
    fake = FakeClassifier(FAKE)

    results = asyncio.run(evaluate(ROWS, fake))

    assert fake.calls == [r["text"] for r in ROWS]
    assert [r.id for r in results] == [1, 2, 3, 4, 5, 6]
    assert all(r.latency_s >= 0 for r in results)


def test_accuracy_counts_failed_rows_as_wrong(metrics: dict) -> None:
    # Correct category: rows 1, 2, 6. Correct urgency: rows 1, 3, 6.
    assert metrics["category_accuracy"] == {"correct": 3, "total": 6, "accuracy": 0.5}
    assert metrics["urgency_accuracy"] == {"correct": 3, "total": 6, "accuracy": 0.5}


def test_hard_only_accuracy_includes_failed_hard_row(metrics: dict) -> None:
    # Hard rows 1, 3, 4; row 4 failed but stays in the denominator.
    assert metrics["hard_category_accuracy"] == {
        "correct": 1,
        "total": 3,
        "accuracy": pytest.approx(1 / 3),
    }
    assert metrics["hard_urgency_accuracy"] == {
        "correct": 2,
        "total": 3,
        "accuracy": pytest.approx(2 / 3),
    }


def test_confusion_matrix_cells(metrics: dict) -> None:
    matrix = metrics["confusion_matrix"]

    assert list(matrix) == CATEGORIES
    assert all(list(cells) == CATEGORIES + ["error"] for cells in matrix.values())
    assert matrix["auth"]["auth"] == 1
    assert matrix["payment"]["payment"] == 1
    assert matrix["delivery"]["auth"] == 1
    assert matrix["delivery"]["delivery"] == 0
    assert matrix["technical"]["error"] == 1
    assert matrix["other"]["other"] == 1
    assert matrix["other"]["error"] == 1
    assert sum(sum(cells.values()) for cells in matrix.values()) == len(ROWS)


def test_error_list_entries(metrics: dict) -> None:
    errors = {e["id"]: e for e in metrics["errors"]}

    assert list(errors) == [2, 3, 4, 5]
    assert errors[3] == {
        "id": 3,
        "text": "обращение 3",
        "expected_category": "delivery",
        "expected_urgency": "low",
        "predicted_category": "auth",
        "predicted_urgency": "low",
        "error": None,
    }
    assert errors[4]["predicted_category"] is None
    assert errors[4]["predicted_urgency"] is None
    assert "model call failed" in errors[4]["error"]
    assert "unexpected" in errors[5]["error"]


# --- CLI ---


def make_dataset_rows() -> list[dict]:
    # Smallest valid set: 30 rows, 6 per category, the first 8 marked hard.
    return [row(i, CATEGORIES[i % 5], URGENCIES[i % 3], hard=i < 8) for i in range(30)]


def write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "dataset.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The validator's v3 few-shot path is cwd-relative; keep the repo out of it.
    # Same for .env and the default prompt: Settings sees only what tests set.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "classify.md").write_text("live prompt", encoding="utf-8")


def test_cli_prints_metrics_with_one_failed_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = make_dataset_rows()
    outcomes = {r["text"]: (r["category"], r["urgency"]) for r in rows}
    outcomes[rows[10]["text"]] = ClassifierError("model call failed")
    fake = FakeClassifier(outcomes)
    path = write_dataset(tmp_path, rows)

    exit_code = main(["--dataset", str(path)], make_classifier=lambda args: fake)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert len(fake.calls) == 30
    assert "category accuracy: 29/30" in out.lower()
    assert "urgency accuracy: 29/30" in out.lower()
    assert "hard category accuracy: 8/8" in out.lower()
    assert "confusion matrix" in out.lower()
    assert "id 10" in out
    assert "model call failed" in out


def test_cli_invalid_dataset_exits_non_zero_without_building_classifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_dataset(tmp_path, make_dataset_rows()[:29])
    built: list[object] = []

    exit_code = main(["--dataset", str(path)], make_classifier=built.append)

    assert exit_code != 0
    assert built == []
    assert "dataset has 29 rows, need at least 30" in capsys.readouterr().out


# --- Saved result file ---

RESULT_NAME = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json")


class FakeGeminiModels:
    """Mimics `client.aio.models` of google-genai: answers each row's text with
    its expected labels, reporting 10 prompt + 5 output + 2 thinking tokens.

    `overrides` maps a text to a different response `text` (e.g. bad JSON).
    `rate_limited` maps a text to how many 429s it gets before an answer.
    """

    def __init__(
        self,
        rows: list[dict],
        overrides: dict[str, str],
        rate_limited: dict[str, int] | None = None,
    ) -> None:
        self.labels = {r["text"]: (r["category"], r["urgency"]) for r in rows}
        self.overrides = overrides
        self.rate_limited = dict(rate_limited or {})

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        if self.rate_limited.get(contents, 0) > 0:
            self.rate_limited[contents] -= 1
            raise api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED")
        usage = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
            thoughts_token_count=2,
            total_token_count=17,
        )
        if contents in self.overrides:
            return SimpleNamespace(
                parsed=None, text=self.overrides[contents], usage_metadata=usage
            )
        category, urgency = self.labels[contents]
        output = ModelOutput(category=category, urgency=urgency, summary="кратко")
        return SimpleNamespace(parsed=output, text=None, usage_metadata=usage)


def read_single_result(results_dir: Path) -> dict:
    [path] = list(results_dir.iterdir())
    assert RESULT_NAME.fullmatch(path.name)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.usefixtures("no_backoff")
def test_cli_saves_result_file_with_run_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "4")
    monkeypatch.setenv("BATCH_CONCURRENCY", "2")
    monkeypatch.setenv("REQUEST_TIMEOUT_S", "12.5")
    rows = make_dataset_rows()
    long_summary = json.dumps(
        {"category": "payment", "urgency": "medium", "summary": "слово " * 40}
    )
    models = FakeGeminiModels(
        rows,
        {rows[1]["text"]: long_summary, rows[10]["text"]: "not json"},
        rate_limited={rows[5]["text"]: 2},
    )
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    dataset = write_dataset(tmp_path, rows)
    results_dir = tmp_path / "results"

    exit_code = main(
        ["--dataset", str(dataset)],
        make_classifier=lambda settings: Classifier(client, "prompt", settings),
        results_dir=results_dir,
    )

    assert exit_code == 0
    result = read_single_result(results_dir)
    prompt_hash = hashlib.sha256(b"live prompt").hexdigest()
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert result["model"] == Settings().gemini_model
    assert result["temperature"] == 0.0
    assert result["prompt_path"] == "prompts/classify.md"
    assert result["prompt_sha256"] == prompt_hash
    assert result["dataset_sha256"] == dataset_hash
    assert result["dataset_rows"] == 30
    assert result["timestamp"]
    # Run conditions as used, so latency can be told apart from 429 backoff.
    assert result["max_retries"] == 4
    assert result["batch_concurrency"] == 2
    assert result["request_timeout_s"] == 12.5
    assert result["category_accuracy"] == {
        "correct": 29,
        "total": 30,
        "accuracy": pytest.approx(29 / 30),
    }
    assert result["hard_urgency_accuracy"]["total"] == 8
    assert result["confusion_matrix"]["auth"]["error"] == 1
    assert [e["id"] for e in result["errors"]] == [10]
    assert "invalid model output" in result["errors"][0]["error"]
    predictions = result["predictions"]
    assert [p["id"] for p in predictions] == list(range(30))
    assert predictions[1]["predicted_category"] == "payment"
    assert len(predictions[1]["summary"]) <= 150
    # Row 5 got two 429s before its answer; the invalid-output row 10 is not retried.
    assert [p["attempts"] for p in predictions] == [3 if i == 5 else 1 for i in range(30)]
    assert result["latency_s"]["total"] == pytest.approx(
        sum(p["latency_s"] for p in predictions)
    )
    assert result["latency_s"]["mean"] == pytest.approx(
        result["latency_s"]["total"] / 30
    )
    # The failed row's call still returned a response, so its tokens count too.
    assert result["token_usage"] == {
        "prompt_tokens": 300,
        "output_tokens": 150,
        "thinking_tokens": 60,
        "total_tokens": 510,
    }
    assert result["summaries_trimmed"] == 1
    out = capsys.readouterr().out
    assert str(results_dir) in out
    assert prompt_hash[:12] in out
    assert dataset_hash[:12] in out


def test_hashes_describe_the_files_as_they_were_when_the_run_started(
    tmp_path: Path,
) -> None:
    rows = make_dataset_rows()
    dataset = write_dataset(tmp_path, rows)
    prompt = tmp_path / "prompts" / "classify.md"
    original_dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()

    class EditingClassifier(FakeClassifier):
        # Someone edits the prompt and the dataset while the run is going.
        async def classify(self, text: str) -> ClassifyResponse:
            prompt.write_text("edited prompt", encoding="utf-8")
            dataset.write_text("edited dataset\n", encoding="utf-8")
            return await super().classify(text)

    fake = EditingClassifier({r["text"]: (r["category"], r["urgency"]) for r in rows})
    results_dir = tmp_path / "results"

    main(
        ["--dataset", str(dataset)],
        make_classifier=lambda settings: fake,
        results_dir=results_dir,
    )

    result = read_single_result(results_dir)
    assert result["prompt_sha256"] == hashlib.sha256(b"live prompt").hexdigest()
    assert result["dataset_sha256"] == original_dataset_hash


def test_second_run_with_the_same_timestamp_does_not_overwrite_the_first(
    tmp_path: Path,
) -> None:
    started = datetime(2026, 11, 2, 18, 15, 17)
    results_dir = tmp_path / "results"

    first = save_result(results_dir, started, {"run": 1})
    second = save_result(results_dir, started, {"run": 2})

    assert first.name == "2026-11-02_18-15-17.json"
    assert second.name == "2026-11-02_18-15-17_2.json"
    assert json.loads(first.read_text(encoding="utf-8")) == {"run": 1}
    assert json.loads(second.read_text(encoding="utf-8")) == {"run": 2}
    assert sorted(p.name for p in results_dir.iterdir()) == [first.name, second.name]


@pytest.mark.usefixtures("no_backoff")
def test_result_file_counts_attempts_of_a_row_that_never_got_an_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_RETRIES", "4")
    rows = make_dataset_rows()
    models = FakeGeminiModels(rows, {}, rate_limited={rows[3]["text"]: 99})
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    results_dir = tmp_path / "results"

    main(
        ["--dataset", str(write_dataset(tmp_path, rows))],
        make_classifier=lambda settings: Classifier(client, "prompt", settings),
        results_dir=results_dir,
    )

    predictions = read_single_result(results_dir)["predictions"]
    assert "model call failed" in predictions[3]["error"]
    assert predictions[3]["attempts"] == 4
    assert predictions[4]["attempts"] == 1


def test_cli_flags_override_config_for_that_run_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "configured-model")
    monkeypatch.setenv("TEMPERATURE", "0.1")
    v1 = tmp_path / "prompts" / "v1.md"
    v1.write_text("frozen v1 prompt", encoding="utf-8")
    rows = make_dataset_rows()
    dataset = write_dataset(tmp_path, rows)
    fake = FakeClassifier({r["text"]: (r["category"], r["urgency"]) for r in rows})
    seen: list[Settings] = []

    def make_classifier(settings: Settings) -> FakeClassifier:
        seen.append(settings)
        return fake

    main(
        ["--dataset", str(dataset)],
        make_classifier=make_classifier,
        results_dir=tmp_path / "default",
    )
    main(
        [
            "--dataset",
            str(dataset),
            "--prompt",
            str(v1),
            "--model",
            "flag-model",
            "--temperature",
            "0.7",
        ],
        make_classifier=make_classifier,
        results_dir=tmp_path / "flags",
    )

    default, flags = seen
    assert (default.gemini_model, default.temperature) == ("configured-model", 0.1)
    assert default.prompt_path == "prompts/classify.md"
    assert (flags.gemini_model, flags.temperature) == ("flag-model", 0.7)
    assert flags.prompt_path == str(v1)

    default_result = read_single_result(tmp_path / "default")
    flags_result = read_single_result(tmp_path / "flags")
    assert default_result["model"] == "configured-model"
    assert default_result["temperature"] == 0.1
    assert flags_result["model"] == "flag-model"
    assert flags_result["temperature"] == 0.7
    assert flags_result["prompt_path"] == str(v1)
    assert flags_result["prompt_sha256"] == hashlib.sha256(v1.read_bytes()).hexdigest()
    # Comparable runs: same dataset, different prompts.
    assert default_result["dataset_sha256"] == flags_result["dataset_sha256"]
    assert default_result["prompt_sha256"] != flags_result["prompt_sha256"]
    # The fake classifier reports no token usage or attempts, so none is stored.
    assert flags_result["token_usage"] is None
    assert {p["attempts"] for p in flags_result["predictions"]} == {None}
