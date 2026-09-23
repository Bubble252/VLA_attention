from scripts.eval_capability_suite import (
    atomic_json_write,
    binary_metrics,
    binary_label,
    parse_choice,
    normalize_answer,
    resume_identity,
    score_records,
    vqa_consensus,
)
from scripts.paired_capability_bootstrap import bootstrap as paired_bootstrap


def test_vqa_consensus_uses_leave_one_annotator_out_rule():
    assert vqa_consensus("the cat", ["cat", "cat", "cat", "dog"] + ["bird"] * 6) == 0.9
    assert vqa_consensus("cat", ["cat", "cat", "dog"] + ["bird"] * 7) == 0.6
    assert vqa_consensus("cat", []) == 0.0


def test_binary_metrics_count_unknown_as_incorrect_without_positive_prediction():
    labels = ["yes", "no", "yes", "no"]
    preds = [binary_label("Yes, it is."), binary_label("no"), "unknown", "maybe"]
    result = binary_metrics(preds, labels)
    assert result["accuracy"] == 0.5
    assert result["unknown_predictions"] == 1


def test_multiple_choice_extraction_and_grouped_scoring():
    assert parse_choice("The answer is B.", ["red", "blue", "green"]) == "B"
    assert parse_choice("A train is visible.", ["car", "train", "bus"]) == "B"
    records = [
        {"id": "a", "task": "worldmedqa", "prompt": "q", "answer": "B", "choices": ["x", "y"], "language": "es", "question_type": "diagnosis"},
        {"id": "b", "task": "pope", "prompt": "q", "label": "no"},
    ]
    scored = score_records(records, ["B", "no"])
    assert scored["worldmedqa"]["accuracy"] == 1.0
    assert scored["worldmedqa"]["accuracy_by_language"]["es"] == 1.0
    assert scored["pope"]["accuracy"] == 1.0


def test_pope_scoring_reports_official_subset_strata():
    records = [
        {"id": "a", "task": "pope", "prompt": "q", "label": "yes", "category": "adversarial"},
        {"id": "b", "task": "pope", "prompt": "q", "label": "no", "category": "popular"},
    ]
    scored = score_records(records, ["yes", "yes"])["pope"]
    assert scored["by_category"]["adversarial"]["accuracy"] == 1.0
    assert scored["by_category"]["popular"]["accuracy"] == 0.0


def test_vqa_answer_normalization_covers_articles_numbers_and_contractions():
    assert normalize_answer("The Two cats.") == "2 cats"
    assert normalize_answer("isnt") == "isn't"
    assert normalize_answer("isn't") == "isn't"


def test_mme_rejects_incomplete_question_pairs_and_reports_category_scores():
    rows = [
        {"id": "a", "task": "mme_pair", "prompt": "q", "label": "yes", "pair_id": "p", "category": "count"},
        {"id": "b", "task": "mme_pair", "prompt": "q", "label": "no", "pair_id": "p", "category": "count"},
    ]
    metrics = score_records(rows, ["yes", "yes"])["mme_pair"]
    assert metrics["mme_score_sum"] == 1
    assert metrics["mme_score_by_category"]["count"] == 1
    try:
        score_records(rows[:1], ["yes"])
    except ValueError as error:
        assert "complete 2-question pairs" in str(error)
    else:
        raise AssertionError("incomplete MME pairs must be rejected")


def test_caption_scorer_collects_each_metric_without_external_process(monkeypatch):
    from scripts.eval_capability_suite import _CaptionScorer

    class Scorer:
        def __init__(self, score):
            self.score = score

        def compute_score(self, refs, hyps):
            return self.score, None

    output, _ = _CaptionScorer([("Bleu", Scorer([0.1, 0.2, 0.3, 0.4])),
                                ("CIDEr", Scorer(0.5))]).compute_score({"1": ["ref"]}, {"1": ["hyp"]})
    assert output == {"Bleu_1": 0.1, "Bleu_2": 0.2, "Bleu_3": 0.3, "Bleu_4": 0.4, "CIDEr": 0.5}


def test_paired_capability_bootstrap_uses_same_ids_and_mme_pair_units():
    baseline = {
        "a": {"id": "a", "task": "worldmedqa", "prediction": "A", "answer": "A", "choices": ["a", "b"]},
        "b": {"id": "b", "task": "worldmedqa", "prediction": "A", "answer": "B", "choices": ["a", "b"]},
    }
    candidate = {
        "a": {"id": "a", "task": "worldmedqa", "prediction": "B", "answer": "A", "choices": ["a", "b"]},
        "b": {"id": "b", "task": "worldmedqa", "prediction": "B", "answer": "B", "choices": ["a", "b"]},
    }
    score = paired_bootstrap(baseline, candidate, "worldmedqa", ["a", "b"], 500, 7)
    assert score["delta"] == 0.0
    rows_a = {
        "a": {"task": "mme_pair", "pair_id": "p1", "label": "yes", "prediction": "yes"},
        "b": {"task": "mme_pair", "pair_id": "p1", "label": "no", "prediction": "no"},
        "c": {"task": "mme_pair", "pair_id": "p2", "label": "yes", "prediction": "no"},
        "d": {"task": "mme_pair", "pair_id": "p2", "label": "no", "prediction": "no"},
    }
    rows_b = {**rows_a, "c": {**rows_a["c"], "prediction": "yes"}}
    paired = paired_bootstrap(rows_a, rows_b, "mme_pair", ["a", "b", "c", "d"], 500, 7)
    assert paired["pair_count"] == 2
    assert paired["pair_accuracy_delta"] == 0.5


def test_scoring_rejects_misaligned_predictions():
    try:
        score_records([{"task": "pope", "label": "yes"}], [])
    except ValueError as error:
        assert "length mismatch" in str(error)
    else:
        raise AssertionError("expected mismatched records/predictions to fail")


def test_partial_checkpoint_is_atomic_and_identity_tracks_prompt_inputs(tmp_path):
    output = tmp_path / "partial.json"
    atomic_json_write(output, {"predictions": ["a"]})
    assert output.exists()
    assert not (tmp_path / "partial.json.tmp").exists()
    first = resume_identity(manifest_sha256="abc", model_id="qwen", model=tmp_path / "model",
                            adapter=None, max_new_tokens=64, model_revision="rev1", max_image_pixels=1000, ids=["one"])
    changed = resume_identity(manifest_sha256="abc", model_id="qwen", model=tmp_path / "model",
                              adapter=None, max_new_tokens=64, model_revision="rev2", max_image_pixels=1000, ids=["one"])
    assert first != changed
