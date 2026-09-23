from scripts.eval_capability_suite import (
    binary_metrics,
    binary_label,
    parse_choice,
    score_records,
    vqa_consensus,
)


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
    records = [
        {"id": "a", "task": "worldmedqa", "prompt": "q", "answer": "B", "choices": ["x", "y"], "language": "es", "question_type": "diagnosis"},
        {"id": "b", "task": "pope", "prompt": "q", "label": "no"},
    ]
    scored = score_records(records, ["B", "no"])
    assert scored["worldmedqa"]["accuracy"] == 1.0
    assert scored["worldmedqa"]["accuracy_by_language"]["es"] == 1.0
    assert scored["pope"]["accuracy"] == 1.0


def test_scoring_rejects_misaligned_predictions():
    try:
        score_records([{"task": "pope", "label": "yes"}], [])
    except ValueError as error:
        assert "length mismatch" in str(error)
    else:
        raise AssertionError("expected mismatched records/predictions to fail")
