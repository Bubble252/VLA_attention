from scripts.build_hf_capability_manifest import normalize_row, stable_score


def test_hf_manifest_normalizes_vqa_and_preserves_task_metadata():
    row = {"question_id": 23, "question": "What color?", "answers": ["red", "red", "blue"],
           "question_type": "color", "answer_type": "other"}
    result = normalize_row(row, "vqa", "23", "images/a.jpg")
    assert result["task"] == "vqa"
    assert result["answers"] == ["red", "red", "blue"]
    assert result["question_type"] == "color"


def test_hf_manifest_normalizes_pope_yes_no_and_subset():
    row = {"question_id": "q1", "question": "Is there a dog?", "answer": "yes", "category": "adversarial"}
    result = normalize_row(row, "pope", "q1", "images/a.jpg")
    assert result["task"] == "pope"
    assert result["label"] == "yes"
    assert result["category"] == "adversarial"


def test_hf_manifest_preserves_mme_pair_id():
    row = {"question_id": "image.png", "question": "Is this present?", "answer": "Yes",
           "category": "count", "_pair_id": "count:image.png:pair3"}
    result = normalize_row(row, "mme", "image.png::3", "images/a.png")
    assert result["task"] == "mme_pair"
    assert result["pair_id"] == "count:image.png:pair3"


def test_stable_sampling_is_deterministic_and_seeded():
    assert stable_score("official-id", 7) == stable_score("official-id", 7)
    assert stable_score("official-id", 7) != stable_score("official-id", 8)
