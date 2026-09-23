import json
import subprocess
import sys


def run_validator(tmp_path, rows):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "manifest.meta.json"
    result = subprocess.run(
        [sys.executable, "scripts/validate_capability_manifest.py", str(manifest),
         "--dataset-root", str(tmp_path), "--benchmark", "unit", "--split", "val",
         "--source-revision", "test-rev", "--output", str(output)],
        text=True, capture_output=True,
    )
    return result, output


def test_validator_writes_manifest_and_prompt_fingerprints(tmp_path):
    (tmp_path / "image.jpg").write_bytes(b"image")
    rows = [{"id": "official-1", "task": "vqa", "image": "image.jpg",
             "prompt": "What is shown?", "answers": ["cat", "cat"]}]
    result, output = run_validator(tmp_path, rows)
    assert result.returncode == 0, result.stderr
    metadata = json.loads(output.read_text())
    assert metadata["n"] == 1
    assert metadata["task_counts"] == {"vqa": 1}
    assert len(metadata["ordered_id_prompt_sha256"]) == 64


def test_validator_rejects_duplicate_ids(tmp_path):
    row = {"id": "duplicate", "task": "vqa", "prompt": "q", "answers": ["a"]}
    result, _ = run_validator(tmp_path, [row, row])
    assert result.returncode != 0
    assert "duplicate IDs" in result.stderr
