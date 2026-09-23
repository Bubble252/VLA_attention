# P4 capability evaluation manifest schema

`scripts/eval_capability_suite.py` consumes normalized JSONL so all checkpoints receive identical prompts and image preprocessing. Build one manifest per official benchmark split, preserve the official sample ID, and save its source revision/checksum beside the manifest. Do not mix examples from different official splits in a file.

Common fields:

```json
{"id":"official-id","task":"vqa","image":"relative/image.jpg","prompt":"...","answers":["..."],"category":"..."}
```

Supported task types and required fields:

| Task | Required fields | Output metric | Notes |
|---|---|---|---|
| `caption` | `id,task,image,prompt,references` | CIDEr, BLEU-1..4, METEOR, ROUGE-L if `pycocoevalcap` installed | Inference only; the current small-sample scorer is a pipeline check, not a final official leaderboard run. |
| `vqa` | `id,task,image,prompt,answers` | VQA leave-one-annotator-out consensus formula | Answer normalization is still generic; final result must be rescored with the official VQAv2 evaluator. |
| `textvqa` | `id,task,image,prompt,answers` | approximate TextVQA consensus | Final result must use official TextVQA answer normalization/scorer. |
| `pope` / `binary` | `id,task,image,prompt,label` | accuracy, yes precision/recall/F1 | Labels must be canonical `yes`/`no`; preserve adversarial/popular/random split metadata. |
| `multiple_choice` | `id,task,image,prompt,answer,choices` | accuracy | Use official answer normalization and option order. |
| `worldmedqa` | `id,task,image,prompt,answer,choices,language,question_type` | accuracy overall/by language/by question type | Never train, tune prompts, or calibrate maps on this split. |
| `mme_pair` | `id,task,image,prompt,label,pair_id` | paired accuracy | An MME pair scores correct only if both positive and negative questions for that pair are correct. |
| `exact_match` | `id,task,image,prompt,answers` | normalized exact match | Use only for datasets whose official metric is exact match. |

Before inference, freeze prompt text, task instruction, image resolution policy, processor revision, `max_new_tokens`, greedy decoding (`do_sample=false`), and answer extraction rules. Store these in the manifest metadata file or run config, and compute a prompt/config SHA256. Each V0–V4 run must cover identical IDs; missing predictions or failed samples invalidate paired comparisons until rerun.

The runner writes an atomic `<output>.partial.json` every 10 completed items by default. To continue after an interruption, repeat the same command with `--resume`; it verifies the manifest hash, ordered IDs, model ID/path/revision, adapter, and generation length before reusing predictions. Change the output path for a different checkpoint/config. The completed report includes per-item latency, aggregate latency/throughput, and CUDA peak allocated/reserved memory when available. Do not use partial reports for benchmark claims.

Validate each manifest before uploading/running:

```bash
python scripts/validate_capability_manifest.py \
  experiment_workspace/manifests/capability/vqav2_val.jsonl \
  --dataset-root /path/to/dataset \
  --benchmark VQAv2 --split validation \
  --source-revision <official-release-or-commit> \
  --require-images
```

The validator writes a sidecar metadata JSON with manifest SHA256, official-ID count, task counts, split, and source revision. If manifests are assembled from downloaded archive data, record archive SHA256 and conversion-script commit as well.

The built-in VQA score uses the leave-one-annotator-out consensus formula, but its answer normalization is generic and does not claim exact parity with the official evaluator. It is useful for pilot smoke tests only. The final table must use official dataset scorers (including exact VQAv2 normalization, TextVQA answer normalization, and MME paired/task scoring). The generic runner preserves raw predictions to support a scorer replacement without rerunning inference. The validator records both the manifest SHA256 and a hash over ordered IDs plus prompts so accidental prompt/order drift is detectable.
