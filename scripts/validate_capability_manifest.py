"""Validate and fingerprint a normalized capability evaluation JSONL manifest."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

REQUIRED={
    "caption": {"references"},
    "vqa": {"answers"},
    "textvqa": {"answers"},
    "pope": {"label"},
    "binary": {"label"},
    "multiple_choice": {"answer","choices"},
    "worldmedqa": {"answer","choices","language","question_type"},
    "mme_pair": {"label","pair_id"},
    "exact_match": {"answers"},
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("manifest",type=Path)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--benchmark",required=True)
    p.add_argument("--split",required=True)
    p.add_argument("--source-revision",required=True)
    p.add_argument("--output",type=Path)
    p.add_argument("--require-images",action="store_true")
    a=p.parse_args()
    raw=a.manifest.read_bytes(); rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows: raise SystemExit("manifest is empty")
    ids=[]; counts=Counter(); missing=[]
    for i,row in enumerate(rows):
        for field in ("id","task","prompt"):
            if not isinstance(row.get(field),str) or not row[field].strip(): raise ValueError(f"row {i}: missing/invalid {field}")
        task=row["task"]
        if task not in REQUIRED: raise ValueError(f"row {i}: unsupported task {task!r}")
        fields=REQUIRED[task]-set(row)
        if fields: raise ValueError(f"row {i}: task={task} missing fields {sorted(fields)}")
        if a.require_images and not row.get("image"): raise ValueError(f"row {i}: image required")
        if row.get("image") is not None and not isinstance(row["image"], str):
            raise ValueError(f"row {i}: image must be a relative path string")
        if row.get("image") and not (a.dataset_root/row["image"]).is_file(): missing.append(str(a.dataset_root/row["image"]))
        if task in ("pope","binary","mme_pair") and str(row["label"]).lower() not in ("yes","no"):
            raise ValueError(f"row {i}: binary label must be yes/no")
        if task in ("multiple_choice","worldmedqa") and (not isinstance(row["choices"],list) or not row["choices"] or not str(row["answer"]).strip()):
            raise ValueError(f"row {i}: choices and answer must be nonempty")
        if task=="caption" and (not isinstance(row["references"],list) or not row["references"] or not all(isinstance(x,str) for x in row["references"])):
            raise ValueError(f"row {i}: references must be a nonempty list of strings")
        if task in ("vqa","exact_match") and (not isinstance(row["answers"],list) or not row["answers"] or not all(isinstance(x,str) for x in row["answers"])):
            raise ValueError(f"row {i}: answers must be nonempty")
        ids.append(row["id"]); counts[task]+=1
    if missing: raise FileNotFoundError("missing image files: "+", ".join(missing[:5]))
    if len(ids)!=len(set(ids)): raise ValueError("duplicate IDs in manifest")
    prompt_blob="\n".join(row["id"]+"\t"+row["prompt"] for row in rows).encode()
    payload={"benchmark":a.benchmark,"split":a.split,"source_revision":a.source_revision,
             "manifest":str(a.manifest),"sha256":hashlib.sha256(raw).hexdigest(),
             "ordered_id_prompt_sha256":hashlib.sha256(prompt_blob).hexdigest(),
             "n":len(rows),"task_counts":dict(counts),"official_id_count":len(set(ids))}
    out=a.output or a.manifest.with_suffix(a.manifest.suffix+".meta.json")
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
