"""Run the scoop-check paper-search step and preserve raw structured records."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/home/bubble/.codex/skills/paper-search/scripts")
from search_papers import search_papers

QUERIES = [
    "vision language action instruction target grounding manipulation",
    "robot policy counterfactual visual evidence intervention",
    "multi object manipulation instruction retargeting action",
]


def main() -> None:
    destination = Path(__file__).with_name("step2_search_records.json")
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite {destination}")
    results = search_papers(
        query=QUERIES,
        start_year=2021,
        end_year=2026,
        max_results=5,
        parallel=True,
    )
    destination.write_text(
        json.dumps(
            {
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "queries": QUERIES,
                "start_year": 2021,
                "end_year": 2026,
                "max_results_per_query_per_source": 5,
                "results": results,
                "known_errors_must_be_read_from_stderr_log": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(destination)
    for source, records in results.items():
        print(f"{source}: {len(records)}")


if __name__ == "__main__":
    main()
