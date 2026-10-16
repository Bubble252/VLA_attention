"""Run the installed paper-search skill and preserve structured provenance."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, '/home/bubble/.codex/skills/paper-search/scripts')
from search_papers import search_papers

QUERIES = [
    'vision language action state aliasing',
    'robot policy representation metric learning',
    'language conditioned action similarity contrastive representation',
]

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    destination = root / 'search_raw.json'
    if destination.exists():
        raise SystemExit('Refusing to overwrite existing search results')
    results = search_papers(query=QUERIES, start_year=2020, end_year=2026,
                            max_results=3, parallel=True)
    destination.write_text(json.dumps({
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'queries': QUERIES, 'years': [2020, 2026], 'max_results_per_query_per_source': 3,
        'results': results,
    }, ensure_ascii=False, indent=2))
    print(destination)
    for source, records in results.items():
        print(source, len(records))
