"""Guard V0--V4 controlled-comparison invariants before creating run manifests."""
import json
from pathlib import Path


def main() -> int:
    config = json.loads(Path("configs/experiments/P4_vlm_v0_v4_template.json").read_text())
    expected = {"V0": {"none"}, "V1": {"caption_cross_entropy"}, "V2": {"caption_cross_entropy", "retention"}, "V3": {"caption_cross_entropy", "semantic_attribution"}, "V4": {"caption_cross_entropy", "retention", "semantic_attribution"}}
    for name, losses in expected.items():
        if set(config["groups"][name]["losses"]) != losses:
            raise SystemExit(f"FAIL {name}: loss set differs from controlled design")
    if config["groups"]["V0"]["train"] or not all(config["groups"][name]["train"] for name in ("V1", "V2", "V3", "V4")):
        raise SystemExit("FAIL: V0/V1-V4 train flags invalid")
    if config["shared"]["data"]["supervision"] != "image + fixed caption prompt -> original Flickr30k caption":
        raise SystemExit("FAIL: main-table supervision drifted from caption SFT")
    print("PASS V0-V4 controlled-comparison template")


if __name__ == "__main__":
    main()
