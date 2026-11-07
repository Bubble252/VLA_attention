import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_goal13_c1_pilot import evaluate_c1_pilot


class Goal13C1PilotTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        postrun = root / "postrun.json"
        postrun.write_text(json.dumps({"postrun_status": "PASS", "arm": "D0"}))
        summary = root / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "num_init_groups": 15,
                    "num_episodes": 60,
                    "seen": {
                        "episodes": 45,
                        "successes": 24,
                        "group_any_successes": 12,
                        "group_all_successes": 4,
                    },
                    "white_right": {
                        "episodes": 15,
                        "successes": 0,
                        "strict_destination_successes": 0,
                    },
                    "groups": [
                        {
                            "shared_init_id": f"init_{index}",
                            "seen_episodes": 3,
                            "seen_successes": 1,
                            "seen_any_success": True,
                            "seen_all_success": False,
                            "white_right_episodes": 1,
                            "white_right_successes": 0,
                            "white_right_strict_destination": 0,
                        }
                        for index in range(15)
                    ],
                }
            )
        )
        return postrun, summary

    def evaluate(self, postrun: Path, summary: Path):
        return evaluate_c1_pilot(
            postrun_path=postrun,
            summary_path=summary,
            expected_groups=15,
        )

    def test_valid_pilot_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, summary = self.make_fixture(Path(directory))
            report = self.evaluate(postrun, summary)
        self.assertEqual(report["contract_status"], "PASS")
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["white_right"]["strict_destination_successes"], 0)

    def test_missing_groups_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, summary = self.make_fixture(Path(directory))
            payload = json.loads(summary.read_text())
            payload["num_init_groups"] = 14
            summary.write_text(json.dumps(payload))
            report = self.evaluate(postrun, summary)
        self.assertEqual(report["contract_status"], "INVALID")
        self.assertIn("num_init_groups", report["failures"])

    def test_wrong_white_right_count_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, summary = self.make_fixture(Path(directory))
            payload = json.loads(summary.read_text())
            payload["white_right"]["episodes"] = 14
            summary.write_text(json.dumps(payload))
            report = self.evaluate(postrun, summary)
        self.assertEqual(report["contract_status"], "INVALID")
        self.assertIn("white_right_episodes", report["failures"])

    def test_duplicate_group_ids_are_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            postrun, summary = self.make_fixture(Path(directory))
            payload = json.loads(summary.read_text())
            payload["groups"][1]["shared_init_id"] = payload["groups"][0]["shared_init_id"]
            summary.write_text(json.dumps(payload))
            report = self.evaluate(postrun, summary)
        self.assertEqual(report["contract_status"], "INVALID")
        self.assertIn("group_ids_unique", report["failures"])


if __name__ == "__main__":
    unittest.main()
