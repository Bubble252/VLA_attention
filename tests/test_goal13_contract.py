import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.audit_goal13_rlds import (
    PROMPTS,
    audit_shards,
    build_report,
    detect_prompt,
    iter_tfrecord_payloads,
    load_split_manifest,
    sha256_bytes,
)
from scripts.build_goal13_contract import (
    default_data_arms,
    evaluate_contract,
    manifest_summary,
)


COMMIT = "4507931a625c844404c4a76128fb116536d8ca7c"
CONFIG_SHA = "a" * 64


def make_manifest():
    prompt_specs = {
        "red_left": (27, 5, 5, "put the red mug on the left plate"),
        "red_right": (34, 5, 5, "put the red mug on the right plate"),
        "white_left": (35, 5, 5, "put the white mug on the left plate"),
    }
    splits = {"train": [], "validation": [], "rollout": []}
    for prompt, (n_train, n_validation, n_rollout, language) in prompt_specs.items():
        for index in range(n_train + n_validation + n_rollout):
            if index < n_train:
                split = "train"
            elif index < n_train + n_validation:
                split = "validation"
            else:
                split = "rollout"
            splits[split].append(
                {
                    "episode_id": f"{prompt}-{index}",
                    "task": "LIVING_ROOM_SCENE5",
                    "language": language,
                    "steps": 100,
                    "source_shard": "shard",
                    "source_record_index": index,
                }
            )
    return {"version": "test", "splits": splits}


def make_audit():
    arms = default_data_arms()
    return {
        "schema": "goal13-rlds-audit-v1",
        "audit_status": "PASS",
        "source_language_counts": dict(arms["expected_source_counts"]),
        "train_language_counts": dict(arms["expected_train_counts"]),
        "split_language_counts": {
            "train": {"red_left": 27, "red_right": 34, "white_left": 35, "white_right": 0},
            "validation": {"red_left": 5, "red_right": 5, "white_left": 5},
            "rollout": {"red_left": 5, "red_right": 5, "white_left": 5},
        },
        "split_overlap_total": 0,
        "white_right_train_count": 0,
        "carrier_episode_count": 48,
        "white_bowl_right_episode_count": 44,
    }


def make_observed_contract(*, include_carrier=False):
    datasets = [
        {
            "name": "libero_90_gate_lerobot",
            "length": 11630,
            "sampling_weight": 0.6666667 if include_carrier else 1.0,
        }
    ]
    if include_carrier:
        datasets.append(
            {
                "name": "libero_90_goal13_carrier_lerobot",
                "length": 4786,
                "sampling_weight": 0.3333333,
            }
        )
    return {"datasets": datasets, "sampling_weight_sum": 1.0}


def make_carrier_rows(count: int = 48):
    return [
        {
            "output_record_index": index,
            "episode_id": f"carrier-{index}",
            "language": "put the yellow and white mug on the right plate",
        }
        for index in range(count)
    ]


def evaluate(*, data_arm="D0", audit=None, observed=None, runtime=None, carrier_rows=None):
    arms = default_data_arms()
    return evaluate_contract(
        audit=audit or make_audit(),
        split_manifest=make_manifest(),
        data_arm=data_arm,
        arm_definition=arms["arms"][data_arm],
        runtime=runtime or {"per_device_batch": 4, "num_processes": 1, "grad_accum": 1, "seed": 17},
        code_commit=COMMIT,
        config_sha256=CONFIG_SHA,
        observed_contract=observed or make_observed_contract(),
        parent_arm=None,
        marker_mode="none",
        counterfactual_mode="none",
        expected_source_counts=arms["expected_source_counts"],
        expected_train_counts=arms["expected_train_counts"],
        expected_manifest_split_counts=arms["expected_manifest_split_counts"],
        expected_carrier_count=48,
        expected_white_bowl_count=44,
        expected_c1_dataset_length=11630,
        c1_dataset_name="libero_90_gate_lerobot",
        expected_carrier_dataset_length=4786 if data_arm in {"D1", "D2"} else None,
        expected_c1_sampling_weight=0.6666667 if data_arm in {"D1", "D2"} else None,
        expected_carrier_sampling_weight=0.3333333 if data_arm in {"D1", "D2"} else None,
        carrier_rows=make_carrier_rows() if carrier_rows is None else carrier_rows,
        expected_raw_transitions=17424 if data_arm in {"D1", "D2"} else 12302,
        expected_trajectories=144 if data_arm in {"D1", "D2"} else 96,
        prompt_text=arms["prompts"],
        expected_seed=17,
        expected_effective_batch=4,
    )


class Goal13AuditTests(unittest.TestCase):
    def test_tfrecord_framing_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.tfrecord-00000-of-00001"
            payloads = [b"first payload", b"second payload"]
            with path.open("wb") as handle:
                for payload in payloads:
                    handle.write(struct.pack("<Q", len(payload)))
                    handle.write(b"\x00\x00\x00\x00")
                    handle.write(payload)
                    handle.write(b"\x00\x00\x00\x00")
            self.assertEqual(list(iter_tfrecord_payloads(path)), payloads)

    def test_prompt_detection_is_episode_level_and_fail_closed(self):
        self.assertEqual(
            detect_prompt(b"xxx put the white mug on the left plate yyy"),
            "white_left",
        )
        self.assertIsNone(detect_prompt(b"unrelated task"))
        with self.assertRaises(ValueError):
            detect_prompt(
                b"put the red mug on the left plate and put the red mug on the right plate"
            )

    def test_manifest_summary_counts_each_split(self):
        summary = manifest_summary(make_manifest())
        self.assertEqual(summary["split_counts"], {"train": 96, "validation": 15, "rollout": 15})
        self.assertEqual(summary["overlap_total"], 0)
        self.assertEqual(summary["language_counts"]["put the red mug on the left plate"], 37)

    def test_audit_end_to_end_passes_with_expected_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data"
            data_path.mkdir()
            c1_payloads = []
            prompt_rows = {
                "red_left": (27, 5, 5),
                "red_right": (34, 5, 5),
                "white_left": (35, 5, 5),
            }
            for prompt, counts in prompt_rows.items():
                for index, split in enumerate(
                    ["train"] * counts[0] + ["validation"] * counts[1] + ["rollout"] * counts[2]
                ):
                    payload = f"{prompt}-{index}-".encode() + PROMPTS[prompt]
                    c1_payloads.append((split, payload))
            carrier_payloads = [
                f"carrier-{index}-".encode() + PROMPTS["yellow_and_white_right"] for index in range(48)
            ]
            white_bowl_payloads = [
                f"white-bowl-{index}-".encode() + PROMPTS["white_bowl_right"] for index in range(44)
            ]
            all_payloads = [payload for _, payload in c1_payloads] + carrier_payloads + white_bowl_payloads
            for shard_index in range(256):
                shard = data_path / f"synthetic.tfrecord-{shard_index:05d}-of-00256"
                with shard.open("wb") as handle:
                    if shard_index < len(all_payloads):
                        payload = all_payloads[shard_index]
                        handle.write(struct.pack("<Q", len(payload)))
                        handle.write(b"\x00\x00\x00\x00")
                        handle.write(payload)
                        handle.write(b"\x00\x00\x00\x00")
            split_rows = {"train": [], "validation": [], "rollout": []}
            for split, payload in c1_payloads:
                episode_id = sha256_bytes(payload)
                language = next(
                    token.decode("utf-8") for token in PROMPTS.values() if token in payload
                )
                split_rows[split].append(
                    {
                        "episode_id": episode_id,
                        "task": "LIVING_ROOM_SCENE5",
                        "language": language,
                        "steps": 100,
                        "source_shard": "synthetic",
                        "source_record_index": 0,
                    }
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"splits": split_rows}))
            manifest_by_id, ids_by_split = load_split_manifest(manifest_path)
            shards = sorted(data_path.glob("*.tfrecord-*"))
            audit = audit_shards(shards, manifest_by_id)
            report = build_report(
                data_path=data_path,
                shard_count=len(shards),
                expected_shards=256,
                audit=audit,
                manifest_by_id=manifest_by_id,
                ids_by_split=ids_by_split,
                expected_source_counts=default_data_arms()["expected_source_counts"],
            )
            self.assertEqual(report["audit_status"], "PASS")
            self.assertEqual(report["source_language_counts"]["red_left"], 37)
            self.assertEqual(report["carrier_episode_count"], 48)
            self.assertEqual(report["white_right_train_count"], 0)


class Goal13ContractTests(unittest.TestCase):
    def test_d0_passes_with_complete_matched_evidence(self):
        contract = evaluate()
        self.assertEqual(contract["contract_status"], "PASS")
        self.assertEqual(contract["arm_status"], "PENDING_GATE")
        self.assertEqual(contract["data"]["train_episode_count"], 96)
        self.assertEqual(contract["data"]["heldout_prompt_train_count"], 0)

    def test_d1_adds_all_carrier_episodes(self):
        contract = evaluate(data_arm="D1", observed=make_observed_contract(include_carrier=True))
        self.assertEqual(contract["contract_status"], "PASS")
        self.assertEqual(contract["data"]["train_episode_count"], 144)
        self.assertEqual(contract["data"]["carrier_episode_count"], 48)

    def test_d1_requires_carrier_manifest(self):
        contract = evaluate(
            data_arm="D1",
            observed=make_observed_contract(include_carrier=True),
            carrier_rows=[],
        )
        self.assertEqual(contract["contract_status"], "INVALID")
        self.assertIn("carrier_manifest_count", contract["failures"])

    def test_d1_wrong_carrier_weight_is_invalid(self):
        observed = make_observed_contract(include_carrier=True)
        observed["datasets"][0]["sampling_weight"] = 0.5
        observed["datasets"][1]["sampling_weight"] = 0.5
        contract = evaluate(data_arm="D1", observed=observed)
        self.assertEqual(contract["contract_status"], "INVALID")
        self.assertIn("c1_sampling_weight", contract["failures"])
        self.assertIn("carrier_sampling_weight", contract["failures"])

    def test_d1_carrier_split_overlap_is_invalid(self):
        rows = make_carrier_rows()
        rows[0]["episode_id"] = "red_left-0"
        contract = evaluate(
            data_arm="D1",
            observed=make_observed_contract(include_carrier=True),
            carrier_rows=rows,
        )
        self.assertEqual(contract["contract_status"], "INVALID")
        self.assertIn("carrier_manifest_split_overlap_zero", contract["failures"])

    def test_white_right_in_train_invalidates_contract(self):
        audit = make_audit()
        audit["train_language_counts"]["white_right"] = 1
        audit["split_language_counts"]["train"]["white_right"] = 1
        audit["white_right_train_count"] = 1
        contract = evaluate(audit=audit)
        self.assertEqual(contract["contract_status"], "INVALID")
        self.assertIn("white_right_train_zero", contract["failures"])
        self.assertIn("arm_white_right_train_zero", contract["failures"])

    def test_missing_observed_dataset_contract_is_invalid(self):
        arms = default_data_arms()
        contract = evaluate_contract(
            audit=make_audit(),
            split_manifest=make_manifest(),
            data_arm="D0",
            arm_definition=arms["arms"]["D0"],
            runtime={"per_device_batch": 4, "num_processes": 1, "grad_accum": 1, "seed": 17},
            code_commit=COMMIT,
            config_sha256=CONFIG_SHA,
            observed_contract=None,
            parent_arm=None,
            marker_mode="none",
            counterfactual_mode="none",
            expected_source_counts=arms["expected_source_counts"],
            expected_train_counts=arms["expected_train_counts"],
            expected_manifest_split_counts=arms["expected_manifest_split_counts"],
            expected_carrier_count=48,
            expected_white_bowl_count=44,
            expected_c1_dataset_length=11630,
            c1_dataset_name="libero_90_gate_lerobot",
            expected_carrier_dataset_length=None,
            expected_c1_sampling_weight=None,
            expected_carrier_sampling_weight=None,
            carrier_rows=None,
            expected_raw_transitions=12302,
            expected_trajectories=96,
            prompt_text=arms["prompts"],
            expected_seed=17,
            expected_effective_batch=4,
        )
        self.assertEqual(contract["contract_status"], "INVALID")
        self.assertIn("observed_dataset_contract", contract["failures"])

    def test_wrong_effective_batch_is_invalid(self):
        contract = evaluate(runtime={"per_device_batch": 2, "num_processes": 1, "grad_accum": 1, "seed": 17})
        self.assertEqual(contract["contract_status"], "INVALID")
        self.assertIn("effective_batch", contract["failures"])


if __name__ == "__main__":
    unittest.main()
