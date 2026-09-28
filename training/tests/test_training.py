from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TRAINING_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import cards_models
import cards_training
import road_evaluation


class TrainingPackageTests(unittest.TestCase):
    def test_configs_are_portable(self) -> None:
        for path in sorted((TRAINING_ROOT / "configs").glob("*.json")):
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"[A-Za-z]:[/\\]")
            json.loads(text)

    def test_cards_role_table_hash(self) -> None:
        path = TRAINING_ROOT / "data" / "protocol" / "cards_trace_roles.csv"
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        config = json.loads((TRAINING_ROOT / "configs" / "cards_training.json").read_text(encoding="utf-8"))
        self.assertEqual(observed, config["data"]["role_csv_sha256"])

    def test_materialized_cards_normal_assignments(self) -> None:
        config = json.loads((TRAINING_ROOT / "configs" / "cards_training.json").read_text(encoding="utf-8"))
        original = config["data"]["role_csv"]
        config["data"]["role_csv"] = str(TRAINING_ROOT / original)
        for fold in config["data"]["outer_folds"]:
            roles = cards_training.roles_for_fold(config, int(fold))
            self.assertEqual(len(roles["references"]), 10)
            self.assertEqual(len(roles["normal_train"]), 9)
            self.assertFalse(
                {row["attack_mechanism_group"] for row in roles["attack_train"]}
                & {row["attack_mechanism_group"] for row in roles["outer_attack"]}
            )
            calibration = {row["trace_name"] for row in roles["normal_calibration"]}
            outer_normal = {row["trace_name"] for row in roles["outer_normal"]}
            self.assertTrue(calibration.isdisjoint(outer_normal))

    def test_training_assembly_success_markers(self) -> None:
        self.assertEqual(cards_training.completion_marker("TRAINING_ASSEMBLY_STATIC_PASS"), "COMPLETED.json")
        self.assertEqual(cards_training.completion_marker("TRAINING_ASSEMBLY_PASS"), "COMPLETED.json")
        self.assertEqual(cards_training.completion_marker("TRAINING_ASSEMBLY_STATIC_FAIL"), "FAILED.json")

    def test_road_event_alarms_are_limited_to_injection_interval(self) -> None:
        timestamps = np.asarray([0.5, 1.0, 1.2, 2.0, 2.2])
        probabilities = np.asarray([0.99, 0.1, 0.8, 0.1, 0.99])
        metadata = {"injection_interval": [1.0, 2.0]}
        row = road_evaluation.event_row("capture", "group", timestamps, probabilities, metadata, "frame")
        self.assertEqual(row["detected"], 1)
        self.assertAlmostEqual(row["ttd_sec"], 0.2)
        outside_only = road_evaluation.event_row(
            "capture",
            "group",
            timestamps,
            np.asarray([0.99, 0.1, 0.1, 0.1, 0.99]),
            metadata,
            "frame",
        )
        self.assertEqual(outside_only["detected"], 0)
        self.assertIsNone(outside_only["ttd_sec"])

    def test_alarm_count_does_not_increase_with_cooldown(self) -> None:
        scores = np.ones(6)
        timestamps = np.asarray([0.0, 0.05, 0.2, 1.1, 1.15, 7.0])
        counts = [cards_models.alarm_count(scores, timestamps, 0.5, value) for value in (0.1, 1.0, 5.0)]
        self.assertGreaterEqual(counts[0], counts[1])
        self.assertGreaterEqual(counts[1], counts[2])

    def test_threshold_does_not_increase_with_budget(self) -> None:
        stream = [(np.asarray([0.1, 0.2, 0.8, 0.9]), np.asarray([0.0, 1.0, 2.0, 3.0]))]
        strict = cards_models.calibrate_threshold(stream, 1200.0, 1.0)
        relaxed = cards_models.calibrate_threshold(stream, 2400.0, 1.0)
        self.assertLessEqual(relaxed["threshold"], strict["threshold"])

    def test_probe_models_and_threshold(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 16, 21)).astype(np.float32)
        y = np.asarray([0] * 40 + [1] * 40, dtype=np.uint8)
        config = json.loads((TRAINING_ROOT / "configs" / "cards_training.json").read_text(encoding="utf-8"))
        for kind in ("logistic_regression", "hgb"):
            model = cards_models.fit_sklearn(x, y, config, kind)
            probability = model.predict_proba(x[:, -1, :])[:, 1]
            self.assertTrue(np.isfinite(probability).all())
        threshold = cards_models.calibrate_threshold(
            [(np.asarray([0.1, 0.2, 0.8, 0.9]), np.asarray([0.0, 1.0, 2.0, 3.0]))],
            1200.0,
            1.0,
        )
        self.assertTrue(np.isfinite(threshold["threshold"]))


if __name__ == "__main__":
    unittest.main()
