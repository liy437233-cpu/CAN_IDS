from __future__ import annotations

import csv
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ReleaseTests(unittest.TestCase):
    def test_endpoint_rules_match_declared_contract(self) -> None:
        script = ROOT / "scripts" / "build_endpoint_contract.py"
        spec = importlib.util.spec_from_file_location("endpoint_contract", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = json.loads((ROOT / "configs" / "endpoint_applicability.json").read_text(encoding="utf-8"))
        for profile in config["datasets"]:
            observed = {row["endpoint"]: row["status"] for row in module.evaluate(profile)}
            self.assertEqual(observed, profile["expected_contract"])

    def test_ctat_microsegments_do_not_define_attack_events(self) -> None:
        script = ROOT / "scripts" / "build_endpoint_contract.py"
        spec = importlib.util.spec_from_file_location("endpoint_contract_ctat", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = json.loads((ROOT / "configs" / "endpoint_applicability.json").read_text(encoding="utf-8"))
        ctat = next(profile for profile in config["datasets"] if profile["dataset_id"] == "CTAT")
        observed = {row["endpoint"]: row["status"] for row in module.evaluate(ctat)}
        self.assertEqual(observed["event_response"], "N/A")

    def test_road_rmttd_censors_undetected_events_at_horizon(self) -> None:
        script = ROOT / "scripts" / "aggregate_formal_results.py"
        spec = importlib.util.spec_from_file_location("aggregate_formal_results", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        events = [
            {"detected": "1", "ttd_sec": "0.25", "event_duration_sec": "1.0"},
            {"detected": "1", "ttd_sec": "8.0", "event_duration_sec": "10.0"},
            {"detected": "0", "ttd_sec": "0.5", "event_duration_sec": "0.5"},
        ]
        self.assertAlmostEqual(module.rmttd(events, 5.0), (0.25 + 5.0 + 5.0) / 3.0)

    def test_source_table_shapes(self) -> None:
        expected = json.loads((ROOT / "configs" / "expected_summary.json").read_text(encoding="utf-8"))["row_counts"]
        for name, count in expected.items():
            if name == "dataset_endpoint_contract.csv":
                continue
            self.assertEqual(len(read_csv(ROOT / "data" / "source_data" / name)), count)

    def test_cards_each_fold_has_zero_detection_mechanism(self) -> None:
        rows = read_csv(ROOT / "data" / "source_data" / "cards_mechanism_tpr.csv")
        zero_folds = {int(row["fold"]) for row in rows if float(row["mechanism_tpr"]) == 0.0}
        self.assertEqual(zero_folds, {0, 1, 2})

    def test_materialized_protocol_rows(self) -> None:
        expected = json.loads((ROOT / "configs" / "expected_summary.json").read_text(encoding="utf-8"))["protocol_row_counts"]
        for name, count in expected.items():
            self.assertEqual(len(read_csv(ROOT / "data" / "protocol" / name)), count)


if __name__ == "__main__":
    unittest.main()
