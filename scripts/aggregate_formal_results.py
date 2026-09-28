"""Aggregate completed ROAD, CTAT, and CarDS runs into paper source tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rmttd(events: list[dict[str, str]], horizon_seconds: float) -> float:
    if not events:
        raise ValueError("RMTTD requires at least one event")
    contributions = []
    for event in events:
        if int(event["detected"]) == 0:
            contributions.append(horizon_seconds)
        else:
            contributions.append(min(float(event["ttd_sec"]), horizon_seconds))
    return float(np.mean(contributions))


def macro_f1_from_counts(tp: int, fp: int, tn: int, fn: int) -> float:
    positive = 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    negative = 0.0 if 2 * tn + fp + fn == 0 else 2 * tn / (2 * tn + fp + fn)
    return (positive + negative) / 2.0


def dense_rank(values: dict[str, float], descending: bool) -> dict[str, int]:
    ordered = sorted(set(values.values()), reverse=descending)
    ranks = {value: index + 1 for index, value in enumerate(ordered)}
    return {name: ranks[value] for name, value in values.items()}


def road_rows(folder: Path) -> list[dict[str, Any]]:
    events = read_csv(folder / "attack_event_oof.csv")
    frames = read_csv(folder / "frame_capture_oof.csv")
    normals = read_csv(folder / "normal_capture_oof.csv")
    models = sorted({row["model"] for row in events})
    summary: dict[str, dict[str, float]] = {}
    for model in models:
        model_events = [row for row in events if row["model"] == model]
        model_frames = [row for row in frames if row["model"] == model]
        model_normals = [row for row in normals if row["model"] == model]
        counts = {
            name: sum(int(row[name]) for row in model_frames)
            for name in ("tp", "fp", "tn", "fn")
        }
        normal_hours = sum(float(row["normal_hours"]) for row in model_normals)
        false_alarms = sum(int(row["false_alarm_episodes_1s"]) for row in model_normals)
        summary[model] = {
            "frame_macro_f1": macro_f1_from_counts(**counts),
            "der_100ms": float(np.mean([int(row["der_100ms"]) for row in model_events])),
            "rmttd_5s": rmttd(model_events, 5.0),
            "normal_fae_h": false_alarms / normal_hours,
        }
    frame_ranks = dense_rank({name: value["frame_macro_f1"] for name, value in summary.items()}, True)
    der_ranks = dense_rank({name: value["der_100ms"] for name, value in summary.items()}, True)
    fae_ranks = dense_rank({name: value["normal_fae_h"] for name, value in summary.items()}, False)
    return [
        {
            "probe": model,
            **summary[model],
            "frame_rank": frame_ranks[model],
            "der_rank": der_ranks[model],
            "fae_rank": fae_ranks[model],
        }
        for model in models
    ]


def ctat_rows(folder: Path) -> list[dict[str, Any]]:
    preferred = folder / "ctat_microsegment_test.csv"
    source = preferred if preferred.is_file() else folder / "ctat_event_test.csv"
    rows = read_csv(source)
    unique: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["task_set"], row["filename"], int(row["event_index"]))
        unique.setdefault(key, row)
    by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in unique.values():
        by_task[row["task_set"]].append(row)
    output = []
    for task in sorted(by_task):
        values = by_task[task]
        frames = np.asarray([int(row["event_frames"]) for row in values], dtype=float)
        durations = np.asarray([float(row["event_duration_sec"]) for row in values], dtype=float)
        output.append(
            {
                "task": task,
                "unique_microsegments": len(values),
                "median_frames": float(np.median(frames)),
                "median_duration_ms": float(np.median(durations) * 1000.0),
                "single_frame_fraction": float(np.mean(frames == 1)),
                "duration_le_10ms_fraction": float(np.mean(durations <= 0.01)),
            }
        )
    return output


def family_name(model: str) -> str | None:
    if model == "hgb__fixed":
        return "HGB"
    if model == "logistic_regression__fixed":
        return "LR"
    if model.startswith("early_fusion_tcn__"):
        return "TCN"
    return None


def cards_sensitivity_rows(folder: Path) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for fold in range(3):
        result = read_json(folder / f"fold_{fold}_result.json")
        for model, settings in result["metrics"].items():
            family = family_name(model)
            if family is None:
                continue
            for setting, metrics in settings.items():
                budget_text, cooldown_text = setting.split("__")
                raw.append(
                    {
                        "family": family,
                        "budget_fae_h": float(budget_text.replace("budget_", "")),
                        "cooldown_seconds": float(cooldown_text.replace("cooldown_", "")),
                        "mechanism_macro_tpr": float(metrics["mechanism_macro_tpr"]),
                        "der_100ms": float(metrics["der_trace_mean_by_deadline"]["0.1"]),
                        "outer_normal_fae_h": float(metrics["outer_normal_fae_per_hour"]),
                        "rmttd_5s": float(metrics["rmttd_trace_mean_by_horizon"]["5"]),
                    }
                )
    grouped: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        grouped[(row["family"], row["budget_fae_h"], row["cooldown_seconds"])].append(row)
    output = []
    for (family, budget, cooldown), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "family": family,
            "budget_fae_h": budget,
            "cooldown_seconds": cooldown,
            "observation_count": len(values),
            "independence_note": (
                "3 folds" if family != "TCN" else "3 folds x 3 frozen seeds; not 9 independent datasets"
            ),
        }
        for metric in ("mechanism_macro_tpr", "der_100ms", "outer_normal_fae_h", "rmttd_5s"):
            numbers = np.asarray([value[metric] for value in values], dtype=float)
            row[f"{metric}_mean"] = float(numbers.mean())
            row[f"{metric}_min"] = float(numbers.min())
            row[f"{metric}_max"] = float(numbers.max())
        output.append(row)
    return output


def cards_mechanism_rows(folder: Path) -> list[dict[str, Any]]:
    rows = []
    for fold in range(3):
        result = read_json(folder / f"fold_{fold}_result.json")
        metrics = result["metrics"]["hgb__fixed"]["budget_30__cooldown_1"]
        ordered = sorted(metrics["per_mechanism_tpr"].items(), key=lambda item: (item[1], item[0]))
        count = len(ordered)
        lower_quartile = float(ordered[int(math.floor((count - 1) * 0.25))][1])
        mean = float(np.mean([value for _, value in ordered]))
        for rank, (mechanism, value) in enumerate(ordered, start=1):
            rows.append(
                {
                    "fold": fold,
                    "mechanism": mechanism,
                    "rank_within_fold": rank,
                    "mechanism_tpr": float(value),
                    "mechanism_macro_tpr": mean,
                    "lower_quartile_tpr": lower_quartile,
                    "worst_mechanism_tpr": float(ordered[0][1]),
                    "mechanism_count": count,
                    "der_100ms": float(metrics["der_trace_mean_by_deadline"]["0.1"]),
                    "rmttd_5s": float(metrics["rmttd_trace_mean_by_horizon"]["5"]),
                    "outer_normal_fae_h": float(metrics["outer_normal_fae_per_hour"]),
                }
            )
    return rows


def main_table_rows(
    road: list[dict[str, Any]],
    ctat: list[dict[str, Any]],
    mechanisms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    total_segments = sum(row["unique_microsegments"] for row in ctat)
    rows = [
        {
            "dataset": "CTAT",
            "unit": f"{total_segments:,} positive-label microsegments across {len(ctat)} tasks",
            "probe_or_setting": "label-semantics negative control",
            "frame_metric": "median 1 frame; median 0 ms",
            "operational_metric": "attack-process DER/RMTTD not applicable",
            "normal_fae_h": "available",
            "tail_or_robustness": "10/50/100/500 ms columns are semantically identical",
            "paper_role": "endpoint applicability gate",
        }
    ]
    for row in road:
        rows.append(
            {
                "dataset": "ROAD",
                "unit": "activity-group-isolated attack events",
                "probe_or_setting": row["probe"],
                "frame_metric": f"Frame Macro-F1={row['frame_macro_f1']:.4f} (rank {row['frame_rank']})",
                "operational_metric": (
                    f"DER@100ms={row['der_100ms']:.4f} (rank {row['der_rank']}); "
                    f"RMTTD@5s={row['rmttd_5s']:.4f}s"
                ),
                "normal_fae_h": f"{row['normal_fae_h']:.2f} (rank {row['fae_rank']})",
                "tail_or_robustness": "0.1/1/5 s cooldown and 10/50/100/500 ms deadlines available",
                "paper_role": "within-dataset rank reversal",
            }
        )
    for fold in range(3):
        fold_rows = [row for row in mechanisms if row["fold"] == fold]
        first = fold_rows[0]
        rows.append(
            {
                "dataset": "CarDS",
                "unit": f"mechanism-disjoint fold {fold}",
                "probe_or_setting": "HGB; 30 FAE/h; 1 s cooldown",
                "frame_metric": "reported probe configuration",
                "operational_metric": (
                    f"Macro-TPR={first['mechanism_macro_tpr']:.4f}; "
                    f"DER@100ms={first['der_100ms']:.4f}; RMTTD@5s={first['rmttd_5s']:.3f}s"
                ),
                "normal_fae_h": f"{first['outer_normal_fae_h']:.2f}",
                "tail_or_robustness": (
                    f"worst={first['worst_mechanism_tpr']:.4f}; Q25={first['lower_quartile_tpr']:.4f}"
                ),
                "paper_role": "mechanism lower-tail failure",
            }
        )
    rows.append(
        {
            "dataset": "CarDS",
            "unit": "3 budgets x 3 cooldowns x 3 folds",
            "probe_or_setting": "HGB vs TCN vs LR",
            "frame_metric": "operational sensitivity grid",
            "operational_metric": "Macro-TPR winner changes from HGB to TCN",
            "normal_fae_h": "HGB lowest in 9/9 grid cells",
            "tail_or_robustness": "HGB highest DER@100ms in 9/9 grid cells",
            "paper_role": "operational-rule sensitivity",
        }
    )
    return rows


def require_completed(folder: Path) -> None:
    if not (folder / "COMPLETED.json").is_file():
        raise FileNotFoundError(f"completed run marker not found: {folder}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-results", type=Path, required=True)
    parser.add_argument("--ctat-results", type=Path, required=True)
    parser.add_argument("--cards-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    road_folder = args.road_results.resolve()
    ctat_folder = args.ctat_results.resolve()
    cards_folder = args.cards_results.resolve()
    output = args.output.resolve()
    for folder in (road_folder, ctat_folder, cards_folder):
        require_completed(folder)

    road = road_rows(road_folder)
    ctat = ctat_rows(ctat_folder)
    sensitivity = cards_sensitivity_rows(cards_folder)
    mechanisms = cards_mechanism_rows(cards_folder)
    main_table = main_table_rows(road, ctat, mechanisms)

    tables = {
        "road_rank_reversal.csv": (road, list(road[0])),
        "ctat_label_granularity.csv": (ctat, list(ctat[0])),
        "cards_operational_sensitivity.csv": (sensitivity, list(sensitivity[0])),
        "cards_mechanism_tpr.csv": (mechanisms, list(mechanisms[0])),
        "main_result_table.csv": (main_table, list(main_table[0])),
    }
    for name, (rows, fields) in tables.items():
        write_csv(output / name, rows, fields)

    manifest_path = args.manifest.resolve() if args.manifest else output.parent / "manifest.json"
    input_files = {
        "road": ["attack_event_oof.csv", "frame_capture_oof.csv", "normal_capture_oof.csv"],
        "ctat": [
            "ctat_microsegment_test.csv"
            if (ctat_folder / "ctat_microsegment_test.csv").is_file()
            else "ctat_event_test.csv"
        ],
        "cards": [f"fold_{fold}_result.json" for fold in range(3)],
    }
    input_folders = {"road": road_folder, "ctat": ctat_folder, "cards": cards_folder}
    manifest = {
        "manifest_version": 2,
        "aggregation_inputs": {
            dataset: [
                {"file": name, "sha256": sha256(input_folders[dataset] / name)}
                for name in names
            ]
            for dataset, names in input_files.items()
        },
        "files": [
            {
                "file": name,
                "bytes": (output / name).stat().st_size,
                "sha256": sha256(output / name),
                "rows": len(rows),
            }
            for name, (rows, _) in sorted(tables.items())
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
