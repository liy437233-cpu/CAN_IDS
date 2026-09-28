"""Rebuild the main table and principal figures from distributed source data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


PALETTE = {
    "blue": (46 / 255, 111 / 255, 158 / 255),
    "orange": (217 / 255, 130 / 255, 43 / 255),
    "gray": (107 / 255, 114 / 255, 128 / 255),
    "risk": (162 / 255, 59 / 255, 59 / 255),
    "ink": (34 / 255, 34 / 255, 34 / 255),
    "charcoal": (75 / 255, 85 / 255, 99 / 255),
    "light_gray": (229 / 255, 231 / 255, 235 / 255),
    "light_blue": (169 / 255, 200 / 255, 222 / 255),
    "highlight": (242 / 255, 210 / 255, 138 / 255),
    "separator": (209 / 255, 213 / 255, 219 / 255),
}
COLORS = {"HGB": PALETTE["blue"], "TCN": PALETTE["orange"], "LR": PALETTE["gray"], "risk": PALETTE["risk"], "ink": PALETTE["ink"]}
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "operational-can-ids-evaluation-v1",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def panel_label(ax: Any, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")


def save_figure(fig: Any, base: Path) -> dict[str, dict[str, Any]]:
    base.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for suffix, kwargs in (
        ("png", {"dpi": 600}),
        ("pdf", {"metadata": {"CreationDate": None, "ModDate": None}}),
        ("svg", {"metadata": {"Date": None}}),
    ):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs[suffix] = {"path": path.relative_to(base.parents[1]).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
    plt.close(fig)
    return outputs


def figure1(contract: list[dict[str, str]], road: list[dict[str, str]], ctat: list[dict[str, str]], base: Path) -> dict[str, dict[str, Any]]:
    fig = plt.figure(figsize=(7.09, 5.1))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.16, top=0.91, hspace=0.58, wspace=0.34)

    endpoints = [
        ("frame_classification", "Frame classification"),
        ("event_response", "Event response"),
        ("independent_normal_alarm_burden", "Independent-normal FAE/h"),
        ("mechanism_tail", "Mechanism-tail risk"),
        ("cross_dataset_pooled_score", "Cross-dataset pooled score"),
    ]
    datasets = ["ROAD", "CTAT", "CarDS", "HCRL-CH"]
    code = {"P": 0, "N/A": 1, "D": 2, "A": 3}
    lookup = {(row["dataset"], row["endpoint"]): row["status"] for row in contract}
    matrix = np.asarray([[code[lookup[(dataset, endpoint)]] for dataset in datasets] for endpoint, _ in endpoints])
    ax_a.imshow(matrix, aspect="auto", cmap=ListedColormap([PALETTE["charcoal"], PALETTE["light_gray"], PALETTE["light_blue"], PALETTE["blue"]]), vmin=-0.5, vmax=3.5)
    for y, (endpoint, _) in enumerate(endpoints):
        for x, dataset in enumerate(datasets):
            status = lookup[(dataset, endpoint)]
            ax_a.text(x, y, status, ha="center", va="center", color="white" if status in {"A", "P"} else COLORS["ink"])
    ax_a.set_xticks(range(len(datasets)), datasets)
    ax_a.set_yticks(range(len(endpoints)), [label for _, label in endpoints])
    ax_a.tick_params(length=0)
    ax_a.set_title("Semantic endpoint applicability and external metadata exercise", pad=7)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    panel_label(ax_a, "a")

    probe_colors = {"frame": PALETTE["risk"], "context": PALETTE["orange"], "timing": PALETTE["blue"]}
    x = np.arange(3)
    for row in road:
        ranks = [int(row["frame_rank"]), int(row["der_rank"]), int(row["fae_rank"])]
        ax_b.plot(x, ranks, marker="o", markersize=4.2, color=probe_colors[row["probe"]], label=row["probe"].capitalize())
    ax_b.set_xticks(x, ["Frame\nMacro-F1", "DER@100ms", "FAE/h\n(lower is better)"])
    ax_b.set_yticks([1, 2, 3])
    ax_b.set_ylim(3.35, 0.65)
    ax_b.set_ylabel("Dense rank (1 = best)")
    ax_b.set_title("ROAD: frame and operational ranks diverge", pad=7)
    ax_b.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.6)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, handlelength=1.2, columnspacing=0.8)
    panel_label(ax_b, "b")

    labels = [row["task"].replace("set_0", "S") for row in ctat]
    counts = np.asarray([int(row["unique_microsegments"]) for row in ctat])
    single = np.asarray([100.0 * float(row["single_frame_fraction"]) for row in ctat])
    short = np.asarray([100.0 * float(row["duration_le_10ms_fraction"]) for row in ctat])
    xpos = np.arange(len(labels))
    ax_c.vlines(xpos, 5000, counts, color=PALETTE["light_blue"], linewidth=8.0, zorder=1)
    ax_c.scatter(xpos, counts, color=PALETTE["blue"], s=28, marker="s", zorder=2)
    ax_c.set_yscale("log")
    ax_c.set_ylim(5000, 300000)
    ax_c.set_ylabel("Positive microsegments (log scale)")
    ax_c.set_xticks(xpos, [f"{label}\n{count:,}\n{one:.0f}/{ten:.0f}" for label, count, one, ten in zip(labels, counts, single, short)])
    ax_c.set_xlabel("Task / segments / 1-frame vs <=10 ms share (%)")
    ax_c.set_yticks([10000, 100000], ["10k", "100k"])
    ax_c.set_title("CTAT: median 1 frame and 0 ms in every task", pad=7)
    ax_c.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.6)
    panel_label(ax_c, "c")
    return save_figure(fig, base)


def figure2(rows: list[dict[str, str]], base: Path) -> dict[str, dict[str, Any]]:
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 3.25))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.78, wspace=0.31)
    settings = [(budget, cooldown) for budget in (10.0, 30.0, 60.0) for cooldown in (0.1, 1.0, 5.0)]
    lookup = {(row["family"], float(row["budget_fae_h"]), float(row["cooldown_seconds"])): row for row in rows}
    metrics = [
        ("mechanism_macro_tpr", "Mechanism Macro-TPR", (0.0, 1.02)),
        ("der_100ms", "DER@100ms", (0.0, 0.36)),
        ("outer_normal_fae_h", "Outer-normal FAE/h", None),
    ]
    x = np.arange(len(settings))
    labels = [f"{budget:g}\n{cooldown:g}" for budget, cooldown in settings]
    for panel_index, (ax, (metric, ylabel, ylim)) in enumerate(zip(axes, metrics)):
        ax.axvspan(3.65, 4.35, color=PALETTE["highlight"], alpha=0.25, zorder=0)
        for separator in (2.5, 5.5):
            ax.axvline(separator, color=PALETTE["separator"], linewidth=0.7, zorder=0)
        for family in ("HGB", "TCN", "LR"):
            family_rows = [lookup[(family, budget, cooldown)] for budget, cooldown in settings]
            mean = np.asarray([float(row[f"{metric}_mean"]) for row in family_rows])
            low = np.asarray([float(row[f"{metric}_min"]) for row in family_rows])
            high = np.asarray([float(row[f"{metric}_max"]) for row in family_rows])
            ax.fill_between(x, low, high, color=COLORS[family], alpha=0.10, linewidth=0)
            ax.plot(x, mean, color=COLORS[family], marker={"HGB": "o", "TCN": "s", "LR": "^"}[family], markersize=3.4)
        if metric == "outer_normal_fae_h":
            ax.plot(x, np.asarray([budget for budget, _ in settings]), color=COLORS["ink"], linestyle="--", linewidth=1.0)
            ax.set_yscale("symlog", linthresh=1.0, linscale=0.8)
            ax.set_yticks([0, 1, 3, 10, 30, 100, 300], ["0", "1", "3", "10", "30", "100", "300"])
        elif ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(x, labels)
        ax.set_xlabel("Calibration budget / cooldown\n(FAE/h / s)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.6)
        ax.set_title(["Detection coverage changes rank", "HGB leads timely detection", "Normal burden can exceed budget"][panel_index], pad=6)
        panel_label(ax, chr(ord("a") + panel_index))
    handles = [Line2D([0], [0], color=COLORS[family], marker={"HGB": "o", "TCN": "s", "LR": "^"}[family], label=family) for family in ("HGB", "TCN", "LR")]
    handles.append(Line2D([0], [0], color=COLORS["ink"], linestyle="--", label="Calibration budget"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.97), ncol=4, frameon=False)
    return save_figure(fig, base)


def figure3(rows: list[dict[str, str]], base: Path) -> dict[str, dict[str, Any]]:
    fig, axes = plt.subplots(1, 3, figsize=(7.09, 2.95), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.77, wspace=0.18)
    for fold, ax in enumerate(axes):
        fold_rows = sorted((row for row in rows if int(row["fold"]) == fold), key=lambda row: int(row["rank_within_fold"]))
        x = np.arange(1, len(fold_rows) + 1)
        y = np.asarray([float(row["mechanism_tpr"]) for row in fold_rows])
        mean = float(fold_rows[0]["mechanism_macro_tpr"])
        q25 = float(fold_rows[0]["lower_quartile_tpr"])
        point_colors = [COLORS["risk"] if value == 0.0 else COLORS["HGB"] for value in y]
        ax.scatter(x, y, color=point_colors, s=18, zorder=3)
        ax.axhline(mean, color=COLORS["HGB"], linestyle="-", linewidth=1.0, label="Macro mean")
        ax.axhline(q25, color=COLORS["risk"], linestyle="--", linewidth=1.0, label="Lower quartile")
        ax.set_xlim(0.3, len(fold_rows) + 0.7)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("Mechanism rank within fold")
        ax.set_title(f"Fold {fold}: {sum(value == 0.0 for value in y)} zero-detection mechanism(s)")
        ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.6)
        panel_label(ax, chr(ord("a") + fold))
    axes[0].set_ylabel("Mechanism TPR")
    axes[0].legend(loc="lower right", frameon=False)
    return save_figure(fig, base)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    contract_path = args.contract.resolve()
    output = args.output.resolve()
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    files = {
        "ctat": data_dir / "ctat_label_granularity.csv",
        "road": data_dir / "road_rank_reversal.csv",
        "cards_sensitivity": data_dir / "cards_operational_sensitivity.csv",
        "cards_mechanism": data_dir / "cards_mechanism_tpr.csv",
        "main_table": data_dir / "main_result_table.csv",
        "contract": contract_path,
    }
    rows = {name: read_csv(path) for name, path in files.items()}
    shutil.copyfile(files["main_table"], output / "main_result_table.csv")
    figures = {
        "figure2": figure1(rows["contract"], rows["road"], rows["ctat"], figure_dir / "figure2_endpoint_applicability_and_rank_divergence"),
        "figure3": figure2(rows["cards_sensitivity"], figure_dir / "figure3_cards_operational_sensitivity"),
        "figure4": figure3(rows["cards_mechanism"], figure_dir / "figure4_cards_mechanism_tail_risk"),
    }
    manifest = {
        "status": "PASS",
        "backend": f"Python/Matplotlib {mpl.__version__}",
        "input_receipts": {name: {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in files.items()},
        "row_counts": {name: len(value) for name, value in rows.items()},
        "figures": figures,
        "boundaries": [
            "No cross-dataset pooled model score",
            "CTAT positive-label microsegments are not treated as physical attack campaigns",
            "TCN seed-fold observations are not treated as independent datasets",
        ],
    }
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
