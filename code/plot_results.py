"""Run from project root: python code/plot_results.py."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
METRICS = ("train_loss", "test_loss", "train_acc", "test_acc")


def read_report(path):
    with path.open(encoding="utf-8") as file:
        report = json.load(file)
    for metric in METRICS:
        values = np.asarray(report[metric], dtype=float)
        if values.ndim != 1 or len(values) != report["epochs"]:
            raise ValueError(f"{path.name}: invalid length for {metric}")
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: non-finite values in {metric}")
    return report


def load_results(results_dir):
    series = {}
    single_path = results_dir / "single.json"
    if single_path.exists():
        series["Single (1 process)"] = read_report(single_path)

    for world_size in (2, 4):
        path = results_dir / f"DDP_{world_size}.json"
        if path.exists():
            series[f"DDP ({world_size} processes)"] = read_report(path)
        elif list(results_dir.glob(f"DDP_{world_size}_*.json")):
            raise ValueError(
                f"Missing {path.name}: DDP must save a combined report before plotting."
            )

    for method in ("ps", "ring"):
        for world_size in (2, 4):
            path = results_dir / f"{method}_{world_size}.json"
            if path.exists():
                series[f"{method.upper()} ({world_size} processes)"] = read_report(path)
    if not series:
        raise ValueError(f"No supported result files found in {results_dir}")
    return series


def plot_metrics(series, output_dir, metric_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for axis, split in zip(axes, ("train", "test")):
        for label, report in series.items():
            values = report[f"{split}_{metric_name}"]
            epochs = range(1, len(values) + 1)
            axis.plot(epochs, values, marker="o", markersize=4, label=label)
        axis.set_title(f"{split.capitalize()} {metric_name}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss" if metric_name == "loss" else "Accuracy (%)")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if metric_name == "acc":
            axis.set_title(f"{split.capitalize()} accuracy")
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for extension in ("png", "pdf"):
        path = output_dir / f"{metric_name}_vs_epochs.{extension}"
        fig.savefig(path, dpi=200)
        print(f"Saved {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "plots")
    args = parser.parse_args()
    series = load_results(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Configurations: " + ", ".join(series))
    plot_metrics(series, args.output_dir, "loss")
    plot_metrics(series, args.output_dir, "acc")


if __name__ == "__main__":
    main()
