"""Plot saved evaluation predictions. Does not fit or retrain models."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "linear": "Linear regression", "random_forest": "Random forest",
    "MLP": "MLP", "GNN": "GNN",
    "MLP_kinetic": "Release-constrained MLP",
    "GNN_kinetic": "Release-constrained GNN",
    "higuchi": "Higuchi", "saturating_exponential": "Saturating exponential",
    "power_law": "Empirical power law",
}
KEYS = ["Time_h", "pH", "Formulation"]


def averaged(frame):
    if frame.duplicated(["model", "seed"] + KEYS).any():
        raise ValueError("Repeated prediction for the same model, seed and condition")
    if frame.groupby(["model"] + KEYS)["observed"].nunique().max() != 1:
        raise ValueError("Observed values differ between seeds")
    return frame.groupby(["model"] + KEYS, as_index=False).agg(
        observed=("observed", "first"), predicted=("predicted", "mean")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--powerlaw", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("exports/manuscript_figures"))
    args = parser.parse_args()
    original = pd.read_csv(args.analysis / "held_out_predictions.csv")
    original = original[original.target == "cumulative_percent_as_supplied"].copy()
    extra = pd.read_csv(args.powerlaw / "held_out_predictions.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    for protocol in ["forward_time", "held_out_profiles"]:
        points = averaged(original[original.protocol == protocol])
        new = extra[extra.protocol == protocol].copy()
        if new.duplicated(["model"] + KEYS).any():
            raise ValueError("Repeated benchmark predictions")
        for model in ["power_law", "saturating_exponential"]:
            reference = points[points.model == "saturating_exponential"]
            candidate = new[new.model == model]
            merged = reference.merge(candidate, on=KEYS, suffixes=("_old", "_new"), validate="one_to_one")
            if len(merged) != len(reference) or len(candidate) != len(reference):
                raise ValueError("Benchmark evaluation conditions do not match")
            if not np.allclose(merged.observed_old, merged.observed_new, rtol=1e-10, atol=1e-12):
                raise ValueError("Benchmark observed values do not match")
            if model == "saturating_exponential" and not np.allclose(
                merged.predicted_old, merged.predicted_new, rtol=1e-6, atol=1e-7
            ):
                raise ValueError("Exponential predictions differ between runs")
        power = new[new.model == "power_law"][points.columns]
        points = pd.concat([points, power], ignore_index=True)
        if set(points.model) != set(LABELS):
            raise ValueError("Unexpected or missing model")
        expected = 24 if protocol == "forward_time" else 36
        if not (points.groupby("model").size() == expected).all():
            raise ValueError("Unexpected number of held-out conditions")
        low = min(0, points.observed.min(), points.predicted.min())
        high = max(points.observed.max(), points.predicted.max())
        pad = (high - low) * 0.06
        limits = (low - pad, high + pad)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
        for letter, (ax, (model, label)) in zip("abcdefghi", zip(axes.flat, LABELS.items())):
            data = points[points.model == model]
            ax.plot(limits, limits, "--", color="0.4", linewidth=1, zorder=0)
            for ph, color in [(5.5, "#0072B2"), (6.5, "#D55E00"), (7.4, "#009E73")]:
                for formulation, marker in [("Test", "o"), ("Control", "^")]:
                    subset = data[(data.pH == ph) & (data.Formulation == formulation)]
                    ax.scatter(subset.observed, subset.predicted, c=color, marker=marker,
                               s=28, alpha=0.75, edgecolors="white", linewidths=0.3)
            ax.set_title(f"({letter}) {label}", fontsize=10)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(labelbottom=True, labelleft=True)
        fig.supxlabel("Observed cumulative release (analysis scale)")
        fig.supylabel("Predicted cumulative release (analysis scale)")
        handles = [plt.Line2D([], [], color=c, marker="s", linestyle="", label=f"pH {p}")
                   for p, c in [(5.5, "#0072B2"), (6.5, "#D55E00"), (7.4, "#009E73")]]
        handles += [plt.Line2D([], [], color="0.3", marker=m, linestyle="", label=l)
                    for m, l in [("o", "Loaded nanocarrier"), ("^", "Free citral")]]
        fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0.03, 0.03, 1, 0.95))
        stem = args.output / f"cumulative_release_{protocol}"
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        points.to_csv(args.output / f"{stem.name}_plotted_values.csv", index=False)
        plt.close(fig)
        print(f"Saved {stem}.png and .pdf; {expected} conditions per model")
    print("COMPLETED. Saved predictions were plotted; no models retrained.")


if __name__ == "__main__":
    main()
