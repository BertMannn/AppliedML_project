from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
from typing import Any

from preprocessing.get_data import get_data

MODEL_PATH: str = "models/gradient_boosting.pkl"
OUTPUT_PATH: str = "uncertainty_eval.png"

UncertaintyDict = dict[str, Any]

def uncertainty(model_path: str = MODEL_PATH, n_bins: int = 10) -> UncertaintyDict:
    _, X_test, _, y_test, _ = get_data(return_numeric_columns=True)

    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")

    labels, proba = model.predict(X_test, return_probability=True)
    class_order = np.array(sorted(model.classes_.keys()))

    y_true = np.asarray(y_test)
    confidence = proba.max(axis=1)
    correct = (np.asarray(labels) == y_true).astype(int)

    y_onehot = (y_true[:, None] == class_order[None, :]).astype(float)
    brier = float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, edges) - 1, 0, n_bins - 1)

    mids: list[float] = []
    accs: list[float] = []
    confs: list[float] = []
    weights: list[float] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum():
            mids.append((edges[b] + edges[b + 1]) / 2)
            accs.append(correct[mask].mean())
            confs.append(confidence[mask].mean())
            weights.append(mask.sum() / len(confidence))

    ece = float(np.sum(np.array(weights) * np.abs(np.array(accs) - np.array(confs))))

    order = np.argsort(-confidence)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    ranks = np.arange(1, n + 1)
    coverage = ranks / n
    accuracy = np.cumsum(correct_sorted) / ranks
    overall_acc = float(accuracy[-1])

    return {
        "accuracy": accuracy,
        "coverage": coverage,
        "overall_acc": overall_acc,
        "bin_mids": np.array(mids),
        "bin_accs": np.array(accs),
        "n_bins": n_bins,
        "brier": brier,
        "ece": ece,
    }


def plot(results: UncertaintyDict, output_path: str = OUTPUT_PATH) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(
        results["bin_mids"],
        results["bin_accs"],
        width=(1 / results["n_bins"]) * 0.9,
        color="#1f5fbf",
        edgecolor="white",
        label="Accuracy per bin",
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(f"Calibration (Brier={results['brier']:.3f}, ECE={results['ece']:.3f})")
    ax1.legend(loc="upper left")

    ax2.plot(results["coverage"], results["accuracy"], color="#1f5fbf", linewidth=2)
    ax2.axhline(
        results["overall_acc"],
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"Overall accuracy = {results['overall_acc']:.3f}",
    )
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Coverage (fraction of most-confident predictions kept)")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Coverage vs Accuracy")
    ax2.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(
        f"Brier={results['brier']:.4f}  ECE={results['ece']:.4f}  overall_acc={results['overall_acc']:.4f}"
    )
    plt.show()


def main() -> None:
    results = uncertainty()
    plot(results)


if __name__ == "__main__":
    main()