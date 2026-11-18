"""
AM321 - Model comparison plots for report section 3.

Run from the repo root AFTER src.models.train_all has produced results:
    python -m src.evaluation.plots

Reads   results/models/comparison.csv, model_selection.csv, runs/*_folds.csv,
        probs/*_<test>.parquet, histories/*.json
Writes  results/models/plots/*.png

Every number plotted comes from those files (no hand-typed values), so each graph
in the report can be regenerated from the repo.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # write files only, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "models"
OUT = RES / "plots"

# ---- style: one accent colour for "a model", a second only for the selected model
# or for a second series (train vs validation). Validated colour-blind safe pair.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_results():
    table = pd.read_csv(RES / "comparison.csv")                     # every metric per model
    sel_path = RES / "model_selection.csv"
    if sel_path.exists():                                            # add the selected flag
        sel = pd.read_csv(sel_path)[["model", "selected"]]
        table = table.merge(sel, on="model", how="left")
    else:
        table["selected"] = False
    table["selected"] = table["selected"].astype(str).str.lower().eq("true")
    folds = pd.concat([pd.read_csv(p) for p in sorted((RES / "runs").glob("*_folds.csv"))],
                      ignore_index=True)
    return table, folds


def label(row):
    return f"{row['model']}  ★" if row["selected"] else row["model"]


def bar_colors(df):
    return [ORANGE if s else BLUE for s in df["selected"]]


def finish(fig, name, note=None):
    if note:
        fig.text(0.0, -0.02, note, fontsize=8, color=INK_2, ha="left", va="top")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("  wrote", OUT / name)


SELECT_NOTE = "★ / orange = model selected by the CV rule (see model_selection.csv)."


# ---------------------------------------------------------------------------
# 1. Cross-validation macro F1 (with fold spread)
# ---------------------------------------------------------------------------
def plot_cv_f1(table, folds):
    df = table.sort_values("cv_macro_f1_mean")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(df) + 1.2))
    ax.barh(y, df["cv_macro_f1_mean"], height=0.6, color=bar_colors(df))
    ax.errorbar(df["cv_macro_f1_mean"], y, xerr=df["cv_macro_f1_std"], fmt="none",
                ecolor=INK_2, elinewidth=1, capsize=3)
    for i, m in enumerate(df["model"]):                       # each fold as a dot
        f = folds.loc[folds["model"] == m, "val_macro_f1"]
        ax.scatter(f, np.full(len(f), i), s=14, color=INK, alpha=0.55, zorder=3, linewidths=0)
    for i, v in enumerate(df["cv_macro_f1_mean"]):
        ax.text(0.01, i, f"{v:.3f}", va="center", ha="left", color="white", fontsize=9, fontweight="bold")
    ax.set_yticks(y, [label(r) for _, r in df.iterrows()])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1 (GroupKFold by trip, training trips)")
    ax.set_title("Cross-validation Macro F1 by model")
    finish(fig, "01_cv_macro_f1.png",
           "Bar = mean over folds, whisker = ±1 std, dots = individual folds. " + SELECT_NOTE)


# ---------------------------------------------------------------------------
# 2. Student-test Top-1 accuracy and Macro F1 (two panels, shared model axis)
# ---------------------------------------------------------------------------
def plot_test_scores(table):
    df = table.sort_values("test_top1_accuracy")
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(df) + 1.2), sharey=True)
    for ax, col, title in [(axes[0], "test_top1_accuracy", "Top-1 accuracy (per GPS point)"),
                           (axes[1], "test_macro_f1", "Macro F1 (per candidate)")]:
        ax.barh(y, df[col], height=0.6, color=bar_colors(df))
        for i, v in enumerate(df[col]):
            ax.text(v - 0.01, i, f"{v:.3f}", va="center", ha="right", color="white", fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=11)
    axes[0].set_yticks(y, [label(r) for _, r in df.iterrows()])
    part = table["test_partition"].iloc[0] if "test_partition" in table else "student_test"
    fig.suptitle(f"Held-out performance on the {part} partition", fontweight="bold", x=0.02, ha="left")
    finish(fig, "02_student_test_top1_f1.png", SELECT_NOTE)


# ---------------------------------------------------------------------------
# 3. Log loss (CV and student-test). Log scale: Naive Bayes is ~50x worse than the rest.
# ---------------------------------------------------------------------------
def plot_log_loss(table):
    df = table.sort_values("test_log_loss", ascending=False)
    y = np.arange(len(df))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(df) + 1.3))
    if "cv_log_loss" in df:
        ax.barh(y + h / 2 + 0.02, df["cv_log_loss"], height=h, color=BLUE, label="Cross-validation (training trips)")
    ax.barh(y - h / 2 - 0.02, df["test_log_loss"], height=h, color=ORANGE, label="Student-test trips")
    for i, (_, r) in enumerate(df.iterrows()):
        if "cv_log_loss" in df:
            ax.text(r["cv_log_loss"] * 1.08, i + h / 2 + 0.02, f"{r['cv_log_loss']:.3f}", va="center", fontsize=8, color=INK_2)
        ax.text(r["test_log_loss"] * 1.08, i - h / 2 - 0.02, f"{r['test_log_loss']:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_xscale("log")
    ax.set_yticks(y, [label(r) for _, r in df.iterrows()])
    ax.set_xlabel("Log loss (log scale, lower is better)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=8)
    ax.set_title("Probability quality: log loss by model", pad=28)
    ax.margins(x=0.25)
    finish(fig, "03_log_loss.png", "★ = selected model. Colours here mean data split, not selection.")


# ---------------------------------------------------------------------------
# 4. Overfitting check: training F1 vs CV F1 (dumbbell)
# ---------------------------------------------------------------------------
def plot_overfitting(table):
    df = table.sort_values("cv_macro_f1_mean")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(df) + 1.2))
    ax.hlines(y, df["cv_macro_f1_mean"], df["cv_train_macro_f1_mean"], color=GRID, linewidth=3, zorder=1)
    ax.scatter(df["cv_train_macro_f1_mean"], y, s=60, color=ORANGE, zorder=3,
               edgecolors="white", linewidths=2, label="Training folds (fit data)")
    ax.scatter(df["cv_macro_f1_mean"], y, s=60, color=BLUE, zorder=3,
               edgecolors="white", linewidths=2, label="Validation folds (unseen trips)")
    for i, (_, r) in enumerate(df.iterrows()):
        gap = r["cv_train_macro_f1_mean"] - r["cv_macro_f1_mean"]
        ax.text(max(r["cv_train_macro_f1_mean"], r["cv_macro_f1_mean"]) + 0.012, i,
                f"gap {gap:+.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [label(r) for _, r in df.iterrows()])
    lo = max(0, df[["cv_macro_f1_mean", "cv_train_macro_f1_mean"]].min().min() - 0.05)
    ax.set_xlim(lo, 1.1)                                   # room on the right for the gap labels
    ax.set_xticks([t for t in np.arange(0, 1.0001, 0.05) if t >= lo])   # F1 never exceeds 1
    ax.set_xlabel("Macro F1 (mean over CV folds)")
    ax.set_title("Overfitting check: training vs validation F1", pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=8)
    finish(fig, "04_overfitting_train_vs_cv.png",
           "Large gap = overfitting (memorises training trips). Both low = underfitting. "
           "KNN with distance weights scores ~1.0 on its own training rows by construction.")


# ---------------------------------------------------------------------------
# 5. Speed vs accuracy (helps justify selection for a live API)
# ---------------------------------------------------------------------------
def plot_speed(table):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for _, r in table.iterrows():
        c = ORANGE if r["selected"] else BLUE
        ax.scatter(r["inference_ms_per_obs"], r["cv_top1_pooled"] if "cv_top1_pooled" in r else r["cv_top1_mean"],
                   s=70, color=c, edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(label(r), (r["inference_ms_per_obs"], r.get("cv_top1_pooled", r["cv_top1_mean"])),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per GPS point (ms, log scale)")
    ax.set_ylabel("CV Top-1 accuracy")
    ax.set_title("Accuracy vs inference speed")
    finish(fig, "05_speed_vs_accuracy.png", "Up and to the left is better. " + SELECT_NOTE)


# ---------------------------------------------------------------------------
# 6. Confusion matrices (student-test, threshold 0.5)
# ---------------------------------------------------------------------------
def plot_confusions(table):
    part = table["test_partition"].iloc[0] if "test_partition" in table else "student_test"
    models = [m for m in table.sort_values("cv_macro_f1_mean", ascending=False)["model"]
              if (RES / "probs" / f"{m}_{part}.parquet").exists()]
    if not models:
        print("  (no probability files - skipping confusion matrices)")
        return
    ncol = min(4, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, models):
        ax.set_visible(True)
        ax.grid(False)
        p = pd.read_parquet(RES / "probs" / f"{m}_{part}.parquet")
        cm = confusion_matrix(p["label"], p["prob"])
        rate = cm / cm.sum(axis=1, keepdims=True)              # row-normalised: recall per true class
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{rate[i, j]:.1%}\n({cm[i, j]:,})", ha="center", va="center", fontsize=8,
                        color="white" if rate[i, j] > 0.55 else INK)
        sel = bool(table.loc[table["model"] == m, "selected"].iloc[0])
        ax.set_title(f"{m}{'  ★' if sel else ''}", fontsize=10)
        ax.set_xticks([0, 1], ["pred: wrong arc", "pred: true arc"], fontsize=7)
        ax.set_yticks([0, 1], ["wrong arc", "true arc"], fontsize=7)
    fig.suptitle(f"Confusion matrices on {part} (threshold 0.5, row-normalised)",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    finish(fig, "06_confusion_matrices.png",
           "Cell = share of that true class (count in brackets). Diagonal = correct.")


# ---------------------------------------------------------------------------
# 7. Training curves - ONLY models that train iteratively (brief: no artificial curves)
# ---------------------------------------------------------------------------
def plot_histories(table):
    hdir = RES / "histories"
    for f in sorted(hdir.glob("*.json")) if hdir.exists() else []:
        m, h = f.stem, json.loads(f.read_text())
        if "val_loss" in h:                                    # LightGBM: train & student-test log loss
            fig, ax = plt.subplots(figsize=(7, 4))
            x = np.arange(1, len(h["train_loss"]) + 1)
            ax.plot(x, h["train_loss"], color=ORANGE, linewidth=2, label="Training trips")
            ax.plot(x, h["val_loss"], color=BLUE, linewidth=2, label="Student-test trips")
            best = int(np.argmin(h["val_loss"]))
            ax.scatter([x[best]], [h["val_loss"][best]], s=40, color=BLUE, zorder=3, edgecolors="white", linewidths=2)
            ax.annotate(f"lowest test loss {h['val_loss'][best]:.4f}\nat round {x[best]}",
                        (x[best], h["val_loss"][best]), xytext=(10, 20), textcoords="offset points",
                        fontsize=8, color=INK_2)
            ax.set_ylabel("Log loss")
            ax.set_xlabel(h.get("x_label", "iteration"))
            ax.set_title(f"{m}: training history")
            ax.legend(frameon=False)
            finish(fig, f"07_history_{m}.png",
                   "Recorded during the final refit; no early stopping, so the test curve did not influence training.")
        else:                                                  # MLP: train loss + internal-validation ACCURACY
            has_val = bool(h.get("val_accuracy"))
            fig, axes = plt.subplots(1, 2 if has_val else 1, figsize=(10 if has_val else 6, 3.8), squeeze=False)
            x = np.arange(1, len(h["train_loss"]) + 1)
            axes[0, 0].plot(x, h["train_loss"], color=ORANGE, linewidth=2)
            axes[0, 0].set_title("Training loss (log loss)", fontsize=11)
            axes[0, 0].set_xlabel(h.get("x_label", "epoch"))
            if has_val:
                axes[0, 1].plot(np.arange(1, len(h["val_accuracy"]) + 1), h["val_accuracy"], color=BLUE, linewidth=2)
                axes[0, 1].set_title("Internal validation accuracy (10% hold-out)", fontsize=11)
                axes[0, 1].set_xlabel(h.get("x_label", "epoch"))
            fig.suptitle(f"{m}: training history", fontweight="bold", x=0.02, ha="left")
            fig.tight_layout()
            finish(fig, f"07_history_{m}.png",
                   "sklearn's MLP records validation ACCURACY (not loss) for early stopping, so the units differ "
                   "and they are shown as separate panels.")


def main():
    table, folds = load_results()
    print(f"Plotting {len(table)} models -> {OUT}")
    plot_cv_f1(table, folds)
    plot_test_scores(table)
    plot_log_loss(table)
    plot_overfitting(table)
    plot_speed(table)
    plot_confusions(table)
    plot_histories(table)


if __name__ == "__main__":
    main()
