"""
AM321 - Build the report section 3.1 comparison table from results/models/comparison.csv.

    python -m src.evaluation.report_table          # prints Markdown; also writes results/models/report_table.md

Columns follow the brief exactly. No number is typed by hand, so every value in the
report can be traced back to the CSV produced by src.models.train_all.
"""
import json

import pandas as pd

from src.models.model_zoo import ROOT

RES = ROOT / "results" / "models"

SHORT_CONFIG = {   # the hyperparameters worth showing in a narrow table cell
    "logreg": ["C", "class_weight"],
    "gaussian_nb": ["var_smoothing"],
    "knn": ["n_neighbors", "weights"],
    "random_forest": ["n_estimators", "min_samples_leaf", "max_features"],
    "lightgbm": ["n_estimators", "learning_rate", "num_leaves"],
    "svm_rbf": ["C", "gamma", "n_components"],
    "mlp": ["hidden_layer_sizes", "alpha", "early_stopping"],
}


def short_config(model, cfg_json, max_rows):
    cfg = json.loads(cfg_json)
    parts = [f"{k}={cfg[k]}" for k in SHORT_CONFIG.get(model, cfg) if k in cfg]
    if pd.notna(max_rows):
        parts.append(f"train rows={int(max_rows):,}")
    return ", ".join(parts)


def build():
    t = pd.read_csv(RES / "comparison.csv")
    sel = RES / "selected_model.csv"
    chosen = pd.read_csv(sel)["model"].iloc[0] if sel.exists() else None
    rows = []
    for _, r in t.iterrows():
        name = f"**{r['model']}** (selected)" if r["model"] == chosen else r["model"]
        rows.append({
            "Model": name,
            "Algorithm family": r["family"],
            "Key configuration": short_config(r["model"], r["config"], r.get("max_train_rows")),
            "CV Macro F1 (mean ± std)": f"{r['cv_macro_f1_mean']:.4f} ± {r['cv_macro_f1_std']:.4f}",
            "Top-1 Accuracy": f"{r['test_top1_accuracy']:.4f}",
            "Route continuity": f"{r['test_route_continuity']:.4f}",
            "Student-test Macro F1": f"{r['test_macro_f1']:.4f}",
            "Log Loss": f"{r['test_log_loss']:.4f}",
            "Inference time (ms / GPS point)": f"{r['inference_ms_per_obs']:.4f}",
        })
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md = "\n".join(lines)               # plain Markdown, no extra package needed
    (RES / "report_table.md").write_text(md + "\n")
    return md


if __name__ == "__main__":
    print(build())
