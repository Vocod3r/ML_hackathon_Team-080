"""FastAPI service for trip map matching and server-side evaluation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.data.io import list_trips, load_trip
from src.data.labels import reference_alignment
from src.evaluation.metrics import route_continuity
from src.pipeline.predict_trip import predict_trip
from src.viz.route_map import create_route_map


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("PATHSNAP_API_CONFIG", ROOT / "configs/api.yaml"))


class NearestArcModel:
    """Fallback scorer for bringing up the service before model artifacts arrive."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        distance = np.asarray(X["d"], dtype=float)
        positive = np.exp(-0.5 * (distance / 20.0) ** 2)
        return np.column_stack((1.0 - positive, positive))


def _read_config():
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {}


def _load_runtime():
    cfg = _read_config()
    data_root = Path(cfg.get("trip_root", ROOT / "data/student_test"))
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    feature_cols = cfg.get("feature_cols")
    if feature_cols:
        feature_path = Path(feature_cols)
        if not feature_path.is_absolute():
            feature_path = ROOT / feature_path
        feature_cols = [line.strip() for line in feature_path.read_text().splitlines() if line.strip()]
    else:
        from src.data.features import FEATURE_COLS
        feature_cols = FEATURE_COLS

    model_path = cfg.get("model_path")
    model = None
    if model_path:
        path = Path(model_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            import joblib
            model = joblib.load(path)
    if model is None:
        model = NearestArcModel()

    hmm_path = Path(cfg.get("hmm_config", ROOT / "configs/hmm.yaml"))
    if not hmm_path.is_absolute():
        hmm_path = ROOT / hmm_path
    hmm_cfg = yaml.safe_load(hmm_path.read_text()) if hmm_path.exists() else {}
    return {"trip_root": data_root, "feature_cols": feature_cols, "model": model,
            "hmm_config": hmm_cfg or {}, "candidate_options": cfg.get("candidate_options", {})}


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("pathsnap.api")
app = FastAPI(title="PathSnap Map Matching API", version="1.0.0")
runtime = _load_runtime()


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/trips")
def get_trips():
    """List available trip directory names."""
    if not runtime["trip_root"].is_dir():
        raise HTTPException(status_code=503, detail=f"Trip directory not found: {runtime['trip_root']}")
    return {"trips": list_trips(runtime["trip_root"])}


@app.post("/predict")
def post_predict(request: PredictRequest):
    """Predict a trip, then evaluate against its reference route inside the API."""
    started = time.perf_counter()
    trip_dir = runtime["trip_root"] / request.trip_id
    if not trip_dir.is_dir() or trip_dir.resolve().parent != runtime["trip_root"].resolve():
        logger.info("predict trip_id=%s status=not_found", request.trip_id)
        raise HTTPException(status_code=404, detail="Unknown trip_id")
    try:
        result = predict_trip(trip_dir, runtime["model"], runtime["feature_cols"],
                              runtime["hmm_config"], candidate_options=runtime["candidate_options"])
        # Load route only after prediction, strictly for server-side scoring/map display.
        evaluated_trip = load_trip(trip_dir, load_route=True)
        alignment = reference_alignment(evaluated_trip)
        ref = alignment.set_index("obs_idx")["ref_arc"]
        predicted = result["predicted_path"].set_index("obs_idx")["arc_id"]
        classifier = result["classifier_path"].set_index("obs_idx")["arc_id"]
        all_obs = np.arange(len(evaluated_trip["track"]))
        pred_full = predicted.reindex(all_obs)
        cls_full = classifier.reindex(all_obs)
        trusted = alignment.set_index("obs_idx")["trusted"].reindex(all_obs).fillna(False)
        valid = pred_full.notna()
        metrics = {
            "top1_accuracy": float((pred_full[valid] == ref.reindex(all_obs)[valid]).mean()) if valid.any() else None,
            "top1_accuracy_trusted": float((pred_full[valid & trusted] == ref.reindex(all_obs)[valid & trusted]).mean()) if (valid & trusted).any() else None,
            "classifier_top1_accuracy": float((cls_full[cls_full.notna()] == ref.reindex(all_obs)[cls_full.notna()]).mean()) if cls_full.notna().any() else None,
            "candidate_miss_rate": float((~result["candidates"].merge(
                alignment[["obs_idx", "ref_arc"]], on="obs_idx", how="left"
            ).assign(is_hit=lambda frame: frame["arc_id"] == frame["ref_arc"])
             .groupby("obs_idx")["is_hit"].any().reindex(all_obs, fill_value=False)).mean()),
            "trusted_observations": int(trusted.sum()),
            "observations": int(len(all_obs)),
        }
        # Score route continuity against the trip-specific directed network.
        continuity_rows = result["predicted_path"].copy()
        continuity_rows["trip_id"] = request.trip_id
        metrics["route_continuity"] = route_continuity(
            continuity_rows.rename(columns={"arc_id": "arc_id"}), {request.trip_id: evaluated_trip["arcs"]})

        map_obj = create_route_map(evaluated_trip, reference_arcs=evaluated_trip["route"],
                                   classifier_arcs=result["classifier_path"]["arc_id"],
                                   predicted_arcs=result["predicted_path"]["arc_id"])
        elapsed = time.perf_counter() - started
        logger.info("predict trip_id=%s observations=%d candidates=%d elapsed_s=%.3f status=ok",
                    request.trip_id, len(all_obs), len(result["candidates"]), elapsed)
        return {"trip_id": request.trip_id,
                "gps_points": evaluated_trip["track"][:, :2].tolist(),
                "predicted_arcs": result["predicted_path"].to_dict(orient="records"),
                "route_geojson": map_obj.get_root().render(), "metrics": metrics,
                "timing": {"seconds": elapsed}}
    except HTTPException:
        raise
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.exception("predict trip_id=%s elapsed_s=%.3f status=error", request.trip_id, elapsed)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "ok"}

