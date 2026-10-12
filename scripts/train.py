"""Trains a mineral classifier from any of this service's processed
datasets and writes a versioned model + metrics/meta report to models/.
Works for all three sensor datasets (Raman/"lab", Sentinel-2, AS7265x) —
feature columns are whatever isn't a known metadata column, so a 200-point
Raman grid and a 7-band Sentinel-2 row both work unchanged.

Classes with too few samples to stratify-split are dropped (reported, not
silently ignored) — this is the honest place a real coverage gap (e.g.
"gold" having 0 RRUFF Raman samples, see mineral_mapping.py) surfaces
before it can be trained into a fake classifier.

Usage:
  python scripts/train.py --dataset data/processed/spectral_dataset.csv \\
      --model-name mineral_classifier --version v1 \\
      --source "RRUFF Raman (excellent+fair_unoriented)"
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipelines import DEPLOYED, FEATURES, MODELS, build_pipeline  # noqa: E402
from scripts.evaluate import specimen  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MIN_SAMPLES_PER_CLASS = 4  # below this, a class can't be stratified across folds
_METADATA_COLUMNS = {"label", "rruff_id", "mineral_name", "source"}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"{path} not found — run the matching build/fetch/parse script first.")
        sys.exit(1)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(ROOT / "data" / "processed" / "spectral_dataset.csv"))
    parser.add_argument("--model-name", default="mineral_classifier", help="artifact filename prefix")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--source", default="RRUFF Raman spectral database (excellent+fair_unoriented tiers)")
    parser.add_argument("--features", default="raw", choices=list(FEATURES),
                        help="feature recipe (app/pipelines.py); pick with scripts/evaluate.py --screen")
    parser.add_argument("--model", default="random_forest", choices=list(MODELS))
    parser.add_argument("--calibrate", choices=["sigmoid", "isotonic"], default=None)
    parser.add_argument("--deployed", choices=list(DEPLOYED),
                        help="use the recorded deployed setup for this model (overrides --features/--model/--calibrate)")
    args = parser.parse_args()

    df = load_dataset(Path(args.dataset))
    feature_cols = [c for c in df.columns if c not in _METADATA_COLUMNS]

    counts = df["label"].value_counts()
    usable_labels = counts[counts >= MIN_SAMPLES_PER_CLASS].index.tolist()
    dropped = counts[counts < MIN_SAMPLES_PER_CLASS]
    if len(dropped):
        print("Dropping classes with too few samples to train/evaluate:")
        for label, n in dropped.items():
            print(f"  {label}: {n} sample(s)")
    df = df[df["label"].isin(usable_labels)]

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy()

    # Held-out check grouped by physical specimen: a random row split puts
    # copies of the same rock on both sides and reports near-perfect scores
    # (it did — 1.00 for the lab model). scripts/evaluate.py gives the full
    # repeated estimate; this is the quick single-split version.
    groups = df.apply(specimen, axis=1).to_numpy()
    train_idx, test_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups))
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    if args.deployed:
        cfg = DEPLOYED[args.deployed]
        args.features, args.model, args.calibrate = cfg["features"], cfg["model"], cfg.get("calibrate")
    pipeline = build_pipeline(args.features, args.model, args.calibrate)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Refit on all usable data for the deployed artifact — the held-out
    # split above is only for the reported metrics.
    pipeline.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"{args.model_name}_{args.version}.joblib"
    joblib.dump(pipeline, model_path)

    meta = {
        "version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": sorted(pipeline.classes_.tolist()),
        "dropped_classes": {k: int(v) for k, v in dropped.items()},
        "feature_columns": feature_cols,
        "training_samples": int(len(df)),
        "samples_per_class": {k: int(v) for k, v in counts.items()},
        "held_out_metrics": report,
        "source": args.source,
        "features": args.features,
        "model": args.model,
        "calibration": args.calibrate,
    }
    meta_path = MODELS_DIR / f"{args.model_name}_{args.version}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
