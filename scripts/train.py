"""Trains the mineral classifier from data/processed/spectral_dataset.csv
and writes a versioned model + metrics/meta report to models/.

Classes with too few samples to stratify-split are dropped (reported, not
silently ignored) — this is the honest place a real coverage gap (e.g.
"gold" having 0 RRUFF Raman samples, see mineral_mapping.py) surfaces
before it can be trained into a fake classifier.

Usage: python scripts/train.py [--version v1]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.constants import GRID_MAX_CM, GRID_MIN_CM, GRID_POINTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "spectral_dataset.csv"
MODELS_DIR = ROOT / "models"
MIN_SAMPLES_PER_CLASS = 4  # below this, train_test_split can't stratify


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print(f"{DATA_PATH} not found — run scripts/build_dataset.py first.")
        sys.exit(1)
    return pd.read_csv(DATA_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    df = load_dataset()
    feature_cols = [c for c in df.columns if c.startswith("f")]

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, zero_division=0))

    # Refit on all usable data for the deployed artifact — the held-out
    # split above is only for the reported metrics.
    pipeline.fit(X, y)

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / f"mineral_classifier_{args.version}.joblib"
    joblib.dump(pipeline, model_path)

    meta = {
        "version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": sorted(pipeline.classes_.tolist()),
        "dropped_classes": {k: int(v) for k, v in dropped.items()},
        "feature_grid": {"min_cm": GRID_MIN_CM, "max_cm": GRID_MAX_CM, "points": GRID_POINTS},
        "training_samples": int(len(df)),
        "samples_per_class": {k: int(v) for k, v in counts.items()},
        "held_out_metrics": report,
        "source": "RRUFF Raman spectral database (excellent_unoriented + fair_unoriented tiers)",
    }
    meta_path = MODELS_DIR / f"mineral_classifier_{args.version}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
