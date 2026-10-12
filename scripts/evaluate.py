"""Honest evaluation for every model in this repo — the yardstick any
"improvement" has to clear.

Rules, because the obvious shortcuts inflate scores here:
- Grouped by physical specimen. Libraries measure one rock several times
  (grain sizes, instruments); if copies of a rock sit in both train and
  test, the model is graded on rocks it has already seen.
- Repeated over several random splits, reported as mean ± std, so a lucky
  split doesn't look like progress.
- Compared against a majority-class baseline ("always guess the most
  common class") — accuracy alone flatters imbalanced data.
- --nested: when we *choose* the best of many configs, the winner's score
  is optimistic (we picked the luckiest). Nested CV picks inside each
  training fold and grades on the untouched test fold, giving an honest
  number for "the procedure of picking the best".

Usage:
  python scripts/evaluate.py                          # current deployed config, all datasets
  python scripts/evaluate.py --screen                 # every features x model combo
  python scripts/evaluate.py --nested --dataset sentinel2
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.pipelines import DEPLOYED, FEATURES, MODELS, build_pipeline  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"
DATASETS = {
    "lab": "spectral_dataset.csv",
    "sentinel2": "sentinel2_dataset.csv",
    "as7265x": "as7265x_dataset.csv",
    "pathfinder_sentinel2": "pathfinder_sentinel2_dataset.csv",
    "pathfinder_as7265x": "pathfinder_as7265x_dataset.csv",
}
META_COLS = {"label", "rruff_id", "mineral_name", "source"}
MIN_SPECIMENS = 5  # a class needs this many distinct rocks to be split 5 ways
SEEDS = (0, 1, 2)


def specimen(row) -> str:
    """Which physical rock a row was measured on."""
    if isinstance(row.get("rruff_id"), str) and row["rruff_id"]:
        return "rruff:" + row["rruff_id"]
    src = str(row["source"]).split(":")[-1]
    if src.startswith("mineral."):  # ECOSTRESS: mineral.class.sub.grain.range.SAMPLE.owner...
        return "eco:" + src.split(".")[5]
    if src.startswith(("splib07a_", "S07SNTL2_")):  # USGS: prefix_Mineral_SampleID_...
        return "usgs:" + "_".join(src.split("_")[1:3])
    return src


def load(name: str):
    df = pd.read_csv(ROOT / DATASETS[name])
    df["group"] = df.apply(specimen, axis=1)
    per_class = df.groupby("label")["group"].nunique()
    dropped = sorted(per_class[per_class < MIN_SPECIMENS].index)
    df = df[df["label"].isin(per_class[per_class >= MIN_SPECIMENS].index)]
    X = df[[c for c in df.columns if c not in META_COLS | {"group"}]].to_numpy(float)
    return X, df["label"].to_numpy(), df["group"].to_numpy(), dropped


def _scores(y_true, y_pred) -> dict:
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def cv_scores(make_model, X, y, groups, seeds=SEEDS) -> dict:
    runs = []
    for seed in seeds:
        y_pred = np.empty_like(y)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
            m = make_model()
            m.fit(X[tr], y[tr])
            y_pred[te] = m.predict(X[te])
        runs.append(_scores(y, y_pred))
    return {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs])) for k in runs[0]}


def nested_scores(candidates: dict, X, y, groups, seeds=SEEDS) -> tuple[dict, list]:
    runs, picks = [], []
    for seed in seeds:
        y_pred = np.empty_like(y)
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
            inner = {name: cv_scores(make, X[tr], y[tr], groups[tr], seeds=(seed,))["macro_f1"][0]
                     for name, make in candidates.items()}
            best = max(inner, key=inner.get)
            picks.append(best)
            m = candidates[best]()
            m.fit(X[tr], y[tr])
            y_pred[te] = m.predict(X[te])
        runs.append(_scores(y, y_pred))
    return {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs])) for k in runs[0]}, picks


def fmt(s: dict) -> str:
    return "  ".join(f"{k}={m:.2f}±{sd:.2f}" for k, (m, sd) in s.items())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS), action="append")
    p.add_argument("--screen", action="store_true", help="every features x model combination")
    p.add_argument("--nested", action="store_true", help="honest score for picking the best combination")
    p.add_argument("--seeds", type=int, default=len(SEEDS), help="random splits to repeat over (nested is slow)")
    a = p.parse_args()
    seeds = tuple(range(a.seeds))

    for name in a.dataset or list(DATASETS):
        X, y, groups, dropped = load(name)
        print(f"\n=== {name}: {len(y)} rows, {len(set(groups))} specimens, {len(set(y))} classes"
              + (f" (dropped, <{MIN_SPECIMENS} specimens: {dropped})" if dropped else ""))
        print(f"  {'baseline (majority)':34s} {fmt(cv_scores(lambda: DummyClassifier(strategy='most_frequent'), X, y, groups))}")
        print(f"  {'original: raw + random_forest':34s} {fmt(cv_scores(lambda: build_pipeline('raw', 'random_forest'), X, y, groups))}")
        cfg = DEPLOYED[name]
        label = "deployed: " + " + ".join(v for v in cfg.values())
        print(f"  {label:34s} {fmt(cv_scores(lambda: build_pipeline(**cfg), X, y, groups))}")
        candidates = {f"{f} + {m}": (lambda f=f, m=m: build_pipeline(f, m)) for f in FEATURES for m in MODELS}
        if a.screen:
            for label, make in candidates.items():
                print(f"  {label:34s} {fmt(cv_scores(make, X, y, groups))}", flush=True)
        if a.nested:
            s, picks = nested_scores(candidates, X, y, groups, seeds=seeds)
            top = pd.Series(picks).value_counts().head(3).to_dict()
            print(f"  {'NESTED (pick best inside folds)':34s} {fmt(s)}\n    picked: {top}")


if __name__ == "__main__":
    main()
