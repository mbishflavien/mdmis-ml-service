"""Feature recipes x model choices, shared by scripts/evaluate.py (which
compares them honestly) and scripts/train.py (which saves the winner).
The feature step lives *inside* the saved sklearn Pipeline, so a live
request is transformed exactly like the training data — nothing in
app/model.py has to know which recipe a model uses.

Feature recipes:
  raw   — the values as they come (reflectance per band, or the
          normalised Raman spectrum).
  shape — mineral identity lives in the spectrum's *shape*; overall
          brightness mostly reflects grain size, wetness and lighting.
          So: spectrum divided by its own mean (brightness removed),
          slopes between neighbouring bands, log band-to-band ratios
          (the SRS's band-ratio idea, for every band pair), plus one
          log-brightness value so albedo isn't thrown away entirely.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

_EPS = 1e-6


def shape_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    pos = np.clip(X, _EPS, None)  # reflectance can dip just below 0 from noise
    mean = pos.mean(axis=1, keepdims=True)
    norm = pos / mean
    slopes = np.diff(norm, axis=1)
    logx = np.log(pos)
    i, j = np.triu_indices(X.shape[1], k=1)
    if len(i) > 400:  # long spectra (200-point Raman): ratios of every pair would be ~20k columns
        i, j = i[:: len(i) // 400 + 1], j[:: len(j) // 400 + 1]
    log_ratios = logx[:, i] - logx[:, j]
    return np.hstack([norm, slopes, log_ratios, np.log(mean)])


FEATURES = {
    "raw": lambda: FunctionTransformer(),
    "shape": lambda: FunctionTransformer(shape_features),
}

# class_weight="balanced" where supported: "unknown"/"background" outnumber
# the target minerals several times over.
MODELS = {
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
    "extra_trees": lambda: ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1),
    "gradient_boosting": lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
    "svm_rbf": lambda: SVC(C=10, gamma="scale", class_weight="balanced", probability=True, random_state=42),
    "logistic": lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
    # Spectral Angle Mapper in nearest-neighbour form (named in the SRS):
    # cosine distance between spectra is a monotone function of their angle.
    "sam_knn": lambda: KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance"),
}


def build_pipeline(features: str = "raw", model: str = "random_forest", calibrate: str | None = None):
    pipe = Pipeline([
        ("features", FEATURES[features]()),
        # Standardising each feature would distort the spectral angle SAM
        # compares, so SAM gets the vectors as they are.
        ("scaler", "passthrough" if model == "sam_knn" else StandardScaler()),
        ("clf", MODELS[model]()),
    ])
    if calibrate is None:
        return pipe
    # The confidence % shown to users drives auto-accept decisions, so it has
    # to mean what it says. The multispectral mineral models were
    # overconfident (said >=90% but were right ~70% of the time); isotonic
    # calibration learns how much to discount their raw scores. cv=3 because
    # some minerals have too few samples per training fold for more.
    return CalibratedClassifierCV(pipe, method=calibrate, cv=3)


# What each deployed model uses, chosen with scripts/evaluate.py (grouped,
# repeated and nested CV) plus a confidence-calibration check. Recorded here
# so evaluation, training and retraining all build the same thing.
DEPLOYED = {
    "lab": dict(features="raw", model="sam_knn"),
    "sentinel2": dict(features="shape", model="sam_knn", calibrate="isotonic"),
    "as7265x": dict(features="shape", model="gradient_boosting", calibrate="isotonic"),
    "pathfinder_sentinel2": dict(features="shape", model="logistic"),
    "pathfinder_as7265x": dict(features="shape", model="logistic"),
}
