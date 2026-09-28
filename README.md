# mdmis-ml-service

Standalone FastAPI microservice that classifies a mineral spectrum into
one of MDMIS's `MINERAL_CHOICES` (see `MDMIS_BACKEND-`'s `app/sites/models.py`),
sitting at the `SENSORS → ML (classification) → BACKEND → FRONTEND` point
in the architecture. Called by the backend's `POST /api/scans/{id}/classify`
route; not called directly by the frontend.

## Data

Seed training data is the public [RRUFF Raman spectral database](https://rruff.net)
(no login/API key) — `scripts/download_data.py` fetches its bulk
`excellent_unoriented`/`fair_unoriented` zips, `scripts/build_dataset.py`
filters out the ~35 mineral names mapped in `scripts/mineral_mapping.py`
onto MDMIS's 10 mineral classes and resamples every spectrum onto a
common 150-1300 cm⁻¹ grid.

**Known coverage gap:** native gold has essentially no Raman signal (metals
aren't Raman-active), so the `gold` class currently has **zero** training
samples and is dropped at train time with a printed warning — this is a
real geology limitation of Raman spectroscopy, not a bug. Detecting gold
in the field needs a different signal (associated alteration minerals,
XRF/geochemistry, or visual inspection), which is a documented gap for a
future iteration, not something this model fakes.

Several other classes (`cassiterite`, `wolframite`) have single-digit
sample counts from public data alone — the held-out metrics in
`models/*.meta.json` are honest about this. This is meant as a seed model
to bootstrap the feedback loop below, not a finished classifier.

## Retraining loop (incremental learning)

"Incremental" here means a **scheduled retrain on accumulated data**, not
per-sample online learning — `RandomForestClassifier` has no `partial_fit`,
and true online learning is far less robust on a dataset this small. When
a geologist confirms a `MineralZone` in MDMIS (`status="lab_confirmed"`),
run:

```bash
python scripts/retrain.py --new-version v2
```

This pulls every `lab_confirmed` zone + its original spectrum from the
backend's `GET /api/scans/retrain-data` endpoint, appends it to the seed
dataset, retrains, and writes `models/mineral_classifier_v2.joblib`. Bump
`MODEL_VERSION` in `.env` (or the backend's `ML_SERVICE_URL` config, once
you're ready to cut over) to switch which version is served.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # set SERVICE_API_KEY to match the backend's ML_SERVICE_API_KEY

python scripts/download_data.py    # ~300MB, one-time
python scripts/build_dataset.py
python scripts/train.py --version v1

uvicorn app.main:app --reload --port 8100
```

## API

- `GET /health` — `{status, service, model_version}`
- `POST /classify` (requires `X-ML-Service-Key` header) —
  `{x_values: [float], intensities: [float], sensor_type: "lab"}` →
  `{mineral_type, confidence_score, confidence_alternatives, grade_pct, model_version}`.
  `grade_pct` is always `null`: ore-grade quantification needs an
  instrument calibration curve this model doesn't have — that's a
  geologist/lab task, not something the classifier should guess at.

## Not yet built

- Only Raman ("lab" sensor_type) spectra are supported. `hyperspectral`,
  `gpr`, `em`, `magnetometer`, `gamma`, `satellite` each need their own
  model/pipeline — out of scope for this v1.
- The frontend has no live wiring to any of this yet (it's 100% mock
  data in `lib/mdmis-data.ts` as of this writing) — that's separate,
  larger work tracked outside this repo.
