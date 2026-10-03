# mdmis-ml-service

Standalone FastAPI microservice that classifies a mineral spectrum into
one of MDMIS's `MINERAL_CHOICES` (see `MDMIS_BACKEND-`'s `app/sites/models.py`),
sitting at the `SENSORS → ML (classification) → BACKEND → FRONTEND` point
in the architecture. Called by the backend's `POST /api/scans/{id}/classify`
route; not called directly by the frontend.

## Which sensors this actually targets

Per `MDMIS_IoT_Budget.docx` (downloaded 2026-09-16, the "Bare-Minimum DIY
Kit"), the pilot's real sensing hardware is:

| Sensor | Hardware status | What it measures |
|---|---|---|
| **Sentinel-2 satellite** | Real, $0 cost, no hardware needed | 13-band multispectral reflectance, 10-60m/px |
| **AS7265x breakout** | Real, $70 | 18-channel point spectrometer, 410-940nm |
| Pi NoIR + filters | Real, $90 | Only 3-4 bands — too coarse to model separately |
| RadiaCode gamma spectrometer | Real, $650 | Gamma energy spectrum (K/U/Th) — different modality, not built yet |
| DIY GPR / VLF-EM / magnetometer | Real, self-built | Subsurface structure/conductivity/magnetics — depth signals, not mineral-identity classifiers (see SRS: best fused as supporting features, not their own classifier) |

An earlier version of this service was trained on RRUFF **Raman**
spectroscopy data. That turned out to be the wrong physical signal —
Raman isn't one of the five field sensors in the budget; it maps to the
SRS's "Lab Spectrometer" row, which is explicitly defined as a *ground-
truth validation* input, not a field-sensing modality. That model is kept
as `models/mineral_classifier_v*.joblib` for that narrower lab/validation
role; it is **not** what the drone or AS7265x will produce.

## Data, per sensor

### Sentinel-2 (`app/sensor_bands.py::SENTINEL2_BANDS`)
- `scripts/fetch_sentinel2.py` pulls real band reflectance via Microsoft
  Planetary Computer's STAC API (free, no account needed for anonymous
  reads) at two sets of points: the 10 seeded Rwandan sites from
  `MDMIS_BACKEND-`'s `app/seed.py` (real coordinates; `primary_mineral`
  there is demo placeholder data, not field-verified — treat the output
  as "proof the pipeline works," not training labels), and the Cuprite,
  Nevada mining district (a real deposit with a published USGS alteration
  map, used as an external sanity check).
- `app/band_ratios.py` implements the SRS's own band-ratio formulas —
  Iron Oxide (B11/B08), Carbonate (B11/B12), Clay (B11/B8A) — a rule-
  based alteration-style flag that needs **zero training data** and works
  today. This is an honest "detected, not identified" signal: it flags
  where alteration geochemistry associated with mineralisation is
  present, not which MINERAL_CHOICES class.
- Supervised classification on top of band values needs labeled Sentinel-
  2-shaped spectra — see "Getting the ECOSTRESS data" below.

### AS7265x (`app/sensor_bands.py::AS7265X_BANDS`)
- The actual ground-level point spectrometer in the budget. Same training
  requirement as Sentinel-2: labeled reflectance spectra resampled to its
  18 exact band centers.

### Getting the ECOSTRESS data (**this is where you can help**)
[NASA JPL's ECOSTRESS Spectral Library](https://speclib.jpl.nasa.gov) has
~3,100 genuine VNIR-SWIR-TIR mineral reflectance spectra (0.35-15.4µm —
covers both Sentinel-2's and the AS7265x's ranges), free and public. I
spent real effort trying to script its bulk download and couldn't: the
download page is a JS shopping-cart UI with no underlying API (verified —
a `plone.restapi` `@search` probe found no indexed spectrum content, and
POSTing the visible form fields to its checkout endpoint didn't trigger a
file either). It needs an actual browser.

**Steps (~2 minutes):**
1. Go to https://speclib.jpl.nasa.gov/download
2. Click **"All Minerals (3104 files)"**
3. Click **"Checkout"**, then download the resulting zip
4. Unzip it into `data/raw/ecostress/` in this repo (so you have
   `data/raw/ecostress/*.spectrum.txt`)
5. Tell me it's there (or push it, or just run the next step yourself):
   ```bash
   python scripts/parse_ecostress.py   # -> data/processed/sentinel2_dataset.csv
                                        #    data/processed/as7265x_dataset.csv
   python scripts/train.py --dataset data/processed/sentinel2_dataset.csv \
       --model-name mineral_classifier_s2 --version v1 --source "ECOSTRESS speclib v1.0"
   python scripts/train.py --dataset data/processed/as7265x_dataset.csv \
       --model-name mineral_classifier_as7265x --version v1 --source "ECOSTRESS speclib v1.0"
   ```

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

# Raman/lab model (existing)
python scripts/download_data.py    # ~300MB, one-time
python scripts/build_dataset.py
python scripts/train.py --version v1

# Sentinel-2: real data, no hardware needed
python scripts/fetch_sentinel2.py   # takes 15-20 min — Planetary Computer's
                                     # STAC search is slow (~90s/call) from
                                     # this network; band reads themselves
                                     # are fast once the search returns

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

- GPR, EM, magnetometer, gamma (RadiaCode) — each needs its own
  model/pipeline; per the SRS these are better used as depth/structure
  features feeding the 3D block than as standalone mineral classifiers.
- Sentinel-2 and AS7265x classifiers are code-complete but need the
  ECOSTRESS data above to actually train.
- The frontend has no live wiring to any of this yet (it's 100% mock
  data in `lib/mdmis-data.ts` as of this writing) — that's separate,
  larger work tracked outside this repo.
