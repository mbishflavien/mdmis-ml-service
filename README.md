# mdmis-ml-service

Standalone FastAPI microservice that classifies a mineral spectrum into
one of MDMIS's `MINERAL_CHOICES` (see `MDMIS_BACKEND-`'s `app/sites/models.py`),
sitting at the `SENSORS → ML (classification) → BACKEND → FRONTEND` point
in the architecture. Called by the backend's `POST /api/scans/{id}/classify`
route; not called directly by the frontend.

## Which sensors this actually targets

Per `MDMIS_IoT_Budget.docx` (downloaded 2026-09-16, the "Bare-Minimum DIY
Kit"), the pilot's real sensing hardware is:

| Sensor | Hardware status | Model in this repo |
|---|---|---|
| **Sentinel-2 satellite** | Real, $0 cost, no hardware needed | `mineral_classifier_s2` |
| **AS7265x breakout** | Real, $70 | `mineral_classifier_as7265x` |
| Lab/Raman (SRS's ground-truth validation role) | N/A — bench reference only | `mineral_classifier` |
| Pi NoIR + filters | Real, $90 | Only 3-4 bands — too coarse to model separately |
| RadiaCode gamma spectrometer | Real, $650 | Different modality (gamma energy spectrum), not built yet |
| DIY GPR / VLF-EM / magnetometer | Real, self-built | Subsurface structure/conductivity — depth signals, best fused as supporting features rather than standalone mineral classifiers (see SRS) |

`sensor_type` on `POST /classify` picks which of the three trained models
and band grids is used — `"lab"`, `"sentinel2"`, or `"as7265x"`. An
unsupported sensor_type (gpr/em/magnetometer/gamma) returns a clear 400
rather than silently routing to the wrong model.

## Data

All three models are trained on the **[USGS Digital Spectral Library
splib07](https://www.sciencebase.gov/catalog/item/586e8c88e4b0f5ce109fccae)**
— the reference library most mineral-mapping remote-sensing research is
actually built on. Downloaded manually (ScienceBase sits behind a
Cloudflare bot-challenge that blocks scripted access — confirmed, not
guessed — but works fine in a real browser):

1. `ASCIIdata_splib07a.zip` ("measured spectra", 20.8MB) — raw lab
   reflectance, 0.2-3.0µm depending on instrument. Used for `as7265x`:
   `scripts/parse_usgs_splib.py` resamples it onto the AS7265x's 18 exact
   band centers (only ASD/BECK-instrument samples cover that 410-940nm
   range — NIC4 starts at 1.12µm, too far into the SWIR).
2. `ASCIIdata_splib07b_rsSentinel2.zip` ("resampled to Sentinel-2 MSI
   characteristics", 1.4MB) — **USGS already did the Sentinel-2
   band-resampling themselves**, onto all 13 official bands. Used for
   `sentinel2` close to as-is (see `app/sensor_bands.py` for which bands
   are kept — B01/B09/B10 are dropped, they're atmospheric-correction
   support bands and B10 isn't even in the live L2A product).

Both unzip into `data/raw/usgs_splib/` (`ASCIIdata_splib07a/` and
`ASCIIdata_splib07b_rsSentinel2/` subfolders), then:

```bash
python scripts/parse_usgs_splib.py
python scripts/train.py --dataset data/processed/sentinel2_dataset.csv \
    --model-name mineral_classifier_s2 --version v1 --source "USGS splib07b resampled to Sentinel-2 MSI"
python scripts/train.py --dataset data/processed/as7265x_dataset.csv \
    --model-name mineral_classifier_as7265x --version v1 --source "USGS splib07a, resampled to AS7265x bands"
```

**Known coverage gap, confirmed independently twice:** `gold`, `coltan`,
and `wolframite` have **zero** usable samples in both USGS splib07 *and*
the earlier RRUFF Raman database. Native gold is Raman/reflectance-silent
(metals don't produce diagnostic spectral features by either method), and
neither library happens to carry columbite-tantalite or wolframite-series
entries. This isn't one dataset's gap — it's a real hole in public
spectral libraries for these three ore minerals that only your own
lab-confirmed field samples (via the retrain loop below) will close.

**Accuracy is much lower than the Raman model** (59-62% vs. 93-97% held
out) — expected, not a bug: Sentinel-2/AS7265x give the classifier 7-18
broad bands to work with, versus a 200-point Raman fingerprint. Multispectral
reflectance is inherently less mineral-specific than Raman spectroscopy;
this is a known tradeoff in the remote-sensing literature, and the honest
numbers are in each `models/*.meta.json`.

**A second source, in progress, not yet needed:** NASA JPL's ECOSTRESS
Spectral Library (order placed 2026-10-04, processed by JPL staff — email
said "within a week") — a nice-to-have second opinion once it lands, not
a blocker anymore now that USGS splib is in.

### Sentinel-2 band-ratio detector (`app/band_ratios.py`)
Implements the SRS's own formulas — Iron Oxide (B11/B08), Carbonate
(B11/B12), Clay (B11/B8A) — a rule-based alteration-style flag that needs
**zero training data**. Gated on NDVI (computed from the same bands):
running it against the 10 real seeded Rwandan sites
(`scripts/fetch_sentinel2.py`) returned a uniform false-positive
`carbonate_alteration` flag everywhere, which turned out to be vegetation
canopy (NDVI 0.26-0.55), not geology — confirmed by the one site where it
worked correctly, Cuprite, NV (NDVI 0.07, bare ground, correctly flagged
its real published alteration zone). Flags are now suppressed above
NDVI 0.2 rather than reported misleadingly. This is also a real argument
for why the drone stays in the pipeline: at Sentinel-2's 10-60m
resolution, Rwanda's vegetation cover hides mineral-alteration signals
that a low-flying drone could still see through to bare/disturbed ground.

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
dataset, retrains, and writes `models/mineral_classifier_v2.joblib`
(the `lab`/Raman model only, currently — extending this to the other two
sensors is the same pattern, not yet wired). Bump `MODEL_VERSION` in
`.env` to switch which version is served.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # set SERVICE_API_KEY to match the backend's ML_SERVICE_API_KEY

# lab/Raman model
python scripts/download_data.py    # ~300MB, one-time
python scripts/build_dataset.py
python scripts/train.py --version v1

# sentinel2 + as7265x models — see "Data" above for getting usgs_splib/ in place
python scripts/parse_usgs_splib.py
python scripts/train.py --dataset data/processed/sentinel2_dataset.csv --model-name mineral_classifier_s2 --version v1
python scripts/train.py --dataset data/processed/as7265x_dataset.csv --model-name mineral_classifier_as7265x --version v1

# optional: real Sentinel-2 band-ratio detector over real coordinates, no training needed
python scripts/fetch_sentinel2.py   # takes 15-20 min — Planetary Computer's STAC
                                     # search is slow (~90s/call) from this network

uvicorn app.main:app --reload --port 8100
```

## API

- `GET /health` — `{status, service, model_versions: {lab, sentinel2, as7265x}}`
- `POST /classify` (requires `X-ML-Service-Key` header) —
  `{x_values: [float], intensities: [float], sensor_type: "lab"|"sentinel2"|"as7265x"}` →
  `{mineral_type, confidence_score, confidence_alternatives, grade_pct, model_version}`.
  `x_values`/`intensities` are Raman shift (cm⁻¹) for `"lab"`, wavelength
  (nm) for the other two. `grade_pct` is always `null`: ore-grade
  quantification needs an instrument calibration curve this model doesn't
  have — that's a geologist/lab task, not something the classifier should
  guess at.

## Not yet built

- GPR, EM, magnetometer, gamma (RadiaCode) — each needs its own
  model/pipeline; per the SRS these are better used as depth/structure
  features feeding the 3D block than as standalone mineral classifiers.
- `scripts/retrain.py` only retrains the `lab` model — extending the loop
  to `sentinel2`/`as7265x` is the same pattern, just not done yet.
- The frontend has no live wiring to any of this yet (it's 100% mock
  data in `lib/mdmis-data.ts` as of this writing) — that's separate,
  larger work tracked outside this repo.
