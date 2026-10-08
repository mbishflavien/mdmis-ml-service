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

**A third source closed most of the gap.** [NASA JPL's ECOSTRESS Spectral
Library](https://speclib.jpl.nasa.gov) (ordered via their site, processed
by JPL staff over ~24h, not instant) adds 204 more VSWIR spectra —
`scripts/parse_ecostress.py` appends them onto the USGS-derived datasets
above (`data/raw/ecostress/ecospeclib_all_minerals.zip`, run after
`parse_usgs_splib.py`). Two real wins:
- **"Columbite Fe^2+Nb_2O_6" — first real `coltan` coverage** across all
  three libraries checked (RRUFF, USGS splib07, ECOSTRESS). Verified
  live: a real columbite spectrum through `/classify` correctly returns
  `coltan` at 72% confidence.
- **"Scheelite CaWO_4" — first real `wolframite`-class coverage in a
  VSWIR reflectance library.** Scheelite isn't literally wolframite, but
  every tungsten-deposit reference treats them as the two ore minerals
  for the same commodity (same skarn/vein systems, same exploration
  target) — the same kind of commodity-target grouping already used for
  `coltan` (columbite+tantalite) and `lithium` (four different pegmatite
  minerals), documented in `scripts/mineral_mapping.py`. Actual
  wolframite/ferberite/huebnerite VSWIR spectra are confirmed absent
  from both USGS splib07 and ECOSTRESS (checked directly under every
  spelling) — but that's not the whole story, see the `lab`/Raman note
  below. Verified live: a real scheelite spectrum correctly returns
  `wolframite` at 63% confidence.

**`gold` is the one real, unresolved case**, and it's worth separating
from the other two: this isn't a "haven't found the right library yet"
gap like coltan/wolframite were — native gold has no diagnostic
feature in *either* Raman or VSWIR reflectance (a metallic lattice has
no absorption bands the way oxides/silicates do), so no spectral library
will ever carry a usable "Gold" entry for a classifier to learn from
directly. The real-world practice (and the only honest path here) is
detecting gold *indirectly*, via the alteration/pathfinder minerals
that do have real signatures (pyrite, arsenopyrite, sericite, silica
veining) and flagging their co-occurrence — not something built yet,
and worth scoping separately before claiming any "gold detection."

**One correction to an earlier claim in this README:** `wolframite`
was never actually at zero everywhere — the `lab`/Raman model
(`mineral_classifier`, trained on RRUFF) has **7 real wolframite samples**
via Ferberite/Huebnerite Raman spectra, and predicts the class
correctly. The zero was specific to VSWIR reflectance libraries
(USGS/ECOSTRESS), which is what matters for the `sentinel2`/`as7265x`
field sensors — but a Raman-capable lab reading was never blind to it.

Current (v3, combined USGS+ECOSTRESS, scheelite included) per-class
counts: `unknown` 185, `gemstone` 55, `copper` 46, `lithium` 39,
`beryl` 13, `cassiterite` 11, `coltan` 6, `wolframite` 6. `coltan` and
`wolframite` are both thin (6 samples each) — expect noisy per-class
metrics on a small held-out split until real field data backs them up —
but both existing as predictable classes *at all* is the real milestone;
v1 structurally could not output either one no matter what the input was.

**Accuracy is much lower than the Raman model** (59-62% vs. 93-97% held
out) — expected, not a bug: Sentinel-2/AS7265x give the classifier 7-18
broad bands to work with, versus a 200-point Raman fingerprint. Multispectral
reflectance is inherently less mineral-specific than Raman spectroscopy;
this is a known tradeoff in the remote-sensing literature, and the honest
numbers are in each `models/*.meta.json`.

### Sentinel-2 band-ratio detector (`app/band_ratios.py`)
Implements the SRS's own formulas — Iron Oxide (B11/B08), Carbonate
(B11/B12), Clay (B11/B8A) — a rule-based alteration-style flag that needs
**zero training data**. Gated on NDVI (computed from the same bands):
running it against the seeded Rwandan sites (`scripts/fetch_sentinel2.py`)
returned a uniform false-positive `carbonate_alteration` flag everywhere,
which turned out to be vegetation canopy, not geology — confirmed by the
one site where it worked correctly, Cuprite, NV (bare ground, correctly
flagged its real published alteration zone). Flags are now suppressed
above NDVI 0.2 rather than reported misleadingly.

**Correction (2026-10-08):** the first fetches ignored Sentinel-2's
−1000 offset (see "Reading translation layer"), so every value read ~0.1
too bright. Re-fetched through the Sentinel-2 adapter, Cuprite is NDVI
0.11 (still bare), iron-oxide ratio 1.22, clay ratio 1.19 — both still
fire, more strongly. The 5 Rwandan sites re-fetched so far are NDVI
0.49–0.88 *and* ESA's own scene classification labels each pixel
"vegetation" — the vegetation conclusion is stronger, not weaker. The
other 5 seeded sites haven't been re-fetched yet. This is also a real
argument for why the drone stays in the pipeline: at Sentinel-2's
10-60m resolution, vegetation cover hides mineral-alteration signals
that a low-flying drone could still see through to bare/disturbed ground.

### Gold pathfinder indicator (`app/pathfinder.py`, `POST /pathfinder`)

Deliberately **not** part of `/classify` or `MINERAL_CHOICES` — it never
outputs "gold". Native gold has no diagnostic feature in Raman or VSWIR
reflectance (a metallic lattice has no absorption bands), so nothing
built here ever will claim to detect it directly. What's built instead
is the same indirect signal real gold exploration actually runs on:
flagging the **alteration minerals** that correlate with gold systems —
iron-oxide gossan (hematite/goethite/jarosite), argillic alteration
(kaolinite/illite/alunite/pyrophyllite), and sulfide pathfinders
(pyrite/arsenopyrite) — against a background class, via a trained
classifier (`scripts/build_pathfinder_dataset.py` +
`scripts/pathfinder_mapping.py`, same USGS+ECOSTRESS sources, 309-310
rows). `pathfinder_score` sums the three gold-associated categories;
`category`/`category_alternatives` give the underlying breakdown. Every
response carries a `caveat` field repeating that this is a follow-up
flag, not a detection.

**Compared directly against `band_ratios.py` on the exact same real
Cuprite pixel** (chosen because the user explicitly asked "once we know
it is the best we got" — this is that comparison, not a guess):

| | iron-oxide signal | clay/argillic signal |
|---|---|---|
| `band_ratios.py` (rule-based) | **fired** (ratio 1.22 > 1.1) | **fired** (ratio 1.19 > 1.1) |
| `pathfinder_classifier_s2` (trained) | 29% | 24% |

(Offset-corrected values. The first version of this comparison used
uncorrected, too-bright readings and showed iron-oxide at only 6% — part
of the classifier's apparent "miss" was that bug, not the model.)

Both see iron-oxide and clay alteration, but the trained classifier
still ranks `background` first (46%), while the ratio approach is
unambiguous. The likely remaining reason: the classifier learned from
*pure* lab mineral specimens, while a real Sentinel-2 pixel is a
*mixture* (minerals diluted among soil/rock across 10-60m) — a
well-known remote-sensing problem (sub-pixel spectral mixing).

The translation layer's QC matters here too: run on the vegetated
Rwandan pixels with QC bypassed, the pathfinder called all five
`iron_oxide_gossan` at 85–88% — a confident false alarm on vegetation.
Through `POST /readings/sentinel2` those pixels are blocked before any
model runs.

**Recommendation, not yet acted on:** for `sentinel2`, `band_ratios.py`
is the better-validated signal right now — it's simpler, needs no
training data, and the one real-world test favors it. The trained
`pathfinder_classifier_s2` is a second opinion, not (yet) the primary
one. For `as7265x`, there's no competing rule-based alternative (the
SRS's ratio formulas need B11/B12, which AS7265x doesn't have), so the
trained classifier is the only option there — but AS7265x's 410-940nm
range structurally can't see the ~2200nm feature `argillic_alteration`
depends on most. (An earlier version of this note quoted a single
held-out split, f1 0.24 vs. 0.47, that overstated the gap; 5-fold
cross-validation grouped by physical specimen puts the two pathfinder
models close overall — macro-F1 0.58 vs. 0.63.) Bundling
this into v3 as a default-on signal should wait for validation against
more than one real site — right now it's a `/pathfinder` endpoint you
can call deliberately, not something wired into the main classify flow.

## Reading translation layer (`app/translation/`)

Each device in the IoT budget outputs something different (raw counts,
images, gamma channel counts, radar traces, µT). Before any model sees a
reading, a per-device **adapter** translates it into one common
`Observation` format (`app/translation/observation.py`): calibrated values
in standard units, location/depth/time, quality-check flags, and
provenance (which adapter and which calibration inputs produced it).

Design rule: **numbers are converted only by deterministic, tested code.**
AI is meant for the fuzzy parts around the numbers — identifying a file's
device/format, extracting units/metadata, drafting adapters for new
devices for human review, explaining QC failures — not for producing the
values (a misread number would spread silently into the 3D block). That
AI part isn't built yet; it needs Anthropic API access, which a Claude
Pro subscription doesn't include.

**AS7265x (done):** `POST /readings/as7265x` takes three readings — the
sample, a **dark** reading (light blocked) and a **white reference panel**
reading (Spectralon ≈0.99, PTFE ≈0.95), each as 18 numbers or the raw
SparkFun serial line — and computes
`reflectance = (sample − dark) / (white − dark) × panel_reflectance`.
This matters: the chip outputs light intensity, not reflectance, and the
models were trained on reflectance, so raw counts would give confident
nonsense. Readings with a saturated channel, white ≤ dark, or reflectance
far outside [0, 1] are flagged and **never sent to the models**;
otherwise the response includes the mineral and pathfinder results.
`tests/test_translation.py` simulates the chip's output for a real
training spectrum and checks the round trip recovers the exact
reflectance and the same prediction. QC thresholds are starting points
until real field readings exist.

**Field requirement:** every AS7265x session needs a white reference
panel and a dark reading taken with the same lamp, distance and gain as
the samples.

**Sentinel-2 (done):** `POST /readings/sentinel2` takes the raw L2A
digital numbers per band (as stored in the product, *not* reflectance),
the scene's processing baseline (STAC `s2:processing_baseline`) and,
optionally, the pixel's Scene Classification (SCL) value. It computes
`reflectance = (DN + offset) / 10000`, where the offset is −1000 for
baseline 04.00+ (ESA, Jan 2022) and 0 before — not just "divide by
10000", which is the bug the first fetches had. QC blocks no-data,
cloud, cloud shadow, cirrus, water, snow, dark-area and vegetation
pixels (SCL), plus anything with NDVI > 0.2. Passing readings get the
mineral model, the pathfinder and the SRS band ratios.
`scripts/fetch_sentinel2.py` now goes through this adapter. Nothing in
it is country-specific: the same adapter serves Rwanda and DRC sites.

**Terrain / 3D-block surface (done):** `app/translation/terrain.py`
loads any elevation GeoTIFF — a WebODM/OpenDroneMap DTM from the drone
(the real target, usually UTM-projected, cm–dm resolution) or the free
Copernicus DEM GLO-30 (30 m) as a stand-in until flights happen — and:
- produces the surface grid in the frontend's existing `DemGrid` shape
  (`frontend/lib/dem-fetcher.ts`: row 0 = south, col 0 = west, edges
  included), so `useDemTerrain` can render real terrain instead of the
  procedural noise the block draws today;
- places readings: `elevation = surface_elevation(lat, lon) − depth_m`,
  which is how a GPR/EM/sample reading at 12 m depth gets its position
  inside the block.

```bash
python scripts/ingest_terrain.py --site RW-RTG-01 --lat -1.7783 --lon 30.0611 --copernicus
python scripts/ingest_terrain.py --site RW-RTG-01 --lat -1.7783 --lon 30.0611 --dtm odm_dtm.tif   # after a drone flight
```

For many sites at once, `--sites-json sites.json --copernicus` (a list of
`{id, lat, lon}`) does a single tile search for all of them (the catalog
search is the slow part, ~90 s) and stitches every 1°×1° tile a site's
box touches — 4 of the 10 seeded sites straddle a tile edge, and reading
one tile left part of their block as gap-filled guesses. All 10 seeded
sites are ingested this way (0% filled); Lake Kivu shows up as a flat
1,461 m floor at the shoreline sites RW-KRG-09 and RW-RTS-10.

The 12 North Kivu sites imported from IPIS open data
(`MDMIS_BACKEND- app/import_ipis.py`) are ingested the same way, 0%
filled: Masisi/Rubaya highlands ~1,500–2,760 m, Walikale lowland forest
~550–870 m, two Lubero sites just north of the equator.

**Caveat — surface, not ground:** Copernicus GLO-30 is a *surface* model
(its files are named `..._DSM_...`). Over forest it measures the
treetops, not the soil — in Walikale's rainforest that can be 30–40 m
above the real ground, which also shifts every depth placed below it. A
drone DTM from WebODM (which separates bare ground from vegetation)
fixes this per site; until then, treat forested-site elevations and
depths as approximate.

Writes `data/terrain/<site>/` (gitignored). API: `GET /terrain/{site_id}`
→ DemGrid; `POST /terrain/{site_id}/place` with `[{lat, lon, depth_m}]`
→ surface and absolute elevation per reading. Holes in the source are
filled from the nearest valid cell and reported as `filledFraction`, so
a filled area is never mistaken for a measurement. Verified on real
Copernicus terrain for RW-RTG-01 (1,478–1,960 m) and on a synthetic
UTM-35S plane with exactly known heights (`tests/test_terrain.py`).
Accuracy caveat: a drone DTM without surveyed ground control points is
accurate in shape and depth but its absolute height can be off by metres
or more.

Not yet: RadiaCode, GPR (time→depth), VLF-EM, magnetometer, Pi NoIR,
drone photo metadata.

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

# sentinel2 + as7265x models — see "Data" above for getting usgs_splib/
# and ecospeclib_all_minerals.zip in place
python scripts/parse_usgs_splib.py
python scripts/parse_ecostress.py   # appends onto the same CSVs
python scripts/train.py --dataset data/processed/sentinel2_dataset.csv --model-name mineral_classifier_s2 --version v3
python scripts/train.py --dataset data/processed/as7265x_dataset.csv --model-name mineral_classifier_as7265x --version v3

# gold pathfinder indicator models (uses the same usgs_splib/ecostress data above)
python scripts/build_pathfinder_dataset.py
python scripts/train.py --dataset data/processed/pathfinder_sentinel2_dataset.csv --model-name pathfinder_classifier_s2 --version v1
python scripts/train.py --dataset data/processed/pathfinder_as7265x_dataset.csv --model-name pathfinder_classifier_as7265x --version v1

# optional: real Sentinel-2 band-ratio detector over real coordinates, no training needed
python scripts/fetch_sentinel2.py   # takes 15-20 min — Planetary Computer's STAC
                                     # search is slow (~90s/call) from this network

uvicorn app.main:app --reload --port 8100
```

## API

- `GET /health` — `{status, service, model_versions: {lab, sentinel2, as7265x}, pathfinder_versions: {sentinel2, as7265x}}`
- `POST /classify` (requires `X-ML-Service-Key` header) —
  `{x_values: [float], intensities: [float], sensor_type: "lab"|"sentinel2"|"as7265x"}` →
  `{mineral_type, confidence_score, confidence_alternatives, grade_pct, model_version}`.
  `x_values`/`intensities` are Raman shift (cm⁻¹) for `"lab"`, wavelength
  (nm) for the other two. `grade_pct` is always `null`: ore-grade
  quantification needs an instrument calibration curve this model doesn't
  have — that's a geologist/lab task, not something the classifier should
  guess at.
- `POST /pathfinder` (requires `X-ML-Service-Key` header) —
  `{x_values: [float], intensities: [float], sensor_type: "sentinel2"|"as7265x"}` →
  `{category, category_score, category_alternatives, pathfinder_score, caveat, model_version}`.
  See "Gold pathfinder indicator" above — separate from `/classify` on
  purpose, never returns "gold" as a value anywhere.

## Not yet built

- GPR, EM, magnetometer, gamma (RadiaCode) — each needs its own
  model/pipeline; per the SRS these are better used as depth/structure
  features feeding the 3D block than as standalone mineral classifiers.
- `scripts/retrain.py` only retrains the `lab` model — extending the loop
  to `sentinel2`/`as7265x` is the same pattern, just not done yet.
- The frontend has no live wiring to any of this yet (it's 100% mock
  data in `lib/mdmis-data.ts` as of this writing) — that's separate,
  larger work tracked outside this repo.
