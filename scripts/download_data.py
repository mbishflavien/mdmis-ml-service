"""Downloads the public RRUFF bulk Raman-spectra archives used as this
model's seed dataset. No login/API key required.

RRUFF (rruff.net) publishes its whole Raman collection as a few quality-
graded zip bundles (each mixing every mineral together); we don't try to
hit per-mineral endpoints — we just cache the bundles here and let
build_dataset.py filter out the ~35 mineral names in mineral_mapping.py
from inside the zip.

Usage: python scripts/download_data.py
"""
import sys
from pathlib import Path
from urllib.request import Request, urlopen

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "rruff_raman"
BASE_URL = "https://www.rruff.net/zipped_data_files/raman/"

# "excellent" + "fair" unoriented single-crystal/powder spectra — good
# coverage without pulling every quality tier (~300MB combined vs >600MB
# for all tiers). "unoriented" avoids duplicate per-crystallographic-axis
# spectra of the same sample, which would bias the class balance.
TIERS = ["excellent_unoriented.zip", "fair_unoriented.zip"]


def download(tier: str) -> None:
    dest = RAW_DIR / tier
    if dest.exists():
        print(f"[skip] {tier} already downloaded ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = BASE_URL + tier
    print(f"[download] {url}")
    req = Request(url, headers={"User-Agent": "mdmis-ml-service/1.0 (dataset fetch)"})
    tmp = dest.with_suffix(".part")
    with urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        written = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            written += len(chunk)
            if total:
                print(f"\r  {written / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)
    print(f"[done] saved to {dest}")


def main() -> None:
    for tier in TIERS:
        download(tier)


if __name__ == "__main__":
    sys.exit(main())
