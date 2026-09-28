"""Maps RRUFF mineral names (its `##NAMES=` field / filename prefix) onto
MDMIS's MINERAL_CHOICES labels (app.constants).

Mineralogical basis for each grouping:
- coltan: colloquial name for columbite-tantalite group ore, so both the
  Fe and Mn end-members of columbite and tantalite map here.
- wolframite: a solid-solution series between the iron end-member
  (ferberite) and manganese end-member (huebnerite); RRUFF catalogs the
  end-members, not "wolframite" itself.
- lithium: the lithium-bearing pegmatite minerals actually seen in the
  field (spodumene, petalite, amblygonite, lepidolite), not elemental
  lithium (which has no mineral form / Raman spectrum).
- cobalt: cobalt ore/secondary minerals.
- copper: copper ore/secondary minerals — native copper is excluded on
  purpose, see the "gold" note below.
- gemstone: gem-quality accessory minerals, deliberately excluding beryl
  (beryl is its own MINERAL_CHOICES class and would otherwise double-map
  since emerald/aquamarine are beryl varieties).
- unknown: common rock-forming/host-rock minerals used as the negative
  ("no target mineral present") class.

Known coverage gap, not a bug: native metals (gold, and to a lesser
extent copper) are Raman-silent — a metallic lattice has no
polarizability-changing vibrational modes, so RRUFF's Raman collection
has essentially no "Gold" entries. This mapping still lists "Gold" so
that gap surfaces explicitly in build_dataset.py's per-class coverage
report rather than silently vanishing. Detecting gold in the field
realistically needs a different signal (alteration-mineral association,
XRF/geochemistry, visual) — not Raman — and is a documented limitation
of this v1 model, not something to fake with synthetic labels.
"""

MINERAL_NAME_MAP: dict[str, str] = {
    "Cassiterite": "cassiterite",
    "Columbite-(Fe)": "coltan",
    "Columbite-(Mn)": "coltan",
    "Tantalite-(Fe)": "coltan",
    "Tantalite-(Mn)": "coltan",
    "Ferberite": "wolframite",
    "Huebnerite": "wolframite",
    "Gold": "gold",
    "Beryl": "beryl",
    "Spodumene": "lithium",
    "Petalite": "lithium",
    "Amblygonite": "lithium",
    "Lepidolite": "lithium",
    "Cobaltite": "cobalt",
    "Erythrite": "cobalt",
    "Skutterudite": "cobalt",
    "Heterogenite": "cobalt",
    "Chalcopyrite": "copper",
    "Malachite": "copper",
    "Azurite": "copper",
    "Cuprite": "copper",
    "Bornite": "copper",
    "Chalcocite": "copper",
    "Corundum": "gemstone",
    "Tourmaline": "gemstone",
    "Topaz": "gemstone",
    "Garnet": "gemstone",
    "Zircon": "gemstone",
    "Spinel": "gemstone",
    "Quartz": "unknown",
    "Microcline": "unknown",
    "Albite": "unknown",
    "Muscovite": "unknown",
    "Biotite": "unknown",
    "Calcite": "unknown",
    "Dolomite": "unknown",
    "Chlorite": "unknown",
}
