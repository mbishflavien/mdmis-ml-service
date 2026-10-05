"""Maps RRUFF mineral names (its `##NAMES=` field / filename prefix) onto
MDMIS's MINERAL_CHOICES labels (app.constants).

Mineralogical basis for each grouping:
- coltan: colloquial name for columbite-tantalite group ore, so both the
  Fe and Mn end-members of columbite and tantalite map here.
- wolframite: a solid-solution series between the iron end-member
  (ferberite) and manganese end-member (huebnerite); RRUFF catalogs the
  end-members, not "wolframite" itself. Scheelite (CaWO4) is a
  mineralogically distinct species, but every tungsten deposit textbook
  pairs it with wolframite as "the two tungsten ore minerals" — they
  occur in the same skarn/vein systems and are mined for the same
  target commodity. Neither USGS splib07 nor ECOSTRESS carries actual
  wolframite/ferberite/huebnerite VSWIR spectra (checked directly, zero
  hits under any spelling), so scheelite is the only real public VSWIR
  data standing in for "tungsten ore" until real field samples arrive.
  This is a commodity-target grouping like coltan/lithium below, not a
  claim that scheelite and wolframite look spectrally identical — a
  lab-confirmed result should still say which species it actually was.
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
    "Columbite": "coltan",  # ECOSTRESS names the Fe end-member bare, e.g. "Columbite Fe^2+Nb_2O_6"
    "Tantalite-(Fe)": "coltan",
    "Tantalite-(Mn)": "coltan",
    "Tantalite": "coltan",
    "Ferberite": "wolframite",
    "Huebnerite": "wolframite",
    "Scheelite": "wolframite",  # see wolframite note above — real public VSWIR data, different species
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
