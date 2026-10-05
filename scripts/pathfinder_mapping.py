"""Maps mineral names onto gold-pathfinder ALTERATION categories — a
completely separate label space from app.constants.MINERAL_CHOICES /
scripts/mineral_mapping.py. This model never outputs "gold"; it outputs
which alteration style a spectrum looks like, which is how gold
exploration actually works in practice (geologists follow alteration
footprints to a target, not a direct "gold spectrum" — see the README
note on why native gold has no such thing in Raman or VSWIR).

Category basis (standard epithermal/orogenic gold-exploration geology):
- iron_oxide_gossan: hematite/goethite/jarosite — the oxidized cap left
  behind when near-surface sulfides (often auriferous pyrite) weather.
  "Gossan" hunting is one of the oldest surface gold-exploration methods.
- argillic_alteration: kaolinite/illite/alunite/pyrophyllite — acid-
  sulfate and phyllic alteration halos that form around epithermal and
  orogenic gold systems. This is exactly the assemblage at Cuprite, NV
  (see app/band_ratios.py's validated real-data check) — alunite+
  kaolinite+iron-oxide is the textbook high-sulfidation epithermal
  signature, a known gold-system analog even though Cuprite itself is
  mined for other things.
- sulfide_pathfinder: pyrite/arsenopyrite — arsenopyrite specifically is
  a classic orogenic-gold pathfinder mineral. These are spectrally
  weaker than the other two categories (sulfides have broad, shallow
  VSWIR features compared to oxides/clays) — held-out accuracy for this
  category should be read with that in mind, not assumed as strong as
  the other two.
- background: the same common rock-forming minerals used as the negative
  class in mineral_mapping.py — no alteration signal present.

None of these categories is "gold is here." They're "this spot has the
kind of chemistry gold systems tend to have" — a follow-up flag for a
geologist, same as a real gossan outcrop is a place to go look closer,
not a gold nugget.
"""

PATHFINDER_NAME_MAP: dict[str, str] = {
    "Hematite": "iron_oxide_gossan",
    "Goethite": "iron_oxide_gossan",
    "Jarosite": "iron_oxide_gossan",
    "Kaolinite": "argillic_alteration",
    "Illite": "argillic_alteration",
    "Alunite": "argillic_alteration",
    "Pyrophyllite": "argillic_alteration",
    "Pyrite": "sulfide_pathfinder",
    "Arsenopyrite": "sulfide_pathfinder",
    "Quartz": "background",
    "Microcline": "background",
    "Albite": "background",
    "Muscovite": "background",
    "Biotite": "background",
    "Calcite": "background",
    "Dolomite": "background",
    "Chlorite": "background",
}

PATHFINDER_CATEGORIES = ("iron_oxide_gossan", "argillic_alteration", "sulfide_pathfinder", "background")
