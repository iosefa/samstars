# update_codes.py   •   Python 3.12
"""
Clean a training layer so that only `class`, `code`, and geometry remain,
no duplicates exist, CRS is EPSG:6635, and there are no missing values.
"""

import geopandas as gpd
import pandas as pd

# ────────────────────────── mappings ──────────────────────────
_CODE_TO_CLASS = {
    "METPOL": 1,
    "GREROB": 2,
    "ACAKOA": 3,
    "DIOSAN": 4,
    "PROSOPISPALLIDA": 5,
    "PROPAL": 5,
    "NELPAL": 5,
    "OTHER":  6,
}
_CLASS_TO_CODE = {v: k for k, v in _CODE_TO_CLASS.items()}


# ─────────────────── helpers ────────────────────
def _canonicalise_code(code: str | None, species_old: str | None = None) -> str | None:
    """
    Return a canonical 6-letter species code.

    • If *code* is "OTHER" (any casing) → use *species_old* when available.
    • If *code* is missing → also try *species_old*.
    • Otherwise just upper-case *code*.

    Always returns the best candidate we can build, even if it is not yet
    in the lookup table.  Unknown codes will later map to class 6.
    """
    # ── 1. CODE exists ─────────────────────────────────────────────
    if not pd.isna(code):
        code_up = str(code).upper()

        if code_up == "OTHER" and not pd.isna(species_old):
            return sanitise_species_old(species_old)

        return code_up

    # ── 2. CODE is missing → fall back to SpeciesOld ───────────────
    if not pd.isna(species_old):
        return sanitise_species_old(species_old)

    # ── 3. Nothing usable ──────────────────────────────────────────
    return None


def code_to_class(code: str | None, species_old: str | None = None) -> int | None:
    """
    Convert *code* (and optionally *species_old*) to its numeric class.

    Unknown or new codes are assigned to class 6.
    """
    canonical = _canonicalise_code(code, species_old)
    if canonical is None:
        return None
    return _CODE_TO_CLASS.get(canonical, 6)


def class_to_code(cls: int | float | None) -> str | None:
    if pd.isna(cls):
        return None
    return _CLASS_TO_CODE.get(int(cls), None)


def sanitise_species_old(value: str | None) -> str | None:
    """
    Turn 'Met pol' → 'METPOL'.
    Returns None if value is None/NaN.
    """
    if pd.isna(value):
        return None
    return str(value).replace(" ", "").upper()


# ─────────────────────────── main ────────────────────────────
INPUT_GPKG  = "./train_classify.gpkg"
OUTPUT_GPKG = "./train_classifier.gpkg"

# 1. Read & re-project ---------------------------------------------------------
gdf = gpd.read_file(INPUT_GPKG).to_crs(6635)

# 2. Pass-1 ─ fill CLASS from CODE (+ SpeciesOld) ----------------------------
gdf["code"] = gdf.apply(
    lambda r: _canonicalise_code(r["code"], r.get("SpeciesOld")),
    axis=1
)

# 2. Pass-1 ─ fill CLASS from the now-clean CODE -----------------------------
needs_class = gdf["class"].isna() & gdf["code"].notna()

gdf.loc[needs_class, "class"] = (
    gdf.loc[needs_class, "code"].apply(code_to_class)
)

print(f"Pass-1  (code ➜ class): {needs_class.sum()} rows fixed.")
# 3. Pass-2 ─ fill CODE  from CLASS ------------------------------------------
needs_code = gdf["code"].isna() & gdf["class"].notna()
gdf.loc[needs_code, "code"] = gdf.loc[needs_code, "class"].apply(class_to_code)
print(f"Pass-2  (class ➜ code): {needs_code.sum()} rows fixed.")

# 4. Pass-3 ─ recover from SpeciesOld ----------------------------------------
if "SpeciesOld" in gdf.columns:
    needs_both = gdf["code"].isna() & gdf["class"].isna() & gdf["SpeciesOld"].notna()
    gdf.loc[needs_both, "code"] = (
        gdf.loc[needs_both, "SpeciesOld"].apply(sanitise_species_old)
    )
    gdf.loc[needs_both, "class"] = gdf.loc[needs_both, "code"].apply(code_to_class)
    print(f"Pass-3 (SpeciesOld): {needs_both.sum()} rows fixed.")

# 5. Report / drop any rows still incomplete ----------------------------------
null_rows = gdf["code"].isna() | gdf["class"].isna()
n_null_rows = int(null_rows.sum())
if n_null_rows:
    print(f"Remaining rows with nulls: {n_null_rows}. They will be removed.")
    gdf = gdf.loc[~null_rows].copy()

# 6. Keep only required columns ----------------------------------------------
gdf = gdf[["class", "code", "geometry"]]

# 7. Remove duplicate point geometries ---------------------------------------
gdf["geom_hex"] = gdf.geometry.apply(lambda geom: geom.wkb_hex)
before = len(gdf)
gdf = gdf.drop_duplicates(subset="geom_hex").drop(columns="geom_hex")
print(f"Removed {before - len(gdf)} duplicate geometries.")

# 8. Final sanity check -------------------------------------------------------
assert not gdf["class"].isna().any()
assert not gdf["code"].isna().any()

# 9. Write out ---------------------------------------------------------------
gdf.to_file(OUTPUT_GPKG, driver="GPKG")
print(f"✓ Clean layer written to {OUTPUT_GPKG}")