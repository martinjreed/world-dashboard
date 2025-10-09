"""
student_hook.py  — simplest version
Edit only the lines under "STUDENT SETTINGS".
Works with app_core.py (hot-reload supported).
"""

# ===================== STUDENT SETTINGS =====================

# Show these indicators in the dropdown (must exist in CSV or be "my_index" below)
VISIBLE_INDICATORS = [
    "life_expectancy_years",
    "internet_users_pct",
    "my_index",  # <- your simple composite
]

# Friendly labels
LABELS = {
    "life_expectancy_years": "Life expectancy (years)",
    "internet_users_pct": "Internet users (% of pop.)",
    "my_index": "My Index (0–100)",
}

# Default map look
DEFAULT_COLOR_SCALE = "Viridis"
DEFAULT_LOG_SCALE   = False

# Optional per-point transform before plotting (keep it simple)
VALUE_TRANSFORM = lambda s: s  # e.g., lambda s: s.clip(lower=0)

# --- Simple composite definition (students tweak just these) ---
# Combine indicators with (sign, weight). sign=+1 if higher is better; -1 if lower is better.
COMPONENTS = [
    ("life_expectancy_years", +1, 0.5),
    ("internet_users_pct",    +1, 0.5),
    # Example to penalize CO₂ if present:
    # ("co2_per_capita_tons",  -1, 0.2),
]
OUTPUT_SCALE = (0, 100)     # scale final score to 0..100
NORMALIZATION = "minmax"    # "minmax" or "zscore" per-year
MIN_COMPONENTS = 1          # require at least this many components present
# =============================================================


# --------- Minimal helper: per-year normalization of a Series ---------
def _norm_per_year(s, mode="minmax"):
    import numpy as np
    if mode == "zscore":
        mu, sd = np.nanmean(s.values), np.nanstd(s.values)
        if not np.isfinite(mu) or not np.isfinite(sd) or sd == 0:
            return s * np.nan
        z = (s - mu) / sd
        z = z.clip(-3, 3)  # clamp for stability
        return (z + 3) / 6.0  # ~[0,1]
    # default: minmax
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if lo is None or hi is None or lo == hi:
        return s * np.nan
    return (s - lo) / (hi - lo)

# --------- Required hook: build any derived metrics and append ----------
def make_derived_metrics(df_long):
    """
    df_long columns: iso3, country, year, indicator, value
    Returns df_long with a single extra indicator: 'my_index'
    (defined by COMPONENTS above). If inputs are missing, it skips.
    """
    import numpy as np
    import pandas as pd

    # Quick exit if no components defined
    if not COMPONENTS:
        return df_long

    years = sorted(df_long["year"].dropna().unique())
    countries = df_long[["iso3", "country"]].drop_duplicates()
    rows = []

    # Pre-split by indicator for speed
    by_ind = {ind: g for ind, g in df_long.groupby("indicator")}

    for yr in years:
        parts = []
        weights = []
        for ind_name, sign, w in COMPONENTS:
            g = by_ind.get(ind_name)
            if g is None:
                continue
            d = g[g["year"] == yr]
            if d.empty:
                continue
            s = d.set_index("iso3")["value"]
            s = _norm_per_year(s, NORMALIZATION)
            s = s if sign > 0 else (1 - s)
            parts.append(s.rename(ind_name))
            weights.append((ind_name, float(w)))

        if not parts:
            continue

        M = pd.concat(parts, axis=1)  # iso3 x components
        w = pd.Series({name: wt for name, wt in weights}, index=M.columns)

        # Weighted average across available components per country
        weighted_sum = (M * w).sum(axis=1, skipna=True)
        weight_sum = M.notna().mul(w, axis=1).sum(axis=1, skipna=True)
        count_non_null = M.notna().sum(axis=1)

        score01 = weighted_sum / weight_sum
        score01 = score01.where(count_non_null >= MIN_COMPONENTS, np.nan)

        # Rescale to OUTPUT_SCALE
        a, b = OUTPUT_SCALE
        score = score01 * (b - a) + a

        for iso3, val in score.items():
            rows.append({"iso3": iso3, "year": int(yr),
                         "indicator": "my_index", "value": val})

    if not rows:
        return df_long

    new_df = pd.DataFrame(rows).merge(countries, on="iso3", how="left")
    return pd.concat([df_long, new_df[["iso3","country","year","indicator","value"]]], ignore_index=True)
