"""
student_hook.py — multi-composite teaching version
Edit only the settings below. Works with app_core.py (hot reload supported).
"""

# ===================== STUDENT SETTINGS =====================

# Indicators visible in the dashboard dropdown (must exist in CSV or be composite names below)
VISIBLE_INDICATORS = [
    "population_density",
    "gdp_per_capita_usd",
    "co2_per_capita_tons",
    "internet_users_pct",
    "renewables_pct_final_energy",
    "life_expectancy_years",
    "urban_pop_pct",
    "students_2024",          # optional (if you merged that extra CSV)
    "wealth_health_index",    # composite #1
    "sustainability_index",   # composite #2
]

# Friendly labels for the UI
LABELS = {
    "population_density": "Population density (people/km²)",
    "gdp_per_capita_usd": "GDP per capita (USD)",
    "co2_per_capita_tons": "CO₂ per capita (tons)",
    "internet_users_pct": "Internet users (% of population)",
    "renewables_pct_final_energy": "Renewables (% of final energy)",
    "life_expectancy_years": "Life expectancy (years)",
    "urban_pop_pct": "Urban population (% of total)",
    "students_2024": "International Students (2024)",
    "wealth_health_index": "Wealth–Health Index (0–100)",
    "sustainability_index": "Sustainability Index (0–100)",
}

# Default visualisation settings
DEFAULT_COLOR_SCALE = "Viridis"
DEFAULT_LOG_SCALE   = False

# Optional pre-plot transform before mapping (keep simple)
VALUE_TRANSFORM = lambda s: s  # e.g., lambda s: s.clip(lower=0)

# ----- Define simple composites here -----
# Each composite:
#   name: {
#     "label": "Shown in UI",
#     "components": [(indicator_name, sign, weight), ...],  # sign +1 higher-is-better, -1 lower-is-better
#     "scale": (min, max),                                   # output scaling
#     "min_components": K                                    # require at least K components present
#   }
COMPOSITES = {
    # Composite #1: wealth + health
    "wealth_health_index": {
        "label": "Wealth–Health Index (0–100)",
        "components": [
            ("gdp_per_capita_usd",  +1, 0.5),
            ("life_expectancy_years", +1, 0.5),
        ],
        "scale": (0, 100),
        "min_components": 2,
    },

    # Composite #2: sustainability = wellbeing + digital access minus emissions
    "sustainability_index": {
        "label": "Sustainability Index (0–100)",
        "components": [
            ("life_expectancy_years", +1, 0.4),
            ("internet_users_pct",    +1, 0.3),
            ("co2_per_capita_tons",   -1, 0.3),
        ],
        "scale": (0, 100),
        "min_components": 2,
    },
}

# Normalise each component per-year before weighting:
#   "minmax" -> rescale to [0,1] by year
#   "zscore" -> standardise to ~[0,1] using a clipped z-score
NORMALIZATION = "minmax"

# ================== END STUDENT SETTINGS =====================


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


# --------- Required hook: build derived metrics and append ------------
def make_derived_metrics(df_long):
    """
    df_long columns: iso3, country, year, indicator, value
    Returns df_long with extra rows for each composite in COMPOSITES.
    """
    import numpy as np
    import pandas as pd

    if not COMPOSITES:
        return df_long

    years = sorted(df_long["year"].dropna().unique())
    countries = df_long[["iso3", "country"]].drop_duplicates()
    rows = []

    # Pre-split by indicator for faster lookups
    by_ind = {ind: g for ind, g in df_long.groupby("indicator")}

    for comp_name, spec in COMPOSITES.items():
        label = spec.get("label", comp_name)
        components = spec.get("components", [])
        out_min, out_max = spec.get("scale", (0, 100))
        min_components = int(spec.get("min_components", 1))

        if not components:
            continue

        for yr in years:
            parts = []
            weights = []
            for ind_name, sign, w in components:
                g = by_ind.get(ind_name)
                if g is None:
                    continue
                d = g[g["year"] == yr]
                if d.empty:
                    continue
                s = d.set_index("iso3")["value"]
                s = _norm_per_year(s, NORMALIZATION)
                s = s if sign > 0 else (1 - s)  # invert if "lower is better"
                parts.append(s.rename(ind_name))
                weights.append((ind_name, float(w)))

            if not parts:
                continue

            # Join components on iso3
            M = pd.concat(parts, axis=1)     # iso3 × components
            w = pd.Series({name: wt for name, wt in weights}, index=M.columns)

            # Weighted average per country, ignoring missing components
            weighted_sum = (M * w).sum(axis=1, skipna=True)
            weight_sum   = M.notna().mul(w, axis=1).sum(axis=1, skipna=True)
            count_nonnull = M.notna().sum(axis=1)

            score01 = weighted_sum / weight_sum
            score01 = score01.where(count_nonnull >= min_components, np.nan)

            # Rescale to (out_min, out_max)
            score = score01 * (out_max - out_min) + out_min

            for iso3, val in score.items():
                rows.append({"iso3": iso3, "year": int(yr),
                             "indicator": comp_name, "value": val})

        # Ensure UI label is available even if not set above
        LABELS.setdefault(comp_name, label)

    if not rows:
        return df_long

    new_df = pd.DataFrame(rows).merge(countries, on="iso3", how="left")
    return pd.concat(
        [df_long, new_df[["iso3", "country", "year", "indicator", "value"]]],
        ignore_index=True
    )

