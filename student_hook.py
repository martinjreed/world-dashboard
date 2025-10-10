"""
student_hook.py — multi-composite teaching version
Edit only the settings below. Works with app_core.py (hot reload supported).
"""

# ===================== STUDENT SETTINGS =====================

# Indicators visible in the dashboard dropdown (must exist in CSV or be composite names below)
VISIBLE_INDICATORS = [
    "students_2024",          # optional (if you merged that extra CSV)
    "population_density",
    "gdp_per_capita_usd",
    "co2_per_capita_tons",
    "internet_users_pct",
    "renewables_pct_final_energy",
    "life_expectancy_years",
    "urban_pop_pct",
    "wealth_health_index",    # composite #1
    "sustainability_index",   # composite #2
]

# Friendly labels for the UI
LABELS = {
    "students_2024": "International Students (2024)",
    "population_density": "Population density (people/km²)",
    "gdp_per_capita_usd": "GDP per capita (USD)",
    "co2_per_capita_tons": "CO₂ per capita (tons)",
    "internet_users_pct": "Internet users (% of population)",
    "renewables_pct_final_energy": "Renewables (% of final energy)",
    "life_expectancy_years": "Life expectancy (years)",
    "urban_pop_pct": "Urban population (% of total)",
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
