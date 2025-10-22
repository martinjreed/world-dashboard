
# Friendly labels for the UI, you can change these (right hand side) as you like
LABELS = {
    "students_2024": "CSEE Student Nationality (2024) (best with Log ticked!)",
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
MAP_PROJECTION = "natural earth"  # Try: "orthographic", "equirectangular", "mercator", "miller"

# ----- Define simple composites here -----
# Each composite:
#   name: {
#     "label": "Shown in UI",
#     "components": [(indicator_name, sign, weight), ...],  # sign +1 higher-is-better, -1 lower-is-better
#     "scale": (min, max),                                   # output scaling
#     "min_components": K                                    # require at least K components present
#   }
# maybe you could add more! What about:
#    Green Development Score , made up of renewables_pct_final_energy and co2_per_capita_tons
#    Urban Wealth Index, made up of gdp_per_capita_usd urban_pop_pct
#    remember the +1 means the metric helps the index, -1 that it harms the index
#    you can see that below for co2_per_capita_ton, more is worse!


# an easy change would be to change the relative values of the metrics
COMPOSITES = {
    # Composite #1: wealth + health
    "wealth_health_index": {
        "label": "Wealth–Health Index (0–100)",
        "components": [
            ("gdp_per_capita_usd",  +1, 0.4),
            ("life_expectancy_years", +1, 0.6),
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
