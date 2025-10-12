# 🌍 Quickstart: World Data Dashboard (Edit `student_hook.py`)

This dashboard maps real-world indicators (GDP, population, CO₂, life expectancy, etc.).  
You can **create new composite metrics** by editing **`student_hook.py`** only.

---

## 1) Run the app

**macOS / Linux**
```bash
./run.sh
```

**Windows**
```bat
run.bat
```

Open: http://127.0.0.1:8050

---

## 2) Where to edit

Use the **Editor (inline)** tab (or your IDE) and open `student_hook.py`. You’ll see:

- `VISIBLE_INDICATORS`: which metrics appear in the dropdown.  
- `LABELS`: nice names for metrics.  
- `COMPOSITES`: **where you define new composite metrics**.  
- `NORMALIZATION`: how the app rescales components before weighting (`"minmax"` or `"zscore"`).  
- `VALUE_TRANSFORM`: optional last-step transform before plotting.

---

## 3) Add a composite metric (your template format)

Each composite in `COMPOSITES` looks like:
```python
COMPOSITES = {
  "your_metric_id": {
    "label": "Shown in UI",
    "components": [
        # ("indicator_name", sign, weight)
        # sign: +1 = higher is better, -1 = lower is better
        ("gdp_per_capita_usd", +1, 0.5),
        ("life_expectancy_years", +1, 0.5),
        # ("co2_per_capita_tons", -1, 0.3),  # example of “lower is better”
    ],
    "scale": (0, 100),      # rescale final score to this range
    "min_components": 2,    # require at least this many present to compute
  },
}
```

**Example A — Wealth–Health Index**
```python
"wealth_health_index": {
  "label": "Wealth–Health Index (0–100)",
  "components": [
      ("gdp_per_capita_usd",  +1, 0.4),
      ("life_expectancy_years", +1, 0.6),
  ],
  "scale": (0, 100),
  "min_components": 2,
},
```

**Example B — Sustainability Index**
```python
"sustainability_index": {
  "label": "Sustainability Index (0–100)",
  "components": [
      ("renewables_pct_final_energy", +1, 0.5),
      ("life_expectancy_years",       +1, 0.2),
      ("co2_per_capita_tons",         -1, 0.3),
  ],
  "scale": (0, 100),
  "min_components": 2,
},
```

After you add a composite:
1. Add the **metric id** to `VISIBLE_INDICATORS` so it appears in the dropdown.
2. Add the **label** to `LABELS` (if not already specified inside `COMPOSITES` by your template).

Click **Save + Reload** in the editor to apply changes.

---

## 4) Normalisation & tips

- The app normalises each component **per-year** using `NORMALIZATION`:
  - `"minmax"` → scales each component to [0, 1] for the selected year, then applies signs and weights.  
  - `"zscore"` → standardises values; good for heavy-tailed metrics.
- Weights don’t need to sum to 1 (they’re rebalanced internally), but keeping them near 1.0 total is intuitive.
- Use `min_components` to avoid noisy scores when data is missing.
- If the map looks blank, pick a year that has data for your metric.

---

## 5) Reset to baseline

If edits break the app, refresh the browser (or use the app’s “Reload student_hook.py” button if available) to restore the baseline.

---

**Requirements:** Python ≥ 3.10. First run needs internet to install packages.

*Created for teaching with Plotly Dash and `dash-ace`.*
