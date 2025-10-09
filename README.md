# 🌍 World Metrics Dashboard (Plotly Dash)

An interactive **offline world data visualisation dashboard** built with Python, Dash, and Plotly.  
It allows students to explore world indicators such as GDP, CO₂ emissions, and life expectancy on a choropleth map and time-series plot — and define **their own composite indicators** using a simple configuration file.

The project is designed for teaching purposes, focusing on data visualisation, analysis, and lightweight coding experimentation.

---

## ✨ Features

- 🗺️ Interactive world map with hoverable countries  
- 📈 Linked time-series plot for selected country  
- ⚙️ Hot-reload of `student_hook.py` (no app restart needed)  
- 🧮 Student-editable composite metrics (weighted indices)  
- 💾 Fully offline once data is prepared  
- 🔁 Optional backup or Git-based reversion for safe experimentation  

---

## 📁 Repository structure

```
world-dashboard/
├── app_core.py               # Main Dash application (core logic + reload)
├── student_hook.py           # Student configuration file (labels, metrics)
├── prepare_world_dataset.py  # Script to fetch and prepare world data
├── add_students_data.py      # (Optional) merge user data such as student counts
├── world_data_long.csv       # Base dataset (output from preparation script)
├── world_data_wide_latest.csv# Optional wide-format summary
├── world_data_long_plus.csv  # (Optional) merged dataset with extra indicators
├── students_2024.csv         # (Optional) input file (Country,Count)
└── README.md
```

---

## 🧩 1. Installation

Set up a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate

pip install dash plotly pandas numpy pandas_datareader
```

> ⚠️ The data preparation step requires internet access (to fetch World Bank / OWID data).  
> After that, the app runs **fully offline** using local CSV files.

---

## 🌐 2. Prepare the dataset

Run:

```bash
python prepare_world_dataset.py
```

This script will:

- Download world data from **World Bank** (and **Our World in Data** as fallback for CO₂)
- Merge key indicators such as:
  - Population density  
  - GDP per capita (USD)  
  - CO₂ emissions per capita  
  - Internet users (% of population)  
  - Renewable energy (% of final energy)  
  - Life expectancy (years)  
  - Urban population (% of total)
- Output:
  - `world_data_long.csv` — main long-format dataset  
  - `world_data_wide_latest.csv` — latest values snapshot  

---

## 🧮 3. (Optional) Add your own dataset

You can merge a custom dataset, e.g. number of students per country.

Example file: **`students_2024.csv`**

```csv
Country,Count
United Kingdom,10500
India,9600
China,8700
Germany,2300
Spain,1200
```

Merge into the dataset:

```bash
python add_students_data.py
```

This creates **`world_data_long_plus.csv`** with a new indicator called `students_2024` (year 2024).

In `app_core.py`, select which dataset to use:

```python
# app_core.py
DATA_LONG_PATH = Path("world_data_long.csv")
# or to use your merged data:
# DATA_LONG_PATH = Path("world_data_long_plus.csv")
```

---

## 🚀 4. Run the dashboard

```bash
python app_core.py
```

Open your browser at:  
👉 [http://127.0.0.1:8050](http://127.0.0.1:8050)

You can now:
- Choose an indicator from the dropdown  
- Adjust the year slider (auto-snaps to available data)  
- Click a country for its historical data  
- Edit `student_hook.py` and hit **Reload student_hook.py** to apply changes instantly  

---

## 🧠 5. Student configuration: `student_hook.py`

Students edit this file to customise the dashboard.

You can:
- Rename indicators (`LABELS`)  
- Choose visible metrics (`VISIBLE_INDICATORS`)  
- Create simple composite metrics (`COMPOSITES`)

Example composite:

```python
COMPOSITES = {
    "wealth_health_index": {
        "label": "Wealth–Health Index (0–100)",
        "components": [
            ("gdp_per_capita_usd", +1, 0.5),
            ("life_expectancy_years", +1, 0.5),
        ],
        "scale": (0, 100),
        "min_components": 2,
    },
}
```

Ensure the composite name also appears in:

```python
VISIBLE_INDICATORS = [
    "gdp_per_capita_usd",
    "life_expectancy_years",
    "wealth_health_index",
]
```

Then click **Reload student_hook.py** in the app — no restart required.

---

## 🧰 6. Recommended Git workflow (safe revert)

Set up Git to track changes and easily undo mistakes:

```bash
git init
git config user.email "you@example.com"
git config user.name  "Your Name"
git add .
git commit -m "Initial dashboard setup"
```

After working edits:

```bash
git add student_hook.py
git commit -m "Student composite v1"
```

If something breaks:

```bash
git checkout -- student_hook.py
```

> 💡 Alternatively, you can use the optional `student_editor.py` Tkinter app for backups and restoration without Git.

---

## 🧭 7. Troubleshooting

| Problem | Likely cause | Fix |
|----------|---------------|------|
| **Map is all grey** | Metric has no data for selected year | Pick a year that has data (the slider auto-snaps to the latest available year). |
| **Reload fails** | Syntax error in `student_hook.py` | Fix syntax and click Reload again. |
| **Some countries blank** | Name mismatch in custom CSV | Match country names to those in `world_data_long.csv`. |
| **Missing CO₂ data** | World Bank deprecated the indicator | The script automatically uses **Our World in Data** fallback. |

---

## 🎓 8. Educational goals

This project is designed to teach:
- Practical data visualisation and analysis with Python  
- Data normalisation, composition, and interpretation  
- Modular code organisation and safe experimentation  
- Responsible software practices (versioning, reproducibility)

By editing only `student_hook.py`, students can explore data reasoning without breaking the main system — ideal for workshops or lab sessions.

---

## 📜 License

Choose a license appropriate for your teaching use (MIT recommended).

---

## 🙏 Acknowledgements

- [World Bank Open Data](https://data.worldbank.org) — for most indicators  
- [Our World in Data (OWID)](https://ourworldindata.org) — for CO₂ fallback  
- [Plotly Dash](https://dash.plotly.com) — interactive dashboard framework  

---

**Author:** Prof. Martin J. Reed  
University of Essex — School of Computer Science and Electronic Engineering  

*This README and project framework were created collaboratively using Vibe Coding in ChatGPT.*
