# ⚡ Quickstart Guide — World Metrics Dashboard

Welcome!  
This is a lightweight data visualisation dashboard built for students.  
Follow these **three quick steps** to get started.

---

## 1️⃣ Prepare your environment

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install dash plotly pandas numpy pandas_datareader
```

---

## 2️⃣ Create the dataset

Run the preparation script (internet required once):

```bash
python prepare_world_dataset.py
```

This downloads world data and saves:
- `world_data_long.csv`
- `world_data_wide_latest.csv`

Optional: add your own CSV (e.g. `students_2024.csv`) and merge it:

```bash
python add_students_data.py
```

---

## 3️⃣ Launch the dashboard

```bash
python app_core.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050).

- Pick an indicator from the dropdown  
- Adjust the year slider  
- Click countries to see their data  
- Edit `student_hook.py` and press **Reload** in the dashboard

---

## ✏️ Editing `student_hook.py`

Change labels, visible metrics, or define your own:

```python
COMPOSITES = {
    "wealth_health_index": {
        "label": "Wealth–Health Index (0–100)",
        "components": [
            ("gdp_per_capita_usd", +1, 0.5),
            ("life_expectancy_years", +1, 0.5),
        ],
    },
}
```

Save → click **Reload student_hook.py** in the dashboard → see your results instantly!

---

## 💡 Tips

| Action | Shortcut |
|--------|-----------|
| Reload your edits | Click “Reload student_hook.py” |
| Undo mistakes | Use Git or backup copies |
| Blank map | Try a different year — not all indicators have full data |

---

Enjoy exploring global data — and experiment safely!  
*Created collaboratively using Vibe Coding in ChatGPT.*
