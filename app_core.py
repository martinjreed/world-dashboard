# app_core.py
# World Metrics Dashboard (Student Edition) with hot-reload of student_hook.py

import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, exceptions

# ---- Student config (initial import) ----
import student_hook as SH

APP_TITLE = "World Metrics Dashboard (Student Edition)"
DATA_LONG_PATH = Path("world_data_long_plus.csv")

# -----------------------------
# Helpers
# -----------------------------
def build_indicator_options(df_long, SH_module):
    present = sorted(df_long["indicator"].dropna().unique())
    visible = [i for i in SH_module.VISIBLE_INDICATORS if i in present] or present
    return [
        {"label": SH_module.LABELS.get(ind, ind.replace("_", " ").title()), "value": ind}
        for ind in visible
    ]

def transform_for_plot(series, SH_module, use_log=False):
    s = series.copy()
    s = SH_module.VALUE_TRANSFORM(s)  # student hook
    if use_log:
        s = s.where(s > 0, np.nan)
        s = np.log10(s)
    return s

# -----------------------------
# Load base data once
# -----------------------------
if not DATA_LONG_PATH.exists():
    raise FileNotFoundError(
        "world_data_long.csv not found. Generate it first with your prep script."
    )

df_base = pd.read_csv(DATA_LONG_PATH)
need = {"iso3", "country", "year", "indicator", "value"}
missing = need - set(df_base.columns)
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df_base["year"] = pd.to_numeric(df_base["year"], errors="coerce").astype("Int64")
df_base["value"] = pd.to_numeric(df_base["value"], errors="coerce")

# Build current dataframe via student hook (mutable by reload)
df_current = SH.make_derived_metrics(df_base.copy())
indicator_options = build_indicator_options(df_current, SH)

year_min = int(df_current["year"].min())
year_max = int(df_current["year"].max())

COLOR_SCALES = [
    "Viridis", "Plasma", "Cividis", "Turbo", "Inferno", "Magma",
    "Blues", "Greens", "Oranges", "Purples", "Reds", "Greys",
    "YlOrRd", "YlGnBu", "PuBuGn"
]

def slider_marks(start, end, step=5):
    return {y: str(y) for y in range(start, end + 1, step)}

# -----------------------------
# App layout
# -----------------------------
app = Dash(__name__, title=APP_TITLE)

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial",
           "padding": "16px", "maxWidth": "1200px", "margin": "0 auto"},
    children=[
        html.H2(APP_TITLE, style={"marginBottom": "8px"}),
        html.P("Edit only student_hook.py to change behaviour (add metrics, labels, defaults). "
               "Use the Reload button after saving your changes."),

        html.Div(
            style={"display": "flex", "gap": "10px", "alignItems": "center",
                   "margin": "6px 0 12px"},
            children=[
                html.Button("🔁 Reload student_hook.py", id="btn-reload", n_clicks=0),
                html.Div(id="reload-status", style={"color": "#555"})
            ]
        ),

        # Controls
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr 1fr", "gap": "12px",
                   "alignItems": "center", "margin": "12px 0 20px 0"},
            children=[
                html.Div([
                    html.Label("Metric"),
                    dcc.Dropdown(
                        id="metric",
                        options=indicator_options,
                        value=indicator_options[0]["value"] if indicator_options else None,
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Color scale"),
                    dcc.Dropdown(
                        id="colorscale",
                        options=[{"label": c, "value": c} for c in COLOR_SCALES],
                        value=SH.DEFAULT_COLOR_SCALE,
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Log scale"),
                    dcc.Checklist(
                        id="logscale",
                        options=[{"label": " Use log₁₀", "value": "log"}],
                        value=["log"] if SH.DEFAULT_LOG_SCALE else [],
                        style={"marginTop": "6px"}
                    ),
                ]),
            ],
        ),

        # Year slider
        html.Div([
            html.Label("Year"),
            dcc.Slider(
                id="year",
                min=year_min,
                max=year_max,
                step=1,
                value=year_max,
                marks=slider_marks(year_min, year_max, step=5),
                tooltip={"placement": "bottom", "always_visible": False},
            )
        ], style={"marginBottom": "16px"}),

        # Map + trend
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px"},
            children=[
                dcc.Graph(
                    id="world_map",
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                    style={"height": "70vh"}
                ),
                dcc.Graph(
                    id="country_trend",
                    config={"displaylogo": False},
                    style={"height": "70vh"}
                ),
            ],
        ),

        dcc.Store(id="last_iso3"),
        dcc.Store(id="settings-store"),  # to push updated options/defaults after reload
    ]
)

# -----------------------------
# Reload callback
# -----------------------------
@app.callback(
    Output("settings-store", "data"),
    Output("reload-status", "children"),
    Input("btn-reload", "n_clicks"),
    prevent_initial_call=True
)
def reload_student(n_clicks):
    """Hot-reload student_hook.py and rebuild df_current + options/defaults."""
    global SH, df_current, indicator_options, year_min, year_max

    try:
        # Reload module (or import if not present)
        if "student_hook" in sys.modules:
            SH = importlib.reload(sys.modules["student_hook"])
        else:
            SH = importlib.import_module("student_hook")

        # Recalculate current dataframe using potentially new derived metrics
        df_current = SH.make_derived_metrics(df_base.copy())

        # Rebuild indicator options and defaults
        indicator_options = build_indicator_options(df_current, SH)
        metric_default = indicator_options[0]["value"] if indicator_options else None
        color_default = SH.DEFAULT_COLOR_SCALE
        log_default = ["log"] if SH.DEFAULT_LOG_SCALE else []

        # Recompute year range (in case transforms dropped rows)
        year_min = int(df_current["year"].min())
        year_max = int(df_current["year"].max())

        payload = {
            "indicator_options": indicator_options,
            "metric_default": metric_default,
            "color_default": color_default,
            "log_default": log_default,
            "labels": SH.LABELS,
            "year_min": year_min,
            "year_max": year_max,
        }
        return payload, "Reloaded student_hook.py ✅"
    except Exception as e:
        return None, f"Reload failed: {e}"

# Apply new settings to controls when settings-store updates
@app.callback(
    Output("metric", "options"),
    Output("metric", "value"),
    Output("colorscale", "value"),
    Output("logscale", "value"),
    Input("settings-store", "data"),
    prevent_initial_call=True
)
def apply_settings(data):
    if not data:
        raise exceptions.PreventUpdate
    return (
        data["indicator_options"],
        data["metric_default"],
        data["color_default"],
        data["log_default"],
    )

# -----------------------------
# Map + Trend callbacks (use df_current + SH each time)
# -----------------------------
@app.callback(
    Output("world_map", "figure"),
    Output("last_iso3", "data"),
    Input("metric", "value"),
    Input("year", "value"),
    Input("colorscale", "value"),
    Input("logscale", "value"),
    State("last_iso3", "data"),
)
def update_map(metric, year, colorscale, log_opts, last_iso3):
    if metric is None or df_current.empty:
        return px.choropleth(), None

    use_log = "log" in (log_opts or [])
    d = df_current[(df_current["indicator"] == metric) & (df_current["year"] == year)].copy()
    # Hide countries with no value automatically from color scaling
    d["plot_value"] = transform_for_plot(d["value"], SH, use_log)

    fig = px.choropleth(
        d,
        locations="iso3",
        color="plot_value",
        hover_name="country",
        color_continuous_scale=colorscale,
        projection="natural earth",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title=SH.LABELS.get(metric, metric)),
    )

    keep = last_iso3 if last_iso3 and (d["iso3"] == last_iso3).any() else None
    return fig, keep

@app.callback(
    Output("country_trend", "figure"),
    Input("world_map", "clickData"),
    Input("last_iso3", "data"),
    Input("metric", "value"),
)
def update_trend(clickData, last_iso3, metric):
    label = SH.LABELS.get(metric, metric)
    dm = df_current[df_current["indicator"] == metric]

    iso3 = None
    if clickData and "points" in clickData and clickData["points"]:
        iso3 = clickData["points"][0].get("location")
    elif last_iso3:
        iso3 = last_iso3

    if iso3:
        dm = dm[dm["iso3"] == iso3].sort_values("year")
        title = f"{dm['country'].iloc[0] if not dm.empty else iso3} — {label}"
    else:
        dm = dm.iloc[0:0]
        title = "Click a country on the map to see its time series"

    # Apply the same pre-plot transform (without log) for consistency
    dm = dm.assign(value_transformed=SH.VALUE_TRANSFORM(dm["value"]))
    fig = px.line(dm, x="year", y="value_transformed", markers=True, hover_data=["country"])
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=label,
        margin=dict(l=40, r=10, t=50, b=40),
    )
    return fig

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print(f"Base rows: {len(df_base):,}, current rows: {len(df_current):,}, "
          f"years {year_min}–{year_max}, indicators: "
          f"{', '.join(sorted(df_current['indicator'].unique()))}")
    app.run(debug=True)  # Dash 3.x

