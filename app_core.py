# app_core.py
# World Metrics Dashboard (Student Edition) with hot-reload of student_hook.py

import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, exceptions, ctx
import base64, os, time
try:
    from dash_ace import DashAceEditor as AceEditor
except Exception:
    try:
        from dash_ace.AceEditor import AceEditor
    except Exception:
        AceEditor = None  # we'll guard later



APP_TITLE = "World Metrics Dashboard (Student Edition)"
DATA_LONG_PATH = Path("world_data_long_plus.csv")




# -----------------------------
# Helpers
# -----------------------------

def _norm_per_year(s, mode="minmax"):
    import numpy as np
    vals = s.values
    if mode == "zscore":
        mu, sd = np.nanmean(vals), np.nanstd(vals)
        if not np.isfinite(mu) or not np.isfinite(sd) or sd == 0:
            return s * np.nan
        z = (s - mu) / sd
        z = z.clip(-3, 3)
        return (z + 3) / 6.0
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return s * np.nan
    return (s - lo) / (hi - lo)

def make_derived_metrics(df_long, SH_module):
    """
    Build derived metrics from COMPOSITES defined in the student hook.
    """
    import numpy as np
    import pandas as pd

    COMPOSITES = getattr(SH_module, "COMPOSITES", {})
    NORMALIZATION = getattr(SH_module, "NORMALIZATION", "minmax")

    if not COMPOSITES:
        return df_long

    years = sorted(df_long["year"].dropna().unique())
    countries = df_long[["iso3", "country"]].drop_duplicates()
    rows = []
    by_ind = {ind: g for ind, g in df_long.groupby("indicator")}

    for comp_name, spec in COMPOSITES.items():
        label = spec.get("label", comp_name)
        components = spec.get("components", [])
        out_min, out_max = spec.get("scale", (0, 100))
        min_components = int(spec.get("min_components", 1))
        if not components:
            continue

        for yr in years:
            parts, weights = [], []
            for ind_name, sign, w in components:
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

            M = pd.concat(parts, axis=1)
            w = pd.Series({name: wt for name, wt in weights}, index=M.columns)
            weighted_sum = (M * w).sum(axis=1, skipna=True)
            weight_sum = M.notna().mul(w, axis=1).sum(axis=1, skipna=True)
            count_nonnull = M.notna().sum(axis=1)

            score01 = weighted_sum / weight_sum.replace(0, np.nan)
            score01 = score01.where(count_nonnull >= min_components, np.nan)
            score = score01 * (out_max - out_min) + out_min

            for iso3, val in score.items():
                rows.append({
                    "iso3": iso3, "year": int(yr),
                    "indicator": comp_name, "value": val
                })

        SH_module.LABELS.setdefault(comp_name, label)

    if not rows:
        return df_long

    new_df = pd.DataFrame(rows).merge(countries, on="iso3", how="left")
    return pd.concat(
        [df_long, new_df[["iso3","country","year","indicator","value"]]],
        ignore_index=True
    )


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

def make_log_ticks(vmin, vmax):
    """
    Given positive min/max in *linear* space, return (tickvals_log10, ticktext_str)
    covering whole decades within [vmin, vmax].
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return [], []
    vmin = float(vmin); vmax = float(vmax)
    if vmin <= 0 or vmax <= 0 or vmin >= vmax:
        return [], []
    pmin = int(np.floor(np.log10(vmin)))
    pmax = int(np.ceil(np.log10(vmax)))
    vals = np.array([10.0**p for p in range(pmin, pmax + 1)], dtype=float)
    tickvals = np.log10(vals)               # positions in log10 space
    ticktext = [f"{int(v):,}" if v >= 1 else f"{v:.3g}" for v in vals]  # pretty labels
    return tickvals.tolist(), ticktext

# to provide in app backups when editing
# Baseline (immutable on refresh) and per-student override
BASE_FILE = "student_hook.py"
USER_FILE = "student_hook_local.py"   # edited by the in-Dash editor
BACKUP_DIR = ".backups"

def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def _write_text_atomic(path, text):
    import os
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)

def _backup(path, text):
    import os, time
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{os.path.basename(path)}.{ts}.bak"
    bp = os.path.join(BACKUP_DIR, name)
    _write_text_atomic(bp, text)
    return bp

def _load_module_from_file(py_path, alias="student_hook_loaded"):
    """Load a module from an arbitrary file path, bypassing import cache."""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(alias, py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


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
# ---- Student config (initial import) ----
import student_hook as SH

df_current = make_derived_metrics(df_base.copy(), SH)
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
        html.Div(id="active-config", style={"fontSize":"12px", "color":"#555"}),
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

# --- Inline Editor Block (always visible at page bottom) ---
if AceEditor is not None:
    editor_block = html.Div([
        html.Hr(),
        html.H3("Editor (inline)"),
        html.Div([
            html.Button("Save", id="ed-save", n_clicks=0, className="btn"),
            html.Button("Save + Backup", id="ed-save-backup", n_clicks=0, className="btn"),
            html.Button("Save + Reload", id="ed-save-reload", n_clicks=0, className="btn"),
            html.Span(id="ed-status", style={"marginLeft": "12px"}),
        ], style={"margin":"8px 0"}),
        
        AceEditor(
            id="ace",
            value=_read_text(USER_FILE) or _read_text(BASE_FILE),
            theme="github",
            mode="python",
            tabSize=4,
            fontSize=14,
            debounceChangePeriod=250,
            showPrintMargin=False,
            highlightActiveLine=True,
            showGutter=True,
            wrapEnabled=False,
            height="50vh",
            width="100%",
            setOptions={"useWorker": True},
            style={"border":"1px solid #ddd","borderRadius":"6px"},
        ),
        dcc.Store(id="editor-reload-ping"),
    ], style={"padding":"10px 12px", "background":"#fafafa", "borderTop":"1px solid #eee"})

    # If your layout is a Div with 'children', append the editor
    if isinstance(app.layout, html.Div):
        # If the Div has children list, append. If not, wrap it.
        if isinstance(app.layout.children, list):
            app.layout.children.append(editor_block)
        else:
            app.layout.children = [app.layout.children, editor_block]
    else:
        # Fallback: wrap previous layout in a Div and add editor below
        app.layout = html.Div([app.layout, editor_block])
else:
    print("WARNING: dash-ace not available; inline editor disabled.")

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
        df_current = make_derived_metrics(df_base.copy(), SH)

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
# fix for invalid year in data (moves slider to latest year with data)
@app.callback(
    Output("year", "min"),
    Output("year", "max"),
    Output("year", "value"),
    Output("year", "marks"),
    Input("metric", "value"),
    Input("settings-store", "data"),  # also react after Reload
    Input("editor-reload-ping","data"),
)
def sync_year_slider(metric, _settings, _ping):
    # Default to current global range if anything odd happens
    global df_current
    try:
        d = df_current[df_current["indicator"] == metric]
        years = sorted(y for y in d["year"].dropna().unique())
        if not years:
            # No data for this metric: keep existing slider state unchanged
            raise exceptions.PreventUpdate
        min_y, max_y = int(years[0]), int(years[-1])
        # Snap value to the latest (max) year with data for this metric
        value = max_y
        marks = slider_marks(min_y, max_y, step=5 if (max_y - min_y) > 8 else 1)
        return min_y, max_y, value, marks
    except exceptions.PreventUpdate:
        raise
    except Exception:
        # On any unexpected error, avoid breaking the UI
        raise exceptions.PreventUpdate

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


    # Filter the data
    d = df_current[(df_current["indicator"] == metric) & (df_current["year"] == year)].copy()
    use_log = "log" in (log_opts or [])
    
    # Keep both the real and log-transformed versions
    d["plot_value"] = np.where(d["value"] > 0, np.log10(d["value"]), np.nan) if use_log else d["value"]
    
    # Build a hover-friendly version of the value (the real one)
    def fmt(v):
        if not np.isfinite(v):
            return ""
        if v >= 1e6:
            return f"{v/1e6:.2f}M"
        elif v >= 1e3:
            return f"{v/1e3:.1f}K"
        elif v >= 1:
            return f"{v:,.0f}"
        else:
            return f"{v:.3f}"

    d["value_display"] = d["value"].apply(fmt)
    
    # Build ticks from the *original* positive values (before log transform)
    orig_pos = d["value"].where(d["value"] > 0).dropna()
    tickvals, ticktext = ([], [])
    if use_log and not orig_pos.empty:
        tickvals, ticktext = make_log_ticks(orig_pos.min(), orig_pos.max())

    fig = px.choropleth(
        d,
        locations="iso3",
        color="plot_value",
        hover_name="country",
        hover_data={
            "value_display": True,   # show real value nicely formatted
            "value": False,          # hide the raw number (for safety)
            "plot_value": False,     # hide log-transformed number
        },
        color_continuous_scale=colorscale,
        projection="natural earth",

    )
    # Colorbar: show real numbers when using log (we plotted log10, so positions are in log space)
    if use_log and tickvals:
        fig.update_layout(coloraxis_colorbar=dict(
            tickvals=tickvals,      # positions in log10 space
            ticktext=ticktext,      # human-friendly labels
            title=SH.LABELS.get(metric, metric),
        ))
    else:
        fig.update_layout(coloraxis_colorbar=dict(
            title=SH.LABELS.get(metric, metric),
        ))
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

@app.callback(
    Output("ed-status", "children"),
    Output("editor-reload-ping", "data"),
    Input("ed-save", "n_clicks"),
    Input("ed-save-backup", "n_clicks"),
    Input("ed-save-reload", "n_clicks"),
    State("ace", "value"),
    prevent_initial_call=True,
)
def editor_actions(n_save, n_save_bak, n_save_rel, text):
    if not ctx.triggered_id:
        raise exceptions.PreventUpdate

    # Always operate on USER_FILE (never touch BASE_FILE)
    # If USER_FILE doesn't exist yet, first Save will create it.
    # Optional syntax check (doesn't block save):
    try:
        compile(text, USER_FILE, "exec")
        syntax_msg = "Syntax OK"
    except Exception as e:
        syntax_msg = f"⚠️ Syntax warning: {e}"

    # Save or Save+Backup
    if ctx.triggered_id in ("ed-save", "ed-save-backup", "ed-save-reload"):
        _write_text_atomic(USER_FILE, text)

    if ctx.triggered_id == "ed-save":
        return f"Saved to {USER_FILE}. {syntax_msg}", None

    if ctx.triggered_id == "ed-save-backup":
        bpath = _backup(USER_FILE, text)
        return f"Saved + backup → {os.path.basename(bpath)}. {syntax_msg}", None

    if ctx.triggered_id == "ed-save-reload":
        # Backup then try to load USER_FILE for this session
        bpath = _backup(USER_FILE, text)
        try:
            SH = _load_module_from_file(USER_FILE, alias="student_hook_override")
        except Exception as e:
            return f"Saved + backup ({os.path.basename(bpath)}), reload failed: {e}", None

        # Recompute with the override module
        global df_current
        try:
            df_current = make_derived_metrics(df_base.copy(), SH)
        except Exception as e:
            return f"Saved + backup ({os.path.basename(bpath)}), recompute error: {e}", None

        # Poke stores so other callbacks (e.g., year slider) refresh
        import time as _t
        return f"Saved + backup ({os.path.basename(bpath)}). Reloaded override ✓ {syntax_msg}", {"ts": _t.time()}

    return "No action.", None

@app.callback(
    Output("active-config", "children"),
    Input("editor-reload-ping", "data")
)
def show_active_config(_ping):
    # If we’ve just reloaded from USER_FILE, say so; otherwise default to BASE_FILE
    return "Config: using override (student_hook_local.py) this session" if _ping else "Config: baseline (student_hook.py)"
# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print(f"Base rows: {len(df_base):,}, current rows: {len(df_current):,}, "
          f"years {year_min}–{year_max}, indicators: "
          f"{', '.join(sorted(df_current['indicator'].unique()))}")
    app.run(debug=True)  # Dash 3.x

