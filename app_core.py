# app_core.py
# World Metrics Dashboard (Student Edition) with hot-reload of student_hook.py

import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State, exceptions, ctx, no_update
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


def safe_log10(x):
    """Log10 for scalars or arrays. Nonpositive -> NaN. No runtime warnings."""
    if np.isscalar(x):
        x = float(x)
        return np.log10(x) if x > 0 else np.nan
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log10(x), np.nan)

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
        s = safe_log10(s)
    return s

def make_log_ticks(vmin, vmax):
    """
    Given positive min/max in linear space, return (tickvals_log10, ticktext)
    for whole decades covering [vmin, vmax].
    """
    try:
        vmin = float(vmin); vmax = float(vmax)
    except Exception:
        return [], []

    # guardrails
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return [], []
    if vmin <= 0 or vmax <= 0 or vmin >= vmax:
        return [], []

    pmin = int(np.floor(np.log10(vmin)))
    pmax = int(np.ceil(np.log10(vmax)))

    vals = (10.0 ** np.arange(pmin, pmax + 1)).astype(float)
    tickvals = np.log10(vals)  # positions in log space
    ticktext = [f"{int(v):,}" if v >= 1 else f"{v:.3g}" for v in vals]
    return tickvals.tolist(), ticktext

# to provide in app backups when editing
# Baseline (immutable on refresh) and per-student override
BASE_FILE = "student_hook.py"
USER_FILE = "student_hook_local.pylocal"   # edited by the in-Dash editor
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
    import os
    import importlib.util
    from importlib.machinery import SourceFileLoader

    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Edited file not found: {py_path}")
    if os.path.isdir(py_path):
        raise IsADirectoryError(f"Expected a file, found directory: {py_path}")

    # Optional: guard against empty file
    try:
        with open(py_path, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        raise OSError(f"Cannot read {py_path}: {e}")
    if not src.strip():
        raise ValueError(f"Edited file is empty: {py_path}")

    # Use an explicit SourceFileLoader so any extension works (.pylocal, .txt, etc.)
    loader = SourceFileLoader(alias, py_path)
    spec = importlib.util.spec_from_loader(alias, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {py_path}")

    mod = importlib.util.module_from_spec(spec)
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

# Use a global pointer to whichever SH is active (baseline at startup)
SH_ACTIVE = SH


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
        dcc.Location(id="url"),

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

from dash import no_update

def _build_metric_options_from_active():
    if df_current is None or df_current.empty:
        return []
    return build_indicator_options(df_current, SH_ACTIVE)

@app.callback(
    Output("metric", "options"),
    Output("metric", "value"),
    Input("url", "href"),                   # initial page load
    Input("editor-reload-ping", "data"),    # after Save + Reload
    State("metric", "value"),
)
def refresh_metric_options(_href, _ping, current_value):
    opts = _build_metric_options_from_active()
    if not opts:
        return [], None
    values = [o["value"] for o in opts]
    new_val = current_value if current_value in values else values[0]
    return opts, new_val

def _empty_map(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=20,r=20,t=30,b=20))
    return fig

@app.callback(
    Output("world_map", "figure"),
    Output("last_iso3", "data"),
    Input("metric", "value"),
    Input("year", "value"),
    Input("colorscale", "value"),
    Input("logscale", "value"),
    Input("editor-reload-ping", "data"),     # ensures re-run after Save+Reload
    State("last_iso3", "data"),
)
def update_map(metric, year, colorscale, log_opts, _ping, last_iso3):
    # pick current dataframe (or use _active_df(cfg_mode) if you have baseline/override)
    df = df_current

    # 1) Validate metric
    if not metric or df.empty or "indicator" not in df.columns:
        return _empty_map("No data"), last_iso3

    d_all = df[df["indicator"] == metric]
    if d_all.empty:
        return _empty_map(f"No data for '{metric}'"), last_iso3

    # 2) Snap to latest available year if current year has no data
    if (year is None) or (year not in set(d_all["year"].dropna().astype(int))):
        yvals = d_all["year"].dropna().astype(int)
        if yvals.empty:
            return _empty_map(f"No years for '{metric}'"), last_iso3
        year = int(yvals.max())

    d = d_all[d_all["year"] == year].copy()
    if d.empty:
        return _empty_map(f"No data for '{metric}' in {year}"), last_iso3

    # 3) Log handling (robust)
    use_log = isinstance(log_opts, (list, tuple, set)) and ("log" in log_opts)
    d["plot_value"] = safe_log10(d["value"]) if use_log else d["value"]

    # 4) Build figure
    fig = px.choropleth(
        d,
        locations="iso3",
        color="plot_value",
        hover_name="country",
        color_continuous_scale=colorscale or "Viridis",
    )

    # Show REAL values in hover (not log)
    label = getattr(SH, "LABELS", {}).get(metric, metric)
    fig.update_traces(
        customdata=np.c_[d["value"].to_numpy()],
        hovertemplate=f"%{{hovertext}}<br>{label}: %{{customdata[0]:,.4g}}<extra></extra>",
    )

    # 5) Log colorbar ticks (real numbers on labels)
    if use_log:
        orig_pos = d["value"].where(d["value"] > 0).dropna()
        if not orig_pos.empty:
            tickvals, ticktext = make_log_ticks(orig_pos.min(), orig_pos.max())
            if tickvals and ticktext:
                fig.update_layout(coloraxis_colorbar=dict(tickvals=tickvals, ticktext=ticktext))

    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), title=f"{label} — {year}")
    return fig, last_iso3

@app.callback(
    Output("country_trend", "figure"),
    Input("world_map", "clickData"),
    Input("last_iso3", "data"),
    Input("metric", "value"),
    Input("editor-reload-ping", "data"),   # re-run after Save+Reload
)
def update_trend(clickData, last_iso3, metric, _ping):
    # pick the active dataframe
    df = df_current  # if you use baseline/override helpers, call _active_df(cfg_mode)

    # 1) Resolve ISO3 from click or stored value
    iso3 = None
    if clickData and isinstance(clickData, dict):
        try:
            iso3 = clickData["points"][0].get("location")
        except Exception:
            iso3 = None
    if not iso3:
        iso3 = last_iso3

    # 2) If still nothing, pick a sensible default: top country by latest non-NaN value
    if not iso3:
        dm = df[df["indicator"] == metric]
        if dm.empty:
            return _empty_fig(f"No data for '{metric}'")
        latest = dm.dropna(subset=["value"])
        if latest.empty:
            return _empty_fig(f"No valid values for '{metric}'")
        # choose the most recent year’s top value
        y = latest["year"].max()
        top = latest[latest["year"] == y].sort_values("value", ascending=False).head(1)
        if top.empty:
            return _empty_fig(f"No values for '{metric}' in {y}")
        iso3 = top["iso3"].iloc[0]

    # 3) Build the time series for this country & metric
    series = df[(df["indicator"] == metric) & (df["iso3"] == iso3)][["year", "value"]].dropna()
    if series.empty:
        return _empty_fig(f"No data for {iso3} on '{metric}'")

    series = series.sort_values("year")
    label = SH.LABELS.get(metric, metric) if hasattr(SH, "LABELS") else metric
    country_name = df.loc[df["iso3"] == iso3, "country"].dropna().head(1).tolist()
    country_name = country_name[0] if country_name else iso3

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series["year"],
        y=series["value"],
        mode="lines+markers",
        name=country_name,
        hovertemplate=f"{country_name}<br>Year=%{{x}}<br>{label}=%{{y}}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{label} — {country_name}",
        xaxis_title="Year",
        yaxis_title=label,
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
    )
    return fig


def _empty_fig(msg: str):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=40, r=20, t=40, b=40),
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
    action = ctx.triggered_id

    # Always save to USER_FILE
    try:
        _write_text_atomic(USER_FILE, text)
    except Exception as e:
        return f"❌ Save failed to {USER_FILE}: {e}", no_update

    # Syntax check (warn but continue)
    try:
        compile(text, USER_FILE, "exec")
        syntax_msg = "Syntax OK"
    except Exception as e:
        syntax_msg = f"⚠️ Syntax warning: {e}"

    if action == "ed-save":
        return f"Saved → {os.path.basename(USER_FILE)}. {syntax_msg}", no_update

    if action == "ed-save-backup":
        bpath = _backup(USER_FILE, text)
        return f"Saved + backup → {os.path.basename(bpath)}. {syntax_msg}", no_update

    if ctx.triggered_id == "ed-save-reload":
        bpath = _backup(USER_FILE, text)
        try:
            SH_override = _load_module_from_file(USER_FILE, alias="student_hook_override")
        except Exception as e:
            return f"Saved + backup ({os.path.basename(bpath)}), reload failed: {e}", None

        # 👇 persist the active module globally for later callbacks
        global SH_ACTIVE
        SH_ACTIVE = SH_override

        # Recompute with the override module
        global df_current
        try:
            df_current = make_derived_metrics(df_base.copy(), SH_ACTIVE)
        except Exception as e:
            return f"Saved + backup ({os.path.basename(bpath)}), recompute error: {e}", None

        import time as _t
        return f"Saved + backup ({os.path.basename(bpath)}). Reloaded override ✓ {syntax_msg}", {"ts": _t.time()}


@app.callback(
    Output("active-config", "children"),
    Input("editor-reload-ping", "data")
)
def show_active_config(_ping):
    # If we’ve just reloaded from USER_FILE, say so; otherwise default to BASE_FILE
    return "Config: using override (student_hook_local.py) this session" if _ping else "Config: baseline (student_hook.py)"
# --- Copy baseline to editable copy on each page load and show it in editor ---
# --- Initialize editor content on page load without clobbering prior edits ---
@app.callback(
    Output("ace", "value"),
    Input("url", "href"),
    prevent_initial_call=False,
)
def _init_page(_href):
    local = _read_text(USER_FILE)
    if local and local.strip():
        return local
    base = _read_text(BASE_FILE)
    if base and base.strip():
        _write_text_atomic(USER_FILE, base)  # seed once
        return base
    return ""

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print(f"Base rows: {len(df_base):,}, current rows: {len(df_current):,}, "
          f"years {year_min}–{year_max}, indicators: "
          f"{', '.join(sorted(df_current['indicator'].unique()))}")
    app.run(debug=True, dev_tools_hot_reload=False)  # Dash 3.x

