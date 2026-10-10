"""
Child Sexual Offence (POCSO) Cases in India - animated Streamlit dashboard
==========================================================================
Data source : child_sexual_offence_cases_india_verified.csv (bundled, no upload)
Run         : streamlit run app.py

Everything shown is taken from, or arithmetically derived from, that one CSV.
Derived values (growth %, shares, "rest of India" remainders) are labelled as such.
"""
import html as html_lib
import inspect
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from scipy import stats

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="POCSO Cases in India - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "child_sexual_offence_cases_india_verified.csv"

ALL_OFF = "All POCSO offences"
SEC_OFF = "Sec 4 & 6 (penetrative)"
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#9D755D", "#EECA3B", "#FF9DA6", "#BAB0AC"]
NAVY = "#1F2A44"

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
@keyframes fadeUp {from {opacity: 0; transform: translateY(16px);} to {opacity: 1; transform: none;}}
@keyframes slideIn {from {opacity: 0; transform: translateX(-24px);} to {opacity: 1; transform: none;}}
div[data-testid="stPlotlyChart"], div[data-testid="stImage"], div[data-testid="stDataFrame"] {
    animation: fadeUp .8s ease both;
}
.hero {animation: slideIn .9s ease both; margin-bottom: .4rem;}
.hero h1 {
    font-size: 2.1rem; margin: 0; line-height: 1.2;
    background: linear-gradient(90deg, #4C78A8, #B279A2, #F58518);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p {opacity: .75; margin: .3rem 0 0 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small compatibility helpers (Streamlit's width API changed across versions)
# --------------------------------------------------------------------------- #
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig, key):
    """Render a Plotly figure stretched to the column width."""
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #
def short_offence(text: str) -> str:
    t = str(text).lower()
    if "all offences" in t:
        return ALL_OFF
    if "sec 4 & 6" in t:
        return SEC_OFF
    if "share" in t:
        return "Victim gender share"
    return str(text)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    raw = pd.read_csv(DATA_PATH)
    raw["cases"] = pd.to_numeric(raw["total_cases_registered"], errors="coerce")
    raw["offence_short"] = raw["offence_category"].map(short_offence)
    raw["scope"] = np.where(raw["geography"].str.startswith("India"), "National", "State")
    raw["date"] = pd.to_datetime(raw["year"].astype(str) + "-12-31")
    return raw


def parse_gender_share(raw: pd.DataFrame):
    """The dataset holds one row with *percentages* of victims by gender (in its notes)."""
    row = raw[raw["cases"].isna()]
    if row.empty:
        return None
    r = row.iloc[0]
    nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", str(r["notes"]))]
    if len(nums) < 4:
        return None
    period = re.search(r"(\d{4})\s*-\s*(\d{4})", str(r["offence_category"]))
    return {
        "girls_range": (nums[0], nums[1]),
        "boys_range": (nums[2], nums[3]),
        "girls": float(np.mean(nums[:2])),
        "boys": float(np.mean(nums[2:4])),
        "period": f"{period.group(1)}-{period.group(2)}" if period else "",
        "source": str(r["source"]),
    }


def build_components(cases: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose each reported national total into parts that add back up to it:
      * listed states (as reported)
      * 'Rest of India' = national total - listed states            (derived)
      * for 2020 (no state rows): Sec 4 & 6 + 'Other offences'       (derived)
    Years that have state rows but no national total are marked as partial.
    """
    allx = cases[cases["offence_short"] == ALL_OFF]
    nat = allx[allx["scope"] == "National"].set_index("year")["cases"]
    sts = allx[allx["scope"] == "State"]
    sec = cases[(cases["scope"] == "National") & (cases["offence_short"] == SEC_OFF)] \
        .set_index("year")["cases"]

    rows = []

    def add(year, comp, val, geo, derived):
        rows.append({"year": int(year), "component": comp, "cases": float(val),
                     "geography": geo, "derived": derived})

    for y in sorted(allx["year"].unique()):
        sy = sts[sts["year"] == y]
        for _, r in sy.iterrows():
            add(y, r["geography"], r["cases"], r["geography"], False)
        if y in nat.index:
            if len(sy):
                add(y, "Rest of India (derived)", nat[y] - sy["cases"].sum(), "India (all)", True)
            elif y in sec.index:
                add(y, "Sec 4 & 6 - penetrative (reported as 'more than')", sec[y], "India (all)", False)
                add(y, "Other POCSO offences (derived)", nat[y] - sec[y], "India (all)", True)
            else:
                add(y, "India - all offences", nat[y], "India (all)", False)

    comp = pd.DataFrame(rows)
    nat_years = set(nat.index)
    comp["year_label"] = [str(y) if y in nat_years else f"{y} (listed states only)"
                          for y in comp["year"]]
    return comp


def change_table(f: pd.DataFrame) -> pd.DataFrame:
    """First-vs-latest reported year for every series with >= 2 reported years."""
    d = f[f["offence_short"] == ALL_OFF].sort_values("year")
    g = d.groupby("geography").agg(
        first_year=("year", "first"), last_year=("year", "last"),
        first_cases=("cases", "first"), last_cases=("cases", "last"), n=("year", "size"),
    )
    g = g[g["n"] >= 2].reset_index()
    if g.empty:
        return g
    span = (g["last_year"] - g["first_year"]).astype(float)
    g["change_pct"] = (g["last_cases"] / g["first_cases"] - 1) * 100
    g["cagr_pct"] = ((g["last_cases"] / g["first_cases"]) ** (1 / span) - 1) * 100
    return g


def compute_kpis(f: pd.DataFrame, total_records: int):
    cards = [{"label": "Records in view", "value": float(len(f)), "dec": 0,
              "sub": f"of {total_records} records with case counts"}]

    nat = f[(f["scope"] == "National") & (f["offence_short"] == ALL_OFF)].sort_values("year")
    if len(nat):
        last = nat.iloc[-1]
        cards.append({"label": "Latest national total", "value": float(last["cases"]), "dec": 0,
                      "sub": f"{int(last['year'])} · all POCSO offences"})
    else:
        cards.append({"label": "Latest national total", "value": None, "sub": "No national record in view"})

    if len(nat) >= 2:
        a, b = nat.iloc[0], nat.iloc[-1]
        yrs = int(b["year"] - a["year"])
        pct = (b["cases"] / a["cases"] - 1) * 100
        cagr = ((b["cases"] / a["cases"]) ** (1 / yrs) - 1) * 100
        cards.append({"label": "Growth in registered cases", "value": float(pct), "dec": 1,
                      "prefix": "+" if pct >= 0 else "", "suffix": "%",
                      "sub": f"{int(a['year'])} → {int(b['year'])} · CAGR {cagr:.1f}%/yr"})
    else:
        cards.append({"label": "Growth in registered cases", "value": None,
                      "sub": "Needs ≥2 national years in view"})

    sts = f[f["scope"] == "State"]
    if len(sts):
        top = sts.loc[sts["cases"].idxmax()]
        cards.append({"label": "Highest state record", "value": float(top["cases"]), "dec": 0,
                      "sub": f"{top['geography']} · {int(top['year'])}"})
    else:
        cards.append({"label": "Highest state record", "value": None, "sub": "No state record in view"})

    best = None
    nat_s = nat.set_index("year")["cases"]
    for y, tot in nat_s.items():
        sy = f[(f["scope"] == "State") & (f["year"] == y) & (f["offence_short"] == ALL_OFF)]
        if len(sy) and (best is None or len(sy) > best[1]):
            best = (int(y), len(sy), sy["cases"].sum() / tot * 100)
    if best:
        cards.append({"label": "Listed states' share of national", "value": float(best[2]), "dec": 1,
                      "suffix": "%", "sub": f"{best[1]} state(s) listed · {best[0]}"})
    else:
        cards.append({"label": "Listed states' share of national", "value": None,
                      "sub": "Needs national + state rows"})
    return cards


KPI_TEMPLATE = """
<style>
  body {margin:0; font-family: "Source Sans Pro", system-ui, sans-serif; background: transparent;}
  .grid {display:grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px;}
  .card {background: linear-gradient(135deg, #1F2A44 0%, #2E3F6B 100%); color: #fff;
         border-radius: 14px; padding: 14px 16px; border-left: 5px solid #F58518;
         box-shadow: 0 4px 14px rgba(0,0,0,.18); opacity: 0; transform: translateY(14px);
         animation: pop .7s ease forwards;}
  .lab {font-size: 12px; letter-spacing: .04em; text-transform: uppercase; opacity: .75;}
  .val {font-size: 30px; font-weight: 700; margin: 4px 0 2px 0; font-variant-numeric: tabular-nums;}
  .sub {font-size: 12px; opacity: .8;}
  @keyframes pop {to {opacity: 1; transform: none;}}
</style>
<div class="grid">__ITEMS__</div>
<script>
  document.querySelectorAll('.val[data-target]').forEach(function (el) {
    var target = parseFloat(el.dataset.target), dec = parseInt(el.dataset.dec || '0'),
        pre = el.dataset.prefix || '', suf = el.dataset.suffix || '', dur = 1500, t0 = performance.now();
    function fmt(v) {
      return pre + v.toLocaleString('en-IN', {minimumFractionDigits: dec, maximumFractionDigits: dec}) + suf;
    }
    function tick(now) {
      var p = Math.min((now - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(target * e);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
</script>
"""


def render_kpis(cards):
    items = []
    for i, c in enumerate(cards):
        label, sub = html_lib.escape(c["label"]), html_lib.escape(c["sub"])
        if c["value"] is None:
            val = '<div class="val">-</div>'
        else:
            val = (f'<div class="val" data-target="{c["value"]}" data-dec="{c.get("dec", 0)}" '
                   f'data-prefix="{c.get("prefix", "")}" data-suffix="{c.get("suffix", "")}">0</div>')
        items.append(f'<div class="card" style="animation-delay:{i * 0.1:.1f}s">'
                     f'<div class="lab">{label}</div>{val}<div class="sub">{sub}</div></div>')
    page = KPI_TEMPLATE.replace("__ITEMS__", "".join(items))
    try:
        components.html(page, height=135)
    except Exception:  # very defensive fallback: static cards without count-up
        cols = st.columns(len(cards))
        for col, c in zip(cols, cards):
            col.metric(c["label"], "-" if c["value"] is None else f"{c['value']:,.{c.get('dec', 0)}f}", c["sub"])


# --------------------------------------------------------------------------- #
# Chart helpers
# --------------------------------------------------------------------------- #
def base_layout(fig, title=None, height=430):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=64 if title else 24, b=10),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    return fig


def update_all_traces(fig, **kw):
    """update_traces() skips animation frames, so patch those too."""
    fig.update_traces(**kw)
    for fr in fig.frames:
        for tr in fr.data:
            tr.update(**kw)


def play_controls(fig, frame_names, duration, active=0, prefix="Year: "):
    """Own play/pause buttons + slider so the sidebar speed control applies to every animation."""
    if not frame_names:
        return fig
    anim = {"frame": {"duration": duration, "redraw": True}, "mode": "immediate",
            "fromcurrent": False,
            "transition": {"duration": int(duration * 0.5), "easing": "cubic-in-out"}}
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False,
            pad={"r": 10, "t": 70}, x=0.1, xanchor="right", y=0, yanchor="top",
            buttons=[
                dict(label="▶ Play", method="animate", args=[None, anim]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}]),
            ])],
        sliders=[dict(
            active=active, x=0.1, len=0.9, xanchor="left", y=0, yanchor="top",
            pad={"b": 10, "t": 60}, currentvalue={"prefix": prefix},
            steps=[dict(label=n, method="animate",
                        args=[[n], {"frame": {"duration": duration, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": int(duration * 0.5)}}])
                   for n in frame_names])],
    )
    return fig


def axis_range(values_min, values_max, log):
    if log:
        return dict(type="log", range=[np.log10(max(values_min * 0.55, 1)), np.log10(values_max * 2.4)])
    return dict(type="linear", range=[0, values_max * 1.18])


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def fig_bar_race(f, mode, speed, off_colors):
    d = f.copy()
    if mode == "States only":
        d = d[d["scope"] == "State"]
    elif mode == "National only":
        d = d[d["scope"] == "National"]
    if d.empty:
        return None
    log = mode.startswith("All")
    years = sorted(int(y) for y in d["year"].unique())
    order = [g for g in GEO_ORDER if g in set(d["geography"])]
    fig = px.bar(
        d, x="geography", y="cases", color="offence_short", animation_frame="year",
        barmode="group", text_auto=",.0f",
        category_orders={"year": years, "geography": order},
        color_discrete_map=off_colors,
        labels={"cases": "Registered cases", "geography": "", "offence_short": "Offence category"},
    )
    update_all_traces(fig, textposition="outside", cliponaxis=False)
    fig.update_yaxes(**axis_range(d["cases"].min(), d["cases"].max(), log))
    base_layout(fig, "Registered cases by geography - press ▶ Play to step through the years", 520)
    return play_controls(fig, [fr.name for fr in fig.frames], speed)


def fig_donut(share):
    labels = ["Girls", "Boys", "Unspecified / rounding"]
    values = [share["girls"], share["boys"], max(0.0, 100 - share["girls"] - share["boys"])]
    custom = [f"{share['girls_range'][0]}%-{share['girls_range'][1]}%",
              f"{share['boys_range'][0]}%-{share['boys_range'][1]}%", "remainder"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62, sort=False, customdata=custom,
        marker=dict(colors=["#E45756", "#4C78A8", "#BAB0AC"], line=dict(color="white", width=2)),
        textinfo="label+percent", textposition="outside",
        hovertemplate="<b>%{label}</b><br>Reported range: %{customdata}<extra></extra>",
        pull=[0.03, 0, 0],
    ))
    fig.add_annotation(text=f"Victims<br><b>{share['period']}</b>", showarrow=False, font=dict(size=15))
    base_layout(fig, "Victim gender split (donut)", 430)
    fig.update_layout(showlegend=False)
    return fig


def fig_pie(comp_f, year_label):
    d = comp_f[comp_f["year_label"] == year_label]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(d))]
    fig = go.Figure(go.Pie(
        labels=d["component"], values=d["cases"], sort=False,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="percent", hovertemplate="<b>%{label}</b><br>%{value:,.0f} cases<br>%{percent}<extra></extra>",
    ))
    base_layout(fig, f"Composition of registered cases - {year_label} (pie)", 430)
    fig.update_layout(legend=dict(orientation="v", x=1.0, y=0.5))
    return fig


def fig_sunburst(comp_f):
    fig = px.sunburst(
        comp_f, path=["year_label", "component"], values="cases", color="year_label",
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(
        insidetextorientation="radial", textinfo="label+percent parent",
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} cases<extra></extra>",
    )
    return base_layout(fig, "Year → component breakdown (sunburst)", 520)


def fig_line_draw(f, log, speed, geo_colors):
    d = f.sort_values("year").copy()
    multi = d["offence_short"].nunique() > 1
    d["series"] = np.where(multi, d["geography"] + " · " + d["offence_short"], d["geography"])
    series = list(dict.fromkeys(d["series"]))
    years = sorted(int(y) for y in d["year"].unique())
    dashes = {ALL_OFF: "solid", SEC_OFF: "dot"}

    def traces_upto(y):
        out = []
        for s in series:
            sd = d[(d["series"] == s) & (d["year"] <= y)]
            geo, off = (sd["geography"].iloc[0], sd["offence_short"].iloc[0]) if len(sd) else \
                (d[d["series"] == s]["geography"].iloc[0], d[d["series"] == s]["offence_short"].iloc[0])
            out.append(go.Scatter(
                x=sd["year"], y=sd["cases"], mode="lines+markers", name=s,
                line=dict(color=geo_colors.get(geo, NAVY), width=3.5, dash=dashes.get(off, "solid")),
                marker=dict(size=10),
                hovertemplate="%{x}: %{y:,.0f} cases<extra>" + s + "</extra>"))
        return out

    frames = [go.Frame(data=traces_upto(y), name=str(y)) for y in years]
    fig = go.Figure(data=traces_upto(years[-1]), frames=frames)
    fig.update_xaxes(tickmode="array", tickvals=years, range=[years[0] - 0.4, years[-1] + 0.4],
                     title="Year")
    fig.update_yaxes(title="Registered cases", **axis_range(d["cases"].min(), d["cases"].max(), log))
    base_layout(fig, "Registered cases over time (line) - ▶ Play redraws the lines year by year", 520)
    return play_controls(fig, [str(y) for y in years], speed, active=len(years) - 1)


def series_options(f):
    opts = {}
    for (geo, off), g in f.groupby(["geography", "offence_short"]):
        if len(g) >= 2:
            opts[geo if off == ALL_OFF else f"{geo} · {off}"] = g.sort_values("year")
    return opts


def fig_timeseries(g, label):
    x, y = g["year"].astype(float).to_numpy(), g["cases"].to_numpy(dtype=float)
    yoy = g["cases"].pct_change() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(
        x=g["date"], y=yoy, name="Change vs previous reported year (%)", secondary_y=True,
        marker_color="rgba(245,133,24,0.45)", width=1000 * 3600 * 24 * 70,
        texttemplate="%{y:+.1f}%", textposition="outside",
        hovertemplate="%{y:+.1f}%<extra>Change</extra>")
    fig.add_scatter(
        x=g["date"], y=g["cases"], mode="lines+markers", name="Registered cases",
        line=dict(color="#4C78A8", width=4), marker=dict(size=11), fill="tozeroy",
        fillcolor="rgba(76,120,168,0.15)", hovertemplate="%{x|%Y}: %{y:,.0f}<extra></extra>")
    info = {"n": len(g), "slope": None, "r2": None}
    if len(g) >= 3:
        res = stats.linregress(x, y)
        fig.add_scatter(x=g["date"], y=res.intercept + res.slope * x, mode="lines", name="Linear trend (scipy)",
                        line=dict(color="#E45756", width=2.5, dash="dash"))
        info.update(slope=float(res.slope), r2=float(res.rvalue ** 2))
    fig.update_xaxes(rangeslider_visible=True, tickformat="%Y", dtick="M12")
    fig.update_yaxes(title_text="Registered cases", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Change (%)", secondary_y=True, showgrid=False)
    return base_layout(fig, f"Time series - {label}", 520), info


def fig_hist(values, nbins):
    counts, edges = np.histogram(values, bins=nbins)
    centers, width = (edges[:-1] + edges[1:]) / 2, edges[1] - edges[0]
    fig = go.Figure(go.Bar(
        x=centers, y=counts, width=width * 0.92, name="Records", marker_color="#4C78A8",
        customdata=np.c_[edges[:-1], edges[1:]],
        hovertemplate="%{customdata[0]:,.0f} - %{customdata[1]:,.0f} cases<br>Records: %{y}<extra></extra>"))
    if len(values) >= 3 and np.ptp(values) > 0:
        kde = stats.gaussian_kde(values)
        xs = np.linspace(edges[0] - width * 0.5, edges[-1] + width * 0.5, 200)
        fig.add_scatter(x=xs, y=kde(xs) * len(values) * width, mode="lines", name="Density (KDE, scaled)",
                        line=dict(color="#F58518", width=3))
    fig.update_layout(bargap=0.04, xaxis_title="Registered cases per record", yaxis_title="Number of records")
    return base_layout(fig, "Distribution of case counts (histogram)", 430)


def fig_rank_scatter(f, speed, geo_colors):
    d = f[f["scope"] == "State"].copy()
    if d.empty:
        return None
    d["rank"] = d.groupby("year")["cases"].rank(ascending=False, method="first").astype(int)
    years = sorted(int(y) for y in d["year"].unique())
    fig = px.scatter(
        d, x="rank", y="cases", size="cases", color="geography", text="geography",
        animation_frame="year", size_max=46, log_y=True,
        category_orders={"year": years}, color_discrete_map=geo_colors,
        labels={"rank": "Rank within year (1 = most cases)", "cases": "Registered cases"})
    update_all_traces(fig, textposition="top center")
    fig.update_xaxes(range=[0.4, d["rank"].max() + 0.8], dtick=1)
    fig.update_yaxes(range=[np.log10(d["cases"].min() * 0.6), np.log10(d["cases"].max() * 1.9)])
    base_layout(fig, "Rank vs cases for states (animated scatter, log scale)", 520)
    fig.update_layout(showlegend=False)
    return play_controls(fig, [fr.name for fr in fig.frames], speed)


def fig_paired(ch):
    lo = min(ch["first_cases"].min(), ch["last_cases"].min()) * 0.7
    hi = max(ch["first_cases"].max(), ch["last_cases"].max()) * 1.5
    fig = go.Figure()
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="No change (y = x)",
                    line=dict(color="grey", dash="dot"))
    for i, r in ch.reset_index(drop=True).iterrows():
        fig.add_scatter(
            x=[r["first_cases"]], y=[r["last_cases"]], mode="markers+text", text=[r["geography"]],
            textposition="top center", name=r["geography"],
            marker=dict(size=18, color=PALETTE[i % len(PALETTE)], line=dict(color="white", width=1.5)),
            hovertemplate=(f"<b>{r['geography']}</b><br>{int(r['first_year'])}: {r['first_cases']:,.0f}"
                           f"<br>{int(r['last_year'])}: {r['last_cases']:,.0f}"
                           f"<br>Change: {r['change_pct']:+.1f}%<extra></extra>"))
    fig.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], title="Cases in first reported year")
    fig.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)], title="Cases in latest reported year")
    base_layout(fig, "First vs latest reported year (scatter) - above the dotted line = increase", 430)
    fig.update_layout(showlegend=False)
    return fig


def fig_radar(f):
    d = f[(f["scope"] == "State") & (f["offence_short"] == ALL_OFF)]
    states = sorted(d["geography"].unique(), key=lambda g: -d[d["geography"] == g]["cases"].max())
    if len(states) < 3:
        return None
    fig = go.Figure()
    for i, y in enumerate(sorted(d["year"].unique())):
        dy = d[d["year"] == y].set_index("geography")["cases"]
        r = [float(dy[s]) if s in dy.index else None for s in states]
        complete = all(v is not None for v in r)
        fig.add_trace(go.Scatterpolar(
            r=r + [r[0]], theta=states + [states[0]], name=str(int(y)), mode="lines+markers",
            fill="toself" if complete else "none", opacity=0.85, connectgaps=False,
            line=dict(color=PALETTE[i % len(PALETTE)], width=3), marker=dict(size=8),
            hovertemplate="%{theta}: %{r:,.0f}<extra>" + str(int(y)) + "</extra>"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, d["cases"].max() * 1.15])))
    return base_layout(fig, "State profile by year (radar)", 480)


def fig_heatmap(f):
    d = f[(f["scope"] == "State") & (f["offence_short"] == ALL_OFF)]
    if d.empty:
        return None
    pv = d.pivot_table(index="geography", columns="year", values="cases", aggfunc="sum")
    pv = pv.loc[pv.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7, max(2.6, 0.55 * len(pv) + 1.2)))
    ax.set_facecolor("#EEF0F4")
    sns.heatmap(pv, annot=True, fmt=",.0f", cmap="YlOrRd", linewidths=0.8, linecolor="white",
                cbar_kws={"label": "Registered cases"}, mask=pv.isna(), ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("State × year heatmap (seaborn) - grey = not in dataset", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Load data, build global lookups
# --------------------------------------------------------------------------- #
if not DATA_PATH.exists():
    st.error(f"Dataset not found at `{DATA_PATH}`. Keep the CSV in the same folder as app.py.")
    st.stop()

raw = load_data()
cases = raw[raw["cases"].notna()].copy()
share_info = parse_gender_share(raw)
comp_all = build_components(cases)

NUMERIC_COLS = ["victims_girls", "victims_boys", "victims_other_gender",
                "cases_solved_chargesheeted", "cases_convicted"]
empty_cols = [c for c in NUMERIC_COLS if c in raw.columns and raw[c].isna().all()]

GEO_ORDER = list(cases.sort_values("cases", ascending=False)["geography"].drop_duplicates())
ALL_GEOS = list(cases["geography"].drop_duplicates())
GEO_COLORS = {g: (NAVY if g.startswith("India") else PALETTE[i % len(PALETTE)])
              for i, g in enumerate(ALL_GEOS)}
OFF_COLORS = {ALL_OFF: "#4C78A8", SEC_OFF: "#E45756"}

years_all = [int(y) for y in sorted(cases["year"].unique())]
offs_all = list(cases["offence_short"].drop_duplicates())
lo_cases, hi_cases = int(cases["cases"].min()), int(cases["cases"].max())

# --------------------------------------------------------------------------- #
# Sidebar - filters (no file upload by design)
# --------------------------------------------------------------------------- #
DEFAULTS = {"f_years": years_all, "f_geos": ALL_GEOS, "f_offs": offs_all, "f_range": (lo_cases, hi_cases)}


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v


with st.sidebar:
    st.header("🎛️ Filters")
    st.multiselect("Year", years_all, default=years_all, key="f_years")
    st.multiselect("Geography", ALL_GEOS, default=ALL_GEOS, key="f_geos")
    st.multiselect("Offence category", offs_all, default=offs_all, key="f_offs")
    st.slider("Registered cases (range)", lo_cases, hi_cases, value=(lo_cases, hi_cases), step=1, key="f_range")
    st.button("↺ Reset filters", on_click=reset_filters)
    st.divider()
    st.header("🎬 Animation")
    speed = st.slider("Frame duration (ms)", 300, 2500, 900, step=100,
                      help="Lower = faster. Applies to every animated chart.")
    st.caption("Use ▶ Play under the animated charts, or drag their year slider.")

mask = (cases["year"].isin(st.session_state["f_years"])
        & cases["geography"].isin(st.session_state["f_geos"])
        & cases["offence_short"].isin(st.session_state["f_offs"])
        & cases["cases"].between(*st.session_state["f_range"]))
f = cases[mask].sort_values(["year", "geography"]).copy()

comp_f = comp_all[
    comp_all["year"].isin(st.session_state["f_years"])
    & comp_all["geography"].isin(st.session_state["f_geos"])
    & comp_all["cases"].between(*st.session_state["f_range"])
].copy() if ALL_OFF in st.session_state["f_offs"] else comp_all.iloc[0:0].copy()

st.sidebar.divider()
st.sidebar.caption(f"Showing **{len(f)}** of {len(cases)} records")

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.markdown(
    f"""<div class="hero"><h1>Child Sexual Offence Cases in India</h1>
    <p>POCSO Act registered-case figures, {years_all[0]}-{years_all[-1]} · NCRB data as compiled in the bundled CSV
    ({len(cases)} case-count records)</p></div>""",
    unsafe_allow_html=True,
)

if f.empty:
    st.warning("No records match the current filters. Widen the filters or press **Reset filters**.")
    st.stop()

render_kpis(compute_kpis(f, len(cases)))

tab_over, tab_trend, tab_geo, tab_dist, tab_data = st.tabs(
    ["📈 Overview", "⏱ Trends", "🗺 Geography", "📊 Distribution & relationships", "🗂 Data & notes"])

# ------------------------------- Overview ---------------------------------- #
with tab_over:
    c1, c2 = st.columns([2, 1])
    with c1:
        mode = st.radio("Scope", ["All records (log scale)", "States only", "National only"],
                        horizontal=True, key="bar_mode")
        fig = fig_bar_race(f, mode, speed, OFF_COLORS)
        if fig is None:
            st.info("No records for this scope under the current filters.")
        else:
            show(fig, "bar_race")
            if mode.startswith("All"):
                st.caption("Log scale: national totals are ~10x larger than single-state counts, so bar "
                           "length is not proportional to value here.")
    with c2:
        if share_info:
            show(fig_donut(share_info), "donut")
            st.caption(f"Midpoint of reported ranges (girls {share_info['girls_range'][0]}-"
                       f"{share_info['girls_range'][1]}%, boys {share_info['boys_range'][0]}-"
                       f"{share_info['boys_range'][1]}%). Percentages, not counts; not affected by filters.")
        else:
            st.info("No gender-share information in the dataset.")

    st.subheader("Composition")
    if comp_f.empty:
        st.info("Composition charts need 'All POCSO offences' selected and at least one matching record.")
    else:
        p1, p2 = st.columns(2)
        with p1:
            labels = list(dict.fromkeys(comp_f.sort_values("year")["year_label"]))
            counts = comp_f.groupby("year_label")["component"].count()
            default_idx = labels.index(counts.idxmax()) if len(labels) else 0
            ylab = st.selectbox("Year for pie chart", labels, index=default_idx, key="pie_year")
            show(fig_pie(comp_f, ylab), "pie")
        with p2:
            show(fig_sunburst(comp_f), "sunburst")
        st.caption("'Rest of India' and 'Other POCSO offences' are **derived** (national total minus the parts listed "
                   "in the dataset). Years labelled 'listed states only' have no national total in the dataset.")

# -------------------------------- Trends ----------------------------------- #
with tab_trend:
    log_line = st.toggle("Log scale", value=False, key="line_log")
    show(fig_line_draw(f, log_line, speed, GEO_COLORS), "line_draw")

    st.subheader("Time series with trend")
    opts = series_options(f)
    if not opts:
        st.info("A time series needs a series with at least 2 reported years. Widen the year filter.")
    else:
        label = st.selectbox("Series", list(opts.keys()), key="ts_series")
        fig_ts, info = fig_timeseries(opts[label], label)
        show(fig_ts, "timeseries")
        if info["slope"] is not None:
            st.caption(f"Linear fit (scipy.stats.linregress): **{info['slope']:+,.0f} cases per year**, "
                       f"R² = {info['r2']:.2f}, n = {info['n']} years. Indicative only - not a forecast.")
        else:
            st.caption(f"Only {info['n']} reported years, so no trend line is fitted.")

    ch = change_table(f)
    if not ch.empty:
        st.subheader("Change between first and latest reported year")
        tbl = ch.rename(columns={"geography": "Geography", "first_year": "From", "last_year": "To",
                                 "first_cases": "Cases (from)", "last_cases": "Cases (to)",
                                 "change_pct": "Change %", "cagr_pct": "CAGR %/yr"}).drop(columns="n")
        st.dataframe(tbl.style.format({"Cases (from)": "{:,.0f}", "Cases (to)": "{:,.0f}",
                                       "Change %": "{:+.1f}", "CAGR %/yr": "{:+.1f}"}), hide_index=True)

# ------------------------------- Geography --------------------------------- #
with tab_geo:
    g1, g2 = st.columns(2)
    with g1:
        fr = fig_radar(f)
        if fr is None:
            st.info("A radar needs at least 3 states in view (all-offence records).")
        else:
            show(fr, "radar")
            st.caption("Each axis is a state; each polygon is a year. Years with fewer than all states "
                       "show as points (the dataset has no value for the missing states).")
    with g2:
        hm = fig_heatmap(f)
        if hm is None:
            st.info("No state-level records in view.")
        else:
            st.pyplot(hm)
            plt.close(hm)

# --------------------- Distribution & relationships ------------------------ #
with tab_dist:
    d1, d2 = st.columns([1, 2])
    with d1:
        scope_h = st.selectbox("Records to include", ["State-level records", "National records", "All records"],
                               key="hist_scope")
    with d2:
        nb = st.slider("Histogram bins", 3, 12, 5, key="hist_bins")
    hv = f if scope_h == "All records" else f[f["scope"] == ("State" if scope_h.startswith("State") else "National")]
    vals = hv["cases"].to_numpy(dtype=float)
    if len(vals) < 2:
        st.info("A histogram needs at least 2 records. Change the scope or widen the filters.")
    else:
        show(fig_hist(vals, nb), "hist")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Records (n)", f"{len(vals)}")
        m2.metric("Mean", f"{vals.mean():,.0f}")
        m3.metric("Median", f"{np.median(vals):,.0f}")
        m4.metric("Skewness", f"{stats.skew(vals):.2f}" if len(vals) >= 3 else "n/a")
        st.caption(f"Very small sample (n = {len(vals)}): the shape is descriptive, not a statistical finding.")

    s1, s2 = st.columns([3, 2])
    with s1:
        fs = fig_rank_scatter(f, speed, GEO_COLORS)
        if fs is None:
            st.info("No state-level records in view for the animated scatter.")
        else:
            show(fs, "rank_scatter")
    with s2:
        ch2 = change_table(f)
        if ch2.empty:
            st.info("The first-vs-latest scatter needs a series with ≥2 reported years.")
        else:
            show(fig_paired(ch2), "paired")

# ------------------------------ Data & notes -------------------------------- #
with tab_data:
    st.subheader("Records in view")
    cols = ["year", "geography", "offence_category", "cases", "source", "notes"]
    view = f[cols].rename(columns={"cases": "total_cases_registered"})
    st.dataframe(view, hide_index=True)
    st.download_button("⬇ Download filtered records (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="pocso_filtered_records.csv", mime="text/csv")

    st.subheader("About this data")
    notes = [
        f"The dataset has **{len(raw)} rows**; {len(cases)} carry a case count, "
        f"{len(raw) - len(cases)} carries only victim-gender percentages (donut chart).",
        "Counts are **registered cases** (not victims, not convictions). A rise can reflect more reporting "
        "and registration as well as more offences.",
        "Coverage is patchy: all-India totals exist for 2020-2022; state figures only for 2021 (5 states), "
        "Uttar Pradesh 2022 and Karnataka 2023.",
        "The 2020 'Sec 4 & 6' value is reported only as 'more than 28,000', so it is a lower bound.",
    ]
    if empty_cols:
        notes.append("Columns present but **entirely empty** (so not charted): "
                     + ", ".join(f"`{c}`" for c in empty_cols) + ".")
    st.markdown("\n".join(f"- {n}" for n in notes))
    st.markdown("**Sources named in the dataset**")
    st.markdown("\n".join(f"- {s}" for s in sorted(raw["source"].dropna().unique())))

st.divider()
st.caption("Built only from the bundled CSV. If a child is in danger in India, CHILDLINE 1098 is the 24×7 helpline.")
