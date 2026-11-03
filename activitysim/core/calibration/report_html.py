# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import base64
import html
import logging
from pathlib import Path

import pandas as pd

from activitysim.core import workflow

from .reporting import (
    CALIBRATION_ITERATION_FILE,
    CALIBRATION_SUMMARY_FILE,
    _component_output_dir,
)

logger = logging.getLogger(__name__)

REPORT_FILE = "calibration/calibration_report.html"

# PNGs written by the standard plots; everything else in a component
# directory is treated as a user-supplied diagnostic and embedded as-is.
_STANDARD_PNG_PREFIXES = (
    "coefficient_progress_set_",
    "final_components_set_",
    "final_pct_change_set_",
)

_CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:0;color:#222}
header{background:#1f3a5f;color:#fff;padding:12px 24px}
nav{display:flex;flex-wrap:wrap;gap:4px;padding:8px 24px;background:#eef1f5;
    position:sticky;top:0;z-index:5}
nav button{border:0;padding:8px 14px;cursor:pointer;background:#d5dce6;border-radius:4px}
nav button.active{background:#1f3a5f;color:#fff}
section{display:none;padding:16px 24px}
section.active{display:block}
table{border-collapse:collapse;font-size:13px}
th,td{border:1px solid #ccc;padding:3px 8px;text-align:right}
th{background:#eef1f5;position:sticky;top:0}
td.l,th.l{text-align:left}
tr.bad td{background:#fde8e8}
tr.ok td{background:#e9f7ec}
.scroll{max-height:420px;overflow:auto;display:inline-block}
.cards span{display:inline-block;margin-right:24px;font-size:15px}
img{max-width:100%;border:1px solid #ddd;margin:6px 0}
h3{margin-top:24px}
"""

_JS = """
function show(id){
  document.querySelectorAll('section').forEach(s=>s.classList.toggle('active',s.id===id));
  document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('active',b.dataset.t===id));
  window.dispatchEvent(new Event('resize'));
}
show(document.querySelector('nav button').dataset.t);
"""


def _latest_records(recs: pd.DataFrame) -> pd.DataFrame:
    """Return rows from the last update of each component."""
    key = ["global_iter", "attempt", "component_iter"]
    ordered = recs.sort_values(key)
    last_key = ordered.drop_duplicates("component", keep="last")[["component"] + key]
    return ordered.merge(last_key, on=["component"] + key)


def _table(df: pd.DataFrame, left_cols: tuple[str, ...] = ()) -> str:
    """Render a dataframe as an HTML table, shading rows by convergence."""
    head = "".join(
        f'<th class="{"l" if c in left_cols else ""}">{html.escape(str(c))}</th>'
        for c in df.columns
        if c != "converged"
    )
    rows = []
    for _, row in df.iterrows():
        cls = "ok" if row.get("converged", False) else "bad"
        cells = []
        for c in df.columns:
            if c == "converged":
                continue
            v = row[c]
            text = f"{v:.4g}" if isinstance(v, float) else html.escape(str(v))
            cells.append(f'<td class="{"l" if c in left_cols else ""}">{text}</td>')
        rows.append(f'<tr class="{cls}">{"".join(cells)}</tr>')
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


class _Figures:
    """Render plotly figures, embedding plotly.js only once."""

    def __init__(self):
        self._first = True

    def html(self, fig) -> str:
        out = fig.to_html(
            full_html=False,
            include_plotlyjs=True if self._first else False,
            default_height="480px",
        )
        self._first = False
        return out


def _overview(summary: pd.DataFrame | None, latest: pd.DataFrame, figs: _Figures):
    import plotly.express as px

    per_comp = (
        latest.groupby("component")
        .agg(
            coefficients=("coefficient", "count"),
            converged=("converged", "sum"),
            max_abs_pct_diff=("pct_difference", lambda s: s.abs().max()),
        )
        .reset_index()
    )
    per_comp["unconverged"] = per_comp["coefficients"] - per_comp["converged"]
    n_bad = int(per_comp["unconverged"].sum())
    out = [
        '<div class="cards">'
        f"<span><b>{len(per_comp)}</b> components</span>"
        f'<span><b>{int(per_comp["coefficients"].sum())}</b> targets</span>'
        f"<span><b>{n_bad}</b> not converged</span></div>",
        "<h3>Components</h3>",
        _table(per_comp.drop(columns="converged"), left_cols=("component",)),
        "<h3>Worst targets (all components, latest iteration)</h3>",
    ]
    scored = latest.copy()
    if "tolerance" in scored and scored["tolerance"].notna().all():
        scored["x_tolerance"] = (scored["difference"] / scored["tolerance"]).abs()
    else:
        scored["x_tolerance"] = scored["difference"].abs()
    worst = scored.sort_values("x_tolerance", ascending=False).head(50)[
        [
            "component",
            "description",
            "target_value",
            "model_value",
            "difference",
            "x_tolerance",
            "coefficient",
            "next_coefficient",
            "converged",
        ]
    ]
    out.append(_table(worst, left_cols=("component", "description", "coefficient")))

    if summary is not None and not summary.empty:
        s = summary.sort_values(["global_iter", "attempt", "component_iter"]).copy()
        s["step"] = (
            "G"
            + s["global_iter"].astype(str)
            + "-A"
            + s["attempt"].astype(str)
            + "-C"
            + s["component_iter"].astype(str)
        )
        fig = px.line(
            s,
            x="step",
            y="max_difference",
            color="component",
            markers=True,
            hover_data=["max_difference_coefficient"],
            title="Convergence: max difference per calibration update",
        )
        fig.update_xaxes(type="category")
        out += ["<h3>Convergence</h3>", figs.html(fig)]
    return "".join(out)


def _component_section(
    name: str, recs: pd.DataFrame, latest: pd.DataFrame, comp_dir: Path, figs: _Figures
) -> str:
    import plotly.graph_objects as go

    out = []
    last = latest[latest.component == name].copy()
    if "tolerance" in last and last["tolerance"].notna().all():
        # convergence is |difference| <= tolerance, so plot on that scale
        last["score"] = last["difference"] / last["tolerance"]
        y_title = "difference / tolerance (within ±1 = converged)"
        title = "Model minus target, in multiples of tolerance"
    else:
        last["score"] = last["difference"]
        y_title = "model - target (absolute)"
        title = "Model minus target (absolute difference)"
    last["abs_score"] = last["score"].abs()
    last = last.sort_values("abs_score", ascending=False)

    # residuals: one bar per target, sorted, so offenders appear first
    fig = go.Figure(
        go.Bar(
            x=last["description"],
            y=last["score"],
            marker_color=["#2e9e55" if c else "#d64545" for c in last["converged"]],
            customdata=last[["target_value", "model_value", "difference"]],
            hovertemplate="%{x}<br>score %{y:.3g}<br>target %{customdata[0]:.4g}"
            "<br>model %{customdata[1]:.4g}<br>diff %{customdata[2]:.4g}"
            "<extra></extra>",
        )
    )
    if y_title.startswith("difference /"):
        fig.add_hrect(y0=-1, y1=1, fillcolor="#2e9e55", opacity=0.08, line_width=0)
    fig.update_layout(
        title=title + " (green = converged)",
        yaxis_title=y_title,
        xaxis={"categoryorder": "array", "categoryarray": list(last["description"])},
    )
    out += [figs.html(fig)]

    # coefficient paths: every coefficient, toggled via the legend
    r = recs.sort_values(["global_iter", "attempt", "component_iter"]).copy()
    r["step"] = (
        "G"
        + r["global_iter"].astype(str)
        + "-A"
        + r["attempt"].astype(str)
        + "-C"
        + r["component_iter"].astype(str)
    )
    fig = go.Figure()
    for coef, g in r.groupby("coefficient", sort=False):
        fig.add_trace(
            go.Scatter(
                x=["Start"] + list(g["step"]),
                y=[g["prev_coefficient"].iloc[0]] + list(g["next_coefficient"]),
                mode="lines+markers",
                name=str(coef),
            )
        )
    fig.update_layout(
        title="Coefficient values by calibration update (click legend to toggle)",
        yaxis_title="Coefficient value",
    )
    out += [figs.html(fig)]

    out += [
        "<h3>Latest iteration</h3>",
        _table(
            last.drop(columns=["score", "abs_score"])[
                [
                    "description",
                    "coefficient",
                    "target_value",
                    "model_value",
                    "difference",
                    "pct_difference",
                    "prev_coefficient",
                    "next_coefficient",
                    "converged",
                ]
            ],
            left_cols=("description", "coefficient"),
        ),
    ]

    extra = sorted(
        p
        for p in comp_dir.glob("*.png")
        if not p.name.startswith(_STANDARD_PNG_PREFIXES)
    )
    if extra:
        out.append("<h3>Component diagnostics</h3>")
        for p in extra:
            data = base64.b64encode(p.read_bytes()).decode()
            out.append(
                f"<details open><summary>{html.escape(p.name)}</summary>"
                f'<img src="data:image/png;base64,{data}"></details>'
            )
    return "".join(out)


def write_calibration_report(state: workflow.State) -> Path | None:
    """Write output/calibration/calibration_report.html; returns its path."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        logger.info("plotly is not installed; skipping calibration HTML report")
        return None

    iter_path = state.get_output_file_path(CALIBRATION_ITERATION_FILE)
    if not iter_path.exists():
        return None
    recs = pd.read_csv(iter_path)
    if recs.empty:
        return None
    if "attempt" not in recs.columns:
        recs["attempt"] = 1

    summary_path = state.get_output_file_path(CALIBRATION_SUMMARY_FILE)
    summary = pd.read_csv(summary_path) if summary_path.exists() else None
    if summary is not None and "attempt" not in summary.columns:
        summary["attempt"] = 1

    latest = _latest_records(recs)
    figs = _Figures()
    components = list(dict.fromkeys(recs["component"]))

    sections = {"overview": _overview(summary, latest, figs)}
    for name in components:
        sections[f"c_{name}"] = _component_section(
            name,
            recs[recs.component == name],
            latest,
            _component_output_dir(state, name),
            figs,
        )

    labels = {"overview": "Overview", **{f"c_{n}": n for n in components}}
    nav = "".join(
        f'<button data-t="{k}" onclick="show(\'{k}\')">{html.escape(labels[k])}</button>'
        for k in sections
    )
    body = "".join(f'<section id="{k}">{v}</section>' for k, v in sections.items())
    doc = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>Calibration report</title><style>{_CSS}</style></head><body>"
        f"<header><h2>Calibration report</h2></header><nav>{nav}</nav>{body}"
        f"<script>{_JS}</script></body></html>"
    )
    path = state.get_output_file_path(REPORT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
