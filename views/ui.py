"""Small shared UI helpers so every page looks and behaves consistently."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Utility-monitoring palette. Amber reads as "instrumentation", red/green as status.
AMBER = "#f5a524"
RED = "#f0553b"
GREEN = "#2ea86a"
BLUE = "#4c8dff"
MUTED = "#8b949e"
GRID = "rgba(139,148,158,0.18)"

PHASE_COLORS = {"a": "#f5a524", "b": "#4c8dff", "c": "#2ea86a"}

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1250px;}
  [data-testid="stMetricValue"] {font-size: 1.65rem;}
  .pg-status {
      border-radius: 10px; padding: 1.1rem 1.3rem; margin: .4rem 0 1rem 0;
      border-left: 5px solid var(--pg-accent);
      background: var(--pg-bg);
  }
  .pg-status .pg-kicker {
      font-size: .72rem; letter-spacing: .16em; text-transform: uppercase;
      color: #8b949e; margin-bottom: .35rem;
  }
  .pg-status .pg-headline {font-size: 1.55rem; font-weight: 700; color: var(--pg-accent);}
  .pg-status .pg-detail {font-size: .95rem; color: #c9d1d9; margin-top: .3rem;}
  .pg-note {
      font-size: .82rem; color: #8b949e; border-left: 3px solid rgba(139,148,158,.35);
      padding-left: .8rem; margin: .5rem 0;
  }
  .pg-pipe {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: .84rem; line-height: 1.75; color: #c9d1d9;
      background: rgba(139,148,158,.08); border-radius: 8px; padding: 1rem 1.2rem;
  }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def status_banner(kicker: str, headline: str, detail: str = "", tone: str = "neutral") -> None:
    """Prominent grid-status block used on the Live Prediction page."""
    accents = {
        "fault": (RED, "rgba(240,85,59,.10)"),
        "normal": (GREEN, "rgba(46,168,106,.10)"),
        "neutral": (AMBER, "rgba(245,165,36,.10)"),
    }
    accent, bg = accents.get(tone, accents["neutral"])
    detail_html = f'<div class="pg-detail">{detail}</div>' if detail else ""
    st.markdown(
        f'<div class="pg-status" style="--pg-accent:{accent}; --pg-bg:{bg};">'
        f'<div class="pg-kicker">{kicker}</div>'
        f'<div class="pg-headline">{headline}</div>'
        f"{detail_html}</div>",
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="pg-note">{text}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 320, showlegend: bool = False) -> go.Figure:
    """Apply one consistent chart theme instead of restyling at every call site."""
    fig.update_layout(
        height=height,
        showlegend=showlegend,
        margin=dict(l=10, r=10, t=34, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c9d1d9", size=12),
        title=dict(font=dict(size=14, color="#e6edf3")),
        hoverlabel=dict(bgcolor="#161b22", font_size=12),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def phase_bar(labels, values, title: str, unit: str) -> go.Figure:
    """Grouped bar for the three phases, coloured consistently across the app."""
    colors = [PHASE_COLORS[label[-1].lower()] for label in labels]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:,.3f}" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(title=title, yaxis_title=unit)
    return style_fig(fig, height=300)
