"""Zero-Day DoS Detection Engine — Cybersecurity Dashboard.

Streamlit application with dark hacker aesthetic for real-time
anomaly detection monitoring and alert visualization.
"""
from __future__ import annotations

import datetime
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Path Setup 
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.dashboard.theme import (
    COLORS,
    FONT_MONO,
    GLOBAL_CSS,
    hex_decoration,
    severity_badge,
    stat_row,
    system_health_bar,
    terminal_block,
    threat_level_banner,
    zero_day_alert_banner,
)
from src.dashboard.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

# Page Config 
st.set_page_config(
    page_title="ZION-TECH — Zero-Day Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Global Styles 
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown(hex_decoration(), unsafe_allow_html=True)


# Session State Defaults 
def _init_state() -> None:
    """Initialize session state on first run."""
    defaults = {
        "metrics": MetricsCalculator(),
        "alerts": [],
        "detection_log": [],
        "model_status": {
            "loaded": False,
            "anomaly_loaded": False,
            "classifier_loaded": False,
        },
        "active_page": "App",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# Sidebar 
def _render_sidebar() -> None:
    """Render the left sidebar with system controls and status."""
    with st.sidebar:
        # Logo / Title
        st.markdown(f"""
        <div style="text-align:center;padding:0.5rem 0 1rem 0;">
            <div style="
                font-family:{FONT_MONO};font-size:1.4rem;font-weight:700;
                color:{COLORS['accent_green']};
                text-shadow:0 0 15px {COLORS['accent_green']}60;
                letter-spacing:0.08em;
            ">🛡️ ZION-TECH</div>
            <div style="
                font-family:{FONT_MONO};font-size:0.6rem;
                color:{COLORS['text_muted']};
                text-transform:uppercase;letter-spacing:0.2em;
                margin-top:2px;
            ">Zero-Day DoS Detection Engine</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{COLORS['border']};'>", unsafe_allow_html=True)

        # System Status Panel 
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.65rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.15em;margin-bottom:0.5rem;
        ">◆ System Status</div>
        """, unsafe_allow_html=True)

        ms = st.session_state["model_status"]
        if ms.get("loaded"):
            st.markdown(system_health_bar("healthy", "All models loaded and operational"), unsafe_allow_html=True)
        elif ms.get("anomaly_loaded") or ms.get("classifier_loaded"):
            st.markdown(system_health_bar("warning", "Partial model load — check classifier"), unsafe_allow_html=True)
        else:
            st.markdown(system_health_bar("critical", "No models loaded"), unsafe_allow_html=True)

        # Model Details 
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.65rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.15em;margin:0.75rem 0 0.5rem 0;
        ">◆ Model Status</div>
        """, unsafe_allow_html=True)

        anomaly_icon = "●" if ms.get("anomaly_loaded") else "○"
        anomaly_color = COLORS["healthy"] if ms.get("anomaly_loaded") else COLORS["text_muted"]
        classifier_icon = "●" if ms.get("classifier_loaded") else "○"
        classifier_color = COLORS["healthy"] if ms.get("classifier_loaded") else COLORS["text_muted"]

        st.markdown(f"""
        <div style="font-family:{FONT_MONO};font-size:0.75rem;">
            <div style="display:flex;justify-content:space-between;padding:0.3rem 0;">
                <span style="color:{COLORS['text_secondary']};">Isolation Forest</span>
                <span style="color:{anomaly_color};">{anomaly_icon} {"ACTIVE" if ms.get("anomaly_loaded") else "INACTIVE"}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:0.3rem 0;">
                <span style="color:{COLORS['text_secondary']};">Random Forest</span>
                <span style="color:{classifier_color};">{classifier_icon} {"ACTIVE" if ms.get("classifier_loaded") else "INACTIVE"}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{COLORS['border']};'>", unsafe_allow_html=True)

        # Quick Stats 
        metrics: MetricsCalculator = st.session_state["metrics"]
        det = metrics.get_detection_metrics()
        alrt = metrics.get_alert_metrics()

        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.65rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.15em;margin-bottom:0.5rem;
        ">◆ Session Stats</div>
        """, unsafe_allow_html=True)

        st.markdown(stat_row("Detections", str(det["total_detections"])), unsafe_allow_html=True)
        st.markdown(stat_row("Anomaly Rate", f"{det['anomaly_rate']:.2%}"), unsafe_allow_html=True)
        st.markdown(stat_row("Zero-Day Rate", f"{det['zero_day_rate']:.2%}"), unsafe_allow_html=True)
        st.markdown(stat_row("Avg Latency", f"{det['avg_latency_ms']:.1f}ms"), unsafe_allow_html=True)
        st.markdown(stat_row("Alerts", str(alrt["total_alerts"])), unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{COLORS['border']};'>", unsafe_allow_html=True)

        # Clock 
        now = datetime.datetime.now()
        st.markdown(f"""
        <div style="text-align:center;padding:0.5rem 0;">
            <div style="
                font-family:{FONT_MONO};font-size:1.1rem;font-weight:600;
                color:{COLORS['accent_cyan']};
            ">{now.strftime('%H:%M:%S')}</div>
            <div style="
                font-family:{FONT_MONO};font-size:0.6rem;
                color:{COLORS['text_muted']};text-transform:uppercase;
                letter-spacing:0.1em;
            ">{now.strftime('%d %b %Y')}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.55rem;
            color:{COLORS['text_muted']};text-align:center;
            margin-top:1rem;padding-top:0.5rem;
            border-top:1px solid {COLORS['border']};
        ">
            ZION-TECH v1.0 — Oyelude Zion Clifford<br/>
            FUTA CYS/20/4940 — Prof. (Mrs.) Alowolodu
        </div>
        """, unsafe_allow_html=True)


_render_sidebar()


# Navigation Tabs 
from src.dashboard.views.overview import render_overview
from src.dashboard.views.alerts import render_alerts
from src.dashboard.views.metrics import render_metrics
from src.dashboard.views.models import render_models

_PAGES = ["App", "Overview", "Alerts", "Metrics", "Models"]
_page_index = _PAGES.index(st.session_state.get("active_page", "App"))

# Inject nav styles (Tactical Command HUD) 
st.markdown(f"""
<style>
/* ══════════════════════════════════════════════════════════════════════════════
   NAV — Tactical Command Strip
   ══════════════════════════════════════════════════════════════════════════════ */

/*  Kill ALL Streamlit width constraints on the radio tree  */
div[data-testid="stRadio"],
div[data-testid="stRadio"] > div,
div[data-testid="stRadio"] > div > div {{
    max-width:none !important;
    width:100% !important;
    padding-left:0 !important;
    padding-right:0 !important;
}}

/*  Container: the command strip  */
div[data-testid="stRadio"] > div {{
    display:flex !important;
    gap:0.5rem !important;
    flex-wrap:nowrap;
    padding:0.75rem 0.5rem;
    background:
        linear-gradient(180deg, {COLORS['bg_secondary']} 0%, {COLORS['bg_primary']}ee 100%);
    border:1px solid {COLORS['border']};
    border-top:2px solid {COLORS['accent_green']}50;
    border-radius:0;
    margin-bottom:1.75rem;
    position:relative;
    overflow:hidden;
    animation:navContainerIn 0.5s cubic-bezier(.4,0,.2,1) both;
    box-shadow:
        inset 0  1px 0 {COLORS['accent_green']}10,
        0 4px 24px #00000040;
}}
@keyframes navContainerIn {{
    0%   {{ opacity:0; transform:translateY(-8px); }}
    100% {{ opacity:1; transform:translateY(0); }}
}}

/*  Top scan-line  */
div[data-testid="stRadio"] > div::before {{
    content:'';
    position:absolute;
    top:0; left:0; right:0;
    height:2px;
    background:linear-gradient(90deg,
        transparent 0%,
        {COLORS['accent_green']}00 20%,
        {COLORS['accent_green']} 50%,
        {COLORS['accent_green']}00 80%,
        transparent 100%);
    animation:navScanSweep 5s linear infinite;
    z-index:10;
}}
@keyframes navScanSweep {{
    0%   {{ transform:translateX(-100%); opacity:0; }}
    8%   {{ opacity:1; }}
    92%  {{ opacity:1; }}
    100% {{ transform:translateX(100%); opacity:0; }}
}}

/*  Bottom accent line  */
div[data-testid="stRadio"] > div::after {{
    content:'';
    position:absolute;
    bottom:0; left:5%; right:5%;
    height:1px;
    background:linear-gradient(90deg,
        transparent,
        {COLORS['accent_green']}40,
        transparent);
}}

/*  Individual button  */
div[data-testid="stRadio"] label {{
    display:flex !important;
    align-items:center;
    justify-content:center;
    flex:1;
    padding:1.5rem 0.75rem;
    border-radius:0;
    font-family:{FONT_MONO} !important;
    font-size:1.1rem;
    font-weight:700;
    letter-spacing:0.1em;
    text-transform:uppercase;
    cursor:pointer;
    text-align:center;
    border:1px solid {COLORS['border']};
    background:{COLORS['bg_card']};
    color:{COLORS['text_muted']};
    transition:
        color 0.25s ease,
        background 0.25s ease,
        box-shadow 0.25s ease,
        transform 0.2s ease,
        border-color 0.25s ease;
    margin:0 !important;
    white-space:nowrap;
    position:relative;
    overflow:hidden;
    animation:navBtnIn 0.45s cubic-bezier(.4,0,.2,1) both;
}}
div[data-testid="stRadio"] label:nth-child(1) {{ animation-delay:0.08s; }}
div[data-testid="stRadio"] label:nth-child(2) {{ animation-delay:0.14s; }}
div[data-testid="stRadio"] label:nth-child(3) {{ animation-delay:0.20s; }}
div[data-testid="stRadio"] label:nth-child(4) {{ animation-delay:0.26s; }}
div[data-testid="stRadio"] label:nth-child(5) {{ animation-delay:0.32s; }}
@keyframes navBtnIn {{
    0%   {{ opacity:0; transform:translateY(6px); }}
    100% {{ opacity:1; transform:translateY(0); }}
}}

/*  Button: subtle inner left-edge highlight  */
div[data-testid="stRadio"] label::before {{
    content:'';
    position:absolute;
    top:20%; bottom:20%;
    left:0; width:2px;
    background:{COLORS['accent_green']}00;
    transition:background 0.25s ease, box-shadow 0.25s ease;
}}

/*  Hover — pure neon red  */
div[data-testid="stRadio"] label:hover {{
    color:#ff0033 !important;
    background:#ff003310 !important;
    border-color:#ff003340 !important;
    box-shadow:
        inset 0 -4px 0 #ff0033,
        inset 0 0 30px #ff003308,
        0 0 25px #ff003318;
    transform:translateY(-1px);
}}
div[data-testid="stRadio"] label:hover::before {{
    background:#ff0033;
    box-shadow:0 0 8px #ff003380;
}}

/*  Selected / Active — neon green  */
div[data-testid="stRadio"] input:checked + span {{
    border:none !important;
    color:{COLORS['accent_green']} !important;
    background:linear-gradient(180deg,
        {COLORS['accent_green']}14 0%,
        {COLORS['accent_green']}06 100%) !important;
    box-shadow:
        inset 0 -4px 0 {COLORS['accent_green']},
        inset 0 0 40px {COLORS['accent_green']}08,
        0 0 30px {COLORS['accent_green']}20 !important;
    text-shadow:0 0 14px {COLORS['accent_green']}60;
}}
div[data-testid="stRadio"] input:checked + span::before {{
    background:{COLORS['accent_green']} !important;
    box-shadow:0 0 10px {COLORS['accent_green']}80 !important;
}}

/*  Focus ring (keyboard a11y)  */
div[data-testid="stRadio"] label:focus-within {{
    outline:2px solid {COLORS['accent_cyan']}60;
    outline-offset:-2px;
}}

/*  Hide duplicate button row  */
div[data-testid="stHorizontalBlock"]:has(div[data-testid="stButton"]) {{
    display:none !important;
}}
</style>
</style>
""", unsafe_allow_html=True)

active_page = st.radio(
    "Navigation",
    _PAGES,
    index=_page_index,
    horizontal=True,
    label_visibility="collapsed",
    key="nav_radio",
)
if active_page != st.session_state.get("active_page"):
    st.session_state["active_page"] = active_page
    st.rerun()


# Page Routing 

_page = st.session_state["active_page"]


def _render_page_header(title: str, subtitle: str = "") -> None:
    """Render a consistent page header with cyber styling."""
    st.markdown(f"""
    <div style="
        padding:0.75rem 0;margin-bottom:1rem;
        border-bottom:1px solid {COLORS['border']};
    ">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="
                width:4px;height:28px;border-radius:2px;
                background:linear-gradient(180deg, {COLORS['accent_green']}, {COLORS['accent_cyan']});
                box-shadow:0 0 8px {COLORS['accent_green']}40;
            "></div>
            <div>
                <div style="
                    font-family:{FONT_MONO};font-size:1.1rem;font-weight:700;
                    color:{COLORS['text_primary']};
                    text-transform:uppercase;letter-spacing:0.08em;
                ">{title}</div>
                {"<div style='font-family:" + FONT_MONO + ";font-size:0.65rem;color:" + COLORS['text_muted'] + ";margin-top:2px;letter-spacing:0.05em;'>" + subtitle + "</div>" if subtitle else ""}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# App (Main Dashboard) 
if _page == "App":
    _render_page_header("ZION-TECH Dashboard", "Real-Time Autonomous Anomaly Detection for Zero-Day DoS Exploits")
    page_data = render_overview(
        st.session_state["metrics"],
        st.session_state["model_status"],
    )

    # Zero-day alert banner
    zd = page_data.get("severity_distribution", {}).get("zero_day", 0)
    st.markdown(zero_day_alert_banner(zd), unsafe_allow_html=True)

    # Threat level
    anomaly_rate = page_data["cards"][1]["value"] if len(page_data["cards"]) > 1 else "0.00%"
    try:
        rate_val = float(str(anomaly_rate).strip("%"))
    except (ValueError, TypeError):
        rate_val = 0.0

    if rate_val > 10:
        threat = "critical"
    elif rate_val > 5:
        threat = "high"
    elif rate_val > 1:
        threat = "medium"
    else:
        threat = "low"

    st.markdown(threat_level_banner(threat, rate_val / 100), unsafe_allow_html=True)

    # Metric cards row
    cols = st.columns(len(page_data["cards"]))
    for i, card in enumerate(page_data["cards"]):
        with cols[i]:
            delta_str = ""
            if "delta" in card:
                d = card["delta"]
                ds = card.get("delta_suffix", "%")
                dp = card.get("delta_positive", False)
                delta_str = f"{d:+.1f}{ds}" if isinstance(d, (int, float)) else str(d)
            st.metric(
                label=card["title"],
                value=card["value"],
                delta=delta_str if delta_str else None,
            )

    st.markdown(f"<hr style='border-color:{COLORS['border']};'>", unsafe_allow_html=True)

    # System status + class distribution
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin-bottom:0.75rem;
        ">◆ System Status</div>
        """, unsafe_allow_html=True)
        st.markdown(page_data["system_status"]["details"], unsafe_allow_html=True)
        st.markdown(system_health_bar(
            page_data["system_status"]["status"],
            page_data["system_status"]["details"],
        ), unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin-bottom:0.75rem;
        ">◆ Traffic Distribution</div>
        """, unsafe_allow_html=True)
        if page_data.get("class_distribution"):
            from src.dashboard.components.charts import create_class_distribution_chart
            import plotly.graph_objects as go

            chart_data = create_class_distribution_chart(page_data["class_distribution"])
            fig = go.Figure(data=[go.Pie(
                labels=chart_data["labels"],
                values=chart_data["values"],
                hole=0.6,
                marker=dict(colors=[
                    COLORS["accent_green"], COLORS["accent_cyan"],
                    COLORS["accent_blue"], COLORS["high"],
                    COLORS["critical"], COLORS["zero_day"],
                    COLORS["medium"], COLORS["low"],
                ]),
                textfont=dict(family=FONT_MONO, size=10, color=COLORS["text_primary"]),
                textinfo="label+percent",
            )])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT_MONO, color=COLORS["text_primary"]),
                showlegend=True,
                legend=dict(
                    font=dict(size=9, family=FONT_MONO, color=COLORS["text_secondary"]),
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(l=20, r=20, t=10, b=10),
                height=280,
            )
            st.plotly_chart(fig, width="stretch")

    # Recent detection log
    log = st.session_state.get("detection_log", [])
    if log:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin:1rem 0 0.5rem 0;
        ">◆ Recent Detections</div>
        """, unsafe_allow_html=True)
        st.markdown(terminal_block(log[-15:]), unsafe_allow_html=True)


# Overview 
elif _page == "Overview":
    _render_page_header("System Overview", "Real-time threat monitoring dashboard")
    page_data = render_overview(
        st.session_state["metrics"],
        st.session_state["model_status"],
    )

    # Zero-day alert banner
    zd = page_data.get("severity_distribution", {}).get("zero_day", 0)
    st.markdown(zero_day_alert_banner(zd), unsafe_allow_html=True)

    # Threat level
    anomaly_rate = page_data["cards"][1]["value"] if len(page_data["cards"]) > 1 else "0.00%"
    try:
        rate_val = float(str(anomaly_rate).strip("%"))
    except (ValueError, TypeError):
        rate_val = 0.0

    if rate_val > 10:
        threat = "critical"
    elif rate_val > 5:
        threat = "high"
    elif rate_val > 1:
        threat = "medium"
    else:
        threat = "low"

    st.markdown(threat_level_banner(threat, rate_val / 100), unsafe_allow_html=True)

    # Metric cards row
    cols = st.columns(len(page_data["cards"]))
    for i, card in enumerate(page_data["cards"]):
        with cols[i]:
            delta_str = ""
            if "delta" in card:
                d = card["delta"]
                ds = card.get("delta_suffix", "%")
                dp = card.get("delta_positive", False)
                delta_str = f"{d:+.1f}{ds}" if isinstance(d, (int, float)) else str(d)
            st.metric(
                label=card["title"],
                value=card["value"],
                delta=delta_str if delta_str else None,
            )

    st.markdown(f"<hr style='border-color:{COLORS['border']};'>", unsafe_allow_html=True)

    # System status + class distribution
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin-bottom:0.75rem;
        ">◆ System Status</div>
        """, unsafe_allow_html=True)
        st.markdown(page_data["system_status"]["details"], unsafe_allow_html=True)
        st.markdown(system_health_bar(
            page_data["system_status"]["status"],
            page_data["system_status"]["details"],
        ), unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin-bottom:0.75rem;
        ">◆ Traffic Distribution</div>
        """, unsafe_allow_html=True)
        if page_data.get("class_distribution"):
            from src.dashboard.components.charts import create_class_distribution_chart
            import plotly.graph_objects as go

            chart_data = create_class_distribution_chart(page_data["class_distribution"])
            fig = go.Figure(data=[go.Pie(
                labels=chart_data["labels"],
                values=chart_data["values"],
                hole=0.6,
                marker=dict(colors=[
                    COLORS["accent_green"], COLORS["accent_cyan"],
                    COLORS["accent_blue"], COLORS["high"],
                    COLORS["critical"], COLORS["zero_day"],
                    COLORS["medium"], COLORS["low"],
                ]),
                textfont=dict(family=FONT_MONO, size=10, color=COLORS["text_primary"]),
                textinfo="label+percent",
            )])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT_MONO, color=COLORS["text_primary"]),
                showlegend=True,
                legend=dict(
                    font=dict(size=9, family=FONT_MONO, color=COLORS["text_secondary"]),
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(l=20, r=20, t=10, b=10),
                height=280,
            )
            st.plotly_chart(fig, width="stretch")

    # Recent detection log
    log = st.session_state.get("detection_log", [])
    if log:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin:1rem 0 0.5rem 0;
        ">◆ Recent Detections</div>
        """, unsafe_allow_html=True)
        st.markdown(terminal_block(log[-15:]), unsafe_allow_html=True)


# Alerts 
elif _page == "Alerts":
    _render_page_header("Threat Alerts", "Alert history and zero-day detection events")
    page_data = render_alerts(st.session_state["alerts"])

    # Summary row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Alerts", page_data["total_alerts"])
    with c2:
        st.metric("Zero-Day Events", page_data["zero_day_alerts"])
    with c3:
        st.metric("Unique Severities", len(page_data["severity_chart"]["labels"]))

    # Zero-day banner
    st.markdown(zero_day_alert_banner(page_data["zero_day_alerts"]), unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{COLORS['border']};'>", unsafe_allow_html=True)

    # Charts
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin-bottom:0.75rem;
        ">◆ Severity Breakdown</div>
        """, unsafe_allow_html=True)
        import plotly.graph_objects as go

        sev_chart = page_data["severity_chart"]
        sev_colors = [COLORS.get(l, COLORS["text_muted"]) for l in sev_chart["labels"]]
        fig = go.Figure(data=[go.Bar(
            x=sev_chart["labels"],
            y=sev_chart["values"],
            marker=dict(
                color=sev_colors,
                line=dict(color=[c + "80" for c in sev_colors], width=1),
            ),
            text=sev_chart["values"],
            textposition="outside",
            textfont=dict(family=FONT_MONO, size=10, color=COLORS["text_primary"]),
        )])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT_MONO, color=COLORS["text_primary"]),
            xaxis=dict(
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                tickfont=dict(size=10, color=COLORS["text_secondary"]),
            ),
            yaxis=dict(
                gridcolor=COLORS["border"],
                linecolor=COLORS["border"],
                tickfont=dict(size=10, color=COLORS["text_secondary"]),
            ),
            margin=dict(l=20, r=20, t=10, b=10),
            height=250,
        )
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin-bottom:0.75rem;
        ">◆ Detection Timeline</div>
        """, unsafe_allow_html=True)
        timeline = page_data["timeline_chart"]["data"]
        if timeline:
            times = [t.get("timestamp", "") for t in timeline]
            scores = [t.get("anomaly_score", 0) for t in timeline]
            colors = [
                COLORS["zero_day"] if t.get("is_zero_day")
                else COLORS.get(t.get("severity", "low"), COLORS["text_muted"])
                for t in timeline
            ]
            fig = go.Figure(data=[go.Scatter(
                x=times,
                y=scores,
                mode="markers",
                marker=dict(
                    color=colors,
                    size=8,
                    line=dict(width=1, color=COLORS["bg_primary"]),
                ),
                text=[
                    f"Zero-Day" if t.get("is_zero_day") else t.get("severity", "N/A")
                    for t in timeline
                ],
                hovertemplate="%{x}<br>Score: %{y:.4f}<br>%{text}<extra></extra>",
            )])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family=FONT_MONO, color=COLORS["text_primary"]),
                xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
                yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
                margin=dict(l=20, r=20, t=10, b=10),
                height=250,
            )
            st.plotly_chart(fig, width="stretch")

    # Alerts table
    st.markdown(f"<hr style='border-color:{COLORS['border']};'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="
        font-family:{FONT_MONO};font-size:0.7rem;
        color:{COLORS['text_muted']};text-transform:uppercase;
        letter-spacing:0.12em;margin-bottom:0.75rem;
    ">◆ Alert Log ({page_data['total_alerts']} entries)</div>
    """, unsafe_allow_html=True)

    if page_data["alerts_table"]["rows"]:
        import pandas as pd
        df = pd.DataFrame(page_data["alerts_table"]["rows"])
        st.dataframe(
            df,
            width="stretch",
            height=min(400, 35 + len(df) * 30),
        )
    else:
        st.markdown(f"""
        <div style="
            text-align:center;padding:2rem;
            font-family:{FONT_MONO};font-size:0.85rem;
            color:{COLORS['text_muted']};
        ">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📭</div>
            No alerts recorded yet. Awaiting detection events.
        </div>
        """, unsafe_allow_html=True)


# Metrics 
elif _page == "Metrics":
    _render_page_header("Detection Metrics", "Performance analysis and detection statistics")
    page_data = render_metrics(st.session_state["metrics"])

    det = page_data["detection_metrics"]
    alrt = page_data["alert_metrics"]

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Detections", det["total_detections"])
    with c2:
        st.metric("Anomaly Rate", f"{det['anomaly_rate']:.2%}")
    with c3:
        st.metric("Avg Latency", f"{det['avg_latency_ms']:.2f} ms")
    with c4:
        st.metric("Total Alerts", alrt["total_alerts"])

    st.markdown(f"<hr style='border-color:{COLORS['border']};'>", unsafe_allow_html=True)

    # Charts
    for i, chart_data in enumerate(page_data["charts"]):
        chart_type = chart_data.get("type", "bar")
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin:1rem 0 0.5rem 0;
        ">◆ {chart_data['title']}</div>
        """, unsafe_allow_html=True)

        import plotly.graph_objects as go

        if chart_type == "histogram":
            fig = go.Figure(data=[go.Histogram(
                x=chart_data["data"],
                nbinsx=30,
                marker=dict(
                    color=COLORS["accent_green"],
                    line=dict(color=COLORS["accent_green"], width=0.5),
                ),
                opacity=0.8,
            )])
            for t_name, t_val in chart_data.get("thresholds", {}).items():
                t_color = COLORS.get(t_name, COLORS["text_muted"])
                fig.add_vline(
                    x=t_val, line_dash="dash", line_color=t_color,
                    annotation_text=t_name.upper(),
                    annotation_font=dict(family=FONT_MONO, size=9, color=t_color),
                )
            fig.update_layout(
                xaxis_title=chart_data.get("x_label", ""),
                yaxis_title=chart_data.get("y_label", ""),
            )

        elif chart_type == "pie":
            fig = go.Figure(data=[go.Pie(
                labels=chart_data["labels"],
                values=chart_data["values"],
                hole=0.5,
                marker=dict(colors=[
                    COLORS["accent_green"], COLORS["accent_cyan"],
                    COLORS["accent_blue"], COLORS["high"],
                    COLORS["critical"], COLORS["zero_day"],
                ]),
                textfont=dict(family=FONT_MONO, size=10, color=COLORS["text_primary"]),
                textinfo="label+percent",
            )])

        elif chart_type == "bar":
            bar_colors = chart_data.get("colors", [COLORS["accent_green"]] * len(chart_data["labels"]))
            fig = go.Figure(data=[go.Bar(
                x=chart_data["labels"],
                y=chart_data["values"],
                marker=dict(color=bar_colors),
                text=chart_data["values"],
                textposition="outside",
                textfont=dict(family=FONT_MONO, size=10, color=COLORS["text_primary"]),
            )])

        elif chart_type == "box":
            fig = go.Figure(data=[go.Box(
                x=chart_data["data"],
                name="Latency",
                marker=dict(color=COLORS["accent_cyan"]),
                line=dict(color=COLORS["accent_cyan"]),
                boxmean=True,
            )])
            stats = chart_data.get("stats", {})
            if stats:
                for stat_name, stat_val in stats.items():
                    fig.add_vline(
                        x=stat_val, line_dash="dot",
                        line_color=COLORS["text_muted"],
                        opacity=0.5,
                    )

        else:
            continue

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT_MONO, color=COLORS["text_primary"]),
            xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
            yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"]),
            margin=dict(l=20, r=20, t=10, b=10),
            height=280,
        )
        st.plotly_chart(fig, width="stretch")


# Models 
elif _page == "Models":
    _render_page_header("Model Registry", "Trained model status and performance comparison")

    # Placeholder model data
    default_models = [
        {
            "name": "Isolation Forest",
            "loaded": st.session_state["model_status"].get("anomaly_loaded", False),
            "version": "1.0",
            "accuracy": 0.0,
            "f1_score": 0.0,
            "detection_rate": 0.0,
            "false_alarm_rate": 0.0,
            "last_trained": "N/A",
        },
        {
            "name": "Random Forest",
            "loaded": st.session_state["model_status"].get("classifier_loaded", False),
            "version": "1.0",
            "accuracy": 0.0,
            "f1_score": 0.0,
            "detection_rate": 0.0,
            "false_alarm_rate": 0.0,
            "last_trained": "N/A",
        },
    ]

    page_data = render_models(
        default_models,
        [],
        {},
    )

    # Model cards
    for mc in page_data["model_cards"]:
        loaded = mc["loaded"]
        status_color = COLORS["healthy"] if loaded else COLORS["critical"]
        status_text = "ONLINE" if loaded else "OFFLINE"

        st.markdown(f"""
        <div style="
            background:{COLORS['bg_card']};
            border:1px solid {COLORS['border']};
            border-left:3px solid {status_color};
            border-radius:8px;padding:1rem 1.25rem;
            margin-bottom:0.75rem;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="
                        font-family:{FONT_MONO};font-size:0.9rem;font-weight:600;
                        color:{COLORS['text_primary']};
                    ">{mc['title']}</div>
                    <div style="
                        font-family:{FONT_MONO};font-size:0.65rem;
                        color:{COLORS['text_muted']};margin-top:2px;
                    ">Version {mc['version']} — Last trained: {mc['last_trained']}</div>
                </div>
                <div style="
                    font-family:{FONT_MONO};font-size:0.7rem;font-weight:600;
                    color:{status_color};
                    display:flex;align-items:center;gap:6px;
                ">
                    <div style="
                        width:8px;height:8px;border-radius:50%;
                        background:{status_color};
                        box-shadow:0 0 8px {status_color};
                    "></div>
                    {status_text}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{COLORS['border']};'>", unsafe_allow_html=True)

    # Comparison table
    if page_data["comparison_table"]["rows"]:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin-bottom:0.75rem;
        ">◆ Model Comparison</div>
        """, unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame(page_data["comparison_table"]["rows"])
        st.dataframe(df, width="stretch")

    # Feature importance tables
    if page_data["importance_tables"]:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin:1rem 0 0.5rem 0;
        ">◆ Feature Importance</div>
        """, unsafe_allow_html=True)
        for model_name, table_data in page_data["importance_tables"].items():
            with st.expander(f"📈 {model_name} — Top Features"):
                import pandas as pd
                df = pd.DataFrame(table_data["rows"])
                st.dataframe(df, width="stretch")

    # Confusion matrices
    if page_data["confusion_tables"]:
        st.markdown(f"""
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.12em;margin:1rem 0 0.5rem 0;
        ">◆ Confusion Matrices</div>
        """, unsafe_allow_html=True)
        for model_name, table_data in page_data["confusion_tables"].items():
            with st.expander(f"🎯 {model_name} — Confusion Matrix"):
                import pandas as pd
                df = pd.DataFrame(table_data["rows"])
                st.dataframe(df, width="stretch")


# Footer 
st.markdown(f"""
<div style="
    margin-top:2rem;padding:1rem 0;
    border-top:1px solid {COLORS['border']};
    text-align:center;
">
    <span style="
        font-family:{FONT_MONO};font-size:0.6rem;
        color:{COLORS['text_muted']};letter-spacing:0.1em;
    ">
        ZION-TECH v1.0 — Real-Time Autonomous Anomaly Detection for Zero-Day DoS Exploits<br/>
        Oyelude Zion Clifford (CYS/20/4940) — Supervisor: Prof. (Mrs.) Alowolodu<br/>
        Federal University of Technology, Akure — 2026
    </span>
</div>
""", unsafe_allow_html=True)
