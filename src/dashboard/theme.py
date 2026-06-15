"""Cybersecurity dashboard theme — dark hacker aesthetic.

Provides CSS constants, color palette, and Streamlit inject helpers
for the zero-day detection engine dashboard.
"""
from __future__ import annotations

from typing import Dict, List

# Color Palette 
COLORS: Dict[str, str] = {
    "bg_primary": "#0a0e17",
    "bg_secondary": "#111827",
    "bg_card": "#1a1f2e",
    "bg_card_hover": "#222940",
    "bg_input": "#0d1321",
    "border": "#1e293b",
    "border_glow": "#00ff9d40",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted": "#475569",
    "accent_green": "#00ff9d",
    "accent_cyan": "#00e5ff",
    "accent_blue": "#3b82f6",
    "critical": "#ff3b3b",
    "critical_glow": "#ff3b3b60",
    "high": "#ff8c00",
    "high_glow": "#ff8c0060",
    "medium": "#ffd600",
    "medium_glow": "#ffd60060",
    "low": "#00ff9d",
    "low_glow": "#00ff9d60",
    "zero_day": "#ff00ff",
    "zero_day_glow": "#ff00ff60",
    "healthy": "#00ff9d",
    "warning": "#ffd600",
    "error": "#ff3b3b",
}

FONT_MONO = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace"
FONT_SANS = "'Inter', 'Segoe UI', system-ui, sans-serif"


# Global CSS 
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/*  Base Reset  */
.stApp {{
    background: {COLORS['bg_primary']} !important;
    color: {COLORS['text_primary']} !important;
    font-family: {FONT_SANS} !important;
}}

/*  Header Bar  */
header[data-testid="stHeader"] {{
    background: {COLORS['bg_secondary']} !important;
    border-bottom: 1px solid {COLORS['border']} !important;
}}

/*  Sidebar  */
section[data-testid="stSidebar"] {{
    background: {COLORS['bg_secondary']} !important;
    border-right: 1px solid {COLORS['border']} !important;
}}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] label {{
    color: {COLORS['text_secondary']} !important;
}}

/*  Radio / Select Boxes  */
.stRadio > div {{
    gap: 0.5rem !important;
}}
.stRadio > div > label {{
    background: {COLORS['bg_card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    color: {COLORS['text_secondary']} !important;
    transition: all 0.2s ease !important;
}}
.stRadio > div > label:hover {{
    border-color: {COLORS['accent_green']} !important;
    color: {COLORS['accent_green']} !important;
}}
.stRadio > div > label[data-checked="true"] {{
    background: {COLORS['bg_card_hover']} !important;
    border-color: {COLORS['accent_green']} !important;
    color: {COLORS['accent_green']} !important;
    box-shadow: 0 0 12px {COLORS['border_glow']} !important;
}}

/*  Metric Cards  */
div[data-testid="stMetric"] {{
    background: {COLORS['bg_card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
    transition: all 0.25s ease !important;
    position: relative;
    overflow: hidden;
}}
div[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, {COLORS['accent_green']}, transparent);
    opacity: 0.6;
}}
div[data-testid="stMetric"]:hover {{
    border-color: {COLORS['accent_green']} !important;
    box-shadow: 0 0 20px {COLORS['border_glow']} !important;
    transform: translateY(-1px);
}}
div[data-testid="stMetric"] label {{
    color: {COLORS['text_muted']} !important;
    font-family: {FONT_MONO} !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {COLORS['accent_green']} !important;
    font-family: {FONT_MONO} !important;
    font-weight: 600 !important;
}}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
    font-family: {FONT_MONO} !important;
}}

/*  DataFrames  */
div[data-testid="stDataFrame"] {{
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px !important;
    overflow: hidden;
}}
div[data-testid="stDataFrame"] table {{
    background: {COLORS['bg_card']} !important;
    color: {COLORS['text_primary']} !important;
    font-family: {FONT_MONO} !important;
    font-size: 0.8rem !important;
}}
div[data-testid="stDataFrame"] th {{
    background: {COLORS['bg_secondary']} !important;
    color: {COLORS['accent_cyan']} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    border-bottom: 1px solid {COLORS['accent_green']}30 !important;
}}
div[data-testid="stDataFrame"] td {{
    border-bottom: 1px solid {COLORS['border']} !important;
}}
div[data-testid="stDataFrame"] tr:hover td {{
    background: {COLORS['bg_card_hover']} !important;
}}

/*  Tabs  */
.stTabs [data-baseweb="tab-list"] {{
    background: {COLORS['bg_secondary']} !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid {COLORS['border']} !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {COLORS['text_muted']} !important;
    font-family: {FONT_MONO} !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.25rem !important;
    border: 1px solid transparent !important;
    transition: all 0.2s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {COLORS['accent_green']} !important;
    border-color: {COLORS['border']} !important;
}}
.stTabs [aria-selected="true"] {{
    background: {COLORS['bg_card']} !important;
    color: {COLORS['accent_green']} !important;
    border-color: {COLORS['accent_green']} !important;
    box-shadow: 0 0 10px {COLORS['border_glow']} !important;
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
}}
.stTabs [data-baseweb="tab-border"] {{
    display: none !important;
}}

/*  Plotly Charts  */
.js-plotly-plot .plotly {{
    background: {COLORS['bg_card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px !important;
}}
.js-plotly-plot .plotly .modebar {{
    background: transparent !important;
}}
.js-plotly-plot .plotly .modebar-btn path {{
    fill: {COLORS['text_muted']} !important;
}}
.js-plotly-plot .plotly .modebar-btn:hover path {{
    fill: {COLORS['accent_green']} !important;
}}

/*  Dividers  */
hr {{
    border: none !important;
    border-top: 1px solid {COLORS['border']} !important;
    margin: 1rem 0 !important;
}}

/*  Scrollbar  */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{
    background: {COLORS['bg_primary']};
}}
::-webkit-scrollbar-thumb {{
    background: {COLORS['border']};
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {COLORS['accent_green']}60;
}}

/*  Expander  */
details {{
    background: {COLORS['bg_card']} !important;
    border: 1px solid {COLORS['border']} !important;
    border-radius: 8px !important;
}}
details summary {{
    color: {COLORS['text_secondary']} !important;
    font-family: {FONT_MONO} !important;
}}

/*  Selection Highlight  */
::selection {{
    background: {COLORS['accent_green']}30 !important;
    color: {COLORS['accent_green']} !important;
}}

/* ══════════════════════════════════════════════════════════════════════════════
   ANIMATIONS — entrance, glow, scanning, transitions
   ══════════════════════════════════════════════════════════════════════════════ */

/*  Keyframes  */
@keyframes fadeInUp {{
    0%   {{ opacity:0; transform:translateY(18px); }}
    100% {{ opacity:1; transform:translateY(0); }}
}}
@keyframes fadeInScale {{
    0%   {{ opacity:0; transform:scale(0.96); }}
    100% {{ opacity:1; transform:scale(1); }}
}}
@keyframes glowPulse {{
    0%,100% {{ box-shadow:0 0 15px {COLORS['accent_green']}15; }}
    50%     {{ box-shadow:0 0 30px {COLORS['accent_green']}30; }}
}}
@keyframes scanLine {{
    0%   {{ transform:translateX(-100%); }}
    100% {{ transform:translateX(100%); }}
}}
@keyframes borderGlow {{
    0%,100% {{ border-color:{COLORS['border']}; }}
    50%     {{ border-color:{COLORS['accent_green']}50; }}
}}
@keyframes slideInLeft {{
    0%   {{ opacity:0; transform:translateX(-20px); }}
    100% {{ opacity:1; transform:translateX(0); }}
}}
@keyframes shimmer {{
    0%   {{ background-position:-200% 0; }}
    100% {{ background-position:200% 0; }}
}}

/*  Metric Card Entrance + Glow  */
div[data-testid="stMetric"] {{
    animation: fadeInUp 0.5s ease-out both;
}}
div[data-testid="stMetric"]:nth-child(1) {{ animation-delay:0.05s; }}
div[data-testid="stMetric"]:nth-child(2) {{ animation-delay:0.12s; }}
div[data-testid="stMetric"]:nth-child(3) {{ animation-delay:0.19s; }}
div[data-testid="stMetric"]:nth-child(4) {{ animation-delay:0.26s; }}
div[data-testid="stMetric"]:nth-child(5) {{ animation-delay:0.33s; }}

/*  DataFrame Entrance  */
div[data-testid="stDataFrame"] {{
    animation: fadeInScale 0.45s ease-out both;
}}

/*  Plotly Chart Entrance  */
.js-plotly-plot {{
    animation: fadeInUp 0.5s ease-out both;
}}

/*  Tab Entrance  */
.stTabs [data-baseweb="tab"] {{
    animation: fadeInUp 0.35s ease-out both;
}}
.stTabs [data-baseweb="tab"]:nth-child(1) {{ animation-delay:0.05s; }}
.stTabs [data-baseweb="tab"]:nth-child(2) {{ animation-delay:0.1s; }}
.stTabs [data-baseweb="tab"]:nth-child(3) {{ animation-delay:0.15s; }}

/*  Expanders Slide-In  */
details {{
    animation: slideInLeft 0.4s ease-out both;
}}

/*  Columns stagger entrance  */
div[data-testid="stHorizontalBlock"] > div {{
    animation: fadeInUp 0.45s ease-out both;
}}
div[data-testid="stHorizontalBlock"] > div:nth-child(1) {{ animation-delay:0.05s; }}
div[data-testid="stHorizontalBlock"] > div:nth-child(2) {{ animation-delay:0.12s; }}
div[data-testid="stHorizontalBlock"] > div:nth-child(3) {{ animation-delay:0.19s; }}
div[data-testid="stHorizontalBlock"] > div:nth-child(4) {{ animation-delay:0.26s; }}
div[data-testid="stHorizontalBlock"] > div:nth-child(5) {{ animation-delay:0.33s; }}

/*  Scanning line on header  */
header[data-testid="stHeader"] {{
    overflow:hidden !important;
    position:relative !important;
}}
header[data-testid="stHeader"]::after {{
    content:'';
    position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent 0%,{COLORS['accent_green']} 50%,transparent 100%);
    animation:scanLine 6s linear infinite;
}}

/*  Sidebar items slide-in  */
section[data-testid="stSidebar"] div[data-testid="stMarkdown"] {{
    animation: slideInLeft 0.3s ease-out both;
}}

/*  Smooth transitions for everything interactive  */
.stButton > button,
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    transition:all 0.25s cubic-bezier(.4,0,.2,1) !important;
}}
.stButton > button:hover {{
    transform:translateY(-1px);
    box-shadow:0 4px 20px {COLORS['accent_green']}25;
}}

/*  Header shimmer text accent  */
div[data-testid="stHeader"] h1,
div[data-testid="stHeader"] h2 {{
    background:linear-gradient(90deg,{COLORS['accent_green']},{COLORS['accent_cyan']},{COLORS['accent_green']});
    background-size:200% 100%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:shimmer 4s linear infinite;
}}
</style>
"""


# Reusable HTML Fragments 
def threat_level_banner(level: str, score: float) -> str:
    """Render a glowing threat-level banner."""
    color = COLORS.get(level, COLORS["medium"])
    return f"""
    <div style="
        background: {COLORS['bg_card']};
        border: 1px solid {color}40;
        border-left: 4px solid {color};
        border-radius: 0;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 0 20px {color}15;
        animation: fadeInUp 0.5s ease-out both, glowPulse 3s ease-in-out infinite;
    ">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="
                width:12px;height:12px;border-radius:50%;
                background:{color};
                box-shadow:0 0 10px {color};
                animation: pulse 2s infinite;
            "></div>
            <span style="
                font-family:{FONT_MONO};font-size:0.7rem;
                color:{COLORS['text_muted']};text-transform:uppercase;
                letter-spacing:0.15em;
            ">THREAT LEVEL</span>
        </div>
        <div style="
            font-family:{FONT_MONO};font-size:1.5rem;font-weight:700;
            color:{color};margin-top:0.5rem;
            text-shadow:0 0 10px {color}60;
        ">{level.upper()}</div>
        <div style="
            font-family:{FONT_MONO};font-size:0.8rem;
            color:{COLORS['text_muted']};margin-top:0.25rem;
        ">Anomaly Score: {score:.4f}</div>
    </div>
    <style>
    @keyframes pulse {{
        0%,100% {{ opacity:1; }}
        50% {{ opacity:0.4; }}
    }}
    @keyframes fadeInUp {{
        0%   {{ opacity:0; transform:translateY(18px); }}
        100% {{ opacity:1; transform:translateY(0); }}
    }}
    @keyframes glowPulse {{
        0%,100% {{ box-shadow:0 0 15px {COLORS['accent_green']}15; }}
        50%     {{ box-shadow:0 0 30px {COLORS['accent_green']}30; }}
    }}
    </style>
    """


def system_health_bar(health: str, detail: str = "") -> str:
    """Render a horizontal system health indicator."""
    color = COLORS.get(health, COLORS["text_muted"])
    return f"""
    <div style="
        background:{COLORS['bg_card']};
        border:1px solid {COLORS['border']};
        border-radius:8px;padding:0.75rem 1rem;
        margin-bottom:0.75rem;
    ">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="
                    width:8px;height:8px;border-radius:50%;
                    background:{color};
                    box-shadow:0 0 8px {color};
                "></div>
                <span style="
                    font-family:{FONT_MONO};font-size:0.75rem;
                    color:{COLORS['text_secondary']};text-transform:uppercase;
                    letter-spacing:0.1em;
                ">SYSTEM STATUS</span>
            </div>
            <span style="
                font-family:{FONT_MONO};font-size:0.75rem;font-weight:600;
                color:{color};
            ">{health.upper()}</span>
        </div>
        {"<div style='font-family:" + FONT_MONO + ";font-size:0.7rem;color:" + COLORS['text_muted'] + ";margin-top:0.4rem;'>" + detail + "</div>" if detail else ""}
    </div>
    """


def terminal_block(lines: List[str], max_lines: int = 20) -> str:
    """Render a terminal-style log block."""
    display = lines[-max_lines:]
    content = "\n".join(
        f'<span style="color:{COLORS["text_muted"]}">{i+1:04d}</span> '
        f'<span style="color:{COLORS["accent_green"]}">$</span> '
        f'<span style="color:{COLORS["text_secondary"]}">{line}</span>'
        for i, line in enumerate(display)
    )
    return f"""
    <div style="
        background:{COLORS['bg_input']};
        border:1px solid {COLORS['border']};
        border-radius:8px;padding:1rem;
        font-family:{FONT_MONO};font-size:0.75rem;
        line-height:1.6;overflow-x:auto;
        max-height:400px;overflow-y:auto;
    ">
        <div style="
            display:flex;align-items:center;gap:6px;
            margin-bottom:0.75rem;padding-bottom:0.5rem;
            border-bottom:1px solid {COLORS['border']};
        ">
            <div style="width:10px;height:10px;border-radius:50%;background:#ff5f57;"></div>
            <div style="width:10px;height:10px;border-radius:50%;background:#febc2e;"></div>
            <div style="width:10px;height:10px;border-radius:50%;background:#28c840;"></div>
            <span style="
                margin-left:8px;font-size:0.65rem;
                color:{COLORS['text_muted']};text-transform:uppercase;
                letter-spacing:0.1em;
            ">detection-engine — live feed</span>
        </div>
        {content}
    </div>
    """


def stat_row(label: str, value: str, color: str = "accent_green") -> str:
    """Render a single key-value stat row."""
    c = COLORS.get(color, COLORS["accent_green"])
    return f"""
    <div style="
        display:flex;justify-content:space-between;
        padding:0.4rem 0;
        border-bottom:1px solid {COLORS['border']}20;
    ">
        <span style="
            font-family:{FONT_MONO};font-size:0.75rem;
            color:{COLORS['text_muted']};text-transform:uppercase;
            letter-spacing:0.05em;
        ">{label}</span>
        <span style="
            font-family:{FONT_MONO};font-size:0.75rem;font-weight:600;
            color:{c};
        ">{value}</span>
    </div>
    """


def severity_badge(severity: str) -> str:
    """Render a colored severity badge."""
    color = COLORS.get(severity, COLORS["text_muted"])
    glow = COLORS.get(f"{severity}_glow", "transparent")
    return f"""
    <span style="
        display:inline-block;
        background:{color}20;
        color:{color};
        border:1px solid {color}40;
        border-radius:4px;
        padding:2px 10px;
        font-family:{FONT_MONO};
        font-size:0.7rem;
        font-weight:600;
        text-transform:uppercase;
        letter-spacing:0.08em;
        box-shadow:0 0 8px {glow};
    ">{severity}</span>
    """


def zero_day_alert_banner(count: int) -> str:
    """Render a pulsing zero-day alert banner."""
    if count == 0:
        return ""
    return f"""
    <div style="
        background:linear-gradient(135deg, {COLORS['zero_day']}15, {COLORS['critical']}10);
        border:1px solid {COLORS['zero_day']}40;
        border-radius:0;padding:1rem 1.5rem;
        margin-bottom:1rem;
        animation: fadeInUp 0.5s ease-out both, zeroday-pulse 3s infinite;
    ">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.2rem;">⚠</span>
            <span style="
                font-family:{FONT_MONO};font-size:0.85rem;font-weight:700;
                color:{COLORS['zero_day']};
                text-shadow:0 0 10px {COLORS['zero_day']}60;
            ">ZERO-DAY THREATS DETECTED: {count}</span>
        </div>
        <div style="
            font-family:{FONT_MONO};font-size:0.7rem;
            color:{COLORS['text_muted']};margin-top:0.3rem;
        ">Unidentified attack patterns — immediate analysis required</div>
    </div>
    <style>
    @keyframes zeroday-pulse {{
        0%,100% {{ border-color: {COLORS['zero_day']}40; }}
        50% {{ border-color: {COLORS['zero_day']}80; box-shadow: 0 0 25px {COLORS['zero_day']}20; }}
    }}
    </style>
    """


def progress_bar(value: float, max_val: float = 1.0, color: str = "accent_green", label: str = "") -> str:
    """Render a thin animated progress bar."""
    pct = min(100, (value / max_val) * 100) if max_val > 0 else 0
    c = COLORS.get(color, COLORS["accent_green"])
    return f"""
    <div style="margin:0.3rem 0;">
        {"<span style='font-family:" + FONT_MONO + ";font-size:0.65rem;color:" + COLORS['text_muted'] + ";text-transform:uppercase;letter-spacing:0.08em;'>" + label + "</span>" if label else ""}
        <div style="
            background:{COLORS['bg_input']};
            border-radius:4px;height:6px;
            overflow:hidden;margin-top:4px;
        ">
            <div style="
                width:{pct}%;height:100%;
                background:linear-gradient(90deg, {c}80, {c});
                border-radius:4px;
                box-shadow:0 0 8px {c}40;
                transition:width 0.5s ease;
            "></div>
        </div>
    </div>
    """


def hex_decoration() -> str:
    """Render a subtle hex-grid decorative overlay."""
    return """
    <div style="
        position:fixed;top:0;left:0;right:0;bottom:0;
        pointer-events:none;z-index:0;
        opacity:0.03;
        background-image:
            linear-gradient(30deg, #00ff9d 12%, transparent 12.5%, transparent 87%, #00ff9d 87.5%, #00ff9d),
            linear-gradient(150deg, #00ff9d 12%, transparent 12.5%, transparent 87%, #00ff9d 87.5%, #00ff9d),
            linear-gradient(30deg, #00ff9d 12%, transparent 12.5%, transparent 87%, #00ff9d 87.5%, #00ff9d),
            linear-gradient(150deg, #00ff9d 12%, transparent 12.5%, transparent 87%, #00ff9d 87.5%, #00ff9d),
            linear-gradient(60deg, #00ff9d77 25%, transparent 25.5%, transparent 75%, #00ff9d77 75%, #00ff9d77),
            linear-gradient(60deg, #00ff9d77 25%, transparent 25.5%, transparent 75%, #00ff9d77 75%, #00ff9d77);
        background-size:80px 140px;
        background-position:0 0, 0 0, 40px 70px, 40px 70px, 0 0, 40px 70px;
    "></div>
    """
