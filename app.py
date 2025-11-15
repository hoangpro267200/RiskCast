# =============================================================================
# RISKCAST v5.5 — ENTERPRISE ULTRA TOOLTIP EDITION
# ESG Logistics Risk Assessment Dashboard
#
# Author: Bùi Xuân Hoàng (original idea)
# Refactor + Multi-Package + Full Explanations + Enterprise UX: Kai assistant
#
# Theme: Premium Green · Mixed Enterprise (Salesforce + Oracle Fusion)
# =============================================================================

import io
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

warnings.filterwarnings("ignore")

# Optional dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


# =============================================================================
# PAGE CONFIG + GLOBAL CSS
# =============================================================================

def app_config():
    st.set_page_config(
        page_title="RISKCAST v5.5 — Enterprise Tooltip Edition",
        page_icon="🛡️",
        layout="wide"
    )
def apply_enterprise_css():
    """RISKCAST Ultra Luxury v3 – Max Glow (Black + Neon Green)."""
    st.markdown(
        """
        <style>
        /* =========================================================
           GLOBAL RESET + TYPO
        ========================================================= */
        * {
            text-rendering: optimizeLegibility !important;
            -webkit-font-smoothing: antialiased !important;
            scroll-behavior: smooth;
        }

        html, body {
            background: #020608 !important;
        }

        .stApp {
            background: radial-gradient(circle at top,
                        #041619 0%,
                        #020608 40%,
                        #000000 100%) !important;
            color: #e9fff4 !important;
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif !important;
            font-size: 16px !important;
        }

        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 3.4rem !important;
            max-width: 1500px !important;
        }

        h1, h2, h3, h4, h5 {
            font-weight: 800 !important;
            letter-spacing: 0.02em;
            color: #eafff8 !important;
        }

        /* =========================================================
           LUXURY GLOW UTILITIES
        ========================================================= */
        .neon-text {
            background: linear-gradient(90deg, #eafff8, #9dffd0, #e0fff7);
            -webkit-background-clip: text;
            color: transparent;
        }

        .neon-soft {
            text-shadow: 0 0 14px rgba(0,255,153,0.45);
        }

        .neon-hard {
            text-shadow:
                0 0 4px rgba(0,255,153,0.9),
                0 0 10px rgba(0,255,153,0.8),
                0 0 25px rgba(0,255,153,0.7);
        }

        .glow-ring {
            box-shadow:
                0 0 0 1px rgba(0,255,153,0.25),
                0 0 20px rgba(0,255,153,0.25),
                0 0 40px rgba(0,255,153,0.30);
        }

        .glass {
            background: linear-gradient(135deg,
                rgba(6, 24, 20, 0.94),
                rgba(0, 0, 0, 0.98)) !important;
            border-radius: 18px;
            border: 1px solid rgba(0,255,153,0.22);
            box-shadow:
                0 16px 40px rgba(0,0,0,0.85),
                0 0 30px rgba(0,255,153,0.22);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }

        /* =========================================================
           HEADER – ULTRA LUXURY
        ========================================================= */
        .rc-header {
            position: relative;
            padding: 1.3rem 1.7rem;
            margin-bottom: 2.1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1.6rem;
            overflow: hidden;
        }

        .rc-header::before {
            content: "";
            position: absolute;
            inset: -1px;
            border-radius: 22px;
            background: radial-gradient(circle at 10% 0%,
                        rgba(0,255,153,0.16) 0%,
                        transparent 45%),
                        radial-gradient(circle at 90% 100%,
                        rgba(0,255,204,0.18) 0%,
                        transparent 55%);
            opacity: 0.85;
            pointer-events: none;
        }

        .rc-header::after {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 22px;
            border: 1px solid rgba(0,255,153,0.45);
            box-shadow:
                0 0 0 1px rgba(0,255,153,0.25),
                0 0 18px rgba(0,255,153,0.55),
                0 18px 45px rgba(0,0,0,0.95);
            pointer-events: none;
        }

        .rc-header-left {
            position: relative;
            z-index: 2;
            display: flex;
            align-items: center;
            gap: 1.3rem;
        }

        .rc-logo {
            width: 82px;
            height: 82px;
            border-radius: 22px;
            background:
                radial-gradient(circle at 30% 20%,
                    #ffffff 0%, #d7fff4 14%, #9affdf 34%, transparent 60%),
                radial-gradient(circle at 70% 80%,
                    #00ffcc 0%, #00e6aa 35%, #00664a 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 1.8rem;
            color: #00140d;
            border: 3px solid #c4ffea;
            box-shadow:
                0 0 12px rgba(0,255,153,0.75),
                0 0 38px rgba(0,255,153,0.85);
            text-shadow:
                0 0 4px rgba(0,0,0,0.55),
                0 0 16px rgba(0,255,153,0.95);
        }

        .rc-title {
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            background: linear-gradient(90deg,
                        #f4fff9,
                        #c2ffe0,
                        #e2fff7);
            -webkit-background-clip: text;
            color: transparent;
        }

        .rc-subtitle {
            margin-top: 4px;
            font-size: 0.95rem;
            color: #c4ffea;
            opacity: 0.95;
            font-weight: 500;
        }

        .rc-badge {
            position: relative;
            z-index: 2;
            background: linear-gradient(135deg, #00ff99, #00e676, #00bfa5);
            padding: 0.7rem 1.5rem;
            border-radius: 999px;
            color: #00140c;
            font-weight: 800;
            font-size: 0.92rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            box-shadow:
                0 0 18px rgba(0,255,153,0.7),
                0 14px 36px rgba(0,0,0,0.9);
            white-space: nowrap;
        }

        /* =========================================================
           SIDEBAR – LUXURY PANEL
        ========================================================= */
        section[data-testid="stSidebar"] {
            background: radial-gradient(circle at top,
                        #031914 0%,
                        #020806 45%,
                        #000000 100%) !important;
            border-right: 1px solid rgba(0,255,153,0.26);
            box-shadow: 10px 0 30px rgba(0,0,0,0.85);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 0.5rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #c8ffec !important;
            font-weight: 800 !important;
        }

        section[data-testid="stSidebar"] label {
            color: #e0fff4 !important;
            font-weight: 600 !important;
        }

        section[data-testid="stSidebar"] .stSlider label {
            font-size: 0.9rem !important;
        }

        /* Sidebar inputs */
        section[data-testid="stSidebar"] 
        .stTextInput > div > div > input,
        section[data-testid="stSidebar"]
        .stNumberInput input,
        section[data-testid="stSidebar"]
        .stSelectbox > div > div {
            background: rgba(0,0,0,0.6) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(0,255,153,0.4) !important;
            color: #e9fff4 !important;
            box-shadow: 0 0 0 1px rgba(0,255,153,0.24);
        }

        /* Sidebar button */
        section[data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(135deg, #00ff99, #00e676, #00c853) !important;
            color: #00140d !important;
            font-weight: 900 !important;
            border-radius: 999px !important;
            border: none !important;
            padding: 0.7rem 1.8rem !important;
            box-shadow:
                0 0 18px rgba(0,255,153,0.65),
                0 12px 30px rgba(0,0,0,0.9) !important;
            transition: all 0.15s ease-out !important;
            font-size: 0.98rem !important;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-1px) scale(1.02);
            box-shadow:
                0 0 26px rgba(0,255,153,0.9),
                0 16px 40px rgba(0,0,0,0.95) !important;
        }

        /* =========================================================
           CORE CARDS / SECTIONS
        ========================================================= */
        .rc-card {
            margin-bottom: 1.5rem;
        }

        .rc-card,
        .result-box,
        .explanation-box,
        .top3-card,
        .rc-risk-card {
            position: relative;
        }

        .rc-card::before,
        .result-box::before,
        .explanation-box::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 18px;
            border: 1px solid rgba(0,255,153,0.25);
            box-shadow:
                0 0 14px rgba(0,255,153,0.25),
                0 12px 26px rgba(0,0,0,0.9);
            opacity: 0.9;
            pointer-events: none;
        }

        .rc-card > *:not(.rc-card::before),
        .result-box > *:not(.result-box::before),
        .explanation-box > *:not(.explanation-box::before) {
            position: relative;
            z-index: 2;
        }

        .rc-card {
            padding: 1.25rem 1.5rem;
            background: radial-gradient(circle at top left,
                        rgba(0,255,153,0.16),
                        transparent 55%),
                        radial-gradient(circle at bottom right,
                        rgba(0,128,96,0.35),
                        rgba(0,0,0,0.98));
            border-radius: 18px;
        }

        .result-box {
            padding: 1.7rem 2.1rem;
            border-radius: 20px;
            background:
                radial-gradient(circle at top left,
                    rgba(0,255,153,0.35),
                    rgba(0,0,0,0.98)),
                radial-gradient(circle at bottom right,
                    rgba(0,255,204,0.22),
                    rgba(0,0,0,1));
            color: #00130d !important;
        }

        .result-box b {
            color: #00130d !important;
        }

        .explanation-box {
            margin-top: 0.8rem;
            padding: 1.2rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg,
                        rgba(1, 20, 14, 0.98),
                        rgba(0, 0, 0, 0.98));
            border-left: 4px solid #00e676;
        }

        .explanation-box h4 {
            margin-bottom: 0.5rem;
            color: #a5ffdc !important;
        }

        .explanation-box p,
        .explanation-box li {
            color: #e0f2f1 !important;
            font-weight: 500;
        }

        /* =========================================================
           TOOLTIP (TEXT + ICON)
        ========================================================= */
        .rc-tooltip {
            text-decoration: underline dotted #00e676;
            cursor: pointer;
            position: relative;
            font-weight: 600;
            color: #aaffdd;
        }

        .rc-tooltip:hover::after {
            content: attr(data-tip);
            position: absolute;
            left: 0;
            bottom: -2.9rem;
            background: rgba(0, 5, 4, 0.98);
            padding: 9px 12px;
            border-radius: 10px;
            border: 1px solid rgba(0,255,153,0.55);
            font-size: 0.84rem;
            color: #d8fff0;
            width: max-content;
            max-width: 320px;
            z-index: 999;
            box-shadow:
                0 0 14px rgba(0,255,153,0.45),
                0 18px 30px rgba(0,0,0,0.9);
        }

        .tooltip-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-left: 6px;
            background: radial-gradient(circle,
                        rgba(0,255,153,0.18),
                        rgba(0, 36, 26, 0.95));
            color: #a5ffdc;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 12px;
            cursor: help;
            border: 1px solid rgba(0,255,153,0.6);
            box-shadow:
                0 0 8px rgba(0,255,153,0.35),
                0 0 16px rgba(0,255,153,0.4);
            position: relative;
            font-weight: 700;
        }

        .tooltip-icon:hover {
            background: radial-gradient(circle,
                        rgba(0,255,160,0.35),
                        rgba(0, 40, 28, 0.95));
        }

        .tooltip-icon:hover::after {
            content: attr(data-tip);
            position: absolute;
            background: rgba(0, 8, 6, 0.98);
            border: 1px solid rgba(0,255,153,0.6);
            padding: 10px 14px;
            border-radius: 10px;
            color: #e0fff5;
            width: 260px;
            left: 22px;
            bottom: -4px;
            font-size: 0.82rem;
            line-height: 1.35rem;
            z-index: 999;
            box-shadow:
                0 0 20px rgba(0,255,153,0.6),
                0 18px 36px rgba(0,0,0,0.95);
        }

        /* =========================================================
           TOP 3 CARDS – MEDAL GLOW
        ========================================================= */
        .top3-card {
            position: relative;
            padding: 1.1rem 1.1rem 1.0rem 1.1rem;
            border-radius: 18px;
            background: radial-gradient(circle at top left,
                        rgba(0,255,153,0.16),
                        rgba(0,0,0,0.96));
            border: 1px solid rgba(0,255,153,0.45);
            box-shadow:
                0 0 18px rgba(0,255,153,0.18),
                0 16px 40px rgba(0,0,0,0.9);
            text-align: center;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            transition: transform 0.18s ease-out,
                        box-shadow 0.18s ease-out,
                        border-color 0.18s ease-out;
            margin-bottom: 1.2rem;
        }

        .top1-card {
            background: radial-gradient(circle at top left,
                        rgba(255,215,0,0.22),
                        rgba(0,0,0,0.96));
            border-color: rgba(255,215,0,0.8);
            box-shadow:
                0 0 24px rgba(255,215,0,0.65),
                0 20px 46px rgba(0,0,0,0.95);
            animation: gold-pulse 2.6s ease-in-out infinite alternate;
        }

        @keyframes gold-pulse {
            0% {
                box-shadow:
                    0 0 10px rgba(255,215,0,0.35),
                    0 18px 40px rgba(0,0,0,0.9);
            }
            100% {
                box-shadow:
                    0 0 28px rgba(255,215,0,0.9),
                    0 24px 54px rgba(0,0,0,0.98);
            }
        }

        .top3-card:hover {
            transform: translateY(-4px) scale(1.02);
            border-color: rgba(0,255,200,0.9);
            box-shadow:
                0 0 24px rgba(0,255,180,0.9),
                0 22px 52px rgba(0,0,0,0.98);
        }

        .top3-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #a5ffdc;
        }

        .top1-title {
            font-size: 1.12rem;
            font-weight: 900;
            color: #ffe680;
            text-shadow:
                0 0 10px rgba(255,210,0,0.7),
                0 0 20px rgba(255,210,0,0.9);
        }

        .top3-sub {
            font-size: 0.92rem;
            margin-top: 5px;
            color: #e0f2f1;
        }

        .badge-icc {
            display: inline-block;
            padding: 4px 11px;
            border-radius: 999px;
            background: linear-gradient(120deg, #00e676, #00bfa5);
            color: #00130d;
            font-weight: 700;
            font-size: 0.86rem;
        }

        .pill-badge {
            display: inline-block;
            padding: 3px 11px;
            border-radius: 999px;
            border: 1px solid rgba(0,255,153,0.65);
            font-size: 0.82rem;
            margin-top: 4px;
            color: #c8ffec;
        }

        /* =========================================================
           DATAFRAME / TABLE
        ========================================================= */
        div[data-testid="stDataFrame"] {
            border-radius: 14px !important;
            border: 1px solid rgba(0,255,153,0.32) !important;
            overflow: hidden !important;
            box-shadow:
                0 0 18px rgba(0,255,153,0.22),
                0 18px 38px rgba(0,0,0,0.9) !important;
            background: radial-gradient(circle at top,
                        rgba(0,255,153,0.16),
                        rgba(0,0,0,0.96)) !important;
        }

        /* Metric */
        [data-testid="stMetricValue"] {
            color: #76ff03 !important;
            font-weight: 900 !important;
            font-size: 1.08rem !important;
        }

        [data-testid="stMetricLabel"] {
            color: #e0f2f1 !important;
            font-weight: 600 !important;
        }

        /* =========================================================
           CHART FRAME (FIX PHÓNG TO + NEON BORDER)
        ========================================================= */
        div[data-testid="stPlotlyChart"] {
            padding: 0.8rem 0.9rem 0.6rem 0.9rem;
            margin: 0.9rem 0 1.5rem 0;
            background: radial-gradient(circle at top,
                        rgba(0,255,153,0.12),
                        rgba(0,0,0,0.98));
            border-radius: 18px;
            border: 1px solid rgba(0,255,153,0.38);
            box-shadow:
                0 0 20px rgba(0,255,153,0.35),
                0 18px 40px rgba(0,0,0,0.95);
        }

        div[data-testid="stPlotlyChart"] > div {
            max-width: 100% !important;
        }

        /* =========================================================
           MAIN ACTION BUTTON
        ========================================================= */
        button[kind="primary"],
        .stButton > button[kind="primary"] {
            background: radial-gradient(circle at 0% 0%,
                        #ffffff 0%, #d7fff4 24%, #00ffbf 55%, #00bfa5 100%) !important;
            color: #00140d !important;
            font-weight: 900 !important;
            border-radius: 999px !important;
            border: none !important;
            padding: 0.9rem 1.9rem !important;
            box-shadow:
                0 0 20px rgba(0,255,153,0.75),
                0 20px 46px rgba(0,0,0,0.95) !important;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            transition: all 0.16s ease-out !important;
            font-size: 1.02rem !important;
        }

        button[kind="primary"]:hover,
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) scale(1.03);
            box-shadow:
                0 0 30px rgba(0,255,170,0.95),
                0 26px 60px rgba(0,0,0,0.98) !important;
        }

        /* =========================================================
           DOWNLOAD BUTTONS
        ========================================================= */
        .stDownloadButton > button {
            border-radius: 999px !important;
            border: 1px solid rgba(0,255,153,0.7) !important;
            background: radial-gradient(circle,
                        rgba(0,255,153,0.18),
                        rgba(0,0,0,0.96)) !important;
            color: #e9fff4 !important;
            font-weight: 700 !important;
            box-shadow:
                0 0 14px rgba(0,255,153,0.35),
                0 14px 30px rgba(0,0,0,0.9) !important;
        }

        .stDownloadButton > button:hover {
            background: radial-gradient(circle,
                        rgba(0,255,153,0.35),
                        rgba(0,10,8,0.98)) !important;
        }

        /* =========================================================
           MOBILE RESPONSIVE
        ========================================================= */
        @media (max-width: 900px) {
            .rc-header {
                flex-direction: column;
                align-items: flex-start;
            }
            .block-container {
                padding-left: 0.7rem !important;
                padding-right: 0.7rem !important;
            }
            .rc-badge {
                margin-top: 0.4rem;
            }
            div[data-testid="stPlotlyChart"] {
                padding: 0.6rem 0.5rem 0.6rem 0.5rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DOMAIN MODELS & CONSTANTS
# =============================================================================

class CriterionType(Enum):
    COST = "cost"
    BENEFIT = "benefit"


@dataclass
class AnalysisParams:
    cargo_value: float
    good_type: str
    route: str
    method: str
    month: int
    priority: str
    use_fuzzy: bool
    use_arima: bool
    use_mc: bool
    use_var: bool
    mc_runs: int
    fuzzy_uncertainty: float


@dataclass
class AnalysisResult:
    results: pd.DataFrame
    weights: pd.Series
    data_adjusted: pd.DataFrame
    var: Optional[float]
    cvar: Optional[float]
    historical: np.ndarray
    forecast: np.ndarray


CRITERIA = [
    "C1: Tỷ lệ phí",
    "C2: Thời gian xử lý",
    "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC",
    "C5: Chăm sóc KH",
    "C6: Rủi ro khí hậu"
]

PRIORITY_PROFILES = {
    "💰 Tiết kiệm chi phí": {
        "C1: Tỷ lệ phí": 0.35,
        "C2: Thời gian xử lý": 0.10,
        "C3: Tỷ lệ tổn thất": 0.15,
        "C4: Hỗ trợ ICC": 0.15,
        "C5: Chăm sóc KH": 0.10,
        "C6: Rủi ro khí hậu": 0.15
    },
    "⚖️ Cân bằng": {
        "C1: Tỷ lệ phí": 0.20,
        "C2: Thời gian xử lý": 0.15,
        "C3: Tỷ lệ tổn thất": 0.20,
        "C4: Hỗ trợ ICC": 0.20,
        "C5: Chăm sóc KH": 0.10,
        "C6: Rủi ro khí hậu": 0.15
    },
    "🛡️ An toàn tối đa": {
        "C1: Tỷ lệ phí": 0.10,
        "C2: Thời gian xử lý": 0.10,
        "C3: Tỷ lệ tổn thất": 0.25,
        "C4: Hỗ trợ ICC": 0.25,
        "C5: Chăm sóc KH": 0.10,
        "C6: Rủi ro khí hậu": 0.20
    }
}

ICC_PACKAGES = {
    "ICC A": {
        "coverage": 1.0,
        "premium_multiplier": 1.5,
        "description": "Bảo vệ toàn diện mọi rủi ro trừ điều khoản loại trừ (All Risks)"
    },
    "ICC B": {
        "coverage": 0.75,
        "premium_multiplier": 1.0,
        "description": "Bảo vệ các rủi ro chính (hỏa hoạn, va chạm, chìm đắm, Named Perils)"
    },
    "ICC C": {
        "coverage": 0.5,
        "premium_multiplier": 0.65,
        "description": "Bảo vệ cơ bản (chỉ các rủi ro lớn như chìm, cháy, va chạm nghiêm trọng)"
    }
}

COST_BENEFIT_MAP = {
    "C1: Tỷ lệ phí": CriterionType.COST,
    "C2: Thời gian xử lý": CriterionType.COST,
    "C3: Tỷ lệ tổn thất": CriterionType.COST,
    "C4: Hỗ trợ ICC": CriterionType.BENEFIT,
    "C5: Chăm sóc KH": CriterionType.BENEFIT,
    "C6: Rủi ro khí hậu": CriterionType.COST
}

SENSITIVITY_MAP = {
    "Chubb": 0.95,
    "PVI": 1.05,
    "BaoViet": 1.00,
    "BaoMinh": 1.02,
    "MIC": 1.03
}


# =============================================================================
# DATA LAYER
# =============================================================================

class DataService:
    """Quản lý dữ liệu đầu vào (lịch sử khí hậu, dữ liệu công ty)."""

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_historical_data() -> pd.DataFrame:
        climate_base = {
            "VN - EU": [0.28, 0.30, 0.35, 0.40, 0.52, 0.60, 0.67, 0.70, 0.75, 0.72, 0.60, 0.48],
            "VN - US": [0.33, 0.36, 0.40, 0.46, 0.55, 0.63, 0.72, 0.78, 0.80, 0.74, 0.62, 0.50],
            "VN - Singapore": [0.18, 0.20, 0.24, 0.27, 0.32, 0.36, 0.40, 0.43, 0.45, 0.42, 0.35, 0.30],
            "VN - China": [0.20, 0.23, 0.27, 0.31, 0.38, 0.42, 0.48, 0.50, 0.53, 0.49, 0.40, 0.34],
            "Domestic": [0.12, 0.13, 0.14, 0.16, 0.20, 0.22, 0.23, 0.25, 0.27, 0.24, 0.20, 0.18]
        }
        df = pd.DataFrame({"month": list(range(1, 13))})
        for route, values in climate_base.items():
            df[route] = values
        return df

    @staticmethod
    @st.cache_data
    def get_company_data() -> pd.DataFrame:
        return (
            pd.DataFrame({
                "Company": ["Chubb", "PVI", "BaoViet", "BaoMinh", "MIC"],
                "C1: Tỷ lệ phí": [0.42, 0.36, 0.40, 0.38, 0.34],
                "C2: Thời gian xử lý": [12, 10, 15, 14, 11],
                "C3: Tỷ lệ tổn thất": [0.07, 0.09, 0.11, 0.10, 0.08],
                "C4: Hỗ trợ ICC": [9, 8, 7, 8, 7],
                "C5: Chăm sóc KH": [9, 8, 7, 7, 6],
            })
            .set_index("Company")
        )


# =============================================================================
# CORE ALGORITHMS
# =============================================================================

class FuzzyAHP:
    @staticmethod
    def apply(weights: pd.Series, uncertainty_pct: float) -> pd.Series:
        factor = uncertainty_pct / 100.0
        w = weights.values
        low = np.maximum(w * (1 - factor), 1e-9)
        high = np.minimum(w * (1 + factor), 0.9999)
        defuzzified = (low + w + high) / 3.0
        normalized = defuzzified / defuzzified.sum()
        return pd.Series(normalized, index=weights.index)


class MonteCarloSimulator:
    @staticmethod
    @st.cache_data(ttl=600)
    def simulate(
        base_risk: float,
        sensitivity_map: Dict[str, float],
        n_simulations: int
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())
        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.03, mu * 0.12)
        sims = rng.normal(loc=mu, scale=sigma, size=(n_simulations, len(companies)))
        sims = np.clip(sims, 0.0, 1.0)
        return companies, sims.mean(axis=0), sims.std(axis=0)


class TOPSISAnalyzer:
    @staticmethod
    def analyze(
        data: pd.DataFrame,
        weights: pd.Series,
        cost_benefit: Dict[str, CriterionType]
    ) -> np.ndarray:
        M = data[list(weights.index)].values.astype(float)
        denom = np.sqrt((M ** 2).sum(axis=0))
        denom[denom == 0] = 1.0
        R = M / denom
        V = R * weights.values
        is_cost = np.array([cost_benefit[c] == CriterionType.COST for c in weights.index])
        ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
        ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))
        d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
        return d_minus / (d_plus + d_minus + 1e-12)


class RiskCalculator:
    @staticmethod
    def calculate_var_cvar(
        loss_rates: np.ndarray,
        cargo_value: float,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        if len(loss_rates) == 0:
            return 0.0, 0.0
        losses = loss_rates * cargo_value
        var = float(np.percentile(losses, confidence * 100))
        tail_losses = losses[losses >= var]
        cvar = float(tail_losses.mean()) if len(tail_losses) > 0 else var
        return var, cvar


class Forecaster:
    @staticmethod
    def forecast(
        historical: pd.DataFrame,
        route: str,
        current_month: int,
        use_arima: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if route not in historical.columns:
            route = historical.columns[1]

        full_series = historical[route].values
        n_total = len(full_series)

        current_month = max(1, min(current_month, n_total))
        hist_series = full_series[:current_month]
        train_series = hist_series.copy()

        if use_arima and ARIMA_AVAILABLE and len(train_series) >= 6:
            try:
                model = ARIMA(train_series, order=(1, 1, 1))
                fitted = model.fit()
                fc = fitted.forecast(1)
                fc_val = float(np.clip(fc[0], 0.0, 1.0))
                return hist_series, np.array([fc_val])
            except Exception:
                pass

        if len(train_series) >= 3:
            trend = (train_series[-1] - train_series[-3]) / 2.0
        elif len(train_series) >= 2:
            trend = train_series[-1] - train_series[-2]
        else:
            trend = 0.0

        next_val = np.clip(train_series[-1] + trend, 0.0, 1.0)
        return hist_series, np.array([next_val])


# =============================================================================
# FUZZY VISUAL UTILITIES
# =============================================================================

def build_fuzzy_table(weights: pd.Series, fuzzy_pct: float) -> pd.DataFrame:
    rows = []
    factor = fuzzy_pct / 100.0
    for crit in weights.index:
        w = float(weights[crit])
        low = max(w * (1 - factor), 0.0)
        high = min(w * (1 + factor), 1.0)
        centroid = (low + w + high) / 3.0
        rows.append([crit, round(low, 4), round(w, 4), round(high, 4), round(centroid, 4)])
    df = pd.DataFrame(rows, columns=["Tiêu chí", "Low", "Mid", "High", "Centroid"])
    return df


def most_uncertain_criterion(weights: pd.Series, fuzzy_pct: float) -> Tuple[str, Dict[str, float]]:
    factor = fuzzy_pct / 100.0
    diff_map: Dict[str, float] = {}
    for crit in weights.index:
        w = float(weights[crit])
        low = w * (1 - factor)
        high = w * (1 + factor)
        diff_map[crit] = float(high - low)
    most_unc = max(diff_map, key=diff_map.get)
    return most_unc, diff_map


def fuzzy_heatmap_premium(diff_map: Dict[str, float]) -> go.Figure:
    values = list(diff_map.values())
    labels = list(diff_map.keys())

    fig = px.imshow(
        [values],
        labels=dict(color="Mức dao động"),
        x=labels,
        y=[""],
        color_continuous_scale=[
            [0.0, "#00331F"],
            [0.2, "#006642"],
            [0.4, "#00AA66"],
            [0.6, "#00DD88"],
            [1.0, "#00FFAA"]
        ]
    )

    fig.update_layout(
        title=dict(
            text="<b>🌿 Heatmap mức dao động Fuzzy (Premium Green)</b>",
            font=dict(size=22, color="#CCFFE6"),
            x=0.5
        ),
        paper_bgcolor="#001a12",
        plot_bgcolor="#001a12",
        margin=dict(l=40, r=40, t=80, b=40),
        coloraxis_colorbar=dict(
            title="Dao động",
            tickfont=dict(color="#CCFFE6")
        )
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(showticklabels=False)
    return fig


def fuzzy_chart_premium(weights: pd.Series, fuzzy_pct: float) -> go.Figure:
    factor = fuzzy_pct / 100.0
    labels = list(weights.index)
    low_vals, mid_vals, high_vals = [], [], []

    for crit in labels:
        w = float(weights[crit])
        low = max(w * (1 - factor), 0.0)
        high = min(w * (1 + factor), 1.0)
        low_vals.append(low)
        mid_vals.append(w)
        high_vals.append(high)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels, y=low_vals,
        mode="lines+markers", name="Low",
        line=dict(width=2, color="#004d40", dash="dot"),
        marker=dict(size=8),
        hovertemplate="Tiêu chí: %{x}<br>Low: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=labels, y=mid_vals,
        mode="lines+markers", name="Mid (gốc)",
        line=dict(width=3, color="#00e676"),
        marker=dict(size=9, symbol="diamond"),
        hovertemplate="Tiêu chí: %{x}<br>Mid: %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=labels, y=high_vals,
        mode="lines+markers", name="High",
        line=dict(width=2, color="#69f0ae", dash="dash"),
        marker=dict(size=8),
        hovertemplate="Tiêu chí: %{x}<br>High: %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>🌿 Fuzzy AHP — Low / Mid / High (±{fuzzy_pct:.0f}%)</b>",
            font=dict(size=22, color="#e6fff7"),
            x=0.5
        ),
        paper_bgcolor="#001a12",
        plot_bgcolor="#001a12",
        legend=dict(
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="#00e676",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=80, b=80),
        font=dict(size=13, color="#e6fff7")
    )
    fig.update_xaxes(showgrid=False, tickangle=-20)
    fig.update_yaxes(
        title="Trọng số",
        range=[0, max(0.4, max(high_vals) * 1.15)],
        showgrid=True,
        gridcolor="#004d40"
    )
    return fig


# =============================================================================
# MULTI-PACKAGE ANALYZER
# =============================================================================

class MultiPackageAnalyzer:
    def __init__(self):
        self.data_service = DataService()
        self.fuzzy_ahp = FuzzyAHP()
        self.mc_simulator = MonteCarloSimulator()
        self.topsis = TOPSISAnalyzer()
        self.risk_calc = RiskCalculator()
        self.forecaster = Forecaster()

    def run_analysis(self, params: AnalysisParams, historical: pd.DataFrame) -> AnalysisResult:
        profile_weights = PRIORITY_PROFILES[params.priority]
        weights = pd.Series(profile_weights, index=CRITERIA)

        if params.use_fuzzy:
            weights = self.fuzzy_ahp.apply(weights, params.fuzzy_uncertainty)

        company_data = self.data_service.get_company_data()

        if params.month in historical["month"].values:
            base_risk = float(
                historical.loc[historical["month"] == params.month, params.route].iloc[0]
            )
        else:
            base_risk = 0.4

        if params.use_mc:
            companies, mc_mean, mc_std = self.mc_simulator.simulate(
                base_risk, SENSITIVITY_MAP, params.mc_runs
            )
            order = [companies.index(c) for c in company_data.index]
            mc_mean, mc_std = mc_mean[order], mc_std[order]
        else:
            mc_mean = np.zeros(len(company_data))
            mc_std = np.zeros(len(company_data))

        all_options = []
        for company in company_data.index:
            for icc_name, icc_data in ICC_PACKAGES.items():
                option = company_data.loc[company].copy()

                base_premium = option["C1: Tỷ lệ phí"]
                option["C1: Tỷ lệ phí"] = base_premium * icc_data["premium_multiplier"]
                option["C4: Hỗ trợ ICC"] = option["C4: Hỗ trợ ICC"] * icc_data["coverage"]

                idx = list(company_data.index).index(company)
                option["C6: Rủi ro khí hậu"] = mc_mean[idx]

                all_options.append({
                    "company": company,
                    "icc_package": icc_name,
                    "coverage": icc_data["coverage"],
                    "premium_rate": option["C1: Tỷ lệ phí"],
                    "estimated_cost": params.cargo_value * option["C1: Tỷ lệ phí"],
                    "C1: Tỷ lệ phí": option["C1: Tỷ lệ phí"],
                    "C2: Thời gian xử lý": option["C2: Thời gian xử lý"],
                    "C3: Tỷ lệ tổn thất": option["C3: Tỷ lệ tổn thất"],
                    "C4: Hỗ trợ ICC": option["C4: Hỗ trợ ICC"],
                    "C5: Chăm sóc KH": option["C5: Chăm sóc KH"],
                    "C6: Rủi ro khí hậu": option["C6: Rủi ro khí hậu"],
                    "C6_std": mc_std[idx]
                })

        data_adjusted = pd.DataFrame(all_options)

        if params.cargo_value > 50_000:
            data_adjusted["C1: Tỷ lệ phí"] *= 1.1
            data_adjusted["estimated_cost"] *= 1.1

        scores = self.topsis.analyze(
            data_adjusted[["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
                           "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]],
            weights,
            COST_BENEFIT_MAP
        )

        data_adjusted["score"] = scores
        data_adjusted["C6_mean"] = data_adjusted["C6: Rủi ro khí hậu"]

        data_adjusted = data_adjusted.sort_values("score", ascending=False).reset_index(drop=True)
        data_adjusted["rank"] = data_adjusted.index + 1

        def categorize_option(row):
            if row["icc_package"] == "ICC C":
                return "💰 Tiết kiệm"
            elif row["icc_package"] == "ICC B":
                return "⚖️ Cân bằng"
            else:
                return "🛡️ An toàn"

        data_adjusted["category"] = data_adjusted.apply(categorize_option, axis=1)

        eps = 1e-9
        cv_c6 = data_adjusted["C6_std"].values / (data_adjusted["C6_mean"].values + eps)
        conf = 1.0 / (1.0 + cv_c6)
        conf = 0.3 + 0.7 * (conf - conf.min()) / (np.ptp(conf) + eps)
        data_adjusted["confidence"] = conf

        var = cvar = None
        if params.use_var:
            var, cvar = self.risk_calc.calculate_var_cvar(
                data_adjusted["C6_mean"].values, params.cargo_value
            )

        hist_series, forecast = self.forecaster.forecast(
            historical, params.route, params.month, use_arima=params.use_arima
        )

        return AnalysisResult(
            results=data_adjusted,
            weights=weights,
            data_adjusted=data_adjusted,
            var=var,
            cvar=cvar,
            historical=hist_series,
            forecast=forecast
        )


# =============================================================================
# CHART FACTORY
# =============================================================================

class ChartFactory:
    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=22, color="#e6fff7"),
                x=0.5
            ),
            font=dict(size=15, color="#e6fff7"),
            plot_bgcolor="#001016",
            paper_bgcolor="#000c11",
            margin=dict(l=70, r=40, t=80, b=70),
            legend=dict(
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="#00e676",
                borderwidth=1
            )
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#00332b",
            tickfont=dict(size=14, color="#e6fff7")
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#00332b",
            tickfont=dict(size=14, color="#e6fff7")
        )
        return fig

    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        colors = ['#00e676', '#69f0ae', '#b9f6ca', '#00bfa5', '#1de9b6', '#64ffda']
        labels_full = list(weights.index)
        labels_short = [c.split(':')[0] for c in labels_full]

        fig = go.Figure(data=[go.Pie(
            labels=labels_full,
            values=weights.values,
            text=labels_short,
            textinfo='text+percent',
            textposition='inside',
            hole=0.18,
            marker=dict(colors=colors, line=dict(color='#00130d', width=2)),
            pull=[0.04] * len(weights),
            hovertemplate="<b>%{label}</b><br>Tỉ trọng: %{percent}<extra></extra>"
        )])

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=20, color="#a5ffdc"),
                x=0.5,
                y=0.98
            ),
            showlegend=True,
            legend=dict(
                title="<b>Các tiêu chí</b>",
                font=dict(size=13, color="#e6fff7")
            ),
            paper_bgcolor="#001016",
            plot_bgcolor="#001016",
            margin=dict(l=0, r=0, t=80, b=0),
            height=480
        )
        return fig

    @staticmethod
    def create_cost_benefit_scatter(results: pd.DataFrame) -> go.Figure:
        color_map = {
            "ICC A": "#ff6b6b",
            "ICC B": "#ffd93d",
            "ICC C": "#6bcf7f"
        }

        fig = go.Figure()

        for icc in ["ICC C", "ICC B", "ICC A"]:
            df_icc = results[results["icc_package"] == icc]
            fig.add_trace(go.Scatter(
                x=df_icc["estimated_cost"],
                y=df_icc["score"],
                mode="markers+text",
                name=icc,
                text=df_icc["company"],
                textposition="top center",
                marker=dict(
                    size=15,
                    color=color_map[icc],
                    line=dict(width=2, color="#000")
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>" +
                    f"Gói: {icc}<br>" +
                    "Chi phí: $%{x:,.0f}<br>" +
                    "Điểm: %{y:.3f}<extra></extra>"
                )
            ))

        fig.update_xaxes(title="<b>Chi phí ước tính ($)</b>")
        fig.update_yaxes(title="<b>Điểm TOPSIS</b>", range=[0, 1])

        fig = ChartFactory._apply_theme(fig, "💰 Chi phí vs Chất lượng (Cost-Benefit Analysis)")
        fig.update_layout(height=480, autosize=False)
        return fig

    @staticmethod
    def create_top_recommendations_bar(results: pd.DataFrame) -> go.Figure:
        df = results.head(5).copy()
        df["label"] = df["company"] + " - " + df["icc_package"]

        fig = go.Figure(data=[go.Bar(
            x=df["score"],
            y=df["label"],
            orientation="h",
            text=[f"{v:.3f}" for v in df["score"]],
            textposition="outside",
            marker=dict(
                color=df["score"],
                colorscale=[[0, '#69f0ae'], [0.5, '#00e676'], [1, '#00c853']],
                line=dict(color='#00130d', width=1)
            ),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<br>Chi phí: $%{customdata:,.0f}<extra></extra>",
            customdata=df["estimated_cost"]
        )])

        fig.update_xaxes(title="<b>Điểm TOPSIS</b>", range=[0, 1])
        fig.update_yaxes(title="<b>Phương án</b>")

        fig = ChartFactory._apply_theme(fig, "🏆 Top 5 Phương án Tốt nhất")
        fig.update_layout(height=440)
        return fig

    @staticmethod
    def create_forecast_chart(
        historical: np.ndarray,
        forecast: np.ndarray,
        route: str,
        selected_month: int
    ) -> go.Figure:
        hist_len = len(historical)
        months_hist = list(range(1, hist_len + 1))
        next_month = selected_month % 12 + 1
        months_fc = [next_month]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=months_hist,
            y=historical,
            mode="lines+markers",
            name="📈 Lịch sử",
            line=dict(color="#00e676", width=3),
            marker=dict(size=9),
            hovertemplate="Tháng %{x}<br>Rủi ro: %{y:.1%}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=months_fc,
            y=forecast,
            mode="lines+markers",
            name="🔮 Dự báo",
            line=dict(color="#ffeb3b", width=3, dash="dash"),
            marker=dict(size=11, symbol="diamond"),
            hovertemplate="Tháng %{x}<br>Dự báo: %{y:.1%}<extra></extra>"
        ))

        fig = ChartFactory._apply_theme(fig, f"Dự báo rủi ro khí hậu — {route}")

        fig.update_xaxes(
            title="<b>Tháng</b>",
            tickmode="linear",
            tick0=1,
            dtick=1,
            range=[1, 12],
            tickvals=list(range(1, 13))
        )

        max_val = max(float(historical.max()), float(forecast.max()))
        fig.update_yaxes(
            title="<b>Mức rủi ro (0–1)</b>",
            range=[0, max(1.0, max_val * 1.15)],
            tickformat=".0%"
        )

        fig.update_layout(height=450, autosize=False)
        return fig

    @staticmethod
    def create_category_comparison(results: pd.DataFrame) -> go.Figure:
        categories = ["💰 Tiết kiệm", "⚖️ Cân bằng", "🛡️ An toàn"]
        avg_scores = []
        avg_costs = []

        for cat in categories:
            df_cat = results[results["category"] == cat]
            if len(df_cat) > 0:
                avg_scores.append(df_cat["score"].mean())
                avg_costs.append(df_cat["estimated_cost"].mean())
            else:
                avg_scores.append(0)
                avg_costs.append(0)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Điểm trung bình",
            x=categories,
            y=avg_scores,
            marker=dict(color='#00e676'),
            yaxis="y",
            hovertemplate="<b>%{x}</b><br>Điểm TB: %{y:.3f}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            name="Chi phí trung bình",
            x=categories,
            y=avg_costs,
            mode="lines+markers",
            marker=dict(size=10, color='#ffeb3b'),
            line=dict(width=3, color='#ffeb3b'),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>Chi phí TB: $%{y:,.0f}<extra></extra>"
        ))

        fig.update_layout(
            title=dict(
                text="<b>📊 So sánh 3 loại phương án</b>",
                font=dict(size=22, color="#e6fff7"),
                x=0.5
            ),
            yaxis=dict(
                title=dict(text="<b>Điểm TOPSIS</b>", font=dict(color="#00e676")),
                range=[0, 1],
                tickfont=dict(color="#00e676")
            ),
            yaxis2=dict(
                title=dict(text="<b>Chi phí ($)</b>", font=dict(color="#ffeb3b")),
                overlaying="y",
                side="right",
                tickfont=dict(color="#ffeb3b")
            ),
            paper_bgcolor="#000c11",
            plot_bgcolor="#001016",
            font=dict(color="#e6fff7"),
            legend=dict(
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="#00e676",
                borderwidth=1
            ),
            margin=dict(l=60, r=60, t=80, b=60),
            height=480,
            autosize=False
        )

        return fig


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

class ReportGenerator:
    @staticmethod
    def generate_pdf(
        results: pd.DataFrame,
        params: AnalysisParams,
        var: Optional[float],
        cvar: Optional[float]
    ) -> bytes:
        try:
            pdf = FPDF()
            pdf.add_page()

            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST v5.5 - Multi-Package Analysis", 0, 1, "C")
            pdf.ln(4)

            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, f"Route: {params.route} | Month: {params.month} | Priority: {params.priority}", 0, 1)
            pdf.cell(0, 6, f"Cargo Value: ${params.cargo_value:,.0f}", 0, 1)
            pdf.ln(4)

            top = results.iloc[0]
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, f"Top Recommendation: {top['company']} - {top['icc_package']}", 0, 1)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, f"Score: {top['score']:.3f} | Cost: ${top['estimated_cost']:,.0f}", 0, 1)
            pdf.cell(0, 6, f"Confidence: {top['confidence']:.2f}", 0, 1)
            pdf.ln(4)

            pdf.set_font("Arial", "B", 10)
            pdf.cell(15, 6, "Rank", 1)
            pdf.cell(40, 6, "Company", 1)
            pdf.cell(25, 6, "ICC", 1)
            pdf.cell(30, 6, "Cost", 1)
            pdf.cell(25, 6, "Score", 1)
            pdf.cell(25, 6, "Conf.", 1, 1)

            pdf.set_font("Arial", "", 9)
            for _, row in results.head(10).iterrows():
                pdf.cell(15, 6, str(int(row["rank"])), 1)
                pdf.cell(40, 6, str(row["company"])[:18], 1)
                pdf.cell(25, 6, str(row["icc_package"]), 1)
                pdf.cell(30, 6, f"${row['estimated_cost']:,.0f}", 1)
                pdf.cell(25, 6, f"{row['score']:.3f}", 1)
                pdf.cell(25, 6, f"{row['confidence']:.2f}", 1, 1)

            if var is not None and cvar is not None:
                pdf.ln(4)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, f"VaR 95%: ${var:,.0f}   |   CVaR 95%: ${cvar:,.0f}", 0, 1)

            return pdf.output(dest="S").encode("latin1")
        except Exception as e:
            st.error(f"Lỗi tạo PDF: {e}")
            return b""

    @staticmethod
    def generate_excel(results: pd.DataFrame, weights: pd.Series) -> bytes:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            results[["rank", "company", "icc_package", "estimated_cost", "score",
                     "confidence", "category"]].to_excel(writer, sheet_name="Results", index=False)
            pd.DataFrame({"weight": weights.values}, index=weights.index).to_excel(
                writer, sheet_name="Weights"
            )
        buffer.seek(0)
        return buffer.getvalue()


# =============================================================================
# STREAMLIT UI
# =============================================================================

class StreamlitUI:
    def __init__(self):
        self.analyzer = MultiPackageAnalyzer()
        self.chart_factory = ChartFactory()
        self.report_gen = ReportGenerator()

    def initialize(self):
        app_config()
        apply_enterprise_css()

    def render_header(self):
        st.markdown(
            """
            <div class="rc-header">
                <div class="rc-header-left">
                    <div class="rc-logo">RC</div>
                    <div>
                        <div class="rc-title">RISKCAST v5.5 — MULTI-PACKAGE ANALYSIS</div>
                        <div class="rc-subtitle">
                            15 phương án (5 Công ty × 3 Gói ICC) · 
                            <span class="rc-tooltip" data-tip="Hệ thống tự điều chỉnh trọng số theo mục tiêu (Tiết kiệm / Cân bằng / An toàn)">Profile-based recommendation</span> ·
                            <span class="rc-tooltip" data-tip="TOPSIS + Monte Carlo + VaR/CVaR + Fuzzy AHP">Hybrid ESG Risk Engine</span>
                        </div>
                    </div>
                </div>
                <div class="rc-badge">
                    🎯 Smart ESG Risk &amp; Insurance Decision
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    def render_sidebar(self) -> AnalysisParams:
        with st.sidebar:
            st.header("📊 Thông tin lô hàng")

            cargo_value = st.number_input("Giá trị (USD)", 1000, value=39_000, step=1_000)
            good_type = st.selectbox(
                "Loại hàng",
                ["Điện tử", "Đông lạnh", "Hàng khô", "Nguy hiểm", "Khác"]
            )
            route = st.selectbox(
                "Tuyến vận chuyển",
                ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"]
            )
            method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
            month = st.selectbox("Tháng", list(range(1, 13)), index=8)

            st.markdown("---")
            st.header("🎯 Mục tiêu của bạn")
            priority = st.selectbox(
                "Chọn mục tiêu ưu tiên",
                list(PRIORITY_PROFILES.keys()),
                help="Hệ thống sẽ tự động điều chỉnh trọng số theo mục tiêu bạn chọn"
            )

            st.markdown("---")
            st.header("⚙️ Cấu hình mô hình")

            use_fuzzy = st.checkbox("Bật Fuzzy AHP", True,
                                    help="Xử lý bất định trong đánh giá chuyên gia bằng tam giác mờ")
            use_arima = st.checkbox("Dùng ARIMA dự báo", True,
                                    help="Dùng ARIMA(1,1,1) để dự báo rủi ro khí hậu tháng kế tiếp")
            use_mc = st.checkbox("Monte Carlo (C6)", True,
                                 help="Mô phỏng nhiều kịch bản rủi ro khí hậu để lấy mean & std")
            use_var = st.checkbox("Tính VaR/CVaR", True,
                                  help="Đo lường tổn thất tối đa & tổn thất trung bình trong tail")

            mc_runs = st.number_input("Số lần Monte Carlo", 500, 10_000, 2_000, 500)
            fuzzy_uncertainty = st.slider("Mức bất định Fuzzy (%)", 0, 50, 15) if use_fuzzy else 15

            return AnalysisParams(
                cargo_value, good_type, route, method, month, priority,
                use_fuzzy, use_arima, use_mc, use_var, mc_runs, fuzzy_uncertainty
            )

    def display_profile_explanation(self, params: AnalysisParams):
        st.markdown('<div class="rc-card">', unsafe_allow_html=True)
        st.subheader(f"📌 Đã chọn mục tiêu: {params.priority}")

        profile_weights = PRIORITY_PROFILES[params.priority]
        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>⚙️ Trọng số tự động theo mục tiêu</h4>
                <ul>
                    <li><b>C1 (Chi phí):</b> {profile_weights['C1: Tỷ lệ phí']:.0%}</li>
                    <li><b>C2 (Thời gian xử lý):</b> {profile_weights['C2: Thời gian xử lý']:.0%}</li>
                    <li><b>C3 (Tỷ lệ tổn thất):</b> {profile_weights['C3: Tỷ lệ tổn thất']:.0%}</li>
                    <li><b>C4 (Hỗ trợ ICC):</b> {profile_weights['C4: Hỗ trợ ICC']:.0%}</li>
                    <li><b>C5 (Chăm sóc KH):</b> {profile_weights['C5: Chăm sóc KH']:.0%}</li>
                    <li><b>C6 (Rủi ro khí hậu):</b> {profile_weights['C6: Rủi ro khí hậu']:.0%}</li>
                </ul>
                <p>
                    <b>💡 Gợi ý dùng trong báo cáo NCKH:</b><br>
                    Trình bày rằng hệ thống áp dụng 
                    <span class="rc-tooltip" 
                    data-tip="Trọng số được xác định trước theo hành vi ra quyết định điển hình của nhà xuất nhập khẩu">
                    hồ sơ ưu tiên (priority profile)</span> để phản ánh mục tiêu thực tế của doanh nghiệp.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    def display_results(self, result: AnalysisResult, params: AnalysisParams):
        st.success("✅ Đã phân tích xong 15 phương án (5 công ty × 3 gói ICC)")

        top = result.results.iloc[0]
        st.markdown(
            f"""
            <div class="result-box">
                🏆 <b>GỢI Ý TỐT NHẤT CHO MỤC TIÊU: {params.priority}</b><br><br>
                <span style="font-size:1.6rem;">{top['company']} - {top['icc_package']}</span><br><br>
                💰 Chi phí: <b>${top['estimated_cost']:,.0f}</b> ({top['premium_rate']:.2%} giá trị hàng)<br>
                📊 Điểm TOPSIS: <b>{top['score']:.3f}</b>
                <span class="tooltip-icon" data-tip="TOPSIS đo mức độ gần với phương án lý tưởng (ideal best) 
và xa phương án tệ nhất (ideal worst). Điểm càng cao càng tốt.">i</span> |
                🎯 Độ tin cậy: <b>{top['confidence']:.2f}</b><br>
                📦 Loại gợi ý: <b>{top['category']}</b><br>
                📜 Gói ICC: <b>{ICC_PACKAGES[top['icc_package']]['description']}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("📋 Giải thích kết quả chi tiết")

        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>🎯 Vì sao <b>{top['company']} - {top['icc_package']}</b> được khuyến nghị?</h4>
                <ul>
                    <li><b>Điểm TOPSIS cao nhất:</b> {top['score']:.3f} (gần phương án lý tưởng nhất).</li>
                    <li><b>Cân bằng theo mục tiêu:</b> {params.priority} 
                        – trọng số được điều chỉnh để ưu tiên mục tiêu này.</li>
                    <li><b>Chi phí &amp; bảo vệ:</b> ${top['estimated_cost']:,.0f} với mức bảo vệ 
                        "<i>{ICC_PACKAGES[top['icc_package']]['description']}</i>".</li>
                    <li><b>Độ tin cậy mô hình:</b> {top['confidence']:.2f} 
                        – dựa trên độ biến động rủi ro khí hậu (Monte Carlo).</li>
                </ul>
                <p>
                    <b>📌 Gợi ý viết NCKH:</b> có thể mô tả đây là 
                    <span class="rc-tooltip" data-tip="Kết quả tích hợp giữa mô hình MC, TOPSIS, VaR/CVaR, Fuzzy AHP và bộ trọng số theo hồ sơ ưu tiên.">
                        phương án tối ưu đa tiêu chí
                    </span> phù hợp khẩu vị rủi ro và ngân sách của doanh nghiệp.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Top 3 Premium Cards
        st.markdown("## 🏅 Top 3 phương án (Premium View)")

        cols = st.columns(3)
        top3 = result.results.head(3)
        medals = ["🥇", "🥈", "🥉"]

        for i, col in enumerate(cols):
            if i >= len(top3):
                continue
            r = top3.iloc[i]
            card_class = "top3-card"
            title_class = "top3-title"
            if i == 0:
                card_class += " top1-card"
                title_class = "top1-title"

            with col:
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="{title_class}">{medals[i]} #{i+1}: {r['company']}</div>
                        <div class="top3-sub">
                            <span class="badge-icc">{r['icc_package']}</span>
                            <div class="pill-badge">{r['category']}</div>
                        </div>
                        <div class="top3-sub" style="color:#7CFFA1; font-size:0.98rem;">
                            💰 Chi phí: <b>${r['estimated_cost']:,.0f}</b>
                        </div>
                        <div class="top3-sub">
                            📊 Điểm TOPSIS: <b>{r['score']:.3f}</b>
                        </div>
                        <div class="top3-sub">
                            🎯 Tin cậy mô hình: <b>{r['confidence']:.2f}</b>
                        </div>
                        <div class="top3-sub">
                            🌪 Rủi ro khí hậu (mean ± std): 
                            <b>{r['C6_mean']:.2%} ± {r['C6_std']:.2%}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Table 15 options
        st.markdown("---")
        st.subheader("📋 Bảng so sánh 15 phương án (đầy đủ)")

        df_display = result.results[["rank", "company", "icc_package", "category",
                                     "estimated_cost", "score", "confidence"]].copy()
        df_display.columns = ["Hạng", "Công ty", "Gói ICC", "Loại", "Chi phí", "Điểm", "Tin cậy"]
        df_display["Chi phí"] = df_display["Chi phí"].apply(lambda x: f"${x:,.0f}")
        df_display = df_display.set_index("Hạng")

        st.dataframe(df_display, use_container_width=True)

        st.markdown(
            """
            <div class="explanation-box">
                <h4>💡 Giải thích 3 loại phương án</h4>
                <ul>
                    <li><b>💰 Tiết kiệm (ICC C):</b> phí thấp nhất, bảo vệ cơ bản – phù hợp hàng giá trị thấp, tuyến ngắn.</li>
                    <li><b>⚖️ Cân bằng (ICC B):</b> phí trung bình, bảo vệ vừa – phù hợp đa số lô hàng thông thường.</li>
                    <li><b>🛡️ An toàn (ICC A):</b> phí cao nhất, bảo vệ toàn diện – phù hợp hàng giá trị cao, tuyến rủi ro lớn.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # VaR/CVaR explanation
        if result.var is not None and result.cvar is not None:
            risk_pct = (result.var / params.cargo_value) * 100
            st.markdown(
                f"""
                <div class="explanation-box">
                    <h4>
                        ⚠️ Đánh giá rủi ro tài chính (VaR / CVaR)
                        <span class="tooltip-icon" data-tip="VaR 95%: tổn thất tối đa có thể xảy ra với mức tin cậy 95%.
CVaR 95%: tổn thất trung bình trong 5% trường hợp xấu nhất.">i</span>
                    </h4>
                    <ul>
                        <li><b>VaR 95%:</b> ${result.var:,.0f} ({risk_pct:.1f}% giá trị hàng).</li>
                        <li><b>CVaR 95%:</b> ${result.cvar:,.0f} – tổn thất trung bình trong vùng tail.</li>
                        <li><b>Nhận định:</b> {'✅ Rủi ro ở mức chấp nhận được.' if risk_pct < 10 else '⚠️ Rủi ro khá cao, nên tăng mức bảo hiểm.'}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

                # Charts section
        st.markdown("---")
        st.subheader("Biểu đồ phân tích")

        # Biểu đồ 1: Chi phí – Chất lượng (Cost–Benefit)
        st.markdown("""
        <h4 style='display:flex;align-items:center;gap:6px;'>
        📉 Chi phí – Chất lượng (Cost–Benefit)
        <span class="tooltip-icon" data-tip="Mỗi điểm là một phương án bảo hiểm (công ty × gói ICC).
Trục X: chi phí ước tính; Trục Y: điểm TOPSIS.
Điểm càng cao và chi phí càng thấp → phương án càng hấp dẫn.">i</span>
        </h4>
        """, unsafe_allow_html=True)
        fig_scatter = self.chart_factory.create_cost_benefit_scatter(result.results)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Biểu đồ 2: So sánh 3 loại phương án
        st.markdown("""
        <h4 style='display:flex;align-items:center;gap:6px;'>
        📊 So sánh 3 loại phương án
        <span class="tooltip-icon" data-tip="So sánh trung bình điểm TOPSIS và trung bình chi phí
của 3 nhóm: Tiết kiệm (ICC C), Cân bằng (ICC B), An toàn (ICC A).">i</span>
        </h4>
        """, unsafe_allow_html=True)
        fig_category = self.chart_factory.create_category_comparison(result.results)
        st.plotly_chart(fig_category, use_container_width=True)

        # Biểu đồ 3: Top 5 phương án tốt nhất
        st.markdown("""
        <h4 style='display:flex;align-items:center;gap:6px;'>
        🏆 Top 5 phương án tốt nhất
        <span class="tooltip-icon" data-tip="Biểu đồ ngang hiển thị 5 phương án có điểm TOPSIS cao nhất.">i</span>
        </h4>
        """, unsafe_allow_html=True)
        fig_top5 = self.chart_factory.create_top_recommendations_bar(result.results)
        st.plotly_chart(fig_top5, use_container_width=True)

        # Biểu đồ 4: Trọng số tiêu chí
        st.markdown("""
        <h4 style='display:flex;align-items:center;gap:6px;'>
        📘 Trọng số tiêu chí
        <span class="tooltip-icon" data-tip="Trọng số được xác định theo hồ sơ ưu tiên (Tiết kiệm / Cân bằng / An toàn).
Nếu bật Fuzzy AHP, mỗi trọng số được mở rộng thành tam giác mờ (Low–Mid–High).">i</span>
        </h4>
        """, unsafe_allow_html=True)
        fig_weights = self.chart_factory.create_weights_pie(
            result.weights,
            "Trọng số tiêu chí (sau khi áp dụng Fuzzy AHP)" if params.use_fuzzy else "Trọng số tiêu chí"
        )
        st.plotly_chart(fig_weights, use_container_width=True)

        # Biểu đồ 5: Dự báo rủi ro khí hậu
        st.markdown("""
        <h4 style='display:flex;align-items:center;gap:6px;'>
        📉 Dự báo rủi ro khí hậu theo tháng
        <span class="tooltip-icon" data-tip="Từ dữ liệu lịch sử rủi ro khí hậu theo tuyến,
mô hình dự báo giá trị tháng kế tiếp (ARIMA hoặc xu hướng tuyến tính).">i</span>
        </h4>
        """, unsafe_allow_html=True)
        fig_forecast = self.chart_factory.create_forecast_chart(
            result.historical, result.forecast, params.route, params.month
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        
        
        # Fuzzy AHP
        if params.use_fuzzy:
            st.markdown("---")
            st.subheader("🌿 Fuzzy AHP — Phân tích bất định trọng số")

            st.markdown(
                """
                <div class="explanation-box">
                    <h4>📚 Fuzzy AHP là gì?</h4>
                    <ul>
                        <li>Sử dụng <b>tam giác mờ (Low – Mid – High)</b> để biểu diễn sự không chắc chắn trong ý kiến chuyên gia.</li>
                        <li>Trọng số cuối cùng được tính bằng 
                            <span class="rc-tooltip" data-tip="(Low + Mid + High) / 3">(Low + Mid + High) / 3</span> 
                            rồi chuẩn hóa lại.</li>
                        <li>Giúp mô hình <b>mềm dẻo hơn</b>, không phụ thuộc duy nhất vào một bộ trọng số cứng.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

            fig_fuzzy = fuzzy_chart_premium(result.weights, params.fuzzy_uncertainty)
            st.plotly_chart(fig_fuzzy, use_container_width=True)

            fuzzy_table = build_fuzzy_table(result.weights, params.fuzzy_uncertainty)
            st.dataframe(fuzzy_table, use_container_width=True)

            most_unc, diff_map = most_uncertain_criterion(result.weights, params.fuzzy_uncertainty)
            st.markdown(
                f"""
                <div style="background:#00331F; padding:15px; border-radius:10px;
                border:2px solid #00FFAA; color:#CCFFE6;">
                    <b>🔍 Tiêu chí dao động mạnh nhất:</b> {most_unc}<br>
                    <small>Độ chênh lệch (Low → High): {diff_map[most_unc]:.4f}</small>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("🔥 Heatmap mức dao động Fuzzy (Premium Green)")
            fig_heat = fuzzy_heatmap_premium(diff_map)
            st.plotly_chart(fig_heat, use_container_width=True)

        # Export
        st.markdown("---")
        st.subheader("📥 Xuất báo cáo")

        col_e1, col_e2 = st.columns(2)

        with col_e1:
            excel_data = self.report_gen.generate_excel(result.results, result.weights)
            st.download_button(
                "📊 Tải Excel (kết quả + trọng số)",
                data=excel_data,
                file_name=f"riskcast_v55_{params.route.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col_e2:
            pdf_data = self.report_gen.generate_pdf(result.results, params, result.var, result.cvar)
            if pdf_data:
                st.download_button(
                    "📄 Tải PDF (tóm tắt báo cáo)",
                    data=pdf_data,
                    file_name=f"riskcast_v55_{params.route.replace(' - ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    def run(self):
        self.initialize()
        self.render_header()

        historical = DataService.load_historical_data()
        params = self.render_sidebar()
        self.display_profile_explanation(params)

        st.markdown("---")
        if st.button("🚀 PHÂN TÍCH 15 PHƯƠNG ÁN", type="primary", use_container_width=True):
            with st.spinner("🔄 Đang phân tích tất cả phương án..."):
                try:
                    result = self.analyzer.run_analysis(params, historical)
                    self.display_results(result, params)
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
                    st.exception(e)


# =============================================================================
# MAIN
# =============================================================================

def main():
    app = StreamlitUI()
    app.run()


if __name__ == "__main__":
    main()
