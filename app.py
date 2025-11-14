# =============================================================================
# RISKCAST v5.3 — ENTERPRISE EDITION (Multi-Package Analysis)
# ESG Logistics Risk Assessment Dashboard
#
# Author: Bùi Xuân Hoàng (original idea)
# Refactor + Multi-Package + Full Explanations + Enterprise UX: Kai assistant
#
# Nổi bật trong v5.3 Enterprise:
#   - Profile-Based Recommendation (3 mục tiêu: Tiết kiệm / Cân bằng / An toàn)
#   - Multi-Package Analysis (5 công ty × 3 gói ICC = 15 phương án)
#   - Smart Ranking Table với badges
#   - Cost-Benefit Scatter Plot
#   - Trade-off Analysis
#   - Fuzzy AHP Enterprise module (heatmap + radar-style line)
#   - Forecast chart nền tối + line neon
#   - TẤT CẢ EXPLANATION BOXES cho NCKH
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
# DOMAIN MODELS & CONSTANTS
# =============================================================================

class CriterionType(Enum):
    """Loại tiêu chí: chi phí (càng thấp càng tốt) hoặc lợi ích (càng cao càng tốt)."""
    COST = "cost"
    BENEFIT = "benefit"


@dataclass
class AnalysisParams:
    """Các tham số đầu vào cho 1 lần phân tích."""
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
    """Kết quả phân tích."""
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

# Profile weights - Trọng số theo mục tiêu
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

# ICC Package definitions
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

# Map loại tiêu chí
COST_BENEFIT_MAP = {
    "C1: Tỷ lệ phí": CriterionType.COST,
    "C2: Thời gian xử lý": CriterionType.COST,
    "C3: Tỷ lệ tổn thất": CriterionType.COST,
    "C4: Hỗ trợ ICC": CriterionType.BENEFIT,
    "C5: Chăm sóc KH": CriterionType.BENEFIT,
    "C6: Rủi ro khí hậu": CriterionType.COST
}

# Độ nhạy rủi ro khí hậu theo công ty
SENSITIVITY_MAP = {
    "Chubb": 0.95,
    "PVI": 1.05,
    "BaoViet": 1.00,
    "BaoMinh": 1.02,
    "MIC": 1.03
}


# =============================================================================
# UI STYLING — ENTERPRISE ESG PREMIUM GREEN
# =============================================================================

def apply_custom_css() -> None:
    """CSS Enterprise: Sidebar, Header, Card, Table, Mobile Hybrid Responsive."""
    st.markdown("""
    <style>
    * {
        text-rendering: optimizeLegibility !important;
        -webkit-font-smoothing: antialiased !important;
    }

    .stApp {
        background: radial-gradient(circle at top, #00ff99 0%, #001a0f 35%, #000c08 100%) !important;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
        color: #e6fff7 !important;
        font-size: 1.05rem !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    h1 { font-size: 2.8rem !important; font-weight: 900 !important; letter-spacing: 0.03em; }
    h2 { font-size: 2.1rem !important; font-weight: 800 !important; }
    h3 { font-size: 1.5rem !important; font-weight: 700 !important; }

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.1rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(0, 255, 153, 0.14), rgba(0, 0, 0, 0.88));
        border: 1px solid rgba(0, 255, 153, 0.45);
        box-shadow: 0 0 0 1px rgba(0, 255, 153, 0.12), 0 18px 45px rgba(0, 0, 0, 0.85);
        margin-bottom: 1.2rem;
        gap: 1.5rem;
    }

    .app-header-left { display: flex; align-items: center; gap: 0.9rem; }

    .app-logo-circle {
        width: 64px; height: 64px; border-radius: 18px;
        background: radial-gradient(circle at 30% 30%, #b9f6ca 0%, #00c853 38%, #00381f 100%);
        display: flex; align-items: center; justify-content: center;
        font-weight: 900; font-size: 1.4rem; color: #00130d;
        box-shadow: 0 0 14px rgba(0, 255, 153, 0.65), 0 0 36px rgba(0, 0, 0, 0.75);
        border: 2px solid #e8f5e9;
    }

    .app-header-title {
        font-size: 1.5rem; font-weight: 800;
        background: linear-gradient(90deg, #e8fffb, #b9f6ca, #e8fffb);
        -webkit-background-clip: text; color: transparent;
        letter-spacing: 0.05em; text-transform: uppercase;
    }

    .app-header-subtitle { font-size: 0.9rem; color: #ccffec; opacity: 0.9; }

    .app-header-badge {
        font-size: 0.86rem; font-weight: 600; padding: 0.55rem 0.9rem;
        border-radius: 999px; background: radial-gradient(circle at 0 0, #00e676, #00bfa5);
        color: #00130d; display: flex; align-items: center; gap: 0.35rem;
        white-space: nowrap; box-shadow: 0 0 14px rgba(0, 255, 153, 0.65), 0 0 22px rgba(0, 0, 0, 0.7);
    }

    section[data-testid="stSidebar"] {
        background: radial-gradient(circle at 0 0, #003322 0%, #000f0a 40%, #000805 100%) !important;
        border-right: 1px solid rgba(0, 230, 118, 0.55);
        box-shadow: 8px 0 22px rgba(0, 0, 0, 0.85);
    }

    section[data-testid="stSidebar"] > div { padding-top: 1.1rem; }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #a5ffdc !important; font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] label {
        color: #e0f2f1 !important; font-weight: 600 !important; font-size: 0.92rem !important;
    }

    .stButton > button {
        background: linear-gradient(120deg, #00ff99, #00e676, #00bfa5) !important;
        color: #00130d !important; font-weight: 800 !important;
        border-radius: 999px !important; border: none !important;
        padding: 0.65rem 1.9rem !important;
        box-shadow: 0 0 14px rgba(0, 255, 153, 0.7), 0 10px 22px rgba(0, 0, 0, 0.85) !important;
        transition: all 0.12s ease-out; font-size: 0.98rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 0 20px rgba(0, 255, 153, 0.95), 0 14px 30px rgba(0, 0, 0, 0.9) !important;
    }

    .premium-card {
        background: radial-gradient(circle at top left, rgba(0, 255, 153, 0.10), rgba(0, 0, 0, 0.95));
        border-radius: 16px; padding: 1.1rem 1.2rem;
        border: 1px solid rgba(0, 255, 153, 0.45);
        box-shadow: 0 0 0 1px rgba(0, 255, 153, 0.08), 0 16px 38px rgba(0, 0, 0, 0.9);
        margin-bottom: 1.2rem;
    }

    .result-box {
        background: radial-gradient(circle at top left,#00ff99,#00bfa5);
        color: #00130d !important; padding: 1.6rem 2rem; border-radius: 18px;
        font-weight: 800; box-shadow: 0 0 22px rgba(0, 255, 153, 0.7), 0 18px 40px rgba(0, 0, 0, 0.9);
        border: 2px solid #b9f6ca; margin-top: 0.6rem;
    }

    .explanation-box {
        background: rgba(0,40,28,0.92); border-left: 4px solid #00e676;
        padding: 1.2rem 1.5rem; border-radius: 12px; margin-top: 0.7rem;
        box-shadow: 0 0 16px rgba(0,0,0,0.7);
    }

    .explanation-box h4 { color: #a5ffdc !important; font-weight: 800; }
    .explanation-box li { color: #e0f2f1 !important; font-weight: 500; margin: 0.25rem 0; }

    div[data-testid="stDataFrame"] {
        border-radius: 14px !important; border: 1px solid rgba(0, 255, 170, 0.45) !important;
        overflow: hidden !important;
        box-shadow: 0 0 0 1px rgba(0, 255, 170, 0.10), 0 16px 40px rgba(0, 0, 0, 0.85) !important;
    }

    [data-testid="stMetricValue"] {
        color: #76ff03 !important; font-weight: 900 !important; font-size: 1.1rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #e0f2f1 !important; font-weight: 600 !important;
    }

    @media (max-width: 900px) {
        .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
        .app-header { flex-direction: column; align-items: flex-start; }
    }
    </style>
    """, unsafe_allow_html=True)


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
            mc_mean = mc_std = np.zeros(len(company_data))

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
# VISUALIZATION
# =============================================================================

class ChartFactory:
    """Tạo các biểu đồ Plotly."""

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

        return ChartFactory._apply_theme(fig, "💰 Chi phí vs Chất lượng (Cost-Benefit Analysis)")

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

        return ChartFactory._apply_theme(fig, "🏆 Top 5 Phương án Tốt nhất")

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
            marker=dict(size=12, color='#ffeb3b'),
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
            )
        )

        return fig


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

class ReportGenerator:
    """Xuất Excel & PDF."""

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
            pdf.cell(0, 10, "RISKCAST v5.3 - Multi-Package Analysis", 0, 1, "C")
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
        st.set_page_config(
            page_title="RISKCAST v5.3 — Multi-Package Analysis",
            page_icon="🛡️",
            layout="wide"
        )
        apply_custom_css()

        # Header
        st.markdown("""
        <div class="app-header">
            <div class="app-header-left">
                <div class="app-logo-circle">R</div>
                <div>
                    <div class="app-header-title">RISKCAST ENTERPRISE v5.3</div>
                    <div class="app-header-subtitle">
                        ESG Logistics Risk Assessment • Multi-Package (ICC A/B/C) • VaR/CVaR • Fuzzy AHP
                    </div>
                </div>
            </div>
            <div class="app-header-badge">
                🟢 Enterprise • Multi-Package • NCKH Ready
            </div>
        </div>
        """, unsafe_allow_html=True)

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

            use_fuzzy = st.checkbox("Bật Fuzzy AHP", True)
            use_arima = st.checkbox("Dùng ARIMA dự báo", True)
            use_mc = st.checkbox("Monte Carlo (C6)", True)
            use_var = st.checkbox("Tính VaR/CVaR", True)

            mc_runs = st.number_input("Số lần Monte Carlo", 500, 10_000, 2_000, 500)
            fuzzy_uncertainty = st.slider("Mức bất định Fuzzy (%)", 0, 50, 15) if use_fuzzy else 15

            return AnalysisParams(
                cargo_value, good_type, route, method, month, priority,
                use_fuzzy, use_arima, use_mc, use_var, mc_runs, fuzzy_uncertainty
            )

    def display_results(self, result: AnalysisResult, params: AnalysisParams):
        st.success("✅ Đã phân tích xong 15 phương án (5 công ty × 3 gói ICC)")

        top = result.results.iloc[0]
        st.markdown(
            f"""
            <div class="result-box">
                🏆 <b>GỢI Ý TỐT NHẤT CHO MỤC TIÊU: {params.priority}</b><br><br>
                <span style="font-size:1.6rem;">{top['company']} - {top['icc_package']}</span><br><br>
                💰 Chi phí: <b>${top['estimated_cost']:,.0f}</b> ({top['premium_rate']:.2%} giá trị hàng)<br>
                📊 Điểm TOPSIS: <b>{top['score']:.3f}</b> | 
                🎯 Độ tin cậy: <b>{top['confidence']:.2f}</b><br>
                📦 Loại: <b>{top['category']}</b>
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
                    <li><b>Điểm TOPSIS cao nhất:</b> {top['score']:.3f} - Cân bằng tốt nhất giữa chi phí và bảo vệ</li>
                    <li><b>Phù hợp với mục tiêu:</b> {params.priority} - Hệ thống đã tối ưu trọng số theo nhu cầu</li>
                    <li><b>Chi phí hợp lý:</b> ${top['estimated_cost']:,.0f} ({top['premium_rate']:.2%} giá trị hàng)</li>
                    <li><b>Độ tin cậy cao:</b> {top['confidence']:.2f} - Kết quả ổn định, ít biến động</li>
                    <li><b>Mức bảo vệ:</b> {ICC_PACKAGES[top['icc_package']]['description']}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="explanation-box">
                <h4>🥇 So sánh Top 3 phương án (giải thích chi tiết):</h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        cols = st.columns(3)
        for idx, col in enumerate(cols):
            if idx < len(result.results):
                row = result.results.iloc[idx]
                with col:
                    medal = ["🥇", "🥈", "🥉"][idx]
                    st.metric(
                        f"{medal} #{idx+1}: {row['company']}",
                        f"{row['icc_package']}",
                        f"${row['estimated_cost']:,.0f}"
                    )
                    st.caption(f"Điểm: {row['score']:.3f} | {row['category']}")
                    st.caption(f"Tin cậy: {row['confidence']:.2f}")

        top3 = result.results.head(3)
        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>📊 Phân tích so sánh Top 3:</h4>
                <ul>
                    <li><b>#1 {top3.iloc[0]['company']} - {top3.iloc[0]['icc_package']}</b>
                        <br>→ Điểm: {top3.iloc[0]['score']:.3f} | Chi phí: ${top3.iloc[0]['estimated_cost']:,.0f}
                        <br>→ Rủi ro khí hậu: {top3.iloc[0]['C6_mean']:.2%} ± {top3.iloc[0]['C6_std']:.2%}
                    </li>
                    <li><b>#2 {top3.iloc[1]['company']} - {top3.iloc[1]['icc_package']}</b>
                        <br>→ Điểm: {top3.iloc[1]['score']:.3f} (kém {top3.iloc[0]['score'] - top3.iloc[1]['score']:.3f})
                        <br>→ Chi phí: ${top3.iloc[1]['estimated_cost']:,.0f} (chênh ${abs(top3.iloc[1]['estimated_cost'] - top3.iloc[0]['estimated_cost']):,.0f})
                    </li>
                    <li><b>#3 {top3.iloc[2]['company']} - {top3.iloc[2]['icc_package']}</b>
                        <br>→ Điểm: {top3.iloc[2]['score']:.3f} (kém {top3.iloc[0]['score'] - top3.iloc[2]['score']:.3f})
                        <br>→ Độ tin cậy: {top3.iloc[2]['confidence']:.2f}
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("📋 Bảng so sánh 15 phương án (đầy đủ)")

        df_display = result.results[["rank", "company", "icc_package", "category",
                                     "estimated_cost", "score", "confidence"]].copy()
        df_display.columns = ["Hạng", "Công ty", "Gói ICC", "Loại", "Chi phí", "Điểm", "Tin cậy"]
        df_display["Chi phí"] = df_display["Chi phí"].apply(lambda x: f"${x:,.0f}")
        df_display = df_display.set_index("Hạng")
        st.dataframe(df_display, use_container_width=True)

        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>💡 Giải thích về 3 loại phương án:</h4>
                <ul>
                    <li><b>💰 Tiết kiệm (ICC C):</b> {ICC_PACKAGES['ICC C']['description']}
                        <br>→ Phí thấp nhất ({ICC_PACKAGES['ICC C']['premium_multiplier']:.0%} baseline)
                        <br>→ Phù hợp: Hàng giá trị thấp, tuyến ngắn, rủi ro thấp
                    </li>
                    <li><b>⚖️ Cân bằng (ICC B):</b> {ICC_PACKAGES['ICC B']['description']}
                        <br>→ Phí trung bình (baseline 100%)
                        <br>→ Phù hợp: Đa số trường hợp, cân bằng chi phí - bảo vệ
                    </li>
                    <li><b>🛡️ An toàn (ICC A):</b> {ICC_PACKAGES['ICC A']['description']}
                        <br>→ Phí cao nhất ({ICC_PACKAGES['ICC A']['premium_multiplier']:.0%} baseline)
                        <br>→ Phù hợp: Hàng giá trị cao, tuyến xa, rủi ro cao
                    </li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        if result.var is not None and result.cvar is not None:
            risk_pct = (result.var / params.cargo_value) * 100
            st.markdown(
                f"""
                <div class="explanation-box">
                    <h4>⚠️ Đánh giá rủi ro tài chính (VaR/CVaR):</h4>
                    <ul>
                        <li><b>VaR 95%:</b> ${result.var:,.0f} ({risk_pct:.1f}% giá trị hàng)
                            <br>→ Tổn thất tối đa ở mức tin cậy 95%
                        </li>
                        <li><b>CVaR 95%:</b> ${result.cvar:,.0f}
                            <br>→ Tổn thất trung bình trong 5% trường hợp xấu nhất
                        </li>
                        <li><b>Nhận định:</b> {'✅ Chấp nhận được - Rủi ro trong ngưỡng kiểm soát' if risk_pct < 10 else '⚠️ Cần xem xét kỹ - Rủi ro cao'}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ==================== CHARTS ====================
        st.markdown("---")
        st.subheader("📊 Biểu đồ phân tích")

        st.markdown("### 📉 Biểu đồ Chi phí – Chất lượng")
        fig_scatter = self.chart_factory.create_cost_benefit_scatter(result.results)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### 🏆 Top 5 Phương án tốt nhất")
        fig_bar = self.chart_factory.create_top_recommendations_bar(result.results)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 📊 So sánh 3 Loại Phương án")
        fig_category = self.chart_factory.create_category_comparison(result.results)
        st.plotly_chart(fig_category, use_container_width=True)

        st.markdown("### 🌦️ Dự báo rủi ro khí hậu tuyến đã chọn")
        fig_forecast = self.chart_factory.create_forecast_chart(
            result.historical, result.forecast, params.route, params.month
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # ==================== PREMIUM VIEW 3.0 ====================
        st.markdown("""
        <style>
        .premium3-wrapper {
            display: flex;
            gap: 32px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 30px;
        }
        .flip-box {
            background: transparent;
            width: 340px;
            height: 430px;
            perspective: 1300px;
        }
        .flip-box-inner {
            position: relative;
            width: 100%;
            height: 100%;
            transition: transform 0.9s;
            transform-style: preserve-3d;
        }
        .flip-box:hover .flip-box-inner {
            transform: rotateY(180deg);
        }
        .flip-box-front,
        .flip-box-back {
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 22px;
            -webkit-backface-visibility: hidden;
            backface-visibility: hidden;
            padding: 26px;
        }
        .flip-box-front {
            background: rgba(0,15,10,0.65);
            border: 1px solid rgba(0,255,153,0.45);
            box-shadow: 0 0 26px rgba(0,255,153,0.25);
            backdrop-filter: blur(18px);
        }
        .flip-box-back {
            background: rgba(0,0,0,0.65);
            border: 1px solid rgba(255,215,0,0.45);
            box-shadow: 0 0 36px rgba(255,215,0,0.25);
            backdrop-filter: blur(20px);
            transform: rotateY(180deg);
        }
        .hero-glow {
            animation: heroPulse 2.8s ease-in-out infinite;
            border-color: rgba(255,230,90,0.85) !important;
            box-shadow: 0 0 40px rgba(255,210,0,0.55),
                        0 0 90px rgba(255,185,0,0.28);
        }
        @keyframes heroPulse {
            0% { box-shadow: 0 0 30px rgba(255,200,0,0.35); }
            100% { box-shadow: 0 0 70px rgba(255,230,130,0.9); }
        }
        .card-title {
            text-align: center;
            font-size: 1.42rem;
            font-weight: 900;
            margin-bottom: 12px;
            color: #aaffea;
        }
        .hero-title {
            color: #ffe680;
            text-shadow: 0 0 14px rgba(255,220,0,0.8);
        }
        .company-avatar {
            width: 72px;
            height: 72px;
            margin: auto;
            border-radius: 999px;
            background-size: cover;
            background-position: center;
            box-shadow: 0 0 18px rgba(0,255,153,0.55),
                        inset 0 0 12px rgba(0,255,153,0.35);
            margin-bottom: 12px;
            animation: holoSpin 6s linear infinite;
        }
        @keyframes holoSpin {
            0% { transform: rotateY(0); }
            100% { transform: rotateY(360deg); }
        }
        .neon-bar {
            height: 10px;
            background: linear-gradient(90deg, #00ffc3, #00e676);
            border-radius: 999px;
            margin: 6px 0;
            box-shadow: 0 0 14px rgba(0,255,153,0.65);
        }
        .drawer {
            margin-top: 12px;
            padding: 10px 14px;
            border-radius: 14px;
            background: rgba(0,0,0,0.45);
            border: 1px solid rgba(0,255,153,0.45);
            color: #e6fff6;
            font-size: 0.92rem;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("## 🪩 Premium View 3.0 — Interactive Flip Edition")

        company_logo = {
            "PVI": "https://i.imgur.com/SzK2e5v.png",
            "Chubb": "https://i.imgur.com/xnPP9kq.png",
            "MIC": "https://i.imgur.com/aCHaHWE.png",
        }

        top3 = result.results.head(3)
        medals = ["🥇", "🥈", "🥉"]
        cols_flip = st.columns(3)

        for i, col in enumerate(cols_flip):
            if i >= len(top3):
                continue
            r = top3.iloc[i]
            avatar = company_logo.get(r["company"], "")
            front_extra = "hero-glow" if i == 0 else ""
            title_extra = "hero-title" if i == 0 else ""

            with col:
                st.markdown(
                    f"""
                    <div class="flip-box">
                        <div class="flip-box-inner">

                            <div class="flip-box-front {front_extra}">
                                <div class="company-avatar" style="background-image:url('{avatar}')"></div>
                                <div class="card-title {title_extra}">
                                    {medals[i]} {r['company']}
                                </div>
                                <div>💰 <b>${r['estimated_cost']:,.0f}</b></div>
                                <div class="neon-bar" style="width:{r['score']*100}%"></div>
                                <div>📊 Điểm: <b>{r['score']:.3f}</b></div>
                                <div class="neon-bar" style="width:{r['confidence']*100}%"></div>
                                <div>🎯 Tin cậy: <b>{r['confidence']:.2f}</b></div>
                            </div>

                            <div class="flip-box-back">
                                <div class="card-title">📘 Chi tiết</div>
                                <div class="drawer">
                                    <b>ICC:</b> {r['icc_package']}<br><br>
                                    🌪 Rủi ro: <b>{r['C6_std']:.2f}</b><br><br>
                                    📌 Nhận định:<br>
                                    - Phù hợp tuyến<br>
                                    - Ổn định chi phí<br>
                                    - Cân bằng rủi ro – giá
                                </div>
                            </div>

                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ======================== FUZZY AHP MODULE ========================
        if params.use_fuzzy:
            st.markdown("---")
            st.subheader("🌿 Fuzzy AHP — Phân tích bất định trọng số (Enterprise Module)")

            st.markdown("""
            <div class="explanation-box">
                <h4>📚 Giải thích về Fuzzy AHP:</h4>
                <ul>
                    <li><b>Mục đích:</b> Xử lý bất định trong đánh giá chuyên gia</li>
                    <li><b>Phương pháp:</b> Chuyển trọng số crisp thành tam giác mờ (Low-Mid-High)</li>
                    <li><b>Defuzzification:</b> Sử dụng phương pháp Centroid để chuyển về crisp</li>
                    <li><b>Ứng dụng:</b> Tăng độ tin cậy kết quả khi chuyên gia không chắc chắn 100%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            fig_fuzzy = fuzzy_chart_premium(result.weights, params.fuzzy_uncertainty)
            st.plotly_chart(fig_fuzzy, use_container_width=True)

            st.subheader("📄 Bảng Low – Mid – High – Centroid (cho NCKH)")
            fuzzy_table = build_fuzzy_table(result.weights, params.fuzzy_uncertainty)
            st.dataframe(fuzzy_table, use_container_width=True)

            most_unc, diff_map = most_uncertain_criterion(result.weights, params.fuzzy_uncertainty)

            st.markdown(
                f"""
                <div style="background:#00331F; padding:15px; border-radius:10px;
                border:2px solid #00FFAA; color:#CCFFE6; font-size:16px; margin-top:0.8rem;">
                🔍 <b>Tiêu chí dao động mạnh nhất (High - Low lớn nhất):</b><br>
                <span style="color:#00FFAA; font-size:20px;"><b>{most_unc}</b></span><br><br>
                💡 <b>Ý nghĩa:</b> Tiêu chí này <b>nhạy cảm nhất</b> khi thay đổi trọng số đầu vào (Fuzzy).<br>
                Mô hình Fuzzy cho thấy tiêu chí này có độ bất định cao,
                nên cần được chuyên gia cân nhắc kỹ khi hiệu chỉnh trọng số.<br><br>
                <b>Giải pháp:</b> Thu thập thêm ý kiến chuyên gia hoặc dữ liệu thực tế để giảm bất định.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("🔥 Heatmap mức dao động Fuzzy (Premium Green)")
            fig_heat = fuzzy_heatmap_premium(diff_map)
            st.plotly_chart(fig_heat, use_container_width=True)

        # ======================== EXPORT ========================
        st.markdown("---")
        st.subheader("📥 Xuất báo cáo")

        col1, col2 = st.columns(2)

        with col1:
            excel_data = self.report_gen.generate_excel(result.results, result.weights)
            st.download_button(
                "📊 Tải Excel",
                data=excel_data,
                file_name=f"riskcast_v53_{params.route.replace(' - ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with col2:
            pdf_data = self.report_gen.generate_pdf(result.results, params, result.var, result.cvar)
            if pdf_data:
                st.download_button(
                    "📄 Tải PDF",
                    data=pdf_data,
                    file_name=f"riskcast_v53_{params.route.replace(' - ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    def run(self):
        self.initialize()
        historical = DataService.load_historical_data()
        params = self.render_sidebar()

        run_btn = st.button("🚀 Chạy phân tích Multi-Package", use_container_width=True)

        if run_btn:
            result = self.analyzer.run_analysis(params, historical)
            self.display_results(result, params)
        else:
            st.info("👈 Nhập thông tin ở sidebar và bấm **“🚀 Chạy phân tích Multi-Package”** để xem kết quả.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    app = StreamlitUI()
    app.run()


if __name__ == "__main__":
    main()
