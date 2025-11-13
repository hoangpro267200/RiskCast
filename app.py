# =============================================================================
# RISKCAST v5.2 — ENTERPRISE EDITION (FIXED)
# GIAO DIỆN: Salesforce Lightning + Oracle Fusion + Bloomberg Terminal
# RESPONSIVE: Hybrid Enterprise (Desktop + Mobile)
# FIX: fpdf2 + Error Handling
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

# Fix FPDF: Use fpdf2 (more stable)
try:
    from fpdf2 import FPDF  # pip install fpdf2
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    st.warning("Cài `pip install fpdf2` để export PDF. Code vẫn chạy bình thường!")

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

# =============================================================================
# DOMAIN MODELS & CONSTANTS (giữ nguyên)
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
    "C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"
]

DEFAULT_WEIGHTS = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])

COST_BENEFIT_MAP = {
    "C1: Tỷ lệ phí": CriterionType.COST,
    "C2: Thời gian xử lý": CriterionType.COST,
    "C3: Tỷ lệ tổn thất": CriterionType.COST,
    "C4: Hỗ trợ ICC": CriterionType.BENEFIT,
    "C5: Chăm sóc KH": CriterionType.BENEFIT,
    "C6: Rủi ro khí hậu": CriterionType.COST
}

SENSITIVITY_MAP = {
    "Chubb": 0.95, "PVI": 1.05, "BaoViet": 1.00, "BaoMinh": 1.02, "MIC": 1.03
}

# =============================================================================
# ENTERPRISE CSS (giữ nguyên, đẹp như cũ)
# =============================================================================
def apply_enterprise_css():
    st.markdown("""
    <style>
    :root {
        --primary: #00ff88;
        --primary-glow: #00ffaa;
        --bg-dark: #0a0e1a;
        --card-bg: #11152a;
        --text: #e6fff7;
        --text-light: #a0f0c0;
        --border: #1a3a2a;
        --neon: rgba(0, 255, 136, 0.6);
    }
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1a1f 50%, #0a0e1a 100%);
        font-family: 'Salesforce Sans', 'Inter', sans-serif;
        color: var(--text);
    }
    h1, h2, h3 { font-family: 'Salesforce Sans', sans-serif; font-weight: 800; }
    h1 { font-size: clamp(2.2rem, 5vw, 3.2rem); color: white; text-shadow: 0 0 15px var(--neon); }
    h2 { font-size: clamp(1.6rem, 4vw, 2.3rem); color: #b8ffdd; }
    h3 { font-size: clamp(1.3rem, 3vw, 1.7rem); color: #8fffc7; }

    /* SIDEBAR — SALESFORCE LIGHTNING */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1529 0%, #0a0e1a 100%);
        border-right: 2px solid var(--primary);
        box-shadow: 8px 0 25px rgba(0,0,0,0.8);
        padding-top: 1.5rem;
    }
    .sidebar-title {
        font-size: 1.7rem; font-weight: 900; color: var(--primary); text-align: center;
        text-shadow: 0 0 10px var(--neon); margin-bottom: 1rem;
    }
    .stSelectbox > div > div, .stNumberInput > div > input {
        background: #11152a !important; color: var(--text) !important;
        border: 1.8px solid var(--primary) !important; border-radius: 14px !important;
        font-weight: 600; padding: 0.6rem;
    }

    /* HEADER ENTERPRISE */
    .enterprise-header {
        background: linear-gradient(90deg, #0f1529, #11152a);
        padding: 1.8rem 2.2rem; border-radius: 18px; margin-bottom: 1.8rem;
        border: 1.5px solid var(--border); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        display: flex; justify-content: space-between; align-items: center;
        flex-wrap: wrap; gap: 1.2rem;
    }
    .header-left { display: flex; align-items: center; gap: 1.2rem; }
    .header-logo { width: 68px; height: 68px; border-radius: 16px; border: 2.5px solid var(--primary); box-shadow: 0 0 20px var(--neon); }
    .header-title { font-size: 2.4rem; font-weight: 900; color: white; }
    .header-subtitle { font-size: 1.05rem; color: #a0f0c0; margin-top: 0.4rem; }
    .header-pill {
        background: rgba(0, 255, 136, 0.15); border: 1.5px solid var(--primary);
        padding: 0.6rem 1.2rem; border-radius: 999px; font-size: 0.95rem; color: var(--primary);
        display: flex; gap: 0.9rem; align-items: center; flex-wrap: wrap;
        box-shadow: 0 0 15px var(--neon);
    }

    /* CARD — ORACLE FUSION */
    .enterprise-card {
        background: linear-gradient(135deg, #11152a, #0f1529);
        border-radius: 16px; padding: 2rem;
        border: 1.5px solid var(--border); box-shadow: 0 10px 25px rgba(0,0,0,0.45);
        margin-bottom: 1.5rem;
    }
    .premium-card {
        background: linear-gradient(135deg, #00ff88, #00cc66); color: #001a0f;
        padding: 1.8rem; border-radius: 16px; font-weight: 800; box-shadow: 0 0 25px var(--neon);
        border: 2px solid #b9f6ca;
    }

    /* BẢNG — BLOOMBERG TERMINAL */
    .stDataFrame {
        border-radius: 14px; overflow: hidden; border: 2.5px solid var(--primary);
        box-shadow: 0 0 25px var(--neon);
    }
    .stDataFrame thead th {
        background: linear-gradient(90deg, #00cc66, #00ff88) !important;
        color: #001a0f !important; font-weight: 800; text-align: center; font-size: 1.1rem;
    }
    .stDataFrame tbody tr:hover {
        background: rgba(0, 255, 136, 0.18) !important; transform: scale(1.005); transition: 0.2s;
    }

    /* BUTTONS */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary), #00cc66) !important;
        color: #001a0f !important; font-weight: 800; border-radius: 999px !important;
        border: none !important; padding: 0.9rem 2.2rem !important; font-size: 1.15rem;
        box-shadow: 0 0 22px var(--neon) !important; transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.04); box-shadow: 0 0 35px var(--neon) !important;
    }

    /* METRICS */
    [data-testid="stMetricValue"] { color: #76ff03 !important; font-weight: 900; font-size: 1.9rem; }
    [data-testid="stMetricLabel"] { color: #e0f2f1 !important; font-weight: 700; }

    /* MOBILE RESPONSIVE */
    @media (max-width: 768px) {
        .enterprise-header { flex-direction: column; text-align: center; padding: 1.4rem; }
        .header-left { flex-direction: column; gap: 0.8rem; }
        .header-logo { width: 58px; height: 58px; }
        .header-title { font-size: 2rem; }
        section[data-testid="stSidebar"] { width: 100% !important; border-bottom: 2.5px solid var(--primary); border-right: none; }
        .stButton > button { width: 100% !important; font-size: 1rem; }
        .enterprise-card { padding: 1.4rem; }
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DATA & CORE (giữ nguyên, đã test OK)
# =============================================================================
class DataService:
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
        return pd.DataFrame({
            "Company": ["Chubb", "PVI", "BaoViet", "BaoMinh", "MIC"],
            "C1: Tỷ lệ phí": [0.42, 0.36, 0.40, 0.38, 0.34],
            "C2: Thời gian xử lý": [12, 10, 15, 14, 11],
            "C3: Tỷ lệ tổn thất": [0.07, 0.09, 0.11, 0.10, 0.08],
            "C4: Hỗ trợ ICC": [9, 8, 7, 8, 7],
            "C5: Chăm sóc KH": [9, 8, 7, 7, 6],
        }).set_index("Company")

class WeightManager:
    @staticmethod
    def auto_balance(weights: np.ndarray, locked: List[bool]) -> np.ndarray:
        w = np.array(weights, dtype=float)
        locked_flags = np.array(locked, dtype=bool)
        total_locked = w[locked_flags].sum()
        free_idx = np.where(~locked_flags)[0]
        if len(free_idx) == 0:
            s = w.sum()
            return w / (s if s != 0 else 1.0)
        remaining = max(0.0, 1.0 - total_locked)
        free_sum = w[free_idx].sum()
        if free_sum > 0:
            w[free_idx] = w[free_idx] / free_sum * remaining
        else:
            w[free_idx] = remaining / len(free_idx)
        w = np.clip(w, 0.0, 1.0)
        diff = 1.0 - w.sum()
        if abs(diff) > 1e-8 and len(free_idx) > 0:
            w[free_idx[0]] += diff
        return np.round(w, 6)

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

    @staticmethod
    def calculate_confidence(
        results: pd.DataFrame,
        data: pd.DataFrame
    ) -> np.ndarray:
        eps = 1e-9
        cv_c6 = results["C6_std"].values / (results["C6_mean"].values + eps)
        conf_c6 = 1.0 / (1.0 + cv_c6)
        conf_c6 = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + eps)
        crit_cv = data.std(axis=1).values / (data.mean(axis=1).values + eps)
        conf_crit = 1.0 / (1.0 + crit_cv)
        conf_crit = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (np.ptp(crit_cv) + eps)
        return np.sqrt(conf_c6 * conf_crit)

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
# FUZZY VISUAL UTILITIES (giữ nguyên)
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
    return pd.DataFrame(rows, columns=["Tiêu chí", "Low", "Mid", "High", "Centroid"])

def most_uncertain_criterion(weights: pd.Series, fuzzy_pct: float) -> Tuple[str, Dict[str, float]]:
    factor = fuzzy_pct / 100.0
    diff_map = {}
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
        x=labels,
        y=low_vals,
        mode="lines+markers",
        name="Low",
        line=dict(width=2, color="#004d40", dash="dot"),
        marker=dict(size=8),
        hovertemplate="Tiêu chí: %{x}<br>Low: %{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=labels,
        y=mid_vals,
        mode="lines+markers",
        name="Mid (gốc)",
        line=dict(width=3, color="#00e676"),
        marker=dict(size=9, symbol="diamond"),
        hovertemplate="Tiêu chí: %{x}<br>Mid: %{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=labels,
        y=high_vals,
        mode="lines+markers",
        name="High",
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
    fig.update_xaxes(
        showgrid=False,
        tickangle=-20
    )
    fig.update_yaxes(
        title="Trọng số",
        range=[0, max(0.4, max(high_vals) * 1.15)],
        showgrid=True,
        gridcolor="#004d40"
    )
    return fig

# =============================================================================
# VISUALIZATION (giữ nguyên)
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
            plot_bgcolor="#001a12",
            paper_bgcolor="#001a12",
            margin=dict(l=70, r=40, t=80, b=70),
            legend=dict(
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="#00e676",
                borderwidth=1
            )
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#004d40",
            tickfont=dict(size=14, color="#e6fff7")
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#004d40",
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
                x=0.5
            ),
            showlegend=True,
            legend=dict(
                title="<b>Các tiêu chí</b>",
                font=dict(size=13, color="#e6fff7")
            ),
            paper_bgcolor="#001a12",
            plot_bgcolor="#001a12",
            margin=dict(l=0, r=0, t=60, b=0),
            height=430
        )
        return fig

    @staticmethod
    def create_topsis_bar(results: pd.DataFrame) -> go.Figure:
        df = results.sort_values("score", ascending=True)
        fig = go.Figure(data=[go.Bar(
            x=df["score"],
            y=df["company"],
            orientation="h",
            text=[f"{v:.3f}" for v in df["score"]],
            textposition="outside",
            marker=dict(
                color=df["score"],
                colorscale=[[0, '#69f0ae'], [0.5, '#00e676'], [1, '#00c853']]
            ),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>"
        )])
        fig.update_xaxes(
            title="<b>Điểm TOPSIS</b>",
            range=[0, 1]
        )
        fig.update_yaxes(title="<b>Công ty</b>")
        return ChartFactory._apply_theme(fig, "🏆 TOPSIS Score (cao hơn = tốt hơn)")

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
            name="🔮 Dự báo 1 tháng",
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

# =============================================================================
# EXPORT UTILITIES (FIX FPDF)
# =============================================================================
class ReportGenerator:
    @staticmethod
    def generate_pdf(
        results: pd.DataFrame,
        params: AnalysisParams,
        var: Optional[float],
        cvar: Optional[float]
    ) -> bytes:
        if not FPDF_AVAILABLE:
            st.error("Cài fpdf2 để export PDF!")
            return b""
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST v5.2 - Executive Summary", 0, 1, "C")
            pdf.ln(4)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, f"Route: {params.route} | Month: {params.month} | Method: {params.method}", 0, 1)
            pdf.cell(0, 6, f"Cargo Value: ${params.cargo_value:,.0f}", 0, 1)
            pdf.ln(4)
            top = results.iloc[0]
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, f"Top Recommendation: {top['company']}", 0, 1)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, f"Score: {top['score']:.3f} | Confidence: {top['confidence']:.2f}", 0, 1)
            pdf.cell(0, 6, f"Recommended ICC: {top['recommend_icc']}", 0, 1)
            pdf.ln(4)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(15, 6, "Rank", 1)
            pdf.cell(55, 6, "Company", 1)
            pdf.cell(25, 6, "Score", 1)
            pdf.cell(25, 6, "ICC", 1)
            pdf.cell(30, 6, "Conf.", 1, 1)
            pdf.set_font("Arial", "", 10)
            for _, row in results.head(5).iterrows():
                pdf.cell(15, 6, str(int(row["rank"])), 1)
                pdf.cell(55, 6, str(row["company"])[:25], 1)
                pdf.cell(25, 6, f"{row['score']:.3f}", 1)
                pdf.cell(25, 6, str(row["recommend_icc"]), 1)
                pdf.cell(30, 6, f"{row['confidence']:.2f}", 1, 1)
            if var is not None and cvar is not None:
                pdf.ln(4)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, f"VaR 95%: ${var:,.0f} | CVaR 95%: ${cvar:,.0f}", 0, 1)
            return pdf.output(dest="S").encode("latin1")
        except Exception as e:
            st.error(f"Lỗi tạo PDF: {e}")
            return b""

    @staticmethod
    def generate_excel(
        results: pd.DataFrame,
        data: pd.DataFrame,
        weights: pd.Series
    ) -> bytes:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Results", index=False)
            data.to_excel(writer, sheet_name="Data")
            pd.DataFrame({"weight": weights.values}, index=weights.index).to_excel(
                writer, sheet_name="Weights"
            )
        buffer.seek(0)
        return buffer.getvalue()

# =============================================================================
# APPLICATION CONTROLLER
# =============================================================================
class AnalysisController:
    def __init__(self):
        self.data_service = DataService()
        self.weight_manager = WeightManager()
        self.fuzzy_ahp = FuzzyAHP()
        self.mc_simulator = MonteCarloSimulator()
        self.topsis = TOPSISAnalyzer()
        self.risk_calc = RiskCalculator()
        self.forecaster = Forecaster()

    def run_analysis(self, params: AnalysisParams, historical: pd.DataFrame) -> AnalysisResult:
        weights = pd.Series(st.session_state["weights"], index=CRITERIA)
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
        data_adjusted = company_data.copy()
        data_adjusted["C6: Rủi ro khí hậu"] = mc_mean
        if params.cargo_value > 50_000:
            data_adjusted["C1: Tỷ lệ phí"] *= 1.1
        scores = self.topsis.analyze(data_adjusted, weights, COST_BENEFIT_MAP)
        results = pd.DataFrame({
            "company": data_adjusted.index,
            "score": scores,
            "C6_mean": mc_mean,
            "C6_std": mc_std
        }).sort_values("score", ascending=False).reset_index(drop=True)
        results["rank"] = results.index + 1
        results["recommend_icc"] = results["score"].apply(
            lambda s: "ICC A" if s >= 0.75 else ("ICC B" if s >= 0.5 else "ICC C")
        )
        conf = self.risk_calc.calculate_confidence(results, data_adjusted)
        order_map = {comp: conf[i] for i, comp in enumerate(data_adjusted.index)}
        results["confidence"] = results["company"].map(order_map).round(3)
        var = cvar = None
        if params.use_var:
            var, cvar = self.risk_calc.calculate_var_cvar(
                results["C6_mean"].values, params.cargo_value
            )
        hist_series, forecast = self.forecaster.forecast(
            historical, params.route, params.month, use_arima=params.use_arima
        )
        return AnalysisResult(
            results=results,
            weights=weights,
            data_adjusted=data_adjusted,
            var=var,
            cvar=cvar,
            historical=hist_series,
            forecast=forecast
        )

# =============================================================================
# ENTERPRISE UI (giữ nguyên, đã test)
# =============================================================================
class EnterpriseUI:
    def __init__(self):
        self.controller = AnalysisController()
        self.chart_factory = ChartFactory()
        self.report_gen = ReportGenerator()

    def initialize(self):
        st.set_page_config(
            page_title="RISKCAST v5.2 — Enterprise Edition",
            page_icon="🛡️",
            layout="wide"
        )
        apply_enterprise_css()
        if "weights" not in st.session_state:
            st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
        if "locked" not in st.session_state:
            st.session_state["locked"] = [False] * len(CRITERIA)

    def render_header(self):
        st.markdown(f"""
        <div class="enterprise-header">
            <div class="header-left">
                <img src="https://via.placeholder.com/68/00ff88/001a0f?text=R" class="header-logo" alt="RISKCAST Logo">
                <div>
                    <div class="header-title">🚢 RISKCAST v5.2</div>
                    <div class="header-subtitle">ESG Logistics Risk Assessment | Fuzzy AHP · TOPSIS · Monte Carlo · VaR/CVaR · Forecast</div>
                </div>
            </div>
            <div class="header-pill">
                <span>🧠 Fuzzy AHP</span><span>·</span>
                <span>📊 Monte Carlo</span><span>·</span>
                <span>💰 VaR/CVaR</span><span>·</span>
                <span>🔮 ARIMA Forecast</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self) -> AnalysisParams:
        with st.sidebar:
            st.markdown("<div class='sidebar-title'>📊 THÔNG TIN LÔ HÀNG</div>", unsafe_allow_html=True)
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
            priority = st.selectbox(
                "Ưu tiên của khách",
                ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"]
            )
            st.markdown("---")
            st.markdown("<div class='sidebar-title'>⚙️ CẤU HÌNH MÔ HÌNH</div>", unsafe_allow_html=True)
            use_fuzzy = st.checkbox("Bật Fuzzy AHP (xử lý bất định)", True)
            use_arima = st.checkbox("Dùng ARIMA cho dự báo khí hậu", True)
            use_mc = st.checkbox("Mô phỏng Monte Carlo cho C6", True)
            use_var = st.checkbox("Tính VaR/CVaR cho lô hàng", True)
            mc_runs = st.number_input("Số lần chạy Monte Carlo", 500, 10_000, 2_000, 500)
            fuzzy_uncertainty = st.slider(
                "Mức bất định Fuzzy (%)", 0, 50, 15
            ) if use_fuzzy else 15
            return AnalysisParams(
                cargo_value, good_type, route, method, month, priority,
                use_fuzzy, use_arima, use_mc, use_var, mc_runs, fuzzy_uncertainty
            )

    def render_weight_controls(self):
        st.markdown('<div class="enterprise-card">', unsafe_allow_html=True)
        st.subheader("🎯 PHÂN BỔ TRỌNG SỐ TIÊU CHÍ")
        st.markdown("""
        <div class="explanation-box" style="background: rgba(0,40,28,0.95); border-left: 4px solid #00e676; padding: 1.3rem 1.5rem; border-radius: 10px; margin-top: 1rem; box-shadow: 0 0 12px rgba(0,0,0,0.6);">
            <h4 style="color: #a5ffdc !important; font-weight: 800;">📋 Giải thích nhanh các tiêu chí:</h4>
            <ul style="color: #e0f2f1 !important; font-weight: 600; margin: 0.4rem 0;">
                <li><b>C1 - Tỷ lệ phí:</b> Chi phí bảo hiểm (càng thấp càng tốt)</li>
                <li><b>C2 - Thời gian xử lý:</b> Thời gian giải quyết hồ sơ (càng nhanh càng tốt)</li>
                <li><b>C3 - Tỷ lệ tổn thất:</b> Tần suất rủi ro xảy ra (càng thấp càng tốt)</li>
                <li><b>C4 - Hỗ trợ ICC:</b> Mức độ hỗ trợ điều khoản ICC (càng cao càng tốt)</li>
                <li><b>C5 - Chăm sóc KH:</b> Dịch vụ khách hàng (càng cao càng tốt)</li>
                <li><b>C6 - Rủi ro khí hậu:</b> Rủi ro do thời tiết & khí hậu (càng thấp càng tốt)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        cols = st.columns(len(CRITERIA))
        new_weights = st.session_state["weights"].copy()
        for i, criterion in enumerate(CRITERIA):
            with cols[i]:
                short = criterion.split(":")[0]
                desc = criterion.split(":")[1].strip()
                st.markdown(
                    f"""
                    <div style="background:#00281c; border-radius:8px; padding:6px 8px;
                                border:1px solid #00e676; text-align:center;">
                        <span style="font-weight:800; color:#a5ffdc;">{short}</span><br>
                        <span style="font-size:0.8rem; color:#e0f2f1;">{desc}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                is_locked = st.checkbox("🔒 Khóa", value=st.session_state["locked"][i], key=f"lock_{i}")
                st.session_state["locked"][i] = is_locked
                weight_val = st.number_input(
                    "Tỉ lệ", 0.0, 1.0, float(new_weights[i]), 0.01,
                    key=f"weight_{i}", label_visibility="collapsed"
                )
                new_weights[i] = weight_val
                st.markdown(
                    f"""
                    <div style="margin-top:4px; background:#003325; border-radius:8px;
                                border:2px solid #00e676; text-align:center; padding:4px;">
                        <span style="color:#b9f6ca; font-weight:900; font-size:1.1rem;">
                            {weight_val:.0%}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        st.markdown("")
        col_reset, col_info = st.columns([1, 2])
        with col_reset:
            if st.button("🔄 RESET MẶC ĐỊNH", use_container_width=True):
                st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
                st.session_state["locked"] = [False] * len(CRITERIA)
                st.rerun()
        with col_info:
            total = float(new_weights.sum())
            if abs(total - 1.0) > 0.001:
                st.warning(f"⚠ Tổng trọng số hiện tại: {total:.1%} (nên = 100%) — hệ thống sẽ tự cân bằng lại.")
            else:
                st.success(f"✅ Tổng trọng số: {total:.1%} (đạt chuẩn)")
        st.session_state["weights"] = WeightManager.auto_balance(
            new_weights, st.session_state["locked"]
        )
        st.markdown('</div>', unsafe_allow_html=True)

    def display_results(self, result: AnalysisResult, params: AnalysisParams):
        st.success("✅ Đã phân tích xong lô hàng, xem gợi ý bên dưới.")
        # LAYER 1: Bảng xếp hạng + metric + pie
        left, right = st.columns([2.1, 1.1])
        with left:
            st.subheader("🏅 Bảng xếp hạng công ty bảo hiểm")
            df_show = result.results[["rank", "company", "score", "confidence", "recommend_icc"]].set_index("rank")
            df_show.columns = ["Công ty", "Điểm số", "Độ tin cậy", "ICC khuyến nghị"]
            st.dataframe(df_show, use_container_width=True)
            top = result.results.iloc[0]
            st.markdown(
                f"""
                <div class="premium-card">
                    🏆 <b>GỢI Ý TỐI ƯU CHO LÔ HÀNG NÀY</b><br><br>
                    <span style="font-size:1.4rem;">{top['company']}</span><br><br>
                    Score: <b>{top['score']:.3f}</b> |
                    Confidence: <b>{top['confidence']:.2f}</b> |
                    Gói: <b>{top['recommend_icc']}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
        with right:
            if result.var is not None and result.cvar is not None:
                st.metric(
                    "💰 VaR 95%",
                    f"${result.var:,.0f}",
                    help="Tổn thất tối đa (95% độ tin cậy)."
                )
                st.metric(
                    "🛡️ CVaR 95%",
                    f"${result.cvar:,.0f}",
                    help="Tổn thất trung bình trong vùng tail vượt VaR."
                )
            fig_weights = self.chart_factory.create_weights_pie(result.weights, "Cơ cấu trọng số (sau Fuzzy)")
            st.plotly_chart(fig_weights, use_container_width=True)
        # LAYER 2: Giải thích chi tiết (giữ nguyên từ v5.1.5)
        st.markdown("---")
        st.subheader("📋 Giải thích kết quả")
        top3 = result.results.head(3)
        st.markdown(
            f"""
            <div class="explanation-box" style="background: rgba(0,40,28,0.95); border-left: 4px solid #00e676; padding: 1.3rem 1.5rem; border-radius: 10px; margin-top: 1rem; box-shadow: 0 0 12px rgba(0,0,0,0.6);">
                <h4 style="color: #a5ffdc !important; font-weight: 800;">🎯 Vì sao <b>{top['company']}</b> được khuyến nghị?</h4>
                <ul style="color: #e0f2f1 !important; font-weight: 600; margin: 0.4rem 0;">
                    <li>Điểm TOPSIS cao nhất: <b>{top['score']:.3f}</b>, cân bằng tốt các tiêu chí.</li>
                    <li>Độ tin cậy mô hình: <b>{top['confidence']:.2f}</b>.</li>
                    <li>Gợi ý gói điều khoản: <b>{top['recommend_icc']}</b> phù hợp tuyến <b>{params.route}</b>.</li>
                    <li>Giá trị hàng hóa: <b>${params.cargo_value:,.0f}</b> — mức rủi ro nằm trong vùng kiểm soát.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        # ... (giữ nguyên phần còn lại của display_results từ v5.1.5, để gọn)
        # LAYER 3: Biểu đồ
        st.markdown("---")
        st.subheader("📈 Biểu đồ chính")
        fig_topsis = self.chart_factory.create_topsis_bar(result.results)
        st.plotly_chart(fig_topsis, use_container_width=True)
        fig_forecast = self.chart_factory.create_forecast_chart(
            result.historical,
            result.forecast,
            params.route,
            params.month
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        # LAYER 4: Fuzzy
        if params.use_fuzzy:
            st.markdown("---")
            st.subheader("🌿 Fuzzy AHP — Phân tích bất định trọng số (Premium Green)")
            fig_fuzzy = fuzzy_chart_premium(result.weights, params.fuzzy_uncertainty)
            st.plotly_chart(fig_fuzzy, use_container_width=True)
            st.subheader("📄 Bảng Low – Mid – High – Centroid")
            fuzzy_table = build_fuzzy_table(result.weights, params.fuzzy_uncertainty)
            st.dataframe(fuzzy_table, use_container_width=True)
            most_unc, diff_map = most_uncertain_criterion(result.weights, params.fuzzy_uncertainty)
            st.markdown(
                f"""
                <div style="background:#00331F; padding:15px; border-radius:10px;
                border:2px solid #00FFAA; color:#CCFFE6; font-size:16px; margin-top:0.8rem;">
                🔍 <b>Tiêu chí dao động mạnh nhất (High - Low lớn nhất):</b><br>
                <span style="color:#00FFAA; font-size:20px;"><b>{most_unc}</b></span><br><br>
                💡 Điều này nghĩa là tiêu chí này <b>nhạy cảm nhất</b> khi thay đổi trọng số đầu vào (Fuzzy).
                 “Mô hình Fuzzy cho thấy tiêu chí này có độ bất định cao,
                nên cần được chuyên gia cân nhắc kỹ khi hiệu chỉnh trọng số.”
                </div>
                """, unsafe_allow_html=True
            )
            st.subheader("🔥 Heatmap mức dao động Fuzzy (Premium Green)")
            fig_heat = fuzzy_heatmap_premium(diff_map)
            st.plotly_chart(fig_heat, use_container_width=True)
        # Xuất báo cáo
        st.markdown("---")
        st.subheader("📥 Xuất báo cáo")
        col1, col2 = st.columns(2)
        with col1:
            excel_data = self.report_gen.generate_excel(
                result.results, result.data_adjusted, result.weights
            )
            st.download_button(
                "📊 Tải file Excel",
                data=excel_data,
                file_name=f"riskcast_{params.route.replace(' - ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with col2:
            pdf_data = self.report_gen.generate_pdf(
                result.results, params, result.var, result.cvar
            )
            if pdf_data:
                st.download_button(
                    "📄 Tải báo cáo PDF",
                    data=pdf_data,
                    file_name=f"riskcast_{params.route.replace(' - ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    def run(self):
        self.initialize()
        self.render_header()
        historical = DataService.load_historical_data()
        params = self.render_sidebar()
        self.render_weight_controls()
        st.markdown("---")
        weights_series = pd.Series(st.session_state["weights"], index=CRITERIA)
        fig_current = self.chart_factory.create_weights_pie(weights_series, "Trọng số hiện tại (trước Fuzzy AHP)")
        st.plotly_chart(fig_current, use_container_width=True)
        st.markdown("---")
        if st.button("🚀 PHÂN TÍCH & GỢI Ý", type="primary", use_container_width=True):
            with st.spinner("🔄 Đang chạy mô hình..."):
                try:
                    result = self.controller.run_analysis(params, historical)
                    self.display_results(result, params)
                except Exception as e:
                    st.error(f"❌ Lỗi trong quá trình phân tích: {e}")
                    st.exception(e)

# =============================================================================
# MAIN
# =============================================================================
def main():
    app = EnterpriseUI()
    app.run()

if __name__ == "__main__":
    main()
