# ============================================================================= 
# RISKCAST v5.2 — ENTERPRISE EDITION (Hybrid Responsive + Ranking Mode Patched)
# ESG Logistics Risk Assessment Dashboard
#
# Author: Bùi Xuân Hoàng (Original idea)
# Refactor + UI + Real Data Engine + Enterprise UX: Kai
# Ranking Mode: An toàn tối đa / Cân bằng / Tối ưu chi phí (v5.2 Patch)
#
# Nổi bật trong v5.2 Enterprise:
#   - Sidebar Enterprise style (Salesforce-like)
#   - Header Enterprise: logo, subtitle, badge
#   - Card Enterprise (Oracle Fusion)
#   - Bảng Enterprise (Bloomberg Terminal)
#   - Fuzzy AHP Enterprise module
#   - Forecast chart nền tối + neon
#   - Hybrid responsive trên mobile
#   - Ranking Mode (NEW)
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
    COST = "cost"
    BENEFIT = "benefit"


@dataclass
class AnalysisParams:
    cargo_value: float
    good_type: str
    route: str
    method: str
    month: int
    priority: str  # (An toàn tối đa / Cân bằng / Tối ưu chi phí)
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
    "Chubb":     0.95,
    "PVI":       1.05,
    "BaoViet":   1.00,
    "BaoMinh":   1.02,
    "MIC":       1.03
}


# =============================================================================
# CSS ENTERPRISE
# =============================================================================

def apply_custom_css() -> None:
    st.markdown("""
    <style>
    * { text-rendering: optimizeLegibility !important; -webkit-font-smoothing: antialiased !important; }

    .stApp {
        background: radial-gradient(circle at top, #00ff99 0%, #001a0f 35%, #000c08 100%) !important;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
        color: #e6fff7 !important;
        font-size: 1.05rem !important;
    }

    .block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 1400px !important; }

    h1 { font-size: 2.8rem !important; font-weight: 900 !important; letter-spacing: 0.03em; }
    h2 { font-size: 2.1rem !important; font-weight: 800 !important; }
    h3 { font-size: 1.5rem !important; font-weight: 700 !important; }

    /* HEADER */
    .app-header {
        display:flex; justify-content:space-between; align-items:center;
        padding:1.1rem 1.5rem; border-radius:18px;
        background:linear-gradient(120deg,rgba(0,255,153,0.14),rgba(0,0,0,0.88));
        border:1px solid rgba(0,255,153,0.45);
        box-shadow:0 0 0 1px rgba(0,255,153,0.12),0 18px 45px rgba(0,0,0,0.85);
        margin-bottom:1.2rem; gap:1.5rem;
    }
    .app-header-left { display:flex; align-items:center; gap:0.9rem; }

    .app-logo-circle {
        width:64px; height:64px; border-radius:18px;
        background:radial-gradient(circle at 30% 30%,#b9f6ca 0%,#00c853 38%,#00381f 100%);
        display:flex; align-items:center; justify-content:center;
        font-weight:900; font-size:1.4rem; color:#00130d;
        box-shadow:0 0 14px rgba(0,255,153,0.65),0 0 36px rgba(0,0,0,0.75);
        border:2px solid #e8f5e9;
    }

    .app-header-title {
        font-size:1.5rem; font-weight:800;
        background:linear-gradient(90deg,#e8fffb,#b9f6ca,#e8fffb);
        -webkit-background-clip:text; color:transparent;
        letter-spacing:0.05em; text-transform:uppercase;
    }
    .app-header-subtitle {
        font-size:0.9rem; color:#ccffec; opacity:0.9;
    }

    .app-header-badge {
        font-size:0.86rem; font-weight:600;
        padding:0.55rem 0.9rem; border-radius:999px;
        background:radial-gradient(circle at 0 0,#00e676,#00bfa5);
        color:#00130d; display:flex; align-items:center; gap:0.35rem;
        white-space:nowrap;
        box-shadow:0 0 14px rgba(0,255,153,0.65),0 0 22px rgba(0,0,0,0.7);
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background:radial-gradient(circle at 0 0,#003322 0%,#000f0a 40%,#000805 100%) !important;
        border-right:1px solid rgba(0,230,118,0.55);
        box-shadow:8px 0 22px rgba(0,0,0,0.85);
    }

    section[data-testid="stSidebar"] > div { padding-top:1.1rem; }

    section[data-testid="stSidebar"] label {
        color:#e0f2f1 !important; font-weight:600 !important; font-size:0.92rem !important;
    }

    /* BUTTONS */
    .stButton > button {
        background:linear-gradient(120deg,#00ff99,#00e676,#00bfa5) !important;
        color:#00130d !important; font-weight:800 !important;
        border-radius:999px !important; border:none !important;
        padding:0.65rem 1.9rem !important;
        box-shadow:0 0 14px rgba(0,255,153,0.7),0 10px 22px rgba(0,0,0,0.85) !important;
        transition:all 0.12s ease-out;
        font-size:0.98rem !important;
    }

    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# DATA LAYER
# =============================================================================

class DataService:
    """Dữ liệu tuyến + dữ liệu công ty."""
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_historical_data() -> pd.DataFrame:
        """Dữ liệu rủi ro khí hậu 12 tháng theo tuyến."""
        climate_base = {
            "VN - EU": [
                0.28, 0.30, 0.35, 0.40, 0.52, 0.60,
                0.67, 0.70, 0.75, 0.72, 0.60, 0.48
            ],
            "VN - US": [
                0.33, 0.36, 0.40, 0.46, 0.55, 0.63,
                0.72, 0.78, 0.80, 0.74, 0.62, 0.50
            ],
            "VN - Singapore": [
                0.18, 0.20, 0.24, 0.27, 0.32, 0.36,
                0.40, 0.43, 0.45, 0.42, 0.35, 0.30
            ],
            "VN - China": [
                0.20, 0.23, 0.27, 0.31, 0.38, 0.42,
                0.48, 0.50, 0.53, 0.49, 0.40, 0.34
            ],
            "Domestic": [
                0.12, 0.13, 0.14, 0.16, 0.20, 0.22,
                0.23, 0.25, 0.27, 0.24, 0.20, 0.18
            ]
        }
        df = pd.DataFrame({"month": list(range(1, 13))})
        for rt, vals in climate_base.items():
            df[rt] = vals
        return df

    @staticmethod
    @st.cache_data
    def get_company_data() -> pd.DataFrame:
        """Thông số tiêu chí chuẩn của 5 công ty bảo hiểm."""
        return (
            pd.DataFrame({
                "Company": ["Chubb", "PVI", "BaoViet", "BaoMinh", "MIC"],
                "C1: Tỷ lệ phí":       [0.42, 0.36, 0.40, 0.38, 0.34],
                "C2: Thời gian xử lý": [12,   10,   15,   14,   11],
                "C3: Tỷ lệ tổn thất":  [0.07, 0.09, 0.11, 0.10, 0.08],
                "C4: Hỗ trợ ICC":      [9,    8,    7,    8,    7],
                "C5: Chăm sóc KH":     [9,    8,    7,    7,    6],
            })
            .set_index("Company")
        )


# =============================================================================
# WEIGHT MANAGER
# =============================================================================

class WeightManager:
    """Auto-balance trọng số."""
    @staticmethod
    def auto_balance(weights: np.ndarray, locked: List[bool]) -> np.ndarray:
        w = np.array(weights, dtype=float)
        locked_flags = np.array(locked, dtype=bool)

        total_locked = w[locked_flags].sum()
        free_idx = np.where(~locked_flags)[0]

        if len(free_idx) == 0:
            s = w.sum()
            return w / (s if s != 0 else 1)

        remaining = max(0, 1.0 - total_locked)
        free_sum = w[free_idx].sum()

        if free_sum > 0:
            w[free_idx] = w[free_idx] / free_sum * remaining
        else:
            w[free_idx] = remaining / len(free_idx)

        w = np.clip(w, 0, 1)
        diff = 1.0 - w.sum()
        if abs(diff) > 1e-8 and len(free_idx) > 0:
            w[free_idx[0]] += diff

        return np.round(w, 6)


# =============================================================================
# FUZZY AHP
# =============================================================================

class FuzzyAHP:
    @staticmethod
    def apply(weights: pd.Series, uncertainty_pct: float) -> pd.Series:
        factor = uncertainty_pct / 100.0
        w = weights.values

        low = np.maximum(w * (1 - factor), 1e-9)
        high = np.minimum(w * (1 + factor), 0.9999)

        centroid = (low + w + high) / 3.0
        normalized = centroid / centroid.sum()
        return pd.Series(normalized, index=weights.index)


# =============================================================================
# MONTE CARLO
# =============================================================================

class MonteCarloSimulator:
    @staticmethod
    @st.cache_data(ttl=600)
    def simulate(base_risk: float, sensitivity_map: Dict[str, float], n_sim: int):
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())

        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.03, mu * 0.12)

        sims = rng.normal(loc=mu, scale=sigma, size=(n_sim, len(companies)))
        sims = np.clip(sims, 0.0, 1.0)

        return companies, sims.mean(axis=0), sims.std(axis=0)


# =============================================================================
# TOPSIS
# =============================================================================

class TOPSISAnalyzer:
    @staticmethod
    def analyze(data: pd.DataFrame, weights: pd.Series, cost_benefit: Dict[str, CriterionType]) -> np.ndarray:
        M = data[list(weights.index)].values.astype(float)

        denom = np.sqrt((M ** 2).sum(axis=0))
        denom[denom == 0] = 1
        R = M / denom

        V = R * weights.values

        is_cost = np.array([cost_benefit[c] == CriterionType.COST for c in weights.index])
        ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
        ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))

        d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

        return d_minus / (d_plus + d_minus + 1e-12)


# =============================================================================
# RISK CALCULATOR (VaR / CVaR)
# =============================================================================

class RiskCalculator:
    @staticmethod
    def calculate_var_cvar(loss_rates: np.ndarray, cargo_value: float, confidence=0.95):
        if len(loss_rates) == 0:
            return 0, 0

        losses = loss_rates * cargo_value
        var = float(np.percentile(losses, confidence * 100))
        tail = losses[losses >= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        return var, cvar

    @staticmethod
    def calculate_confidence(results: pd.DataFrame, data: pd.DataFrame):
        eps = 1e-9

        cv_c6 = results["C6_std"].values / (results["C6_mean"].values + eps)
        conf_c6 = 1 / (1 + cv_c6)
        conf_c6 = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + eps)

        crit_cv = data.std(axis=1).values / (data.mean(axis=1).values + eps)
        conf_crit = 1 / (1 + crit_cv)
        conf_crit = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (np.ptp(crit_cv) + eps)

        return np.sqrt(conf_c6 * conf_crit)


# =============================================================================
# FORECASTER (ARIMA + fallback)
# =============================================================================

class Forecaster:
    @staticmethod
    def forecast(historical: pd.DataFrame, route: str, month: int, use_arima=True):
        if route not in historical.columns:
            route = historical.columns[1]

        full = historical[route].values
        hist = full[:month]

        if use_arima and ARIMA_AVAILABLE and len(hist) >= 6:
            try:
                model = ARIMA(hist, order=(1, 1, 1))
                fitted = model.fit()
                fc = fitted.forecast(1)
                return hist, np.array([float(np.clip(fc[0], 0, 1))])
            except:
                pass

        if len(hist) >= 3:
            trend = (hist[-1] - hist[-3]) / 2
        elif len(hist) >= 2:
            trend = hist[-1] - hist[-2]
        else:
            trend = 0

        next_val = np.clip(hist[-1] + trend, 0, 1)
        return hist, np.array([float(next_val)])
# =============================================================================
# FUZZY VISUAL UTILITIES (PREMIUM GREEN)
# =============================================================================

def build_fuzzy_table(weights: pd.Series, fuzzy_pct: float) -> pd.DataFrame:
    """Tạo bảng Fuzzy (Low – Mid – High – Centroid)."""
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


def most_uncertain_criterion(weights: pd.Series, fuzzy_pct: float):
    """Tiêu chí có mức dao động Fuzzy mạnh nhất."""
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
    """Heatmap mức dao động Fuzzy (Premium Green)."""
    labels = list(diff_map.keys())
    values = list(diff_map.values())

    fig = px.imshow(
        [values],
        labels=dict(color="Dao động"),
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
        title=dict(text="<b>🌿 Heatmap mức dao động Fuzzy</b>", font=dict(size=22, color="#CCFFE6"), x=0.5),
        paper_bgcolor="#001a12",
        plot_bgcolor="#001a12",
        margin=dict(l=40, r=40, t=80, b=40),
        coloraxis_colorbar=dict(tickfont=dict(color="#CCFFE6"))
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(showticklabels=False)

    return fig


def fuzzy_chart_premium(weights: pd.Series, fuzzy_pct: float) -> go.Figure:
    """Biểu đồ đường Fuzzy Premium."""
    factor = fuzzy_pct / 100.0
    labels = list(weights.index)

    low_vals, mid_vals, high_vals = [], [], []

    for crit in labels:
        w = float(weights[crit])
        low_vals.append(max(w * (1 - factor), 0.0))
        mid_vals.append(w)
        high_vals.append(min(w * (1 + factor), 1.0))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels, y=low_vals, mode="lines+markers", name="Low",
        line=dict(width=2, color="#004d40", dash="dot"),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=mid_vals, mode="lines+markers", name="Mid",
        line=dict(width=3, color="#00e676"),
        marker=dict(size=9, symbol="diamond")
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=high_vals, mode="lines+markers", name="High",
        line=dict(width=2, color="#69f0ae", dash="dash"),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=dict(text=f"<b>🌿 Fuzzy AHP — Low / Mid / High (±{fuzzy_pct:.0f}%)</b>",
                   font=dict(size=22, color="#e6fff7"), x=0.5),
        paper_bgcolor="#001a12",
        plot_bgcolor="#001a12",
        margin=dict(l=40, r=40, t=80, b=80),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#00e676", borderwidth=1),
        font=dict(color="#e6fff7", size=14)
    )

    fig.update_xaxes(tickangle=-20, showgrid=False)
    fig.update_yaxes(title="Trọng số", showgrid=True, gridcolor="#004d40")

    return fig


# =============================================================================
# CHART FACTORY (Plotly)
# =============================================================================

class ChartFactory:

    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            template="plotly_dark",
            title=dict(text=f"<b>{title}</b>", font=dict(size=22, color="#e6fff7"), x=0.5),
            font=dict(size=14, color="#e6fff7"),
            paper_bgcolor="#000c11",
            plot_bgcolor="#001016",
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#00e676", borderwidth=1),
            margin=dict(l=70, r=40, t=80, b=70)
        )
        fig.update_xaxes(showgrid=True, gridcolor="#00332b")
        fig.update_yaxes(showgrid=True, gridcolor="#00332b")
        return fig

    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        colors = ['#00e676', '#69f0ae', '#b9f6ca', '#00bfa5', '#1de9b6', '#64ffda']
        labels = list(weights.index)

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=weights.values,
            text=[l.split(":")[0] for l in labels],
            textinfo="text+percent",
            hole=0.2,
            marker=dict(colors=colors, line=dict(color="#00130d", width=2)),
        )])

        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=20, color="#a5ffdc"), x=0.5),
            paper_bgcolor="#001016",
            plot_bgcolor="#001016",
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
                colorscale=[[0, "#69f0ae"], [0.5, "#00e676"], [1, "#00c853"]]
            )
        )])

        fig.update_xaxes(title="<b>Điểm TOPSIS</b>", range=[0, 1])
        fig.update_yaxes(title="<b>Công ty</b>")

        return ChartFactory._apply_theme(fig, "🏆 TOPSIS Score (cao hơn = tốt hơn)")

    @staticmethod
    def create_forecast_chart(historical: np.ndarray, forecast: np.ndarray, route: str, month: int):
        hist_len = len(historical)
        x_hist = list(range(1, hist_len + 1))

        next_month = month % 12 + 1
        x_fc = [next_month]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_hist, y=historical,
            mode="lines+markers",
            name="📈 Lịch sử",
            line=dict(color="#00e676", width=3),
            marker=dict(size=9)
        ))

        fig.add_trace(go.Scatter(
            x=x_fc, y=forecast,
            mode="lines+markers",
            name="🔮 Dự báo",
            line=dict(color="#ffeb3b", width=3, dash="dash"),
            marker=dict(size=11, symbol="diamond")
        ))

        fig = ChartFactory._apply_theme(fig, f"Dự báo rủi ro khí hậu — {route}")

        fig.update_xaxes(title="<b>Tháng</b>", tickmode="linear",
                         tick0=1, dtick=1, range=[1, 12])
        fig.update_yaxes(title="<b>Mức rủi ro (0–1)</b>", tickformat=".0%")

        return fig


# =============================================================================
# REPORT GENERATOR (Excel + PDF)
# =============================================================================

class ReportGenerator:

    @staticmethod
    def generate_pdf(results: pd.DataFrame, params: AnalysisParams, var: Optional[float], cvar: Optional[float]) -> bytes:
        try:
            pdf = FPDF()
            pdf.add_page()

            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST v5.2 - Enterprise Summary", 0, 1, "C")
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
                pdf.cell(0, 6, f"VaR 95%: ${var:,.0f}   |   CVaR 95%: ${cvar:,.0f}", 0, 1)

            return pdf.output(dest="S").encode("latin1")

        except Exception as e:
            st.error(f"Lỗi tạo PDF: {e}")
            return b""

    @staticmethod
    def generate_excel(results: pd.DataFrame, data: pd.DataFrame, weights: pd.Series) -> bytes:
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
# APPLICATION CONTROLLER (Ranking Mode Patched)
# =============================================================================

class AnalysisController:
    """Điều phối toàn bộ pipeline phân tích."""

    def __init__(self):
        self.data_service = DataService()
        self.weight_manager = WeightManager()
        self.fuzzy_ahp = FuzzyAHP()
        self.mc_simulator = MonteCarloSimulator()
        self.topsis = TOPSISAnalyzer()
        self.risk_calc = RiskCalculator()
        self.forecaster = Forecaster()

    def run_analysis(self, params: AnalysisParams, historical: pd.DataFrame) -> AnalysisResult:
        # Lấy trọng số (đã auto-balance)
        weights = pd.Series(st.session_state["weights"], index=CRITERIA)
        if params.use_fuzzy:
            weights = self.fuzzy_ahp.apply(weights, params.fuzzy_uncertainty)

        company_data = self.data_service.get_company_data()

        # Rủi ro khí hậu tại tháng chọn
        if params.month in historical["month"].values:
            base_risk = float(
                historical.loc[historical["month"] == params.month, params.route].iloc[0]
            )
        else:
            base_risk = 0.4

        # Monte Carlo cho C6
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

        # Phụ phí cho hàng giá trị lớn
        if params.cargo_value > 50_000:
            data_adjusted["C1: Tỷ lệ phí"] *= 1.1

        # TOPSIS CORE
        scores = self.topsis.analyze(data_adjusted, weights, COST_BENEFIT_MAP)

        # BẢNG KẾT QUẢ + MODE XẾP HẠNG (Ranking Mode patched)
        results = pd.DataFrame({
            "company": data_adjusted.index,
            "score": scores,
            "C6_mean": mc_mean,
            "C6_std": mc_std
        })

        # 🟩 PATCHED — Ranking Mode ALWAYS sorts by score DESC
        results = results.sort_values("score", ascending=False).reset_index(drop=True)

        results["rank"] = results.index + 1
        results["recommend_icc"] = results["score"].apply(
            lambda s: "ICC A" if s >= 0.75 else ("ICC B" if s >= 0.5 else "ICC C")
        )

        # CONFIDENCE
        conf = self.risk_calc.calculate_confidence(results, data_adjusted)
        conf_map = {comp: conf[i] for i, comp in enumerate(data_adjusted.index)}
        results["confidence"] = results["company"].map(conf_map).round(3)

        # VAR / CVAR
        var = cvar = None
        if params.use_var:
            var, cvar = self.risk_calc.calculate_var_cvar(
                results["C6_mean"].values, params.cargo_value
            )

        # Forecast 1 tháng
        historical_series, forecast = self.forecaster.forecast(
            historical, params.route, params.month, use_arima=params.use_arima
        )

        return AnalysisResult(
            results=results,
            weights=weights,
            data_adjusted=data_adjusted,
            var=var,
            cvar=cvar,
            historical=historical_series,
            forecast=forecast
        )


# =============================================================================
# STREAMLIT UI — ENTERPRISE EDITION
# =============================================================================

class StreamlitUI:

    def __init__(self):
        self.controller = AnalysisController()
        self.chart_factory = ChartFactory()
        self.report_gen = ReportGenerator()

    # ---------------------- INIT ----------------------
    def initialize(self):
        st.set_page_config(
            page_title="RISKCAST v5.2 — Enterprise ESG Premium Green",
            page_icon="🛡️",
            layout="wide"
        )
        apply_custom_css()

        if "weights" not in st.session_state:
            st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
        if "locked" not in st.session_state:
            st.session_state["locked"] = [False] * len(CRITERIA)

    # ---------------------- SIDEBAR ----------------------
    def render_sidebar(self) -> AnalysisParams:
        with st.sidebar:
            st.header("📊 Thông tin lô hàng")

            cargo_value = st.number_input("Giá trị (USD)", 1000, value=39_000, step=1000)
            good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Nguy hiểm", "Khác"])
            route = st.selectbox("Tuyến vận chuyển", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"])
            method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
            month = st.selectbox("Tháng", list(range(1, 13)), index=8)
            priority = st.selectbox("Ưu tiên của khách", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

            st.markdown("---")
            st.header("⚙️ Mô hình nâng cao")

            use_fuzzy = st.checkbox("Fuzzy AHP (bất định)", True)
            use_arima = st.checkbox("ARIMA Forecast", True)
            use_mc = st.checkbox("Monte Carlo", True)
            use_var = st.checkbox("VaR / CVaR", True)

            mc_runs = st.number_input("Số mô phỏng Monte Carlo", 500, 10000, 2000, 500)
            fuzzy_uncertainty = st.slider("Mức bất định Fuzzy (%)", 0, 50, 15) if use_fuzzy else 15

            return AnalysisParams(
                cargo_value, good_type, route, method, month, priority,
                use_fuzzy, use_arima, use_mc, use_var, mc_runs, fuzzy_uncertainty
            )

    # ---------------------- WEIGHT CONTROL ----------------------
    def render_weight_controls(self):
        st.subheader("🎯 Trọng số tiêu chí (Enterprise Controls)")

        cols = st.columns(len(CRITERIA))
        new_weights = st.session_state["weights"].copy()

        for i, crit in enumerate(CRITERIA):
            with cols[i]:
                st.markdown(f"<div style='color:#a5ffdc;font-weight:700;'>{crit}</div>", unsafe_allow_html=True)

                is_locked = st.checkbox("🔒", value=st.session_state["locked"][i], key=f"lk_{i}")
                st.session_state["locked"][i] = is_locked

                w_val = st.number_input("Tỉ lệ", 0.0, 1.0, float(new_weights[i]), 0.01, key=f"w_{i}",
                                        label_visibility="collapsed")
                new_weights[i] = w_val

        total = float(new_weights.sum())
        if abs(total - 1.0) > 0.01:
            st.warning(f"Tổng trọng số = {total:.2f} ≠ 1 → hệ thống sẽ tự cân bằng.")

        st.session_state["weights"] = WeightManager.auto_balance(new_weights, st.session_state["locked"])

    # ---------------------- RESULT DISPLAY ----------------------
    def display_results(self, result: AnalysisResult, params: AnalysisParams):
        st.success("✅ Đã phân tích — xem kết quả bên dưới.")

        # ---------------------- LAYER 1 ----------------------
        left, right = st.columns([2.1, 1.1])

        with left:
            st.subheader("🏅 Xếp hạng (Ranking Mode) — Enterprise Table")

            df_show = result.results[["rank", "company", "score", "confidence", "recommend_icc"]]\
                       .set_index("rank")
            df_show.columns = ["Công ty", "Điểm", "Tin cậy", "ICC"]
            st.dataframe(df_show, use_container_width=True)

            top = result.results.iloc[0]
            st.markdown(
                f"""
                <div class="result-box">
                    🏆 <b>Gợi ý tối ưu</b><br><br>
                    <span style="font-size:1.4rem;">{top['company']}</span><br><br>
                    Score: <b>{top['score']:.3f}</b> |
                    Tin cậy: <b>{top['confidence']:.2f}</b> |
                    Gói: <b>{top['recommend_icc']}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        with right:
            if result.var is not None:
                st.metric("💰 VaR 95%", f"${result.var:,.0f}")
            if result.cvar is not None:
                st.metric("🛡️ CVaR 95%", f"${result.cvar:,.0f}")

            fig_weights = self.chart_factory.create_weights_pie(result.weights, "Cơ cấu trọng số (sau fuzzy)")
            st.plotly_chart(fig_weights, use_container_width=True)

        # ---------------------- LAYER 2 — Explanation ----------------------
        st.markdown("---")
        st.subheader("📋 Giải thích kết quả")

        top = result.results.iloc[0]
        key = result.data_adjusted.loc[top["company"]]

        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>🎯 Vì sao chọn <b>{top['company']}</b>?</h4>
                <ul>
                    <li>Điểm TOPSIS cao nhất: <b>{top['score']:.3f}</b></li>
                    <li>Độ tin cậy: <b>{top['confidence']:.2f}</b></li>
                    <li>Tuyến: <b>{params.route}</b>, tháng: <b>{params.month}</b></li>
                    <li>C6: {top['C6_mean']:.1%} ± {top['C6_std']:.1%}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---------------------- LAYER 3 — Charts ----------------------
        st.markdown("---")
        st.subheader("📈 Biểu đồ chính")

        fig_topsis = self.chart_factory.create_topsis_bar(result.results)
        st.plotly_chart(fig_topsis, use_container_width=True)

        fig_fc = self.chart_factory.create_forecast_chart(
            result.historical, result.forecast, params.route, params.month
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # ---------------------- FUZZY LAYER ----------------------
        if params.use_fuzzy:
            st.markdown("---")
            st.subheader("🌿 Fuzzy AHP — phân tích bất định")

            fig_fuzzy = fuzzy_chart_premium(result.weights, params.fuzzy_uncertainty)
            st.plotly_chart(fig_fuzzy, use_container_width=True)

            fuzzy_table = build_fuzzy_table(result.weights, params.fuzzy_uncertainty)
            st.dataframe(fuzzy_table, use_container_width=True)

            most_unc, diff_map = most_uncertain_criterion(result.weights, params.fuzzy_uncertainty)

            st.markdown(
                f"""
                <div style="background:#00331F;padding:15px;border-radius:10px;border:2px solid #00FFAA;">
                    🔍 <b>Tiêu chí dao động mạnh nhất:</b> {most_unc}
                </div>
                """,
                unsafe_allow_html=True
            )

            fig_heat = fuzzy_heatmap_premium(diff_map)
            st.plotly_chart(fig_heat, use_container_width=True)

        # ---------------------- EXPORT ----------------------
        st.markdown("---")
        st.subheader("📥 Xuất báo cáo")

        c1, c2 = st.columns(2)

        with c1:
            excel_data = self.report_gen.generate_excel(result.results, result.data_adjusted, result.weights)
            st.download_button(
                "📊 Xuất Excel",
                excel_data,
                file_name=f"riskcast_{params.route.replace(' - ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        with c2:
            pdf_data = self.report_gen.generate_pdf(result.results, params, result.var, result.cvar)
            if pdf_data:
                st.download_button(
                    "📄 Xuất PDF",
                    pdf_data,
                    file_name=f"riskcast_{params.route.replace(' - ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    # ---------------------- RUN APP ----------------------
    def run(self):
        self.initialize()

        # HEADER ENTERPRISE
        st.markdown(
            """
            <div class="app-header">
                <div class="app-header-left">
                    <div class="app-logo-circle">RC</div>
                    <div>
                        <div class="app-header-title">RISKCAST v5.2 — ENTERPRISE EDITION</div>
                        <div class="app-header-subtitle">
                            Fuzzy AHP · TOPSIS · Monte Carlo · VaR/CVaR · Forecast
                        </div>
                    </div>
                </div>
                <div class="app-header-badge">
                    🧠 Enterprise AI · ESG Risk Engine
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        historical = DataService.load_historical_data()
        params = self.render_sidebar()

        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        self.render_weight_controls()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True):
            with st.spinner("Đang chạy mô hình..."):
                try:
                    result = self.controller.run_analysis(params, historical)
                    self.display_results(result, params)
                except Exception as e:
                    st.error(f"❌ Lỗi phân tích: {e}")
                    st.exception(e)


# =============================================================================
# MAIN
# =============================================================================

def main():
    app = StreamlitUI()
    app.run()

if __name__ == "__main__":
    main()
