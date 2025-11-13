# =============================================================================
# RISKCAST v5.1.5 — Premium Green DARK Edition
# Author: Bùi Xuân Hoàng — Premium Chart Upgrade by Kai
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

# Optional ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


# =============================================================================
# CONSTANTS & DATA MODELS
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
    forecast_months: np.ndarray
    fuzzy_table: Optional[pd.DataFrame]


CRITERIA = [
    "C1: Tỷ lệ phí",
    "C2: Thời gian xử lý",
    "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC",
    "C5: Chăm sóc KH",
    "C6: Rủi ro khí hậu",
]

DEFAULT_WEIGHTS = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])

COST_BENEFIT_MAP: Dict[str, CriterionType] = {
    "C1: Tỷ lệ phí": CriterionType.COST,
    "C2: Thời gian xử lý": CriterionType.COST,
    "C3: Tỷ lệ tổn thất": CriterionType.COST,
    "C4: Hỗ trợ ICC": CriterionType.BENEFIT,
    "C5: Chăm sóc KH": CriterionType.BENEFIT,
    "C6: Rủi ro khí hậu": CriterionType.COST,
}

SENSITIVITY_MAP: Dict[str, float] = {
    "Chubb": 0.95,
    "PVI": 1.10,
    "InternationalIns": 1.20,
    "BaoViet": 1.05,
    "Aon": 0.90,
}


# =============================================================================
# CSS — PREMIUM GREEN DARK THEME
# =============================================================================

def apply_custom_css():
    """Premium Green DARK UI v2 — Xanh đậm neon, bo góc đẹp."""
    st.markdown(
        """
        <style>

        /* BACKGROUND ---------------------------------------------------------*/
        .stApp {
            background: linear-gradient(180deg,#002B1A 0%, #001F13 40%, #00150D 100%) !important;
            font-family: 'Inter', sans-serif;
            color: #E6FFE9 !important;
        }

        .block-container {
            padding: 2rem 2.5rem !important;
            background: rgba(0, 50, 30, 0.30);
            backdrop-filter: blur(5px);
            border-radius: 18px;
            border: 1.5px solid #00D387;
            box-shadow: 0 0 25px rgba(0,255,150,0.15);
        }

        /* HEADERS -----------------------------------------------------------*/
        h1 {
            color: #00FFAA !important;
            font-weight: 900 !important;
            text-shadow: 0 0 15px rgba(0,255,160,0.6);
        }
        h2, h3 {
            color: #7CFFCE !important;
            font-weight: 800;
            text-shadow: 0 0 10px rgba(0,255,140,0.4);
        }

        p, label, span, .stMarkdown {
            color: #D7FFEE !important;
            font-weight: 600;
        }

        /* SIDEBAR -----------------------------------------------------------*/
        section[data-testid="stSidebar"] {
            background: #001E14 !important;
            border-right: 3px solid #00FFAA;
        }
        section[data-testid="stSidebar"] h2 {
            color: #00FFAA !important;
            background: #003822;
            padding: 12px;
            border-radius: 10px;
            text-align: center;
            font-weight: 900;
            border: 1.3px solid #00D387;
        }

        section[data-testid="stSidebar"] label {
            color: #CFFFEF !important;
            font-size: 1rem;
        }

        /* INPUT FIELDS -------------------------------------------------------*/
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] select {
            background: #002C1A !important;
            color: #D0FFE8 !important;
            border: 1.5px solid #00FFAA !important;
            border-radius: 8px !important;
        }

        /* BUTTONS ------------------------------------------------------------*/
        .stButton > button {
            background: linear-gradient(135deg,#00A86B,#00FFAA) !important;
            color: #002214 !important;
            border-radius: 12px !important;
            padding: 12px 18px !important;
            font-size: 1.15rem !important;
            font-weight: 900 !important;
            border: none !important;
            box-shadow: 0 0 18px rgba(0,255,160,0.4) !important;
            transition: all 0.2s ease-in-out;
        }
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.03);
            box-shadow: 0 0 25px rgba(0,255,160,0.7) !important;
        }

        /* TABLE --------------------------------------------------------------*/
        .stDataFrame table {
            border-radius: 12px !important;
            overflow: hidden !important;
        }
        .stDataFrame thead tr th {
            background: #004A32 !important;
            color: #CFFFF0 !important;
            font-weight: 900;
        }
        .stDataFrame tbody tr td {
            background: rgba(0,40,25,0.4) !important;
            color: #E6FFF5 !important;
        }

        /* RESULT BOX ---------------------------------------------------------*/
        .result-box {
            background: linear-gradient(135deg,#00FFAA,#00C47A);
            padding: 20px;
            text-align: center;
            border-radius: 16px;
            color: #002313 !important;
            font-weight: 900;
            font-size: 1.4rem;
            box-shadow: 0 0 25px rgba(0,255,170,0.45);
        }

        /* METRICS ------------------------------------------------------------*/
        [data-testid="stMetricValue"] {
            color: #00FFAA !important;
            text-shadow: 0 0 8px rgba(0,255,160,0.5);
            font-weight: 900;
        }
        [data-testid="stMetricLabel"] {
            color: #CFFFF0 !important;
        }

        /* EXPLANATION BOX ----------------------------------------------------*/
        .explanation-box {
            background: rgba(0, 45, 25, 0.45);
            padding: 1.3rem;
            border-left: 6px solid #00FFAA;
            border-radius: 10px;
            box-shadow: 0 0 18px rgba(0,255,160,0.15);
        }
        .explanation-box h4 {
            color: #00FFAA !important;
            font-weight: 900;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
# =============================================================================
# CORE ALGORITHMS — PREMIUM DARK v5.1.5
# =============================================================================


# ============================
# 1. TRỌNG SỐ (AUTO-BALANCE)
# ============================

class WeightManager:
    """Tự cân bằng trọng số sao cho tổng = 1, có khóa từng tiêu chí."""

    @staticmethod
    def auto_balance(weights: np.ndarray, locked: List[bool]) -> np.ndarray:
        w = np.array(weights, dtype=float)
        locked_flags = np.array(locked, dtype=bool)

        total_locked = w[locked_flags].sum()
        free_idx = np.where(~locked_flags)[0]

        # Không còn tiêu chí nào free
        if len(free_idx) == 0:
            total = w.sum() or 1.0
            return np.round(w / total, 6)

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



# ============================
# 2. FUZZY AHP
# ============================

class FuzzyAHP:
    """
    Fuzzy AHP tam giác: low, mid, high.
    Tạo thêm:
      - centroid (defuzzified)
      - range (high - low)
    """

    @staticmethod
    def build_fuzzy_table(weights: pd.Series, uncertainty_pct: float) -> pd.DataFrame:
        factor = uncertainty_pct / 100
        base = weights.values

        # raw fuzzy
        low_raw = np.maximum(base * (1 - factor), 1e-9)
        mid_raw = base.copy()
        high_raw = np.minimum(base * (1 + factor), 0.999)

        # normalize helper
        def normalize(arr):
            s = arr.sum()
            return arr / s if s > 0 else np.full_like(arr, 1/len(arr))

        low = normalize(low_raw)
        mid = normalize(mid_raw)
        high = normalize(high_raw)

        centroid_raw = (low_raw + mid_raw + high_raw) / 3
        centroid = normalize(centroid_raw)

        rng = high - low

        df = pd.DataFrame({
            "low": low,
            "mid": mid,
            "high": high,
            "centroid": centroid,
            "range": rng,
        }, index=weights.index)

        return df

    @staticmethod
    def apply(weights: pd.Series, uncertainty_pct: float):
        fuzzy_table = FuzzyAHP.build_fuzzy_table(weights, uncertainty_pct)
        centroid_weight = fuzzy_table["centroid"]
        return centroid_weight, fuzzy_table



# ============================
# 3. MONTE CARLO
# ============================

class MonteCarloSimulator:
    """Monte Carlo cho rủi ro khí hậu C6."""

    @staticmethod
    @st.cache_data(ttl=600)
    def simulate(base_risk: float, sensitivity_map: Dict[str, float], n_simulations: int):
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())

        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.025, mu * 0.12)

        sims = rng.normal(mu, sigma, size=(n_simulations, len(companies)))
        sims = np.clip(sims, 0.0, 1.0)

        return companies, sims.mean(axis=0), sims.std(axis=0)



# ============================
# 4. TOPSIS
# ============================

class TOPSISAnalyzer:
    """Chuẩn TOPSIS: normalize → weight → ideal → distance."""

    @staticmethod
    def analyze(data: pd.DataFrame, weights: pd.Series, cost_benefit: Dict[str, CriterionType]):

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



# ============================
# 5. VaR / CVaR + CONFIDENCE
# ============================

class RiskCalculator:

    @staticmethod
    def calculate_var_cvar(loss_rates: np.ndarray, cargo_value: float, confidence=0.95):
        if len(loss_rates) == 0:
            return 0.0, 0.0

        losses = loss_rates * cargo_value
        var = float(np.percentile(losses, confidence * 100))
        tail = losses[losses >= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        return var, cvar

    @staticmethod
    def calculate_confidence(results: pd.DataFrame, data: pd.DataFrame):
        eps = 1e-9

        cv_c6 = results["C6_std"] / (results["C6_mean"] + eps)
        conf_c6 = 1 / (1 + cv_c6)
        conf_c6 = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + eps)

        crit_cv = data.std(axis=1) / (data.mean(axis=1) + eps)
        conf_crit = 1 / (1 + crit_cv)
        conf_crit = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (np.ptp(conf_crit) + eps)

        return np.sqrt(conf_c6 * conf_crit)



# ============================
# 6. FORECASTER (FIX MONTH 14)
# ============================

class Forecaster:
    """Chỉ dự báo 1 tháng — luôn chính xác và không nhảy sang tháng 14."""

    @staticmethod
    def forecast(historical: pd.DataFrame, route: str, current_month: int, months_ahead=1, use_arima=True):

        if route not in historical.columns:
            route = historical.columns[1]

        series = historical[route].values  # 12 tháng

        # ARIMA hoặc fallback
        if use_arima and ARIMA_AVAILABLE and len(series) >= 6:
            try:
                model = ARIMA(series, order=(1,1,1))
                fitted = model.fit()
                fc = fitted.forecast(months_ahead)
                fc = np.clip(fc, 0, 1)
            except:
                fc = np.array([series[-1]])
        else:
            if len(series) >= 3:
                trend = (series[-1] - series[-3]) / 3
            else:
                trend = 0
            fc = np.array([np.clip(series[-1] + trend, 0, 1)])

        next_month = (current_month % 12) + 1
        return series, fc, np.array([next_month])
# =============================================================================
# VISUALIZATION — PREMIUM GREEN DARK v5.1.5
# =============================================================================

class ChartFactory:
    """Toàn bộ chart style Premium Green DARK Neon."""

    # -------------------------------------------------------------
    # APPLY PREMIUM THEME
    # -------------------------------------------------------------
    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=24, color="#00FFAA", family="Arial Black"),
                x=0.5,
            ),
            font=dict(size=15, color="#CFFFF0", family="Inter"),
            paper_bgcolor="#002016",
            plot_bgcolor="#001A12",
            margin=dict(l=60, r=40, t=80, b=70),
            legend=dict(
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="#00FFAA",
                borderwidth=1.6,
                font=dict(size=13, color="#CFFFF0"),
            ),
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,255,160,0.08)",
            linecolor="#00FFAA",
            linewidth=1.3,
            zeroline=False,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(0,255,160,0.08)",
            linecolor="#00FFAA",
            linewidth=1.3,
            zeroline=False,
        )
        return fig


    # -------------------------------------------------------------
    # PIE CHART — TRỌNG SỐ (PREMIUM GREEN)
    # -------------------------------------------------------------
    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        colors = [
            "#00FFAA", "#00D387", "#00A86B",
            "#008F5D", "#66FFCA", "#33FFB2"
        ]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=weights.index,
                    values=weights.values,
                    hole=0.45,
                    marker=dict(colors=colors, line=dict(color="#001A12", width=3)),
                    textinfo="percent",
                    textfont=dict(size=15, color="#001A12", family="Inter"),
                    pull=[0.03]*len(weights),
                    hovertemplate="<b>%{label}</b><br>Tỉ trọng: %{value:.2%}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=22, color="#00FFAA"),
                x=0.5,
            ),
            paper_bgcolor="#002016",
        )
        return fig


    # -------------------------------------------------------------
    # TOPSIS BAR — NEON PREMIUM
    # -------------------------------------------------------------
    @staticmethod
    def create_topsis_bar(results: pd.DataFrame) -> go.Figure:
        df = results.sort_values("score", ascending=True)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df["score"],
                    y=df["company"],
                    orientation="h",
                    text=df["score"].apply(lambda x: f"{x:.3f}"),
                    textposition="outside",
                    marker=dict(
                        color=df["score"],
                        colorscale=[
                            [0.00, "#004D36"],
                            [0.25, "#00A06B"],
                            [0.50, "#00D387"],
                            [0.75, "#00FFAA"],
                            [1.00, "#66FFD1"],
                        ],
                        line=dict(color="#00FFAA", width=1.8),
                    ),
                    hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
                )
            ]
        )

        fig = ChartFactory._apply_theme(fig, "🏆 TOPSIS Score (cao hơn = tốt hơn)")
        fig.update_xaxes(range=[0, 1], title="Điểm TOPSIS")
        fig.update_yaxes(title="Công ty")
        return fig


    # -------------------------------------------------------------
    # FORECAST — CLIMATE RISK (1 THÁNG)
    # -------------------------------------------------------------
    @staticmethod
    def create_forecast_chart(historical, forecast, forecast_months, route: str):

        months_hist = list(range(1, len(historical)+1))
        months_fc = list(forecast_months)

        fig = go.Figure()

        # Lịch sử
        fig.add_trace(
            go.Scatter(
                x=months_hist,
                y=historical,
                mode="lines+markers",
                name="📈 Lịch sử",
                line=dict(color="#00FFAA", width=3),
                marker=dict(size=9, color="#00D387", line=dict(width=2, color="#001A12")),
                hovertemplate="Tháng %{x}<br>Rủi ro: %{y:.2%}<extra></extra>",
            )
        )

        # Dự báo
        fig.add_trace(
            go.Scatter(
                x=months_fc,
                y=forecast,
                mode="lines+markers",
                name="🔮 Dự báo 1 tháng",
                line=dict(color="#FFB44C", width=3, dash="dash"),
                marker=dict(
                    size=11,
                    color="#FF9800",
                    symbol="diamond",
                    line=dict(width=2.5, color="#FFF"),
                ),
                hovertemplate="Tháng %{x}<br>Dự báo: %{y:.2%}<extra></extra>",
            )
        )

        ymax = max(1.0, float(max(historical.max(), forecast.max()) * 1.15))

        fig = ChartFactory._apply_theme(fig, f"📊 Dự báo rủi ro khí hậu — {route}")
        fig.update_xaxes(
            title="Tháng",
            tickmode="linear",
            tickvals=list(range(1, 13)),
            dtick=1,
        )
        fig.update_yaxes(
            title="Mức rủi ro (0–1)",
            range=[0, ymax],
            tickformat=".0%",
        )
        return fig


    # -------------------------------------------------------------
    # FUZZY HEATMAP (LOW–MID–HIGH–CENTROID)
    # -------------------------------------------------------------
    @staticmethod
    def create_fuzzy_heatmap(fuzzy_table: pd.DataFrame):
        data = fuzzy_table[["low", "mid", "high", "centroid"]].values
        z_text = np.round(data, 3).astype(str)

        fig = px.imshow(
            data,
            x=["Low", "Mid", "High", "Centroid"],
            y=fuzzy_table.index,
            text=z_text,
            aspect="auto",
            color_continuous_scale=[
                (0.0, "#002015"),
                (0.2, "#005A3A"),
                (0.4, "#00A06B"),
                (0.6, "#00D387"),
                (0.8, "#00FFAA"),
                (1.0, "#66FFD1"),
            ],
        )
        fig.update_traces(texttemplate="%{text}", textfont=dict(size=11, color="#001A12"))
        fig = ChartFactory._apply_theme(fig, "🌿 Fuzzy Heatmap (Low–Mid–High–Centroid)")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Tiêu chí")
        return fig


    # -------------------------------------------------------------
    # FUZZY TRIANGLE CHART
    # -------------------------------------------------------------
    @staticmethod
    def create_fuzzy_triangle(fuzzy_table: pd.DataFrame, criterion: str):
        """Vẽ tam giác Fuzzy cho 1 tiêu chí."""

        row = fuzzy_table.loc[criterion]
        x = [row["low"], row["mid"], row["high"]]
        y = [0, 1, 0]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                fill="tozeroy",
                name="Fuzzy",
                line=dict(color="#00FFAA", width=3),
                marker=dict(size=10, color="#00D387", line=dict(width=2, color="#001A12")),
            )
        )
        fig = ChartFactory._apply_theme(fig, f"🔺 Fuzzy Triangle — {criterion}")
        fig.update_xaxes(title="Trọng số", range=[0, max(0.3, row["high"]+0.05)])
        fig.update_yaxes(title="Membership", range=[0, 1.2])
        return fig
# =============================================================================
# STREAMLIT UI & MAIN — RISKCAST v5.1.5 Premium Chart
# =============================================================================

class StreamlitUI:
    """Quản lý toàn bộ UI Streamlit cho RISKCAST v5.1.5."""

    def __init__(self):
        self.controller = AnalysisController()
        self.chart_factory = ChartFactory()
        self.report_gen = ReportGenerator()

    # ---------------------------------------------------------
    # INIT
    # ---------------------------------------------------------
    def initialize(self):
        st.set_page_config(
            page_title="RISKCAST v5.1.5 — ESG Risk Assessment",
            page_icon="🛡️",
            layout="wide",
        )
        apply_custom_css()

        if "weights" not in st.session_state:
            st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
        if "locked" not in st.session_state:
            st.session_state["locked"] = [False] * len(CRITERIA)

    # ---------------------------------------------------------
    # SIDEBAR
    # ---------------------------------------------------------
    def render_sidebar(self) -> AnalysisParams:
        with st.sidebar:
            st.header("📊 Thông tin lô hàng")

            cargo_value = st.number_input(
                "Giá trị lô hàng (USD)",
                min_value=1000,
                value=39000,
                step=1000,
            )
            good_type = st.selectbox(
                "Loại hàng",
                ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"],
            )
            route = st.selectbox(
                "Tuyến",
                ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"],
            )
            method = st.selectbox("Phương thức vận tải", ["Sea", "Air", "Truck"])
            month = st.selectbox("Tháng (1–12)", list(range(1, 13)), index=8)
            priority = st.selectbox(
                "Ưu tiên của chủ hàng",
                ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"],
            )

            st.markdown("---")
            st.header("⚙️ Cấu hình mô hình")

            use_fuzzy = st.checkbox("Bật Fuzzy AHP", True)
            use_arima = st.checkbox("Dùng ARIMA dự báo C6", True)
            use_mc = st.checkbox("Monte Carlo cho C6", True)
            use_var = st.checkbox("Tính VaR & CVaR", True)

            mc_runs = st.number_input(
                "Số vòng Monte Carlo",
                min_value=500,
                max_value=10000,
                value=2000,
                step=500,
            )
            fuzzy_uncertainty = (
                st.slider("Mức bất định Fuzzy (%)", 0, 50, 15) if use_fuzzy else 15
            )

            return AnalysisParams(
                cargo_value=cargo_value,
                good_type=good_type,
                route=route,
                method=method,
                month=month,
                priority=priority,
                use_fuzzy=use_fuzzy,
                use_arima=use_arima,
                use_mc=use_mc,
                use_var=use_var,
                mc_runs=mc_runs,
                fuzzy_uncertainty=fuzzy_uncertainty,
            )

    # ---------------------------------------------------------
    # WEIGHT CONTROLS
    # ---------------------------------------------------------
    def render_weight_controls(self):
        st.subheader("🎯 Phân bổ trọng số tiêu chí")

        st.markdown(
            """
            <div class="explanation-box">
                <h4>📋 Ý nghĩa các tiêu chí:</h4>
                <ul>
                    <li><b>C1 – Tỷ lệ phí:</b> phần trăm phí bảo hiểm (càng thấp càng tốt).</li>
                    <li><b>C2 – Thời gian xử lý:</b> số ngày giải quyết hồ sơ (càng nhanh càng tốt).</li>
                    <li><b>C3 – Tỷ lệ tổn thất:</b> xác suất/tần suất tổn thất (càng thấp càng tốt).</li>
                    <li><b>C4 – Hỗ trợ ICC:</b> mức hỗ trợ điều khoản ICC A/B/C (càng cao càng tốt).</li>
                    <li><b>C5 – Chăm sóc khách hàng:</b> dịch vụ CSKH, hỗ trợ claim (càng cao càng tốt).</li>
                    <li><b>C6 – Rủi ro khí hậu:</b> rủi ro thiên tai, bão, thời tiết xấu trên tuyến (càng thấp càng tốt).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(len(CRITERIA))
        new_weights = st.session_state["weights"].copy()

        for i, criterion in enumerate(CRITERIA):
            with cols[i]:
                short = criterion.split(":")[0]
                detail = criterion.split(":")[1].strip() if ":" in criterion else ""

                st.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        padding:8px;
                        margin-bottom:6px;
                        background:#e8f5e9;
                        border-radius:8px;
                        border:2px solid #66bb6a;">
                        <b style="color:#1b5e20;">{short}</b><br>
                        <span style="font-size:0.8rem; color:#33691e;">{detail}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                lock = st.checkbox(
                    "🔒 Khóa",
                    value=st.session_state["locked"][i],
                    key=f"lock_{i}",
                )
                st.session_state["locked"][i] = lock

                w_val = st.number_input(
                    "Tỉ lệ",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(new_weights[i]),
                    step=0.01,
                    key=f"weight_{i}",
                    label_visibility="collapsed",
                )
                new_weights[i] = w_val

                st.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        background:#e3f2fd;
                        padding:6px;
                        border-radius:6px;
                        border:2px solid #42a5f5;">
                        <span style="color:#1565c0; font-weight:800;">{w_val:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        col_reset, col_info = st.columns([1, 2])

        with col_reset:
            if st.button("🔄 Reset trọng số mặc định", use_container_width=True):
                st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
                st.session_state["locked"] = [False] * len(CRITERIA)
                st.rerun()

        with col_info:
            total = float(new_weights.sum())
            if abs(total - 1.0) > 0.01:
                st.warning(f"⚠️ Tổng trọng số hiện tại: {total:.1%} (mục tiêu = 100%)")
            else:
                st.success(f"✅ Tổng trọng số: {total:.1%}")

        st.session_state["weights"] = WeightManager.auto_balance(
            new_weights,
            st.session_state["locked"],
        )

    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------
    def display_results(self, result: AnalysisResult, params: AnalysisParams):
        st.success("✅ Đã hoàn tất phân tích RISKCAST!")

        left, right = st.columns([2, 1])

        with left:
            st.subheader("🏅 Bảng xếp hạng TOPSIS")
            df_view = result.results[
                ["rank", "company", "score", "confidence", "recommend_icc"]
            ].set_index("rank")
            df_view.columns = ["Công ty", "Điểm số", "Độ tin cậy", "ICC"]
            st.dataframe(df_view, use_container_width=True)

            top = result.results.iloc[0]
            st.markdown(
                f"""
                <div class="result-box">
                    🏆 <b>KHUYẾN NGHỊ HÀNG ĐẦU</b><br><br>
                    <span style="font-size:1.5rem;">{top['company']}</span><br><br>
                    Score: <b>{top['score']:.3f}</b> |
                    Confidence: <b>{top['confidence']:.2f}</b> |
                    <b>{top['recommend_icc']}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            if result.var is not None and result.cvar is not None:
                st.metric(
                    "💰 VaR 95%",
                    f"${result.var:,.0f}",
                    help="Tổn thất tối đa với độ tin cậy 95%.",
                )
                st.metric(
                    "🛡️ CVaR 95%",
                    f"${result.cvar:,.0f}",
                    help="Tổn thất trung bình khi tổn thất vượt VaR.",
                )

            fig_weights = self.chart_factory.create_weights_pie(
                result.weights, "⚖️ Trọng số sử dụng cuối cùng"
            )
            st.plotly_chart(fig_weights, use_container_width=True)

        # Giải thích chi tiết
        st.markdown("---")
        st.subheader("📋 Giải thích kết quả cho hội đồng")

        top3 = result.results.head(3)
        top = result.results.iloc[0]

        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>🎯 Vì sao <b>{top['company']}</b> đứng hạng 1?</h4>
                <ul>
                    <li>Điểm TOPSIS cao nhất: <b>{top['score']:.3f}</b>, cân bằng tốt giữa chi phí, rủi ro và dịch vụ.</li>
                    <li>Độ tin cậy mô hình: <b>{top['confidence']:.2f}</b> (gần 1 càng tốt).</li>
                    <li>ICC khuyến nghị: <b>{top['recommend_icc']}</b>, phù hợp tuyến <b>{params.route}</b> và ưu tiên <b>{params.priority}</b>.</li>
                    <li>Giá trị lô hàng: <b>${params.cargo_value:,.0f}</b>, phù hợp với mức bảo hiểm đề xuất.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        comp_text = f"""
        <div class="explanation-box">
            <h4>📊 So sánh Top 3 công ty:</h4>
            <ul>
                <li><b>#1 {top3.iloc[0]['company']}</b> — Score: {top3.iloc[0]['score']:.3f}, C6_mean: {top3.iloc[0]['C6_mean']:.2%}</li>
                <li><b>#2 {top3.iloc[1]['company']}</b> — Score: {top3.iloc[1]['score']:.3f} (kém {top3.iloc[0]['score'] - top3.iloc[1]['score']:.3f}), C6_mean: {top3.iloc[1]['C6_mean']:.2%}</li>
                <li><b>#3 {top3.iloc[2]['company']}</b> — Score: {top3.iloc[2]['score']:.3f} (kém {top3.iloc[0]['score'] - top3.iloc[2]['score']:.3f}), C6_mean: {top3.iloc[2]['C6_mean']:.2%}</li>
            </ul>
        </div>
        """
        st.markdown(comp_text, unsafe_allow_html=True)

        key = result.data_adjusted.loc[top["company"]]
        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>🔑 Các yếu tố quyết định cho {top['company']}:</h4>
                <ul>
                    <li>Tỷ lệ phí: <b>{key['C1: Tỷ lệ phí']:.2%}</b> – {"cạnh tranh" if key['C1: Tỷ lệ phí'] < 0.30 else "khá cao"}.</li>
                    <li>Thời gian xử lý: <b>{key['C2: Thời gian xử lý']:.0f} ngày</b> – {"nhanh" if key['C2: Thời gian xử lý'] < 6 else "trung bình"}.</li>
                    <li>Tỷ lệ tổn thất: <b>{key['C3: Tỷ lệ tổn thất']:.2%}</b> – {"tốt" if key['C3: Tỷ lệ tổn thất'] < 0.08 else "chấp nhận được"}.</li>
                    <li>Hỗ trợ ICC: <b>{key['C4: Hỗ trợ ICC']:.0f}/10</b>.</li>
                    <li>Chăm sóc KH: <b>{key['C5: Chăm sóc KH']:.0f}/10</b>.</li>
                    <li>Rủi ro khí hậu C6: <b>{top['C6_mean']:.2%} ± {top['C6_std']:.2%}</b>.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result.var is not None and result.cvar is not None:
            risk_ratio = result.var / params.cargo_value if params.cargo_value > 0 else 0
            st.markdown(
                f"""
                <div class="explanation-box">
                    <h4>⚠️ Đánh giá rủi ro tài chính (VaR / CVaR):</h4>
                    <ul>
                        <li>VaR 95% ≈ <b>${result.var:,.0f}</b> — 95% trường hợp tổn thất không vượt mức này.</li>
                        <li>CVaR 95% ≈ <b>${result.cvar:,.0f}</b> — tổn thất trung bình nếu vượt qua VaR.</li>
                        <li>Tỷ lệ rủi ro so với giá trị lô hàng: <b>{risk_ratio*100:.1f}%</b>.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Biểu đồ TOPSIS + Forecast
        st.markdown("---")
        st.subheader("📈 Biểu đồ phân tích")

        fig_topsis = self.chart_factory.create_topsis_bar(result.results)
        st.plotly_chart(fig_topsis, use_container_width=True)

        fig_fc = self.chart_factory.create_forecast_chart(
            result.historical,
            result.forecast,
            result.forecast_months,
            params.route,
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Fuzzy AHP Visual
        if result.fuzzy_table is not None:
            st.markdown("---")
            st.subheader("🌿 Fuzzy AHP – Premium Green")

            fuzzy_df = result.fuzzy_table.copy()
            display_df = fuzzy_df.copy()
            display_df.columns = ["Low", "Mid", "High", "Centroid", "Biên độ (High-Low)"]

            st.markdown("**Bảng tham số Fuzzy cho từng tiêu chí:**")
            st.dataframe(display_df.style.format("{:.3f}"), use_container_width=True)

            strongest = fuzzy_df["range"].idxmax()
            max_range = float(fuzzy_df["range"].max())
            st.markdown(
                f"""
                <div class="explanation-box">
                    <h4>🔥 Tiêu chí dao động mạnh nhất (Fuzzy):</h4>
                    <p>
                        <b>{strongest}</b> có biên độ Fuzzy (High–Low) lớn nhất: 
                        <b>{max_range:.3f}</b>. Điều này nghĩa là đánh giá chuyên gia về tiêu chí này 
                        còn nhiều bất định → cần giải thích kỹ hơn trong phần thuyết trình.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Heatmap
            fig_fuzzy = self.chart_factory.create_fuzzy_heatmap(fuzzy_df)
            st.plotly_chart(fig_fuzzy, use_container_width=True)

            # Triangle cho 1 tiêu chí
            chosen_crit = st.selectbox(
                "Chọn tiêu chí để xem đồ thị Fuzzy Triangle:",
                list(fuzzy_df.index),
            )
            fig_tri = self.chart_factory.create_fuzzy_triangle(
                fuzzy_df, chosen_crit
            )
            st.plotly_chart(fig_tri, use_container_width=True)

        # Export
        st.markdown("---")
        st.subheader("📥 Xuất báo cáo")

        col1, col2 = st.columns(2)

        with col1:
            excel_data = self.report_gen.generate_excel(
                result.results,
                result.data_adjusted,
                result.weights,
                result.fuzzy_table,
            )
            st.download_button(
                "📊 Tải Excel (Results + Fuzzy)",
                data=excel_data,
                file_name=f"riskcast_{params.route.replace(' - ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col2:
            pdf_data = self.report_gen.generate_pdf(
                result.results,
                params,
                result.var,
                result.cvar,
            )
            if pdf_data:
                st.download_button(
                    "📄 Tải PDF Executive Summary",
                    data=pdf_data,
                    file_name=f"riskcast_summary_{params.route.replace(' - ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    # ---------------------------------------------------------
    # RUN
    # ---------------------------------------------------------
    def run(self):
        self.initialize()

        st.title("🚢 RISKCAST v5.1.5 — ESG Logistics Risk Assessment")
        st.markdown(
            "**Decision Support System cho lựa chọn bảo hiểm vận tải quốc tế (Fuzzy + Monte Carlo + VaR/CVaR + ARIMA).**"
        )
        st.markdown("---")

        historical = DataService.load_historical_data()
        params = self.render_sidebar()
        self.render_weight_controls()

        st.markdown("---")
        weights_series = pd.Series(st.session_state["weights"], index=CRITERIA)
        fig_current = self.chart_factory.create_weights_pie(
            weights_series, "📊 Trọng số hiện tại (trước Fuzzy)"
        )
        st.plotly_chart(fig_current, use_container_width=True)

        st.markdown("---")
        if st.button("🚀 PHÂN TÍCH & GỢI Ý", type="primary", use_container_width=True):
            with st.spinner("🔄 Đang chạy mô hình RISKCAST..."):
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
    app = StreamlitUI()
    app.run()


if __name__ == "__main__":
    main()
