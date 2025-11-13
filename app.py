# =============================================================================
# RISKCAST v5.1.4 — ESG Logistics Risk Assessment Dashboard (Fuzzy Premium Green)
# Author: Bùi Xuân Hoàng — Refactored with OOP + Fuzzy Visualization by Kai
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
    """Loại tiêu chí: tối thiểu (cost) hay tối đa (benefit)."""
    COST = "cost"
    BENEFIT = "benefit"


@dataclass
class AnalysisParams:
    """Container lưu các tham số phân tích do user nhập từ Sidebar."""
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
    """Kết quả phân tích main pipeline."""
    results: pd.DataFrame
    weights: pd.Series
    data_adjusted: pd.DataFrame
    var: Optional[float]
    cvar: Optional[float]
    historical: np.ndarray
    forecast: np.ndarray
    forecast_months: np.ndarray
    fuzzy_table: Optional[pd.DataFrame]


# Các tiêu chí chính của mô hình
CRITERIA = [
    "C1: Tỷ lệ phí",
    "C2: Thời gian xử lý",
    "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC",
    "C5: Chăm sóc KH",
    "C6: Rủi ro khí hậu",
]

# Trọng số mặc định
DEFAULT_WEIGHTS = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])

# Mapping cost / benefit
COST_BENEFIT_MAP: Dict[str, CriterionType] = {
    "C1: Tỷ lệ phí": CriterionType.COST,
    "C2: Thời gian xử lý": CriterionType.COST,
    "C3: Tỷ lệ tổn thất": CriterionType.COST,
    "C4: Hỗ trợ ICC": CriterionType.BENEFIT,
    "C5: Chăm sóc KH": CriterionType.BENEFIT,
    "C6: Rủi ro khí hậu": CriterionType.COST,
}

# Hệ số nhạy cảm rủi ro khí hậu theo hãng
SENSITIVITY_MAP: Dict[str, float] = {
    "Chubb": 0.95,
    "PVI": 1.10,
    "InternationalIns": 1.20,
    "BaoViet": 1.05,
    "Aon": 0.90,
}


# =============================================================================
# UI STYLING — PREMIUM GREEN
# =============================================================================

def apply_custom_css() -> None:
    """CSS giao diện Premium Green + high contrast cho hội đồng dễ nhìn."""
    st.markdown(
        '''
        <style>
            * {
                text-rendering: optimizeLegibility !important;
                -webkit-font-smoothing: antialiased !important;
            }

            .stApp {
                background: linear-gradient(135deg,#e8f5e9 0%,#ffffff 40%,#e3f2fd 100%) !important;
                font-family: "Inter","Segoe UI",Arial,sans-serif !important;
            }

            .block-container {
                background: #ffffff !important;
                padding: 2rem 2.5rem !important;
                border-radius: 18px;
                box-shadow: 0 4px 22px rgba(0,0,0,0.12);
                max-width: 1400px;
                margin: 1.5rem auto;
                border: 2px solid #a5d6a7;
            }

            h1 {
                color: #1b5e20 !important;
                font-weight: 900 !important;
                font-size: 2.8rem !important;
                letter-spacing: -0.02em;
            }

            h2 {
                color: #004d40 !important;
                font-weight: 800 !important;
                font-size: 2rem !important;
            }

            h3 {
                color: #1b5e20 !important;
                font-weight: 700 !important;
                font-size: 1.5rem !important;
            }

            p, span, div, label, .stMarkdown {
                color: #0d1b2a !important;
                font-weight: 600 !important;
            }

            .stButton > button {
                background: linear-gradient(135deg,#2e7d32,#1b5e20) !important;
                color: #ffffff !important;
                border-radius: 10px !important;
                padding: 0.85rem 2.4rem !important;
                font-weight: 800 !important;
                font-size: 1.05rem !important;
                border: 2px solid #1b5e20 !important;
                box-shadow: 0 4px 14px rgba(27,94,32,0.35) !important;
                text-transform: uppercase;
            }

            .stButton > button:hover {
                background: linear-gradient(135deg,#1b5e20,#004d40) !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 7px 20px rgba(0,77,64,0.45) !important;
            }

            .result-box {
                background: linear-gradient(135deg,#c8e6c9,#a5d6a7);
                color: #0d1b2a !important;
                padding: 1.8rem 2.2rem;
                border-radius: 14px;
                font-weight: 800 !important;
                font-size: 1.25rem !important;
                text-align: center;
                margin: 1.5rem 0;
                box-shadow: 0 5px 18px rgba(56,142,60,0.35);
                border: 3px solid #66bb6a;
            }

            .stDataFrame {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 3px 14px rgba(0,0,0,0.12);
                border: 2px solid #cfd8dc !important;
            }

            .stDataFrame thead tr th {
                background-color: #1b5e20 !important;
                color: #ffffff !important;
                font-weight: 800 !important;
                font-size: 1.05rem !important;
            }

            .stDataFrame tbody tr td {
                color: #0d1b2a !important;
                font-weight: 650 !important;
                font-size: 1rem !important;
            }

            section[data-testid="stSidebar"] {
                background: #ffffff !important;
                border-right: 3px solid #2e7d32;
            }

            section[data-testid="stSidebar"] h2 {
                color: #1b5e20 !important;
                font-weight: 900 !important;
                background: #e8f5e9 !important;
                padding: 12px !important;
                border-radius: 10px !important;
                margin-bottom: 16px !important;
                border: 1px solid #a5d6a7;
            }

            section[data-testid="stSidebar"] label {
                color: #0d1b2a !important;
                font-weight: 750 !important;
                font-size: 1.02rem !important;
            }

            section[data-testid="stSidebar"] input,
            section[data-testid="stSidebar"] select {
                background: #ffffff !important;
                color: #0d1b2a !important;
                font-weight: 650 !important;
                border: 2px solid #2e7d32 !important;
            }

            [data-testid="stMetricValue"] {
                color: #1b5e20 !important;
                font-weight: 900 !important;
                font-size: 2.3rem !important;
            }

            [data-testid="stMetricLabel"] {
                color: #0d1b2a !important;
                font-weight: 800 !important;
                font-size: 1.1rem !important;
            }

            .explanation-box {
                background: #edf7ed;
                border-left: 6px solid #2e7d32;
                padding: 1.4rem 1.6rem;
                margin: 1.4rem 0;
                border-radius: 10px;
            }

            .explanation-box h4 {
                color: #1b5e20 !important;
                font-weight: 800 !important;
                margin-bottom: 0.8rem !important;
            }
        </style>
        ''',
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA LAYER
# =============================================================================

class DataService:
    """Quản lý dữ liệu demo: lịch sử rủi ro khí hậu + dữ liệu hãng bảo hiểm."""

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_historical_data() -> pd.DataFrame:
        """Tạo dữ liệu lịch sử rủi ro khí hậu theo tuyến (12 tháng)."""
        climate_base = {
            "VN - EU": [0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.42, 0.48, 0.60, 0.68, 0.58, 0.45],
            "VN - US": [0.30, 0.33, 0.36, 0.40, 0.45, 0.50, 0.56, 0.62, 0.75, 0.72, 0.60, 0.52],
            "VN - Singapore": [0.15, 0.16, 0.18, 0.20, 0.22, 0.26, 0.30, 0.32, 0.35, 0.34, 0.28, 0.25],
            "VN - China": [0.18, 0.19, 0.21, 0.24, 0.26, 0.30, 0.34, 0.36, 0.40, 0.38, 0.32, 0.28],
            "Domestic": [0.10] * 12,
        }

        df = pd.DataFrame({"month": list(range(1, 13))})
        for route, values in climate_base.items():
            df[route] = values
        return df

    @staticmethod
    @st.cache_data
    def get_company_data() -> pd.DataFrame:
        """Dữ liệu baseline các hãng bảo hiểm."""
        return (
            pd.DataFrame(
                {
                    "Company": ["Chubb", "PVI", "InternationalIns", "BaoViet", "Aon"],
                    "C1: Tỷ lệ phí": [0.30, 0.28, 0.26, 0.32, 0.24],
                    "C2: Thời gian xử lý": [6, 5, 8, 7, 4],
                    "C3: Tỷ lệ tổn thất": [0.08, 0.06, 0.09, 0.10, 0.07],
                    "C4: Hỗ trợ ICC": [9, 8, 6, 9, 7],
                    "C5: Chăm sóc KH": [9, 8, 5, 7, 6],
                }
            )
            .set_index("Company")
        )


# =============================================================================
# CORE ALGORITHMS
# =============================================================================

class WeightManager:
    """Quản lý, cân bằng trọng số để tổng luôn = 1."""

    @staticmethod
    def auto_balance(weights: np.ndarray, locked: List[bool]) -> np.ndarray:
        w = np.array(weights, dtype=float)
        locked_flags = np.array(locked, dtype=bool)

        total_locked = w[locked_flags].sum()
        free_idx = np.where(~locked_flags)[0]

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


class FuzzyAHP:
    """
    Fuzzy AHP với số mờ tam giác (low, mid, high).
    Premium Green: thêm bảng, heatmap, highlight biên độ dao động.
    """

    @staticmethod
    def build_fuzzy_table(weights: pd.Series, uncertainty_pct: float) -> pd.DataFrame:
        """
        Tạo bảng Fuzzy:
        - low, mid, high (đã chuẩn hóa)
        - centroid (trọng số defuzzified)
        - range = high - low (mức dao động).
        """
        factor = uncertainty_pct / 100.0
        base = weights.values

        # Số mờ gốc
        low_raw = np.maximum(base * (1 - factor), 1e-9)
        mid_raw = base.copy()
        high_raw = np.minimum(base * (1 + factor), 0.9999)

        # Chuẩn hóa từng thành phần
        def normalize(arr: np.ndarray) -> np.ndarray:
            s = arr.sum()
            if s <= 0:
                return np.full_like(arr, 1.0 / len(arr))
            return arr / s

        low = normalize(low_raw)
        mid = normalize(mid_raw)
        high = normalize(high_raw)

        # Centroid (defuzzified)
        centroid_raw = (low_raw + mid_raw + high_raw) / 3.0
        centroid = normalize(centroid_raw)

        # Biên độ
        rng = high - low

        df = pd.DataFrame(
            {
                "criterion": weights.index,
                "low": low,
                "mid": mid,
                "high": high,
                "centroid": centroid,
                "range": rng,
            }
        ).set_index("criterion")

        return df

    @staticmethod
    def apply(weights: pd.Series, uncertainty_pct: float) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Trả về:
        - centroid_weight: trọng số defuzzified dùng cho TOPSIS
        - fuzzy_table: bảng Fuzzy chi tiết.
        """
        fuzzy_table = FuzzyAHP.build_fuzzy_table(weights, uncertainty_pct)
        centroid_weight = fuzzy_table["centroid"]
        return centroid_weight, fuzzy_table


class MonteCarloSimulator:
    """Monte Carlo cho rủi ro khí hậu (C6)."""

    @staticmethod
    @st.cache_data(ttl=600)
    def simulate(
        base_risk: float,
        sensitivity_map: Dict[str, float],
        n_simulations: int,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())

        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.03, mu * 0.12)

        sims = rng.normal(loc=mu, scale=sigma, size=(n_simulations, len(companies)))
        sims = np.clip(sims, 0.0, 1.0)

        return companies, sims.mean(axis=0), sims.std(axis=0)


class TOPSISAnalyzer:
    """Phân tích TOPSIS."""

    @staticmethod
    def analyze(
        data: pd.DataFrame,
        weights: pd.Series,
        cost_benefit: Dict[str, CriterionType],
    ) -> np.ndarray:
        M = data[list(weights.index)].values.astype(float)

        # Chuẩn hóa
        denom = np.sqrt((M ** 2).sum(axis=0))
        denom[denom == 0] = 1.0
        R = M / denom

        # Áp trọng số
        V = R * weights.values

        is_cost = np.array([cost_benefit[c] == CriterionType.COST for c in weights.index])
        ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
        ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))

        d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

        return d_minus / (d_plus + d_minus + 1e-12)


class RiskCalculator:
    """Tính VaR / CVaR + độ tin cậy."""

    @staticmethod
    def calculate_var_cvar(
        loss_rates: np.ndarray,
        cargo_value: float,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        if len(loss_rates) == 0:
            return 0.0, 0.0

        losses = loss_rates * cargo_value
        var = float(np.percentile(losses, confidence * 100))
        tail_losses = losses[losses >= var]
        cvar = float(tail_losses.mean()) if len(tail_losses) > 0 else var
        return var, cvar

    @staticmethod
    def calculate_confidence(results: pd.DataFrame, data: pd.DataFrame) -> np.ndarray:
        eps = 1e-9

        # Confidence từ C6
        cv_c6 = results["C6_std"].values / (results["C6_mean"].values + eps)
        conf_c6 = 1.0 / (1.0 + cv_c6)
        conf_c6 = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + eps)

        # Confidence từ biến động các tiêu chí
        crit_cv = data.std(axis=1).values / (data.mean(axis=1).values + eps)
        conf_crit = 1.0 / (1.0 + crit_cv)
        conf_crit = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (np.ptp(conf_crit) + eps)

        return np.sqrt(conf_c6 * conf_crit)


class Forecaster:
    """Dự báo rủi ro khí hậu: chỉ 1 bước (1 tháng) như bạn yêu cầu."""

    @staticmethod
    def forecast(
        historical: pd.DataFrame,
        route: str,
        current_month: int,
        months_ahead: int = 1,
        use_arima: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if route not in historical.columns:
            route = historical.columns[1]

        series = historical[route].values  # 12 tháng
        # ARIMA hoặc trend đơn giản, nhưng chỉ lấy 1 step dự báo
        if use_arima and ARIMA_AVAILABLE and len(series) >= 6:
            try:
                model = ARIMA(series, order=(1, 1, 1))
                fitted = model.fit()
                fc = fitted.forecast(months_ahead)
                fc = np.clip(fc, 0.0, 1.0)
            except Exception:
                fc = np.array([series[-1]])
        else:
            if len(series) >= 3:
                trend = (series[-1] - series[-3]) / 3.0
            else:
                trend = 0.0
            last = series[-1]
            fc = np.array([np.clip(last + trend, 0.0, 1.0)])

        # Tháng dự báo: tháng tiếp theo (mod 12)
        next_month = (current_month % 12) + 1
        forecast_months = np.array([next_month])

        return series, fc, forecast_months


# =============================================================================
# VISUALIZATION
# =============================================================================

class ChartFactory:
    """Tạo các biểu đồ Plotly với theme Premium Green."""

    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=22, color="#1b5e20", family="Arial Black"),
                x=0.5,
            ),
            font=dict(size=15, color="#0d1b2a", family="Arial"),
            margin=dict(l=70, r=40, t=80, b=70),
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#cfd8dc",
                borderwidth=2,
                font=dict(size=13, color="#0d1b2a"),
            ),
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#e0e0e0",
            gridwidth=1,
            linecolor="#90a4ae",
            linewidth=2,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#e0e0e0",
            gridwidth=1,
            linecolor="#90a4ae",
            linewidth=2,
        )
        return fig

    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        colors = ["#2e7d32", "#43a047", "#66bb6a", "#9ccc65", "#c0ca33", "#00897b"]
        labels = [c for c in weights.index]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=weights.values,
                    marker=dict(colors=colors, line=dict(color="white", width=3)),
                    textinfo="percent",
                    textfont=dict(size=14, color="#0d1b2a"),
                    pull=[0.03] * len(weights),
                    hovertemplate="<b>%{label}</b><br>Tỉ trọng: %{value:.2%}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=20, color="#1b5e20", family="Arial Black"),
                x=0.5,
            )
        )
        return fig

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
                        colorscale=[[0, "#c8e6c9"], [0.5, "#43a047"], [1, "#1b5e20"]],
                        line=dict(color="#0d1b2a", width=1.5),
                    ),
                    hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
                )
            ]
        )
        fig.update_xaxes(title="TOPSIS Score", range=[0, 1])
        fig.update_yaxes(title="Công ty")
        return ChartFactory._apply_theme(fig, "🏆 TOPSIS Score (cao hơn = tốt hơn)")

    @staticmethod
    def create_forecast_chart(
        historical: np.ndarray,
        forecast: np.ndarray,
        forecast_months: np.ndarray,
        route: str,
    ) -> go.Figure:
        months_hist = list(range(1, len(historical) + 1))
        months_fc = list(forecast_months)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=months_hist,
                y=historical,
                mode="lines+markers",
                name="📈 Lịch sử",
                line=dict(color="#1b5e20", width=3),
                marker=dict(size=9, color="#2e7d32", line=dict(width=2, color="white")),
                hovertemplate="Tháng %{x}<br>Rủi ro: %{y:.2%}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=months_fc,
                y=forecast,
                mode="lines+markers",
                name="🔮 Dự báo (1 tháng)",
                line=dict(color="#ef6c00", width=3, dash="dash"),
                marker=dict(
                    size=11,
                    color="#ff9800",
                    symbol="diamond",
                    line=dict(width=2, color="white"),
                ),
                hovertemplate="Tháng %{x}<br>Dự báo: %{y:.2%}<extra></extra>",
            )
        )

        fig = ChartFactory._apply_theme(fig, f"📊 Dự báo rủi ro khí hậu tuyến {route}")
        fig.update_xaxes(
            title="Tháng",
            tickmode="linear",
            tickvals=list(range(1, 13)),
            dtick=1,
        )
        fig.update_yaxes(
            title="Mức rủi ro (0–1)",
            range=[0, max(1.0, float(max(historical.max(), forecast.max()) * 1.15))],
            tickformat=".0%",
        )
        return fig

    @staticmethod
    def create_fuzzy_heatmap(fuzzy_table: pd.DataFrame) -> go.Figure:
        """Heatmap Fuzzy: low / mid / high / centroid theo tiêu chí."""
        data = fuzzy_table[["low", "mid", "high", "centroid"]].values
        z_text = np.round(data, 3).astype(str)
        fig = px.imshow(
            data,
            x=["Low", "Mid", "High", "Centroid"],
            y=fuzzy_table.index,
            text=z_text,
            aspect="auto",
            color_continuous_scale="Greens",
        )
        fig.update_traces(texttemplate="%{text}", textfont=dict(size=11, color="black"))
        fig = ChartFactory._apply_theme(fig, "🌿 Fuzzy AHP Heatmap (Low–Mid–High–Centroid)")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Tiêu chí")
        return fig


# =============================================================================
# EXPORT UTILITIES
# =============================================================================

class ReportGenerator:
    """Xuất Excel + PDF để nộp kèm NCKH."""

    @staticmethod
    def generate_pdf(
        results: pd.DataFrame,
        params: AnalysisParams,
        var: Optional[float],
        cvar: Optional[float],
    ) -> bytes:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST v5.1.4 - Executive Summary", 0, 1, "C")
            pdf.ln(5)

            pdf.set_font("Arial", "", 11)
            pdf.cell(
                0,
                6,
                f"Route: {params.route} | Month: {params.month} | Method: {params.method}",
                0,
                1,
            )
            pdf.cell(
                0,
                6,
                f"Cargo Value: ${params.cargo_value:,.0f} | Priority: {params.priority}",
                0,
                1,
            )
            pdf.ln(4)

            top = results.iloc[0]
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"Top Recommendation: {top['company']}", 0, 1)
            pdf.set_font("Arial", "", 11)
            pdf.cell(
                0,
                6,
                f"Score: {top['score']:.3f} | Confidence: {top['confidence']:.2f} | ICC: {top['recommend_icc']}",
                0,
                1,
            )
            pdf.ln(4)

            pdf.set_font("Arial", "B", 10)
            pdf.cell(20, 6, "Rank", 1)
            pdf.cell(55, 6, "Company", 1)
            pdf.cell(25, 6, "Score", 1)
            pdf.cell(30, 6, "Confidence", 1)
            pdf.cell(30, 6, "ICC", 1, 1)

            pdf.set_font("Arial", "", 9)
            for _, row in results.head(5).iterrows():
                pdf.cell(20, 6, str(int(row["rank"])), 1)
                pdf.cell(55, 6, str(row["company"])[:22], 1)
                pdf.cell(25, 6, f"{row['score']:.3f}", 1)
                pdf.cell(30, 6, f"{row['confidence']:.2f}", 1)
                pdf.cell(30, 6, str(row["recommend_icc"]), 1, 1)

            if var is not None and cvar is not None:
                pdf.ln(5)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(
                    0,
                    6,
                    f"VaR 95%: ${var:,.0f}   |   CVaR 95%: ${cvar:,.0f}",
                    0,
                    1,
                )

            return pdf.output(dest="S").encode("latin1")
        except Exception as e:
            st.error(f"Lỗi tạo PDF: {e}")
            return b""

    @staticmethod
    def generate_excel(
        results: pd.DataFrame,
        data: pd.DataFrame,
        weights: pd.Series,
        fuzzy_table: Optional[pd.DataFrame],
    ) -> bytes:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Results", index=False)
            data.to_excel(writer, sheet_name="Data")
            pd.DataFrame({"weight": weights.values}, index=weights.index).to_excel(
                writer, sheet_name="Weights"
            )
            if fuzzy_table is not None:
                fuzzy_table.to_excel(writer, sheet_name="Fuzzy_AHP")
        buffer.seek(0)
        return buffer.getvalue()


# =============================================================================
# APPLICATION CONTROLLER
# =============================================================================

class AnalysisController:
    """Orchestrate toàn bộ pipeline phân tích."""

    def __init__(self):
        self.data_service = DataService()
        self.weight_manager = WeightManager()
        self.fuzzy_ahp = FuzzyAHP()
        self.mc_simulator = MonteCarloSimulator()
        self.topsis = TOPSISAnalyzer()
        self.risk_calc = RiskCalculator()
        self.forecaster = Forecaster()

    def run_analysis(self, params: AnalysisParams, historical: pd.DataFrame) -> AnalysisResult:
        # Trọng số gốc từ session
        base_weights = pd.Series(st.session_state["weights"], index=CRITERIA)

        fuzzy_table = None
        if params.use_fuzzy:
            weights, fuzzy_table = self.fuzzy_ahp.apply(base_weights, params.fuzzy_uncertainty)
        else:
            weights = base_weights.copy()

        company_data = self.data_service.get_company_data()

        # Rủi ro khí hậu base
        if params.month in historical["month"].values:
            base_risk = float(
                historical.loc[historical["month"] == params.month, params.route].iloc[0]
            )
        else:
            base_risk = 0.4

        # Monte Carlo cho C6
        if params.use_mc:
            companies, mc_mean, mc_std = self.mc_simulator.simulate(
                base_risk,
                SENSITIVITY_MAP,
                params.mc_runs,
            )
            order = [companies.index(c) for c in company_data.index]
            mc_mean = mc_mean[order]
            mc_std = mc_std[order]
        else:
            mc_mean = np.zeros(len(company_data))
            mc_std = np.zeros(len(company_data))

        data_adjusted = company_data.copy()
        data_adjusted["C6: Rủi ro khí hậu"] = mc_mean

        # Cargo giá trị lớn → phí tăng nhẹ
        if params.cargo_value > 50000:
            data_adjusted["C1: Tỷ lệ phí"] *= 1.1

        # TOPSIS
        scores = self.topsis.analyze(data_adjusted, weights, COST_BENEFIT_MAP)

        results = (
            pd.DataFrame(
                {
                    "company": data_adjusted.index,
                    "score": scores,
                    "C6_mean": mc_mean,
                    "C6_std": mc_std,
                }
            )
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )
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
                results["C6_mean"].values,
                params.cargo_value,
            )

        hist_series, fc_values, fc_months = self.forecaster.forecast(
            historical,
            params.route,
            current_month=params.month,
            months_ahead=1,
            use_arima=params.use_arima,
        )

        return AnalysisResult(
            results=results,
            weights=weights,
            data_adjusted=data_adjusted,
            var=var,
            cvar=cvar,
            historical=hist_series,
            forecast=fc_values,
            forecast_months=fc_months,
            fuzzy_table=fuzzy_table,
        )


# =============================================================================
# STREAMLIT UI
# =============================================================================

class StreamlitUI:
    """Quản lý toàn bộ UI Streamlit."""

    def __init__(self):
        self.controller = AnalysisController()
        self.chart_factory = ChartFactory()
        self.report_gen = ReportGenerator()

    def initialize(self):
        st.set_page_config(
            page_title="RISKCAST v5.1.4 — ESG Risk Assessment",
            page_icon="🛡️",
            layout="wide",
        )
        apply_custom_css()
        if "weights" not in st.session_state:
            st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
        if "locked" not in st.session_state:
            st.session_state["locked"] = [False] * len(CRITERIA)

    def render_sidebar(self) -> AnalysisParams:
        with st.sidebar:
            st.header("📊 Thông tin lô hàng")

            cargo_value = st.number_input(
                "Giá trị lô hàng (USD)", min_value=1000, value=39000, step=1000
            )
            good_type = st.selectbox(
                "Loại hàng",
                ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"],
            )
            route = st.selectbox(
                "Tuyến",
                ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"],
            )
            method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
            month = st.selectbox("Tháng", list(range(1, 13)), index=8)
            priority = st.selectbox(
                "Ưu tiên",
                ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"],
            )

            st.markdown("---")
            st.header("⚙️ Cấu hình mô hình")

            use_fuzzy = st.checkbox("Bật Fuzzy AHP", True)
            use_arima = st.checkbox("Dùng ARIMA cho dự báo C6", True)
            use_mc = st.checkbox("Chạy Monte Carlo cho C6", True)
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

            # Highlight tiêu chí dao động mạnh nhất
            strongest = fuzzy_df["range"].idxmax()
            max_range = float(fuzzy_df["range"].max())
            st.markdown(
                f"""
                <div class="explanation-box">
                    <h4>🔥 Tiêu chí dao động mạnh nhất (Fuzzy):</h4>
                    <p>
                        <b>{strongest}</b> có biên độ Fuzzy (High–Low) lớn nhất: 
                        <b>{max_range:.3f}</b>. Điều này có nghĩa đây là tiêu chí mà 
                        đánh giá chuyên gia còn nhiều bất định → cần giải thích kỹ hơn 
                        trong phần thuyết trình.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_fuzzy = self.chart_factory.create_fuzzy_heatmap(fuzzy_df)
            st.plotly_chart(fig_fuzzy, use_container_width=True)

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

    def run(self):
        self.initialize()

        st.title("🚢 RISKCAST v5.1.4 — ESG Logistics Risk Assessment")
        st.markdown("**Decision Support System cho lựa chọn bảo hiểm vận tải quốc tế (Fuzzy + Monte Carlo + VaR/CVaR).**")
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
