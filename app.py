# =============================================================================
# RISKCAST v5.1 — ESG Logistics Risk Assessment Dashboard (PREMIUM GREEN UI)
# Clean Architecture + Fuzzy AHP + TOPSIS + Monte Carlo + VaR/CVaR + ARIMA
# Tác giả gốc: Bùi Xuân Hoàng  |  Refactor & UI: Kai
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
    """Loại tiêu chí: chi phí (càng thấp càng tốt) hay lợi ích (càng cao càng tốt)."""
    COST = "cost"
    BENEFIT = "benefit"


@dataclass
class AnalysisParams:
    """Các tham số đầu vào chính của mô hình."""
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
    """Kết quả phân tích tổng hợp."""
    results: pd.DataFrame
    weights: pd.Series
    data_adjusted: pd.DataFrame
    var: Optional[float]
    cvar: Optional[float]
    historical: np.ndarray
    forecast: np.ndarray


# Danh sách tiêu chí
CRITERIA = [
    "C1: Tỷ lệ phí",
    "C2: Thời gian xử lý",
    "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC",
    "C5: Chăm sóc KH",
    "C6: Rủi ro khí hậu",
]

# Trọng số mặc định (đã chuẩn hóa = 1)
DEFAULT_WEIGHTS = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])

# Mapping cost / benefit
COST_BENEFIT_MAP = {
    "C1: Tỷ lệ phí":    CriterionType.COST,
    "C2: Thời gian xử lý": CriterionType.COST,
    "C3: Tỷ lệ tổn thất":   CriterionType.COST,
    "C4: Hỗ trợ ICC":       CriterionType.BENEFIT,
    "C5: Chăm sóc KH":      CriterionType.BENEFIT,
    "C6: Rủi ro khí hậu":   CriterionType.COST,
}

# Độ nhạy cảm khí hậu của từng công ty bảo hiểm
SENSITIVITY_MAP = {
    "Chubb": 0.95,
    "PVI": 1.10,
    "InternationalIns": 1.20,
    "BaoViet": 1.05,
    "Aon": 0.90,
}

# Bảng màu ESG Green
COLORS = {
    "bg_grad_top": "#021F18",
    "bg_grad_bottom": "#013220",
    "card_bg": "#06271D",
    "primary": "#00C853",      # Green main
    "primary_dark": "#009624",
    "accent": "#A7FFEB",      # Mint accent
    "warning": "#FFC400",
    "danger": "#FF5252",
    "text_main": "#E8F5E9",
    "text_muted": "#B2DFDB",
    "border": "#1B5E20",
}

# =============================================================================
# UI STYLING (PREMIUM GREEN)
# =============================================================================

def apply_custom_css() -> None:
    """CSS giao diện Premium ESG Green, sidebar trắng, nội dung dark green."""
    st.markdown(
        f"""
    <style>
        * {{
            text-rendering: optimizeLegibility !important;
            -webkit-font-smoothing: antialiased !important;
        }}
        .stApp {{
            background: radial-gradient(circle at 0% 0%, #004D40 0%, #000000 55%);
            font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
        }}
        .block-container {{
            background: linear-gradient(145deg, {COLORS["bg_grad_top"]} 0%, {COLORS["bg_grad_bottom"]} 70%);
            border-radius: 18px;
            padding: 1.8rem 2.4rem !important;
            margin-top: 1.5rem;
            box-shadow: 0 24px 60px rgba(0,0,0,0.65);
            border: 1px solid {COLORS["border"]};
            max-width: 1500px;
        }}

        /* Typography main */
        h1 {{
            color: {COLORS["accent"]} !important;
            font-weight: 900 !important;
            letter-spacing: .06em;
            text-transform: uppercase;
            font-size: 2.4rem !important;
        }}
        h2 {{
            color: {COLORS["text_main"]} !important;
            font-weight: 800 !important;
            font-size: 1.8rem !important;
        }}
        h3 {{
            color: {COLORS["text_main"]} !important;
            font-weight: 700 !important;
            font-size: 1.35rem !important;
        }}

        p, span, div, label, .stMarkdown {{
            color: {COLORS["text_muted"]} !important;
            font-weight: 500;
        }}

        /* Buttons */
        .stButton > button[kind="primary"],
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["primary_dark"]}) !important;
            color: #001510 !important;
            border-radius: 999px !important;
            padding: 0.7rem 1.8rem !important;
            font-weight: 800 !important;
            border: 1px solid #00FF95 !important;
            box-shadow: 0 10px 30px rgba(0, 255, 149, 0.35);
            text-transform: uppercase;
            letter-spacing: .08em;
            font-size: 0.9rem !important;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px) scale(1.01);
            box-shadow: 0 18px 40px rgba(0, 255, 149, 0.55);
        }}

        /* Sidebar: nền trắng, chữ đen */
        section[data-testid="stSidebar"] {{
            background: #FFFFFF !important;
            border-right: 3px solid #00C853;
        }}
        section[data-testid="stSidebar"] h2 {{
            color: #00695C !important;
            font-weight: 900 !important;
            font-size: 1.35rem !important;
            padding: 0.5rem 0.3rem;
            border-bottom: 2px solid #A7FFEB;
            margin-bottom: 0.9rem;
        }}
        section[data-testid="stSidebar"] label {{
            color: #004D40 !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
        }}
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] select {{
            background: #FFFFFF !important;
            color: #000000 !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            border: 1.5px solid #B0BEC5 !important;
        }}
        section[data-testid="stSidebar"] input:focus,
        section[data-testid="stSidebar"] select:focus {{
            outline: none !important;
            border: 1.5px solid #00C853 !important;
            box-shadow: 0 0 0 2px rgba(0,200,83,0.25);
        }}

        /* Dataframe */
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #004D40;
            box-shadow: 0 10px 35px rgba(0,0,0,0.65);
        }}
        .stDataFrame thead tr th {{
            background-color: #004D40 !important;
            color: #E0F2F1 !important;
            font-weight: 800 !important;
        }}
        .stDataFrame tbody tr td {{
            background-color: #001E1A;
            color: #E0F2F1 !important;
            font-weight: 600;
        }}

        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: #00E676 !important;
            font-weight: 900 !important;
            font-size: 2rem !important;
            text-shadow: 0 0 18px rgba(0,230,118,0.8);
        }}
        [data-testid="stMetricLabel"] {{
            color: {COLORS["text_main"]} !important;
            font-weight: 700 !important;
        }}

        /* Explanation cards */
        .explanation-box {{
            background: rgba(1, 67, 55, 0.9);
            border-radius: 12px;
            border: 1px solid #00C853;
            padding: 1rem 1.3rem;
            margin: 0.8rem 0 1.1rem 0;
            box-shadow: 0 12px 30px rgba(0,0,0,0.6);
        }}
        .explanation-box h4 {{
            color: #A7FFEB !important;
            font-weight: 800 !important;
            margin-bottom: 0.6rem;
        }}
        .explanation-box li {{
            color: {COLORS["text_muted"]} !important;
            margin: 0.3rem 0;
        }}

    </style>
    """,
        unsafe_allow_html=True,
    )

# =============================================================================
# DATA LAYER
# =============================================================================

class DataService:
    """Quản lý dữ liệu (tuyến, rủi ro khí hậu, dữ liệu doanh nghiệp bảo hiểm)."""

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_historical_data() -> pd.DataFrame:
        """Sinh dữ liệu lịch sử rủi ro khí hậu giả lập."""
        climate_base = {
            "VN - EU":        [0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.42, 0.48, 0.60, 0.68, 0.58, 0.45],
            "VN - US":        [0.30, 0.33, 0.36, 0.40, 0.45, 0.50, 0.56, 0.62, 0.75, 0.72, 0.60, 0.52],
            "VN - Singapore": [0.15, 0.16, 0.18, 0.20, 0.22, 0.26, 0.30, 0.32, 0.35, 0.34, 0.28, 0.25],
            "VN - China":     [0.18, 0.19, 0.21, 0.24, 0.26, 0.30, 0.34, 0.36, 0.40, 0.38, 0.32, 0.28],
            "Domestic":       [0.10] * 12,
        }

        df = pd.DataFrame({"month": list(range(1, 13))})
        for route, values in climate_base.items():
            df[route] = values
        return df

    @staticmethod
    @st.cache_data
    def get_company_data() -> pd.DataFrame:
        """Dữ liệu cơ bản của các công ty bảo hiểm (demo)."""
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
    """Quản lý & cân bằng trọng số tiêu chí."""

    @staticmethod
    def auto_balance(weights: np.ndarray, locked: List[bool]) -> np.ndarray:
        """Chuẩn hóa trọng số về tổng = 1, tôn trọng những tiêu chí đã lock."""
        w = np.array(weights, dtype=float)
        locked_flags = np.array(locked, dtype=bool)

        total_locked = w[locked_flags].sum()
        free_idx = np.where(~locked_flags)[0]

        if len(free_idx) == 0:
            return w / (w.sum() or 1.0)

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
    """Xử lý bất định trọng số bằng tam giác mờ (TFN)."""

    @staticmethod
    def apply(weights: pd.Series, uncertainty_pct: float) -> pd.Series:
        factor = uncertainty_pct / 100.0
        w = weights.values

        low = np.maximum(w * (1 - factor), 1e-9)
        high = np.minimum(w * (1 + factor), 0.9999)

        # Defuzzify bằng công thức trung tâm trọng lực
        defuzzified = (low + w + high) / 3.0
        normalized = defuzzified / defuzzified.sum()

        return pd.Series(normalized, index=weights.index)


class MonteCarloSimulator:
    """Mô phỏng Monte Carlo cho rủi ro khí hậu."""

    @staticmethod
    @st.cache_data(ttl=600)
    def simulate(
        base_risk: float, sensitivity_map: Dict[str, float], n_simulations: int
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())

        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.03, mu * 0.12)

        simulations = rng.normal(loc=mu, scale=sigma, size=(n_simulations, len(companies)))
        simulations = np.clip(simulations, 0.0, 1.0)

        return companies, simulations.mean(axis=0), simulations.std(axis=0)


class TOPSISAnalyzer:
    """Thuật toán TOPSIS để xếp hạng phương án."""

    @staticmethod
    def analyze(
        data: pd.DataFrame, weights: pd.Series, cost_benefit: Dict[str, CriterionType]
    ) -> np.ndarray:
        # Ma trận quyết định
        M = data[list(weights.index)].values.astype(float)

        # Chuẩn hóa
        denom = np.sqrt((M ** 2).sum(axis=0))
        denom[denom == 0] = 1.0
        R = M / denom

        # Nhân trọng số
        V = R * weights.values

        # Xác định điểm lý tưởng tốt / xấu
        is_cost = np.array([cost_benefit[c] == CriterionType.COST for c in weights.index])
        ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
        ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))

        d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

        scores = d_minus / (d_plus + d_minus + 1e-12)
        return scores


class RiskCalculator:
    """Tính VaR/CVaR và độ tin cậy."""

    @staticmethod
    def calculate_var_cvar(
        loss_rates: np.ndarray, cargo_value: float, confidence: float = 0.95
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

        # Độ biến thiên rủi ro khí hậu (C6)
        cv_c6 = results["C6_std"].values / (results["C6_mean"].values + eps)
        conf_c6 = 1.0 / (1.0 + cv_c6)
        conf_c6 = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + eps)

        # Độ biến thiên các tiêu chí khác
        crit_cv = data.std(axis=1).values / (data.mean(axis=1).values + eps)
        conf_crit = 1.0 / (1.0 + crit_cv)
        conf_crit = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (np.ptp(conf_crit) + eps)

        return np.sqrt(conf_c6 * conf_crit)


class Forecaster:
    """Dự báo rủi ro khí hậu bằng ARIMA hoặc xu hướng tuyến tính."""

    @staticmethod
    def forecast(
        historical: pd.DataFrame,
        route: str,
        months_ahead: int = 3,
        use_arima: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if route not in historical.columns:
            route = historical.columns[1]

        series = historical[route].values

        if use_arima and ARIMA_AVAILABLE and len(series) >= 6:
            try:
                model = ARIMA(series, order=(1, 1, 1))
                fitted = model.fit()
                forecast = np.clip(fitted.forecast(months_ahead), 0, 1)
                return series, np.asarray(forecast)
            except Exception:
                pass

        # fallback
        trend = (series[-1] - series[-3]) / 3.0 if len(series) >= 3 else 0.0
        forecast = np.array(
            [np.clip(series[-1] + (i + 1) * trend, 0, 1) for i in range(months_ahead)]
        )
        return series, forecast

# =============================================================================
# VISUALIZATION
# =============================================================================

class ChartFactory:
    """Tạo các biểu đồ Plotly với theme ESG green."""

    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=20, color="#E8F5E9", family="Inter, sans-serif"),
            ),
            font=dict(size=14, color="#E0F2F1"),
            paper_bgcolor="#001712",
            plot_bgcolor="#001712",
            margin=dict(l=65, r=40, t=70, b=60),
            legend=dict(
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="#004D40",
                borderwidth=1,
                font=dict(size=13, color="#E0F2F1"),
            ),
        )
        fig.update_xaxes(
            gridcolor="#004D40",
            zerolinecolor="#004D40",
            linecolor="#00796B",
        )
        fig.update_yaxes(
            gridcolor="#004D40",
            zerolinecolor="#004D40",
            linecolor="#00796B",
        )
        return fig

    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        colors = ["#00C853", "#69F0AE", "#AEEA00", "#FFEB3B", "#FFAB00", "#FF6D00"]

        full_labels = list(weights.index)
        short_labels = [c.split(":")[0] for c in weights.index]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=full_labels,
                    values=weights.values,
                    marker=dict(colors=colors, line=dict(color="#001712", width=3)),
                    text=short_labels,
                    textinfo="text+percent",
                    textposition="inside",
                    insidetextorientation="radial",
                    pull=[0.05] * len(weights),
                    hovertemplate="<b>%{label}</b><br>Tỉ trọng: %{percent}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=20, color="#A7FFEB", family="Inter, sans-serif"),
                x=0.5,
            ),
            showlegend=True,
            legend=dict(
                title=dict(
                    text="<b>Các tiêu chí</b>",
                    font=dict(size=14, color="#A7FFEB"),
                ),
                font=dict(size=13, color="#E0F2F1"),
                bgcolor="rgba(0,23,18,0.9)",
                bordercolor="#00C853",
                borderwidth=1,
            ),
            paper_bgcolor="#001712",
            plot_bgcolor="#001712",
        )

        return fig

    @staticmethod
    def create_topsis_bar(results: pd.DataFrame) -> go.Figure:
        df_sorted = results.sort_values("score")
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_sorted["score"],
                    y=df_sorted["company"],
                    orientation="h",
                    text=df_sorted["score"].apply(lambda x: f"{x:.3f}"),
                    textposition="outside",
                    marker=dict(
                        color=df_sorted["score"],
                        colorscale=[[0, "#004D40"], [0.5, "#00C853"], [1, "#B2FF59"]],
                        line=dict(color="#001712", width=2),
                    ),
                    hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
                )
            ]
        )
        fig.update_xaxes(range=[0, 1], title="Điểm TOPSIS")
        fig.update_yaxes(title="Công ty bảo hiểm")
        return ChartFactory._apply_theme(fig, "🏆 Điểm TOPSIS (cao hơn = tốt hơn)")

    @staticmethod
    def create_forecast_chart(
        historical: np.ndarray, forecast: np.ndarray, route: str
    ) -> go.Figure:
        months_hist = list(range(1, len(historical) + 1))
        months_fc = list(
            range(len(historical) + 1, len(historical) + len(forecast) + 1)
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=months_hist,
                y=historical,
                mode="lines+markers",
                name="Lịch sử",
                line=dict(color="#00E676", width=3),
                marker=dict(size=9),
                hovertemplate="Tháng %{x}<br>Rủi ro: %{y:.1%}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=months_fc,
                y=forecast,
                mode="lines+markers",
                name="Dự báo",
                line=dict(color="#FFCA28", width=3, dash="dash"),
                marker=dict(size=10, symbol="diamond"),
                hovertemplate="Tháng %{x}<br>Dự báo: %{y:.1%}<extra></extra>",
            )
        )

        fig.update_xaxes(title="Tháng")
        fig.update_yaxes(
            title="Mức rủi ro (0–1)",
            range=[0, max(1, float(max(historical.max(), forecast.max()) * 1.15))],
            tickformat=".0%",
        )

        return ChartFactory._apply_theme(fig, f"Dự báo rủi ro khí hậu — {route}")

# =============================================================================
# EXPORT UTILITIES
# =============================================================================

class ReportGenerator:
    """Xuất Excel / PDF cho hội đồng & khách hàng."""

    @staticmethod
    def generate_pdf(
        results: pd.DataFrame, params: AnalysisParams, var: Optional[float], cvar: Optional[float]
    ) -> bytes:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST v5.1 - Executive Summary", 0, 1, "C")
            pdf.ln(4)

            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, f"Route: {params.route} | Month: {params.month} | Method: {params.method}", 0, 1)
            pdf.cell(0, 6, f"Cargo Value: ${params.cargo_value:,.0f}", 0, 1)
            pdf.cell(0, 6, f"Priority: {params.priority}", 0, 1)
            pdf.ln(4)

            top = results.iloc[0]
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, f"Top Recommendation: {top['company']}", 0, 1)
            pdf.set_font("Arial", "", 11)
            pdf.cell(
                0,
                6,
                f"Score: {top['score']:.3f} | Confidence: {top['confidence']:.2f} | ICC: {top['recommend_icc']}",
                0,
                1,
            )
            pdf.ln(4)

            pdf.set_font("Arial", "B", 11)
            pdf.cell(15, 7, "Rank", 1)
            pdf.cell(55, 7, "Company", 1)
            pdf.cell(25, 7, "Score", 1)
            pdf.cell(25, 7, "Conf.", 1)
            pdf.cell(25, 7, "ICC", 1, 1)

            pdf.set_font("Arial", "", 10)
            for _, row in results.head(5).iterrows():
                pdf.cell(15, 7, str(int(row["rank"])), 1)
                pdf.cell(55, 7, str(row["company"])[:20], 1)
                pdf.cell(25, 7, f"{row['score']:.3f}", 1)
                pdf.cell(25, 7, f"{row['confidence']:.2f}", 1)
                pdf.cell(25, 7, str(row["recommend_icc"]), 1, 1)

            if var and cvar:
                pdf.ln(4)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 7, f"VaR 95%: ${var:,.0f}", 0, 1)
                pdf.cell(0, 7, f"CVaR 95%: ${cvar:,.0f}", 0, 1)

            return pdf.output(dest="S").encode("latin1")
        except Exception as e:
            st.error(f"PDF generation error: {e}")
            return b""

    @staticmethod
    def generate_excel(
        results: pd.DataFrame, data: pd.DataFrame, weights: pd.Series
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
# CONTROLLER
# =============================================================================

class AnalysisController:
    """Orchestrator: nhận params, chạy mô hình, trả về AnalysisResult."""

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

        base_risk = (
            float(
                historical.loc[historical["month"] == params.month, params.route].iloc[0]
            )
            if params.month in historical["month"].values
            else 0.4
        )

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

        # Nếu giá trị lô hàng lớn, phí thường tăng nhẹ
        if params.cargo_value > 50_000:
            data_adjusted["C1: Tỷ lệ phí"] *= 1.1

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
                results["C6_mean"].values, params.cargo_value
            )

        hist_series, forecast = self.forecaster.forecast(
            historical, params.route, use_arima=params.use_arima
        )

        return AnalysisResult(
            results=results,
            weights=weights,
            data_adjusted=data_adjusted,
            var=var,
            cvar=cvar,
            historical=hist_series,
            forecast=forecast,
        )

# =============================================================================
# STREAMLIT UI
# =============================================================================

class StreamlitUI:
    def __init__(self):
        self.controller = AnalysisController()
        self.chart_factory = ChartFactory()
        self.report_gen = ReportGenerator()

    def initialize(self):
        st.set_page_config(
            page_title="RISKCAST v5.1 — ESG Risk Assessment",
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
            st.header("📦 Thông tin lô hàng")

            cargo_value = st.number_input(
                "Giá trị lô hàng (USD)",
                min_value=1000,
                value=39_000,
                step=1000,
            )
            good_type = st.selectbox(
                "Loại hàng",
                ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"],
            )
            route = st.selectbox(
                "Tuyến vận chuyển",
                ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"],
            )
            method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
            month = st.selectbox("Tháng", list(range(1, 13)), index=8)
            priority = st.selectbox(
                "Ưu tiên của khách", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"]
            )

            st.markdown("---")
            st.header("⚙️ Cấu hình mô hình")

            use_fuzzy = st.checkbox("Bật Fuzzy AHP (xử lý bất định)", True)
            use_arima = st.checkbox("Dùng ARIMA cho dự báo khí hậu", True)
            use_mc = st.checkbox("Mô phỏng Monte Carlo cho C6", True)
            use_var = st.checkbox("Tính VaR/CVaR cho lô hàng", True)

            mc_runs = st.number_input("Số vòng Monte Carlo", 500, 10_000, 2000, 500)
            fuzzy_uncertainty = (
                st.slider("Biên độ Fuzzy (%)", 0, 50, 15) if use_fuzzy else 15
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

        cols = st.columns(len(CRITERIA))
        new_weights = st.session_state["weights"].copy()

        for i, crit in enumerate(CRITERIA):
            with cols[i]:
                short = crit.split(":")[0]
                desc = crit.split(":")[1].strip()

                st.markdown(
                    f"""
                    <div style="background:rgba(0,150,136,0.18); padding:6px 8px;
                                border-radius:10px; border:1px solid #00BFA5;
                                text-align:center; margin-bottom:4px;">
                        <div style="color:#A7FFEB; font-weight:800; font-size:0.95rem;">
                            {short}
                        </div>
                        <div style="color:#E0F2F1; font-size:0.78rem;">
                            {desc}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                is_locked = st.checkbox(
                    "Khóa", value=st.session_state["locked"][i], key=f"lock_{i}"
                )
                st.session_state["locked"][i] = is_locked

                weight_val = st.number_input(
                    "Tỉ lệ",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(new_weights[i]),
                    step=0.01,
                    key=f"weight_{i}",
                    label_visibility="collapsed",
                )
                new_weights[i] = weight_val

                st.markdown(
                    f"""
                    <div style="background:#00332A; padding:4px 0;
                                border-radius:999px; border:1px solid #00C853;
                                text-align:center; margin-top:2px;">
                        <span style="color:#69F0AE; font-weight:800; font-size:0.95rem;">
                            {weight_val:.0%}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        col_reset, col_info = st.columns([1, 2])
        with col_reset:
            if st.button("🔄 Reset mặc định", use_container_width=True):
                st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
                st.session_state["locked"] = [False] * len(CRITERIA)
                st.experimental_rerun()

        with col_info:
            total = float(sum(new_weights))
            if abs(total - 1.0) > 0.01:
                st.warning(f"⚠️ Tổng trọng số hiện tại ≈ {total:.2f} (sẽ tự cân bằng về 1.0).")
            else:
                st.success(f"✅ Tổng trọng số ≈ {total:.2f} (hợp lệ).")

        st.session_state["weights"] = WeightManager.auto_balance(
            new_weights, st.session_state["locked"]
        )

    def display_results(self, result: AnalysisResult, params: AnalysisParams):
        st.success("✅ Đã hoàn tất phân tích RISKCAST v5.1!")

        left, right = st.columns([2, 1])

        with left:
            st.subheader("🏅 Bảng xếp hạng công ty bảo hiểm")

            df_show = result.results[
                ["rank", "company", "score", "confidence", "recommend_icc"]
            ].set_index("rank")
            df_show.columns = ["Công ty", "Điểm số", "Độ tin cậy", "ICC"]
            st.dataframe(df_show, use_container_width=True)

            top = result.results.iloc[0]
            st.markdown(
                f"""
                <div class="explanation-box" style="border-color:#00E676;">
                    <h4>🏆 Khuyến nghị hàng đầu</h4>
                    <ul>
                        <li><b>Công ty:</b> {top['company']}</li>
                        <li><b>Score:</b> {top['score']:.3f} | 
                            <b>Confidence:</b> {top['confidence']:.2f}</li>
                        <li><b>Gói ICC gợi ý:</b> {top['recommend_icc']}</li>
                        <li><b>Tuyến:</b> {params.route} | <b>Giá trị lô hàng:</b> ${params.cargo_value:,.0f}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            if result.var and result.cvar:
                st.metric(
                    "💰 VaR 95%",
                    f"${result.var:,.0f}",
                    help="Tổn thất tối đa với độ tin cậy 95%.",
                )
                st.metric(
                    "🛡️ CVaR 95%",
                    f"${result.cvar:,.0f}",
                    help="Tổn thất trung bình khi đã vượt VaR.",
                )

            fig_weights = self.chart_factory.create_weights_pie(
                result.weights, "Cơ cấu trọng số cuối cùng"
            )
            st.plotly_chart(fig_weights, use_container_width=True)

        # Giải thích chi tiết
        st.markdown("---")
        st.subheader("📋 Giải thích chi tiết cho hội đồng / khách hàng")

        top3 = result.results.head(3)

        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>🎯 Vì sao {top3.iloc[0]['company']} đứng #1?</h4>
                <ul>
                    <li>Điểm TOPSIS cao nhất: <b>{top3.iloc[0]['score']:.3f}</b></li>
                    <li>Độ tin cậy mô hình: <b>{top3.iloc[0]['confidence']:.2f}</b></li>
                    <li>Rủi ro khí hậu trung bình: <b>{top3.iloc[0]['C6_mean']:.2%}</b>
                        (±{top3.iloc[0]['C6_std']:.2%})</li>
                    <li>Phù hợp với ưu tiên khách hàng: <b>{params.priority}</b></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>📊 So sánh Top 3 phương án</h4>
                <ul>
                    <li><b>#{1} {top3.iloc[0]['company']}</b> — Score {top3.iloc[0]['score']:.3f}</li>
                    <li><b>#{2} {top3.iloc[1]['company']}</b> — Score {top3.iloc[1]['score']:.3f}
                        (kém {top3.iloc[0]['score'] - top3.iloc[1]['score']:.3f} điểm)</li>
                    <li><b>#{3} {top3.iloc[2]['company']}</b> — Score {top3.iloc[2]['score']:.3f}
                        (kém {top3.iloc[0]['score'] - top3.iloc[2]['score']:.3f} điểm)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result.var and result.cvar:
            st.markdown(
                f"""
                <div class="explanation-box">
                    <h4>⚠️ Đánh giá rủi ro tài chính (VaR/CVaR)</h4>
                    <ul>
                        <li>VaR 95% ≈ <b>${result.var:,.0f}</b></li>
                        <li>CVaR 95% ≈ <b>${result.cvar:,.0f}</b></li>
                        <li>Tỷ lệ VaR / Giá trị hàng ≈ 
                            <b>{result.var / params.cargo_value * 100:.1f}%</b></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Charts
        st.markdown("---")
        st.subheader("📈 Biểu đồ phân tích")

        fig_topsis = self.chart_factory.create_topsis_bar(result.results)
        st.plotly_chart(fig_topsis, use_container_width=True)

        fig_forecast = self.chart_factory.create_forecast_chart(
            result.historical, result.forecast, params.route
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Export
        st.markdown("---")
        st.subheader("📥 Xuất báo cáo")

        c1, c2 = st.columns(2)
        with c1:
            excel_bytes = self.report_gen.generate_excel(
                result.results, result.data_adjusted, result.weights
            )
            st.download_button(
                "📊 Tải file Excel",
                data=excel_bytes,
                file_name=f"riskcast_v5_1_{params.route.replace(' - ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with c2:
            pdf_bytes = self.report_gen.generate_pdf(
                result.results, params, result.var, result.cvar
            )
            if pdf_bytes:
                st.download_button(
                    "📄 Tải báo cáo PDF",
                    data=pdf_bytes,
                    file_name=f"riskcast_v5_1_{params.route.replace(' - ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    def run(self):
        self.initialize()

        st.title("🚢 RISKCAST v5.1 — ESG Logistics Risk Assessment")
        st.markdown(
            "Hệ thống hỗ trợ ra quyết định bảo hiểm vận tải quốc tế kết hợp "
            "**Fuzzy AHP – TOPSIS – Monte Carlo – VaR/CVaR – ARIMA**."
        )
        st.markdown("---")

        historical = DataService.load_historical_data()
        params = self.render_sidebar()
        self.render_weight_controls()

        st.markdown("---")
        weights_series = pd.Series(st.session_state["weights"], index=CRITERIA)
        fig_current = self.chart_factory.create_weights_pie(
            weights_series, "Trọng số hiện tại (sau auto-balance)"
        )
        st.plotly_chart(fig_current, use_container_width=True)

        st.markdown("---")
        if st.button("🚀 PHÂN TÍCH & GỢI Ý", type="primary", use_container_width=True):
            with st.spinner("Đang chạy mô hình RISKCAST v5.1..."):
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
