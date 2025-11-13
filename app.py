# =============================================================================
# RISKCAST v5.2 — ENTERPRISE EDITION
# GIAO DIỆN: Salesforce Lightning + Oracle Fusion + Bloomberg Terminal
# RESPONSIVE: Hybrid Enterprise (Desktop + Mobile)
# TỐI ƯU: NCKH cấp Bộ | Startup Pitch | S24 Ultra 2K
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
# ENTERPRISE CSS — SALESFORCE + ORACLE + BLOOMBERG
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
    .sidebar-icon { font-size: 1.9rem; margin-right: 0.6rem; }
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
    .rank-badge {
        background: var(--primary); color: #001a0f; padding: 0.5rem 0.9rem; border-radius: 999px;
        font-weight: 900; font-size: 0.95rem; box-shadow: 0 0 10px var(--neon);
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
# DATA & CORE (giữ nguyên từ v5.1.5)
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

# ... (giữ nguyên toàn bộ class WeightManager, FuzzyAHP, MonteCarloSimulator, TOPSISAnalyzer, RiskCalculator, Forecaster)

# =============================================================================
# FUZZY VISUAL UTILITIES (PREMIUM GREEN)
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
    diff_map = {crit: float(w * (1 + factor) - w * (1 - factor)) for crit, w in weights.items()}
    most_unc = max(diff_map, key=diff_map.get)
    return most_unc, diff_map

def fuzzy_heatmap_premium(diff_map: Dict[str, float]) -> go.Figure:
    values = list(diff_map.values())
    labels = list(diff_map.keys())
    fig = px.imshow([values], x=labels, y=[""], color_continuous_scale=[
        [0.0, "#00331F"], [0.2, "#006642"], [0.4, "#00AA66"], [0.6, "#00DD88"], [1.0, "#00FFAA"]
    ])
    fig.update_layout(
        title="<b>Heatmap mức dao động Fuzzy (Premium Green)</b>",
        paper_bgcolor="#001a12", plot_bgcolor="#001a12", margin=dict(l=40, r=40, t=80, b=40),
        coloraxis_colorbar=dict(title="Dao động", tickfont=dict(color="#CCFFE6"))
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(showticklabels=False)
    return fig

def fuzzy_chart_premium(weights: pd.Series, fuzzy_pct: float) -> go.Figure:
    factor = fuzzy_pct / 100.0
    labels = list(weights.index)
    low_vals = [max(float(w) * (1 - factor), 0.0) for w in weights]
    mid_vals = [float(w) for w in weights]
    high_vals = [min(float(w) * (1 + factor), 1.0) for w in weights]
    fig = go.Figure()
    for vals, name, color, dash in zip([low_vals, mid_vals, high_vals], ["Low", "Mid (gốc)", "High"], ["#004d40", "#00e676", "#69f0ae"], ["dot", None, "dash"]):
        fig.add_trace(go.Scatter(x=labels, y=vals, mode="lines+markers", name=name, line=dict(width=2.5 if name == "Mid (gốc)" else 2, color=color, dash=dash), marker=dict(size=9 if name == "Mid (gốc)" else 8)))
    fig.update_layout(
        title=f"<b>Fuzzy AHP — Low / Mid / High (±{fuzzy_pct:.0f}%)</b>",
        paper_bgcolor="#001a12", plot_bgcolor="#001a12", legend=dict(bgcolor="rgba(0,0,0,0.35)", bordercolor="#00e676", borderwidth=1),
        margin=dict(l=40, r=40, t=80, b=80), font=dict(size=13, color="#e6fff7")
    )
    fig.update_xaxes(showgrid=False, tickangle=-20)
    fig.update_yaxes(title="Trọng số", range=[0, max(0.4, max(high_vals) * 1.15)], showgrid=True, gridcolor="#004d40")
    return fig

# =============================================================================
# VISUALIZATION
# =============================================================================
class ChartFactory:
    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            template="plotly_dark", title=dict(text=f"<b>{title}</b>", font=dict(size=22, color="#e6fff7"), x=0.5),
            font=dict(size=15, color="#e6fff7"), plot_bgcolor="#001a12", paper_bgcolor="#001a12",
            margin=dict(l=70, r=40, t=80, b=70), legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#00e676", borderwidth=1)
        )
        fig.update_xaxes(showgrid=True, gridcolor="#004d40", tickfont=dict(size=14, color="#e6fff7"))
        fig.update_yaxes(showgrid=True, gridcolor="#004d40", tickfont=dict(size=14, color="#e6fff7"))
        return fig

    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        colors = ['#00e676', '#69f0ae', '#b9f6ca', '#00bfa5', '#1de9b6', '#64ffda']
        fig = go.Figure(data=[go.Pie(labels=weights.index, values=weights.values, text=[c.split(':')[0] for c in weights.index], textinfo='text+percent', textposition='inside', hole=0.2, marker=dict(colors=colors, line=dict(color='#00130d', width=2)), pull=[0.05]*len(weights))])
        fig.update_layout(title=dict(text=f"<b>{title}</b>", font=dict(size=20, color="#a5ffdc"), x=0.5), showlegend=True, legend=dict(title="<b>Các tiêu chí</b>", font=dict(size=13, color="#e6fff7")), paper_bgcolor="#001a12", plot_bgcolor="#001a12", margin=dict(l=0, r=0, t=60, b=0), height=450)
        return fig

    @staticmethod
    def create_topsis_bar(results: pd.DataFrame) -> go.Figure:
        df = results.sort_values("score", ascending=True)
        fig = go.Figure(data=[go.Bar(x=df["score"], y=df["company"], orientation="h", text=[f"{v:.3f}" for v in df["score"]], textposition="outside", marker=dict(color=df["score"], colorscale=[[0, '#69f0ae'], [0.5, '#00e676'], [1, '#00c853']]))])
        fig.update_xaxes(title="<b>Điểm TOPSIS</b>", range=[0, 1])
        fig.update_yaxes(title="<b>Công ty</b>")
        return ChartFactory._apply_theme(fig, "TOPSIS Score (cao hơn = tốt hơn)")

    @staticmethod
    def create_forecast_chart(historical: np.ndarray, forecast: np.ndarray, route: str, selected_month: int) -> go.Figure:
        hist_len = len(historical)
        months_hist = list(range(1, hist_len + 1))
        next_month = selected_month % 12 + 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months_hist, y=historical, mode="lines+markers", name="Lịch sử", line=dict(color="#00e676", width=3.5), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[next_month], y=forecast, mode="lines+markers", name="Dự báo", line=dict(color="#ffeb3b", width=3.5, dash="dash"), marker=dict(size=12, symbol="diamond")))
        fig = ChartFactory._apply_theme(fig, f"Dự báo rủi ro khí hậu — {route}")
        fig.update_xaxes(title="<b>Tháng</b>", tickmode="linear", tick0=1, dtick=1, range=[1, 12], tickvals=list(range(1, 13)))
        max_val = max(float(historical.max()), float(forecast.max()))
        fig.update_yaxes(title="<b>Mức rủi ro (0–1)</b>", range=[0, max(1.0, max_val * 1.15)], tickformat=".0%")
        return fig

# =============================================================================
# EXPORT UTILITIES
# =============================================================================
class ReportGenerator:
    @staticmethod
    def generate_pdf(results: pd.DataFrame, params: AnalysisParams, var: Optional[float], cvar: Optional[float]) -> bytes:
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
            pdf.cell(15, 6, "Rank", 1); pdf.cell(55, 6, "Company", 1); pdf.cell(25, 6, "Score", 1); pdf.cell(25, 6, "ICC", 1); pdf.cell(30, 6, "Conf.", 1, 1)
            pdf.set_font("Arial", "", 10)
            for _, row in results.head(5).iterrows():
                pdf.cell(15, 6, str(int(row["rank"])), 1)
                pdf.cell(55, 6, str(row["company"])[:25], 1)
                pdf.cell(25, 6, f"{row['score']:.3f}", 1)
                pdf.cell(25, 6, str(row["recommend_icc"]), 1)
                pdf.cell(30, 6, f"{row['confidence']:.2f}", 1, 1)
            if var and cvar:
                pdf.ln(4); pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, f"VaR 95%: ${var:,.0f} | CVaR 95%: ${cvar:,.0f}", 0, 1)
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
            pd.DataFrame({"weight": weights.values}, index=weights.index).to_excel(writer, sheet_name="Weights")
        buffer.seek(0)
        return buffer.getvalue()

# =============================================================================
# APPLICATION CONTROLLER & UI
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
        base_risk = float(historical.loc[historical["month"] == params.month, params.route].iloc[0]) if params.month in historical["month"].values else 0.4
        mc_mean = mc_std = np.zeros(len(company_data))
        if params.use_mc:
            companies, mc_mean, mc_std = self.mc_simulator.simulate(base_risk, SENSITIVITY_MAP, params.mc_runs)
            order = [companies.index(c) for c in company_data.index]
            mc_mean, mc_std = mc_mean[order], mc_std[order]
        data_adjusted = company_data.copy()
        data_adjusted["C6: Rủi ro khí hậu"] = mc_mean
        if params.cargo_value > 50_000:
            data_adjusted["C1: Tỷ lệ phí"] *= 1.1
        scores = self.topsis.analyze(data_adjusted, weights, COST_BENEFIT_MAP)
        results = pd.DataFrame({"company": data_adjusted.index, "score": scores, "C6_mean": mc_mean, "C6_std": mc_std}).sort_values("score", ascending=False).reset_index(drop=True)
        results["rank"] = results.index + 1
        results["recommend_icc"] = results["score"].apply(lambda s: "ICC A" if s >= 0.75 else ("ICC B" if s >= 0.5 else "ICC C"))
        conf = self.risk_calc.calculate_confidence(results, data_adjusted)
        results["confidence"] = [conf[data_adjusted.index.get_loc(c)] for c in results["company"]]; results["confidence"] = results["confidence"].round(3)
        var = cvar = None
        if params.use_var:
            var, cvar = self.risk_calc.calculate_var_cvar(results["C6_mean"].values, params.cargo_value)
        hist_series, forecast = self.forecaster.forecast(historical, params.route, params.month, params.use_arima)
        return AnalysisResult(results=results, weights=weights, data_adjusted=data_adjusted, var=var, cvar=cvar, historical=hist_series, forecast=forecast)

class EnterpriseUI:
    def __init__(self):
        self.controller = AnalysisController()
        self.chart_factory = ChartFactory()
        self.report_gen = ReportGenerator()

    def initialize(self):
        st.set_page_config(page_title="RISKCAST v5.2 Enterprise", page_icon="Shield", layout="wide")
        apply_enterprise_css()
        if "weights" not in st.session_state:
            st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
        if "locked" not in st.session_state:
            st.session_state["locked"] = [False] * len(CRITERIA)

    def render_header(self):
        st.markdown(f"""
        <div class="enterprise-header">
            <div class="header-left">
                <img src="https://via.placeholder.com/68/00ff88/001a0f?text=R" class="header-logo">
                <div>
                    <div class="header-title">RISKCAST v5.2</div>
                    <div class="header-subtitle">Enterprise ESG Risk Assessment • Fuzzy AHP • Real Data Engine</div>
                </div>
            </div>
            <div class="header-pill">
                <span>Fuzzy AHP</span><span>·</span>
                <span>Monte Carlo</span><span>·</span>
                <span>VaR/CVaR</span><span>·</span>
                <span>Forecast</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self) -> AnalysisParams:
        with st.sidebar:
            st.markdown("<div class='sidebar-title'>LÔ HÀNG</div>", unsafe_allow_html=True)
            cargo_value = st.number_input("Giá trị (USD)", 1000, value=39_000, step=1_000)
            good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Nguy hiểm", "Khác"])
            route = st.selectbox("Tuyến vận chuyển", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"])
            method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
            month = st.selectbox("Tháng", list(range(1, 13)), index=8)
            priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])
            st.markdown("---")
            st.markdown("<div class='sidebar-title'>CẤU HÌNH MÔ HÌNH</div>", unsafe_allow_html=True)
            use_fuzzy = st.checkbox("Bật Fuzzy AHP", True)
            use_arima = st.checkbox("Dùng ARIMA", True)
            use_mc = st.checkbox("Monte Carlo", True)
            use_var = st.checkbox("VaR/CVaR", True)
            mc_runs = st.number_input("Số lần chạy MC", 500, 10_000, 2_000, 500)
            fuzzy_uncertainty = st.slider("Mức bất định (%)", 0, 50, 15) if use_fuzzy else 15
            return AnalysisParams(cargo_value, good_type, route, method, month, priority, use_fuzzy, use_arima, use_mc, use_var, mc_runs, fuzzy_uncertainty)

    def render_weight_controls(self):
        with st.container():
            st.markdown("<div class='enterprise-card'>", unsafe_allow_html=True)
            st.subheader("Phân bổ trọng số tiêu chí")
            st.markdown("""
            <div style="background:rgba(0,40,28,0.95); border-left:4px solid #00e676; padding:1.3rem; border-radius:10px; margin:1rem 0;">
                <h4 style="color:#a5ffdc; margin:0;">Giải thích nhanh:</h4>
                <ul style="color:#e0f2f1; margin:0.5rem 0;">
                    <li><b>C1 - Tỷ lệ phí:</b> Chi phí bảo hiểm (càng thấp càng tốt)</li>
                    <li><b>C2 - Thời gian xử lý:</b> Thời gian giải quyết (càng nhanh càng tốt)</li>
                    <li><b>C3 - Tỷ lệ tổn thất:</b> Tần suất rủi ro (càng thấp càng tốt)</li>
                    <li><b>C4 - Hỗ trợ ICC:</b> Mức độ hỗ trợ (càng cao càng tốt)</li>
                    <li><b>C5 - Chăm sóc KH:</b> Dịch vụ khách hàng (càng cao càng tốt)</li>
                    <li><b>C6 - Rủi ro khí hậu:</b> Rủi ro thời tiết (càng thấp càng tốt)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            cols = st.columns(len(CRITERIA))
            new_weights = st.session_state["weights"].copy()
            for i, criterion in enumerate(CRITERIA):
                with cols[i]:
                    short = criterion.split(":")[0]
                    desc = criterion.split(":")[1].strip()
                    st.markdown(f"<div style='background:#00281c; border-radius:10px; padding:8px; border:1.5px solid #00e676; text-align:center;'><span style='font-weight:800; color:#a5ffdc;'>{short}</span><br><span style='font-size:0.85rem; color:#e0f2f1;'>{desc}</span></div>", unsafe_allow_html=True)
                    st.session_state["locked"][i] = st.checkbox("Khóa", value=st.session_state["locked"][i], key=f"lock_{i}")
                    weight_val = st.number_input("Tỉ lệ", 0.0, 1.0, float(new_weights[i]), 0.01, key=f"weight_{i}", label_visibility="collapsed")
                    new_weights[i] = weight_val
                    st.markdown(f"<div style='margin-top:6px; background:#003325; border-radius:10px; border:2.5px solid #00e676; text-align:center; padding:5px;'><span style='color:#b9f6ca; font-weight:900; font-size:1.15rem;'>{weight_val:.0%}</span></div>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("RESET MẶC ĐỊNH", use_container_width=True):
                    st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
                    st.session_state["locked"] = [False] * len(CRITERIA)
                    st.rerun()
            with col2:
                total = float(new_weights.sum())
                if abs(total - 1.0) > 0.001:
                    st.warning(f"Tổng trọng số: {total:.1%} → sẽ tự cân bằng")
                else:
                    st.success(f"Tổng trọng số: {total:.1%}")
            st.session_state["weights"] = WeightManager.auto_balance(new_weights, st.session_state["locked"])
            st.markdown("</div>", unsafe_allow_html=True)

    def display_results(self, result: AnalysisResult, params: AnalysisParams):
        st.success("Đã phân tích xong!")
        left, right = st.columns([2.1, 1.1])
        with left:
            st.subheader("Bảng xếp hạng")
            df_show = result.results[["rank", "company", "score", "confidence", "recommend_icc"]].set_index("rank")
            df_show.columns = ["Công ty", "Điểm số", "Độ tin cậy", "ICC"]
            st.dataframe(df_show, use_container_width=True)
            top = result.results.iloc[0]
            st.markdown(f"<div class='premium-card'>GỢI Ý TỐI ƯU<br><span style='font-size:1.5rem;'>{top['company']}</span><br>Score: <b>{top['score']:.3f}</b> | Conf: <b>{top['confidence']:.2f}</b> | Gói: <b>{top['recommend_icc']}</b></div>", unsafe_allow_html=True)
        with right:
            if result.var and result.cvar:
                st.metric("VaR 95%", f"${result.var:,.0f}")
                st.metric("CVaR 95%", f"${result.cvar:,.0f}")
            fig_pie = self.chart_factory.create_weights_pie(result.weights, "Trọng số (sau Fuzzy)")
            st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("---")
        st.subheader("Biểu đồ chính")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(self.chart_factory.create_topsis_bar(result.results), use_container_width=True)
        with col2:
            st.plotly_chart(self.chart_factory.create_forecast_chart(result.historical, result.forecast, params.route, params.month), use_container_width=True)
        if params.use_fuzzy:
            st.markdown("---")
            st.subheader("Fuzzy AHP")
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fuzzy_chart_premium(result.weights, params.fuzzy_uncertainty), use_container_width=True)
            with col2:
                st.dataframe(build_fuzzy_table(result.weights, params.fuzzy_uncertainty), use_container_width=True)
            most_unc, diff_map = most_uncertain_criterion(result.weights, params.fuzzy_uncertainty)
            st.markdown(f"<div style='background:#00331F; padding:18px; border-radius:12px; border:2.5px solid #00FFAA; color:#CCFFE6; font-size:17px; margin-top:1rem;'>Tiêu chí dao động mạnh nhất: <span style='color:#00FFAA; font-size:22px;'><b>{most_unc}</b></span><br>Nên cân nhắc kỹ khi điều chỉnh trọng số.</div>", unsafe_allow_html=True)
            st.plotly_chart(fuzzy_heatmap_premium(diff_map), use_container_width=True)
        st.markdown("---")
        st.subheader("Xuất báo cáo")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Tải Excel", data=self.report_gen.generate_excel(result.results, result.data_adjusted, result.weights), file_name=f"riskcast_{params.route.replace(' - ', '_')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with col2:
            pdf_data = self.report_gen.generate_pdf(result.results, params, result.var, result.cvar)
            if pdf_data:
                st.download_button("Tải PDF", data=pdf_data, file_name=f"riskcast_{params.route.replace(' - ', '_')}.pdf", mime="application/pdf", use_container_width=True)

    def run(self):
        self.initialize()
        self.render_header()
        historical = DataService.load_historical_data()
        params = self.render_sidebar()
        self.render_weight_controls()
        weights_series = pd.Series(st.session_state["weights"], index=CRITERIA)
        st.plotly_chart(self.chart_factory.create_weights_pie(weights_series, "Trọng số hiện tại"), use_container_width=True)
        if st.button("PHÂN TÍCH & GỢI Ý", type="primary", use_container_width=True):
            with st.spinner("Đang chạy mô hình..."):
                result = self.controller.run_analysis(params, historical)
                self.display_results(result, params)

# =============================================================================
# MAIN
# =============================================================================
def main():
    app = EnterpriseUI()
    app.run()

if __name__ == "__main__":
    main()
