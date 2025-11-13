# =============================================================================
# RISKCAST v5.1 — ESG Logistics Risk Assessment Dashboard (UI PRO)
# Refactored & UI Optimized by Grok (xAI)
# Original Author: Bùi Xuân Hoàng
# =============================================================================
import io
import time
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
    "Chubb": 0.95, "PVI": 1.10, "InternationalIns": 1.20,
    "BaoViet": 1.05, "Aon": 0.90
}

# =============================================================================
# UI STYLING (PRO)
# =============================================================================
def apply_pro_css():
    st.markdown("""
    <style>
        /* Global */
        * { text-rendering: optimizeLegibility !important; -webkit-font-smoothing: antialiased !important; }
        .stApp { background: #F8FAFC !important; font-family: 'Inter', 'Segoe UI', sans-serif !important; }
        .block-container { padding: 1.5rem 2rem !important; max-width: 1600px; margin: auto; }

        /* Header */
        .header-box {
            background: linear-gradient(90deg, #0052A3, #003D7A);
            padding: 1.8rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            margin-bottom: 2rem;
        }
        .header-box h1 { color: white; margin: 0; font-weight: 900; font-size: 2.8rem; text-shadow: 2px 2px 10px rgba(0,0,0,0.5); }
        .header-box p { color: #E3F2FD; margin: 8px 0 0; font-weight: 600; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #FFFFFF !important;
            border-right: 3px solid #0052A3;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        section[data-testid="stSidebar"] h2 { color: #0052A3; background: #FFF9E6; padding: 12px; border-radius: 8px; margin-bottom: 15px; text-align: center; font-weight: 900; }

        /* Buttons */
        .stButton > button {
            background: #0052A3 !important; color: white !important; border-radius: 10px !important;
            padding: 0.9rem 2.5rem !important; font-weight: 800 !important; font-size: 1.1rem !important;
            border: 2px solid #0052A3 !important; transition: all 0.3s !important; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        .stButton > button:hover {
            background: #003D7A !important; transform: translateY(-3px) !important; box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
        }

        /* Result Box */
        .result-box {
            background: linear-gradient(135deg, #FFB800, #FF9800);
            color: #000 !important; padding: 2rem; border-radius: 16px; text-align: center;
            font-weight: 800; font-size: 1.4rem; margin: 1.5rem 0; box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            border: 3px solid #FF9800;
        }

        /* Quick Insight */
        .insight-box {
            background: #E8F5E8; padding: 1.2rem; border-radius: 12px; border-left: 6px solid #4CAF50;
            text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            font-weight: 700; font-size: 1.1rem; color: #0052A3; border-bottom: 3px solid #0052A3;
        }
        .stTabs [data-baseweb="tab"]:hover { color: #003D7A; }

        /* Metrics */
        [data-testid="stMetricValue"] { color: #0052A3 !important; font-weight: 900 !important; font-size: 2.4rem !important; }
        [data-testid="stMetricLabel"] { color: #000 !important; font-weight: 800 !important; }

        /* Footer */
        .footer {
            text-align: center; padding: 2rem; color: #666; font-size: 0.9rem; margin-top: 3rem;
            border-top: 1px solid #E0E0E0;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .block-container { padding: 1rem !important; }
            h1 { font-size: 2.2rem !important; }
            .stButton > button { font-size: 1rem !important; padding: 0.7rem !important; }
        }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DATA LAYER
# =============================================================================
class DataService:
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_all_data():
        climate_base = {
            "VN - EU": [0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.42, 0.48, 0.60, 0.68, 0.58, 0.45],
            "VN - US": [0.30, 0.33, 0.36, 0.40, 0.45, 0.50, 0.56, 0.62, 0.75, 0.72, 0.60, 0.52],
            "VN - Singapore": [0.15, 0.16, 0.18, 0.20, 0.22, 0.26, 0.30, 0.32, 0.35, 0.34, 0.28, 0.25],
            "VN - China": [0.18, 0.19, 0.21, 0.24, 0.26, 0.30, 0.34, 0.36, 0.40, 0.38, 0.32, 0.28],
            "Domestic": [0.10] * 12
        }
        historical = pd.DataFrame({"month": range(1, 13)})
        for route, values in climate_base.items():
            historical[route] = values

        company_data = pd.DataFrame({
            "Company": ["Chubb", "PVI", "InternationalIns", "BaoViet", "Aon"],
            "C1: Tỷ lệ phí": [0.30, 0.28, 0.26, 0.32, 0.24],
            "C2: Thời gian xử lý": [6, 5, 8, 7, 4],
            "C3: Tỷ lệ tổn thất": [0.08, 0.06, 0.09, 0.10, 0.07],
            "C4: Hỗ trợ ICC": [9, 8, 6, 9, 7],
            "C5: Chăm sóc KH": [9, 8, 5, 7, 6],
        }).set_index("Company")

        return historical, company_data

# =============================================================================
# CORE ALGORITHMS (Giữ nguyên - đã tối ưu)
# =============================================================================
class WeightManager:
    @staticmethod
    def auto_balance(weights: np.ndarray, locked: List[bool]) -> np.ndarray:
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
    def simulate(base_risk: float, sensitivity_map: Dict[str, float], n_simulations: int):
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())
        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.03, mu * 0.12)
        simulations = rng.normal(loc=mu, scale=sigma, size=(n_simulations, len(companies)))
        simulations = np.clip(simulations, 0.0, 1.0)
        return companies, simulations.mean(axis=0), simulations.std(axis=0)

class TOPSISAnalyzer:
    @staticmethod
    def analyze(data: pd.DataFrame, weights: pd.Series, cost_benefit: Dict[str, CriterionType]) -> np.ndarray:
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
    def calculate_var_cvar(loss_rates: np.ndarray, cargo_value: float, confidence: float = 0.95):
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
        cv_c6 = results["C6_std"].values / (results["C6_mean"].values + eps)
        conf_c6 = 1.0 / (1.0 + cv_c6)
        conf_c6 = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + eps)
        crit_cv = data.std(axis=1).values / (data.mean(axis=1).values + eps)
        conf_crit = 1.0 / (1.0 + crit_cv)
        conf_crit = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (np.ptp(conf_crit) + eps)
        return np.sqrt(conf_c6 * conf_crit)

class Forecaster:
    @staticmethod
    def forecast(historical: pd.DataFrame, route: str, months_ahead: int = 3, use_arima: bool = True):
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
        trend = (series[-1] - series[-3]) / 3.0 if len(series) >= 3 else 0.0
        forecast = np.array([np.clip(series[-1] + (i + 1) * trend, 0, 1) for i in range(months_ahead)])
        return series, forecast

# =============================================================================
# VISUALIZATION (PRO)
# =============================================================================
class ChartFactory:
    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            template="plotly_white",
            title=dict(text=f"<b>{title}</b>", font=dict(size=22, color="#000", family="Arial Black"), x=0.5),
            font=dict(size=16, color="#000", family="Arial", weight=700),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(font=dict(size=16, color="#000", weight="bold"), bgcolor="white", bordercolor="#000", borderwidth=2)
        )
        fig.update_xaxes(showgrid=True, gridcolor="#E0E0E0", linecolor="#000", linewidth=2)
        fig.update_yaxes(showgrid=True, gridcolor="#E0E0E0", linecolor="#000", linewidth=2)
        return fig

    @staticmethod
    def create_topsis_bar(results: pd.DataFrame) -> go.Figure:
        sorted_res = results.sort_values("score")
        fig = go.Figure(data=[go.Bar(
            x=sorted_res["score"], y=sorted_res["company"], orientation="h",
            text=sorted_res["score"].apply(lambda x: f"<b>{x:.3f}</b>"),
            textposition="outside", textfont=dict(size=18, color="#000", weight="bold"),
            marker=dict(color=sorted_res["score"], colorscale=[[0, '#90CAF9'], [1, '#003D7A']], line=dict(color='#000', width=2)),
            hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
        )])
        fig.update_xaxes(title="<b>TOPSIS Score</b>", range=[0, 1])
        fig.update_yaxes(title="<b>Công ty</b>")
        return ChartFactory._apply_theme(fig, "TOPSIS Score (cao hơn = tốt hơn)")

    @staticmethod
    def create_forecast_chart(historical: np.ndarray, forecast: np.ndarray, route: str) -> go.Figure:
        fig = go.Figure()
        months_hist = list(range(1, len(historical) + 1))
        months_fc = list(range(len(historical) + 1, len(historical) + len(forecast) + 1))
        fig.add_trace(go.Scatter(x=months_hist, y=historical, mode="lines+markers", name="Lịch sử",
                                 line=dict(color="#0052A3", width=4), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=months_fc, y=forecast, mode="lines+markers", name="Dự báo",
                                 line=dict(color="#FF5252", width=4, dash="dash"), marker=dict(size=12, symbol="diamond")))
        fig = ChartFactory._apply_theme(fig, f"Dự báo rủi ro khí hậu: {route}")
        fig.update_xaxes(title="<b>Tháng</b>", tickmode="linear", tickvals=list(range(1, 13)))
        fig.update_yaxes(title="<b>Mức rủi ro</b>", tickformat='.0%')
        return fig

# =============================================================================
# REPORT & EXPORT
# =============================================================================
class ReportGenerator:
    @staticmethod
    def generate_pdf(results: pd.DataFrame, params: AnalysisParams, var: Optional[float], cvar: Optional[float]) -> bytes:
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST v5.1 - Executive Summary", ln=1, align='C')
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"Route: {params.route} | Month: {params.month} | Value: ${params.cargo_value:,.0f}", ln=1)
            top = results.iloc[0]
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"Top: {top['company']} | Score: {top['score']:.3f}", ln=1)
            return pdf.output(dest='S').encode('latin1')
        except:
            return b""

    @staticmethod
    def generate_excel(results: pd.DataFrame, data: pd.DataFrame, weights: pd.Series) -> bytes:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Results", index=False)
            data.to_excel(writer, sheet_name="Data")
            pd.DataFrame({"weight": weights}).to_excel(writer, sheet_name="Weights")
        buffer.seek(0)
        return buffer.getvalue()

# =============================================================================
# MAIN APP (UI PRO)
# =============================================================================
def main():
    st.set_page_config(page_title="RISKCAST v5.1", page_icon="anchor", layout="wide")
    apply_pro_css()

    # Load data
    historical, company_data = DataService.load_all_data()

    # Layout
    left_sidebar, main_col, right_panel = st.columns([0.35, 1, 0.4])

    with main_col:
        st.markdown("""
        <div class="header-box">
            <h1>anchor RISKCAST v5.1</h1>
            <p>ESG Logistics Risk Assessment • AI-Powered Decision Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

    # === SIDEBAR ===
    with left_sidebar:
        st.image("https://img.icons8.com/fluency/48/ship.png", width=60)
        st.markdown("<h2>INPUT</h2>", unsafe_allow_html=True)

        cargo_value = st.number_input("Giá trị (USD)", 1000, value=39000, step=1000)
        good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Nguy hiểm", "Khác"])
        route = st.selectbox("Tuyến", historical.columns[1:].tolist())
        method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
        month = st.selectbox("Tháng", list(range(1, 13)), index=8)
        priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

        st.markdown("---")
        st.markdown("### Cấu hình")
        use_fuzzy = st.checkbox("Fuzzy AHP", True)
        use_arima = st.checkbox("ARIMA", True)
        use_mc = st.checkbox("Monte Carlo", True)
        use_var = st.checkbox("VaR/CVaR", True)
        mc_runs = st.slider("MC Runs", 500, 5000, 2000, 500)
        fuzzy_uncertainty = st.slider("Fuzzy (%)", 0, 50, 15) if use_fuzzy else 15

        # Progress
        progress = sum([cargo_value>0, route, month, sum(st.session_state.get("weights", [0])*6)>0.9]) / 4
        st.progress(progress)
        st.caption(f"Hoàn thành: {int(progress*100)}%")

    # === WEIGHT CONTROLS ===
    if "weights" not in st.session_state:
        st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
    if "locked" not in st.session_state:
        st.session_state["locked"] = [False] * len(CRITERIA)

    with main_col:
        st.subheader("Phân bổ trọng số")
        cols = st.columns(len(CRITERIA))
        new_weights = st.session_state["weights"].copy()
        for i, crit in enumerate(CRITERIA):
            with cols[i]:
                short = crit.split(':')[0]
                st.markdown(f"<div style='text-align:center; background:#FFF9E6; padding:8px; border-radius:6px; border:2px solid #FFB800;'><b style='color:#0052A3;'>{short}</b></div>", unsafe_allow_html=True)
                st.session_state["locked"][i] = st.checkbox("Khóa", key=f"lock_{i}")
                val = st.number_input("Tỉ lệ", 0.0, 1.0, float(new_weights[i]), 0.01, key=f"w_{i}", label_visibility="collapsed")
                new_weights[i] = val
                st.markdown(f"<div style='text-align:center; background:#E3F2FD; padding:8px; border-radius:6px; border:3px solid #0052A3;'><b style='color:#0052A3;'>{val:.1%}</b></div>", unsafe_allow_html=True)

        if st.button("Reset", use_container_width=True):
            st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
            st.session_state["locked"] = [False] * len(CRITERIA)
            st.rerun()

        st.session_state["weights"] = WeightManager.auto_balance(new_weights, st.session_state["locked"])

    # === ANALYSIS BUTTON ===
    if st.button("rocket PHÂN TÍCH & GỢI Ý", type="primary", use_container_width=True):
        with st.spinner("Đang chạy AI phân tích..."):
            time.sleep(1.5)

            # Run analysis (giữ nguyên logic cũ)
            weights = pd.Series(st.session_state["weights"], index=CRITERIA)
            if use_fuzzy:
                weights = FuzzyAHP.apply(weights, fuzzy_uncertainty)

            base_risk = float(historical.loc[historical["month"] == month, route].iloc[0])
            mc_mean = mc_std = np.zeros(len(company_data))
            if use_mc:
                _, mc_mean, mc_std = MonteCarloSimulator.simulate(base_risk, SENSITIVITY_MAP, mc_runs)
                order = [list(SENSITIVITY_MAP.keys()).index(c) for c in company_data.index]
                mc_mean, mc_std = mc_mean[order], mc_std[order]

            data_adj = company_data.copy()
            data_adj["C6: Rủi ro khí hậu"] = mc_mean
            if cargo_value > 50000:
                data_adj["C1: Tỷ lệ phí"] *= 1.1

            scores = TOPSISAnalyzer.analyze(data_adj, weights, COST_BENEFIT_MAP)
            results = pd.DataFrame({
                "company": data_adj.index, "score": scores,
                "C6_mean": mc_mean, "C6_std": mc_std
            }).sort_values("score", ascending=False).reset_index(drop=True)
            results["rank"] = results.index + 1
            results["recommend_icc"] = results["score"].apply(lambda s: "ICC A" if s >= 0.75 else ("ICC B" if s >= 0.5 else "ICC C"))

            conf = RiskCalculator.calculate_confidence(results, data_adj)
            results["confidence"] = pd.Series(conf, index=results.index).round(3)

            var, cvar = (None, None)
            if use_var:
                var, cvar = RiskCalculator.calculate_var_cvar(results["C6_mean"].values, cargo_value)

            hist_series, forecast = Forecaster.forecast(historical, route, use_arima=use_arima)

            result = AnalysisResult(results, weights, data_adj, var, cvar, hist_series, forecast)
            st.session_state["result"] = result
            st.session_state["params"] = AnalysisParams(cargo_value, good_type, route, method, month, priority,
                                                       use_fuzzy, use_arima, use_mc, use_var, mc_runs, fuzzy_uncertainty)

        st.success("Hoàn tất!")
        st.balloons()

    # === DISPLAY RESULTS ===
    if "result" in st.session_state:
        result = st.session_state["result"]
        params = st.session_state["params"]
        top = result.results.iloc[0]

        with right_panel:
            st.markdown("### lightning Quick Insight")
            st.markdown(f"""
            <div class="insight-box">
                <h3 style="margin:0; color:#1B5E20;">{top['company']}</h3>
                <p style="margin:5px 0;"><b>Score:</b> {top['score']:.3f}</p>
                <p style="margin:5px 0;"><b>{top['recommend_icc']}</b></p>
            </div>
            """, unsafe_allow_html=True)
            if result.var:
                st.metric("VaR 95%", f"${result.var:,.0f}")
                st.metric("CVaR 95%", f"${result.cvar:,.0f}")

        with main_col:
            tab1, tab2, tab3, tab4 = st.tabs(["Kết quả", "Biểu đồ", "Báo cáo", "Giải thích AI"])

            with tab1:
                df_display = result.results[["rank", "company", "score", "confidence", "recommend_icc"]].set_index("rank")
                df_display.columns = ["Công ty", "Score", "Độ tin cậy", "ICC"]
                st.dataframe(df_display, use_container_width=True)

                st.markdown(f"""
                <div class="result-box">
                    KHUYẾN NGHỊ HÀNG ĐẦU<br>
                    <span style="font-size:1.8rem;">{top['company']}</span><br>
                    Score: <b>{top['score']:.3f}</b> | ICC: <b>{top['recommend_icc']}</b>
                </div>
                """, unsafe_allow_html=True)

            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    medals = ["gold", "silver", "award"]
                    for i in range(min(3, len(result.results))):
                        st.markdown(f"<h3 style='text-align:center;'>{medals[i]} #{i+1} {result.results.iloc[i]['company']}</h3>", unsafe_allow_html=True)
                    st.plotly_chart(ChartFactory.create_topsis_bar(result.results), use_container_width=True)
                with col2:
                    st.plotly_chart(ChartFactory.create_forecast_chart(result.historical, result.forecast, params.route), use_container_width=True)

            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    excel = ReportGenerator.generate_excel(result.results, result.data_adjusted, result.weights)
                    st.download_button("Tải Excel", excel, f"riskcast_{params.route}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with col2:
                    pdf = ReportGenerator.generate_pdf(result.results, params, result.var, result.cvar)
                    if pdf:
                        st.download_button("Tải PDF", pdf, f"riskcast_{params.route}.pdf", "application/pdf")

            with tab4:
                st.info(f"""
                **AI phân tích:**  
                Tuyến **{params.route}** tháng **{params.month}** có rủi ro khí hậu **{result.results.iloc[0]['C6_mean']:.1%}**.  
                **{top['company']}** được chọn vì:  
                - Phí cạnh tranh: **{result.data_adjusted.loc[top['company'], 'C1: Tỷ lệ phí']:.1%}**  
                - Xử lý nhanh: **{result.data_adjusted.loc[top['company'], 'C2: Thời gian xử lý']:.0f} ngày**  
                - ICC: **{result.data_adjusted.loc[top['company'], 'C4: Hỗ trợ ICC']:.0f}/10**  
                **Khuyến nghị:** Mua **{top['recommend_icc']}**
                """)

    # === FOOTER ===
    st.markdown("""
    <div class="footer">
        <b>RISKCAST v5.1</b> • Powered by <b>Streamlit + TOPSIS + ARIMA + Monte Carlo</b><br>
        © 2025 Bùi Xuân Hoàng • ESG Logistics Intelligence
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
