# =============================================================================
# RISKCAST v5.3 — ENTERPRISE EDITION (Multi-Package Analysis)
# Full file — Part 1/6
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
    COST = "cost"       # Càng thấp càng tốt
    BENEFIT = "benefit" # Càng cao càng tốt


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


# 6 tiêu chí
CRITERIA = [
    "C1: Tỷ lệ phí",
    "C2: Thời gian xử lý",
    "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC",
    "C5: Chăm sóc KH",
    "C6: Rủi ro khí hậu"
]

# Trọng số theo 3 mục tiêu
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

# ICC Packages
ICC_PACKAGES = {
    "ICC A": {
        "coverage": 1.0,
        "premium_multiplier": 1.5,
        "description": "Bảo vệ toàn diện mọi rủi ro (All Risks)"
    },
    "ICC B": {
        "coverage": 0.75,
        "premium_multiplier": 1.0,
        "description": "Bảo vệ các rủi ro chính (Named Perils)"
    },
    "ICC C": {
        "coverage": 0.5,
        "premium_multiplier": 0.65,
        "description": "Bảo vệ cơ bản, chi phí thấp (Major Losses Only)"
    }
}

# Cost/Benefit map
COST_BENEFIT_MAP = {
    "C1: Tỷ lệ phí": CriterionType.COST,
    "C2: Thời gian xử lý": CriterionType.COST,
    "C3: Tỷ lệ tổn thất": CriterionType.COST,
    "C4: Hỗ trợ ICC": CriterionType.BENEFIT,
    "C5: Chăm sóc KH": CriterionType.BENEFIT,
    "C6: Rủi ro khí hậu": CriterionType.COST
}

# Công ty nhạy cảm khí hậu
SENSITIVITY_MAP = {
    "Chubb": 0.95,
    "PVI": 1.05,
    "BaoViet": 1.00,
    "BaoMinh": 1.02,
    "MIC": 1.03
}


# =============================================================================
# CUSTOM CSS (ENTERPRISE PREMIUM UI)
# =============================================================================

def apply_custom_css():
    st.markdown("""
    <style>
    * {
        text-rendering: optimizeLegibility !important;
        -webkit-font-smoothing: antialiased !important;
    }

    .stApp {
        background: radial-gradient(circle at top, #00ff99 0%, #001a0f 32%, #000c08 100%);
        font-family: 'Inter', sans-serif;
        color: #e6fff7;
    }

    .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }

    h1 { font-size: 2.7rem !important; font-weight: 900; }
    h2 { font-size: 2.0rem !important; font-weight: 800; }
    h3 { font-size: 1.4rem !important; font-weight: 700; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: radial-gradient(circle at top left, #003322 0%, #000f0a 40%, #000805 100%);
        border-right: 1px solid rgba(0,255,170,0.4);
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #a5ffdc !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(120deg, #00ff99, #00e676, #00bfa5);
        color: #00130d !important;
        font-weight: 800;
        border-radius: 999px;
        padding: 0.55rem 1.7rem;
        border: none;
        box-shadow: 0 0 14px rgba(0,255,153,0.7);
        transition: 0.12s;
    }

    .stButton > button:hover {
        transform: translateY(-2px) scale(1.03);
        box-shadow: 0 0 20px rgba(0,255,153,1);
    }

    </style>
    """, unsafe_allow_html=True)
# =============================================================================
# PART 2/6 — DATA SERVICE + FUZZY + MONTE CARLO + TOPSIS + VAR/CVAR + FORECAST
# =============================================================================


# =============================================================================
# DATA LAYER
# =============================================================================

class DataService:
    """Quản lý dữ liệu đầu vào (lịch sử khí hậu + dữ liệu công ty)."""

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_historical_data() -> pd.DataFrame:
        """
        Dữ liệu rủi ro khí hậu theo tuyến (12 tháng), chuẩn hoá 0–1.
        Mô phỏng theo mức độ bão, sóng, mưa, chậm trễ năm 2023–2024.
        """
        climate_base = {
            "VN - EU":         [0.28,0.30,0.35,0.40,0.52,0.60,0.67,0.70,0.75,0.72,0.60,0.48],
            "VN - US":         [0.33,0.36,0.40,0.46,0.55,0.63,0.72,0.78,0.80,0.74,0.62,0.50],
            "VN - Singapore":  [0.18,0.20,0.24,0.27,0.32,0.36,0.40,0.43,0.45,0.42,0.35,0.30],
            "VN - China":      [0.20,0.23,0.27,0.31,0.38,0.42,0.48,0.50,0.53,0.49,0.40,0.34],
            "Domestic":        [0.12,0.13,0.14,0.16,0.20,0.22,0.23,0.25,0.27,0.24,0.20,0.18]
        }

        df = pd.DataFrame({"month": list(range(1, 13))})
        for route, vals in climate_base.items():
            df[route] = vals
        return df


    @staticmethod
    @st.cache_data
    def get_company_data() -> pd.DataFrame:
        """
        Thông số từng công ty bảo hiểm theo chuẩn benchmark ngành 2023–2024.
        """
        return (
            pd.DataFrame({
                "Company": ["Chubb","PVI","BaoViet","BaoMinh","MIC"],
                "C1: Tỷ lệ phí":     [0.42,0.36,0.40,0.38,0.34],
                "C2: Thời gian xử lý":[12,10,15,14,11],
                "C3: Tỷ lệ tổn thất": [0.07,0.09,0.11,0.10,0.08],
                "C4: Hỗ trợ ICC":     [9,8,7,8,7],
                "C5: Chăm sóc KH":    [9,8,7,7,6],
            })
            .set_index("Company")
        )


# =============================================================================
# FUZZY AHP
# =============================================================================

class FuzzyAHP:
    """
    Áp dụng tam giác mờ ±% cho trọng số tiêu chí.
    """

    @staticmethod
    def apply(weights: pd.Series, uncertainty_pct: float) -> pd.Series:
        factor = uncertainty_pct / 100
        w = weights.values

        low  = np.maximum(w * (1 - factor), 1e-9)
        high = np.minimum(w * (1 + factor), 0.9999)

        centroid = (low + w + high) / 3
        normalized = centroid / centroid.sum()

        return pd.Series(normalized, index=weights.index)


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloSimulator:
    """
    Mô phỏng rủi ro khí hậu C6 theo phân phối chuẩn, 2000+ lần.
    """

    @staticmethod
    @st.cache_data(ttl=600)
    def simulate(base_risk: float, sensitivity_map: Dict[str, float], n_sim: int):
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())

        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.03, mu * 0.12)

        sims = rng.normal(mu, sigma, size=(n_sim, len(companies)))
        sims = np.clip(sims, 0, 1)

        return companies, sims.mean(axis=0), sims.std(axis=0)


# =============================================================================
# TOPSIS
# =============================================================================

class TOPSISAnalyzer:
    """
    TOPSIS chuẩn 5 bước: normalize → weighted → ideal best/worst → khoảng cách → điểm số.
    """

    @staticmethod
    def analyze(data: pd.DataFrame, weights: pd.Series, cb_map: Dict[str, CriterionType]):
        M = data[list(weights.index)].astype(float).values

        denom = np.sqrt((M ** 2).sum(axis=0))
        denom[denom == 0] = 1
        R = M / denom       # normalize
        V = R * weights.values  # weighted

        is_cost = np.array([cb_map[c] == CriterionType.COST for c in weights.index])

        ideal_best  = np.where(is_cost, V.min(axis=0), V.max(axis=0))
        ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))

        d_plus  = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

        return d_minus / (d_plus + d_minus + 1e-12)


# =============================================================================
# VAR / CVAR
# =============================================================================

class RiskCalculator:

    @staticmethod
    def calculate_var_cvar(loss_rates: np.ndarray, cargo_value: float, conf=0.95):
        if len(loss_rates) == 0:
            return 0, 0

        losses = loss_rates * cargo_value
        var = float(np.percentile(losses, conf * 100))

        tail = losses[losses >= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var

        return var, cvar


# =============================================================================
# FORECASTER (ARIMA + fallback linear)
# =============================================================================

class Forecaster:

    @staticmethod
    def forecast(historical: pd.DataFrame, route: str, month: int, use_arima=True):
        if route not in historical.columns:
            route = historical.columns[1]

        full = historical[route].values
        month = max(1, min(month, len(full)))

        hist = full[:month]

        if use_arima and ARIMA_AVAILABLE and len(hist) >= 6:
            try:
                model = ARIMA(hist, order=(1,1,1))
                fc = model.fit().forecast(1)
                return hist, np.array([float(np.clip(fc[0], 0, 1))])
            except:
                pass

        # Fallback: linear trend
        if len(hist) >= 3:
            trend = (hist[-1] - hist[-3]) / 2
        elif len(hist) >= 2:
            trend = hist[-1] - hist[-2]
        else:
            trend = 0

        next_val = np.clip(hist[-1] + trend, 0, 1)
        return hist, np.array([next_val])
# =============================================================================
# PART 3/6 — MULTI-PACKAGE ANALYZER (FULL FIXED VERSION)
# =============================================================================


class MultiPackageAnalyzer:
    """
    Phân tích 15 phương án (5 công ty × 3 gói ICC).
    """

    def __init__(self):
        self.data_service = DataService()
        self.fuzzy_ahp = FuzzyAHP()
        self.mc_sim = MonteCarloSimulator()
        self.topsis = TOPSISAnalyzer()
        self.risk_calc = RiskCalculator()
        self.forecaster = Forecaster()

    def run_analysis(self, params: AnalysisParams, historical: pd.DataFrame) -> AnalysisResult:
        # ---------------------
        # 1) Lấy trọng số theo mục tiêu
        # ---------------------
        profile_weights = PRIORITY_PROFILES[params.priority]
        weights = pd.Series(profile_weights, index=CRITERIA)

        if params.use_fuzzy:
            weights = self.fuzzy_ahp.apply(weights, params.fuzzy_uncertainty)

        # ---------------------
        # 2) Load data công ty
        # ---------------------
        company_data = self.data_service.get_company_data()

        # ---------------------
        # 3) Base risk tháng + tuyến
        # ---------------------
        if params.month in historical["month"].values:
            base_risk = float(
                historical.loc[historical["month"] == params.month, params.route].iloc[0]
            )
        else:
            base_risk = 0.4

        # ---------------------
        # 4) Monte Carlo (C6)
        # ---------------------
        if params.use_mc:
            companies, mc_mean_raw, mc_std_raw = self.mc_sim.simulate(
                base_risk, SENSITIVITY_MAP, params.mc_runs
            )
            # Sắp xếp lại theo index company_data
            order = [companies.index(c) for c in company_data.index]
            mc_mean = mc_mean_raw[order]
            mc_std = mc_std_raw[order]
        else:
            mc_mean = np.zeros(len(company_data))
            mc_std = np.zeros(len(company_data))

        # ---------------------
        # 5) Sinh 15 phương án (company × ICC)
        # ---------------------
        all_options = []
        for comp_idx, company in enumerate(company_data.index):
            row = company_data.loc[company]

            for icc_name, icc_data in ICC_PACKAGES.items():
                option = row.copy()

                # Adjust phí theo ICC
                prem = option["C1: Tỷ lệ phí"]
                option["C1: Tỷ lệ phí"] = prem * icc_data["premium_multiplier"]

                # Adjust hỗ trợ ICC
                option["C4: Hỗ trợ ICC"] = option["C4: Hỗ trợ ICC"] * icc_data["coverage"]

                # C6 từ Monte Carlo
                option["C6: Rủi ro khí hậu"] = mc_mean[comp_idx]

                # Ghi lại vào dict
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

                    "C6_std": mc_std[comp_idx]
                })

        data_adj = pd.DataFrame(all_options)

        # ---------------------
        # 6) Phụ phí nếu hàng > 50k
        # ---------------------
        if params.cargo_value > 50_000:
            data_adj["C1: Tỷ lệ phí"] *= 1.10
            data_adj["estimated_cost"] *= 1.10

        # ---------------------
        # 7) TOPSIS
        # ---------------------
        scores = self.topsis.analyze(
            data_adj[
                ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
                 "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]
            ],
            weights,
            COST_BENEFIT_MAP
        )

        data_adj["score"] = scores
        data_adj["C6_mean"] = data_adj["C6: Rủi ro khí hậu"]

        # ---------------------
        # 8) Ranking
        # ---------------------
        data_adj = data_adj.sort_values("score", ascending=False).reset_index(drop=True)
        data_adj["rank"] = data_adj.index + 1

        # ---------------------
        # 9) Category theo ICC
        # ---------------------
        def categorize(row):
            if row["icc_package"] == "ICC C":
                return "💰 Tiết kiệm"
            elif row["icc_package"] == "ICC B":
                return "⚖️ Cân bằng"
            else:
                return "🛡️ An toàn"

        data_adj["category"] = data_adj.apply(categorize, axis=1)

        # ---------------------
        # 10) Độ tin cậy
        # ---------------------
        eps = 1e-9
        cv = data_adj["C6_std"].values / (data_adj["C6_mean"].values + eps)
        conf = 1.0 / (1.0 + cv)
        conf = 0.3 + 0.7 * (conf - conf.min()) / (np.ptp(conf) + eps)
        data_adj["confidence"] = conf

        # ---------------------
        # 11) VaR / CVaR
        # ---------------------
        var = cvar = None
        if params.use_var:
            var, cvar = self.risk_calc.calculate_var_cvar(
                data_adj["C6_mean"].values, params.cargo_value
            )

        # ---------------------
        # 12) Forecast
        # ---------------------
        hist, forecast = self.forecaster.forecast(
            historical, params.route, params.month, use_arima=params.use_arima
        )

        return AnalysisResult(
            results=data_adj,
            weights=weights,
            data_adjusted=data_adj,
            var=var,
            cvar=cvar,
            historical=hist,
            forecast=forecast
        )
# =============================================================================
# PART 4/6 — CHART FACTORY (FULL FIXED)
# =============================================================================


class ChartFactory:
    """Tạo các biểu đồ Plotly với theme Enterprise Premium Green."""

    # ------------------------- THEME ÁP DỤNG CHUNG -------------------------
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

    # ------------------------- PIE WEIGHTS -------------------------
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

    # ------------------------- COST-BENEFIT SCATTER -------------------------
    @staticmethod
    def create_cost_benefit_scatter(results: pd.DataFrame) -> go.Figure:
        """Chi phí vs Điểm TOPSIS (màu theo gói ICC)."""

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

        return ChartFactory._apply_theme(fig, "💰 Chi phí vs Chất lượng (Cost–Benefit Analysis)")

    # ------------------------- TOP 5 BAR -------------------------
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

    # ------------------------- FORECAST CHART -------------------------
    @staticmethod
    def create_forecast_chart(historical: np.ndarray, forecast: np.ndarray, route: str, selected_month: int) -> go.Figure:
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
            range=[1, 12]
        )

        max_val = max(float(historical.max()), float(forecast.max()))
        fig.update_yaxes(
            title="<b>Mức rủi ro (0–1)</b>",
            range=[0, max(1.0, max_val * 1.15)],
            tickformat=".0%"
        )

        return fig

    # ------------------------- CATEGORY COMPARISON -------------------------
    @staticmethod
    def create_category_comparison(results: pd.DataFrame) -> go.Figure:
        categories = ["💰 Tiết kiệm", "⚖️ Cân bằng", "🛡️ An toàn"]
        avg_scores, avg_costs = [], []

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
# PART 5/6 — REPORT GENERATOR + UI UTILITIES
# =============================================================================


# ================================================
# REPORT GENERATOR — PDF EXPORT (FPDF)
# ================================================

class ReportGenerator:
    """
    Tạo PDF report Enterprise: kết quả xếp hạng + phân tích giải thích.
    """

    @staticmethod
    def generate_pdf(analysis: AnalysisResult, filename="RiskCast_Report.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # ===== TITLE =====
        pdf.set_font("Arial", "B", 18)
        pdf.set_text_color(0, 150, 100)
        pdf.cell(0, 12, "RISKCAST Enterprise — Insurance Analysis Report", 0, 1, "C")
        pdf.ln(4)

        # ===== SUMMARY =====
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 160, 120)
        pdf.cell(0, 10, "Top 3 Recommendations", 0, 1)

        pdf.set_font("Arial", "", 12)
        top3 = analysis.results.head(3)

        for _, row in top3.iterrows():
            pdf.set_text_color(0, 0, 0)
            txt = (
                f"- {row['company']} ({row['icc_package']}): "
                f"Score={row['score']:.3f}, "
                f"Cost=${row['estimated_cost']:,.0f}"
            )
            pdf.multi_cell(0, 8, txt)
        pdf.ln(4)

        # ===== FULL TABLE =====
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0, 160, 120)
        pdf.cell(0, 10, "Full Ranking Table", 0, 1)

        df = analysis.results[[
            "rank", "company", "icc_package",
            "score", "estimated_cost", "C6_mean"
        ]]

        pdf.set_font("Arial", "", 10)
        for _, row in df.iterrows():
            pdf.set_text_color(0, 0, 0)
            txt = (
                f"#{int(row['rank'])} — {row['company']} ({row['icc_package']}): "
                f"Score={row['score']:.3f}, "
                f"Cost=${row['estimated_cost']:,.0f}, "
                f"ClimateRisk={row['C6_mean']:.2f}"
            )
            pdf.multi_cell(0, 7, txt)

        # ===== VAR / CVAR =====
        pdf.ln(5)
        if analysis.var is not None:
            pdf.set_font("Arial", "B", 13)
            pdf.set_text_color(120, 30, 0)
            pdf.cell(0, 10, "Risk Metrics (VaR / CVaR)", 0, 1)

            pdf.set_font("Arial", "", 12)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 8, f"VaR 95%: ${analysis.var:,.0f}")
            pdf.multi_cell(0, 8, f"CVaR 95%: ${analysis.cvar:,.0f}")

        # Save
        pdf.output(filename)
        return filename


# ================================================
# UI UTILITIES
# ================================================

def render_explanation_box(title: str, content: str):
    """Khung giải thích đẹp Premium Green."""
    st.markdown(f"""
    <div style="
        border: 1px solid #00e676;
        background: rgba(0, 255, 153, 0.07);
        padding: 18px;
        border-radius: 16px;
        margin-top: 12px;
        box-shadow: 0 0 12px rgba(0,255,153,0.15);
    ">
        <h3 style="color:#a5ffdc; margin-top:0;"><b>{title}</b></h3>
        <div style="font-size:1.08rem; color:#e6fff7; line-height:1.5;">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_top3_card_html(rank: int, row):
    """
    FIXED HTML CARD — bảo đảm không bị escape, dùng riêng cho render Top 3 Premium.
    """
    medals = ["🥇", "🥈", "🥉"]
    medal = medals[rank - 1]

    card_html = f"""
    <div style="
        border-radius: 18px;
        padding: 22px;
        background: linear-gradient(145deg, #003322, #001a12);
        border: 1px solid rgba(0,255,153,0.35);
        box-shadow: 0 0 18px rgba(0,255,153,0.25);
        margin-bottom: 15px;
    ">
        <div style="font-size:1.5rem; font-weight:800; color:#a5ffdc;">
            {medal} #{rank}: {row['company']} — {row['icc_package']}
        </div>

        <div style="margin-top:10px; font-size:1.1rem; color:#e6fff7;">
            📊 Điểm: <b>{row['score']:.3f}</b> ·
            💰 Chi phí: <b>${row['estimated_cost']:,.0f}</b> <br>
            🌪 Biến động khí hậu: <b>{row['C6_std']:.3f}</b> ·
            🎯 Tin cậy: <b>{row['confidence']:.2f}</b>
        </div>
    </div>
    """

    return card_html
# =============================================================================
# PART 6/6 — STREAMLIT UI + MAIN
# =============================================================================

def app():
    apply_custom_css()
    st.title("🟩 RISKCAST v5.3 — Enterprise Edition")

    st.markdown("""
    <h3 style='color:#a5ffdc; font-weight:700;'>
        ESG Logistics — Multi-Package Insurance Analysis Dashboard
    </h3>
    """, unsafe_allow_html=True)

    # ===============================
    # SIDEBAR INPUT FORM
    # ===============================
    st.sidebar.header("⚙️ Cài đặt phân tích")
    cargo_value = st.sidebar.number_input("Giá trị lô hàng ($)", 1000, 500000, 50000, 1000)

    good_type = st.sidebar.selectbox("Loại hàng", ["Electronics", "Furniture", "Food", "Chemicals", "Garments"])

    route = st.sidebar.selectbox("Tuyến vận chuyển", [
        "VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"
    ])

    method = st.sidebar.selectbox("Phương thức", ["Sea", "Air"])

    month = st.sidebar.slider("Tháng vận chuyển", 1, 12, 9)

    priority = st.sidebar.selectbox(
        "Mục tiêu ưu tiên",
        ["💰 Tiết kiệm chi phí", "⚖️ Cân bằng", "🛡️ An toàn tối đa"]
    )

    # FLAGS
    use_fuzzy = st.sidebar.checkbox("Fuzzy AHP (Không chắc chắn)", True)
    fuzzy_uncertainty = st.sidebar.slider("Mức độ mờ (%)", 1, 30, 12)

    use_mc = st.sidebar.checkbox("Monte Carlo (C6 khí hậu)", True)
    mc_runs = st.sidebar.slider("Số lần mô phỏng", 500, 6000, 2000, 500)

    use_arima = st.sidebar.checkbox("Dự báo ARIMA", True)
    use_var = st.sidebar.checkbox("Tính VaR / CVaR", False)

    run_btn = st.sidebar.button("🚀 Chạy phân tích", use_container_width=True)

    # ===============================
    # LOAD DATA
    # ===============================
    historical = DataService.load_historical_data()
    analyzer = MultiPackageAnalyzer()

    if run_btn:
        with st.spinner("⏳ Đang chạy mô hình phân tích..."):
            params = AnalysisParams(
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
                fuzzy_uncertainty=fuzzy_uncertainty
            )

            analysis = analyzer.run_analysis(params, historical)

        st.success("✨ Hoàn tất!")

        # ===============================
        # TOP 3 RECOMMENDATIONS
        # ===============================
        st.subheader("🏆 Top 3 Phương án Tối ưu")
        top3 = analysis.results.head(3)

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        for i in range(3):
            with cols[i]:
                card = render_top3_card_html(i + 1, top3.iloc[i])
                st.markdown(card, unsafe_allow_html=True)

        render_explanation_box(
            "Vì sao có kết quả này?",
            """
            • **TOPSIS** đánh giá dựa trên 6 tiêu chí quan trọng trong bảo hiểm vận chuyển.<br>
            • **Monte Carlo** mô phỏng rủi ro khí hậu (C6) dựa trên biến động theo mùa và tuyến vận chuyển.<br>
            • **Fuzzy AHP** điều chỉnh trọng số theo mức độ không chắc chắn.<br>
            • Gói **ICC A**, **ICC B**, **ICC C** ảnh hưởng trực tiếp đến mức bảo vệ và phí bảo hiểm.<br>
            """
        )

        st.markdown("---")

        # ===============================
        # FULL TABLE (15 options)
        # ===============================
        st.subheader("📋 Bảng 15 Phương án (5 công ty × 3 gói ICC)")

        df_show = analysis.results.copy()
        st.dataframe(df_show, use_container_width=True)

        # ===============================
        # CHARTS
        # ===============================
        st.subheader("📊 Phân tích Đa chiều")

        chart1 = ChartFactory.create_cost_benefit_scatter(analysis.results)
        st.plotly_chart(chart1, use_container_width=True)

        chart2 = ChartFactory.create_category_comparison(analysis.results)
        st.plotly_chart(chart2, use_container_width=True)

        chart3 = ChartFactory.create_top_recommendations_bar(analysis.results)
        st.plotly_chart(chart3, use_container_width=True)

        chart4 = ChartFactory.create_forecast_chart(
            analysis.historical,
            analysis.forecast,
            route,
            month
        )
        st.plotly_chart(chart4, use_container_width=True)

        # ===============================
        # EXPORT PDF
        # ===============================
        st.markdown("### 📄 Xuất báo cáo phân tích (PDF)")

        if st.button("Tải PDF", type="primary"):
            filename = ReportGenerator.generate_pdf(analysis)
            with open(filename, "rb") as f:
                st.download_button(
                    "📥 Download PDF",
                    data=f,
                    file_name="RiskCast_Report.pdf",
                    mime="application/pdf"
                )


# Run app
if __name__ == "__main__":
    app()
