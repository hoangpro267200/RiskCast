# =============================================================================
# RISKCAST v5.4 — ENTERPRISE EDITION (Card Layout FIX)
# Tác giả: Bùi Xuân Hoàng (ý tưởng gốc)
# Enterprise UX, Shield Logo, Layout Stable v5.4: Kai assistant
# =============================================================================

import io
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Optional ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
def app_config():
    st.set_page_config(
        page_title="RISKCAST v5.4 — Enterprise Edition",
        page_icon="🛡️",
        layout="wide"
    )


# -----------------------------------------------------------------------------
# ENTERPRISE CSS (đã thêm card tách biểu đồ)
# -----------------------------------------------------------------------------
def apply_enterprise_css():
    st.markdown("""
    <style>

    /* BASE */
    .stApp {
        background: #0e1613 !important;
        color: #eafff5 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .block-container {
        padding-top: 1rem !important;
        max-width: 1500px;
    }

    /* SHIELD LOGO */
    .rc-logo-shield {
        width: 82px;
        height: 82px;
        border-radius: 18px;
        background: radial-gradient(circle at 40% 40%,
            #ffffff 0%, #ccfff2 14%, #00e6a7 48%, #003826 96%);
        border: 3px solid #baffea;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 1.8rem;
        color: #001c12;
        box-shadow: 0 0 22px rgba(0,255,180,0.55);
    }

    /* HEADER */
    .rc-header {
        padding: 1.5rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(18,44,36,0.95), rgba(5,15,12,0.98));
        border: 1px solid rgba(0,255,153,0.22);
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }

    .rc-title {
        font-size: 1.55rem;
        font-weight: 900;
        background: linear-gradient(90deg, #eafff8, #b9ffdd);
        -webkit-background-clip: text;
        color: transparent;
        text-transform: uppercase;
    }

    .rc-subtitle {
        font-size: 0.95rem;
        color: #caffea;
        opacity: 0.9;
    }

    /* CARD (FIX chính để tách 2 biểu đồ) */
    .rc-card {
        background: linear-gradient(140deg, rgba(10,25,21,0.75), rgba(5,15,12,0.9));
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        border: 1px solid rgba(0,255,153,0.18);
        box-shadow: 0 8px 22px rgba(0,0,0,0.55);
        margin-bottom: 1.6rem;
    }

    </style>
    """, unsafe_allow_html=True)
# -----------------------------------------------------------------------------
# DOMAIN MODELS & CONSTANTS
# -----------------------------------------------------------------------------
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
    var: float | None
    cvar: float | None
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

PRIORITY_PROFILES: Dict[str, Dict[str, float]] = {
    "💰 Tiết kiệm chi phí": {
        "C1: Tỷ lệ phí": 0.35,
        "C2: Thời gian xử lý": 0.10,
        "C3: Tỷ lệ tổn thất": 0.15,
        "C4: Hỗ trợ ICC": 0.15,
        "C5: Chăm sóc KH": 0.10,
        "C6: Rủi ro khí hậu": 0.15,
    },
    "⚖️ Cân bằng": {
        "C1: Tỷ lệ phí": 0.20,
        "C2: Thời gian xử lý": 0.15,
        "C3: Tỷ lệ tổn thất": 0.20,
        "C4: Hỗ trợ ICC": 0.20,
        "C5: Chăm sóc KH": 0.10,
        "C6: Rủi ro khí hậu": 0.15,
    },
    "🛡️ An toàn tối đa": {
        "C1: Tỷ lệ phí": 0.10,
        "C2: Thời gian xử lý": 0.10,
        "C3: Tỷ lệ tổn thất": 0.25,
        "C4: Hỗ trợ ICC": 0.25,
        "C5: Chăm sóc KH": 0.10,
        "C6: Rủi ro khí hậu": 0.20,
    },
}

ICC_PACKAGES = {
    "ICC A": {
        "coverage": 1.0,
        "premium_multiplier": 1.5,
        "description": "Bảo vệ toàn diện mọi rủi ro trừ điều khoản loại trừ (All Risks)",
    },
    "ICC B": {
        "coverage": 0.75,
        "premium_multiplier": 1.0,
        "description": "Bảo vệ các rủi ro chính (hỏa hoạn, va chạm, chìm đắm, Named Perils)",
    },
    "ICC C": {
        "coverage": 0.5,
        "premium_multiplier": 0.65,
        "description": "Bảo vệ cơ bản (chỉ các rủi ro lớn như chìm, cháy, va chạm nghiêm trọng)",
    },
}

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
    "PVI": 1.05,
    "BaoViet": 1.00,
    "BaoMinh": 1.02,
    "MIC": 1.03,
}


# -----------------------------------------------------------------------------
# DATA SERVICE
# -----------------------------------------------------------------------------
class DataService:
    """Quản lý dữ liệu công ty + lịch sử khí hậu."""

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_historical_data() -> pd.DataFrame:
        climate_base = {
            "VN - EU": [0.28, 0.30, 0.35, 0.40, 0.52, 0.60, 0.67, 0.70, 0.75, 0.72, 0.60, 0.48],
            "VN - US": [0.33, 0.36, 0.40, 0.46, 0.55, 0.63, 0.72, 0.78, 0.80, 0.74, 0.62, 0.50],
            "VN - Singapore": [0.18, 0.20, 0.24, 0.27, 0.32, 0.36, 0.40, 0.43, 0.45, 0.42, 0.35, 0.30],
            "VN - China": [0.20, 0.23, 0.27, 0.31, 0.38, 0.42, 0.48, 0.50, 0.53, 0.49, 0.40, 0.34],
            "Domestic": [0.12, 0.13, 0.14, 0.16, 0.20, 0.22, 0.23, 0.25, 0.27, 0.24, 0.20, 0.18],
        }
        df = pd.DataFrame({"month": list(range(1, 13))})
        for route, vals in climate_base.items():
            df[route] = vals
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


# -----------------------------------------------------------------------------
# CORE ALGORITHMS
# -----------------------------------------------------------------------------
class FuzzyAHP:
    @staticmethod
    def apply(weights: pd.Series, uncertainty_pct: float) -> pd.Series:
        """Đơn giản hóa: tam giác mờ (low, mid, high) → centroid rồi chuẩn hóa."""
        factor = uncertainty_pct / 100.0
        w = weights.values.astype(float)

        low = np.maximum(w * (1 - factor), 1e-9)
        high = np.minimum(w * (1 + factor), 0.9999)
        centroid = (low + w + high) / 3.0

        centroid = centroid / centroid.sum()
        return pd.Series(centroid, index=weights.index)


class MonteCarloSimulator:
    @staticmethod
    @st.cache_data(ttl=600)
    def simulate(
        base_risk: float,
        sensitivity_map: Dict[str, float],
        n_sim: int,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())
        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.03, mu * 0.12)

        sims = rng.normal(loc=mu, scale=sigma, size=(n_sim, len(companies)))
        sims = np.clip(sims, 0.0, 1.0)

        return companies, sims.mean(axis=0), sims.std(axis=0)


class TOPSISAnalyzer:
    @staticmethod
    def analyze(
        data: pd.DataFrame,
        weights: pd.Series,
        cost_benefit: Dict[str, CriterionType],
    ) -> np.ndarray:
        cols = list(weights.index)
        M = data[cols].values.astype(float)

        denom = np.sqrt((M ** 2).sum(axis=0))
        denom[denom == 0] = 1.0
        R = M / denom

        V = R * weights.values

        is_cost = np.array([cost_benefit[c] == CriterionType.COST for c in cols])
        ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
        ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))

        d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

        return d_minus / (d_plus + d_minus + 1e-9)


class RiskCalculator:
    @staticmethod
    def calculate_var_cvar(
        loss_rates: np.ndarray,
        cargo_value: float,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        if len(loss_rates) == 0:
            return 0.0, 0.0

        losses = np.asarray(loss_rates) * cargo_value
        var = float(np.percentile(losses, confidence * 100))

        tail = losses[losses >= var]
        cvar = float(tail.mean()) if len(tail) > 0 else var
        return var, cvar


class Forecaster:
    @staticmethod
    def forecast(
        historical_df: pd.DataFrame,
        route: str,
        current_month: int,
        use_arima: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if route not in historical_df.columns:
            route = historical_df.columns[1]

        full_series = historical_df[route].values
        n_total = len(full_series)

        current_month = max(1, min(current_month, n_total))
        hist = full_series[:current_month].astype(float)

        if use_arima and ARIMA_AVAILABLE and len(hist) >= 6:
            try:
                model = ARIMA(hist, order=(1, 1, 1))
                fitted = model.fit()
                fc = fitted.forecast(1)
                val = float(np.clip(fc[0], 0.0, 1.0))
                return hist, np.array([val])
            except Exception:
                pass

        if len(hist) >= 3:
            trend = (hist[-1] - hist[-3]) / 2.0
        elif len(hist) == 2:
            trend = hist[-1] - hist[-2]
        else:
            trend = 0.0

        next_val = float(np.clip(hist[-1] + trend, 0.0, 1.0))
        return hist, np.array([next_val])
# -----------------------------------------------------------------------------
# MULTI-PACKAGE ANALYZER
# 5 công ty × 3 gói ICC = 15 phương án
# -----------------------------------------------------------------------------
class MultiPackageAnalyzer:
    @staticmethod
    def build_dataset(
        company_data: pd.DataFrame,
        route: str,
        month: int,
        cargo_value: float,
    ) -> pd.DataFrame:
        rows = []
        companies = company_data.index.tolist()

        base_climate = DataService.load_historical_data()
        hist_vals = base_climate[route].values
        month = max(1, min(month, 12))
        climate_risk = hist_vals[month - 1]  # 0 → 1

        for comp in companies:
            base_row = company_data.loc[comp].to_dict()

            for icc, desc in ICC_PACKAGES.items():
                premium_multiplier = desc["premium_multiplier"]
                coverage = desc["coverage"]

                premium_cost = float(base_row["C1: Tỷ lệ phí"] * premium_multiplier)

                rows.append({
                    "Company": comp,
                    "ICC": icc,
                    "PremiumCost": premium_cost,
                    "ClimateRisk": climate_risk,
                    **base_row,
                })

        return pd.DataFrame(rows)

    @staticmethod
    def analyze(
        params: AnalysisParams,
        company_df: pd.DataFrame,
        climate_df: pd.DataFrame,
    ) -> AnalysisResult:

        df = MultiPackageAnalyzer.build_dataset(
            company_data=company_df,
            route=params.route,
            month=params.month,
            cargo_value=params.cargo_value
        )

        w = pd.Series(PRIORITY_PROFILES[params.priority])

        if params.use_fuzzy:
            w = FuzzyAHP.apply(w, params.fuzzy_uncertainty)

        df["C6: Rủi ro khí hậu"] = df["ClimateRisk"]

        criteria_cols = w.index.tolist()
        data_adjusted = df.copy()

        cb_m = COST_BENEFIT_MAP
        scores = TOPSISAnalyzer.analyze(
            data_adjusted[criteria_cols],
            w,
            cb_m
        )

        data_adjusted["TOPSIS"] = scores

        data_adjusted["TotalCost"] = (
            data_adjusted["C1: Tỷ lệ phí"] * 10000 * 
            data_adjusted["PremiumCost"] * 0.85
        )

        if params.use_mc:
            base_risk = float(data_adjusted["C3: Tỷ lệ tổn thất"].mean())
            comps, mu, sig = MonteCarloSimulator.simulate(
                base_risk=base_risk,
                sensitivity_map=SENSITIVITY_MAP,
                n_sim=params.mc_runs
            )
            mc_loss = mu
        else:
            mc_loss = None

        if params.use_var:
            ratio = float(data_adjusted["C3: Tỷ lệ tổn thất"].mean())
            simulated_losses = np.random.normal(
                loc=ratio, scale=max(0.02, ratio * 0.15), size=500
            )
            simulated_losses = np.clip(simulated_losses, 0, 1)

            var, cvar = RiskCalculator.calculate_var_cvar(
                simulated_losses,
                cargo_value=params.cargo_value,
                confidence=0.95,
            )
        else:
            var, cvar = None, None

        hist, forecast = Forecaster.forecast(
            climate_df,
            route=params.route,
            current_month=params.month,
            use_arima=params.use_arima,
        )

        return AnalysisResult(
            results=data_adjusted.sort_values("TOPSIS", ascending=False).reset_index(drop=True),
            weights=w,
            data_adjusted=data_adjusted,
            var=var,
            cvar=cvar,
            historical=hist,
            forecast=forecast,
        )
# -----------------------------------------------------------------------------
# CHART FACTORY
# -----------------------------------------------------------------------------
class ChartFactory:
    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=22, color="#e6fff7"),
                x=0.5,
            ),
            font=dict(size=14, color="#e6fff7"),
            plot_bgcolor="#001016",
            paper_bgcolor="#000c11",
            margin=dict(l=60, r=40, t=70, b=60),
            legend=dict(
                bgcolor="rgba(0,0,0,0.35)",
                bordercolor="#00e676",
                borderwidth=1,
                font=dict(size=12),
            ),
            height=460,
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#00332b",
            tickfont=dict(size=12, color="#e6fff7"),
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#00332b",
            tickfont=dict(size=12, color="#e6fff7"),
        )
        return fig

    # ------------------------- PIE TRỌNG SỐ -------------------------
    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        labels_full = list(weights.index)
        labels_short = [c.split(":")[0] for c in labels_full]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels_full,
                    values=weights.values,
                    text=labels_short,
                    textinfo="text+percent",
                    textposition="inside",
                    hole=0.2,
                    marker=dict(
                        colors=[
                            "#00e676",
                            "#69f0ae",
                            "#b9f6ca",
                            "#1de9b6",
                            "#00bfa5",
                            "#64ffda",
                        ],
                        line=dict(color="#00130d", width=2),
                    ),
                    pull=[0.04] * len(weights),
                    hovertemplate="<b>%{label}</b><br>Tỷ trọng: %{percent}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=20, color="#a5ffdc"),
                x=0.5,
                y=0.96,
            ),
            showlegend=True,
            legend=dict(
                title="<b>Các tiêu chí</b>",
                font=dict(size=13, color="#e6fff7"),
            ),
            paper_bgcolor="#001016",
            plot_bgcolor="#001016",
            margin=dict(l=0, r=0, t=70, b=0),
            height=460,
        )
        return fig

    # ---------------------- SCATTER COST–BENEFIT --------------------
    @staticmethod
    def create_cost_benefit_scatter(results: pd.DataFrame) -> go.Figure:
        # Đảm bảo có cột chi phí và điểm
        df = results.copy()
        if "TotalCost" not in df.columns:
            df["TotalCost"] = df.get("EstimatedCost", 0.0)
        if "TOPSIS" not in df.columns:
            df["TOPSIS"] = df.get("score", 0.0)

        color_map = {
            "ICC A": "#ff6b6b",
            "ICC B": "#ffd93d",
            "ICC C": "#6bcf7f",
        }

        fig = go.Figure()

        for icc in ["ICC C", "ICC B", "ICC A"]:
            d = df[df["ICC"] == icc]
            if d.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=d["TotalCost"],
                    y=d["TOPSIS"],
                    mode="markers+text",
                    name=icc,
                    text=d["Company"],
                    textposition="top center",
                    marker=dict(
                        size=15,
                        color=color_map.get(icc, "#ffffff"),
                        line=dict(width=2, color="#000000"),
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"Gói: {icc}<br>"
                        "Chi phí: $%{x:,.0f}<br>"
                        "Điểm TOPSIS: %{y:.3f}<extra></extra>"
                    ),
                )
            )

        fig.update_xaxes(title="<b>Chi phí ước tính ($)</b>")
        fig.update_yaxes(title="<b>Điểm TOPSIS</b>", range=[0, 1])

        return ChartFactory._apply_theme(fig, "💰 Chi phí vs Chất lượng (Cost–Benefit)")

    # ---------------------- SO SÁNH 3 LOẠI PHƯƠNG ÁN ----------------
    @staticmethod
    def create_category_comparison(results: pd.DataFrame) -> go.Figure:
        df = results.copy()
        if "TotalCost" not in df.columns:
            df["TotalCost"] = df.get("EstimatedCost", 0.0)
        if "TOPSIS" not in df.columns:
            df["TOPSIS"] = df.get("score", 0.0)

        # Xác định loại từ ICC
        def map_cat(icc: str) -> str:
            if icc == "ICC C":
                return "💰 Tiết kiệm"
            if icc == "ICC B":
                return "⚖️ Cân bằng"
            return "🛡️ An toàn"

        df["Category"] = df["ICC"].apply(map_cat)

        categories = ["💰 Tiết kiệm", "⚖️ Cân bằng", "🛡️ An toàn"]
        avg_scores = []
        avg_costs = []

        for cat in categories:
            sub = df[df["Category"] == cat]
            if len(sub) == 0:
                avg_scores.append(0.0)
                avg_costs.append(0.0)
            else:
                avg_scores.append(sub["TOPSIS"].mean())
                avg_costs.append(sub["TotalCost"].mean())

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Điểm trung bình",
                x=categories,
                y=avg_scores,
                marker=dict(color="#00e676"),
                yaxis="y",
                hovertemplate="<b>%{x}</b><br>Điểm TB: %{y:.3f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                name="Chi phí trung bình",
                x=categories,
                y=avg_costs,
                mode="lines+markers",
                marker=dict(size=9, color="#ffeb3b"),
                line=dict(width=3, color="#ffeb3b"),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Chi phí TB: $%{y:,.0f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text="<b>📊 So sánh 3 loại phương án</b>",
                font=dict(size=22, color="#e6fff7"),
                x=0.5,
            ),
            paper_bgcolor="#000c11",
            plot_bgcolor="#001016",
            font=dict(color="#e6fff7"),
            margin=dict(l=60, r=60, t=70, b=60),
            height=460,
            legend=dict(
                bgcolor="rgba(0,0,0,0.35)",
                bordercolor="#00e676",
                borderwidth=1,
            ),
            yaxis=dict(
                title="<b>Điểm TOPSIS</b>",
                range=[0, 1],
                tickfont=dict(color="#00e676"),
            ),
            yaxis2=dict(
                title="<b>Chi phí ($)</b>",
                overlaying="y",
                side="right",
                tickfont=dict(color="#ffeb3b"),
            ),
        )

        return fig

    # ---------------------- TOP 5 BAR HORIZONTAL --------------------
    @staticmethod
    def create_top5_bar(results: pd.DataFrame) -> go.Figure:
        df = results.copy()
        if "TOPSIS" not in df.columns:
            df["TOPSIS"] = df.get("score", 0.0)
        if "TotalCost" not in df.columns:
            df["TotalCost"] = df.get("EstimatedCost", 0.0)

        df = df.sort_values("TOPSIS", ascending=False).head(5)
        df["Label"] = df["Company"] + " - " + df["ICC"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df["TOPSIS"],
                    y=df["Label"],
                    orientation="h",
                    text=[f"{v:.3f}" for v in df["TOPSIS"]],
                    textposition="outside",
                    marker=dict(
                        color=df["TOPSIS"],
                        colorscale=[[0, "#69f0ae"], [0.5, "#00e676"], [1, "#00c853"]],
                        line=dict(color="#00130d", width=1),
                    ),
                    hovertemplate="<b>%{y}</b><br>Điểm: %{x:.3f}<br>Chi phí: $%{customdata:,.0f}<extra></extra>",
                    customdata=df["TotalCost"],
                )
            ]
        )

        fig.update_xaxes(title="<b>Điểm TOPSIS</b>", range=[0, 1])
        fig.update_yaxes(title="<b>Phương án</b>")

        return ChartFactory._apply_theme(fig, "🏆 Top 5 phương án tốt nhất")

    # ---------------------- FORECAST CHART --------------------------
    @staticmethod
    def create_forecast_chart(
        historical: np.ndarray,
        forecast: np.ndarray,
        route: str,
        selected_month: int,
    ) -> go.Figure:
        hist = np.asarray(historical, dtype=float)
        fc = np.asarray(forecast, dtype=float)

        months_hist = list(range(1, len(hist) + 1))
        next_month = selected_month % 12 + 1
        months_fc = [next_month]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=months_hist,
                y=hist,
                mode="lines+markers",
                name="📈 Lịch sử",
                line=dict(color="#00e676", width=3),
                marker=dict(size=8),
                hovertemplate="Tháng %{x}<br>Rủi ro: %{y:.1%}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=months_fc,
                y=fc,
                mode="lines+markers",
                name="🔮 Dự báo",
                line=dict(color="#ffeb3b", width=3, dash="dash"),
                marker=dict(size=10, symbol="diamond"),
                hovertemplate="Tháng %{x}<br>Dự báo: %{y:.1%}<extra></extra>",
            )
        )

        fig = ChartFactory._apply_theme(
            fig,
            f"Dự báo rủi ro khí hậu — {route}",
        )

        fig.update_xaxes(
            title="<b>Tháng</b>",
            tickmode="linear",
            tick0=1,
            dtick=1,
            range=[1, 12],
            tickvals=list(range(1, 13)),
        )

        max_val = max(float(hist.max()), float(fc.max()))
        fig.update_yaxes(
            title="<b>Mức rủi ro (0–1)</b>",
            range=[0, max(1.0, max_val * 1.15)],
            tickformat=".0%",
        )

        return fig
# -----------------------------------------------------------------------------
# STREAMLIT UI — HEADER + SIDEBAR + INPUT FORM
# -----------------------------------------------------------------------------

def render_header():
    st.markdown("""
    <div class="rc-header">
        <div class="rc-header-left">
            <div class="rc-logo-shield">RC</div>
            <div>
                <div class="rc-title">RISKCAST v5.4 — ENTERPRISE EDITION</div>
                <div class="rc-subtitle">ESG Logistics · ICC Packages · Multi-Scenario Risk Engine</div>
            </div>
        </div>
        <div class="rc-badge">Enterprise</div>
    </div>
    """, unsafe_allow_html=True)


def sidebar_inputs() -> AnalysisParams:
    st.sidebar.title("🔧 Cấu hình phân tích")

    cargo_value = st.sidebar.number_input(
        "💼 Giá trị lô hàng (USD)",
        min_value=1000.0,
        max_value=5_000_000.0,
        value=50_000.0,
        step=1000.0,
        help="Dùng để tính thiệt hại VaR/CVaR và chi phí ước tính."
    )

    good_type = st.sidebar.selectbox(
        "📦 Loại hàng hóa",
        ["Hàng dễ vỡ", "Hàng giá trị cao", "Hàng thường"]
    )

    route = st.sidebar.selectbox(
        "🌍 Tuyến vận chuyển",
        ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"]
    )

    method = st.sidebar.selectbox(
        "🚢 Phương thức",
        ["Đường biển", "Đường không", "Đa phương thức"]
    )

    month = st.sidebar.slider(
        "🗓️ Tháng hiện tại",
        1, 12, 9
    )

    priority = st.sidebar.selectbox(
        "🎯 Ưu tiên phân tích",
        list(PRIORITY_PROFILES.keys())
    )

    # -------------------- MODULE OPTIONS --------------------
    st.sidebar.markdown("### ⚙️ Thuật toán nâng cao")

    use_fuzzy = st.sidebar.checkbox(
        "Fuzzy AHP (độ bất định)", value=True)

    fuzzy_uncertainty = 10.0
    if use_fuzzy:
        fuzzy_uncertainty = st.sidebar.slider(
            "Độ bất định fuzzy (%)",
            5, 40, 15
        )

    use_arima = st.sidebar.checkbox(
        "ARIMA Forecast", value=True
    )

    use_mc = st.sidebar.checkbox(
        "Monte Carlo Simulation", value=True
    )

    mc_runs = 500
    if use_mc:
        mc_runs = st.sidebar.slider(
            "Số lần mô phỏng Monte Carlo",
            200, 2000, 800, step=100
        )

    use_var = st.sidebar.checkbox(
        "VaR / CVaR", value=True
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
# -----------------------------------------------------------------------------
# MAIN LAYOUT — TÁCH 2 BIỂU ĐỒ, DÙNG CARD ENTERPRISE
# -----------------------------------------------------------------------------

def render_results(result: AnalysisResult, params: AnalysisParams):
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📊 Kết quả phân tích 15 phương án")

    # ========================= TOP RECOMMENDATION =============================
    top = result.results.iloc[0]

    st.markdown(f"""
    <div class="result-box">
        🏆 <b>Phương án tối ưu</b><br><br>
        <span style="font-size:1.6rem;">{top['Company']} — {top['ICC']}</span><br><br>
        💰 Chi phí ước tính: <b>{top['PremiumCost']:.2%}</b><br>
        📊 Điểm TOPSIS: <b>{top['TOPSIS']:.3f}</b><br>
        🌪 Rủi ro khí hậu: <b>{top['ClimateRisk']:.2%}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗂️ Bảng xếp hạng 15 phương án")
    df_show = result.results.copy()

    df_show = df_show[[
        "Company", "ICC", "PremiumCost", "TOPSIS", "ClimateRisk"
    ]].rename(columns={
        "Company": "Công ty",
        "ICC": "Gói ICC",
        "PremiumCost": "Phí bảo hiểm",
        "TOPSIS": "Điểm",
        "ClimateRisk": "Rủi ro khí hậu"
    })

    st.dataframe(df_show, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)



    # =========================================================================
    # 🔥 CARD LAYER — TÁCH 2 BIỂU ĐỒ THÀNH 2 HỘP ĐỘC LẬP (KHÔNG DÍNH NHAU)
    # =========================================================================
    st.subheader("📉 Phân tích trực quan")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='rc-card'>", unsafe_allow_html=True)
        st.markdown("#### 💰 Cost–Benefit Analysis")
        fig_cb = ChartFactory.cost_benefit(result.results)
        st.plotly_chart(fig_cb, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='rc-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 So sánh 3 nhóm ICC")
        fig_cat = ChartFactory.category_compare(result.results)
        st.plotly_chart(fig_cat, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)



    # =========================================================================
    # 🔥 HÀNG 2 — PIE + FORECAST (CŨNG TÁCH CARD RIÊNG)
    # =========================================================================
    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.markdown("<div class='rc-card'>", unsafe_allow_html=True)
        st.markdown("#### 🧮 Trọng số tiêu chí")
        fig_pie = ChartFactory.weights_pie(result.weights)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='rc-card'>", unsafe_allow_html=True)
        st.markdown("#### 🌦 Dự báo rủi ro khí hậu")
        fig_fc = ChartFactory.forecast(
            result.historical,
            result.forecast,
            params.route,
            params.month
        )
        st.plotly_chart(fig_fc, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)



    # =========================================================================
    # 🔥 RISK CARD — VaR/CVaR (THÀNH 1 CARD ĐẸP)
    # =========================================================================
    if params.use_var:
        st.markdown("<div class='rc-card'>", unsafe_allow_html=True)
        st.markdown("### ⚠️ Đánh giá rủi ro tài chính (VaR / CVaR)")

        st.write(f"**VaR 95%:** ${result.var:,.0f}")
        st.write(f"**CVaR 95%:** ${result.cvar:,.0f}")

        ratio = (result.var / params.cargo_value) * 100
        st.write(f"**Tỷ lệ rủi ro / giá trị hàng:** {ratio:.1f}%")

        if ratio < 10:
            st.success("Rủi ro ở mức chấp nhận được.")
        else:
            st.error("Rủi ro cao — nên chọn gói ICC A để tăng bảo vệ.")

        st.markdown("</div>", unsafe_allow_html=True)
# -----------------------------------------------------------------------------
# MAIN APP — KẾT NỐI TẤT CẢ MODULE
# -----------------------------------------------------------------------------

def main():
    # Cấu hình page + CSS Premium
    app_config()
    apply_enterprise_css()

    # HEADER
    render_header()

    # SIDEBAR INPUTS
    params = sidebar_inputs()

    # NÚT PHÂN TÍCH
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("🚀 PHÂN TÍCH 15 PHƯƠNG ÁN", use_container_width=True):
        with st.spinner("🔄 Đang chạy mô hình phân tích..."):
            try:
                climate_df = DataService.load_historical_data()
                company_df = DataService.get_company_data()

                result = MultiPackageAnalyzer.analyze(
                    params=params,
                    company_df=company_df,
                    climate_df=climate_df
                )

                render_results(result, params)

            except Exception as e:
                st.error(f"❌ Lỗi khi phân tích: {e}")
                st.exception(e)

    else:
        st.info("⬅️ Nhập thông tin ở sidebar và nhấn nút **PHÂN TÍCH** để chạy mô hình.")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
