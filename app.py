# =============================================================================
# RISKCAST v5.3.1 — ENTERPRISE EDITION (Multi-Package Analysis)
# ESG Logistics Risk Assessment Dashboard
#
# Author: Bùi Xuân Hoàng (original idea)
# Refactor + Multi-Package + Smart Recommendation: Kai assistant
#
# Điểm mới v5.3.1:
#   - Giữ toàn bộ logic Multi-Package (5 công ty × 3 gói ICC = 15 phương án)
#   - Fix toàn bộ title font + theme Enterprise Premium Green
#   - Chuẩn bị nền tảng cho phần giải thích chi tiết (Top 3, điểm mạnh, VaR/CVaR, Fuzzy…)
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
    """Kết quả phân tích tổng hợp."""
    results: pd.DataFrame          # Bảng 15 phương án sau khi xếp hạng
    weights: pd.Series             # Trọng số tiêu chí (đã Fuzzy nếu bật)
    data_adjusted: pd.DataFrame    # Dữ liệu đã điều chỉnh theo gói ICC
    var: Optional[float]           # VaR (nếu tính)
    cvar: Optional[float]          # CVaR (nếu tính)
    historical: np.ndarray         # Chuỗi rủi ro lịch sử (C6) theo tháng
    forecast: np.ndarray           # Giá trị dự báo tháng tiếp theo


# Danh sách tiêu chí (đồng bộ với các cột dữ liệu)
CRITERIA = [
    "C1: Tỷ lệ phí",
    "C2: Thời gian xử lý",
    "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC",
    "C5: Chăm sóc KH",
    "C6: Rủi ro khí hậu"
]

# Profile weights - Trọng số theo mục tiêu người dùng
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

# Định nghĩa 3 gói ICC
ICC_PACKAGES = {
    "ICC A": {
        "coverage": 1.0,        # Bảo vệ toàn diện
        "premium_multiplier": 1.5,
        "description": "Bảo vệ toàn diện mọi rủi ro trừ điều khoản loại trừ."
    },
    "ICC B": {
        "coverage": 0.75,       # Bảo vệ vừa phải
        "premium_multiplier": 1.0,
        "description": "Bảo vệ các rủi ro chính (hỏa hoạn, va chạm, chìm đắm)."
    },
    "ICC C": {
        "coverage": 0.5,        # Bảo vệ cơ bản
        "premium_multiplier": 0.65,
        "description": "Bảo vệ cơ bản (chỉ các rủi ro lớn)."
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

# Độ nhạy rủi ro khí hậu theo công ty (dùng cho Monte Carlo)
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
    st.markdown(
        """
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

        h1 {
            font-size: 2.8rem !important;
            font-weight: 900 !important;
            letter-spacing: 0.03em;
        }
        h2 {
            font-size: 2.1rem !important;
            font-weight: 800 !important;
        }
        h3 {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
        }

        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.1rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(120deg,
                                        rgba(0, 255, 153, 0.14),
                                        rgba(0, 0, 0, 0.88));
            border: 1px solid rgba(0, 255, 153, 0.45);
            box-shadow:
                0 0 0 1px rgba(0, 255, 153, 0.12),
                0 18px 45px rgba(0, 0, 0, 0.85);
            margin-bottom: 1.2rem;
            gap: 1.5rem;
        }

        .app-header-left {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .app-logo-circle {
            width: 64px;
            height: 64px;
            border-radius: 18px;
            background: radial-gradient(circle at 30% 30%,
                                        #b9f6ca 0%,
                                        #00c853 38%,
                                        #00381f 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 1.4rem;
            color: #00130d;
            box-shadow:
                0 0 14px rgba(0, 255, 153, 0.65),
                0 0 36px rgba(0, 0, 0, 0.75);
            border: 2px solid #e8f5e9;
        }

        .app-header-title {
            font-size: 1.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #e8fffb, #b9f6ca, #e8fffb);
            -webkit-background-clip: text;
            color: transparent;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .app-header-subtitle {
            font-size: 0.9rem;
            color: #ccffec;
            opacity: 0.9;
        }

        .app-header-badge {
            font-size: 0.86rem;
            font-weight: 600;
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            background: radial-gradient(circle at 0 0, #00e676, #00bfa5);
            color: #00130d;
            display: flex;
            align-items: center;
            gap: 0.35rem;
            white-space: nowrap;
            box-shadow:
                0 0 14px rgba(0, 255, 153, 0.65),
                0 0 22px rgba(0, 0, 0, 0.7);
        }

        section[data-testid="stSidebar"] {
            background: radial-gradient(circle at 0 0,
                                        #003322 0%,
                                        #000f0a 40%,
                                        #000805 100%) !important;
            border-right: 1px solid rgba(0, 230, 118, 0.55);
            box-shadow: 8px 0 22px rgba(0, 0, 0, 0.85);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.1rem;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #a5ffdc !important;
            font-weight: 800 !important;
        }

        section[data-testid="stSidebar"] label {
            color: #e0f2f1 !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }

        .stButton > button {
            background: linear-gradient(120deg, #00ff99, #00e676, #00bfa5) !important;
            color: #00130d !important;
            font-weight: 800 !important;
            border-radius: 999px !important;
            border: none !important;
            padding: 0.65rem 1.9rem !important;
            box-shadow:
                0 0 14px rgba(0, 255, 153, 0.7),
                0 10px 22px rgba(0, 0, 0, 0.85) !important;
            transition: all 0.12s ease-out;
            font-size: 0.98rem !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px) scale(1.02);
            box-shadow:
                0 0 20px rgba(0, 255, 153, 0.95),
                0 14px 30px rgba(0, 0, 0, 0.9) !important;
        }

        .premium-card {
            background: radial-gradient(circle at top left,
                                        rgba(0, 255, 153, 0.10),
                                        rgba(0, 0, 0, 0.95));
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(0, 255, 153, 0.45);
            box-shadow:
                0 0 0 1px rgba(0, 255, 153, 0.08),
                0 16px 38px rgba(0, 0, 0, 0.9);
            margin-bottom: 1.2rem;
        }

        .result-box {
            background: radial-gradient(circle at top left,#00ff99,#00bfa5);
            color: #00130d !important;
            padding: 1.6rem 2rem;
            border-radius: 18px;
            font-weight: 800;
            box-shadow:
                0 0 22px rgba(0, 255, 153, 0.7),
                0 18px 40px rgba(0, 0, 0, 0.9);
            border: 2px solid #b9f6ca;
            margin-top: 0.6rem;
        }

        .explanation-box {
            background: rgba(0,40,28,0.92);
            border-left: 4px solid #00e676;
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            margin-top: 0.7rem;
            box-shadow: 0 0 16px rgba(0,0,0,0.7);
        }

        .explanation-box h4 {
            color: #a5ffdc !important;
            font-weight: 800;
        }

        .explanation-box li {
            color: #e0f2f1 !important;
            font-weight: 500;
            margin: 0.25rem 0;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px !important;
            border: 1px solid rgba(0, 255, 170, 0.45) !important;
            overflow: hidden !important;
            box-shadow:
                0 0 0 1px rgba(0, 255, 170, 0.10),
                0 16px 40px rgba(0, 0, 0, 0.85) !important;
        }

        [data-testid="stMetricValue"] {
            color: #76ff03 !important;
            font-weight: 900 !important;
            font-size: 1.1rem !important;
        }

        [data-testid="stMetricLabel"] {
            color: #e0f2f1 !important;
            font-weight: 600 !important;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 0.8rem !important;
                padding-right: 0.8rem !important;
            }
            .app-header {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA LAYER
# =============================================================================

class DataService:
    """Quản lý dữ liệu đầu vào (climate risk + thông số công ty)."""

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_historical_data() -> pd.DataFrame:
        """Dữ liệu rủi ro khí hậu theo tuyến (12 tháng)."""
        climate_base = {
            "VN - EU": [0.28, 0.30, 0.35, 0.40, 0.52, 0.60, 0.67, 0.70, 0.75, 0.72, 0.60, 0.48],
            "VN - US": [0.33, 0.36, 0.40, 0.46, 0.55, 0.63, 0.72, 0.78, 0.80, 0.74, 0.62, 0.50],
            "VN - Singapore": [0.18, 0.20, 0.24, 0.27, 0.32, 0.36, 0.40, 0.43, 0.45, 0.42, 0.35, 0.30],
            "VN - China": [0.20, 0.23, 0.27, 0.31, 0.38, 0.42, 0.48, 0.50, 0.53, 0.49, 0.40, 0.34],
            "Domestic": [0.12, 0.13, 0.14, 0.16, 0.20, 0.22, 0.23, 0.25, 0.27, 0.24, 0.20, 0.18],
        }
        df = pd.DataFrame({"month": list(range(1, 13))})
        for route, values in climate_base.items():
            df[route] = values
        return df

    @staticmethod
    @st.cache_data
    def get_company_data() -> pd.DataFrame:
        """Thông số cơ bản của từng công ty bảo hiểm."""
        return (
            pd.DataFrame(
                {
                    "Company": ["Chubb", "PVI", "BaoViet", "BaoMinh", "MIC"],
                    "C1: Tỷ lệ phí": [0.42, 0.36, 0.40, 0.38, 0.34],
                    "C2: Thời gian xử lý": [12, 10, 15, 14, 11],
                    "C3: Tỷ lệ tổn thất": [0.07, 0.09, 0.11, 0.10, 0.08],
                    "C4: Hỗ trợ ICC": [9, 8, 7, 8, 7],
                    "C5: Chăm sóc KH": [9, 8, 7, 7, 6],
                }
            ).set_index("Company")
        )


# =============================================================================
# CORE ALGORITHMS
# =============================================================================

class FuzzyAHP:
    """Áp dụng Fuzzy AHP (tam giác) trên trọng số tiêu chí."""

    @staticmethod
    def apply(weights: pd.Series, uncertainty_pct: float) -> pd.Series:
        """
        Nhân trọng số với khoảng Low / Mid / High rồi giải mờ (defuzzify).
        uncertainty_pct: % dao động quanh trọng số gốc (ví dụ 15%).
        """
        factor = uncertainty_pct / 100.0
        w = weights.values

        low = np.maximum(w * (1 - factor), 1e-9)
        high = np.minimum(w * (1 + factor), 0.9999)

        # Tam giác Fuzzy: (low, mid, high) → centroid
        defuzzified = (low + w + high) / 3.0

        normalized = defuzzified / defuzzified.sum()
        return pd.Series(normalized, index=weights.index)


class MonteCarloSimulator:
    """Mô phỏng Monte Carlo cho rủi ro khí hậu (C6)."""

    @staticmethod
    @st.cache_data(ttl=600)
    def simulate(
        base_risk: float,
        sensitivity_map: Dict[str, float],
        n_simulations: int,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Trả về:
        - danh sách company
        - mean rủi ro C6
        - std rủi ro C6
        """
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())

        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.03, mu * 0.12)

        sims = rng.normal(loc=mu, scale=sigma, size=(n_simulations, len(companies)))
        sims = np.clip(sims, 0.0, 1.0)

        return companies, sims.mean(axis=0), sims.std(axis=0)


class TOPSISAnalyzer:
    """Phân tích TOPSIS để xếp hạng các phương án."""

    @staticmethod
    def analyze(
        data: pd.DataFrame,
        weights: pd.Series,
        cost_benefit: Dict[str, CriterionType],
    ) -> np.ndarray:
        """
        data: DataFrame [n_phương_án × n_tiêu_chí]
        weights: Series trọng số (đã chuẩn hóa)
        cost_benefit: map tiêu chí → COST/BENEFIT
        """
        # Ma trận quyết định
        M = data[list(weights.index)].values.astype(float)

        # Chuẩn hóa vector
        denom = np.sqrt((M ** 2).sum(axis=0))
        denom[denom == 0] = 1.0
        R = M / denom

        # Trọng số
        V = R * weights.values

        is_cost = np.array([cost_benefit[c] == CriterionType.COST for c in weights.index])

        ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
        ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))

        d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

        return d_minus / (d_plus + d_minus + 1e-12)


class RiskCalculator:
    """Tính VaR, CVaR và các chỉ báo rủi ro."""

    @staticmethod
    def calculate_var_cvar(
        loss_rates: np.ndarray,
        cargo_value: float,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Tính VaR/CVaR dựa trên phân phối loss_rates."""
        if len(loss_rates) == 0:
            return 0.0, 0.0

        losses = loss_rates * cargo_value
        var = float(np.percentile(losses, confidence * 100))

        tail_losses = losses[losses >= var]
        cvar = float(tail_losses.mean()) if len(tail_losses) > 0 else var

        return var, cvar


class Forecaster:
    """Dự báo rủi ro khí hậu 1 tháng tiếp theo (ARIMA hoặc heuristic)."""

    @staticmethod
    def forecast(
        historical: pd.DataFrame,
        route: str,
        current_month: int,
        use_arima: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if route not in historical.columns:
            # Fallback: chọn cột tuyến đầu tiên (trừ cột month)
            route = historical.columns[1]

        full_series = historical[route].values
        n_total = len(full_series)

        current_month = max(1, min(current_month, n_total))
        hist_series = full_series[:current_month]
        train_series = hist_series.copy()

        # Ưu tiên ARIMA nếu đủ dữ liệu
        if use_arima and ARIMA_AVAILABLE and len(train_series) >= 6:
            try:
                model = ARIMA(train_series, order=(1, 1, 1))
                fitted = model.fit()
                fc = fitted.forecast(1)
                fc_val = float(np.clip(fc[0], 0.0, 1.0))
                return hist_series, np.array([fc_val])
            except Exception:
                pass  # fallback xuống heuristic

        # Heuristic: dựa trên xu hướng gần nhất
        if len(train_series) >= 3:
            trend = (train_series[-1] - train_series[-3]) / 2.0
        elif len(train_series) >= 2:
            trend = train_series[-1] - train_series[-2]
        else:
            trend = 0.0

        next_val = np.clip(train_series[-1] + trend, 0.0, 1.0)
        return hist_series, np.array([next_val])

# ======================= END OF PART 1/6 =======================
# Tiếp theo: MultiPackageAnalyzer (PART 2)
# =============================================================================
# PART 2 — MULTI-PACKAGE ANALYZER
# =============================================================================

class MultiPackageAnalyzer:
    """Phân tích tất cả các phương án (Công ty × Gói ICC)."""

    def __init__(self):
        self.data_service = DataService()
        self.fuzzy_ahp = FuzzyAHP()
        self.mc_simulator = MonteCarloSimulator()
        self.topsis = TOPSISAnalyzer()
        self.risk_calc = RiskCalculator()
        self.forecaster = Forecaster()

    def run_analysis(self, params: AnalysisParams, historical: pd.DataFrame) -> AnalysisResult:
        # 1) Trọng số theo profile lựa chọn
        profile_weights = PRIORITY_PROFILES[params.priority]
        weights = pd.Series(profile_weights, index=CRITERIA)

        # 2) Fuzzy Weighting (nếu bật)
        if params.use_fuzzy:
            weights = self.fuzzy_ahp.apply(weights, params.fuzzy_uncertainty)

        # 3) Tải dữ liệu công ty
        company_data = self.data_service.get_company_data()

        # 4) Lấy base risk theo tháng & tuyến
        if params.month in historical["month"].values:
            base_risk = float(
                historical.loc[historical["month"] == params.month, params.route].iloc[0]
            )
        else:
            base_risk = 0.4  # fallback an toàn

        # 5) Monte Carlo mô phỏng rủi ro khí hậu
        if params.use_mc:
            companies, mc_mean, mc_std = self.mc_simulator.simulate(
                base_risk, SENSITIVITY_MAP, params.mc_runs
            )
            # reorder theo index công ty
            order = [companies.index(c) for c in company_data.index]
            mc_mean, mc_std = mc_mean[order], mc_std[order]
        else:
            mc_mean = mc_std = np.zeros(len(company_data))

        # ============================================================
        # 6) Tạo 15 phương án (5 công ty × 3 gói ICC)
        # ============================================================
        all_options = []

        for company in company_data.index:
            for icc_name, icc_data in ICC_PACKAGES.items():

                row = company_data.loc[company].copy()

                # Điều chỉnh phí theo gói ICC
                row["C1: Tỷ lệ phí"] *= icc_data["premium_multiplier"]

                # Điều chỉnh hỗ trợ ICC theo độ phủ bảo hiểm
                row["C4: Hỗ trợ ICC"] *= icc_data["coverage"]

                # Rủi ro khí hậu Monte Carlo
                idx = list(company_data.index).index(company)
                row["C6: Rủi ro khí hậu"] = mc_mean[idx]

                all_options.append({
                    "company": company,
                    "icc_package": icc_name,
                    "coverage": icc_data["coverage"],
                    "premium_rate": row["C1: Tỷ lệ phí"],
                    "estimated_cost": params.cargo_value * row["C1: Tỷ lệ phí"],

                    # tiêu chí
                    "C1: Tỷ lệ phí": row["C1: Tỷ lệ phí"],
                    "C2: Thời gian xử lý": row["C2: Thời gian xử lý"],
                    "C3: Tỷ lệ tổn thất": row["C3: Tỷ lệ tổn thất"],
                    "C4: Hỗ trợ ICC": row["C4: Hỗ trợ ICC"],
                    "C5: Chăm sóc KH": row["C5: Chăm sóc KH"],
                    "C6: Rủi ro khí hậu": row["C6: Rủi ro khí hậu"],

                    "C6_mean": row["C6: Rủi ro khí hậu"],
                    "C6_std": mc_std[idx],
                })

        data_adjusted = pd.DataFrame(all_options)

        # 7) Phụ phí nếu hàng > 50k
        if params.cargo_value > 50_000:
            data_adjusted["C1: Tỷ lệ phí"] *= 1.10
            data_adjusted["estimated_cost"] *= 1.10

        # ============================================================
        # 8) TOPSIS Ranking
        # ============================================================
        topsis_scores = self.topsis.analyze(
            data_adjusted[CRITERIA],
            weights,
            COST_BENEFIT_MAP
        )

        data_adjusted["score"] = topsis_scores
        data_adjusted = data_adjusted.sort_values("score", ascending=False).reset_index(drop=True)
        data_adjusted["rank"] = data_adjusted.index + 1

        # 9) Phân loại: Tiết kiệm / Cân bằng / An toàn
        def categorize(row):
            if row["icc_package"] == "ICC C":
                return "💰 Tiết kiệm"
            elif row["icc_package"] == "ICC B":
                return "⚖️ Cân bằng"
            return "🛡️ An toàn"

        data_adjusted["category"] = data_adjusted.apply(categorize, axis=1)

        # ============================================================
        # 10) Confidence Score
        # ============================================================
        eps = 1e-9
        cv_c6 = data_adjusted["C6_std"].values / (data_adjusted["C6_mean"].values + eps)
        conf = 1.0 / (1.0 + cv_c6)

        # scale 0.3 – 1.0
        conf = 0.3 + 0.7 * (conf - conf.min()) / (np.ptp(conf) + eps)
        data_adjusted["confidence"] = conf

        # ============================================================
        # 11) VaR / CVaR (nếu bật)
        # ============================================================
        var = cvar = None
        if params.use_var:
            var, cvar = self.risk_calc.calculate_var_cvar(
                data_adjusted["C6_mean"].values,
                params.cargo_value,
            )

        # ============================================================
        # 12) Forecast tháng tiếp theo
        # ============================================================
        hist_series, forecast = self.forecaster.forecast(
            historical,
            params.route,
            params.month,
            use_arima=params.use_arima
        )

        # ============================================================
        # Trả kết quả
        # ============================================================
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
# PART 3 — CHART FACTORY (BIỂU ĐỒ)
# =============================================================================

class ChartFactory:
    """Tạo các biểu đồ Premium Green cho RISKCAST."""

    # ==============================
    # GLOBAL THEME
    # ==============================
    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        fig.update_layout(
            template="plotly_dark",
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=22, color="#e6fff7", family="Inter"),
                x=0.5,
            ),
            font=dict(size=15, color="#e6fff7", family="Inter"),
            plot_bgcolor="#001016",
            paper_bgcolor="#000c11",
            margin=dict(l=70, r=40, t=80, b=70),
            legend=dict(
                bgcolor="rgba(0,0,0,0.35)",
                bordercolor="#00e676",
                borderwidth=1,
                font=dict(color="#e6fff7", size=14)
            )
        )

        fig.update_xaxes(
            showgrid=True,
            gridcolor="#00332b",
            tickfont=dict(size=14, color="#e6fff7"),
            zeroline=False
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#00332b",
            tickfont=dict(size=14, color="#e6fff7"),
            zeroline=False
        )

        return fig

    # ==============================
    # 1) WEIGHTS PIE (Fuzzy / Profile)
    # ==============================
    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        colors = [
            '#00e676', '#69f0ae', '#b9f6ca',
            '#00bfa5', '#1de9b6', '#64ffda'
        ]

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
                    hole=0.24,
                    marker=dict(colors=colors, line=dict(color="#00130d", width=2)),
                    pull=[0.04] * len(weights),
                    hovertemplate="<b>%{label}</b><br>Tỉ trọng: %{percent}<extra></extra>",
                )
            ]
        )

        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=20, color="#a5ffdc"),
                x=0.5
            ),
            showlegend=True,
            legend=dict(
                title="<b>Tiêu chí</b>",
                font=dict(size=13, color="#e6fff7")
            ),
            paper_bgcolor="#001016",
            plot_bgcolor="#001016",
            margin=dict(l=0, r=0, t=60, b=0),
            height=430,
        )

        return fig

    # ==============================
    # 2) COST-BENEFIT SCATTER
    # ==============================
    @staticmethod
    def create_cost_benefit_scatter(results: pd.DataFrame) -> go.Figure:
        color_map = {
            "ICC A": "#ff6b6b",   # đỏ
            "ICC B": "#ffd93d",   # vàng
            "ICC C": "#6bcf7f",   # xanh
        }

        fig = go.Figure()

        for icc in ["ICC C", "ICC B", "ICC A"]:
            df_icc = results[results["icc_package"] == icc]

            fig.add_trace(
                go.Scatter(
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
                        "<b>%{text}</b><br>"
                        f"Gói: {icc}<br>"
                        "Chi phí: $%{x:,.0f}<br>"
                        "Điểm: %{y:.3f}<extra></extra>"
                    )
                )
            )

        fig.update_xaxes(title="<b>Chi phí ước tính ($)</b>")
        fig.update_yaxes(title="<b>Điểm TOPSIS</b>", range=[0, 1])

        return ChartFactory._apply_theme(fig, "💰 Chi phí vs Chất lượng (Cost-Benefit Analysis)")

    # ==============================
    # 3) CATEGORY COMPARISON (Tiết kiệm – Cân bằng – An toàn)
    # ==============================
    @staticmethod
    def create_category_comparison(results: pd.DataFrame) -> go.Figure:
        categories = ["💰 Tiết kiệm", "⚖️ Cân bằng", "🛡️ An toàn"]
        avg_scores, avg_costs = [], []

        for cat in categories:
            df = results[results["category"] == cat]
            avg_scores.append(df["score"].mean() if len(df) else 0)
            avg_costs.append(df["estimated_cost"].mean() if len(df) else 0)

        fig = go.Figure()

        # Cột: điểm TOPSIS
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

        # Line: chi phí TB
        fig.add_trace(
            go.Scatter(
                name="Chi phí trung bình",
                x=categories,
                y=avg_costs,
                mode="lines+markers",
                marker=dict(size=12, color="#ffeb3b"),
                line=dict(width=3, color="#ffeb3b"),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Chi phí TB: $%{y:,.0f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=dict(
                text="<b>📊 So sánh 3 loại phương án</b>",
                font=dict(size=22, color="#e6fff7"),
                x=0.5
            ),
            yaxis=dict(
                title="<b>Điểm TOPSIS</b>",
                range=[0, 1],
                titlefont=dict(color="#00e676"),
                tickfont=dict(color="#00e676"),
            ),
            yaxis2=dict(
                title="<b>Chi phí ($)</b>",
                overlaying="y",
                side="right",
                titlefont=dict(color="#ffeb3b"),
                tickfont=dict(color="#ffeb3b"),
            ),
            paper_bgcolor="#000c11",
            plot_bgcolor="#001016",
            font=dict(color="#e6fff7"),
            legend=dict(
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="#00e676",
                borderwidth=1,
            ),
        )

        return fig

    # ==============================
    # 4) TOP 5 BAR CHART
    # ==============================
    @staticmethod
    def create_top_recommendations_bar(results: pd.DataFrame) -> go.Figure:
        df = results.head(5).copy()
        df["label"] = df["company"] + " - " + df["icc_package"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df["score"],
                    y=df["label"],
                    orientation="h",
                    text=[f"{v:.3f}" for v in df["score"]],
                    textposition="outside",
                    marker=dict(
                        color=df["score"],
                        colorscale=[[0, "#69f0ae"], [0.5, "#00e676"], [1, "#00c853"]],
                        line=dict(color="#00130d", width=1),
                    ),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Score: %{x:.3f}<br>"
                        "Chi phí: $%{customdata:,.0f}<extra></extra>"
                    ),
                    customdata=df["estimated_cost"],
                )
            ]
        )

        fig.update_xaxes(title="<b>Điểm TOPSIS</b>", range=[0, 1])
        fig.update_yaxes(title="<b>Phương án</b>")

        return ChartFactory._apply_theme(fig, "🏆 Top 5 Phương án Tốt nhất")

    # ==============================
    # 5) FORECAST CHART (Fix jump 2-4-6)
    # ==============================
    @staticmethod
    def create_forecast_chart(historical: np.ndarray, forecast: np.ndarray,
                              route: str, selected_month: int) -> go.Figure:

        hist_len = len(historical)
        months_hist = list(range(1, hist_len + 1))

        next_month = selected_month % 12 + 1
        months_fc = [next_month]

        fig = go.Figure()

        # Lịch sử
        fig.add_trace(
            go.Scatter(
                x=months_hist,
                y=historical,
                mode="lines+markers",
                name="📈 Lịch sử",
                line=dict(color="#00e676", width=3),
                marker=dict(size=9),
                hovertemplate="Tháng %{x}<br>Rủi ro: %{y:.1%}<extra></extra>",
            )
        )

        # Dự báo
        fig.add_trace(
            go.Scatter(
                x=months_fc,
                y=forecast,
                mode="lines+markers",
                name="🔮 Dự báo",
                line=dict(color="#ffeb3b", width=3, dash="dash"),
                marker=dict(size=11, symbol="diamond"),
                hovertemplate="Tháng %{x}<br>Dự báo: %{y:.1%}<extra></extra>",
            )
        )

        fig = ChartFactory._apply_theme(fig, f"Dự báo rủi ro khí hậu — {route}")

        # Fix: tháng 1 → 12 (không bị nhảy 2-4-6 nữa)
        fig.update_xaxes(
            title="<b>Tháng</b>",
            tickmode="array",
            tickvals=list(range(1, 13)),
            ticktext=list(range(1, 13)),
            range=[1, 12]
        )

        max_val = max(float(max(historical)), float(max(forecast)))
        fig.update_yaxes(
            title="<b>Mức rủi ro (0–1)</b>",
            range=[0, max(1.0, max_val * 1.15)],
            tickformat=".0%"
        )

        return fig
# =============================================================================
# STREAMLIT UI & MAIN (PART 4)
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
            fuzzy_uncertainty = st.slider(
                "Mức bất định Fuzzy (%)",
                0, 50, 15
            ) if use_fuzzy else 15

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
                fuzzy_uncertainty=fuzzy_uncertainty
            )

    def _find_safest_option(self, results: pd.DataFrame) -> pd.Series:
        """
        Chọn phương án 'an toàn nhất':
        - Ưu tiên các gói ICC A (bảo hiểm rộng nhất)
        - Trong ICC A: chọn phương án có độ tin cậy cao nhất
        - Nếu không có ICC A (edge case) → chọn phương án có confidence cao nhất toàn bảng
        """
        df_icc_a = results[results["icc_package"] == "ICC A"]
        if len(df_icc_a) > 0:
            return df_icc_a.loc[df_icc_a["confidence"].idxmax()]
        return results.loc[results["confidence"].idxmax()]

    def _render_reason_table_for_top(self, result: AnalysisResult, top_row: pd.Series):
        """
        Bảng giải thích chi tiết theo từng tiêu chí cho phương án được khuyến nghị.
        Giống style bảng giải thích cũ: Tiêu chí – Loại (Chi phí/Lợi ích) – Trọng số – Giá trị.
        """
        rows = []
        for crit in CRITERIA:
            if crit in result.data_adjusted.columns and crit in result.weights.index:
                crit_type = "Chi phí (càng thấp càng tốt)" \
                    if COST_BENEFIT_MAP[crit] == CriterionType.COST else \
                    "Lợi ích (càng cao càng tốt)"

                val = top_row[crit]
                if isinstance(val, (int, float)):
                    # Hiển thị đẹp hơn cho tỷ lệ
                    if "Tỷ lệ" in crit or "rủi ro" in crit.lower():
                        display_val = f"{val:.3f}"
                    else:
                        display_val = f"{val:.2f}"
                else:
                    display_val = str(val)

                rows.append({
                    "Tiêu chí": crit,
                    "Loại tiêu chí": crit_type,
                    "Trọng số": f"{result.weights[crit]:.0%}",
                    "Giá trị của phương án": display_val
                })

        if rows:
            df_reason = pd.DataFrame(rows)
            st.markdown("#### 🔍 Bảng giải thích theo từng tiêu chí (phương án được chọn)")
            st.dataframe(df_reason, hide_index=True, use_container_width=True)

    def display_results(self, result: AnalysisResult, params: AnalysisParams):
        st.success("✅ Đã phân tích xong 15 phương án (5 công ty × 3 gói ICC)")

        # Top recommendation theo mục tiêu của user
        top = result.results.iloc[0]

        # Phương án an toàn nhất (ưu tiên ICC A + độ tin cậy)
        safest = self._find_safest_option(result.results)

        st.markdown(
            f"""
            <div class="result-box">
                🏆 <b>GỢI Ý TỐT NHẤT CHO MỤC TIÊU: {params.priority}</b><br><br>
                <span style="font-size:1.6rem;">{top['company']} - {top['icc_package']}</span><br><br>
                💰 Chi phí: <b>${top['estimated_cost']:,.0f}</b> ({top['premium_rate']:.2%} giá trị hàng)<br>
                📊 Điểm TOPSIS: <b>{top['score']:.3f}</b> · 
                🎯 Độ tin cậy: <b>{top['confidence']:.2f}</b><br>
                📦 Loại phương án: <b>{top['category']}</b><br><br>

                🛡️ <b>PHƯƠNG ÁN AN TOÀN NHẤT (ưu tiên phạm vi bảo hiểm & độ tin cậy)</b><br>
                👉 <b>{safest['company']} - {safest['icc_package']}</b><br>
                Chi phí: <b>${safest['estimated_cost']:,.0f}</b> · 
                Điểm: <b>{safest['score']:.3f}</b> · 
                Tin cậy: <b>{safest['confidence']:.2f}</b><br>
                <span style="font-size:0.9rem;opacity:0.9;">
                    (Hệ thống chọn phương án có gói ICC A và độ tin cậy cao nhất. 
                    Nếu không có ICC A, chọn phương án có độ tin cậy cao nhất toàn bộ.)
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Bảng so sánh 15 phương án
        st.markdown("---")
        st.subheader("📋 Bảng so sánh 15 phương án")

        df_display = result.results[[
            "rank", "company", "icc_package", "category",
            "estimated_cost", "score", "confidence"
        ]].copy()
        df_display.columns = ["Hạng", "Công ty", "Gói ICC", "Loại", "Chi phí", "Điểm", "Tin cậy"]
        df_display["Chi phí"] = df_display["Chi phí"].apply(lambda x: f"${x:,.0f}")
        df_display = df_display.set_index("Hạng")

        st.dataframe(df_display, use_container_width=True)

        # Giải thích tổng quan
        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>💡 Giải thích kết quả</h4>
                <ul>
                    <li><b>{top['company']} - {top['icc_package']}</b> có điểm tổng hợp cao nhất 
                        theo trọng số mục tiêu <b>{params.priority}</b>.</li>
                    <li>Chi phí <b>${top['estimated_cost']:,.0f}</b> phản ánh tỷ lệ phí bảo hiểm 
                        nhân với giá trị lô hàng, có điều chỉnh phụ phí nếu lô hàng lớn.</li>
                    <li>Độ tin cậy <b>{top['confidence']:.2f}</b> dựa trên biến động rủi ro khí hậu 
                        (Monte Carlo) và độ ổn định của các tiêu chí.</li>
                    <li>Hệ thống đã phân tích <b>15 phương án</b> (5 công ty × 3 gói ICC) để đưa ra 
                        gợi ý tốt nhất và phương án an toàn nhất.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Bảng giải thích chi tiết theo tiêu chí cho phương án top
        self._render_reason_table_for_top(result, top)

        # So sánh Top 3
        st.markdown(
            """
            <div class="explanation-box">
                <h4>🥇 So sánh Top 3 phương án</h4>
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
                    st.caption(f"Điểm: {row['score']:.3f} · {row['category']}")

        # Biểu đồ
        st.markdown("---")
        st.subheader("📊 Biểu đồ phân tích")

        col1, col2 = st.columns(2)

        with col1:
            fig_scatter = self.chart_factory.create_cost_benefit_scatter(result.results)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            fig_category = self.chart_factory.create_category_comparison(result.results)
            st.plotly_chart(fig_category, use_container_width=True)

        fig_top = self.chart_factory.create_top_recommendations_bar(result.results)
        st.plotly_chart(fig_top, use_container_width=True)

        # Trọng số & VaR/CVaR
        col1, col2 = st.columns(2)
        with col1:
            fig_weights = self.chart_factory.create_weights_pie(
                result.weights,
                f"Trọng số áp dụng ({params.priority})"
            )
            st.plotly_chart(fig_weights, use_container_width=True)

        with col2:
            if result.var is not None and result.cvar is not None:
                st.metric("💰 VaR 95%", f"${result.var:,.0f}")
                st.metric("🛡️ CVaR 95%", f"${result.cvar:,.0f}")
                risk_pct = (result.var / params.cargo_value) * 100 if params.cargo_value > 0 else 0.0
                st.metric("📊 Rủi ro / Giá trị lô hàng", f"{risk_pct:.1f}%")

        # Forecast
        st.markdown("---")
        fig_forecast = self.chart_factory.create_forecast_chart(
            result.historical, result.forecast, params.route, params.month
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Export
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
            pdf_data = self.report_gen.generate_pdf(
                result.results,
                params,
                result.var,
                result.cvar
            )
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

        # Header
        st.markdown(
            """
            <div class="app-header">
                <div class="app-header-left">
                    <div class="app-logo-circle">RC</div>
                    <div>
                        <div class="app-header-title">RISKCAST v5.3 — MULTI-PACKAGE ANALYSIS</div>
                        <div class="app-header-subtitle">
                            15 Phương án (5 Công ty × 3 Gói ICC) · Profile-Based Recommendation · Smart Ranking · Cost-Benefit Analysis
                        </div>
                    </div>
                </div>
                <div class="app-header-badge">
                    <span>🎯 Smart Recommendation</span>
                    <span>·</span>
                    <span>15 Phương án</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        historical = DataService.load_historical_data()
        params = self.render_sidebar()

        # Hiển thị profile trọng số
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader(f"📌 Đã chọn mục tiêu: {params.priority}")

        profile_weights = PRIORITY_PROFILES[params.priority]
        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>Trọng số tự động được điều chỉnh:</h4>
                <ul>
                    <li>C1 (Chi phí): <b>{profile_weights['C1: Tỷ lệ phí']:.0%}</b></li>
                    <li>C2 (Thời gian): <b>{profile_weights['C2: Thời gian xử lý']:.0%}</b></li>
                    <li>C3 (Tổn thất): <b>{profile_weights['C3: Tỷ lệ tổn thất']:.0%}</b></li>
                    <li>C4 (Hỗ trợ ICC): <b>{profile_weights['C4: Hỗ trợ ICC']:.0%}</b></li>
                    <li>C5 (Chăm sóc KH): <b>{profile_weights['C5: Chăm sóc KH']:.0%}</b></li>
                    <li>C6 (Khí hậu): <b>{profile_weights['C6: Rủi ro khí hậu']:.0%}</b></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        if st.button("🚀 PHÂN TÍCH 15 PHƯƠNG ÁN", type="primary", use_container_width=True):
            with st.spinner("🔄 Đang phân tích tất cả phương án..."):
                try:
                    result = self.analyzer.run_analysis(params, historical)
                    self.display_results(result, params)
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
                    st.exception(e)


# =============================================================================
# MAIN
# =============================================================================

def main():
    app = StreamlitUI()
    app.run()


if __name__ == "__main__":
    main()
# =============================================================================
# PART 5 — REPORT GENERATOR (Excel + PDF)
# =============================================================================

from io import BytesIO
from fpdf import FPDF
import pandas as pd

class ReportGenerator:
    """Xuất Excel + PDF cho RISKCAST."""

    # =========================
    # 1) EXPORT EXCEL
    # =========================
    def generate_excel(self, results: pd.DataFrame, weights: pd.Series):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            results.to_excel(writer, index=False, sheet_name="Results")
            weights.to_frame("Weight").to_excel(writer, sheet_name="Weights")

            workbook = writer.book
            fmt = workbook.add_format({"num_format": "0.000"})
            ws = writer.sheets["Results"]
            ws.set_column("A:Z", 18, fmt)

        return output.getvalue()

    # =========================
    # 2) EXPORT PDF
    # =========================
    def generate_pdf(self, results: pd.DataFrame, params, var, cvar):
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST Report v5.3", ln=True)

            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(
                0, 8,
                f"Tuyến: {params.route}\n"
                f"Giá trị hàng: ${params.cargo_value:,.0f}\n"
                f"Mục tiêu: {params.priority}\n"
            )

            # ====== TOP 5 ======
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 10, "Top 5 Phương Án", ln=True)

            pdf.set_font("Arial", "", 11)
            top5 = results.head(5)
            for _, row in top5.iterrows():
                pdf.multi_cell(
                    0, 7,
                    f"- {row['company']} - {row['icc_package']}:  "
                    f"Score {row['score']:.3f},  "
                    f"Cost ${row['estimated_cost']:,.0f},  "
                    f"Tin cậy {row['confidence']:.2f}"
                )

            # ====== VAR / CVAR ======
            if var is not None and cvar is not None:
                pdf.ln(5)
                pdf.set_font("Arial", "B", 13)
                pdf.cell(0, 10, "Rủi ro tài chính (VaR/CVaR)", ln=True)

                pdf.set_font("Arial", "", 11)
                pdf.multi_cell(
                    0, 7,
                    f"VaR 95%: ${var:,.0f}\n"
                    f"CVaR 95%: ${cvar:,.0f}\n"
                )

            buffer = BytesIO()
            pdf.output(buffer)
            return buffer.getvalue()

        except Exception as e:
            print("PDF ERROR:", e)
            return None
