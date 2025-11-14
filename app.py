# =============================================================================
# RISKCAST v5.3 — ENTERPRISE EDITION (Multi-Package Analysis)
# ESG Logistics Risk Assessment Dashboard
#
# Author: Bùi Xuân Hoàng (original idea)
# Refactor + Multi-Package + Full Explanations + Enterprise UX: Kai assistant
#
# Nổi bật trong v5.3 Enterprise:
#   - Profile-Based Recommendation (3 mục tiêu: Tiết kiệm / Cân bằng / An toàn)
#   - Multi-Package Analysis (5 công ty × 3 gói ICC = 15 phương án)
#   - Smart Ranking Table với badges
#   - Cost-Benefit Scatter Plot
#   - Trade-off Analysis
#   - Fuzzy AHP Enterprise module (heatmap + radar-style line)
#   - Forecast chart nền tối + line neon
#   - Full Explanations cho NCKH
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
    """Kết quả phân tích."""
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
    "C6: Rủi ro khí hậu"
]

# Profile weights - Trọng số theo mục tiêu
PRIORITY_PROFILES = {
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

# ICC Package definitions
ICC_PACKAGES = {
    "ICC A": {
        "coverage": 1.0,              # Bảo vệ toàn diện 100%
        "premium_multiplier": 1.5,    # Phí cao nhất (+50%)
        "description": "Bảo vệ toàn diện mọi rủi ro trừ điều khoản loại trừ (All Risks)",
    },
    "ICC B": {
        "coverage": 0.75,             # Bảo vệ vừa phải 75%
        "premium_multiplier": 1.0,    # Phí trung bình (baseline)
        "description": "Bảo vệ các rủi ro chính (hỏa hoạn, va chạm, chìm đắm, Named Perils)",
    },
    "ICC C": {
        "coverage": 0.5,              # Bảo vệ cơ bản 50%
        "premium_multiplier": 0.65,   # Phí thấp nhất (-35%)
        "description": "Bảo vệ cơ bản (chỉ các rủi ro lớn như chìm, cháy, va chạm nghiêm trọng)",
    },
}

# Map loại tiêu chí
COST_BENEFIT_MAP = {
    "C1: Tỷ lệ phí": CriterionType.COST,          # Chi phí - càng thấp càng tốt
    "C2: Thời gian xử lý": CriterionType.COST,    # Chi phí - càng nhanh càng tốt
    "C3: Tỷ lệ tổn thất": CriterionType.COST,     # Chi phí - càng thấp càng tốt
    "C4: Hỗ trợ ICC": CriterionType.BENEFIT,      # Lợi ích - càng cao càng tốt
    "C5: Chăm sóc KH": CriterionType.BENEFIT,     # Lợi ích - càng cao càng tốt
    "C6: Rủi ro khí hậu": CriterionType.COST,     # Chi phí - càng thấp càng tốt
}

# Độ nhạy rủi ro khí hậu theo công ty
SENSITIVITY_MAP = {
    "Chubb": 0.95,
    "PVI": 1.05,
    "BaoViet": 1.00,
    "BaoMinh": 1.02,
    "MIC": 1.03,
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

    h1 { font-size: 2.8rem !important; font-weight: 900 !important; letter-spacing: 0.03em; }
    h2 { font-size: 2.1rem !important; font-weight: 800 !important; }
    h3 { font-size: 1.5rem !important; font-weight: 700 !important; }

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.1rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(0, 255, 153, 0.14), rgba(0, 0, 0, 0.88));
        border: 1px solid rgba(0, 255, 153, 0.45);
        box-shadow: 0 0 0 1px rgba(0, 255, 153, 0.12), 0 18px 45px rgba(0, 0, 0, 0.85);
        margin-bottom: 1.2rem;
        gap: 1.5rem;
    }

    .app-header-left { display: flex; align-items: center; gap: 0.9rem; }

    .app-logo-circle {
        width: 64px; height: 64px; border-radius: 18px;
        background: radial-gradient(circle at 30% 30%, #b9f6ca 0%, #00c853 38%, #00381f 100%);
        display: flex; align-items: center; justify-content: center;
        font-weight: 900; font-size: 1.4rem; color: #00130d;
        box-shadow: 0 0 14px rgba(0, 255, 153, 0.65), 0 0 36px rgba(0, 0, 0, 0.75);
        border: 2px solid #e8f5e9;
    }

    .app-header-title {
        font-size: 1.5rem; font-weight: 800;
        background: linear-gradient(90deg, #e8fffb, #b9f6ca, #e8fffb);
        -webkit-background-clip: text; color: transparent;
        letter-spacing: 0.05em; text-transform: uppercase;
    }

    .app-header-subtitle { font-size: 0.9rem; color: #ccffec; opacity: 0.9; }

    .app-header-badge {
        font-size: 0.86rem; font-weight: 600; padding: 0.55rem 0.9rem;
        border-radius: 999px; background: radial-gradient(circle at 0 0, #00e676, #00bfa5);
        color: #00130d; display: flex; align-items: center; gap: 0.35rem;
        white-space: nowrap; box-shadow: 0 0 14px rgba(0, 255, 153, 0.65), 0 0 22px rgba(0, 0, 0, 0.7);
    }

    section[data-testid="stSidebar"] {
        background: radial-gradient(circle at 0 0, #003322 0%, #000f0a 40%, #000805 100%) !important;
        border-right: 1px solid rgba(0, 230, 118, 0.55);
        box-shadow: 8px 0 22px rgba(0, 0, 0, 0.85);
    }

    section[data-testid="stSidebar"] > div { padding-top: 1.1rem; }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #a5ffdc !important; font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] label {
        color: #e0f2f1 !important; font-weight: 600 !important; font-size: 0.92rem !important;
    }

    .stButton > button {
        background: linear-gradient(120deg, #00ff99, #00e676, #00bfa5) !important;
        color: #00130d !important; font-weight: 800 !important;
        border-radius: 999px !important; border: none !important;
        padding: 0.65rem 1.9rem !important;
        box-shadow: 0 0 14px rgba(0, 255, 153, 0.7), 0 10px 22px rgba(0, 0, 0, 0.85) !important;
        transition: all 0.12s ease-out; font-size: 0.98rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 0 20px rgba(0, 255, 153, 0.95), 0 14px 30px rgba(0, 0, 0, 0.9) !important;
    }

    .premium-card {
        background: radial-gradient(circle at top left, rgba(0, 255, 153, 0.10), rgba(0, 0, 0, 0.95));
        border-radius: 16px; padding: 1.1rem 1.2rem;
        border: 1px solid rgba(0, 255, 153, 0.45);
        box-shadow: 0 0 0 1px rgba(0, 255, 153, 0.08), 0 16px 38px rgba(0, 0, 0, 0.9);
        margin-bottom: 1.2rem;
    }

    .result-box {
        background: radial-gradient(circle at top left,#00ff99,#00bfa5);
        color: #00130d !important; padding: 1.6rem 2rem; border-radius: 18px;
        font-weight: 800; box-shadow: 0 0 22px rgba(0, 255, 153, 0.7), 0 18px 40px rgba(0, 0, 0, 0.9);
        border: 2px solid #b9f6ca; margin-top: 0.6rem;
    }

    .explanation-box {
        background: rgba(0,40,28,0.92); border-left: 4px solid #00e676;
        padding: 1.2rem 1.5rem; border-radius: 12px; margin-top: 0.7rem;
        box-shadow: 0 0 16px rgba(0,0,0,0.7);
    }

    .explanation-box h4 { color: #a5ffdc !important; font-weight: 800; }
    .explanation-box li { color: #e0f2f1 !important; font-weight: 500; margin: 0.25rem 0; }

    div[data-testid="stDataFrame"] {
        border-radius: 14px !important; border: 1px solid rgba(0, 255, 170, 0.45) !important;
        overflow: hidden !important;
        box-shadow: 0 0 0 1px rgba(0, 255, 170, 0.10), 0 16px 40px rgba(0, 0, 0, 0.85) !important;
    }

    [data-testid="stMetricValue"] {
        color: #76ff03 !important; font-weight: 900 !important; font-size: 1.1rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #e0f2f1 !important; font-weight: 600 !important;
    }

    @media (max-width: 900px) {
        .block-container { padding-left: 0.8rem !important; padding-right: 0.8rem !important; }
        .app-header { flex-direction: column; align-items: flex-start; }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA LAYER — INDUSTRY STANDARD LEVEL 1
# =============================================================================

class DataService:
    """Quản lý dữ liệu đầu vào (lịch sử khí hậu, dữ liệu công ty)."""

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_historical_data() -> pd.DataFrame:
        """
        Dữ liệu rủi ro khí hậu theo tuyến (12 tháng), chuẩn hóa 0–1.
        Mô phỏng theo mức độ bão, sóng, mưa, chậm trễ năm 2023.
        """
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
        """
        Thông số cơ bản của từng công ty bảo hiểm.

        C1: Tỷ lệ phí bảo hiểm (premium rate, %, dạng thập phân 0.34–0.42)
        C2: Thời gian xử lý claim (ngày)
        C3: Tỷ lệ tổn thất (loss ratio, %, dạng thập phân)
        C4: Hỗ trợ ICC (điểm 1–10)
        C5: Chăm sóc khách hàng (điểm 1–10)
        """
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
# CORE ALGORITHMS (P1) — FUZZY AHP + MONTE CARLO
# =============================================================================

class FuzzyAHP:
    """
    Áp dụng Fuzzy AHP (tam giác) trên trọng số.

    - Chuyển trọng số crisp (w) thành tam giác (low, mid, high)
    - Defuzzify bằng centroid: (low + mid + high) / 3
    - Chuẩn hóa lại để tổng = 1
    """

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
    """
    Mô phỏng Monte Carlo cho rủi ro khí hậu (C6).
    """

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
        sigma = np.maximum(0.03, mu * 0.12)  # 12% coefficient of variation
        sims = rng.normal(loc=mu, scale=sigma, size=(n_simulations, len(companies)))
        sims = np.clip(sims, 0.0, 1.0)
        return companies, sims.mean(axis=0), sims.std(axis=0)
# =============================================================================
# CORE ALGORITHMS (P2) — TOPSIS + RISK + FORECAST
# =============================================================================

class TOPSISAnalyzer:
    """
    Phân tích TOPSIS (Technique for Order Preference by Similarity to Ideal Solution).

    Bước:
    1. Chuẩn hóa ma trận quyết định (vector normalization)
    2. Nhân trọng số → ma trận V
    3. Xác định ideal best / ideal worst
       - Cost: best = min, worst = max
       - Benefit: best = max, worst = min
    4. Tính khoảng cách d+ (tới best) và d− (tới worst)
    5. Điểm TOPSIS C = d− / (d+ + d−). Càng gần 1 càng tốt.
    """

    @staticmethod
    def analyze(
        data: pd.DataFrame,
        weights: pd.Series,
        cost_benefit: Dict[str, CriterionType],
    ) -> np.ndarray:
        # Ma trận M (n x m)
        M = data[list(weights.index)].values.astype(float)

        # 1. Chuẩn hóa vector
        denom = np.sqrt((M ** 2).sum(axis=0))
        denom[denom == 0] = 1.0
        R = M / denom

        # 2. Nhân trọng số
        V = R * weights.values

        # 3. Ideal best / worst
        is_cost = np.array(
            [cost_benefit[c] == CriterionType.COST for c in weights.index]
        )
        ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
        ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))

        # 4. Khoảng cách
        d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

        # 5. Điểm TOPSIS
        return d_minus / (d_plus + d_minus + 1e-12)


class RiskCalculator:
    """
    Tính VaR, CVaR & độ tin cậy.

    - VaR 95%: tổn thất tối đa ở mức tin cậy 95%
    - CVaR 95%: tổn thất trung bình trong 5% trường hợp xấu nhất
    """

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
    def calculate_confidence(
        results: pd.DataFrame,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Tính độ tin cậy dựa trên:
        - Độ biến động tương đối (CV) của C6
        - Độ biến động tương đối của các tiêu chí khác
        """
        eps = 1e-9

        # CV của C6
        cv_c6 = results["C6_std"].values / (results["C6_mean"].values + eps)
        conf_c6 = 1.0 / (1.0 + cv_c6)
        conf_c6 = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + eps)

        # CV của toàn bộ tiêu chí (hàng theo phương án)
        crit_cv = data.std(axis=1).values / (data.mean(axis=1).values + eps)
        conf_crit = 1.0 / (1.0 + crit_cv)
        conf_crit = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (np.ptp(crit_cv) + eps)

        # Gộp lại bằng trung bình hình học
        return np.sqrt(conf_c6 * conf_crit)


class Forecaster:
    """
    Dự báo rủi ro khí hậu 1 tháng tiếp theo.

    - Ưu tiên ARIMA(1,1,1) nếu đủ data + đã cài statsmodels.
    - Nếu lỗi / thiếu lib / data ít → fallback linear trend.
    """

    @staticmethod
    def forecast(
        historical: pd.DataFrame,
        route: str,
        current_month: int,
        use_arima: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Nếu route không tồn tại (phòng hờ) → lấy cột thứ 2
        if route not in historical.columns:
            route = historical.columns[1]

        full_series = historical[route].values
        n_total = len(full_series)

        # Bound tháng
        if current_month < 1:
            current_month = 1
        if current_month > n_total:
            current_month = n_total

        hist_series = full_series[:current_month]
        train_series = hist_series.copy()

        # Thử ARIMA khi:
        # - được bật
        # - có lib
        # - có ít nhất 6 điểm dữ liệu
        if use_arima and ARIMA_AVAILABLE and len(train_series) >= 6:
            try:
                model = ARIMA(train_series, order=(1, 1, 1))
                fitted = model.fit()
                fc = fitted.forecast(1)
                fc_val = float(np.clip(fc[0], 0.0, 1.0))
                return hist_series, np.array([fc_val])
            except Exception:
                pass  # lỗi thì fallback xuống dưới

        # Fallback: Linear trend
        if len(train_series) >= 3:
            trend = (train_series[-1] - train_series[-3]) / 2.0
        elif len(train_series) >= 2:
            trend = train_series[-1] - train_series[-2]
        else:
            trend = 0.0

        next_val = np.clip(train_series[-1] + trend, 0.0, 1.0)
        return hist_series, np.array([next_val])


# =============================================================================
# FUZZY VISUAL UTILITIES (PREMIUM GREEN)
# =============================================================================

def build_fuzzy_table(weights: pd.Series, fuzzy_pct: float) -> pd.DataFrame:
    """
    Tạo bảng Fuzzy: Low – Mid – High – Centroid cho từng tiêu chí.

    - Low  = w × (1 - factor)
    - Mid  = w
    - High = w × (1 + factor)
    - Centroid = (Low + Mid + High) / 3
    """
    rows = []
    factor = fuzzy_pct / 100.0
    for crit in weights.index:
        w = float(weights[crit])
        low = max(w * (1 - factor), 0.0)
        high = min(w * (1 + factor), 1.0)
        centroid = (low + w + high) / 3.0
        rows.append(
            [crit, round(low, 4), round(w, 4), round(high, 4), round(centroid, 4)]
        )

    df = pd.DataFrame(
        rows, columns=["Tiêu chí", "Low", "Mid", "High", "Centroid"]
    )
    return df


def most_uncertain_criterion(
    weights: pd.Series, fuzzy_pct: float
) -> Tuple[str, Dict[str, float]]:
    """
    Xác định tiêu chí có độ dao động mạnh nhất (High - Low lớn nhất).
    """
    factor = fuzzy_pct / 100.0
    diff_map: Dict[str, float] = {}
    for crit in weights.index:
        w = float(weights[crit])
        low = w * (1 - factor)
        high = w * (1 + factor)
        diff_map[crit] = float(high - low)

    most_unc = max(diff_map, key=diff_map.get)
    return most_unc, diff_map


def fuzzy_heatmap_premium(diff_map: Dict[str, float]) -> go.Figure:
    """
    Heatmap Premium Green thể hiện mức dao động Fuzzy (High - Low).
    """
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
            [1.0, "#00FFAA"],
        ],
    )

    fig.update_layout(
        title=dict(
            text="<b>🌿 Heatmap mức dao động Fuzzy (Premium Green)</b>",
            font=dict(size=22, color="#CCFFE6"),
            x=0.5,
        ),
        paper_bgcolor="#001a12",
        plot_bgcolor="#001a12",
        margin=dict(l=40, r=40, t=80, b=40),
        coloraxis_colorbar=dict(
            title="Dao động",
            tickfont=dict(color="#CCFFE6"),
        ),
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(showticklabels=False)
    return fig


def fuzzy_chart_premium(weights: pd.Series, fuzzy_pct: float) -> go.Figure:
    """
    Biểu đồ Fuzzy Premium: Low / Mid / High cho từng tiêu chí.
    """
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

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=low_vals,
            mode="lines+markers",
            name="Low",
            line=dict(width=2, color="#004d40", dash="dot"),
            marker=dict(size=8),
            hovertemplate="Tiêu chí: %{x}<br>Low: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=mid_vals,
            mode="lines+markers",
            name="Mid (gốc)",
            line=dict(width=3, color="#00e676"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="Tiêu chí: %{x}<br>Mid: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=high_vals,
            mode="lines+markers",
            name="High",
            line=dict(width=2, color="#69f0ae", dash="dash"),
            marker=dict(size=8),
            hovertemplate="Tiêu chí: %{x}<br>High: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>🌿 Fuzzy AHP — Low / Mid / High (±{fuzzy_pct:.0f}%)</b>",
            font=dict(size=22, color="#e6fff7"),
            x=0.5,
        ),
        paper_bgcolor="#001a12",
        plot_bgcolor="#001a12",
        legend=dict(
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="#00e676",
            borderwidth=1,
        ),
        margin=dict(l=40, r=40, t=80, b=80),
        font=dict(size=13, color="#e6fff7"),
    )
    fig.update_xaxes(showgrid=False, tickangle=-20)
    fig.update_yaxes(
        title="Trọng số",
        range=[0, max(0.4, max(high_vals) * 1.15)],
        showgrid=True,
        gridcolor="#004d40",
    )
    return fig
# =============================================================================
# ANALYZER (P3) — MULTI PACKAGE + FULL PIPELINE
# =============================================================================

class MultiPackageAnalyzer:
    """
    Pipeline chính:
    - Ghép 5 công ty × 3 gói ICC → 15 phương án
    - Áp trọng số (Fuzzy nếu bật)
    - Ghép rủi ro khí hậu (Monte Carlo nếu bật)
    - Tính TOPSIS
    - Tính chi phí ước tính
    - Tính độ tin cậy / biến động
    - Trả DataFrame kết quả cuối
    """

    @staticmethod
    def run(params: AnalysisParams) -> AnalysisResult:
        # Load data
        companies_df = DataService.get_company_data()
        climate_df = DataService.load_historical_data()

        # Lấy rủi ro khí hậu theo tháng & tuyến
        base_risk = float(
            climate_df[params.route].iloc[params.month - 1]
        )

        # --------------- Build 15 rows (5 công ty × 3 gói ICC) ----------------
        rows = []
        for company in companies_df.index:
            base_row = companies_df.loc[company]

            for icc_name, icc in ICC_PACKAGES.items():
                new_row = base_row.copy()
                # Premium rate điều chỉnh theo gói ICC
                new_row["C1: Tỷ lệ phí"] *= icc["premium_multiplier"]
                rows.append(
                    {
                        "company": company,
                        "icc_package": icc_name,
                        **new_row.to_dict(),
                    }
                )

        df = pd.DataFrame(rows)

        # ----------------------- Trọng số gốc (profile) -----------------------
        w = pd.Series(PRIORITY_PROFILES[params.priority])

        # -------------------- Fuzzy AHP nếu bật -----------------------
        if params.use_fuzzy:
            w = FuzzyAHP.apply(w, params.fuzzy_uncertainty)

        # ---------------------- Rủi ro khí hậu (C6) ----------------------
        if params.use_mc:
            comp_names, means, stds = MonteCarloSimulator.simulate(
                base_risk,
                SENSITIVITY_MAP,
                params.mc_runs,
            )
            # Ghép theo từng dòng
            df["C6_mean"] = df["company"].map(
                {comp: means[i] for i, comp in enumerate(comp_names)}
            )
            df["C6_std"] = df["company"].map(
                {comp: stds[i] for i, comp in enumerate(comp_names)}
            )
            df["C6: Rủi ro khí hậu"] = df["C6_mean"]
        else:
            df["C6_mean"] = base_risk
            df["C6_std"] = 0.02
            df["C6: Rủi ro khí hậu"] = base_risk

        # ---------------------------------------------------------------------
        # TOPSIS
        # ---------------------------------------------------------------------
        criteria_cols = list(w.index)
        data_matrix = df[criteria_cols]
        topsis_scores = TOPSISAnalyzer.analyze(
            data_matrix, w, COST_BENEFIT_MAP
        )
        df["score"] = topsis_scores

        # ---------------------------------------------------------------------
        # Chi phí ước tính
        # ---------------------------------------------------------------------
        df["estimated_cost"] = (
            df["C1: Tỷ lệ phí"] * params.cargo_value * (1 + df["C3: Tỷ lệ tổn thất"])
        )

        # ---------------------------------------------------------------------
        # Category dạng badge
        # ---------------------------------------------------------------------
        def cat(x):
            if x >= 0.75:
                return "Tối ưu"
            elif x >= 0.60:
                return "Tốt"
            else:
                return "Trung bình"

        df["category"] = df["score"].apply(cat)

        # ---------------------------------------------------------------------
        # Độ tin cậy
        # ---------------------------------------------------------------------
        df["confidence"] = RiskCalculator.calculate_confidence(
            df, data_matrix
        )

        # ---------------------------------------------------------------------
        # Sắp xếp theo TOPSIS
        # ---------------------------------------------------------------------
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        # ---------------------------------------------------------------------
        # Forecast rủi ro khí hậu
        # ---------------------------------------------------------------------
        hist, fc = Forecaster.forecast(
            climate_df, params.route, params.month, params.use_arima
        )

        # ---------------------------------------------------------------------
        # VaR & CVaR
        # ---------------------------------------------------------------------
        var_val, cvar_val = (0.0, 0.0)
        if params.use_var:
            # Loss rate ≈ C3: Tỷ lệ tổn thất × rủi ro khí hậu
            loss_rates = df["C3: Tỷ lệ tổn thất"].values * df["C6_mean"].values
            var_val, cvar_val = RiskCalculator.calculate_var_cvar(
                loss_rates, params.cargo_value
            )

        return AnalysisResult(
            results=df,
            weights=w,
            data_adjusted=data_matrix,
            var=var_val,
            cvar=cvar_val,
            historical=np.array(hist),
            forecast=np.array(fc),
        )


# =============================================================================
# CHART FACTORY — PREMIUM UI
# =============================================================================

class ChartFactory:
    """
    Tạo tất cả chart của RISKCAST Premium v5.3:
    - Ranking bar
    - Category comparison
    - Cost-benefit scatter
    - Forecast neon chart
    - Weight pie
    """

    # --------------------------- RANKING BAR ---------------------------
    @staticmethod
    def create_ranking_bar(df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()

        colors = ["#00ff99" if i < 1 else "#69f0ae" for i in range(len(df))]

        fig.add_trace(
            go.Bar(
                x=df["score"],
                y=df["company"] + " (" + df["icc_package"] + ")",
                orientation="h",
                text=df["score"].round(3),
                marker=dict(color=colors),
            )
        )

        fig.update_layout(
            title="<b>📊 Xếp hạng TOPSIS</b>",
            xaxis=dict(title="Điểm", color="#CCFFE6"),
            yaxis=dict(title="", color="#CCFFE6"),
            paper_bgcolor="#001a12",
            plot_bgcolor="#001a12",
            font=dict(color="#E0F2F1"),
            margin=dict(l=80, r=40, t=80, b=40),
        )
        return fig

    # ---------------------- CATEGORY COMPARISON ----------------------
    @staticmethod
    def create_category_comparison(df: pd.DataFrame) -> go.Figure:
        fig = px.bar(
            df,
            x="category",
            color="category",
            color_discrete_map={
                "Tối ưu": "#00FFAA",
                "Tốt": "#69F0AE",
                "Trung bình": "#2E7D32",
            },
        )
        fig.update_layout(
            title="<b>💎 Phân bố theo loại phương án</b>",
            paper_bgcolor="#001a12",
            plot_bgcolor="#001a12",
            font=dict(color="#E0F2F1"),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    # -------------------- COST-BENEFIT SCATTER -----------------------
    @staticmethod
    def create_cost_scatter(df: pd.DataFrame) -> go.Figure:
        fig = px.scatter(
            df,
            x="estimated_cost",
            y="score",
            color="category",
            size="C6_std",
            hover_name="company",
            color_discrete_map={
                "Tối ưu": "#00FFAA",
                "Tốt": "#69F0AE",
                "Trung bình": "#2E7D32",
            },
        )
        fig.update_layout(
            title="<b>⚖️ Phân tích đánh đổi Chi phí – Điểm</b>",
            xaxis_title="Chi phí ước tính",
            yaxis_title="Điểm TOPSIS",
            paper_bgcolor="#001a12",
            plot_bgcolor="#001a12",
            font=dict(color="#E0F2F1"),
            margin=dict(l=60, r=40, t=80, b=60),
        )
        return fig

    # --------------------------- FORECAST -----------------------------
    @staticmethod
    def create_forecast_chart(hist: np.ndarray, fc: np.ndarray, route: str, month: int) -> go.Figure:
        fig = go.Figure()

        months = list(range(1, len(hist) + 1))

        fig.add_trace(
            go.Scatter(
                x=months,
                y=hist,
                mode="lines+markers",
                name="Dữ liệu thực",
                line=dict(color="#00FFAA", width=3),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[month + 1],
                y=[fc[0]],
                mode="markers",
                name="Dự báo",
                marker=dict(size=14, color="#FFD600"),
            )
        )

        fig.update_layout(
            title=f"<b>🌦 Dự báo rủi ro khí hậu tuyến {route}</b>",
            xaxis_title="Tháng",
            yaxis_title="Rủi ro",
            paper_bgcolor="#001a12",
            plot_bgcolor="#001a12",
            font=dict(color="#E0F2F1"),
            margin=dict(l=60, r=40, t=80, b=60),
        )
        return fig

    # --------------------------- WEIGHTS ------------------------------
    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        fig = px.pie(
            values=weights.values,
            names=weights.index,
            title=f"<b>{title}</b>",
            hole=0.40,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(
            paper_bgcolor="#001a12",
            font=dict(color="#E0F2F1"),
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return fig
# =============================================================================
# EXPORT & REPORT
# =============================================================================

class ReportGenerator:
    """Xuất Excel & PDF cho RISKCAST v5.3."""
    
    @staticmethod
    def generate_pdf(
        results: pd.DataFrame,
        params: AnalysisParams,
        var: float,
        cvar: float
    ) -> bytes:
        try:
            pdf = FPDF()
            pdf.add_page()
            
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST v5.3 - Multi-Package Analysis", 0, 1, "C")
            pdf.ln(4)
            
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, f"Route: {params.route} | Month: {params.month}", 0, 1)
            pdf.cell(0, 6, f"Priority: {params.priority}", 0, 1)
            pdf.cell(0, 6, f"Cargo Value: ${params.cargo_value:,.0f}", 0, 1)
            pdf.ln(4)
            
            top = results.iloc[0]
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, "Top Recommendation:", 0, 1)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, f"{top['company']} - {top['icc_package']}", 0, 1)
            pdf.cell(0, 6, f"Score: {top['score']:.3f}", 0, 1)
            pdf.cell(0, 6, f"Cost: ${top['estimated_cost']:,.0f}", 0, 1)
            pdf.cell(0, 6, f"Confidence: {top['confidence']:.2f}", 0, 1)
            pdf.ln(4)
            
            pdf.set_font("Arial", "B", 10)
            pdf.cell(12, 6, "Rank", 1)
            pdf.cell(38, 6, "Company", 1)
            pdf.cell(20, 6, "ICC", 1)
            pdf.cell(32, 6, "Cost", 1)
            pdf.cell(22, 6, "Score", 1)
            pdf.cell(22, 6, "Conf.", 1)
            pdf.cell(22, 6, "Cat.", 1, 1)
            
            pdf.set_font("Arial", "", 9)
            for i, row in results.head(10).iterrows():
                pdf.cell(12, 6, str(i + 1), 1)
                pdf.cell(38, 6, str(row["company"])[:18], 1)
                pdf.cell(20, 6, str(row["icc_package"]), 1)
                pdf.cell(32, 6, f"${row['estimated_cost']:,.0f}", 1)
                pdf.cell(22, 6, f"{row['score']:.3f}", 1)
                pdf.cell(22, 6, f"{row['confidence']:.2f}", 1)
                pdf.cell(22, 6, str(row["category"]), 1, 1)
            
            if var is not None and cvar is not None:
                pdf.ln(4)
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 6, f"VaR 95%: ${var:,.0f}   |   CVaR 95%: ${cvar:,.0f}", 0, 1)
            
            return pdf.output(dest="S").encode("latin1")
        except Exception as e:
            st.error(f"Lỗi tạo PDF: {e}")
            return b""
    
    @staticmethod
    def generate_excel(results: pd.DataFrame, weights: pd.Series) -> bytes:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Sheet kết quả
            out_cols = [
                "company", "icc_package", "category",
                "estimated_cost", "score", "confidence",
                "C6_mean", "C6_std"
            ]
            (results[out_cols]
             .rename(columns={
                 "company": "Company",
                 "icc_package": "ICC Package",
                 "category": "Category",
                 "estimated_cost": "Estimated Cost",
                 "score": "Score",
                 "confidence": "Confidence",
                 "C6_mean": "C6 Mean",
                 "C6_std": "C6 Std"
             })
             ).to_excel(writer, sheet_name="Results", index=False)

            # Sheet trọng số
            pd.DataFrame(
                {"Criterion": weights.index, "Weight": weights.values}
            ).to_excel(writer, sheet_name="Weights", index=False)
        
        buffer.seek(0)
        return buffer.getvalue()


# =============================================================================
# STREAMLIT UI
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
            layout="wide",
        )
        apply_custom_css()
    
    def render_sidebar(self) -> AnalysisParams:
        with st.sidebar:
            st.header("📦 Thông tin lô hàng")
            cargo_value = st.number_input(
                "Giá trị lô hàng (USD)",
                min_value=1000,
                value=39_000,
                step=1_000,
            )
            good_type = st.selectbox(
                "Loại hàng",
                ["Điện tử", "Đông lạnh", "Hàng khô", "Nguy hiểm", "Khác"],
            )
            route = st.selectbox(
                "Tuyến vận chuyển",
                ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"],
            )
            method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
            month = st.selectbox("Tháng", list(range(1, 13)), index=8)
            
            st.markdown("---")
            st.header("🎯 Mục tiêu ưu tiên")
            priority = st.selectbox(
                "Chọn profile",
                list(PRIORITY_PROFILES.keys()),
            )
            
            st.markdown("---")
            st.header("⚙️ Cấu hình mô hình")
            use_fuzzy = st.checkbox("Bật Fuzzy AHP", True)
            use_arima = st.checkbox("Dùng ARIMA dự báo", True)
            use_mc = st.checkbox("Monte Carlo cho C6", True)
            use_var = st.checkbox("Tính VaR / CVaR", True)
            
            mc_runs = st.number_input(
                "Số lần Monte Carlo",
                min_value=500,
                max_value=15_000,
                value=2_000,
                step=500,
            )
            fuzzy_uncertainty = (
                st.slider("Mức bất định Fuzzy (%)", 0, 50, 15)
                if use_fuzzy
                else 15
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
    
    def _render_header(self):
        st.markdown(
            """
            <div class="app-header">
                <div class="app-header-left">
                    <div class="app-logo-circle">RC</div>
                    <div>
                        <div class="app-header-title">
                            RISKCAST v5.3 — MULTI-PACKAGE ANALYSIS
                        </div>
                        <div class="app-header-subtitle">
                            5 Công ty × 3 Gói ICC = 15 phương án · Profile-Based Recommendation ·
                            Cost-Benefit · Fuzzy AHP · VaR/CVaR · Full Explanations cho NCKH
                        </div>
                    </div>
                </div>
                <div class="app-header-badge">
                    <span>🛡️ ESG Logistics Risk</span>
                    <span>·</span>
                    <span>Smart Insurance Decision</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    def _render_profile_explanation(self, params: AnalysisParams):
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.subheader(f"📌 Mục tiêu hiện tại: {params.priority}")
        
        profile = PRIORITY_PROFILES[params.priority]
        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>⚙️ Trọng số tiêu chí (auto theo profile):</h4>
                <ul>
                    <li><b>C1: Tỷ lệ phí</b> — {profile['C1: Tỷ lệ phí']:.0%}</li>
                    <li><b>C2: Thời gian xử lý</b> — {profile['C2: Thời gian xử lý']:.0%}</li>
                    <li><b>C3: Tỷ lệ tổn thất</b> — {profile['C3: Tỷ lệ tổn thất']:.0%}</li>
                    <li><b>C4: Hỗ trợ ICC</b> — {profile['C4: Hỗ trợ ICC']:.0%}</li>
                    <li><b>C5: Chăm sóc KH</b> — {profile['C5: Chăm sóc KH']:.0%}</li>
                    <li><b>C6: Rủi ro khí hậu</b> — {profile['C6: Rủi ro khí hậu']:.0%}</li>
                </ul>
                <p><b>💡 Ý nghĩa:</b> Hệ thống tự điều chỉnh trọng số theo hành vi người dùng:
                ưu tiên chi phí / cân bằng / an toàn tối đa.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _render_top_summary(self, result: AnalysisResult, params: AnalysisParams):
        df = result.results
        top = df.iloc[0]
        
        st.markdown(
            f"""
            <div class="result-box">
                🏆 <b>GỢI Ý TỐI ƯU CHO MỤC TIÊU: {params.priority}</b><br><br>
                <span style="font-size:1.6rem;">
                    {top['company']} — {top['icc_package']}
                </span><br><br>
                💰 Chi phí ước tính: <b>${top['estimated_cost']:,.0f}</b><br>
                📊 Điểm TOPSIS: <b>{top['score']:.3f}</b> · 
                🎯 Độ tin cậy: <b>{top['confidence']:.2f}</b><br>
                🧩 Phân loại: <b>{top['category']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            f"""
            <div class="explanation-box">
                <h4>🧠 Vì sao phương án này được chọn?</h4>
                <ul>
                    <li>Điểm TOPSIS <b>cao nhất</b> trong 15 phương án.</li>
                    <li>Chi phí bảo hiểm phù hợp với giá trị hàng: 
                        <b>{top['C1: Tỷ lệ phí']:.2%}</b> × Cargo Value.</li>
                    <li>Độ tin cậy mô phỏng cao: <b>{top['confidence']:.2f}</b>.</li>
                    <li>Gói ICC: <b>{top['icc_package']}</b> 
                        — {ICC_PACKAGES[top['icc_package']]['description']}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    def _render_top3_cards(self, result: AnalysisResult):
        df = result.results.copy()
        top3 = df.head(3)
        
        # CSS Premium card + tooltip
        st.markdown(
            """
            <style>
            .top3-card {
                position: relative;
                background: radial-gradient(circle at top left, rgba(0,255,153,0.12), rgba(0,0,0,0.78));
                border: 1px solid rgba(0,255,153,0.45);
                padding: 20px 22px;
                border-radius: 18px;
                box-shadow: 0 0 18px rgba(0,255,153,0.18);
                margin-bottom: 18px;
                text-align: center;
                backdrop-filter: blur(14px);
                -webkit-backdrop-filter: blur(14px);
                transition: transform 0.18s ease-out, box-shadow 0.18s ease-out, border-color 0.18s ease-out;
            }
            .top1-card {
                background: radial-gradient(circle at top left, rgba(255,215,0,0.20), rgba(0,0,0,0.82));
                border: 1px solid rgba(255,215,0,0.7);
                box-shadow: 0 0 26px rgba(255,215,0,0.45);
                animation: gold-pulse 2.4s ease-in-out infinite alternate;
            }
            @keyframes gold-pulse {
                0% {
                    box-shadow: 0 0 10px rgba(255,215,0,0.35);
                    border-color: rgba(255,215,0,0.6);
                }
                100% {
                    box-shadow: 0 0 26px rgba(255,215,0,0.75);
                    border-color: rgba(255,255,255,0.95);
                }
            }
            .top3-card:hover {
                transform: translateY(-4px) scale(1.03);
                box-shadow: 0 0 26px rgba(0,255,153,0.35);
                border-color: rgba(0,255,200,0.85);
            }
            .top3-title {
                font-size: 1.25rem;
                font-weight: 800;
                color: #a5ffdc;
            }
            .top1-title {
                font-size: 1.3rem;
                font-weight: 900;
                color: #ffe680;
                text-shadow: 0 0 10px rgba(255,210,0,0.7);
            }
            .top3-sub {
                font-size: 1rem;
                margin-top: 6px;
                color: #e0f2f1;
            }
            .badge-icc {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: linear-gradient(120deg, #00e676, #00bfa5);
                color: #00130d;
                font-weight: 700;
                font-size: 0.9rem;
            }
            .pill-badge {
                display: inline-block;
                padding: 3px 10px;
                border-radius: 999px;
                border: 1px solid rgba(0,255,153,0.5);
                font-size: 0.85rem;
                margin-top: 4px;
                color: #c8ffec;
            }
            .top3-btn {
                margin-top: 10px;
                padding: 6px 14px;
                border-radius: 999px;
                border: 1px solid rgba(0,255,153,0.7);
                background: rgba(0,0,0,0.65);
                color: #c8ffec;
                font-size: 0.9rem;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.15s ease-out, transform 0.15s ease-out, box-shadow 0.15s ease-out;
            }
            .top3-btn:hover {
                background: linear-gradient(120deg, #00ff99, #00e676);
                color: #00130d;
                transform: translateY(-1px);
                box-shadow: 0 0 12px rgba(0,255,153,0.7);
            }
            .info-tt {
                position: relative;
                display: inline-block;
                cursor: pointer;
            }
            .info-tt .info-text {
                opacity: 0;
                visibility: hidden;
                width: 250px;
                background: rgba(0,0,0,0.9);
                color: #e0f2f1;
                text-align: left;
                border-radius: 8px;
                padding: 10px 12px;
                border: 1px solid rgba(0,255,153,0.45);
                position: absolute;
                z-index: 999;
                bottom: 125%;
                left: 50%;
                transform: translateX(-50%);
                font-size: 0.85rem;
                transition: opacity 0.18s ease-out;
            }
            .info-tt:hover .info-text {
                opacity: 1;
                visibility: visible;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("## 🏅 Top 3 phương án (Premium View)")
        cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        
        for i, col in enumerate(cols):
            if i >= len(top3):
                continue
            r = top3.iloc[i]
            
            card_class = "top3-card"
            title_class = "top3-title"
            if i == 0:
                card_class += " top1-card"
                title_class = "top1-title"
            
            with col:
                st.markdown(
                    f"""
<div class="{card_class}">
    <div class="{title_class}">{medals[i]} #{i+1}: {r['company']}</div>

    <div class="top3-sub info-tt">
        <b class="badge-icc">{r['icc_package']}</b>
        <span class="info-text">
            <b>Loại điều khoản ICC</b><br><br>
            • ICC A: Bảo vệ rộng nhất (All Risks).<br>
            • ICC B: Mức trung bình – Named Perils.<br>
            • ICC C: Cơ bản – bảo vệ ít, phí thấp.<br><br>
            Gói càng cao → phạm vi bảo vệ càng rộng.
        </span>
    </div>

    <div class="top3-sub info-tt" style="color:#7CFFA1; font-size:1.05rem;">
        💰 Chi phí kỳ vọng: <b>${r['estimated_cost']:,.0f}</b>
        <span class="info-text">
            <b>Ý nghĩa chi phí</b><br><br>
            Ước tính dựa trên:<br>
            • Tỷ lệ phí bảo hiểm (C1).<br>
            • Tỷ lệ tổn thất lịch sử (C3).<br><br>
            Giúp DN so sánh giữa tiết kiệm và mức bảo vệ.
        </span>
    </div>

    <div class="top3-sub info-tt">
        📊 Điểm: <b>{r['score']:.3f}</b> · 
        <span class="pill-badge">{r['category']}</span>
        <span class="info-text">
            <b>Điểm TOPSIS</b><br><br>
            Tính từ 6 tiêu chí C1–C6 với trọng số profile.<br>
            Điểm càng cao → phương án càng gần "phương án lý tưởng".
        </span>
    </div>

    <div class="top3-sub info-tt">
        🎯 Tin cậy: <b>{r['confidence']:.2f}</b>
        <span class="info-text">
            <b>Độ tin cậy mô phỏng</b><br><br>
            Dựa trên độ biến động giữa C6_mean và C6_std.<br>
            0.70+ → rất ổn định.<br>
            0.40–0.69 → trung bình.<br>
            &lt; 0.40 → kết quả dễ dao động.
        </span>
    </div>

    <div class="top3-sub info-tt">
        🌪 Biến động rủi ro: <b>{r['C6_std']:.2f}</b>
        <span class="info-text">
            <b>Độ biến động rủi ro khí hậu</b><br><br>
            Càng cao → rủi ro khó dự đoán, tail risk lớn.<br>
            Quan trọng với hàng giá trị cao / tuyến xa.
        </span>
    </div>

    <button class="top3-btn">📘 Xem phân tích chi tiết</button>
</div>
""",
                    unsafe_allow_html=True,
                )
    
    def _render_charts(self, result: AnalysisResult, params: AnalysisParams):
        st.markdown("---")
        st.subheader("📊 Phân tích trực quan")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_rank = self.chart_factory.create_ranking_bar(result.results)
            st.plotly_chart(fig_rank, use_container_width=True)
        with col2:
            fig_cat = self.chart_factory.create_category_comparison(result.results)
            st.plotly_chart(fig_cat, use_container_width=True)
        
        st.markdown("### ⚖️ Cost – Benefit Analysis")
        fig_cost = self.chart_factory.create_cost_scatter(result.results)
        st.plotly_chart(fig_cost, use_container_width=True)
        
        st.markdown("### 🌦 Dự báo rủi ro khí hậu")
        fig_fc = self.chart_factory.create_forecast_chart(
            result.historical, result.forecast, params.route, params.month
        )
        st.plotly_chart(fig_fc, use_container_width=True)
        
        st.markdown("### 🧮 Trọng số tiêu chí")
        fig_w = self.chart_factory.create_weights_pie(
            result.weights, "Trọng số tiêu chí theo profile"
        )
        st.plotly_chart(fig_w, use_container_width=True)
    
    def _render_fuzzy_block(self, result: AnalysisResult, params: AnalysisParams):
        if not params.use_fuzzy:
            return
        
        st.markdown("---")
        st.subheader("🌿 Fuzzy AHP — Phân tích bất định trọng số")
        
        st.markdown(
            """
            <div class="explanation-box">
                <h4>📚 Ý nghĩa Fuzzy AHP trong đề tài:</h4>
                <ul>
                    <li>Mô hình hóa <b>bất định trong đánh giá chuyên gia</b>.</li>
                    <li>Mỗi trọng số không phải 1 điểm cố định mà là <b>tam giác mờ</b> (Low–Mid–High).</li>
                    <li>Sau đó defuzzify (Centroid) để quay về trọng số crisp dùng cho TOPSIS.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        fig_fuzzy = fuzzy_chart_premium(result.weights, params.fuzzy_uncertainty)
        st.plotly_chart(fig_fuzzy, use_container_width=True)
        
        fuzzy_table = build_fuzzy_table(result.weights, params.fuzzy_uncertainty)
        st.subheader("📄 Bảng Low – Mid – High – Centroid (phụ lục NCKH)")
        st.dataframe(fuzzy_table, use_container_width=True)
        
        most_unc, diff_map = most_uncertain_criterion(
            result.weights, params.fuzzy_uncertainty
        )
        st.markdown(
            f"""
            <div style="background:#00331F; padding:15px; border-radius:10px;
            border:2px solid #00FFAA; color:#CCFFE6; font-size:16px; margin-top:0.8rem;">
            🔍 <b>Tiêu chí dao động mạnh nhất (High - Low lớn nhất):</b><br>
            <span style="color:#00FFAA; font-size:20px;"><b>{most_unc}</b></span><br><br>
            💡 <b>Ý nghĩa:</b> Tiêu chí này <b>nhạy cảm nhất</b> khi thay đổi đánh giá chuyên gia.<br>
            Gợi ý: cần thu thập thêm dữ liệu / ý kiến để giảm bất định.
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.subheader("🔥 Heatmap mức dao động Fuzzy")
        fig_heat = fuzzy_heatmap_premium(diff_map)
        st.plotly_chart(fig_heat, use_container_width=True)
    
    def _render_var_block(self, result: AnalysisResult, params: AnalysisParams):
        if not params.use_var:
            return
        
        var_val, cvar_val = result.var, result.cvar
        if var_val is None or cvar_val is None:
            return
        
        risk_pct = (var_val / params.cargo_value) * 100
        st.markdown("---")
        st.subheader("⚠️ Đánh giá rủi ro tài chính (VaR / CVaR)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 VaR 95%", f"${var_val:,.0f}")
        with col2:
            st.metric("🛡️ CVaR 95%", f"${cvar_val:,.0f}")
        with col3:
            st.metric("📊 Rủi ro / Giá trị hàng", f"{risk_pct:.1f}%")
        
        st.markdown(
            f"""
            <div class="explanation-box">
                <ul>
                    <li><b>VaR 95%</b>: tổn thất tối đa dự kiến trong 95% trường hợp.</li>
                    <li><b>CVaR 95%</b>: tổn thất trung bình trong 5% kịch bản xấu nhất.</li>
                    <li><b>Nhận định:</b> {"✅ Rủi ro ở mức chấp nhận được." if risk_pct < 10 else "⚠️ Rủi ro tương đối cao, cần xem lại mức bảo vệ / tuyến / gói ICC."}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    def _render_export(self, result: AnalysisResult, params: AnalysisParams):
        st.markdown("---")
        st.subheader("📥 Xuất báo cáo")
        col1, col2 = st.columns(2)
        
        with col1:
            excel_bytes = self.report_gen.generate_excel(
                result.results, result.weights
            )
            st.download_button(
                "📊 Tải Excel kết quả",
                data=excel_bytes,
                file_name=f"riskcast_v53_{params.route.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with col2:
            pdf_bytes = self.report_gen.generate_pdf(
                result.results, params, result.var, result.cvar
            )
            if pdf_bytes:
                st.download_button(
                    "📄 Tải PDF tóm tắt",
                    data=pdf_bytes,
                    file_name=f"riskcast_v53_{params.route.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
    
    def run(self):
        self.initialize()
        self._render_header()
        
        params = self.render_sidebar()
        self._render_profile_explanation(params)
        
        st.markdown("---")
        if st.button("🚀 PHÂN TÍCH 15 PHƯƠNG ÁN", type="primary", use_container_width=True):
            with st.spinner("🔄 Đang chạy Multi-Package Analysis..."):
                try:
                    result = MultiPackageAnalyzer.run(params)
                    st.success("✅ Phân tích thành công!")
                    self._render_top_summary(result, params)
                    self._render_top3_cards(result)
                    self._render_charts(result, params)
                    self._render_fuzzy_block(result, params)
                    self._render_var_block(result, params)
                    self._render_export(result, params)
                except Exception as e:
                    st.error(f"❌ Lỗi khi chạy phân tích: {e}")
                    st.exception(e)


# =============================================================================
# MAIN
# =============================================================================

def main():
    app = StreamlitUI()
    app.run()


if __name__ == "__main__":
    main()
