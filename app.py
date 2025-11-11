# =============================================================
# RISKCAST v5.0 — ESG Logistics Risk Assessment Dashboard
# Optimized by Claude - Enhanced Architecture & Performance
# Original Author: Bùi Xuân Hoàng
# =============================================================

import io
import warnings
from typing import Dict, List, Tuple, Optional
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
# CONFIGURATION & CONSTANTS
# =============================================================================

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
    "C1: Tỷ lệ phí": "cost",
    "C2: Thời gian xử lý": "cost",
    "C3: Tỷ lệ tổn thất": "cost",
    "C4: Hỗ trợ ICC": "benefit",
    "C5: Chăm sóc KH": "benefit",
    "C6: Rủi ro khí hậu": "cost"
}

SENSITIVITY_MAP = {
    "Chubb": 0.95,
    "PVI": 1.10,
    "InternationalIns": 1.20,
    "BaoViet": 1.05,
    "Aon": 0.90
}

COLOR_PALETTE = {
    "primary": "#0066CC",        # Xanh dương sáng
    "secondary": "#FFB800",      # Vàng gold
    "accent": "#FF6B35",         # Cam rõ nét
    "success": "#00C853",        # Xanh lá sáng
    "info": "#00B8D4",          # Xanh cyan
    "text_dark": "#1A1A1A",     # Đen đậm
    "text_medium": "#424242",   # Xám đậm
    "bg_light": "#F8FAFC",      # Nền sáng
    "border": "#E0E7FF"         # Viền xanh nhạt
}

# =============================================================================
# UI STYLING
# =============================================================================

def apply_custom_css():
    """Enhanced CSS with crystal-clear typography and modern design"""
    st.markdown("""
    <style>
        /* Global Reset for Maximum Clarity */
        * {
            opacity: 1 !important;
            text-rendering: optimizeLegibility !important;
            -webkit-font-smoothing: antialiased !important;
            -moz-osx-font-smoothing: grayscale !important;
        }
        
        /* App Background - Clean White/Light Blue */
        .stApp {
            background: linear-gradient(135deg, #F0F4FF 0%, #FFFFFF 100%) !important;
            font-family: 'Inter', 'Segoe UI', 'Arial', sans-serif !important;
        }
        
        /* Main Container - Bright White Card */
        .block-container {
            background: #FFFFFF !important;
            padding: 2.5rem 3rem !important;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 102, 204, 0.08);
            border: 2px solid #E0E7FF;
            max-width: 1400px;
            margin: 2rem auto;
        }
        
        /* Typography - Bold & Clear with Bright Colors */
        h1 {
            color: #0066CC !important;
            font-weight: 800 !important;
            font-size: 2.75rem !important;
            letter-spacing: -0.03em !important;
            margin-bottom: 0.5rem !important;
            text-shadow: 0 2px 4px rgba(0,102,204,0.1);
        }
        
        h2 {
            color: #1A1A1A !important;
            font-weight: 700 !important;
            font-size: 1.9rem !important;
            margin: 1.5rem 0 1rem 0 !important;
        }
        
        h3 {
            color: #424242 !important;
            font-weight: 700 !important;
            font-size: 1.4rem !important;
        }
        
        /* Text Elements - High Contrast */
        .stMarkdown, p, label, .stText {
            color: #1A1A1A !important;
            font-weight: 500 !important;
            font-size: 1.05rem !important;
            line-height: 1.7 !important;
        }
        
        /* Subtitle Text */
        .stMarkdown strong {
            color: #424242 !important;
            font-weight: 600 !important;
        }
        
        /* Enhanced Buttons - Bright Blue & Gold */
        .stButton > button {
            background: linear-gradient(135deg, #0066CC 0%, #0052A3 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.85rem 2.5rem !important;
            font-weight: 700 !important;
            font-size: 1.15rem !important;
            letter-spacing: 0.3px !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 4px 14px rgba(0, 102, 204, 0.25) !important;
            text-transform: uppercase;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #0052A3 0%, #003D7A 100%) !important;
            transform: translateY(-3px) !important;
            box-shadow: 0 6px 20px rgba(0, 102, 204, 0.35) !important;
        }
        
        /* Result Box - Gold Gradient */
        .result-box {
            background: linear-gradient(135deg, #FFB800 0%, #FFA000 100%);
            color: #1A1A1A !important;
            padding: 1.8rem 2.5rem;
            border-radius: 12px;
            font-weight: 800;
            font-size: 1.3rem;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: 0 6px 20px rgba(255, 184, 0, 0.3);
            border: 3px solid #FFD54F;
        }
        
        /* Data Tables - Clean White */
        .stDataFrame {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
            background: white !important;
            border: 1px solid #E0E7FF;
        }
        
        .stDataFrame th {
            background: #0066CC !important;
            color: white !important;
            font-weight: 700 !important;
            padding: 12px !important;
        }
        
        .stDataFrame td {
            padding: 10px !important;
            color: #1A1A1A !important;
        }
        
        /* Sidebar - Bright & Clean */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%) !important;
            border-right: 3px solid #0066CC;
            padding: 2rem 1.5rem;
        }
        
        section[data-testid="stSidebar"] h2 {
            color: #0066CC !important;
            font-weight: 800 !important;
            font-size: 1.5rem !important;
            border-bottom: 3px solid #FFB800;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        section[data-testid="stSidebar"] label {
            color: #1A1A1A !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
        }
        
        /* Metric Cards - Bright Colors */
        [data-testid="stMetricValue"] {
            color: #0066CC !important;
            font-weight: 900 !important;
            font-size: 2.2rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #424242 !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
        }
        
        /* Input Fields - High Contrast */
        .stNumberInput input, .stSelectbox select, .stSlider {
            border: 2px solid #0066CC !important;
            border-radius: 8px !important;
            padding: 0.6rem !important;
            font-weight: 600 !important;
            color: #1A1A1A !important;
            background: white !important;
        }
        
        .stNumberInput input:focus, .stSelectbox select:focus {
            border-color: #FFB800 !important;
            box-shadow: 0 0 0 3px rgba(255, 184, 0, 0.2) !important;
        }
        
        /* Checkboxes - Bright Blue */
        .stCheckbox {
            font-weight: 600 !important;
            color: #1A1A1A !important;
        }
        
        .stCheckbox input:checked {
            background-color: #0066CC !important;
        }
        
        /* Success/Info Messages - High Visibility */
        .stSuccess {
            background: #E8F5E9 !important;
            border-radius: 10px !important;
            border-left: 5px solid #00C853 !important;
            font-weight: 600 !important;
            color: #1B5E20 !important;
            padding: 1rem 1.5rem !important;
        }
        
        .stInfo {
            background: #E1F5FE !important;
            border-radius: 10px !important;
            border-left: 5px solid #00B8D4 !important;
            font-weight: 600 !important;
            color: #01579B !important;
            padding: 1rem 1.5rem !important;
        }
        
        /* Spinner - Blue */
        .stSpinner > div {
            border-top-color: #0066CC !important;
        }
        
        /* Horizontal Divider */
        hr {
            border: none;
            height: 3px;
            background: linear-gradient(90deg, #0066CC, #FFB800);
            margin: 2rem 0;
        }
        
        /* Download Buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #FFB800 0%, #FFA000 100%) !important;
            color: #1A1A1A !important;
            font-weight: 700 !important;
            border: 2px solid #FFD54F !important;
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #FFA000 0%, #FF8F00 100%) !important;
            transform: scale(1.05) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & GENERATION
# =============================================================================

@st.cache_data
def load_sample_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample historical and claims data"""
    months = list(range(1, 13))
    
    # Historical climate risk data by route
    climate_base = {
        "VN - EU": [0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.42, 0.48, 0.60, 0.68, 0.58, 0.45],
        "VN - US": [0.30, 0.33, 0.36, 0.40, 0.45, 0.50, 0.56, 0.62, 0.75, 0.72, 0.60, 0.52],
        "VN - Singapore": [0.15, 0.16, 0.18, 0.20, 0.22, 0.26, 0.30, 0.32, 0.35, 0.34, 0.28, 0.25],
        "VN - China": [0.18, 0.19, 0.21, 0.24, 0.26, 0.30, 0.34, 0.36, 0.40, 0.38, 0.32, 0.28],
        "Domestic": [0.10] * 12
    }
    
    historical = pd.DataFrame({"month": months})
    for route, values in climate_base.items():
        historical[route] = values
    
    # Claims data
    rng = np.random.default_rng(42)
    loss_rates = np.clip(rng.normal(loc=0.08, scale=0.02, size=2000), 0, 0.5)
    claims = pd.DataFrame({"loss_rate": loss_rates})
    
    return historical, claims

@st.cache_data
def get_company_data() -> pd.DataFrame:
    """Get insurance company baseline data"""
    return pd.DataFrame({
        "Company": ["Chubb", "PVI", "InternationalIns", "BaoViet", "Aon"],
        "C1: Tỷ lệ phí": [0.30, 0.28, 0.26, 0.32, 0.24],
        "C2: Thời gian xử lý": [6, 5, 8, 7, 4],
        "C3: Tỷ lệ tổn thất": [0.08, 0.06, 0.09, 0.10, 0.07],
        "C4: Hỗ trợ ICC": [9, 8, 6, 9, 7],
        "C5: Chăm sóc KH": [9, 8, 5, 7, 6],
    }).set_index("Company")

# =============================================================================
# CORE ALGORITHMS
# =============================================================================

def auto_balance_weights(weights: np.ndarray, locked: List[bool]) -> np.ndarray:
    """
    Auto-balance weights to sum to 1.0, respecting locked weights
    
    Args:
        weights: Current weight values
        locked: Boolean flags indicating which weights are locked
        
    Returns:
        Balanced weights array
    """
    w = np.array(weights, dtype=float)
    locked_flags = np.array(locked, dtype=bool)
    
    total_locked = w[locked_flags].sum()
    free_idx = np.where(~locked_flags)[0]
    
    if len(free_idx) == 0:
        return w / w.sum() if w.sum() != 0 else np.ones_like(w) / len(w)
    
    remaining = max(0.0, 1.0 - total_locked)
    free_sum = w[free_idx].sum()
    
    if free_sum == 0:
        w[free_idx] = remaining / len(free_idx)
    else:
        w[free_idx] = w[free_idx] / free_sum * remaining
    
    w = np.clip(w, 0.0, 1.0)
    diff = 1.0 - w.sum()
    
    if abs(diff) > 1e-8:
        w[free_idx[0]] += diff
    
    return np.round(w, 6)

def defuzzify_triangular(low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Defuzzify triangular fuzzy numbers using centroid method"""
    return (low + mid + high) / 3.0

def apply_fuzzy_ahp(weights: pd.Series, uncertainty_pct: float) -> pd.Series:
    """
    Apply Fuzzy AHP with triangular fuzzy numbers
    
    Args:
        weights: Base weights
        uncertainty_pct: Uncertainty percentage (0-100)
        
    Returns:
        Defuzzified weights
    """
    factor = uncertainty_pct / 100.0
    low = np.maximum(weights.values * (1 - factor), 1e-9)
    high = np.minimum(weights.values * (1 + factor), 0.9999)
    
    defuzzified = defuzzify_triangular(low, weights.values, high)
    normalized = defuzzified / defuzzified.sum()
    
    return pd.Series(normalized, index=weights.index)

@st.cache_data
def monte_carlo_simulation(
    base_risk: float,
    sensitivity_map: Dict[str, float],
    n_simulations: int
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Vectorized Monte Carlo simulation for climate risk
    
    Args:
        base_risk: Base climate risk value
        sensitivity_map: Company sensitivity factors
        n_simulations: Number of Monte Carlo runs
        
    Returns:
        Tuple of (company_names, mean_risks, std_risks)
    """
    rng = np.random.default_rng(2025)
    companies = list(sensitivity_map.keys())
    
    # Calculate mean and std for each company
    mu = np.array([base_risk * sensitivity_map[c] for c in companies])
    sigma = np.maximum(0.03, mu * 0.12)
    
    # Vectorized simulation
    simulations = rng.normal(loc=mu, scale=sigma, size=(n_simulations, len(companies)))
    simulations = np.clip(simulations, 0.0, 1.0)
    
    mean_risks = simulations.mean(axis=0)
    std_risks = simulations.std(axis=0)
    
    return companies, mean_risks, std_risks

def topsis_analysis(
    data: pd.DataFrame,
    weights: pd.Series,
    cost_benefit: Dict[str, str]
) -> np.ndarray:
    """
    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    
    Args:
        data: Decision matrix
        weights: Criteria weights
        cost_benefit: Cost/benefit mapping for each criterion
        
    Returns:
        TOPSIS scores for each alternative
    """
    # Normalize decision matrix
    M = data[list(weights.index)].values.astype(float)
    denom = np.sqrt((M ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = M / denom
    
    # Weighted normalized matrix
    w = weights.values
    V = R * w
    
    # Determine ideal best and worst
    is_cost = np.array([cost_benefit[c] == "cost" for c in weights.index])
    ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
    ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))
    
    # Calculate distances
    d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    
    # Calculate relative closeness
    scores = d_minus / (d_plus + d_minus + 1e-12)
    
    return scores

def calculate_var_cvar(
    loss_rates: np.ndarray,
    cargo_value: float,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    
    Args:
        loss_rates: Array of loss rate scenarios
        cargo_value: Total cargo value
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (VaR, CVaR)
    """
    losses = loss_rates * cargo_value
    
    if len(losses) == 0:
        return 0.0, 0.0
    
    var = float(np.percentile(losses, confidence * 100))
    tail_losses = losses[losses >= var]
    cvar = float(tail_losses.mean()) if len(tail_losses) > 0 else var
    
    return var, cvar

def forecast_climate_risk(
    historical_data: pd.DataFrame,
    route: str,
    months_ahead: int = 3,
    use_arima: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecast future climate risk using ARIMA or simple trend
    
    Args:
        historical_data: Historical risk data
        route: Selected route
        months_ahead: Number of months to forecast
        use_arima: Whether to use ARIMA (if available)
        
    Returns:
        Tuple of (historical_series, forecast_values)
    """
    if route not in historical_data.columns:
        route = historical_data.columns[1]
    
    series = historical_data[route].values
    
    # Try ARIMA if available and enabled
    if use_arima and ARIMA_AVAILABLE and len(series) >= 6:
        try:
            model = ARIMA(series, order=(1, 1, 1))
            fitted = model.fit()
            forecast = fitted.forecast(months_ahead)
            forecast = np.clip(forecast, 0, 1)
            return series, np.asarray(forecast)
        except Exception:
            pass
    
    # Fallback to simple trend
    if len(series) >= 3:
        trend = (series[-1] - series[-3]) / 3.0
    else:
        trend = 0.0
    
    forecast = np.array([
        np.clip(series[-1] + (i + 1) * trend, 0, 1)
        for i in range(months_ahead)
    ])
    
    return series, forecast

def calculate_confidence_scores(
    results: pd.DataFrame,
    data: pd.DataFrame
) -> np.ndarray:
    """
    Calculate confidence scores based on variability
    
    Args:
        results: Results dataframe with C6 statistics
        data: Original decision matrix
        
    Returns:
        Array of confidence scores
    """
    eps = 1e-9
    
    # Confidence from C6 variability
    cv_c6 = np.where(
        results["C6_mean"].values == 0,
        0.0,
        results["C6_std"].values / (results["C6_mean"].values + eps)
    )
    conf_c6 = 1.0 / (1.0 + cv_c6)
    
    # Scale to [0.3, 1.0] range
    conf_c6 = np.atleast_1d(conf_c6)
    rng = np.ptp(conf_c6)
    if rng > 0:
        conf_c6_scaled = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / rng
    else:
        conf_c6_scaled = np.full_like(conf_c6, 0.65)
    
    # Confidence from criteria variability
    crit_cv = data.std(axis=1).values / (data.mean(axis=1).values + eps)
    conf_crit = np.atleast_1d(1.0 / (1.0 + crit_cv))
    
    rng2 = np.ptp(conf_crit)
    if rng2 > 0:
        conf_crit_scaled = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / rng2
    else:
        conf_crit_scaled = np.full_like(conf_crit, 0.65)
    
    # Geometric mean of both confidence measures
    conf_final = np.sqrt(conf_c6_scaled * conf_crit_scaled)
    
    return conf_final

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_plotly_figure(fig: go.Figure, title: str) -> go.Figure:
    """Apply consistent styling to Plotly figures"""
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=title,
            font=dict(size=22, color="#0066CC", family="Inter, Arial", weight="bold"),
            x=0.5,
            xanchor="center"
        ),
        font=dict(
            family="Inter, Arial, sans-serif",
            size=15,
            color="#1A1A1A"
        ),
        legend=dict(
            font=dict(size=14, color="#1A1A1A"),
            bgcolor="rgba(255,255,255,0.98)",
            bordercolor="#E0E7FF",
            borderwidth=2
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=70, r=70, t=90, b=70),
        hoverlabel=dict(
            bgcolor="white",
            font_size=15,
            bordercolor="#0066CC"
        )
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#F0F4FF",
        linecolor="#E0E7FF",
        linewidth=2,
        tickfont=dict(size=14, color="#1A1A1A", family="Inter, Arial")
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F0F4FF",
        linecolor="#E0E7FF",
        linewidth=2,
        tickfont=dict(size=14, color="#1A1A1A", family="Inter, Arial")
    )
    
    return fig

def create_weights_pie_chart(weights: pd.Series, title: str) -> go.Figure:
    """Create pie chart for weight distribution"""
    # Bright color palette
    colors = ['#0066CC', '#FFB800', '#00C853', '#FF6B35', '#00B8D4', '#9C27B0']
    
    fig = px.pie(
        values=weights.values,
        names=weights.index,
        title=title,
        color_discrete_sequence=colors
    )
    
    fig.update_traces(
        textposition='outside',
        textinfo='percent+label',
        textfont=dict(size=14, color='#1A1A1A', family="Inter, Arial", weight="bold"),
        marker=dict(line=dict(color='white', width=3))
    )
    
    return create_plotly_figure(fig, title)

def create_topsis_bar_chart(results: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart for TOPSIS scores"""
    fig = px.bar(
        results.sort_values("score"),
        x="score",
        y="company",
        orientation="h",
        title="🏆 TOPSIS Score (cao hơn = tốt hơn)",
        text="score",
        color="score",
        color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#0066CC'], [1, '#003D7A']]
    )
    
    fig.update_traces(
        texttemplate="<b>%{text:.3f}</b>",
        textposition="outside",
        textfont=dict(size=17, color="#1A1A1A", weight="bold"),
        marker_line_width=0,
        hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
    )
    
    fig.update_xaxes(title="<b>TOPSIS Score</b>", range=[0, 1])
    fig.update_yaxes(title="<b>Công ty</b>")
    
    return create_plotly_figure(fig, "🏆 TOPSIS Score (cao hơn = tốt hơn)")

def create_forecast_chart(
    historical: np.ndarray,
    forecast: np.ndarray,
    route: str
) -> go.Figure:
    """Create line chart for climate risk forecast"""
    months_hist = list(range(1, len(historical) + 1))
    months_fc = list(range(len(historical) + 1, len(historical) + len(forecast) + 1))
    months_fc = [min(m, 12) for m in months_fc]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months_hist,
        y=historical,
        mode="lines+markers",
        name="📈 Lịch sử",
        line=dict(color="#0066CC", width=4),
        marker=dict(size=10, symbol="circle", color="#0066CC",
                   line=dict(width=2, color='white')),
        hovertemplate='<b>Tháng %{x}</b><br>Rủi ro: %{y:.2%}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=months_fc,
        y=forecast,
        mode="lines+markers",
        name="🔮 Dự báo",
        line=dict(color="#FF6B35", width=4, dash="dash"),
        marker=dict(size=12, symbol="diamond", color="#FF6B35",
                   line=dict(width=2, color='white')),
        hovertemplate='<b>Tháng %{x}</b><br>Dự báo: %{y:.2%}<extra></extra>'
    ))
    
    fig = create_plotly_figure(fig, f"📊 Dự báo rủi ro khí hậu: {route}")
    
    fig.update_xaxes(
        title="<b>Tháng</b>",
        tickmode="linear",
        tickvals=list(range(1, 13)),
        dtick=1
    )
    
    fig.update_yaxes(
        title="<b>Mức rủi ro (0-1)</b>",
        range=[0, max(1, max(historical.max(), forecast.max()) * 1.1)],
        tickformat='.0%'
    )
    
    return fig

# =============================================================================
# PDF EXPORT
# =============================================================================

def generate_pdf_report(
    results: pd.DataFrame,
    route: str,
    month: int,
    method: str,
    cargo_value: float,
    priority: str,
    var: Optional[float],
    cvar: Optional[float]
) -> bytes:
    """Generate PDF report with results"""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 12, "RISKCAST v5.0 - Executive Summary", 0, 1, "C")
        pdf.ln(8)
        
        # Metadata
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Thông tin phân tích:", 0, 1)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 7, f"Tuyến: {route}", 0, 1)
        pdf.cell(0, 7, f"Tháng: {month} | Phương thức: {method}", 0, 1)
        pdf.cell(0, 7, f"Giá trị lô hàng: ${cargo_value:,}", 0, 1)
        pdf.cell(0, 7, f"Ưu tiên: {priority}", 0, 1)
        pdf.ln(8)
        
        # Top recommendation
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Khuyến nghị hàng đầu:", 0, 1)
        pdf.set_font("Arial", "", 11)
        top = results.iloc[0]
        pdf.multi_cell(
            0, 7,
            f"{top['company']} - Score: {top['score']:.3f} | "
            f"Confidence: {top['confidence']:.2f} | {top['recommend_icc']}"
        )
        pdf.ln(8)
        
        # Results table
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Bảng xếp hạng:", 0, 1)
        
        # Table header
        pdf.set_font("Arial", "B", 10)
        pdf.cell(15, 7, "Rank", 1, 0, "C")
        pdf.cell(50, 7, "Company", 1, 0, "C")
        pdf.cell(25, 7, "Score", 1, 0, "C")
        pdf.cell(30, 7, "Confidence", 1, 0, "C")
        pdf.cell(30, 7, "ICC Level", 1, 1, "C")
        
        # Table rows
        pdf.set_font("Arial", "", 10)
        for _, row in results.iterrows():
            pdf.cell(15, 7, str(int(row["rank"])), 1, 0, "C")
            pdf.cell(50, 7, str(row["company"])[:20], 1, 0, "L")
            pdf.cell(25, 7, f"{row['score']:.3f}", 1, 0, "C")
            pdf.cell(30, 7, f"{row['confidence']:.2f}", 1, 0, "C")
            pdf.cell(30, 7, str(row["recommend_icc"]), 1, 1, "C")
        
        # Risk metrics
        if var and cvar:
            pdf.ln(8)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Chỉ số rủi ro:", 0, 1)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 7, f"VaR 95%: ${var:,.0f}", 0, 1)
            pdf.cell(0, 7, f"CVaR 95%: ${cvar:,.0f}", 0, 1)
        
        pdf_output = pdf.output(dest='S').encode('latin1')
        return pdf_output
        
    except Exception as e:
        st.error(f"Lỗi tạo PDF: {e}")
        return b""

# =============================================================================
# EXCEL EXPORT
# =============================================================================

def generate_excel_report(
    results: pd.DataFrame,
    data_adjusted: pd.DataFrame,
    weights: pd.Series
) -> bytes:
    """Generate Excel report with multiple sheets"""
    excel_buffer = io.BytesIO()
    
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="Results", index=False)
        data_adjusted.to_excel(writer, sheet_name="Adjusted_Data")
        pd.DataFrame({"weight": weights.values}, index=weights.index).to_excel(
            writer, sheet_name="Weights"
        )
    
    excel_buffer.seek(0)
    return excel_buffer.getvalue()

# =============================================================================
# STREAMLIT APP
# =============================================================================

def initialize_session_state():
    """Initialize session state variables"""
    if "weights" not in st.session_state:
        st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
    
    if "locked" not in st.session_state:
        st.session_state["locked"] = [False] * len(CRITERIA)

def render_sidebar() -> Dict:
    """Render sidebar with input controls"""
    with st.sidebar:
        st.header("📊 Thông tin lô hàng")
        
        cargo_value = st.number_input(
            "Giá trị lô hàng (USD)",
            min_value=1000,
            value=39000,
            step=1000
        )
        
        good_type = st.selectbox(
            "Loại hàng",
            ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"]
        )
        
        route = st.selectbox(
            "Tuyến",
            ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"]
        )
        
        method = st.selectbox(
            "Phương thức",
            ["Sea", "Air", "Truck"]
        )
        
        month = st.selectbox(
            "Tháng (1-12)",
            list(range(1, 13)),
            index=8
        )
        
        priority = st.selectbox(
            "Ưu tiên",
            ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"]
        )
        
        st.markdown("---")
        st.header("⚙️ Cấu hình mô hình")
        
        use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN)", True)
        use_arima = st.checkbox("Dùng ARIMA (nếu có)", True)
        use_mc = st.checkbox("Chạy Monte Carlo cho C6", True)
        use_var = st.checkbox("Tính VaR và CVaR", True)
        
        mc_runs = st.number_input(
            "Số vòng Monte Carlo",
            min_value=500,
            max_value=10000,
            value=2000,
            step=500
        )
        
        fuzzy_uncertainty = 15
        if use_fuzzy:
            fuzzy_uncertainty = st.slider(
                "Bất định TFN (%)",
                min_value=0,
                max_value=50,
                value=15
            )
        
        return {
            "cargo_value": cargo_value,
            "good_type": good_type,
            "route": route,
            "method": method,
            "month": month,
            "priority": priority,
            "use_fuzzy": use_fuzzy,
            "use_arima": use_arima,
            "use_mc": use_mc,
            "use_var": use_var,
            "mc_runs": mc_runs,
            "fuzzy_uncertainty": fuzzy_uncertainty
        }

def render_weight_controls():
    """Render weight adjustment controls"""
    st.subheader("🎯 Phân bổ trọng số (Realtime)")
    
    cols = st.columns(len(CRITERIA))
    new_weights = st.session_state["weights"].copy()
    
    for i, criterion in enumerate(CRITERIA):
        with cols[i]:
            st.markdown(f"**{criterion}**")
            
            lock_key = f"lock_{i}"
            weight_key = f"weight_{i}"
            
            is_locked = st.checkbox("🔒 Lock", value=st.session_state["locked"][i], key=lock_key)
            st.session_state["locked"][i] = is_locked
            
            weight_val = st.number_input(
                "Tỉ lệ",
                min_value=0.0,
                max_value=1.0,
                value=float(new_weights[i]),
                step=0.01,
                key=weight_key,
                label_visibility="collapsed"
            )
            new_weights[i] = weight_val
    
    # Reset button
    if st.button("🔄 Reset trọng số mặc định"):
        st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
        st.session_state["locked"] = [False] * len(CRITERIA)
        st.rerun()
    else:
        st.session_state["weights"] = auto_balance_weights(
            new_weights,
            st.session_state["locked"]
        )

def run_analysis(params: Dict, historical: pd.DataFrame):
    """Execute main analysis workflow"""
    
    # Get weights
    weights = pd.Series(st.session_state["weights"], index=CRITERIA)
    
    # Apply Fuzzy AHP if enabled
    if params["use_fuzzy"]:
        weights = apply_fuzzy_ahp(weights, params["fuzzy_uncertainty"])
    
    # Get company data
    company_data = get_company_data()
    
    # Get base climate risk
    base_climate = float(
        historical.loc[
            historical["month"] == params["month"],
            params["route"]
        ].iloc[0]
    ) if params["month"] in historical["month"].values else 0.4
    
    # Run Monte Carlo simulation
    if params["use_mc"]:
        companies, mc_mean, mc_std = monte_carlo_simulation(
            base_climate,
            SENSITIVITY_MAP,
            params["mc_runs"]
        )
        
        # Reorder to match company_data index
        order = [companies.index(c) for c in company_data.index]
        mc_mean = mc_mean[order]
        mc_std = mc_std[order]
    else:
        mc_mean = np.zeros(len(company_data))
        mc_std = np.zeros(len(company_data))
    
    # Add C6 to data
    data_adjusted = company_data.copy()
    data_adjusted["C6: Rủi ro khí hậu"] = mc_mean
    
    # Adjust for high-value cargo
    if params["cargo_value"] > 50000:
        data_adjusted["C1: Tỷ lệ phí"] *= 1.1
    
    # Run TOPSIS
    scores = topsis_analysis(data_adjusted, weights, COST_BENEFIT_MAP)
    
    # Prepare results
    results = pd.DataFrame({
        "company": data_adjusted.index,
        "score": scores,
        "C6_mean": mc_mean,
        "C6_std": mc_std
    }).sort_values("score", ascending=False).reset_index(drop=True)
    
    results["rank"] = results.index + 1
    
    # ICC recommendations
    results["recommend_icc"] = results["score"].apply(
        lambda s: "ICC A" if s >= 0.75 else ("ICC B" if s >= 0.5 else "ICC C")
    )
    
    # Calculate confidence scores
    conf_scores = calculate_confidence_scores(results, data_adjusted)
    order_map = {comp: conf_scores[i] for i, comp in enumerate(data_adjusted.index)}
    results["confidence"] = results["company"].map(order_map).round(3)
    
    # Calculate VaR/CVaR
    var, cvar = None, None
    if params["use_var"]:
        var, cvar = calculate_var_cvar(
            results["C6_mean"].values,
            params["cargo_value"],
            confidence=0.95
        )
    
    # Forecast
    hist_series, forecast = forecast_climate_risk(
        historical,
        params["route"],
        months_ahead=3,
        use_arima=params["use_arima"]
    )
    
    return {
        "results": results,
        "weights": weights,
        "data_adjusted": data_adjusted,
        "var": var,
        "cvar": cvar,
        "historical": hist_series,
        "forecast": forecast
    }

def display_results(analysis: Dict, params: Dict):
    """Display analysis results"""
    
    results = analysis["results"]
    weights = analysis["weights"]
    
    st.success("✅ **Hoàn tất phân tích thành công!**")
    
    # Main layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("🏅 Kết quả xếp hạng TOPSIS")
        
        display_df = results[["rank", "company", "score", "confidence", "recommend_icc"]].set_index("rank")
        display_df.columns = ["Công ty", "Điểm số", "Độ tin cậy", "ICC khuyến nghị"]
        st.dataframe(display_df, use_container_width=True)
        
        # Top recommendation - Gold highlight
        top = results.iloc[0]
        st.markdown(
            f"""<div class='result-box'>
            🏆 <b>KHUYẾN NGHỊ HÀNG ĐẦU</b> 🏆<br><br>
            <span style='font-size:1.5rem;'><b>{top['company']}</b></span><br>
            Điểm số: <b>{top['score']:.3f}</b> | 
            Độ tin cậy: <b>{top['confidence']:.2f}</b> | 
            <b>{top['recommend_icc']}</b>
            </div>""",
            unsafe_allow_html=True
        )
    
    with right_col:
        if analysis["var"] and analysis["cvar"]:
            st.metric(
                "💰 VaR 95%", 
                f"${analysis['var']:,.0f}", 
                help="Value at Risk - Tổn thất tối đa với độ tin cậy 95%"
            )
            st.metric(
                "🛡️ CVaR 95%", 
                f"${analysis['cvar']:,.0f}", 
                help="Conditional VaR - Tổn thất trung bình vượt VaR"
            )
        
        # Weight distribution pie chart
        fig_weights = create_weights_pie_chart(weights, "⚖️ Phân bổ trọng số cuối")
        st.plotly_chart(fig_weights, use_container_width=True)
    
    # Charts
    st.markdown("---")
    st.subheader("📈 Biểu đồ Phân tích Chi tiết")
    
    # TOPSIS bar chart
    fig_topsis = create_topsis_bar_chart(results)
    st.plotly_chart(fig_topsis, use_container_width=True)
    
    # Forecast chart
    fig_forecast = create_forecast_chart(
        analysis["historical"],
        analysis["forecast"],
        params["route"]
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Export buttons
    st.markdown("---")
    st.subheader("📥 Xuất báo cáo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        excel_data = generate_excel_report(
            results,
            analysis["data_adjusted"],
            weights
        )
        st.download_button(
            "📊 Tải xuống Excel",
            data=excel_data,
            file_name=f"riskcast_report_{params['route'].replace(' - ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col2:
        pdf_data = generate_pdf_report(
            results,
            params["route"],
            params["month"],
            params["method"],
            params["cargo_value"],
            params["priority"],
            analysis["var"],
            analysis["cvar"]
        )
        
        if pdf_data:
            st.download_button(
                "📄 Tải xuống PDF",
                data=pdf_data,
                file_name=f"riskcast_summary_{params['route'].replace(' - ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

def main():
    """Main application entry point"""
    
    # Page config
    st.set_page_config(
        page_title="RISKCAST v5.0 — ESG Risk Assessment",
        page_icon="📊",
        layout="wide"
    )
    
    # Apply styling
    apply_custom_css()
    
    # Initialize state
    initialize_session_state()
    
    # Title
    st.title("🚢 RISKCAST v5.0 — ESG Logistics Risk Assessment")
    st.markdown("**Advanced Decision Support System for Insurance Selection**")
    st.markdown("---")
    
    # Load data
    historical, claims = load_sample_data()
    
    # Sidebar inputs
    params = render_sidebar()
    
    # Weight controls
    render_weight_controls()
    
    # Display current weights as pie chart
    st.markdown("---")
    weights_series = pd.Series(st.session_state["weights"], index=CRITERIA)
    fig_current_weights = create_weights_pie_chart(
        weights_series,
        "📊 Phân bổ trọng số hiện tại"
    )
    st.plotly_chart(fig_current_weights, use_container_width=True)
    
    st.markdown("---")
    
    # Analysis button
    if st.button("🚀 PHÂN TÍCH & GỢI Ý", type="primary", use_container_width=True):
        with st.spinner("🔄 Đang thực hiện phân tích..."):
            try:
                analysis = run_analysis(params, historical)
                display_results(analysis, params)
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình phân tích: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
