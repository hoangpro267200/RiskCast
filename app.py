# =============================================================================
# RISKCAST v5.0 — ESG Logistics Risk Assessment Dashboard (OPTIMIZED)
# Refactored by Claude - Clean Architecture & Performance
# Original Author: Bùi Xuân Hoàng
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
    """Criterion optimization type"""
    COST = "cost"
    BENEFIT = "benefit"

@dataclass
class AnalysisParams:
    """Analysis parameters container"""
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
    """Analysis results container"""
    results: pd.DataFrame
    weights: pd.Series
    data_adjusted: pd.DataFrame
    var: Optional[float]
    cvar: Optional[float]
    historical: np.ndarray
    forecast: np.ndarray

# Constants
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

# Simplified color palette
COLORS = {
    "primary": "#0066CC",
    "secondary": "#FFB800",
    "accent": "#FF6B35",
    "text": "#1A1A1A",
    "bg": "#FFFFFF"
}

# =============================================================================
# UI STYLING (OPTIMIZED)
# =============================================================================

def apply_custom_css() -> None:
    """Streamlined CSS with essential styling only"""
    st.markdown("""
    <style>
        /* Global Settings - High Contrast */
        * {
            text-rendering: optimizeLegibility !important;
            -webkit-font-smoothing: antialiased !important;
        }
        
        .stApp {
            background: #FFFFFF !important;
            font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
        }
        
        .block-container {
            background: #FFFFFF !important;
            padding: 2rem 2.5rem !important;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
            max-width: 1400px;
            margin: 1.5rem auto;
        }
        
        /* Typography - HIGH CONTRAST */
        h1 { 
            color: #0052A3 !important; 
            font-weight: 900 !important; 
            font-size: 2.8rem !important;
            text-shadow: none !important;
        }
        h2 { 
            color: #000000 !important; 
            font-weight: 800 !important; 
            font-size: 2rem !important;
        }
        h3 { 
            color: #1A1A1A !important; 
            font-weight: 700 !important;
            font-size: 1.5rem !important;
        }
        
        /* Text - Maximum Contrast */
        p, span, div, label, .stMarkdown {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        /* Buttons - High Contrast */
        .stButton > button {
            background: #0052A3 !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            padding: 0.85rem 2.5rem !important;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
            transition: all 0.2s !important;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2) !important;
            border: 2px solid #0052A3 !important;
        }
        
        .stButton > button:hover {
            background: #003D7A !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3) !important;
        }
        
        /* Result Box - Maximum Visibility */
        .result-box {
            background: linear-gradient(135deg, #FFB800, #FF9800);
            color: #000000 !important;
            padding: 2rem 2.5rem;
            border-radius: 12px;
            font-weight: 800 !important;
            font-size: 1.3rem !important;
            text-align: center;
            margin: 1.5rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.25);
            border: 3px solid #FF9800;
        }
        
        /* Tables - High Contrast */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            border: 2px solid #E0E0E0 !important;
        }
        
        .stDataFrame thead tr th {
            background-color: #0052A3 !important;
            color: #FFFFFF !important;
            font-weight: 800 !important;
            font-size: 1.1rem !important;
        }
        
        .stDataFrame tbody tr td {
            color: #000000 !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
        }
        
        /* Sidebar - High Contrast */
        section[data-testid="stSidebar"] {
            background: #F5F5F5 !important;
            border-right: 3px solid #0052A3;
        }
        
        section[data-testid="stSidebar"] h2 {
            color: #000000 !important;
            font-weight: 900 !important;
        }
        
        section[data-testid="stSidebar"] label {
            color: #000000 !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
        }
        
        /* Metrics - Bold */
        [data-testid="stMetricValue"] {
            color: #0052A3 !important;
            font-weight: 900 !important;
            font-size: 2.5rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #000000 !important;
            font-weight: 800 !important;
            font-size: 1.2rem !important;
        }
        
        /* Explanations Box */
        .explanation-box {
            background: #F0F7FF;
            border-left: 5px solid #0052A3;
            padding: 1.5rem;
            margin: 1.5rem 0;
            border-radius: 8px;
            color: #000000 !important;
            font-weight: 600 !important;
        }
        
        .explanation-box h4 {
            color: #0052A3 !important;
            font-weight: 800 !important;
            margin-bottom: 1rem !important;
        }
        
        .explanation-box ul {
            color: #000000 !important;
        }
        
        .explanation-box li {
            margin: 0.8rem 0;
            color: #000000 !important;
            font-weight: 600 !important;
            line-height: 1.6;
        }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# DATA LAYER
# =============================================================================

class DataService:
    """Centralized data management"""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_historical_data() -> pd.DataFrame:
        """Generate historical climate risk data"""
        climate_base = {
            "VN - EU": [0.20, 0.22, 0.25, 0.28, 0.32, 0.36, 0.42, 0.48, 0.60, 0.68, 0.58, 0.45],
            "VN - US": [0.30, 0.33, 0.36, 0.40, 0.45, 0.50, 0.56, 0.62, 0.75, 0.72, 0.60, 0.52],
            "VN - Singapore": [0.15, 0.16, 0.18, 0.20, 0.22, 0.26, 0.30, 0.32, 0.35, 0.34, 0.28, 0.25],
            "VN - China": [0.18, 0.19, 0.21, 0.24, 0.26, 0.30, 0.34, 0.36, 0.40, 0.38, 0.32, 0.28],
            "Domestic": [0.10] * 12
        }
        
        df = pd.DataFrame({"month": list(range(1, 13))})
        for route, values in climate_base.items():
            df[route] = values
        return df
    
    @staticmethod
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
# CORE ALGORITHMS (OPTIMIZED)
# =============================================================================

class WeightManager:
    """Weight balancing and management"""
    
    @staticmethod
    def auto_balance(weights: np.ndarray, locked: List[bool]) -> np.ndarray:
        """Auto-balance weights to sum to 1.0"""
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
    """Fuzzy Analytical Hierarchy Process"""
    
    @staticmethod
    def apply(weights: pd.Series, uncertainty_pct: float) -> pd.Series:
        """Apply triangular fuzzy numbers"""
        factor = uncertainty_pct / 100.0
        w = weights.values
        
        low = np.maximum(w * (1 - factor), 1e-9)
        high = np.minimum(w * (1 + factor), 0.9999)
        
        # Defuzzify using centroid method
        defuzzified = (low + w + high) / 3.0
        normalized = defuzzified / defuzzified.sum()
        
        return pd.Series(normalized, index=weights.index)

class MonteCarloSimulator:
    """Vectorized Monte Carlo simulation"""
    
    @staticmethod
    @st.cache_data(ttl=600)
    def simulate(
        base_risk: float,
        sensitivity_map: Dict[str, float],
        n_simulations: int
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Run Monte Carlo simulation for climate risk"""
        rng = np.random.default_rng(2025)
        companies = list(sensitivity_map.keys())
        
        mu = np.array([base_risk * sensitivity_map[c] for c in companies])
        sigma = np.maximum(0.03, mu * 0.12)
        
        simulations = rng.normal(loc=mu, scale=sigma, size=(n_simulations, len(companies)))
        simulations = np.clip(simulations, 0.0, 1.0)
        
        return companies, simulations.mean(axis=0), simulations.std(axis=0)

class TOPSISAnalyzer:
    """TOPSIS multi-criteria decision analysis"""
    
    @staticmethod
    def analyze(
        data: pd.DataFrame,
        weights: pd.Series,
        cost_benefit: Dict[str, CriterionType]
    ) -> np.ndarray:
        """Execute TOPSIS algorithm"""
        M = data[list(weights.index)].values.astype(float)
        
        # Normalize
        denom = np.sqrt((M ** 2).sum(axis=0))
        denom[denom == 0] = 1.0
        R = M / denom
        
        # Apply weights
        V = R * weights.values
        
        # Ideal solutions
        is_cost = np.array([cost_benefit[c] == CriterionType.COST for c in weights.index])
        ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
        ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))
        
        # Calculate distances
        d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
        d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
        
        # Relative closeness
        return d_minus / (d_plus + d_minus + 1e-12)

class RiskCalculator:
    """Risk metrics calculations"""
    
    @staticmethod
    def calculate_var_cvar(
        loss_rates: np.ndarray,
        cargo_value: float,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate VaR and CVaR"""
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
        data: pd.DataFrame
    ) -> np.ndarray:
        """Calculate confidence scores"""
        eps = 1e-9
        
        # C6 confidence
        cv_c6 = results["C6_std"].values / (results["C6_mean"].values + eps)
        conf_c6 = 1.0 / (1.0 + cv_c6)
        conf_c6 = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + eps)
        
        # Criteria confidence
        crit_cv = data.std(axis=1).values / (data.mean(axis=1).values + eps)
        conf_crit = 1.0 / (1.0 + crit_cv)
        conf_crit = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (np.ptp(conf_crit) + eps)
        
        return np.sqrt(conf_c6 * conf_crit)

class Forecaster:
    """Climate risk forecasting"""
    
    @staticmethod
    def forecast(
        historical: pd.DataFrame,
        route: str,
        months_ahead: int = 3,
        use_arima: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forecast future climate risk"""
        if route not in historical.columns:
            route = historical.columns[1]
        
        series = historical[route].values
        
        # Try ARIMA
        if use_arima and ARIMA_AVAILABLE and len(series) >= 6:
            try:
                model = ARIMA(series, order=(1, 1, 1))
                fitted = model.fit()
                forecast = np.clip(fitted.forecast(months_ahead), 0, 1)
                return series, np.asarray(forecast)
            except Exception:
                pass
        
        # Fallback to trend
        trend = (series[-1] - series[-3]) / 3.0 if len(series) >= 3 else 0.0
        forecast = np.array([
            np.clip(series[-1] + (i + 1) * trend, 0, 1)
            for i in range(months_ahead)
        ])
        
        return series, forecast

# =============================================================================
# VISUALIZATION (OPTIMIZED)
# =============================================================================

class ChartFactory:
    """Centralized chart creation"""
    
    @staticmethod
    def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
        """Apply consistent theme to figures with high contrast"""
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=f"<b>{title}</b>", 
                font=dict(size=22, color="#000000", family="Arial Black, Arial"), 
                x=0.5
            ),
            font=dict(size=16, color="#000000", family="Arial, sans-serif", weight=700),
            margin=dict(l=80, r=80, t=100, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(
                font=dict(size=16, color="#000000", weight="bold"),
                bgcolor="white",
                bordercolor="#000000",
                borderwidth=2
            )
        )
        
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            linecolor="#000000",
            linewidth=2,
            tickfont=dict(size=16, color="#000000", family="Arial", weight="bold"),
            title_font=dict(size=18, color="#000000", weight="bold")
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            linecolor="#000000",
            linewidth=2,
            tickfont=dict(size=16, color="#000000", family="Arial", weight="bold"),
            title_font=dict(size=18, color="#000000", weight="bold")
        )
        
        return fig
    
    @staticmethod
    def create_weights_pie(weights: pd.Series, title: str) -> go.Figure:
        """Create weight distribution pie chart with bold labels"""
        colors = ['#0052A3', '#FF9800', '#00C853', '#FF5252', '#00BCD4', '#9C27B0']
        
        fig = go.Figure(data=[go.Pie(
            labels=weights.index,
            values=weights.values,
            marker=dict(colors=colors, line=dict(color='#000000', width=3)),
            textfont=dict(size=18, color="#000000", family="Arial Black", weight="bold"),
            textposition='outside',
            textinfo='label+percent',
            insidetextorientation='radial',
            pull=[0.05] * len(weights)  # Slight pull for better label visibility
        )])
        
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                font=dict(size=22, color="#000000", family="Arial Black"),
                x=0.5
            ),
            font=dict(size=16, color="#000000", weight="bold"),
            showlegend=True,
            legend=dict(
                font=dict(size=16, color="#000000", weight="bold"),
                bgcolor="white",
                bordercolor="#000000",
                borderwidth=2
            )
        )
        
        return fig
    
    @staticmethod
    def create_topsis_bar(results: pd.DataFrame) -> go.Figure:
        """Create TOPSIS score bar chart with bold text"""
        fig = go.Figure(data=[go.Bar(
            x=results.sort_values("score")["score"],
            y=results.sort_values("score")["company"],
            orientation="h",
            text=results.sort_values("score")["score"].apply(lambda x: f"<b>{x:.3f}</b>"),
            textposition="outside",
            textfont=dict(size=20, color="#000000", family="Arial Black", weight="bold"),
            marker=dict(
                color=results.sort_values("score")["score"],
                colorscale=[[0, '#90CAF9'], [0.5, '#0052A3'], [1, '#003D7A']],
                line=dict(color='#000000', width=2)
            ),
            hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
        )])
        
        fig.update_xaxes(
            title="<b>TOPSIS Score</b>", 
            range=[0, 1],
            tickfont=dict(size=18, color="#000000", weight="bold")
        )
        fig.update_yaxes(
            title="<b>Công ty</b>",
            tickfont=dict(size=18, color="#000000", weight="bold")
        )
        
        return ChartFactory._apply_theme(fig, "🏆 TOPSIS Score (cao hơn = tốt hơn)")
    
    @staticmethod
    def create_forecast_chart(
        historical: np.ndarray,
        forecast: np.ndarray,
        route: str
    ) -> go.Figure:
        """Create forecast line chart with bold labels"""
        fig = go.Figure()
        
        months_hist = list(range(1, len(historical) + 1))
        months_fc = [min(m, 12) for m in range(len(historical) + 1, len(historical) + len(forecast) + 1)]
        
        fig.add_trace(go.Scatter(
            x=months_hist, 
            y=historical, 
            mode="lines+markers+text", 
            name="📈 Lịch sử",
            line=dict(color="#0052A3", width=4),
            marker=dict(size=12, color="#0052A3", line=dict(width=3, color='white')),
            text=[f"{val:.1%}" for val in historical],
            textposition="top center",
            textfont=dict(size=14, color="#000000", weight="bold"),
            hovertemplate='<b>Tháng %{x}</b><br>Rủi ro: %{y:.2%}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=months_fc, 
            y=forecast, 
            mode="lines+markers+text", 
            name="🔮 Dự báo",
            line=dict(color="#FF5252", width=4, dash="dash"),
            marker=dict(size=14, color="#FF5252", symbol="diamond", line=dict(width=3, color='white')),
            text=[f"{val:.1%}" for val in forecast],
            textposition="top center",
            textfont=dict(size=14, color="#000000", weight="bold"),
            hovertemplate='<b>Tháng %{x}</b><br>Dự báo: %{y:.2%}<extra></extra>'
        ))
        
        fig = ChartFactory._apply_theme(fig, f"📊 Dự báo rủi ro khí hậu: {route}")
        
        fig.update_xaxes(
            title="<b>Tháng</b>",
            tickmode="linear",
            tickvals=list(range(1, 13)),
            dtick=1
        )
        
        fig.update_yaxes(
            title="<b>Mức rủi ro</b>",
            range=[0, max(1, max(historical.max(), forecast.max()) * 1.15)],
            tickformat='.0%'
        )
        
        return fig

# =============================================================================
# EXPORT UTILITIES (OPTIMIZED)
# =============================================================================

class ReportGenerator:
    """Report generation utilities"""
    
    @staticmethod
    def generate_pdf(
        results: pd.DataFrame,
        params: AnalysisParams,
        var: Optional[float],
        cvar: Optional[float]
    ) -> bytes:
        """Generate PDF report with simplified layout"""
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST v5.0 - Executive Summary", 0, 1, "C")
            pdf.ln(5)
            
            # Metadata
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"Route: {params.route} | Month: {params.month} | Method: {params.method}", 0, 1)
            pdf.cell(0, 6, f"Cargo Value: ${params.cargo_value:,.0f}", 0, 1)
            pdf.ln(5)
            
            # Top recommendation
            top = results.iloc[0]
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"Top Recommendation: {top['company']}", 0, 1)
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"Score: {top['score']:.3f} | Confidence: {top['confidence']:.2f}", 0, 1)
            pdf.ln(5)
            
            # Rankings table
            pdf.set_font("Arial", "B", 10)
            pdf.cell(20, 6, "Rank", 1)
            pdf.cell(60, 6, "Company", 1)
            pdf.cell(30, 6, "Score", 1)
            pdf.cell(30, 6, "ICC", 1, 1)
            
            pdf.set_font("Arial", "", 9)
            for _, row in results.head(5).iterrows():
                pdf.cell(20, 6, str(int(row["rank"])), 1)
                pdf.cell(60, 6, str(row["company"])[:25], 1)
                pdf.cell(30, 6, f"{row['score']:.3f}", 1)
                pdf.cell(30, 6, str(row["recommend_icc"]), 1, 1)
            
            # Risk metrics
            if var and cvar:
                pdf.ln(5)
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 6, f"VaR 95%: ${var:,.0f} | CVaR 95%: ${cvar:,.0f}", 0, 1)
            
            return pdf.output(dest='S').encode('latin1')
        except Exception as e:
            st.error(f"PDF generation error: {e}")
            return b""
    
    @staticmethod
    def generate_excel(
        results: pd.DataFrame,
        data: pd.DataFrame,
        weights: pd.Series
    ) -> bytes:
        """Generate Excel report"""
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
# APPLICATION CONTROLLER
# =============================================================================

class AnalysisController:
    """Main analysis orchestration"""
    
    def __init__(self):
        self.data_service = DataService()
        self.weight_manager = WeightManager()
        self.fuzzy_ahp = FuzzyAHP()
        self.mc_simulator = MonteCarloSimulator()
        self.topsis = TOPSISAnalyzer()
        self.risk_calc = RiskCalculator()
        self.forecaster = Forecaster()
    
    def run_analysis(self, params: AnalysisParams, historical: pd.DataFrame) -> AnalysisResult:
        """Execute complete analysis pipeline"""
        
        # Prepare weights
        weights = pd.Series(st.session_state["weights"], index=CRITERIA)
        if params.use_fuzzy:
            weights = self.fuzzy_ahp.apply(weights, params.fuzzy_uncertainty)
        
        # Get company data
        company_data = self.data_service.get_company_data()
        
        # Base climate risk
        base_risk = float(
            historical.loc[historical["month"] == params.month, params.route].iloc[0]
        ) if params.month in historical["month"].values else 0.4
        
        # Monte Carlo simulation
        if params.use_mc:
            companies, mc_mean, mc_std = self.mc_simulator.simulate(
                base_risk, SENSITIVITY_MAP, params.mc_runs
            )
            order = [companies.index(c) for c in company_data.index]
            mc_mean, mc_std = mc_mean[order], mc_std[order]
        else:
            mc_mean = mc_std = np.zeros(len(company_data))
        
        # Adjust data
        data_adjusted = company_data.copy()
        data_adjusted["C6: Rủi ro khí hậu"] = mc_mean
        
        if params.cargo_value > 50000:
            data_adjusted["C1: Tỷ lệ phí"] *= 1.1
        
        # TOPSIS analysis
        scores = self.topsis.analyze(data_adjusted, weights, COST_BENEFIT_MAP)
        
        # Prepare results
        results = pd.DataFrame({
            "company": data_adjusted.index,
            "score": scores,
            "C6_mean": mc_mean,
            "C6_std": mc_std
        }).sort_values("score", ascending=False).reset_index(drop=True)
        
        results["rank"] = results.index + 1
        results["recommend_icc"] = results["score"].apply(
            lambda s: "ICC A" if s >= 0.75 else ("ICC B" if s >= 0.5 else "ICC C")
        )
        
        # Confidence scores
        conf = self.risk_calc.calculate_confidence(results, data_adjusted)
        order_map = {comp: conf[i] for i, comp in enumerate(data_adjusted.index)}
        results["confidence"] = results["company"].map(order_map).round(3)
        
        # Risk metrics
        var = cvar = None
        if params.use_var:
            var, cvar = self.risk_calc.calculate_var_cvar(
                results["C6_mean"].values, params.cargo_value
            )
        
        # Forecast
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
            forecast=forecast
        )

# =============================================================================
# STREAMLIT UI
# =============================================================================

class StreamlitUI:
    """Streamlit interface management"""
    
    def __init__(self):
        self.controller = AnalysisController()
        self.chart_factory = ChartFactory()
        self.report_gen = ReportGenerator()
    
    def initialize(self):
        """Initialize session state and UI"""
        st.set_page_config(
            page_title="RISKCAST v5.0",
            page_icon="📊",
            layout="wide"
        )
        apply_custom_css()
        
        if "weights" not in st.session_state:
            st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
        if "locked" not in st.session_state:
            st.session_state["locked"] = [False] * len(CRITERIA)
    
    def render_sidebar(self) -> AnalysisParams:
        """Render sidebar controls"""
        with st.sidebar:
            st.header("📊 Thông tin lô hàng")
            
            cargo_value = st.number_input("Giá trị (USD)", 1000, value=39000, step=1000)
            good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Nguy hiểm", "Khác"])
            route = st.selectbox("Tuyến", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"])
            method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
            month = st.selectbox("Tháng", list(range(1, 13)), index=8)
            priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])
            
            st.markdown("---")
            st.header("⚙️ Cấu hình")
            
            use_fuzzy = st.checkbox("Fuzzy AHP", True)
            use_arima = st.checkbox("ARIMA", True)
            use_mc = st.checkbox("Monte Carlo", True)
            use_var = st.checkbox("VaR/CVaR", True)
            
            mc_runs = st.number_input("MC Runs", 500, 10000, 2000, 500)
            fuzzy_uncertainty = st.slider("Fuzzy (%)", 0, 50, 15) if use_fuzzy else 15
            
            return AnalysisParams(
                cargo_value, good_type, route, method, month, priority,
                use_fuzzy, use_arima, use_mc, use_var, mc_runs, fuzzy_uncertainty
            )
    
    def render_weight_controls(self):
        """Render weight adjustment UI with clear labels"""
        st.subheader("🎯 Phân bổ trọng số")
        
        cols = st.columns(len(CRITERIA))
        new_weights = st.session_state["weights"].copy()
        
        for i, criterion in enumerate(CRITERIA):
            with cols[i]:
                # Bold, clear criterion name
                st.markdown(f"<div style='background:#F0F7FF; padding:8px; border-radius:6px; border:2px solid #0052A3; margin-bottom:8px;'><b style='color:#000000; font-size:1.1rem;'>{criterion.split(':')[0]}</b></div>", unsafe_allow_html=True)
                
                is_locked = st.checkbox("🔒 Lock", value=st.session_state["locked"][i], key=f"lock_{i}")
                st.session_state["locked"][i] = is_locked
                
                weight_val = st.number_input(
                    "Tỉ lệ", 0.0, 1.0, float(new_weights[i]), 0.01,
                    key=f"weight_{i}", label_visibility="collapsed"
                )
                new_weights[i] = weight_val
                
                # Display percentage clearly
                st.markdown(f"<div style='text-align:center; color:#000000; font-weight:800; font-size:1.2rem;'>{weight_val:.1%}</div>", unsafe_allow_html=True)
        
        if st.button("🔄 Reset về mặc định", use_container_width=True):
            st.session_state["weights"] = DEFAULT_WEIGHTS.copy()
            st.session_state["locked"] = [False] * len(CRITERIA)
            st.rerun()
        else:
            st.session_state["weights"] = WeightManager.auto_balance(
                new_weights, st.session_state["locked"]
            )
    
    def display_results(self, result: AnalysisResult, params: AnalysisParams):
        """Display analysis results with detailed explanations"""
        st.success("✅ **Phân tích hoàn tất!**")
        
        # Main results
        left, right = st.columns([2, 1])
        
        with left:
            st.subheader("🏅 Kết quả xếp hạng")
            display_df = result.results[["rank", "company", "score", "confidence", "recommend_icc"]].set_index("rank")
            display_df.columns = ["Công ty", "Điểm số", "Độ tin cậy", "ICC"]
            st.dataframe(display_df, use_container_width=True)
            
            top = result.results.iloc[0]
            st.markdown(
                f"""<div class='result-box'>
                🏆 <b>KHUYẾN NGHỊ HÀNG ĐẦU</b><br><br>
                <span style='font-size:1.6rem; font-weight:900;'>{top['company']}</span><br><br>
                <span style='font-size:1.2rem;'>
                Score: <b>{top['score']:.3f}</b> | 
                Confidence: <b>{top['confidence']:.2f}</b> | 
                <b>{top['recommend_icc']}</b>
                </span>
                </div>""",
                unsafe_allow_html=True
            )
        
        with right:
            if result.var and result.cvar:
                st.metric("💰 VaR 95%", f"${result.var:,.0f}", 
                         help="Tổn thất tối đa với độ tin cậy 95%")
                st.metric("🛡️ CVaR 95%", f"${result.cvar:,.0f}",
                         help="Tổn thất trung bình vượt VaR")
            
            fig_weights = self.chart_factory.create_weights_pie(result.weights, "⚖️ Trọng số")
            st.plotly_chart(fig_weights, use_container_width=True)
        
        # DETAILED EXPLANATION SECTION
        st.markdown("---")
        st.subheader("📋 GIẢI THÍCH CHI TIẾT KẾT QUẢ")
        
        top_3 = result.results.head(3)
        
        # Why this ranking?
        st.markdown(f"""
        <div class='explanation-box'>
            <h4>🎯 Tại sao {top['company']} được khuyến nghị?</h4>
            <ul>
                <li><b>Điểm TOPSIS cao nhất ({top['score']:.3f}):</b> Công ty này cân bằng tốt nhất giữa tất cả tiêu chí đánh giá</li>
                <li><b>Độ tin cậy {top['confidence']:.2%}:</b> Mức độ ổn định và dự đoán được cao</li>
                <li><b>Khuyến nghị {top['recommend_icc']}:</b> Phù hợp với mức rủi ro của tuyến {params.route}</li>
                <li><b>Giá trị hàng hóa ${params.cargo_value:,}:</b> Đủ điều kiện cho mức bảo hiểm này</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison explanation
        comparison_text = f"""
        <div class='explanation-box'>
            <h4>📊 So sánh Top 3:</h4>
            <ul>
                <li><b>#{1} {top_3.iloc[0]['company']} (Score: {top_3.iloc[0]['score']:.3f}):</b> 
                    Cân bằng tốt nhất, rủi ro khí hậu thấp ({top_3.iloc[0]['C6_mean']:.2%})</li>
                <li><b>#{2} {top_3.iloc[1]['company']} (Score: {top_3.iloc[1]['score']:.3f}):</b> 
                    Kém {(top_3.iloc[0]['score'] - top_3.iloc[1]['score']):.3f} điểm, 
                    rủi ro khí hậu cao hơn ({top_3.iloc[1]['C6_mean']:.2%})</li>
                <li><b>#{3} {top_3.iloc[2]['company']} (Score: {top_3.iloc[2]['score']:.3f}):</b> 
                    Kém {(top_3.iloc[0]['score'] - top_3.iloc[2]['score']):.3f} điểm,
                    độ tin cậy thấp hơn ({top_3.iloc[2]['confidence']:.2f})</li>
            </ul>
        </div>
        """
        st.markdown(comparison_text, unsafe_allow_html=True)
        
        # Key factors explanation
        key_factors = result.data_adjusted.loc[top['company']]
        st.markdown(f"""
        <div class='explanation-box'>
            <h4>🔑 Các yếu tố quyết định cho {top['company']}:</h4>
            <ul>
                <li><b>Tỷ lệ phí:</b> {key_factors['C1: Tỷ lệ phí']:.2%} - {"Cạnh tranh" if key_factors['C1: Tỷ lệ phí'] < 0.30 else "Cao"}</li>
                <li><b>Thời gian xử lý:</b> {key_factors['C2: Thời gian xử lý']:.0f} ngày - {"Nhanh" if key_factors['C2: Thời gian xử lý'] < 6 else "Trung bình"}</li>
                <li><b>Tỷ lệ tổn thất:</b> {key_factors['C3: Tỷ lệ tổn thất']:.2%} - {"Tốt" if key_factors['C3: Tỷ lệ tổn thất'] < 0.08 else "Chấp nhận được"}</li>
                <li><b>Hỗ trợ ICC:</b> {key_factors['C4: Hỗ trợ ICC']:.0f}/10 - {"Xuất sắc" if key_factors['C4: Hỗ trợ ICC'] >= 8 else "Tốt"}</li>
                <li><b>Chăm sóc KH:</b> {key_factors['C5: Chăm sóc KH']:.0f}/10 - {"Xuất sắc" if key_factors['C5: Chăm sóc KH'] >= 8 else "Tốt"}</li>
                <li><b>Rủi ro khí hậu:</b> {top['C6_mean']:.2%} ± {top['C6_std']:.2%} - {"Thấp" if top['C6_mean'] < 0.30 else "Trung bình"}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk assessment
        if result.var and result.cvar:
            st.markdown(f"""
            <div class='explanation-box'>
                <h4>⚠️ Đánh giá rủi ro tài chính:</h4>
                <ul>
                    <li><b>VaR 95% = ${result.var:,.0f}:</b> Có 95% khả năng tổn thất không vượt quá mức này</li>
                    <li><b>CVaR 95% = ${result.cvar:,.0f}:</b> Nếu tổn thất vượt VaR, trung bình sẽ ở mức này</li>
                    <li><b>Tỷ lệ rủi ro:</b> {(result.var/params.cargo_value)*100:.1f}% giá trị hàng hóa</li>
                    <li><b>Khuyến nghị:</b> {"Chấp nhận được" if result.var/params.cargo_value < 0.10 else "Cần xem xét kỹ"}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            excel_data = self.report_gen.generate_excel(
                result.results, result.data_adjusted, result.weights
            )
            st.download_button(
                "📊 Tải Excel",
                data=excel_data,
                file_name=f"riskcast_{params.route.replace(' - ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col2:
            pdf_data = self.report_gen.generate_pdf(
                result.results, params, result.var, result.cvar
            )
            if pdf_data:
                st.download_button(
                    "📄 Tải PDF",
                    data=pdf_data,
                    file_name=f"riskcast_{params.route.replace(' - ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    def run(self):
        """Main application flow"""
        self.initialize()
        
        st.title("🚢 RISKCAST v5.0 — ESG Risk Assessment")
        st.markdown("**Advanced Decision Support System**")
        st.markdown("---")
        
        # Load data
        historical = DataService.load_historical_data()
        
        # Sidebar
        params = self.render_sidebar()
        
        # Weight controls
        self.render_weight_controls()
        
        # Current weights display
        st.markdown("---")
        weights_series = pd.Series(st.session_state["weights"], index=CRITERIA)
        fig_current = self.chart_factory.create_weights_pie(weights_series, "📊 Trọng số hiện tại")
        st.plotly_chart(fig_current, use_container_width=True)
        
        st.markdown("---")
        
        # Analysis button
        if st.button("🚀 PHÂN TÍCH & GỢI Ý", type="primary", use_container_width=True):
            with st.spinner("🔄 Đang phân tích..."):
                try:
                    result = self.controller.run_analysis(params, historical)
                    self.display_results(result, params)
                except Exception as e:
                    st.error(f"❌ Lỗi: {e}")
                    st.exception(e)

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Application entry point"""
    app = StreamlitUI()
    app.run()

if __name__ == "__main__":
    main()
