# =============================================================
# RISKCAST v4.9 — ESG Logistics Dashboard (PREMIUM BLUE UI)
# Author: Bùi Xuân Hoàng — UI + Chart Clarity Enhanced by Kai
# =============================================================
import io
import uuid
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import os
import base64
warnings.filterwarnings("ignore")

# Optional libs
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# ---------------- PREMIUM UI (BLUE INSURANCE STYLE) ----------
st.set_page_config(page_title="RISKCAST v4.9 — ESG Insurance", layout="wide")

st.markdown("""
<style>
    /* ====== FIXED CLEAR TEXT STYLING ====== */
    * {
        opacity: 1 !important;
        text-shadow: 0 0 1px rgba(0,0,0,0) !important;
        -webkit-font-smoothing: antialiased !important;
    }
    
    /* ====== BACKGROUND APP ====== */
    .stApp {
        background: linear-gradient(135deg, #d9e9ff 0%, #f4fbff 100%) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        color: #003060 !important;
    }
    
    /* ====== MAIN CONTENT CONTAINER — CRYSTAL CLEAR ====== */
    .block-container {
        background: rgba(255,255,255,0.98) !important;
        backdrop-filter: blur(4px);
        padding: 2.5rem !important;
        border-radius: 16px;
        box-shadow: 0px 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.8);
        opacity: 1 !important;
    }
    
    /* ====== TITLES ====== */
    h1, h2, h3 {
        color: #2A6FDB !important;
        font-weight: 700 !important;
        text-shadow: none !important;
        letter-spacing: -0.02em !important;
    }
    
    /* ====== TEXT ELEMENTS ====== */
    .stMarkdown, .stText, .stLabel, .stSelectbox, .stNumberInput, .stSlider {
        color: #003060 !important;
        font-weight: 500 !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }
    
    /* ====== BUTTON ====== */
    .stButton > button {
        background: linear-gradient(135deg, #2A6FDB 0%, #1e57b2 100%) !important;
        color: #ffffff !important;
        border-radius: 10px;
        padding: 10px 24px;
        border: none;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(42, 111, 219, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e57b2 0%, #164a9e 100%) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(42, 111, 219, 0.4);
    }
    
    /* ====== RESULT BOX ====== */
    .result-box {
        background: linear-gradient(90deg, #2A6FDB, #1e57b2);
        color: white !important;
        padding: 18px 24px;
        border-radius: 12px;
        font-weight: 700;
        text-align: center;
        font-size: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 16px rgba(42, 111, 219, 0.3);
        border: 2px solid rgba(255,255,255,0.3);
        opacity: 1 !important;
    }
    
    /* ====== TABLE / DATAFRAME ====== */
    .stDataFrame, .stTable {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #003060 !important;
        background: white !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* ====== SIDEBAR ====== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%) !important;
        border-right: 3px solid #e6edf7;
    }
    section[data-testid="stSidebar"] * {
        color: #003060 !important;
        font-weight: 600 !important;
    }
    
    /* ====== METRIC CARDS ====== */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #003060 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- ENHANCED PLOTLY CONFIGURATION -----------------
def enhance_fig(fig, title=None, font_size=16, title_size=22):
    """Enhanced figure configuration for maximum clarity"""
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="Segoe UI, Arial, sans-serif", 
            size=font_size, 
            color="#003060"
        ),
        title=dict(
            text=title or fig.layout.title.text,
            font=dict(
                size=title_size, 
                family="Segoe UI, Arial, sans-serif", 
                color="#2A6FDB",
                weight="bold"
            ),
            x=0.5, 
            xanchor="center",
            y=0.95
        ),
        legend=dict(
            font=dict(size=font_size-1, color="#003060"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#e6edf7",
            borderwidth=2,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=100, b=80),
        hoverlabel=dict(
            font_size=font_size,
            bgcolor="white",
            bordercolor="#2A6FDB"
        )
    )
    fig.update_xaxes(
        title_font=dict(size=font_size+1, weight="bold"),
        tickfont=dict(size=font_size, color="#003060"),
        gridcolor="#f0f0f0",
        linecolor="#e6edf7",
        linewidth=2
    )
    fig.update_yaxes(
        title_font=dict(size=font_size+1, weight="bold"),
        tickfont=dict(size=font_size, color="#003060"),
        gridcolor="#f0f0f0",
        linecolor="#e6edf7",
        linewidth=2
    )
    return fig

def fig_to_png_bytes(fig, width=1400, height=600, scale=3):
    """Convert Plotly fig to high-quality PNG bytes"""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale, engine="kaleido")
    except Exception as e:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.write_image(tmp.name, width=width, height=height, scale=scale)
                with open(tmp.name, "rb") as f:
                    data = f.read()
                os.unlink(tmp.name)
                return data
        except Exception:
            st.error(f"Image export error: {e}")
            return None

# ================= Sidebar inputs =================
with st.sidebar:
    st.header("📊 Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị lô hàng (USD)", min_value=1000, value=39000, step=1000, key="sid_cargo_value")
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"], key="sid_good_type")
    route = st.selectbox("Tuyến", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"], key="sid_route")
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"], key="sid_method")
    month = st.selectbox("Tháng (1-12)", list(range(1, 13)), index=8, key="sid_month")
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"], key="sid_priority")
    
    st.markdown("---")
    st.header("⚙️ Mô hình")
    use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN)", True, key="sid_use_fuzzy")
    use_arima = st.checkbox("Dùng ARIMA (nếu có)", True, key="sid_use_arima")
    use_mc = st.checkbox("Chạy Monte Carlo cho C6", True, key="sid_use_mc")
    use_var = st.checkbox("Tính VaR và CVaR", True, key="sid_use_var")
    mc_runs = st.number_input("Số vòng Monte Carlo", 500, 10000, value=2000, step=500, key="sid_mc_runs")

# ================= Helper functions =================
def auto_balance(weights, locked):
    w = np.array(weights, dtype=float)
    locked_flags = np.array(locked, dtype=bool)
    total_locked = w[locked_flags].sum()
    free_idx = np.where(~locked_flags)[0]
    
    if len(free_idx) == 0:
        return (w / w.sum()) if w.sum() != 0 else np.ones_like(w)/len(w)
    
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

def defuzzify_centroid(low, mid, high):
    return (low + mid + high) / 3.0

# ================= Sample data (demo) =================
@st.cache_data
def load_data():
    months = list(range(1, 13))
    base = {
        "VN - EU": [0.2, 0.22, 0.25, 0.28, 0.32, 0.36, 0.42, 0.48, 0.60, 0.68, 0.58, 0.45],
        "VN - US": [0.3, 0.33, 0.36, 0.4, 0.45, 0.5, 0.56, 0.62, 0.75, 0.72, 0.6, 0.52],
        "VN - Singapore": [0.15, 0.16, 0.18, 0.2, 0.22, 0.26, 0.30, 0.32, 0.35, 0.34, 0.28, 0.25],
        "VN - China": [0.18, 0.19, 0.21, 0.24, 0.26, 0.30, 0.34, 0.36, 0.40, 0.38, 0.32, 0.28],
        "Domestic": [0.1] * 12
    }
    hist = pd.DataFrame({"month": months})
    for k, v in base.items():
        hist[k] = v
    
    rng = np.random.default_rng(123)
    losses = np.clip(rng.normal(loc=0.08, scale=0.02, size=2000), 0, 0.5)
    claims = pd.DataFrame({"loss_rate": losses})
    return hist, claims

historical, claims = load_data()

# ================= Criteria & initial weights =================
criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

if "weights" not in st.session_state:
    st.session_state["weights"] = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15], dtype=float)

if "locked" not in st.session_state:
    st.session_state["locked"] = [False] * len(criteria)

st.subheader("🎯 Phân bổ trọng số (Realtime)")
cols = st.columns(len(criteria))
new_w = st.session_state["weights"].copy()

for i, c in enumerate(criteria):
    key_lock = f"lock_{i}_v9"
    key_w = f"w_{i}_v9"
    with cols[i]:
        st.markdown(f"**{c}**")
        st.checkbox("🔒 Lock", value=st.session_state["locked"][i], key=key_lock)
        val = st.number_input("Tỉ lệ", min_value=0.0, max_value=1.0, 
                            value=float(new_w[i]), step=0.01, key=key_w,
                            label_visibility="collapsed")
        new_w[i] = val

for i in range(len(criteria)):
    st.session_state["locked"][i] = st.session_state.get(f"lock_{i}_v9", False)

if st.button("🔄 Reset trọng số mặc định", key="reset_weights_v9"):
    st.session_state["weights"] = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15], dtype=float)
    st.session_state["locked"] = [False] * len(criteria)
    st.rerun()
else:
    st.session_state["weights"] = auto_balance(new_w, st.session_state["locked"])

weights = pd.Series(st.session_state["weights"], index=criteria)

# Biểu đồ Pie Weights (Realtime) - RÕ CHỮ
fig_weights = px.pie(
    values=weights.values,
    names=weights.index,
    title="📊 Phân bổ trọng số (Realtime)",
    color_discrete_sequence=px.colors.sequential.Blues_r
)
fig_weights = enhance_fig(fig_weights, title="📊 Phân bổ trọng số (Realtime)", 
                         font_size=14, title_size=18)

# Hiển thị bằng PNG chất lượng cao
png_weights = fig_to_png_bytes(fig_weights, width=800, height=500, scale=3)
if png_weights:
    st.image(png_weights, use_container_width=True)
else:
    st.plotly_chart(fig_weights, use_container_width=True, key="fig_weights_main")

# ================= Insurance companies demo =================
df = pd.DataFrame({
    "Company": ["Chubb", "PVI", "InternationalIns", "BaoViet", "Aon"],
    "C1: Tỷ lệ phí": [0.30, 0.28, 0.26, 0.32, 0.24],
    "C2: Thời gian xử lý": [6, 5, 8, 7, 4],
    "C3: Tỷ lệ tổn thất": [0.08, 0.06, 0.09, 0.10, 0.07],
    "C4: Hỗ trợ ICC": [9, 8, 6, 9, 7],
    "C5: Chăm sóc KH": [9, 8, 5, 7, 6],
}).set_index("Company")

sensitivity = {"Chubb": 0.95, "PVI": 1.10, "InternationalIns": 1.20, "BaoViet": 1.05, "Aon": 0.90}
route_key = route
base_climate = float(historical.loc[historical["month"] == month, route_key].iloc[0]) if month in historical['month'].values else 0.4

df_adj = df.copy().astype(float)

# ---------------- Monte Carlo vectorized ----------------
@st.cache_data
def monte_carlo_climate(base, sens_map, runs):
    rng = np.random.default_rng(2025)
    names = list(sens_map.keys())
    mu = np.array([base * sens_map[n] for n in names], dtype=float)
    sigma = np.maximum(0.03, mu * 0.12)
    sims = rng.normal(loc=mu, scale=sigma, size=(int(runs), len(names)))
    sims = np.clip(sims, 0.0, 1.0)
    return names, sims.mean(axis=0), sims.std(axis=0)

if use_mc:
    names_mc, mc_mean, mc_std = monte_carlo_climate(base_climate, sensitivity, mc_runs)
    order = [names_mc.index(n) for n in df_adj.index]
    mc_mean = mc_mean[order]
    mc_std = mc_std[order]
else:
    mc_mean = np.zeros(len(df_adj), dtype=float)
    mc_std = np.zeros(len(df_adj), dtype=float)

df_adj["C6: Rủi ro khí hậu"] = mc_mean

if cargo_value > 50000:
    df_adj["C1: Tỷ lệ phí"] *= 1.1

# ================= TOPSIS =================
def topsis(df_input, weight_series, cost_flags):
    M = df_input[list(weight_series.index)].values.astype(float)
    denom = np.sqrt((M**2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = M / denom
    w = np.array(weight_series.values, dtype=float)
    V = R * w
    
    is_cost = np.array([cost_flags[c] == "cost" for c in weight_series.index])
    ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
    ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))
    
    d_plus = np.sqrt(((V - ideal_best)**2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst)**2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)
    return score

cost_flags = {c: ("cost" if c in ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất", "C6: Rủi ro khí hậu"] else "benefit") for c in criteria}

# ================= VaR / CVaR =================
def compute_var_cvar(loss_rates, cargo_value, alpha=0.95):
    losses = np.array(loss_rates, dtype=float) * float(cargo_value)
    if losses.size == 0:
        return None, None
    var = np.percentile(losses, alpha * 100)
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size > 0 else float(var)
    return float(var), float(cvar)

# ================= Forecast (ARIMA fallback) - FIXED =================
def forecast_route(route_key, months_ahead=3):
    """Fixed forecast function - ensures valid month ranges"""
    if route_key not in historical.columns:
        route_key = historical.columns[1]  # fallback to first available route
    
    series = historical[route_key].values
    
    # Ensure we don't forecast beyond reasonable bounds
    if use_arima and ARIMA_AVAILABLE and len(series) >= 6:
        try:
            model = ARIMA(series, order=(1,1,1)).fit()
            fc = model.forecast(months_ahead)
            fc = np.clip(fc, 0, 1)  # Ensure values are between 0-1
            return np.asarray(series), np.asarray(fc)
        except Exception:
            pass
    
    # Simple fallback with bounds
    last = np.array(series)
    trend = (last[-1] - last[-3]) / 3.0 if len(last) >= 3 else 0.0
    fc = np.array([max(0, min(1, last[-1] + (i+1)*trend)) for i in range(months_ahead)])
    return last, fc

# ================= Main action =================
if st.button("🚀 PHÂN TÍCH & GỢI Ý", key="run_analysis_v9"):
    with st.spinner("🔄 Đang chạy mô phỏng và tối ưu..."):
        w = pd.Series(st.session_state["weights"], index=criteria)
        
        if use_fuzzy:
            f = st.sidebar.slider("Bất định TFN (%)", 0, 50, 15, key="sid_tfn_v9")
            low = np.maximum(w * (1 - f/100.0), 1e-9)
            high = np.minimum(w * (1 + f/100.0), 0.9999)
            defuz = defuzzify_centroid(low, w.values, high)
            w = pd.Series(defuz / defuz.sum(), index=w.index)
        
        scores = topsis(df_adj, w, cost_flags)
        
        results = pd.DataFrame({
            "company": df_adj.index,
            "score": scores,
            "C6_mean": mc_mean,
            "C6_std": mc_std
        }).sort_values("score", ascending=False).reset_index(drop=True)
        
        results["rank"] = results.index + 1
        results["recommend_icc"] = results["score"].apply(
            lambda s: "ICC A" if s >= 0.75 else ("ICC B" if s >= 0.5 else "ICC C")
        )
        
        eps = 1e-9
        cv_c6 = np.where(results["C6_mean"].values == 0, 0.0, 
                        results["C6_std"].values / (results["C6_mean"].values + eps))
        conf_c6 = 1.0 / (1.0 + cv_c6)
        conf_c6 = np.atleast_1d(conf_c6)
        rng = np.ptp(conf_c6)
        conf_c6_scaled = (0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (rng + eps)) if rng > 0 else np.full_like(conf_c6, 0.65)
        
        crit_cv = df_adj.std(axis=1).values / (df_adj.mean(axis=1).values + eps)
        conf_crit = np.atleast_1d(1.0 / (1.0 + crit_cv))
        rng2 = np.ptp(conf_crit)
        conf_crit_scaled = (0.3 + 0.7 * (conf_crit - conf_crit.min()) / (rng2 + eps)) if rng2 > 0 else np.full_like(conf_crit, 0.65)
        
        conf_final = np.sqrt(conf_c6_scaled * conf_crit_scaled)
        order_map = {comp: float(conf_final[i]) for i, comp in enumerate(df_adj.index)}
        results["confidence"] = results["company"].map(order_map).round(3)
        
        var95, cvar95 = (compute_var_cvar(results["C6_mean"].values, cargo_value, alpha=0.95) 
                        if use_var else (None, None))
        
        # FIXED: Proper forecast with valid month ranges
        hist_series, fc = forecast_route(route)
        months_hist = list(range(1, len(hist_series) + 1))
        months_fc = list(range(len(hist_series) + 1, len(hist_series) + 1 + len(fc)))
        
        # Ensure forecast months don't exceed 12
        months_fc = [min(m, 12) for m in months_fc]

        # Biểu đồ TOPSIS - CHỮ TO, ĐẬM, RÕ
        fig_topsis = px.bar(
            results.sort_values("score"),
            x="score", y="company", orientation="h",
            title="🏆 TOPSIS Score (cao hơn = tốt hơn)",
            text="score", color="score",
            color_continuous_scale="Blues_r"
        )
        fig_topsis.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            textfont=dict(size=18, color="black", family="Arial Black", weight="bold"),
            marker_line_width=2,
            marker_line_color="white"
        )
        fig_topsis = enhance_fig(fig_topsis, font_size=16, title_size=22)

        # Biểu đồ Forecast - FIXED
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=months_hist, y=hist_series,
            mode="lines+markers", name="📈 Lịch sử",
            line=dict(color="#2A6FDB", width=4),
            marker=dict(size=10, symbol="circle")
        ))
        fig_fc.add_trace(go.Scatter(
            x=months_fc, y=fc,
            mode="lines+markers", name="🔮 Dự báo",
            line=dict(color="#FF6B6B", width=4, dash="dash"),
            marker=dict(size=11, symbol="diamond")
        ))
        fig_fc = enhance_fig(fig_fc, title=f"📊 Dự báo rủi ro khí hậu: {route}", 
                           font_size=15, title_size=20)
        fig_fc.update_xaxes(
            title="Tháng", 
            tickmode='linear',
            tickvals=list(range(1, 13)),  # Fixed: Only show valid months 1-12
            tickfont=dict(size=15)
        )
        fig_fc.update_yaxes(
            title="Mức rủi ro (0-1)", 
            range=[0, 1], 
            tickfont=dict(size=15)
        )

        # Pie chart right
        fig_weights_right = px.pie(
            values=w.values, names=w.index,
            title="⚖️ Trọng số cuối cùng",
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        fig_weights_right = enhance_fig(fig_weights_right, title="⚖️ Trọng số cuối cùng", 
                                      font_size=13, title_size=16)

        st.success("✅ Hoàn tất phân tích")
        
        left, right = st.columns((2, 1))
        
        with left:
            st.subheader("🏅 Kết quả xếp hạng TOPSIS")
            display_results = results[["rank", "company", "score", "confidence", "recommend_icc"]].set_index("rank")
            st.dataframe(display_results, use_container_width=True)
            
            top_company = results.iloc[0]
            st.markdown(
                f"<div class='result-box'>"
                f"<b>🎯 ĐỀ XUẤT TỐT NHẤT:</b> {top_company['company']} "
                f"— Score: {top_company['score']:.3f} "
                f"— Confidence: {top_company['confidence']:.2f} "
                f"— {top_company['recommend_icc']}"
                f"</div>", 
                unsafe_allow_html=True
            )
        
        with right:
            if var95 and cvar95:
                st.metric("💰 VaR 95%", f"${var95:,.0f}", delta="Risk Exposure")
                st.metric("🛡️ CVaR 95%", f"${cvar95:,.0f}", delta="Expected Shortfall")
            else:
                st.info("📊 VaR/CVaR chưa được tính")
            
            png_right = fig_to_png_bytes(fig_weights_right, width=600, height=400, scale=3)
            if png_right:
                st.image(png_right, use_container_width=True)
            else:
                st.plotly_chart(fig_weights_right, use_container_width=True, key="fig_weights_right_v9")

        # Hiển thị biểu đồ chính bằng PNG
        st.subheader("📈 Biểu đồ Phân tích")
        
        png_topsis = fig_to_png_bytes(fig_topsis, width=1400, height=600, scale=3)
        if png_topsis:
            st.image(png_topsis, use_container_width=True)
        else:
            st.plotly_chart(fig_topsis, use_container_width=True, key="fig_topsis_v9")
        
        png_fc = fig_to_png_bytes(fig_fc, width=1400, height=600, scale=3)
        if png_fc:
            st.image(png_fc, use_container_width=True)
        else:
            st.plotly_chart(fig_fc, use_container_width=True, key="fig_fc_v9")

        # ---------------- Export Excel ----------------
        excel_out = io.BytesIO()
        with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Result", index=False)
            df_adj.to_excel(writer, sheet_name="Adjusted_Data")
            pd.DataFrame({"weight": w.values}, index=w.index).to_excel(writer, sheet_name="Weights")
        excel_out.seek(0)
        
        st.download_button(
            "📥 Xuất Excel (Kết quả)", 
            excel_out, 
            file_name="riskcast_result.xlsx", 
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
            key="dl_excel_v9"
        )

        # ---------------- Export PDF - FIXED & SIMPLIFIED ----------------
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            
            # Title
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RISKCAST v4.9 - Executive Summary", 0, 1, "C")
            pdf.ln(10)
            
            # Basic info
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 8, f"Route: {route}", 0, 1)
            pdf.cell(0, 8, f"Month: {month} | Method: {method}", 0, 1)
            pdf.cell(0, 8, f"Cargo value: ${cargo_value:,}", 0, 1)
            pdf.cell(0, 8, f"Priority: {priority}", 0, 1)
            pdf.ln(10)
            
            # Recommendation
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "TOP RECOMMENDATION:", 0, 1)
            pdf.set_font("Arial", "", 12)
            top_rec = f"{results.iloc[0]['company']} (Score: {results.iloc[0]['score']:.3f}, Confidence: {results.iloc[0]['confidence']:.2f})"
            pdf.cell(0, 8, top_rec, 0, 1)
            pdf.ln(10)
            
            # Results table
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Ranking Results:", 0, 1)
            
            pdf.set_font("Arial", "B", 10)
            pdf.cell(15, 8, "Rank", 1)
            pdf.cell(50, 8, "Company", 1)
            pdf.cell(25, 8, "Score", 1)
            pdf.cell(25, 8, "Confidence", 1)
            pdf.cell(30, 8, "ICC Level", 1)
            pdf.ln()
            
            pdf.set_font("Arial", "", 10)
            for idx, row in results.iterrows():
                pdf.cell(15, 8, str(int(row["rank"])), 1)
                pdf.cell(50, 8, str(row["company"])[:20], 1)
                pdf.cell(25, 8, f"{row['score']:.3f}", 1)
                pdf.cell(25, 8, f"{row['confidence']:.2f}", 1)
                pdf.cell(30, 8, str(row["recommend_icc"]), 1)
                pdf.ln()
            
            # Risk metrics
            if var95 and cvar95:
                pdf.ln(10)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Risk Metrics:", 0, 1)
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 8, f"VaR 95%: ${var95:,.0f}", 0, 1)
                pdf.cell(0, 8, f"CVaR 95%: ${cvar95:,.0f}", 0, 1)
            
            pdf_output = pdf.output(dest='S').encode('latin1')
            
            st.download_button(
                "📄 Xuất PDF báo cáo",
                data=pdf_output,
                file_name="RISKCAST_Report.pdf",
                mime="application/pdf",
                key="dl_pdf_v9"
            )
            
        except Exception as e:
            st.error(f"Lỗi tạo PDF: {e}")

# ================= Footer =================
