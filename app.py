# app.py — RISKCAST v4.6.1 — Green ESG Pro (Fixed & Optimized)
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# === Optional Imports with Fallback ===
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ---------------- Page Config + Modern Green ESG CSS ----------------
st.set_page_config(page_title="RISKCAST v4.6.1 — ESG Pro", layout="wide", page_icon="Green Leaf")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a2e0a 0%, #041a04 100%);
        color: #e8f5e9;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    .big-title { font-size: 2.8rem; font-weight: 800; color: #a8e6cf; text-align: center; margin-bottom: 0.5rem; }
    .subtitle { color: #81c784; text-align: center; font-weight: 500; margin-bottom: 1.5rem; }
    .card {
        background: rgba(20, 60, 20, 0.6);
        backdrop-filter: blur(10px);
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(100, 230, 100, 0.15);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin: 0.8rem 0;
    }
    .result-box {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 6px solid #00e676;
        color: #e8f5e9;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #1b5e20);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #388e3c, #2e7d32);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
    }
    .footer { text-align: center; color: #81c784; font-size: 0.8rem; margin-top: 3rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown('<h1 class="big-title">RISKCAST v4.6.1 — ESG Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fuzzy AHP • TOPSIS • Monte Carlo • ARIMA • VaR/CVaR • ESG Focus</p>', unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị lô hàng (USD)", min_value=1000, value=39000, step=1000, format="%d")
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"])
    route = st.selectbox("Tuyến", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"])
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
    month = st.selectbox("Tháng (1-12)", list(range(1,13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

    st.markdown("---")
    st.header("Mô hình")
    use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN)", True)
    use_arima = st.checkbox("Dùng ARIMA dự báo", True)
    use_var = st.checkbox("Tính VaR & CVaR", True)
    use_mc = st.checkbox("Monte Carlo cho C6", True)
    mc_runs = st.slider("Số vòng Monte Carlo", 500, 5000, 2000, step=500)

# ---------------- Helper Functions ----------------
@st.cache_data(show_spinner=False)
def load_sample_data():
    months = list(range(1,13))
    base = {
        "VN - EU": [0.20,0.22,0.25,0.28,0.32,0.36,0.42,0.48,0.60,0.68,0.58,0.45],
        "VN - US": [0.30,0.33,0.36,0.40,0.45,0.50,0.56,0.62,0.75,0.72,0.60,0.52],
        "VN - Singapore": [0.15,0.16,0.18,0.20,0.22,0.26,0.30,0.32,0.35,0.34,0.28,0.25],
        "Domestic": [0.10,0.10,0.10,0.12,0.12,0.14,0.16,0.18,0.20,0.18,0.14,0.12],
        "VN - China": [0.18,0.19,0.21,0.24,0.26,0.30,0.34,0.36,0.40,0.38,0.32,0.28],
    }
    hist = pd.DataFrame({"month": months, **base})
    rng = np.random.default_rng(42)
    losses = np.clip(rng.normal(0.08, 0.02, 2000), 0, 0.5)
    claims = pd.DataFrame({"loss_rate": losses})
    return hist, claims

historical, claims = load_sample_data()

def auto_balance(weights, locked):
    w = np.array(weights, dtype=float)
    locked = np.array(locked, dtype=bool)
    locked_sum = w[locked].sum()
    free_idx = np.where(~locked)[0]
    remaining = max(0.0, 1.0 - locked_sum)

    if len(free_idx) == 0:
        return np.round(w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w), 6)

    free_sum = w[free_idx].sum()
    if free_sum == 0:
        w[free_idx] = remaining / len(free_idx)
    else:
        w[free_idx] = w[free_idx] / free_sum * remaining

    w = np.clip(w, 0.0, 1.0)
    diff = 1.0 - w.sum()
    if abs(diff) > 1e-8 and len(free_idx) > 0:
        w[free_idx[0]] += diff
    return np.round(w, 6)

def defuzzify_centroid(l, m, u):
    return (l + m + u) / 3.0

def safe_plotly_to_png(fig, width=800, height=500):
    try:
        return fig.to_image(format="png", width=width, height=height)
    except:
        return None

# ---------------- Criteria & Weights ----------------
criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

if "weights" not in st.session_state:
    st.session_state.weights = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])
if "locked" not in st.session_state:
    st.session_state.locked = [False] * 6

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Phân bổ trọng số (Lock & Auto-balance)")

cols = st.columns(6)
new_w = st.session_state.weights.copy()

for i, c in enumerate(criteria):
    with cols[i]:
        st.markdown(f"**{c.split(':')[1].strip()}**")
        st.checkbox("Lock", key=f"lock_{i}", value=st.session_state.locked[i])
        val = st.number_input("w", min_value=0.0, max_value=1.0, value=float(new_w[i]), step=0.01, key=f"w_{i}", label_visibility="collapsed")
        new_w[i] = val

for i in range(6):
    st.session_state.locked[i] = st.session_state[f"lock_{i}"]

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Reset", use_container_width=True):
        st.session_state.weights = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])
        st.session_state.locked = [False] * 6
        st.success("Đã reset!")
with col2:
    pass

st.session_state.weights = auto_balance(new_w, st.session_state.locked)
weights_series = pd.Series(st.session_state.weights, index=criteria)

fig_pie = px.pie(values=weights_series, names=weights_series.index, color_discrete_sequence=px.colors.sequential.Greens)
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(fig_pie, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Company Data ----------------
df = pd.DataFrame({
    "Company": ["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí": [0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý": [6,5,8,7,4],
    "C3: Tỷ lệ tổn thất": [0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC": [9,8,6,9,7],
    "C5: Chăm sóc KH": [9,8,5,7,6],
}).set_index("Company")

sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}
base_climate = float(historical.loc[historical['month']==month, route].iloc[0])

# ---------------- Monte Carlo C6 ----------------
df_adj = df.copy().astype(float)
mc_mean = np.array([base_climate * sensitivity.get(c, 1.0) for c in df.index], dtype=float)

if use_mc:
    rng = np.random.default_rng(2025)
    mc_sims = np.zeros((mc_runs, len(df)))
    for i, mu in enumerate(mc_mean):
        sigma = max(0.03, mu * 0.12)
        sims = rng.normal(mu, sigma, mc_runs)
        sims = np.clip(sims, 0.0, 1.0)
        mc_sims[:, i] = sims
    mc_mean = mc_sims.mean(axis=0)
    mc_std = mc_sims.std(axis=0)
else:
    mc_std = np.zeros_like(mc_mean)

df_adj["C6: Rủi ro khí hậu"] = mc_mean

if cargo_value > 50000:
    df_adj["C1: Tỷ lệ phí"] *= 1.1

# ---------------- TOPSIS ----------------
def topsis(matrix, weights, cost_benefit):
    M = matrix.values.astype(float)
    norm = np.sqrt((M**2).sum(axis=0))
    norm[norm == 0] = 1
    R = M / norm
    V = R * weights.values
    is_cost = np.array([cost_benefit[c] == "cost" for c in weights.index])
    ideal_best = np.where(is_cost, V.min(0), V.max(0))
    ideal_worst = np.where(is_cost, V.max(0), V.min(0))
    d_plus = np.sqrt(((V - ideal_best)**2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst)**2).sum(axis=1))
    return d_minus / (d_plus + d_minus + 1e-12)

cost_flags = {c: "cost" if c in ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

# ---------------- VaR/CVaR ----------------
def compute_var_cvar(loss_rates, value, alpha=0.95):
    losses = np.array(loss_rates) * value
    var = np.percentile(losses, alpha * 100)
    cvar = losses[losses >= var].mean() if len(losses[losses >= var]) > 0 else var
    return var, cvar

# ---------------- Forecast ----------------
@st.cache_data(show_spinner=False)
def forecast_route_cached(_historical, route_key, months_ahead=3):
    series = _historical[route_key].values
    if use_arima and ARIMA_AVAILABLE:
        try:
            model = ARIMA(series, order=(1,1,1)).fit()
            fc = model.forecast(months_ahead)
            return series.tolist(), fc.tolist()
        except:
            pass
    last = np.array(series[-6:])
    trend = (series[-1] - series[-6]) / 6
    fc = [max(0, series[-1] + (i+1)*trend) for i in range(months_ahead)]
    return series.tolist(), fc

# ---------------- RUN ANALYSIS ----------------
if st.button("PHÂN TÍCH & GỢI Ý", use_container_width=True, type="primary"):
    progress = st.progress(0)
    status = st.empty()

    status.info("Đang chuẩn bị dữ liệu...")
    progress.progress(20)

    # Fuzzy AHP
    weights = weights_series.copy()
    if use_fuzzy:
        f = st.sidebar.slider("Độ bất định TFN (%)", 0, 50, 15, key="fuzzy_slider")
        low = np.maximum(weights * (1 - f/100), 1e-6)
        high = np.minimum(weights * (1 + f/100), 0.9999)
        defuz = defuzzify_centroid(low, weights, high)
        weights = pd.Series(defuz / defuz.sum(), index=weights.index)
    progress.progress(40)

    # TOPSIS
    status.info("Đang tính TOPSIS...")
    scores = topsis(df_adj[criteria], weights, cost_flags)
    progress.progress(60)

    # Results
    results = pd.DataFrame({
        "company": df_adj.index,
        "score": scores,
        "C6_mean": mc_mean,
        "C6_std": mc_std
    }).sort_values("score", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1
    results["recommend_icc"] = results["score"].apply(lambda x: "ICC A" if x >= 0.75 else ("ICC B" if x >= 0.5 else "ICC C"))

    # === CONFIDENCE - ĐÃ SỬA LỖI .ptp() ===
    cv_c6 = np.where(mc_mean == 0, 0.0, mc_std / mc_mean)
    conf_c6 = 1.0 / (1.0 + cv_c6)
    conf_c6 = np.array(conf_c6, dtype=float)
    if conf_c6.ptp() > 0:
        conf_c6_scaled = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (conf_c6.ptp() + 1e-9)
    else:
        conf_c6_scaled = np.full_like(conf_c6, 0.65)

    crit_cv = df_adj.std(axis=1).values / (df_adj.mean(axis=1).values + 1e-9)
    conf_crit = 1.0 / (1.0 + crit_cv)
    conf_crit = np.array(conf_crit, dtype=float)
    if conf_crit.ptp() > 0:
        conf_crit_scaled = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (conf_crit.ptp() + 1e-9)
    else:
        conf_crit_scaled = np.full_like(conf_crit, 0.65)

    conf_final = np.sqrt(conf_c6_scaled * conf_crit_scaled)
    order_map = {comp: conf_final[i] for i, comp in enumerate(df_adj.index)}
    results["confidence"] = results["company"].map(order_map).round(3)

    # VaR
    var95, cvar95 = compute_var_cvar(mc_mean, cargo_value) if use_var else (None, None)

    # Forecast
    status.info("Dự báo rủi ro...")
    hist_series, fc = forecast_route_cached(historical, route)
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=list(range(1,13)), y=hist_series, mode='lines+markers', name='Lịch sử', line=dict(color='#81c784')))
    fig_forecast.add_trace(go.Scatter(x=list(range(13,16)), y=fc, mode='lines+markers', name='Dự báo', line=dict(color='#00e676', dash='dot')))
    fig_forecast.update_layout(template="plotly_dark", title=f"Dự báo rủi ro: {route}", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    progress.progress(80)

    # TOPSIS Chart
    fig_topsis = px.bar(results, y="company", x="score", orientation='h', color="score",
                        color_continuous_scale="Greens", title="TOPSIS Ranking")
    fig_topsis.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    progress.progress(100)
    status.success("Hoàn tất!")

    # ---------------- Display Results ----------------
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Xếp hạng TOPSIS")
        st.dataframe(results[["rank","company","score","confidence","recommend_icc"]].set_index("rank").style.format({"score": "{:.3f}", "confidence": "{:.2f}"}), use_container_width=True)
        st.markdown(f"""
        <div class='result-box'>
            <strong>ĐỀ XUẤT:</strong> {results.iloc[0]['company']} — 
            Score: <strong>{results.iloc[0]['score']:.3f}</strong> — 
            ICC: <strong>{results.iloc[0]['recommend_icc']}</strong> — 
            Confidence: <strong>{results.iloc[0]['confidence']:.2f}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Tổng quan")
        st.metric("VaR 95%", f"${var95:,.0f}" if var95 else "N/A")
        st.metric("CVaR 95%", f"${cvar95:,.0f}" if cvar95 else "N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    st.plotly_chart(fig_topsis, use_container_width=True)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ---------------- Export ----------------
    col1, col2 = st.columns(2)
    with col1:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            results.to_excel(writer, sheet_name="Results", index=False)
            df_adj.to_excel(writer, sheet_name="Data")
            pd.DataFrame({"Criteria": criteria, "Weight": weights}).to_excel(writer, sheet_name="Weights", index=False)
        excel_buffer.seek(0)
        st.download_button("Xuất Excel", excel_buffer, "riskcast_v4.6.1.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col2:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "RISKCAST v4.6.1 - ESG Report", ln=1, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 8, f"Recommended: {results.iloc[0]['company']} | Score: {results.iloc[0]['score']:.3f} | VaR: ${var95:,.0f}" if var95 else "")
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("Xuất PDF", pdf_bytes, "riskcast_report.pdf", "application/pdf")

    progress.empty()
    status.empty()

# ---------------- Footer ----------------
st.markdown("""
<div class='footer'>
    <hr style='border: 1px solid #2e7d32; margin: 2rem 0;'>
    <strong>RISKCAST v4.6.1 — Green ESG Pro</strong> • Fixed .ptp() error • Author: Bùi Xuân Hoàng
</div>
""", unsafe_allow_html=True)
