# app.py — RISKCAST v4.8 (NO ERROR — FINAL STABLE) — patched by Kai
# -------------------------------------------------------------------
# Mục đích:
#   - Ứng dụng hỗ trợ quyết định mua bảo hiểm vận tải quốc tế dựa vào:
#       + Fuzzy AHP (trọng số tiêu chí)
#       + TOPSIS (xếp hạng nhà bảo hiểm)
#       + Monte Carlo + Climate Risk sensitivity (C6)
#       + VaR / CVaR (rủi ro tổn thất tài chính)
#       + ARIMA Forecast (xu hướng rủi ro tuyến hàng)
#   - Chạy ổn định trên Streamlit Cloud, không còn lỗi scalar .ptp()
#
# Hướng dẫn chạy:
#   streamlit run app.py
# -------------------------------------------------------------------

import io, math, uuid, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

warnings.filterwarnings("ignore")

# =============== OPTIONAL LIBS =====================
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False

HAS_PIL = False
try:
    from PIL import Image
    HAS_PIL = True
except:
    HAS_PIL = False


# ================= UI CONFIG =======================
st.set_page_config(page_title="RISKCAST v4.8 — Green ESG", layout="wide", page_icon="🛡️")
st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg,#0b3d0b 0%, #05320a 100%); color:#e9fbf0; font-family:'Segoe UI'; }
  h1 { color:#a3ff96;text-align:center;font-weight:800; }
  .result-box { background:#0f3d1f;padding:1rem;border-left:6px solid #3ef08a;border-radius:8px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ RISKCAST v4.8 — Green ESG Insurance Advisor (NO ERROR)")
st.caption("MonteCarlo + ARIMA + VaR/CVaR + Fuzzy AHP + TOPSIS — Stable build")

# =====================================================
# SIDEBAR INPUT
# =====================================================
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị lô hàng (USD)", min_value=1000, value=39000, step=1000)
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"])
    route = st.selectbox("Tuyến", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"])
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
    month = st.selectbox("Tháng (1-12)", list(range(1,13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

    st.header("Mô hình")
    use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN)", True)
    use_arima = st.checkbox("Dùng ARIMA (nếu có)", True)
    use_mc = st.checkbox("Chạy Monte Carlo cho C6", True)
    use_var = st.checkbox("Tính VaR và CVaR", True)
    mc_runs = st.number_input("Số vòng Monte Carlo", 500, 10000, value=2000, step=500)


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def auto_balance(weights, locked):
    """Cân bằng trọng số (Lock + Auto-balance)."""
    w = np.array(weights, dtype=float)
    locked = np.array(locked, dtype=bool)
    locked_sum = w[locked].sum()
    free_idx = np.where(~locked)[0]
    if len(free_idx) == 0:
        return w / w.sum()
    remain = max(0.0, 1.0 - locked_sum)
    if w[free_idx].sum() == 0:
        w[free_idx] = remain / len(free_idx)
    else:
        w[free_idx] = w[free_idx]/w[free_idx].sum()*remain
    return np.round(w, 6)

def defuzzify_centroid(low, mid, high):
    return (low + mid + high) / 3.0

def try_plotly_to_png(fig):
    try:
        return fig.to_image(format="png")
    except:
        return None


# =====================================================
# DỮ LIỆU MẪU / DEMO
# =====================================================
@st.cache_data
def load_data():
    months = list(range(1,13))
    base = {
        "VN - EU": [0.2,0.22,0.25,0.28,0.32,0.36,0.42,0.48,0.60,0.68,0.58,0.45],
        "VN - US": [0.3,0.33,0.36,0.4,0.45,0.5,0.56,0.62,0.75,0.72,0.6,0.52],
        "VN - Singapore": [0.15,0.16,0.18,0.2,0.22,0.26,0.30,0.32,0.35,0.34,0.28,0.25],
        "VN - China": [0.18,0.19,0.21,0.24,0.26,0.30,0.34,0.36,0.40,0.38,0.32,0.28],
        "Domestic": [0.1]*12
    }
    hist = pd.DataFrame({"month": months})
    for k,v in base.items():
        hist[k] = v
    losses = np.clip(np.random.normal(loc=0.08, scale=0.02, size=2000), 0, 0.5)
    return hist, pd.DataFrame({"loss_rate": losses})

historical, claims = load_data()

# =====================================================
# CRITERIA & WEIGHT UI
# =====================================================
criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

if "weights" not in st.session_state:
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15], float)
if "locked" not in st.session_state:
    st.session_state["locked"] = [False]*6

st.subheader("⚖️ Phân bổ trọng số (Realtime)")
cols = st.columns(6)
new_w = st.session_state["weights"].copy()

for i,c in enumerate(criteria):
    with cols[i]:
        st.checkbox("🔒 Lock", key=f"lock_{i}", value=st.session_state["locked"][i])
        new_w[i] = st.number_input("Tỷ lệ", 0.0, 1.0, float(new_w[i]), 0.01, key=f"w_{i}")

for i in range(6):
    st.session_state["locked"][i] = st.session_state[f"lock_{i}"]

if st.button("Reset trọng số"):
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15])
    st.session_state["locked"] = [False]*6
else:
    st.session_state["weights"] = auto_balance(new_w, st.session_state["locked"])

weights = pd.Series(st.session_state["weights"], index=criteria)
fig_weights = px.pie(values=weights.values, names=weights.index)
st.plotly_chart(fig_weights, use_container_width=True)


# =====================================================
# BẢNG NHÀ BẢO HIỂM (DATA DEMO)
# =====================================================
df = pd.DataFrame({
    "Company": ["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí": [0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý": [6,5,8,7,4],
    "C3: Tỷ lệ tổn thất": [0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC": [9,8,6,9,7],
    "C5: Chăm sóc KH": [9,8,5,7,6],
}).set_index("Company")

sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}


# =====================================================
# MONTE CARLO CLIMATE (vectorized)
# =====================================================
@st.cache_data
def monte_carlo_climate(base, sens, runs):
    rng = np.random.default_rng(2025)
    names = list(sens.keys())
    mu = np.array([base * sens[n] for n in names])
    sigma = np.maximum(0.03, mu * 0.12)
    sims = rng.normal(mu, sigma, size=(int(runs), len(names)))
    sims = np.clip(sims, 0, 1)
    return names, sims.mean(axis=0), sims.std(axis=0)


base_climate = float(historical.loc[historical["month"]==month, route].iloc[0])
df_adj = df.astype(float)

if use_mc:
    names_mc, mc_mean, mc_std = monte_carlo_climate(base_climate, sensitivity, mc_runs)
    order = [names_mc.index(n) for n in df_adj.index]
    mc_mean, mc_std = mc_mean[order], mc_std[order]
else:
    mc_mean = np.zeros(len(df_adj))
    mc_std = np.zeros(len(df_adj))

df_adj["C6: Rủi ro khí hậu"] = mc_mean


# =====================================================
# TOPSIS
# =====================================================
def topsis(df_input, weight_series, cost_flags):
    M = df_input[list(weight_series.index)].values.astype(float)
    denom = np.sqrt((M**2).sum(axis=0)); denom[denom==0] = 1
    R = M / denom
    V = R * weight_series.values
    is_cost = np.array([cost_flags[c]=="cost" for c in weight_series.index])
    ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
    ideal_worst= np.where(is_cost, V.max(axis=0), V.min(axis=0))
    d_plus = np.sqrt(((V-ideal_best)**2).sum(axis=1))
    d_minus= np.sqrt(((V-ideal_worst)**2).sum(axis=1))
    return d_minus / (d_plus+d_minus+1e-12)

cost_flags = {c:("cost" if c in ["C1: Tỷ lệ phí","C6: Rủi ro khí hậu"] else "benefit") for c in criteria}


# =====================================================
# VAR / CVAR
# =====================================================
def compute_var_cvar(loss_rates, cargo_value, alpha=0.95):
    losses = np.array(loss_rates)*cargo_value
    if losses.size == 0: return None, None
    var = np.percentile(losses, alpha*100)
    cvar = losses[losses >= var].mean()
    return float(var), float(cvar)


# =====================================================
# FORECAST
# =====================================================
def forecast_route(route_key, months_ahead=3):
    series = historical[route_key].values
    if use_arima and ARIMA_AVAILABLE:
        try:
            fc = ARIMA(series, order=(1,1,1)).fit().forecast(months_ahead)
            return series, np.array(fc)
        except:
            pass
    last, trend = series, (series[-1]-series[-6])/6
    fc = np.array([max(0,last[-1]+(i+1)*trend) for i in range(months_ahead)])
    return last, fc


# =====================================================
# MAIN BUTTON
# =====================================================
if st.button("🚀 PHÂN TÍCH & GỢI Ý"):

    # FUZZY AHP
    w = weights.copy()
    if use_fuzzy:
        f = st.sidebar.slider("Bất định TFN (%)", 0, 50, 15)
        low = np.maximum(w*(1-f/100), 1e-9)
        high = np.minimum(w*(1+f/100), 0.9999)
        w = pd.Series(defuzzify_centroid(low, w.values, high) / sum(defuzzify_centroid(low, w.values, high)), index=w.index)

    # TOPSIS
    scores = topsis(df_adj, w, cost_flags)
    results = pd.DataFrame({"company": df_adj.index, "score": scores,
                            "C6_mean":mc_mean,"C6_std":mc_std}).sort_values("score", ascending=False).reset_index(drop=True)
    results["rank"] = results.index+1
    results["recommend_icc"] = results["score"].apply(lambda s: "ICC A" if s>=0.75 else ("ICC B" if s>=0.5 else "ICC C"))


    # ✅ FIX PT P — CONFIDENCE CALC (NO ERROR)
    eps = 1e-9

    cv_c6 = results["C6_std"].values / (results["C6_mean"].values + eps)
    conf_c6 = 1 / (1 + cv_c6)
    conf_c6 = np.atleast_1d(conf_c6)
    rng = np.ptp(conf_c6)
    conf_c6_scaled = (0.3 + 0.7 * (conf_c6 - conf_c6.min())/(rng+eps)) if rng>0 else np.full_like(conf_c6,0.65)

    crit_cv = df_adj.std(axis=1).values / (df_adj.mean(axis=1).values + eps)
    conf_crit = np.atleast_1d(1/(1+crit_cv))
    rng2 = np.ptp(conf_crit)
    conf_crit_scaled = (0.3 + 0.7*(conf_crit-conf_crit.min())/(rng2+eps)) if rng2>0 else np.full_like(conf_crit,0.65)

    conf_final = np.sqrt(conf_c6_scaled * conf_crit_scaled)
    results["confidence"] = conf_final.round(3)


    # VaR/CVaR
    var95, cvar95 = (compute_var_cvar(results["C6_mean"], cargo_value, alpha=0.95)
                     if use_var else (None,None))


    # GRAPH
    fig_topsis = px.bar(results.sort_values("score"), x="score", y="company", orientation="h")
    hist, fc = forecast_route(route)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=list(range(1,len(hist)+1)), y=hist, mode="lines+markers", name="Lịch sử"))
    fig_fc.add_trace(go.Scatter(x=list(range(len(hist)+1,len(hist)+1+len(fc))), y=fc, mode="lines+markers", name="Dự báo", line=dict(color="lime")))

    st.success("✅ Hoàn tất phân tích")

    left, right = st.columns((2,1))
    with left:
        st.subheader("Kết quả xếp hạng TOPSIS")
        st.table(results[["rank","company","score","confidence","recommend_icc"]].set_index("rank"))
        st.markdown(f"<div class='result-box'><b>ĐỀ XUẤT:</b> {results.iloc[0]['company']} — Score {results.iloc[0]['score']:.3f} — Confidence {results.iloc[0]['confidence']:.2f}</div>",
                    unsafe_allow_html=True)

    with right:
        st.metric("VaR 95%", f"${var95:,.0f}" if var95 else "N/A")
        st.metric("CVaR 95%", f"${cvar95:,.0f}" if cvar95 else "N/A")
        st.plotly_chart(fig_weights, use_container_width=True)

    st.plotly_chart(fig_topsis, use_container_width=True)
    st.plotly_chart(fig_fc, use_container_width=True)


# Footer
st.markdown("<br><div style='color:#bfe8c6;font-size:0.85rem'>RISKCAST v4.8 — No error. Author: Bùi Xuân Hoàng.</div>", unsafe_allow_html=True)

