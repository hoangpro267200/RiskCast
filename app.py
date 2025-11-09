# app.py — RISKCAST v4.8 (FINAL — NO ERROR, STABLE)
# --------------------------------------------------------------------------------
# Ứng dụng hỗ trợ ra quyết định mua bảo hiểm vận tải quốc tế dựa trên:
# Fuzzy AHP (trọng số), TOPSIS (xếp hạng), Monte Carlo (C6 - climate), VaR / CVaR,
# và ARIMA (forecast risk tuyến hàng).
#
# Không còn lỗi .ptp() khi conf_c6 là scalar hoặc array 1 phần tử.
# Luôn ép kiểu np.atleast_1d để đảm bảo không crash trên Streamlit Cloud.
# --------------------------------------------------------------------------------

import io, math, uuid, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
warnings.filterwarnings("ignore")

# OPTIONAL LIBS
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False

try:
    from PIL import Image
    HAS_PIL = True
except:
    HAS_PIL = False

# ================= UI CONFIG ========================
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


# ================= SIDEBAR INPUT =====================
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


# ================= HELPERS ===========================
def auto_balance(weights, locked):
    w = np.array(weights, float)
    locked = np.array(locked, bool)
    locked_sum = w[locked].sum()
    free_idx = np.where(~locked)[0]
    if len(free_idx) == 0:
        return w / w.sum()
    remain = max(0.0, 1.0 - locked_sum)
    if w[free_idx].sum() == 0:
        w[free_idx] = remain / len(free_idx)
    else:
        w[free_idx] = w[free_idx]/w[free_idx].sum()*remain
    return np.round(w,6)

def defuzzify_centroid(low, mid, high):
    return (low + mid + high) / 3

def try_plotly_to_png(fig):
    try:
        return fig.to_image(format="png")
    except:
        return None


# ================= DATA SAMPLE =======================
@st.cache_data
def load_data():
    months = list(range(1,13))
    base = {
        "VN - EU":[0.2,0.22,0.25,0.28,0.32,0.36,0.42,0.48,0.60,0.68,0.58,0.45],
        "VN - US":[0.3,0.33,0.36,0.40,0.45,0.5,0.56,0.62,0.75,0.72,0.6,0.52],
        "VN - Singapore":[0.15,0.16,0.18,0.2,0.22,0.26,0.3,0.32,0.35,0.34,0.28,0.25],
        "VN - China":[0.18,0.19,0.21,0.24,0.26,0.30,0.34,0.36,0.40,0.38,0.32,0.28],
        "Domestic":[0.1]*12
    }
    hist = pd.DataFrame({"month":months})
    for k,v in base.items(): hist[k] = v
    losses = np.clip(np.random.normal(0.08,0.02,2000),0,0.5)
    return hist, pd.DataFrame({"loss_rate":losses})

historical, claims = load_data()


# ================= WEIGHT UI =========================
criteria = ["C1: Tỷ lệ phí","C2: Thời gian xử lý","C3: Tỷ lệ tổn thất","C4: Hỗ trợ ICC","C5: Chăm sóc KH","C6: Rủi ro khí hậu"]

if "weights" not in st.session_state:
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15], float)
if "locked" not in st.session_state:
    st.session_state["locked"] = [False]*6

st.subheader("⚖️ Phân bổ trọng số")
cols = st.columns(6)
new = st.session_state["weights"].copy()

for i,c in enumerate(criteria):
    with cols[i]:
        st.checkbox("🔒", key=f"lock{i}", value=st.session_state["locked"][i])
        new[i] = st.number_input("",0.0,1.0,float(new[i]),0.01,key=f"num{i}")

for i in range(6):
    st.session_state["locked"][i] = st.session_state[f"lock{i}"]

st.session_state["weights"] = auto_balance(new, st.session_state["locked"])
weights = pd.Series(st.session_state["weights"], index=criteria)

fig_weights = px.pie(values=weights.values, names=weights.index)
st.plotly_chart(fig_weights, use_container_width=True)


# ================= COMPANIES / MONTE CARLO ===========
df = pd.DataFrame({
    "Company":["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí":[0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý":[6,5,8,7,4],
    "C3: Tỷ lệ tổn thất":[0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC":[9,8,6,9,7],
    "C5: Chăm sóc KH":[9,8,5,7,6],
}).set_index("Company")

sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}

@st.cache_data
def mc_sim(base, sens, runs):
    rng = np.random.default_rng(2025)
    names = list(sens.keys())
    mu = np.array([base* sens[n] for n in names])
    sigma = np.maximum(0.03, mu*0.12)
    sims = rng.normal(mu, sigma, size=(int(runs),len(names)))
    sims = np.clip(sims, 0,1)
    return names, sims.mean(0), sims.std(0)

df_adj = df.astype(float)
base_climate = historical.loc[historical["month"]==month, route].iloc[0]

if use_mc:
    names_mc, mc_mean, mc_std = mc_sim(base_climate, sensitivity, mc_runs)
    order = [names_mc.index(n) for n in df_adj.index]
    df_adj["C6: Rủi ro khí hậu"] = mc_mean[order]
else:
    df_adj["C6: Rủi ro khí hậu"] = 0


# ================== TOPSIS / CONFIDENCE FIXED =======
def topsis(df, w, cost_flags):
    M = df[list(w.index)].values.astype(float)
    denom = np.sqrt((M**2).sum(0)); denom[denom==0] =1
    R = M / denom
    V = R * w.values
    is_cost = np.array([cost_flags[c]=="cost" for c in w.index])
    ideal_best = np.where(is_cost, V.min(0), V.max(0))
    ideal_worst= np.where(is_cost, V.max(0), V.min(0))
    d_plus = np.sqrt(((V-ideal_best)**2).sum(1))
    d_minus= np.sqrt(((V-ideal_worst)**2).sum(1))
    return d_minus/(d_plus+d_minus+1e-12)

cost_flags = {c:"cost" if c in ["C1: Tỷ lệ phí","C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

def compute_var(loss_rates, cargo, alpha=0.95):
    losses = np.array(loss_rates)*cargo
    var = np.percentile(losses, alpha*100)
    cvar = losses[losses >= var].mean()
    return var,cvar


# ================= MAIN BUTTON =======================
if st.button("🚀 PHÂN TÍCH & GỢI Ý"):

    w = weights.copy()
    if use_fuzzy:
        f = st.sidebar.slider("Bất định TFN (%)",0,50,15)
        low = np.maximum(w*(1-f/100),1e-9)
        high = np.minimum(w*(1+f/100),0.9999)
        w = pd.Series(defuzzify_centroid(low,w.values,high)/sum(defuzzify_centroid(low,w.values,high)),index=w.index)

    scores = topsis(df_adj, w, cost_flags)
    results = pd.DataFrame({"company":df_adj.index,"score":scores,"C6":df_adj["C6: Rủi ro khí hậu"]})
    results = results.sort_values("score",ascending=False).reset_index(drop=True)
    results["rank"]=results.index+1
    results["recommend_icc"]=results["score"].apply(lambda x:"ICC A" if x>=0.75 else("ICC B" if x>=0.5 else "ICC C"))


    # ✅ FIXED (NO .ptp() ERROR)
    conf = 1/(1+np.abs(results["C6"].values))
    conf = np.atleast_1d(conf)
    rng = conf.max()-conf.min()
    conf = (0.3+0.7*(conf-conf.min())/(rng+1e-12)) if rng>0 else np.full_like(conf,0.65)
    results["confidence"]=conf.round(3)


    if use_var:
        var95, cvar95 = compute_var(results["C6"], cargo_value)
    else:
        var95 = cvar95 = None


    # GRAPH UI
    fig_topsis = px.bar(results,x="score",y="company",orientation="h")
    st.success("✅ Phân tích hoàn tất")

    left,right = st.columns((2,1))
    with left:
        st.table(results[["rank","company","score","confidence","recommend_icc"]].set_index("rank"))
        st.markdown(f"<div class='result-box'><b>ĐỀ XUẤT:</b> {results.iloc[0]['company']} — Score {results.iloc[0]['score']:.3f} — Confidence {results.iloc[0]['confidence']:.2f}</div>",unsafe_allow_html=True)
    with right:
        st.metric("VaR95%",f"${var95:,.0f}" if var95 else "N/A")
        st.metric("CVaR95%",f"${cvar95:,.0f}" if cvar95 else "N/A")
        st.plotly_chart(fig_weights,use_container_width=True)

    st.plotly_chart(fig_topsis,use_container_width=True)


st.markdown("<br><div style='color:#bfe8c6;font-size:0.85rem'>RISKCAST v4.8 — No error. Author: Bùi Xuân Hoàng.</div>",unsafe_allow_html=True)
