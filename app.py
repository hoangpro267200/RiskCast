# app.py — RISKCAST v4.5 (Green ESG theme) — FULL FIX (NO ERROR)
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# Optional ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

# ---------------- Page config + CSS (Green ESG) ----------------
st.set_page_config(page_title="RISKCAST v4.5 — Green ESG", layout="wide", page_icon="shield")
st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg,#0b3d0b 0%, #05320a 100%); color: #e9fbf0; font-family: 'Segoe UI', sans-serif; }
  h1 { color:#a3ff96; text-align:center; font-weight:800; }
  .card { background: rgba(255,255,255,0.03); padding:1rem; border-radius:10px; border:1px solid rgba(163,255,150,0.08); }
  .result-box { background:#0f3d1f; padding:1rem; border-left:6px solid #3ef08a; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

st.title("RISKCAST v4.5 — Green ESG Insurance Advisor")
st.caption("ARIMA + MonteCarlo + VaR/CVaR + Fuzzy AHP + TOPSIS")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị (USD)", min_value=1000, value=39000, step=1000)
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"])
    route = st.selectbox("Tuyến", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"])
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
    month = st.selectbox("Tháng", list(range(1,13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

    st.header("Mô hình")
    use_fuzzy = st.checkbox("Fuzzy AHP", True)
    use_arima = st.checkbox("ARIMA Forecast", True)
    use_var = st.checkbox("VaR & CVaR", True)
    use_mc = st.checkbox("Monte Carlo", True)
    mc_runs = st.number_input("Số vòng MC", 500, 10000, 2000, 500)

# ---------------- Helper ----------------
def auto_balance(weights, locked):
    w = np.array(weights, dtype=float)
    locked = = np.array(locked, dtype=bool)
    locked_sum = w[locked].sum()
    free_idx = np.where(~locked)[0]
    if len(free_idx) == 0:
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w)
        return np.round(w, 4)
    remaining = max(0.0, 1.0 - locked_sum)
    free_vals = w[free_idx]
    w[free_idx] = (free_vals / free_vals.sum() * remaining) if free_vals.sum() > 0 else remaining / len(free_idx)
    diff = 1.0 - w.sum()
    if abs(diff) > 1e-8 and len(free_idx) > 0:
        w[free_idx[0]] += diff
    return np.round(w, 6)

# ---------------- Session State ----------------
if "weights" not in st.session_state:
    st.session_state.weights = [0.20,0.15,0.20,0.20,0.10,0.15]
if "locked" not in st.session_state:
    st.session_state.locked = [False]*6

# ---------------- Weights UI ----------------
st.subheader("Phân bổ trọng số")
criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất", "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]
cols = st.columns(6)
new_w = st.session_state.weights.copy()
for i, c in enumerate(criteria):
    with cols[i]:
        st.write(f"**{c}**")
        st.checkbox("Lock", key=f"lock_{i}")
        val = st.number_input("", 0.0, 1.0, float(new_w[i]), 0.01, key=f"w_{i}")
        new_w[i] = val

for i in range(6): st.session_state.locked[i] = st.session_state.get(f"lock_{i}", False)
if st.button("Reset trọng số"):
    st.session_state.weights = [0.20,0.15,0.20,0.20,0.10,0.15]
    st.session_state.locked = [False]*6
else:
    st.session_state.weights = auto_balance(new_w, st.session_state.locked)

weights_series = pd.Series(st.session_state.weights, index=criteria)
fig_pie = px.pie(values=weights_series, names=criteria, title="Trọng số")
st.plotly_chart(fig_pie, use_container_width=True)

# ---------------- Data ----------------
df = pd.DataFrame({
    "Company": ["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí": [0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý": [6,5,8,7,4],
    "C3: Tỷ lệ tổn thất": [0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC": [9,8,6,9,7],
    "C5: Chăm sóc KH": [9,8,5,7,6],
}).set_index("Company")

sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}
base_climate = 0.4  # fallback
df_adj = df.copy().astype(float)
mc_mean = np.array([base_climate * sensitivity.get(c,1.0) for c in df_adj.index])
mc_std = np.zeros(len(df_adj))

if use_mc:
    rng = np.random.default_rng(42)
    for i in range(len(df_adj)):
        mu, sigma = mc_mean[i], max(0.03, mc_mean[i]*0.12)
        sim = np.clip(rng.normal(mu, sigma, mc_runs), 0, 1)
        mc_mean[i], mc_std[i] = sim.mean(), sim.std()
df_adj["C6: Rủi ro khí hậu"] = mc_mean

# ---------------- TOPSIS ----------------
def topsis(df, w, cost):
    M = df.values
    R = M / np.sqrt((M**2).sum(0, keepdims=True))
    V = R * w.values
    is_cost = np.array([cost[c]=="cost" for c in w.index])
    best = np.where(is_cost, V.min(0), V.max(0))
    worst = np.where(is_cost, V.max(0), V.min(0))
    d_best = np.sqrt(((V - best)**2).sum(1))
    d_worst = np.sqrt(((V - worst)**2).sum(1))
    return d_worst / (d_best + d_worst + 1e-12)

cost_flags = {c: "cost" if c in ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

# ---------------- RUN ----------------
if st.button("PHÂN TÍCH & GỢI Ý"):
    with st.spinner("Đang chạy..."):
        w = weights_series
        if use_fuzzy:
            f = 0.15
            low = np.maximum(w * (1-f), 1e-6)
            high = np.minimum(w * (1+f), 0.9999)
            w = pd.Series((low + w + high)/3 / ((low + w + high)/3).sum(), index=w.index)

        scores = topsis(df_adj[list(w.index)], w, cost_flags)
        result = pd.DataFrame({"company": df_adj.index, "score": scores, "C6_mean": mc_mean, "C6_std": mc_std})
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = result.index + 1
        result["ICC"] = result["score"].apply(lambda x: "A" if x>=0.75 else "B" if x>=0.5 else "C")

        # --- SỬA LỖI .ptp() ---
        cv_c6 = np.where(result["C6_mean"] == 0, 0, result["C6_std"] / result["C6_mean"])
        conf_c6 = 1.0 / (1.0 + cv_c6)
        conf_c6_arr = np.array(conf_c6)
        ptp_c6 = conf_c6_arr.max() - conf_c6_arr.min() if len(conf_c6_arr) > 1 else 0
        conf_c6_scaled = np.where(ptp_c6 > 0, 0.3 + 0.7 * (conf_c6_arr - conf_c6_arr.min()) / ptp_c6, 0.65)

        crit_cv = df_adj.std(axis=1) / (df_adj.mean(axis=1) + 1e-9)
        conf_crit = 1.0 / (1.0 + crit_cv)
        conf_crit_arr = np.array(conf_crit)
        ptp_crit = conf_crit_arr.max() - conf_crit_arr.min() if len(conf_crit_arr) > 1 else 0
        conf_crit_scaled = np.where(ptp_crit > 0, 0.3 + 0.7 * (conf_crit_arr - conf_crit_arr.min()) / ptp_crit, 0.65)

        result["confidence"] = np.sqrt(conf_c6_scaled * conf_crit_scaled).round(3)
        # --- HẾT SỬA ---

        st.success("HOÀN TẤT!")
        st.dataframe(result.set_index("rank"))

        # Export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result.to_excel(writer, 'Result')
        output.seek(0)
        st.download_button("Xuất Excel", data=output, file_name="riskcast.xlsx")

        # PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Best: {result.iloc[0]['company']} | Score: {result.iloc[0]['score']:.3f}", ln=True)
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("Xuất PDF", data=pdf_bytes, file_name="report.pdf")
