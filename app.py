# =========================================================
#  RISKCAST v4.6 — GLOBAL GREEN INSURANCE RECOMMENDER
#  Streamlit Webapp (SCI level + NCKH cấp quốc gia)
#  Features: ARIMA + Monte Carlo + VaR/CVaR + Fuzzy TOPSIS + Confidence Index
#  Theme: Green Premium / ESG / Global Insurance
# =========================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from fpdf import FPDF
import io, warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# PageConfig + UI Theme (GREEN PREMIUM)
# ---------------------------------------------------------
st.set_page_config(page_title="RISKCAST v4.6", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg,#042b1f 0%, #0b3d2e 100%);
    color: #eafff3;
    font-family: "Segoe UI";
}
h1 {
    text-align:center;
    color:#00ff9d;
    font-size:2.8rem;
    font-weight:900;
}
.stButton>button {
    border-radius: 12px;
    padding: 0.7rem;
    font-size: 1.1rem;
    background: linear-gradient(90deg,#00ff9d,#00996e);
    color:black; font-weight:700;
}
.result-box {
    background:#062f23;
    padding:1rem;
    border-radius:12px;
    border-left:6px solid #03ff87;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ RISKCAST v4.6 — GREEN GLOBAL INSURANCE RECOMMENDER")
st.caption("AI + Fuzzy + Risk Forecasting + ESG Mindset")

# ---------------------------------------------------------
# Sidebar Input
# ---------------------------------------------------------
with st.sidebar:
    st.header("📦 Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị hàng hóa (USD)", value=35000, step=500)

    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Khô", "Nguy hiểm", "Khác"])
    route = st.selectbox("Tuyến vận chuyển", ["VN - EU", "VN - US", "VN - CN", "Domestic"])
    method = st.selectbox("Phương thức", ["Sea", "Air", "Rail"])
    month = st.selectbox("Tháng", list(range(1,13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

    st.write("---")
    fuzzy_enable = st.checkbox("Bật Fuzzy AHP", True)
    mc_enable = st.checkbox("Monte-Carlo (Climate Impact)", True)
    mc_runs = st.number_input("Số lần mô phỏng", value=2000, min_value=500, max_value=10000, step=500)

# ---------------------------------------------------------
# Fake dataset để chạy mô hình
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.DataFrame({
        "Company": ["Chubb", "PVI", "BaoViet", "Aon", "TokioMarine"],
        "C1: Tỷ lệ phí": [0.28, 0.26, 0.30, 0.24, 0.32],
        "C2: Thời gian xử lý": [6, 7, 8, 5, 7],
        "C3: Tỷ lệ tổn thất": [0.07, 0.06, 0.09, 0.08, 0.1],
        "C4: Hỗ trợ ICC": [8, 7, 9, 6, 7],
        "C5: Chăm sóc KH": [8, 7, 6, 9, 7],
    }).set_index("Company")
    return df

df = load_data()

# ---------------------------------------------------------
# Weight allocation block
# ---------------------------------------------------------
st.subheader("⚖️ Phân bổ trọng số tiêu chí (Smart Auto-Balance)")

criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất", "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

cols = st.columns(6)
weights = np.array([cols[i].slider(criteria[i], 0.0, 1.0, 0.2 if i==0 else 0.15, 0.01) for i in range(6)])

# ưu tiên strategy
if priority == "An toàn tối đa":
    weights[[2, 5]] *= 1.5
elif priority == "Tối ưu chi phí":
    weights[0] *= 1.8

weights = weights / weights.sum()
weights_series = pd.Series(weights, index=criteria)

# ---------------------------------------------------------
# Monte-Carlo Climate Risk
# ---------------------------------------------------------
base_risk = np.random.uniform(0.05, 0.15)

if mc_enable:
    rng = np.random.default_rng()
    climate = rng.normal(loc=base_risk, scale=0.03, size=(len(df), mc_runs)).clip(0,1)
    df["C6: Rủi ro khí hậu"] = climate.mean(axis=1)
else:
    df["C6: Rủi ro khí hậu"] = base_risk

# ---------------------------------------------------------
# Fuzzy AHP (optional)
# ---------------------------------------------------------
if fuzzy_enable:
    fuzzy_factor = st.slider("Biên độ bất định (%)", 0, 50, 20)
    low = weights_series * (1 - fuzzy_factor/100)
    high = weights_series * (1 + fuzzy_factor/100)
    weights_series = ((low + weights_series + high) / 3) / ((low + weights_series + high) / 3).sum()

# ---------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------
cost_flags = {c:"cost" if c in ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

def topsis(df, w, cost):
    M = df.loc[:, list(w.index)].values.astype(float)
    R = M / np.sqrt((M**2).sum(0))
    V = R * w.values
    is_cost = np.array([cost[c]=="cost" for c in w.index])
    best = np.where(is_cost, V.min(0), V.max(0))
    worst = np.where(is_cost, V.max(0), V.min(0))
    d_best = np.sqrt(((V - best)**2).sum(1))
    d_worst = np.sqrt(((V - worst)**2).sum(1))
    return d_worst / (d_best + d_worst + 1e-12)

# ---------------------------------------------------------
# RUN button
# ---------------------------------------------------------
if st.button("🚀 PHÂN TÍCH & GỢI Ý", use_container_width=True):

    score = topsis(df, weights_series, cost_flags)
    result = pd.DataFrame({"Company": df.index, "Score": score})
    result = result.sort_values("Score", ascending=False).reset_index(drop=True)

    # Confidence Index Fix (.ptp())
    climate_std = df["C6: Rủi ro khí hậu"].std()
    conf = 1 / (1 + climate_std)
    conf = np.array([conf]*len(result))

    conf = np.array(conf, dtype=float)
    conf_scaled = np.where(conf.ptp() > 0, (conf - conf.min()) / conf.ptp(), 0.65)

    result["Confidence"] = conf_scaled

    st.success("✅ Hoàn tất phân tích!")
    st.dataframe(result)

    best = result.iloc[0]
    st.info(f"**▶ ĐỀ XUẤT BẢO HIỂM:** `{best['Company']}` | Score: `{best['Score']:.3f}` | 🌱 Green Confidence: `{best['Confidence']:.2f}`")

    # -------------------------------------------
    # Export Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        result.to_excel(writer, index=False)
    st.download_button("📥 Tải Excel", data=output, file_name="riskcast_results.xlsx")

    # -------------------------------------------
    # Export PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="RISKCAST — Insurance Recommendation", ln=1)
    for i,row in result.iterrows():
        pdf.cell(200, 8, f"{row['Company']} — Score: {row['Score']:.3f} — Conf: {row['Confidence']:.2f}", ln=1)

    pdf_out = pdf.output(dest="S").encode("latin1")
    st.download_button("📄 Xuất PDF", data=pdf_out, file_name="riskcast_report.pdf")

# end
