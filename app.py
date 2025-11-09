# =======================================================================
# RISKCAST v3.4 — Smart Weights + Fuzzy AHP + Monte-Carlo + TOPSIS + PDF Export
# =======================================================================

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from fpdf import FPDF
import requests
import plotly.io as pio   # dùng để export chart vào ảnh (PDF)
pio.kaleido.scope.default_format = "png"

# --------------------------------------------------
# PAGE STYLE
# --------------------------------------------------
st.set_page_config(page_title="RISKCAST v3.4", layout="wide", page_icon="🛡️")
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg,#00101F 0%, #0E2A47 100%); color: #E7F4FF; }
    h1 { color: #7bd3ff; text-align: center; font-weight: 800; }
    .footer { text-align:center; margin-top: 3rem; color:#aaa; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ RISKCAST v3.4 — Hệ thống đề xuất bảo hiểm thông minh")


# --------------------------------------------------
# SIDEBAR INPUT
# --------------------------------------------------
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị lô hàng (USD)", value=39000, step=1000, format="%d")
    good_type = st.selectbox("Loại hàng", ["Điện tử","Đông lạnh","Hàng khô","Hàng nguy hiểm","Khác"])
    route = st.selectbox("Tuyến vận chuyển", ["VN - EU","VN - US","VN - Singapore","VN - China","Domestic"])
    method = st.selectbox("Phương thức vận tải", ["Sea","Air","Truck"])
    month = st.selectbox("Tháng vận chuyển", list(range(1,13)), index=8)
    priority = st.selectbox("Ưu tiên của bạn", ["An toàn tối đa","Cân bằng","Tối ưu chi phí"])
    use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN → defuzzify)", True)
    use_mc = st.checkbox("Bật Monte-Carlo rủi ro khí hậu", True)
    mc_runs = st.number_input("Số vòng Monte-Carlo", 200, 20000, 2000, 200)


# --------------------------------------------------
# CRITERIA
# --------------------------------------------------
criteria = [
    "C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"
]

criteria_tooltip = {
    "C1: Tỷ lệ phí": "Phí bảo hiểm — càng thấp càng tốt",
    "C2: Thời gian xử lý": "Tốc độ xử lý claim — càng nhanh càng tốt",
    "C3: Tỷ lệ tổn thất": "Tỷ lệ từ chối claim — càng thấp càng tốt",
    "C4: Hỗ trợ ICC": "Mức độ bao phủ ICC A/B/C — càng tốt càng an toàn",
    "C5: Chăm sóc KH": "Dịch vụ hỗ trợ claim — càng tốt càng an tâm",
    "C6: Rủi ro khí hậu": "Rủi ro thiên tai theo tuyến + mùa — càng thấp càng tốt",
}

cost_flags = {c: "cost" if c in ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu", "C3: Tỷ lệ tổn thất"] else "benefit" for c in criteria}


# --------------------------------------------------
# SMART SLIDER (Auto normalize + lock + input + reset)
# --------------------------------------------------
st.subheader("🎚️ Phân bổ trọng số (Smart slider + Lock + Reset)")

if "weights" not in st.session_state:
    st.session_state.weights = np.array([0.20,0.15,0.20,0.20,0.10,0.15], dtype=float)

if "locked" not in st.session_state:
    st.session_state.locked = [False] * 6

w = st.session_state.weights
locked = st.session_state.locked

if st.button("🔄 Reset trọng số về mặc định"):
    st.session_state.weights = np.array([0.20,0.15,0.20,0.20,0.10,0.15], dtype=float)
    st.session_state.locked = [False]*6
    st.experimental_rerun()

cols = st.columns(6)
for i in range(6):
    with cols[i]:
        st.markdown(f"**{criteria[i]}** <br><small style='color:#86c6ff'>{criteria_tooltip[criteria[i]]}</small>", unsafe_allow_html=True)
        locked[i] = st.checkbox("🔒 Lock", locked[i], key=f"lock_{i}")

        inp = st.number_input("Input", 0.0, 1.0, float(w[i]), 0.01, key=f"inp_{i}")
        slid = st.slider("", 0.0, 1.0, float(inp), 0.01, key=f"sld_{i}")

        if not locked[i]:
            diff = slid - w[i]
            w[i] = slid
            idx = [j for j in range(6) if not locked[j] and j != i]
            if len(idx) > 0:
                remain = w[idx]
                w[idx] = remain * ((remain.sum() - diff) / max(remain.sum(),1e-9))

w = w / w.sum()
weights_series = pd.Series(w, index=criteria)
st.session_state.weights = w


# Realtime WEIGHT CHART
df_weight = pd.DataFrame({"criterion":criteria, "weight":w})
fig_w1 = px.bar(df_weight, y="criterion", x="weight", color="weight", color_continuous_scale="Blues")
fig_w2 = px.line_polar(df_weight, r="weight", theta="criterion", line_close=True)

st.plotly_chart(fig_w1, use_container_width=True)
st.plotly_chart(fig_w2, use_container_width=True)


# --------------------------------------------------
# INSURANCE SAMPLE DATA
# --------------------------------------------------
sample = {
    "Company":["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí":[0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý":[6,5,8,7,4],
    "C3: Tỷ lệ tổn thất":[0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC":[9,8,6,9,7],
    "C5: Chăm sóc KH":[9,8,5,7,6],
}
df = pd.DataFrame(sample).set_index("Company")

sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}
climate_map = {("VN - EU",9):0.65,("VN - US",9):0.75,("Domestic",9):0.20}
base_climate = climate_map.get((route,month),0.40)


# --------------------------------------------------
# MONTE CARLO — C6
# --------------------------------------------------
if use_mc:
    rng = np.random.default_rng(42)
    mc = np.zeros((len(df), mc_runs))
    for i, comp in enumerate(df.index):
        mu = base_climate * sensitivity[comp]
        sd = max(mu*0.12, 0.03)
        mc[i] = np.clip(rng.normal(mu, sd, mc_runs), 0, 1)
    df["C6: Rủi ro khí hậu"] = mc.mean(1)
    mc_std = mc.std(1)
else:
    df["C6: Rủi ro khí hậu"] = [base_climate*sensitivity[c] for c in df.index]
    mc_std = np.ones(len(df)) * 0.01


# --------------------------------------------------
# FUZZY AHP (defuzzify TFN)
# --------------------------------------------------
if use_fuzzy:
    fuzz = 0.12
    low = w*(1-fuzz)
    high = w*(1+fuzz)
    w = (low + w + high)/3
    w = w / w.sum()
    weights_series = pd.Series(w, index=criteria)


# --------------------------------------------------
# TOPSIS
# --------------------------------------------------
def topsis(df_data, w, cost_flags):
    M = df_data[list(w.index)].astype(float).values
    R = M / np.sqrt((M ** 2).sum(axis=0))
    V = R * w.values

    is_cost = np.array([cost_flags[c] == "cost" for c in w.index])
    best = np.where(is_cost, V.min(0), V.max(0))
    worst = np.where(is_cost, V.max(0), V.min(0))

    dp = np.sqrt(((V - best)**2).sum(1))
    dm = np.sqrt(((V - worst)**2).sum(1))

    s = dm / (dp + dm + 1e-12)
    r = pd.DataFrame({"company":df_data.index,"score":s})
    r = r.sort_values(by="score", ascending=False).reset_index(drop=True)
    r["rank"] = r.index + 1
    return r


# --------------------------------------------------
# RUN ANALYSIS
# --------------------------------------------------
if st.button("🚀 PHÂN TÍCH NGAY"):
    res = topsis(df, weights_series, cost_flags)

    res["ICC"] = res["score"].apply(lambda x:"ICC A" if x>=0.75 else "ICC B" if x>=0.5 else "ICC C")
    res["Risk"] = res["score"].apply(lambda x:"THẤP" if x>=0.75 else "TRUNG BÌNH" if x>=0.5 else "CAO")

    cv = mc_std / (df["C6: Rủi ro khí hậu"].values + 1e-9)
    conf = 1/(1+cv)
    conf = 0.3 + 0.7*(conf-conf.min()) / ((conf.max()-conf.min())+1e-9)
    res["confidence"] = conf

    st.dataframe(res.set_index("rank"))

    # vẽ biểu đồ kết quả
    fig_bar = px.bar(res, y="company", x="score", color="score", orientation="h", title="Ranking TOPSIS")
    st.plotly_chart(fig_bar, use_container_width=True)

    # =============================
    # EXPORT TO PDF (BẢNG + BIỂU ĐỒ)
    # =============================
    fig_bar.write_image("bar.png")
    fig_w2.write_image("radar.png")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"RISKCAST v3.4 - Insurance Suggestion Report", align="C", ln=1)

    pdf.set_font("Arial","",10)
    pdf.cell(0,6,f"Route: {route}   Method: {method}   Month: {month}", ln=1)
    pdf.ln(4)

    # Bảng kết quả
    pdf.set_font("Arial","B",11)
    pdf.cell(0,6,"Kết quả TOPSIS", ln=1)
    pdf.set_font("Arial","",9)

    for _, r in res.iterrows():
        pdf.cell(80,6,f"{r['rank']} - {r['company']}",1)
        pdf.cell(40,6,f"{r['ICC']}",1)
        pdf.cell(40,6,f"{r['Risk']}",1)
        pdf.cell(30,6,f"{r['confidence']:.2f}",1)
        pdf.ln()

    pdf.ln(6)
    pdf.cell(0,6,"Biểu đồ Ranking TOPSIS")
    pdf.image("bar.png", x=10, w=180)
    pdf.ln(65)

    pdf.cell(0,6,"Biểu đồ Radar (Trọng số)", ln=1)
    pdf.image("radar.png", x=20, w=160)

    pdf.output("RISKCAST_report.pdf")
    st.success("✅ PDF đã xuất thành công!")
    st.download_button("⬇️ Tải PDF", data=open("RISKCAST_report.pdf","rb"), file_name="RISKCAST_v3.4.pdf")

st.markdown("<div class='footer'>RISKCAST v3.4 — Bùi Xuân Hoàng • AI Decision Support System</div>", unsafe_allow_html=True)
