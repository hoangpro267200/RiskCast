# ===================== RISKCAST v3.3 (FULL INTEGRATED) =====================
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from fpdf import FPDF
import requests

# ----------------------------
# PAGE CONFIG + STYLE
# ----------------------------
st.set_page_config(page_title="RISKCAST v3.3", layout="wide", page_icon="shield")
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg,#021023 0%, #082c4a 100%); color: #e6f0ff; font-family: 'Segoe UI'; }
    .block-container { padding: 1.5rem 2rem; }
    h1 { color: #7bd3ff; text-align: center; font-weight: 800; font-size: 2.8rem; }
    .stButton>button { background: linear-gradient(90deg,#00c6ff,#7b2ff7); color: white;
                       font-weight:bold; border-radius: 15px; padding: 0.8rem; font-size: 1.1rem; }
    .result-box { background: #1a2a44; padding: 1.5rem; border-radius: 15px;
                  border-left: 6px solid #00d4ff; margin: 1.5rem 0; box-shadow: 0 4px 12px rgba(0,212,255,0.3); }
    .footer { text-align: center; margin-top: 3rem; color: #aaa; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("RISKCAST v3.3 — HỆ THỐNG ĐỀ XUẤT BẢO HIỂM THÔNG MINH")
st.caption("Fuzzy AHP · Smart Weights · Monte-Carlo Climate · TOPSIS · Confidence Score")

# ----------------------------
# SIDEBAR INPUT
# ----------------------------
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị (USD)", value=39000, step=1000, format="%d")
    good_type  = st.selectbox("Loại hàng", ["Điện tử","Đông lạnh","Hàng khô","Hàng nguy hiểm","Khác"])
    route      = st.selectbox("Tuyến", ["VN - EU","VN - US","VN - Singapore","VN - China","Domestic"])
    method     = st.selectbox("Phương thức", ["Sea","Air","Truck"])
    month      = st.selectbox("Tháng", list(range(1,13)), index=8)
    priority   = st.selectbox("Ưu tiên", ["An toàn tối đa","Cân bằng","Tối ưu chi phí"])
    use_fuzzy  = st.checkbox("Bật Fuzzy AHP (TFN → Defuzzify)", True)
    use_mc     = st.checkbox("Bật Monte-Carlo Climate", True)
    mc_runs    = st.number_input("Số vòng Monte-Carlo", 200, 20000, 2000, 100)
    fetch_noaa = st.checkbox("Lấy dữ liệu NOAA (nếu API khả dụng)", False)
    st.markdown("---")

# ----------------------------
# CRITERIA & COST/BENEFIT
# ----------------------------
criteria = [
    "C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"
]
cost_flags = {c:"cost" if c in ["C1: Tỷ lệ phí","C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

criteria_tooltips = {
    "C1: Tỷ lệ phí"        : "Phí bảo hiểm – càng thấp càng tốt",
    "C2: Thời gian xử lý"  : "Tốc độ bồi thường – càng nhanh càng tốt",
    "C3: Tỷ lệ tổn thất"   : "Tỷ lệ claim thất bại – càng thấp càng tốt",
    "C4: Hỗ trợ ICC"       : "ICC A/B/C – phạm vi bảo vệ, càng rộng càng tốt",
    "C5: Chăm sóc KH"      : "Mức độ hỗ trợ/CSKH – càng tốt càng an tâm",
    "C6: Rủi ro khí hậu"   : "Rủi ro theo tuyến & mùa – càng thấp càng tốt"
}

# ----------------------------
# SMART SLIDER + LOCK + INPUT + RESET + REALTIME
# ----------------------------
st.subheader("🎛️ Phân bổ trọng số tiêu chí (Smart Slider + Lock + Reset)")

if "weights" not in st.session_state:
    st.session_state.weights = np.array([0.20,0.15,0.20,0.20,0.10,0.15])
if "locked" not in st.session_state:
    st.session_state.locked = [False]*6

w = st.session_state.weights
locked = st.session_state.locked

# Reset button
if st.button("🔄 Reset trọng số về mặc định"):
    st.session_state.weights = np.array([0.20,0.15,0.20,0.20,0.10,0.15])
    st.session_state.locked = [False]*6
    st.experimental_rerun()

cols = st.columns(6)
for i in range(6):
    with cols[i]:
        st.markdown(f"**{criteria[i]}**  \n<small style='color:#7bd3ff'>{criteria_tooltips[criteria[i]]}</small>", unsafe_allow_html=True)
        locked[i] = st.checkbox("🔒 Lock", value=locked[i], key=f"lock_{i}")
        inp = st.number_input("Input", 0.0,1.0,float(w[i]),0.01,key=f"in_{i}")
        slid = st.slider("",0.0,1.0,inp,0.01,key=f"sl_{i}")

        if not locked[i]:
            diff = slid - w[i]
            w[i] = slid
            if abs(diff)>1e-9:
                idx_free = [j for j in range(6) if j!=i and not locked[j]]
                rem = w[idx_free]
                s = rem.sum()
                if s>0:
                    w[idx_free] = rem * ((s - diff)/s)

w = w / w.sum()
st.session_state.weights = w
st.session_state.locked  = locked
weights_series = pd.Series(w, index=criteria)

# realtime chart
dfw = pd.DataFrame({"criterion":criteria,"weight":w})
c1,c2 = st.columns(2)
with c1:
    fig1 = px.bar(dfw, x="weight", y="criterion", orientation="h",color="weight",
                  color_continuous_scale="Blues", title="Trọng số realtime")
    st.plotly_chart(fig1,True)
with c2:
    fig2 = px.line_polar(dfw,r="weight",theta="criterion",line_close=True,title="Radar trọng số")
    st.plotly_chart(fig2,True)

# ----------------------------
# DATA INSURANCE SAMPLE
# ----------------------------
sample = {
    "Company":["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí":[0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý":[6,5,8,7,4],
    "C3: Tỷ lệ tổn thất":[0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC":[9,8,6,9,7],
    "C5: Chăm sóc KH":[9,8,5,7,6]
}
df = pd.DataFrame(sample).set_index("Company")
sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}

climate_map = {("VN - EU",9):0.65,("VN - US",9):0.75,("Domestic",9):0.20}
base_climate = climate_map.get((route,month),0.40)

if cargo_value>50000: df["C1: Tỷ lệ phí"]*=1.2
if route in ["VN - US","VN - EU"]: df["C2: Thời gian xử lý"]*=1.3
if good_type in ["Điện tử","Hàng nguy hiểm"]: df["C3: Tỷ lệ tổn thất"]*=1.5

# MC Climate
if use_mc:
    mc = np.zeros((len(df),mc_runs))
    rng = np.random.default_rng(42)
    for i,c in enumerate(df.index):
        mu = base_climate*sensitivity[c]
        sd = max(0.03,mu*0.12)
        mc[i] = np.clip(rng.normal(mu,sd,mc_runs),0,1)
    df["C6: Rủi ro khí hậu"]=mc.mean(1)
    mc_std = mc.std(1)
else:
    df["C6: Rủi ro khí hậu"]=[base_climate*sensitivity[c] for c in df.index]
    mc_std=np.zeros(len(df))+1e-4

# TOPSIS
def topsis(data,w,cost):
    M=data[w.index].values
    R=M/np.sqrt((M**2).sum(0))
    V=R*w.values
    cost_arr=np.array([cost[c]=="cost" for c in w.index])
    best=np.where(cost_arr,V.min(0),V.max(0))
    worst=np.where(cost_arr,V.max(0),V.min(0))
    dp=np.sqrt(((V-best)**2).sum(1))
    dm=np.sqrt(((V-worst)**2).sum(1))
    s=dm/(dp+dm+1e-12)
    r=pd.DataFrame({"company":data.index,"score":s})
    r=r.sort_values("score",False).reset_index(drop=True)
    r["rank"]=r.index+1
    return r

# RUN
if st.button("🚀 PHÂN TÍCH NGAY",True):
    res = topsis(df,weights_series,cost_flags)
    res["ICC"]=res["score"].map(lambda x:"ICC A" if x>0.75 else "ICC B" if x>0.5 else "ICC C")
    res["Risk"]=res["score"].map(lambda x:"THẤP" if x>0.75 else "TRUNG" if x>0.5 else "CAO")

    mean=df["C6: Rủi ro khí hậu"].values
    cv=np.where(mean==0,0,mc_std/mean)
    conf=1/(1+cv)
    conf=0.3+0.7*(conf-conf.min())/((conf.max()-conf.min())+1e-9)
    critcv=df[w.index].std(1)/(df[w.index].mean(1)+1e-9)
    crit_conf=1/(1+critcv)
    fin=np.sqrt(conf*crit_conf.values)
    mp=dict(zip(df.index,fin))
    res["confidence"]=res["company"].map(mp)
    res["score_pct"]=(res["score"]*100).round(2)
    res=res[["rank","company","score","score_pct","ICC","Risk","confidence"]]

    st.dataframe(res,use_container_width=True)
    st.success("✅ Hoàn tất phân tích RISKCAST v3.3!")

# Footer
st.markdown("<div class='footer'>RISKCAST v3.3 · Bùi Xuân Hoàng</div>",unsafe_allow_html=True)
