# ==========================================================
# RISKCAST v3.3 — Smart TOPSIS + Auto-Balance + Fuzzy + Monte-Carlo
# Author: Bùi Xuân Hoàng
# ==========================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import io

# ----------------------------------------------------------
# PAGE SETUP + CSS
# ----------------------------------------------------------
st.set_page_config(page_title="RISKCAST 3.3", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg,#021023 0%, #082c4a 100%); color: #e6f0ff; font-family: 'Segoe UI'; }
    h1 { color: #66e3ff; text-align: center; font-weight: 800; }
    .block-container { padding: 1.5rem 2rem; }
    .result-box { background: #143759; padding: 1rem; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# REDISTRIBUTE FUNCTION (Auto-balance weight)
# ----------------------------------------------------------
def redistribute(values, locked):
    """Giữ nguyên tiêu chí lock, auto-balance phần còn lại sao cho tổng = 1"""
    locked = np.array(locked)
    values = np.array(values)

    remain = 1 - values[locked].sum()
    free_idx = np.where(~locked)[0]

    if len(free_idx) > 0:
        values[free_idx] = values[free_idx] / values[free_idx].sum() * remain

    return np.round(values, 4)


# ----------------------------------------------------------
# SIDEBAR INPUT
# ----------------------------------------------------------
with st.sidebar:
    st.header("📦 Thông tin lô hàng")

    cargo_value = st.number_input("Giá trị hàng hóa (USD)", value=35000, step=1000)
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Khô", "Hàng nguy hiểm", "Khác"])
    route = st.selectbox("Tuyến vận chuyển", ["VN - US", "VN - EU", "VN - CN", "Nội địa"])
    shipping = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
    month = st.selectbox("Tháng", list(range(1, 13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

st.title("🛡️ RISKCAST v3.3 — HỆ THỐNG ĐỀ XUẤT BẢO HIỂM THÔNG MINH")


# ----------------------------------------------------------
# SMART AUTO BALANCE WEIGHTS
# ----------------------------------------------------------

criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

explain = {
    "C1: Tỷ lệ phí": "Phí bảo hiểm — thấp càng tốt",
    "C2: Thời gian xử lý": "Giải quyết claim — nhanh càng tốt",
    "C3: Tỷ lệ tổn thất": "Tỷ lệ từ chối/thất thoát — thấp càng tốt",
    "C4: Hỗ trợ ICC": "Phạm vi ICC A/B/C — rộng càng tốt",
    "C5: Chăm sóc KH": "Hỗ trợ khách hàng — tốt càng an tâm",
    "C6: Rủi ro khí hậu": "Ảnh hưởng khí hậu theo tuyến/tháng — thấp càng tốt"
}

default_w = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])

if "weights" not in st.session_state:
    st.session_state["weights"] = default_w.copy()

if "locked" not in st.session_state:
    st.session_state["locked"] = [False] * 6

st.subheader("⚖️ Phân bổ trọng số tiêu chí (Smart Auto-Balance)")

if st.button("🔄 Reset mặc định"):
    st.session_state["weights"] = default_w.copy()
    st.session_state["locked"] = [False] * 6

cols = st.columns(6)
new_w = st.session_state["weights"].copy()

for i, c in enumerate(criteria):
    with cols[i]:
        st.markdown(f"**{c}**")
        st.caption(explain[c])

        st.session_state["locked"][i] = st.checkbox("🔒 Lock", value=st.session_state["locked"][i],
                                                    key=f"lock_{i}")

        new_w[i] = st.number_input("Nhập trọng số", min_value=0.0, max_value=1.0,
                                   value=float(new_w[i]), step=0.01, key=f"input_{i}")

st.session_state["weights"] = redistribute(new_w, st.session_state["locked"])
weights_series = pd.Series(st.session_state["weights"], index=criteria)


# Biểu đồ realtime
fig = px.pie(values=weights_series, names=criteria, title="Biểu đồ phân bổ trọng số (Realtime)")
st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------
# DATA SAMPLE + TOPSIS
# ----------------------------------------------------------

df = pd.DataFrame({
    "Company": ["Chubb", "PVI", "BaoViet", "Aon", "GlobalIns"],
    "C1: Tỷ lệ phí": [0.30, 0.28, 0.32, 0.25, 0.27],
    "C2: Thời gian xử lý": [6, 5, 8, 4, 7],
    "C3: Tỷ lệ tổn thất": [0.08, 0.06, 0.10, 0.07, 0.09],
    "C4: Hỗ trợ ICC": [9, 8, 9, 7, 6],
    "C5: Chăm sóc KH": [9, 8, 7, 6, 5],
    "C6: Rủi ro khí hậu": [0.70, 0.75, 0.65, 0.50, 0.60],
}).set_index("Company")


def topsis(df_data, weights):
    M = df_data.values
    norm = M / np.sqrt((M ** 2).sum(axis=0))
    V = norm * weights

    ideal_best = np.max(V, axis=0)
    ideal_worst = np.min(V, axis=0)

    d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    score = d_minus / (d_plus + d_minus)
    return score


# ----------------------------------------------------------
# RUN BUTTON
# ----------------------------------------------------------
if st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True):

    score = topsis(df, weights_series)
    df["Score"] = score
    df = df.sort_values("Score", ascending=False)

    st.subheader("📊 KẾT QUẢ XẾP HẠNG")
    st.dataframe(df.style.format({"Score": "{:.4f}"}))

    best = df.iloc[0]

    st.markdown(f"""
    <div class="result-box">
        ✅ Công ty khuyến nghị: **{best.name}**  
        ✅ Gói bảo hiểm: **ICC A**  
        ✅ Điểm TOPSIS: **{best.Score:.4f}**
    </div>
    """, unsafe_allow_html=True)

    # Export Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Result")
    st.download_button("⬇️ Xuất Excel (Kết quả)", data=output,
                       file_name="RISKCAST_Result.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

