# ==========================================================
# ✅ RISKCAST v3.5 — Smart Auto-Balance + TOPSIS + PDF Export
# Author: Bùi Xuân Hoàng (R&D Logistics - University Project)
# ==========================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from fpdf import FPDF
import io

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="RISKCAST 3.5", page_icon="🛡", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg,#021023 0%, #082c4a 100%); color: #e6f0ff; font-family: 'Segoe UI'; }
    h1 { text-align:center; font-weight:800; color:#7bd3ff; }
    .result-box { background:#18324a; padding:1.2rem; border-radius:10px; border-left:5px solid #00eaff; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# AUTO-BALANCE WEIGHT FUNCTION
# ----------------------------------------------------------
def auto_balance(values, locked):
    """Đảm bảo tổng trọng số luôn = 1 và giữ nguyên tiêu chí đã lock."""
    values = np.array(values)
    locked = np.array(locked)

    remaining = 1 - values[locked].sum()
    free_idx = np.where(~locked)[0]

    if len(free_idx) > 0:
        values[free_idx] = values[free_idx] / values[free_idx].sum() * remaining

    return np.round(values, 4)


# ----------------------------------------------------------
# SIDEBAR INPUT
# ----------------------------------------------------------
with st.sidebar:
    st.header("📦 Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị hàng hóa (USD)", value=35000, step=1000)
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Nguy hiểm", "Khác"])
    route = st.selectbox("Tuyến vận chuyển", ["VN - US", "VN - EU", "VN - CN", "Nội địa"])
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
    month = st.selectbox("Tháng", list(range(1, 13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Tối ưu chi phí", "Cân bằng"])

st.title("🛡 RISKCAST v3.5 — HỆ THỐNG ĐỀ XUẤT BẢO HIỂM THÔNG MINH")

criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

tooltip = {
    "C1: Tỷ lệ phí": "Phí bảo hiểm — càng thấp càng tốt",
    "C2: Thời gian xử lý": "Thời gian xử lý claim — càng nhanh càng tốt",
    "C3: Tỷ lệ tổn thất": "Tỷ lệ từ chối / thất thoát — thấp càng tốt",
    "C4: Hỗ trợ ICC": "Phạm vi ICC (A/B/C) — càng rộng càng tốt",
    "C5: Chăm sóc KH": "Hỗ trợ khách hàng — càng tốt càng an tâm",
    "C6: Rủi ro khí hậu": "Ảnh hưởng khí hậu theo tuyến / tháng — càng thấp càng tốt"
}


# ----------------------------------------------------------
# SMART SLIDER + LOCK UI
# ----------------------------------------------------------
st.subheader("⚖️ Phân bổ trọng số tiêu chí (Smart Auto-Balance + Lock)")

default = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])

if "weights" not in st.session_state:
    st.session_state["weights"] = default.copy()

if "locked" not in st.session_state:
    st.session_state["locked"] = [False] * 6

cols = st.columns(6)
new_values = st.session_state["weights"].copy()

if st.button("🔄 Reset trọng số về mặc định"):
    st.session_state["weights"] = default.copy()
    st.session_state["locked"] = [False] * 6

for i, c in enumerate(criteria):
    with cols[i]:
        st.markdown(f"**{c}**")
        st.caption(tooltip[c])

        st.session_state["locked"][i] = st.checkbox("🔒 Lock", st.session_state["locked"][i], key=f"lock{i}")

        new_values[i] = st.number_input("Nhập trọng số", min_value=0.0, max_value=1.0,
                                         value=float(new_values[i]), step=0.01, key=f"input{i}")

st.session_state["weights"] = auto_balance(new_values, st.session_state["locked"])
weights = pd.Series(st.session_state["weights"], index=criteria)

# Biểu đồ realtime
fig = px.pie(values=weights, names=weights.index, title="Biểu đồ phân bổ trọng số (Realtime)")
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
    "C6: Rủi ro khí hậu": [0.72, 0.75, 0.70, 0.50, 0.60],
}).set_index("Company")


def topsis(df_data, weights):
    M = df_data.values
    norm = M / np.sqrt((M ** 2).sum(axis=0))
    weights = np.array(weights.values)
    V = norm * weights
    ideal_best = np.max(V, axis=0)
    ideal_worst = np.min(V, axis=0)
    d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    return d_minus / (d_plus + d_minus)


# ----------------------------------------------------------
# RUN MODEL
# ----------------------------------------------------------
if st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True):

    df["Score"] = topsis(df, weights)
    result = df.sort_values("Score", ascending=False)

    st.subheader("📊 KẾT QUẢ XẾP HẠNG")
    st.dataframe(result.style.format({"Score": "{:.4f}"}))

    best = result.iloc[0]

    st.markdown(f"""
    <div class="result-box">
    ✅ Công ty đề xuất: <strong>{best.name}</strong><br>
    ✅ Gói bảo hiểm IC: <strong>ICC A</strong><br>
    ✅ Điểm TOPSIS: <strong>{best.Score:.4f}</strong>
    </div>
    """, unsafe_allow_html=True)

    # ===================== EXPORT PDF ======================
    class PDF(FPDF):
        pass

    pdf = PDF()
    pdf.add_page()

    # font unicode (bắt buộc file fonts/DejaVuSans.ttf phải có trong dự án)
    pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    pdf.cell(0, 8, "RISKCAST v3.5 — Báo cáo đề xuất bảo hiểm", ln=1)
    pdf.ln(4)
    pdf.set_font("DejaVu", size=10)

    # table
    for idx, r in result.iterrows():
        pdf.cell(40, 8, idx)
        pdf.cell(40, 8, f"{r.Score:.4f}")
        pdf.ln()

    # pie chart export
    fig.write_image("chart.png")
    pdf.image("chart.png", x=10, w=180)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    st.download_button("⬇️ Xuất PDF", data=pdf_bytes, file_name="RISKCAST_Report.pdf",
                       mime="application/pdf")

    # Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        result.to_excel(writer, sheet_name="Result")
    st.download_button("⬇️ Xuất Excel", data=buffer,
                       file_name="RISKCAST_Result.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

