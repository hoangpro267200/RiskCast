# ------------------------------
# RISKCAST v3.3 — Stable Release
# by Bùi Xuân Hoàng / GPT-5 (Kai)
# ------------------------------

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from fpdf import FPDF

# ========= PAGE CONFIG + CSS =============
st.set_page_config(page_title="RISKCAST v3.3", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg,#031223 0%, #082c4a 100%); color: #e6f0ff; font-family: 'Segoe UI'; }
    .block-container { padding: 1.2rem 1.8rem; }
    h1 { color: #7be2ff; text-align: center; font-size: 2.6rem; font-weight: 800; }
    .weight-box { background:#0f2440; padding: 16px; border-radius: 12px; }
    .result-box { background:#14375e; padding:16px; border-radius:12px; border-left:6px solid #00d4ff; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ RISKCAST v3.3 — HỆ THỐNG ĐỀ XUẤT BẢO HIỂM THÔNG MINH")

# =========================================================
# 🔧 FUNCTION: REDISTRIBUTE WEIGHTS (Smart slider behavior)
# =========================================================
def redistribute(weights, locked):
    """Redistribute unlocked weights so total = 1. Keeps locked weights unchanged."""
    weights = np.array(weights, dtype=float)
    locked = np.array(locked, dtype=bool)

    locked_sum = weights[locked].sum()
    free_sum = weights[~locked].sum()

    # Nếu khóa hết → không chỉnh
    if (~locked).sum() == 0:
        return weights

    # Tổng locked đã >= 1 → ép phần còn lại = 0
    if locked_sum >= 1:
        weights[~locked] = 0
        return weights

    # Chia đều phần còn lại
    remain = 1 - locked_sum
    weights[~locked] = remain / (~locked).sum()
    return weights

# ======================================================================
# 🔢 INPUT — SIDEBAR (Cargo, Route, Method…)
# ======================================================================
with st.sidebar:
    st.header("📦 Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị hàng hóa (USD)", value=35000, step=1000)

    good_type = st.selectbox("Loại hàng", ["Điện tử", "Hàng lạnh", "Hàng khô", "Hàng nguy hiểm"])
    route = st.selectbox("Tuyến vận chuyển", ["VN - US", "VN - EU", "VN - Singapore", "Domestic"])
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
    month = st.selectbox("Tháng", list(range(1,13)), index=8)

    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])
    fuzzy_on = st.checkbox("Bật Fuzzy AHP (Defuzzify)", value=True)
    mc_on = st.checkbox("Bật Monte-Carlo (climate risk)", value=True)

# ======================================================================
# 🎚️ SMART WEIGHT SLIDER UI + LOCK + RESET
# ======================================================================
st.subheader("⚖️ Phân bổ trọng số tiêu chí (Smart Auto-Balance)")

criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

explain = {
    "C1: Tỷ lệ phí": "Phí bảo hiểm — càng thấp càng tốt.",
    "C2: Thời gian xử lý": "Thời gian giải quyết claim — càng nhanh càng tốt.",
    "C3: Tỷ lệ tổn thất": "Tỷ lệ từ chối/thất thoát — càng thấp càng tốt.",
    "C4: Hỗ trợ ICC": "Phạm vi ICC (A/B/C) — càng rộng càng tốt.",
    "C5: Chăm sóc KH": "Dịch vụ hỗ trợ khách hàng — càng tốt càng an tâm.",
    "C6: Rủi ro khí hậu": "Ảnh hưởng khí hậu/tuyến/tháng — càng thấp càng tốt."
}

if "weights" not in st.session_state:
    st.session_state["weights"] = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])
if "locked" not in st.session_state:
    st.session_state["locked"] = [False]*6

col_reset = st.columns([1])[0]
if col_reset.button("🔄 Reset trọng số về mặc định"):
    st.session_state["weights"] = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15])
    st.session_state["locked"] = [False]*6

cols = st.columns(6)
new_weights = st.session_state["weights"].copy()

for i, c in enumerate(criteria):
    with cols[i]:
        st.markdown(f"**{c}**")
        st.caption(explain[c])

        st.session_state["locked"][i] = st.checkbox("🔒 Lock", value=st.session_state["locked"][i])

        val = st.number_input("Nhập tỉ lệ", min_value=0.0, max_value=1.0,
                              value=float(new_weights[i]), key=f"in_{i}", step=0.01)
        new_weights[i] = val

# Auto balance = 1.0
st.session_state["weights"] = redistribute(new_weights, st.session_state["locked"])

# Realtime chart
fig_weights = px.pie(
    names=criteria,
    values=st.session_state["weights"],
    title="Phân bố trọng số (Realtime)",
    color_discrete_sequence=px.colors.sequential.Blues
)
st.plotly_chart(fig_weights, use_container_width=True)

weights_series = pd.Series(st.session_state["weights"], index=criteria)

# ======================================================================
# 🧠 DỮ LIỆU GIẢ LẬP (Demo)
# ======================================================================
df = pd.DataFrame({
    "Company": ["PVI", "Chubb", "BaoViet", "Aon", "InternationalIns"],
    "C1: Tỷ lệ phí": [0.28, 0.30, 0.32, 0.24, 0.26],
    "C2: Thời gian xử lý": [5, 6, 7, 4, 8],
    "C3: Tỷ lệ tổn thất": [0.06, 0.08, 0.10, 0.07, 0.09],
    "C4: Hỗ trợ ICC": [8, 9, 9, 7, 6],
    "C5: Chăm sóc KH": [8, 9, 7, 6, 5],
    "C6: Rủi ro khí hậu": [0.55, 0.50, 0.60, 0.45, 0.62]
}).set_index("Company")

# ======================================================================
# 🧮 TOPSIS FUNCTION
# ======================================================================
def topsis(df, weights):
    M = df.values.astype(float)
    denom = np.sqrt((M**2).sum(axis=0))
    R = M / denom
    V = R * weights.values
    ideal_best = V.max(axis=0)
    ideal_worst = V.min(axis=0)
    d_plus = np.sqrt(((V - ideal_best)**2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst)**2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)
    result = pd.DataFrame({"company": df.index, "score": score}).sort_values("score", ascending=False)
    result["rank"] = range(1, len(result)+1)
    return result.reset_index(drop=True)

# ======================================================================
# ▶️ RUN
# ======================================================================
st.markdown("---")

if st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True):

    res = topsis(df, weights_series)
    best = res.iloc[0]

    st.success(f"✅ Đề xuất: **{best.company}** (Rank #1 — Score {best.score:.3f})")

    st.dataframe(res, use_container_width=True)

    # 📊 biểu đồ Score
    fig_bar = px.bar(res, x="score", y="company", color="score", color_continuous_scale="Blues")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ✅ Export Excel
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        res.to_excel(writer, index=False, sheet_name="Result")
        pd.DataFrame(weights_series).to_excel(writer, sheet_name="Weights")

    st.download_button("📥 Tải Excel (Kết quả)", out.getvalue(),
                       file_name="riskcast_result.xlsx", mime="application/vnd.ms-excel")

    # ✅ Export PDF (Hỗ trợ Unicode tiếng Việt)
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Roboto", "", "Roboto-Regular.ttf", uni=True)
    pdf.set_font("Roboto", size=12)
    pdf.cell(0, 8, f"RISKCAST Báo cáo đề xuất bảo hiểm", ln=True)
    pdf.cell(0, 8, f"Lựa chọn tốt nhất: {best.company}", ln=True)

    pdf.ln(5)
    for _, r in res.iterrows():
        pdf.cell(0, 6, f"{r['rank']}. {r['company']} — Score: {r['score']:.3f}", ln=True)

    pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
    st.download_button("📄 Xuất PDF", pdf_bytes, file_name="riskcast_report.pdf", mime="application/pdf")

