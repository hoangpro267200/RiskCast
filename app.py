# app.py — RISKCAST v4.0 PRO | UI Premium + Session State + Lock/Reset
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import io
from risk_engine import RiskEngine  # Tách thuật toán

# -----------------------
# Page Config + CSS XANH LÁ PREMIUM
# -----------------------
st.set_page_config(page_title="RISKCAST v4.0 PRO", layout="wide", page_icon="leaf")
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0a2e1c 0%, #1a4731 100%); color: #e8f5e9; font-family: 'Segoe UI'; }
    h1 { color: #a5d6a7; text-align: center; font-weight: 800; font-size: 3rem; }
    .stButton>button { background: linear-gradient(90deg, #4caf50, #81c784); color: white; font-weight: bold; border-radius: 12px; padding: 0.8rem; font-size: 1.1rem; }
    .result-box { background: #1b5e20; padding: 1.5rem; border-radius: 15px; border-left: 6px solid #81c784; margin: 1.5rem 0; box-shadow: 0 4px 12px rgba(129,199,132,0.3); }
    .footer { text-align: center; margin-top: 3rem; color: #a5d6a7; font-size: 0.9rem; }
    .lock-btn { background: #2e7d32; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("RISKCAST v4.0 PRO — HỆ THỐNG BẢO HIỂM THÔNG MINH")
st.caption("**Full Science | UI Premium | Session State | Lock Weights | Export PDF/Excel**")

# -----------------------
# Session State Init
# -----------------------
if 'weights' not in st.session_state:
    st.session_state.weights = [0.20, 0.15, 0.20, 0.20, 0.10, 0.15]
if 'locked' not in st.session_state:
    st.session_state.locked = [False] * 6
if 'result' not in st.session_state:
    st.session_state.result = None

# -----------------------
# Sidebar Input
# -----------------------
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị (USD)", value=39000, step=1000)
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"])
    route = st.selectbox("Tuyến", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"])
    month = st.selectbox("Tháng", list(range(1, 13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

    st.header("Tùy chỉnh")
    use_fuzzy = st.checkbox("Fuzzy AHP", True)
    use_mc = st.checkbox("Monte Carlo", True)
    mc_runs = st.number_input("Số vòng MC", 500, 10000, 2000, 500)

# -----------------------
# Trọng số + Lock + Reset + Auto Normalize
# -----------------------
st.subheader("Trọng số tiêu chí (Lock + Reset)")
criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất", "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]
cols = st.columns(6)
new_weights = []
for i, c in enumerate(criteria):
    with cols[i]:
        locked = st.session_state.locked[i]
        val = st.session_state.weights[i]
        if locked:
            st.write(f"**{c}**")
            st.write(f"`{val:.3f}`")
            if st.button("Mở", key=f"unlock_{i}"):
                st.session_state.locked[i] = False
                st.rerun()
        else:
            new_val = st.slider("", 0.0, 1.0, val, 0.01, key=f"w_{i}")
            new_weights.append(new_val)
            if st.button("Khóa", key=f"lock_{i}"):
                st.session_state.locked[i] = True
                st.rerun()

# Cập nhật trọng số + Auto normalize
if not all(st.session_state.locked):
    total = sum(new_weights) if new_weights else sum(st.session_state.weights)
    st.session_state.weights = [w / total for w in (new_weights or st.session_state.weights)]

col1, col2 = st.columns(2)
with col1:
    if st.button("RESET TRỌNG SỐ"):
        st.session_state.weights = [0.20, 0.15, 0.20, 0.20, 0.10, 0.15]
        st.session_state.locked = [False] * 6
        st.rerun()
with col2:
    st.write(f"**Tổng:** {sum(st.session_state.weights):.3f}")

# -----------------------
# RUN ANALYSIS
# -----------------------
if st.button("PHÂN TÍCH TOÀN DIỆN", use_container_width=True):
    with st.spinner("Đang tính toán..."):
        engine = RiskEngine(
            cargo_value=cargo_value,
            route=route,
            month=month,
            good_type=good_type,
            priority=priority,
            weights=st.session_state.weights,
            use_fuzzy=use_fuzzy,
            use_mc=use_mc,
            mc_runs=mc_runs
        )
        result = engine.run()
        st.session_state.result = result
        st.success("HOÀN TẤT!")

# -----------------------
# Hiển thị kết quả (từ session state)
# -----------------------
if st.session_state.result is not None:
    result = st.session_state.result
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(result[['rank', 'company', 'score_pct', 'ICC', 'Risk', 'confidence']].set_index('rank'), use_container_width=True)
    with col2:
        fig = px.bar(result, x='score', y='company', color='score', color_continuous_scale='Greens', title="Xếp hạng TOPSIS")
        st.plotly_chart(fig, use_container_width=True)

    best = result.iloc[0]
    st.markdown(f"""
    <div class="result-box">
    <h3>ĐỀ XUẤT TỐI ƯU</h3>
    <p><strong>Công ty:</strong> {best['company']} | <strong>Score:</strong> {best['score']:.3f} ({best['score_pct']}%) | <strong>Conf:</strong> {best['confidence']:.2f}</p>
    <p><strong>ICC:</strong> {best['ICC']} | <strong>Rủi ro:</strong> {best['Risk']}</p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------
    # EXPORT PDF + EXCEL
    # -----------------------
    output_xlsx = io.BytesIO()
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        result.to_excel(writer, 'Result', index=False)
        pd.DataFrame({'Criteria': criteria, 'Weight': st.session_state.weights}).to_excel(writer, 'Weights', index=False)
    output_xlsx.seek(0)
    st.download_button("Xuất Excel", data=output_xlsx, file_name="riskcast_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "BÁO CÁO RISKCAST v4.0 PRO", ln=True, align='C')
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, f"Giá trị: ${cargo_value:,} | Tuyến: {route} | Tháng: {month}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 10)
    for i, row in result.iterrows():
        pdf.cell(0, 8, f"{int(row['rank'])}. {row['company']} — Score: {row['score']:.3f} — ICC: {row['ICC']}", ln=True)
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    st.download_button("Xuất PDF", data=pdf_bytes, file_name="riskcast_report.pdf", mime="application/pdf")

# Footer
st.markdown("<div class='footer'><strong>RISKCAST v4.0 PRO</strong> – Premium UI | Session State | Lock Weights</div>", unsafe_allow_html=True)
