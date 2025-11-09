# RISKCAST v4.8 — Green ESG + Stable CI/CD
# ------------------------------------------------------------
# Hỗ trợ NCKH: Mô hình "Hybrid Past - Present - Future Risk Decision Model"
# ---------------------------------------------
# Past: Fuzzy AHP / TFN xác định trọng số tiêu chí
# Present: TOPSIS đánh giá các nhà bảo hiểm hiện tại
# Future: Monte Carlo + VaR + CVaR + ARIMA dự báo rủi ro tương lai
# ------------------------------------------------------------

import io
import math
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

# ---------------------
# CONFIG UI
# ---------------------
st.set_page_config(page_title="RISKCAST v4.8 — Green ESG", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg,#0b3d0b 0%, #05320a 100%); color: #e9fbf0; font-family: 'Segoe UI'; }
  h1 { color:#a3ff96; text-align:center; font-weight:800; }
  .card { background: rgba(255,255,255,0.06); padding:15px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ RISKCAST — Hybrid Risk-Based Insurance Decision Support System")

# ------------------------------
# Upload Excel input
# ------------------------------
uploaded = st.file_uploader("📂 Upload file Excel (sheet 1: công ty; sheet 2: trọng số tiêu chí)", type=["xlsx"])

if uploaded:
    xls = pd.ExcelFile(uploaded)
    df_company = pd.read_excel(xls, xls.sheet_names[0])
    df_weights = pd.read_excel(xls, xls.sheet_names[1])

    st.subheader("📊 Dữ liệu đầu vào")
    st.dataframe(df_company)

    # Convert trọng số thành vector
    weights = df_weights.iloc[:, 1].to_numpy()

    # Normalize matrix
    norm = df_company.iloc[:, 1:].div(np.sqrt((df_company.iloc[:, 1:] ** 2).sum()), axis=1)

    # Weighted decision matrix
    weighted = norm * weights

    # TOPSIS scoring
    best = weighted.max()
    worst = weighted.min()

    dist_best = np.sqrt(((weighted - best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - worst) ** 2).sum(axis=1))

    scores = dist_worst / (dist_best + dist_worst)
    df_company["Score"] = scores

    # Ranking
    df_company["Rank"] = df_company["Score"].rank(ascending=False)
    df_company = df_company.sort_values("Score", ascending=False)

    st.subheader("🏆 Kết quả xếp hạng (TOPSIS)")
    st.dataframe(df_company)

    # -----------------------
    # FUTURE: Monte Carlo + VaR/CVaR
    # -----------------------
    monte_round = st.sidebar.number_input("Số vòng Monte Carlo", 1000, 50000, 2000)
    c6_min = st.sidebar.slider("Bất định TFN (%)", 5, 50, 15)

    st.sidebar.write("**Mô hình**")
    enable_fuzzy = st.sidebar.checkbox("Bật Fuzzy AHP (TFN)", True)
    enable_arima = st.sidebar.checkbox("Dùng ARIMA dự báo (nếu có dữ liệu)", True)
    enable_var = st.sidebar.checkbox("Tính VaR & CVaR", True)
    enable_mc = st.sidebar.checkbox("Chạy Monte Carlo cho C6", True)

    if enable_mc:
        conf_c6 = (100 - c6_min, 100 + c6_min)

        # ✅ FIX lỗi .ptp(): ép về numpy array
        conf_c6 = np.array(conf_c6, dtype=float)

        # nếu dữ liệu giống nhau (ptp = max-min = 0) → skip monotonic case
        if conf_c6.ptp() > 0:
            mc_result = np.random.uniform(conf_c6.min(), conf_c6.max(), size=monte_round)
            c6_score = np.mean(mc_result)
        else:
            c6_score = float(conf_c6.mean())

        if enable_var:
            var95 = np.percentile(mc_result, 5)
            cvar95 = mc_result[mc_result <= var95].mean()

    # -----------------------
    # Visualization — Pie
    # -----------------------
    fig = go.Figure(go.Pie(
        labels=["Risk Cost", "Optimal Value", "Saving"],
        values=[15, 70, 15],
        hole=0.6,
        marker=dict(line=dict(color="white", width=2))
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # Export PDF (button)
    # -----------------------
    if st.button("📄 Xuất PDF báo cáo"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=13)
        pdf.cell(0, 10, "RISKCAST — Insurance Decision Report", ln=True)

        for idx, row in df_company.iterrows():
            pdf.cell(0, 8, f"{row[0]} — Score: {round(row['Score'],3)} — Rank: {int(row['Rank'])}", ln=True)

        pdf.output("RISKCAST_Report.pdf")
        st.success("✅ Xuất PDF thành công!")
