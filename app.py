# app.py — RISKCAST v2.5
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from fpdf import FPDF

# -----------------------
# Page config + CSS
# -----------------------
st.set_page_config(page_title="RISKCAST v2.5", layout="wide", page_icon="🛡")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg,#021023 0%, #082c4a 70%); color: #e6f0ff; }
    .block-container { padding: 1.2rem 2rem; }
    h1 { color: #7bd3ff; text-align: center; font-weight: 800; }
    .stButton>button { background: linear-gradient(90deg,#00c6ff,#7b2ff7); color: white;
                       font-weight:bold; border-radius: 12px; padding: 0.7rem; }
    .result-box { background: #1a2a44; padding: 1.2rem; border-radius: 12px;
                  border-left: 5px solid #00d4ff; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.title("🛡 RISKCAST v2.5 — HỆ THỐNG ĐỀ XUẤT BẢO HIỂM THÔNG MINH")
st.caption("Nhập lô hàng → Phân tích rủi ro → Xếp hạng công ty bảo hiểm → Xuất PDF/Excel")


# -----------------------
# Sidebar Input
# -----------------------
with st.sidebar:
    st.header("📦 Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị (USD)", value=31000, step=1000)
    good_type = st.selectbox("Loại hàng", ["Đông lạnh", "Hàng khô", "Điện tử", "Hàng nguy hiểm", "Khác"])
    route = st.selectbox("Tuyến", ["VN - Singapore", "VN - US", "VN - EU", "VN - China", "Domestic"])
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
    month = st.selectbox("Tháng", list(range(1, 13)), index=10)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])


# -----------------------
# Criteria & Weights
# -----------------------
criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất", "C4: Hỗ trợ ICC", "C5: Chăm sóc KH"]

cost_flags = {
    "C1: Tỷ lệ phí": "cost",
    "C2: Thời gian xử lý": "benefit",
    "C3: Tỷ lệ tổn thất": "benefit",
    "C4: Hỗ trợ ICC": "benefit",
    "C5: Chăm sóc KH": "benefit"
}

st.subheader("⚖ Điều chỉnh trọng số tiêu chí")
cols = st.columns(5)
weights = [cols[i].slider(criteria[i], 0.0, 1.0, 0.2, 0.01) for i in range(5)]
w = np.array(weights)

# Boost theo ưu tiên
if priority == "An toàn tối đa":
    w[1] *= 1.4; w[4] *= 1.3
elif priority == "Tối ưu chi phí":
    w[0] *= 1.5

w = w / w.sum()
weights_series = pd.Series(w, index=criteria)


# -----------------------
# Data mẫu của 5 công ty bảo hiểm
# -----------------------
sample = {
    "Company": ["PVI", "BaoViet", "Aon", "Chubb", "InternationalIns"],
    "C1: Tỷ lệ phí": [0.22, 0.24, 0.20, 0.28, 0.26],
    "C2: Thời gian xử lý": [7, 5, 4, 6, 8],
    "C3: Tỷ lệ tổn thất": [0.08, 0.10, 0.06, 0.12, 0.09],
    "C4: Hỗ trợ ICC": [8, 9, 7, 9, 6],
    "C5: Chăm sóc KH": [7, 8, 6, 9, 5],
}
df = pd.DataFrame(sample).set_index("Company")

# Điều chỉnh theo thông tin lô hàng
df_adj = df.copy()
if cargo_value > 50000:
    df_adj["C1: Tỷ lệ phí"] *= 1.2
if route in ["VN - US", "VN - EU"]:
    df_adj["C2: Thời gian xử lý"] *= 1.3
if good_type in ["Hàng nguy hiểm", "Điện tử"]:
    df_adj["C3: Tỷ lệ tổn thất"] *= 1.5


# -----------------------
# TOPSIS FUNCTION
# -----------------------
def topsis(df_data, weights, cost_flags):
    M = df_data[list(weights.index)].astype(float).values
    denom = np.sqrt((M ** 2).sum(axis=0)); denom[denom == 0] = 1
    R = M / denom
    V = R * weights.values
    is_cost = np.array([cost_flags[c] == "cost" for c in weights.index])
    ideal_best = np.where(is_cost, np.min(V, 0), np.max(V, 0))
    ideal_worst = np.where(is_cost, np.max(V, 0), np.min(V, 0))
    d_plus = np.sqrt(((V - ideal_best) ** 2).sum(1))
    d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(1))
    score = d_minus / (d_plus + d_minus + 1e-12)

    res = pd.DataFrame({
        'rank': range(1, len(df_data) + 1),
        'company': df_data.index,
        'score': score
    }).sort_values('score', ascending=False).reset_index(drop=True)

    return res


# -----------------------
# RUN ANALYSIS
# -----------------------
if st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True):
    with st.spinner("Đang tính toán bằng TOPSIS..."):

        result = topsis(df_adj, weights_series, cost_flags)

        result["ICC"] = result["score"].apply(lambda x: "ICC A" if x >= 0.75 else "ICC B" if x >= 0.5 else "ICC C")
        result["Risk"] = result["score"].apply(lambda x: "THẤP" if x >= 0.75 else "TRUNG BÌNH" if x >= 0.5 else "CAO")

        st.success("✅ HOÀN TẤT PHÂN TÍCH!")

        st.dataframe(result.set_index("rank"), use_container_width=True)

        fig = px.bar(
            result.sort_values("score"),
            x="score", y="company", color="score",
            color_continuous_scale="Blues", title="Xếp hạng công ty bảo hiểm"
        )
        st.plotly_chart(fig, use_container_width=True)

        best = result.iloc[0]
        st.markdown(f"""### 🏆 ĐỀ XUẤT:
        ✅ Công ty tối ưu: **{best['company']}**  
        ✅ Loại bảo hiểm: **{best['ICC']}**  
        ✅ Mức rủi ro: **{best['Risk']}**
        """)

        # -----------------------
        # EXPORT EXCEL
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result.to_excel(writer, sheet_name="Result", index=False)
            df_adj.to_excel(writer, sheet_name="Adjusted_Data")
        output.seek(0)

        st.download_button("📥 Xuất Excel", data=output,
                           file_name="riskcast_result.xlsx")

        # -----------------------
        # EXPORT PDF
        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 14)
                self.cell(0, 10, "BÁO CÁO ĐỀ XUẤT BẢO HIỂM - RISKCAST v2.5",
                          ln=True, align="C")

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, f"Giá trị hàng: {cargo_value:,} USD | Tuyến: {route}", ln=True)
        pdf.cell(0, 8, f"Phương thức: {method} | Ưu tiên: {priority}", ln=True)
        pdf.ln(6)

        pdf.set_font("Arial", "B", 10)
        pdf.cell(20, 7, "Rank", 1)
        pdf.cell(50, 7, "Company", 1)
        pdf.cell(30, 7, "Score", 1)
        pdf.cell(30, 7, "ICC", 1)
        pdf.cell(30, 7, "Risk", 1)
        pdf.ln()

        pdf.set_font("Arial", "", 9)
        for _, r in result.iterrows():
            pdf.cell(20, 7, str(int(r["rank"])), 1)
            pdf.cell(50, 7, r["company"], 1)
            pdf.cell(30, 7, f"{r['score']:.4f}", 1)
            pdf.cell(30, 7, r["ICC"], 1)
            pdf.cell(30, 7, r["Risk"], 1)
            pdf.ln()

        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("📄 Xuất PDF", data=pdf_bytes,
                           file_name="riskcast_report.pdf",
                           mime="application/pdf")
