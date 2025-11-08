import streamlit as st
import pandas as pd

st.set_page_config(page_title="RISKCAST Demo", layout="wide")

st.title("🚢 RISKCAST — Demo Web App")
st.write("Chào Hoàng, hệ thống đã sẵn sàng xử lý dữ liệu bảo hiểm!")

# STEP 1 — Upload file
uploaded_file = st.file_uploader("📂 Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    st.success("✅ File đã upload thành công!")

    # Chọn sheet trọng số & sheet công ty
    weight_sheet = st.selectbox("📌 Chọn sheet chứa trọng số (Fuzzy AHP)", sheet_names)
    company_sheet = st.selectbox("🏢 Chọn sheet chứa dữ liệu công ty (TOPSIS)", sheet_names)

    # Hiển thị 2 sheet đã chọn
    df_weights = pd.read_excel(uploaded_file, sheet_name=weight_sheet)
    df_company = pd.read_excel(uploaded_file, sheet_name=company_sheet)

    st.subheader("📊 Trọng số (FAHP)")
    st.dataframe(df_weights, use_container_width=True)

    st.subheader("🏢 Dữ liệu công ty (TOPSIS)")
    st.dataframe(df_company, use_container_width=True)

    # Button xử lý thuật toán
    if st.button("🚀 Run FAHP + TOPSIS"):
        st.success("✅ Thuật toán đang chạy... chuẩn bị dữ liệu đầu vào!")
        # (chỗ này tí nữa mình sẽ thêm thuật toán FAHP + TOPSIS)
else:
    st.info("⬆️ Hãy upload file Excel để hệ thống xử lý.")

