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
    st.write(f"File này có **{len(sheet_names)} sheet**: {', '.join(sheet_names)}")

    # Hiển thị từng sheet
    for sheet in sheet_names:
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        st.subheader(f"📄 Sheet: {sheet}")
        st.dataframe(df, use_container_width=True)

else:
    st.info("⬆️ Hãy upload file Excel để hệ thống xử lý.")
