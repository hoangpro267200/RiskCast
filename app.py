import streamlit as st
import pandas as pd

st.set_page_config(page_title="🚢 RISKCAST", layout="wide")

st.title("🚢 RISKCAST — Fuzzy AHP + TOPSIS Demo")
st.write("Chào Hoàng, hệ thống đang sẵn sàng xử lý dữ liệu!")

uploaded_file = st.file_uploader("📂 Upload file Excel (xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📊 Dữ liệu đã upload")
    st.dataframe(df, use_container_width=True)

    st.success("✅ File đã được tải lên thành công!")
else:
    st.info("⬆️ Hãy upload file Excel để bắt đầu xử lý.")
