import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="🚢 RISKCAST", layout="wide")

st.title("🚢 DEMO RISKCAST")
st.write("Chào Hoàng, hệ thống đang sẵn sàng xử lý dữ liệu!")

uploaded_file = st.file_uploader("📂 Upload file Excel (xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("📊 Dữ liệu gốc")
    st.dataframe(df, use_container_width=True)

    st.subheader("🔧 Normalize dữ liệu (Min-Max)")

    # Normalize từng cột (trừ cột đầu nếu là tên công ty)
    df_norm = df.copy()
    for col in df.columns[1:]:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    st.dataframe(df_norm, use_container_width=True)

    st.success("✅ Normalize thành công! Tiếp theo sẽ là Fuzzy AHP.")
else:
    st.info("⬆️ Hãy upload file Excel để bắt đầu xử lý.")
