import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="RISKCAST Demo – Fuzzy AHP + TOPSIS", layout="wide")

# ====== UI ======
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4843/4843098.png", width=80)
    st.title("🚢 RISKCAST")
    st.write("AI Web App hỗ trợ quyết định mua bảo hiểm (FAHP + TOPSIS)")
    st.markdown("---")
    st.write("👤 Owner: **Hoàng Bùi (R&D)**")
    st.write("🧠 Strategy: Risk-based decision + Optimization")
    st.markdown("---")
    st.write("📄 Upload Excel để xử lý")

st.title("🚢 DEMO RISKCAST")
st.write("Chào **Hoàng**, hệ thống đã sẵn sàng xử lý dữ liệu bảo hiểm!")

# STEP 1 — Upload file Excel
uploaded_file = st.file_uploader("📂 Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    st.success(f"✅ File đã upload thành công ({len(sheet_names)} sheets)")
    st.write("Sheets:", ", ".join(sheet_names))

    # chọn sheet
    weight_sheet = st.selectbox("📌 Chọn sheet chứa trọng số (FAHP)", sheet_names)
    company_sheet = st.selectbox("🏢 Chọn sheet chứa dữ liệu công ty (TOPSIS)", sheet_names)

    # load dữ liệu
    df_weights = pd.read_excel(uploaded_file, sheet_name=weight_sheet)
    df_company = pd.read_excel(uploaded_file, sheet_name=company_sheet)

    st.subheader("📊 Trọng số (FAHP)")
    st.dataframe(df_weights, use_container_width=True)

    st.subheader("🏢 Dữ liệu công ty (TOPSIS)")
    st.dataframe(df_company, use_container_width=True)

    # ==============================================
    # 🚀 FAHP + TOPSIS
    # ==============================================
    if st.button("🚀 Chạy FAHP + TOPSIS"):
        with st.spinner("Đang xử lý..."):

            # ----- FAHP → tính trọng số -----
            try:
                mat = df_weights.copy()
                A = mat.values.astype(float)
                n = A.shape[0]

                geo = np.prod(A, axis=1) ** (1.0 / n)
                w = geo / np.sum(geo)      # final weight FAHP

                criteria = list(df_company.columns[1:])
                company = df_company.iloc[:, 0]

                # ----- TOPSIS -----
                X = df_company.iloc[:, 1:].astype(float)

                norm = X / np.sqrt((X**2).sum())
                weighted = norm * w

                ideal_best = weighted.max()
                ideal_worst = weighted.min()

                dist_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
                dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))

                score = dist_worst / (dist_best + dist_worst)

                df_result = pd.DataFrame({
                    "Company": company,
                    "TOPSIS Score": score
                }).sort_values(by="TOPSIS Score", ascending=False)

                st.subheader("🏆 KẾT QUẢ XẾP HẠNG (TOPSIS)")
                st.dataframe(df_result, use_container_width=True)

                # ===== BIỂU ĐỒ RANKING =====
                st.subheader("📈 Ranking Chart")
                fig, ax = plt.subplots()
                ax.bar(df_result["Company"], df_result["TOPSIS Score"])
                ax.set_ylabel("Score")
                ax.set_xlabel("Company")
                ax.set_title("TOPSIS Ranking Result")
                st.pyplot(fig)

                # ===== Xuất Excel =====
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                    df_weights.to_excel(writer, sheet_name="weight_raw")
                    df_company.to_excel(writer, sheet_name="company_raw")
                    df_result.to_excel(writer, sheet_name="topsis_result", index=False)
                towrite.seek(0)

                st.download_button(
                    label="⬇️ Tải file Excel kết quả",
                    data=towrite,
                    file_name="riskcast_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                st.success("🎉 Hoàn tất FAHP + TOPSIS!")

            except Exception as e:
                st.error(f"❌ Có lỗi khi chạy thuật toán: **{e}**")


    



