import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="RISKCAST Demo", layout="wide")

st.title("🚢 RISKCAST – Demo Web App")
st.write("Chào Hoàng, hệ thống đã sẵn sàng xử lý dữ liệu bảo hiểm!")

# STEP 1 — Upload Excel file
uploaded_file = st.file_uploader("📂 Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    st.success("✅ File đã upload thành công!")
    st.write(f"👉 File này có **{len(sheet_names)} sheet**: {', '.join(sheet_names)}")

    # chọn sheet
    weight_sheet = st.selectbox("📌 Chọn sheet chứa trọng số (Fuzzy AHP)", sheet_names)
    company_sheet = st.selectbox("🏢 Chọn sheet chứa dữ liệu công ty (TOPSIS)", sheet_names)

    # hiển thị data
    df_weights = pd.read_excel(uploaded_file, sheet_name=weight_sheet)
    df_company = pd.read_excel(uploaded_file, sheet_name=company_sheet)

    st.subheader("📊 Trọng số (FAHP)")
    st.dataframe(df_weights, use_container_width=True)

    st.subheader("🏢 Dữ liệu công ty (TOPSIS)")
    st.dataframe(df_company, use_container_width=True)

    # RUN FAHP + TOPSIS
    if st.button("🚀 Chạy mô hình FAHP + TOPSIS"):

        with st.spinner("⏳ Đang xử lý… vui lòng đợi…"):

            try:
                # ========== STEP 1: FAHP ==============
                mat = df_weights.copy()

                # Kiểm tra numeric
                if np.all(np.isfinite(mat.select_dtypes(include=[np.number]).values)):
                    A = mat.values.astype(float)
                    n = A.shape[0]

                    # geometric mean
                    geo = np.prod(A, axis=1)**(1/n)
                    W = geo / np.sum(geo)

                    criteria = mat.index if hasattr(mat, "index") else [f"C{i+1}" for i in range(n)]

                else:
                    raise ValueError("❌ Sheet trọng số không phải số!")

                # ========== STEP 2: TOPSIS ==============
                df = df_company.copy()

                cols = df.select_dtypes(include=[np.number]).columns
                data = df[cols].values.astype(float)

                norm = data / np.sqrt((data**2).sum(axis=0))
                weighted = norm * W.reshape(-1, 1)

                ideal_best = np.max(weighted, axis=0)
                ideal_worst = np.min(weighted, axis=0)

                dist_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
                dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))

                score = dist_worst / (dist_best + dist_worst)

                df_result = df_company.copy()
                df_result["TOPSIS Score"] = score
                df_result["Rank"] = df_result["TOPSIS Score"].rank(ascending=False).astype(int)
                df_result = df_result.sort_values(by="Rank")

                # ========== EXPORT EXCEL ==========
                output = io.BytesIO()
                writer = pd.ExcelWriter(output, engine="openpyxl")

                df_weights.to_excel(writer, sheet_name="FAHP_raw")
                df_company.to_excel(writer, sheet_name="Company_raw")
                df_result.to_excel(writer, sheet_name="TOPSIS_result", index=False)

                writer.close()
                output.seek(0)

                st.download_button(
                    label="⬇️ Tải file Excel kết quả",
                    data=output,
                    file_name="riskcast_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.success("✅ Hoàn tất: FAHP + TOPSIS đã chạy xong.")

            except Exception as e:
                st.error(f"❌ Có lỗi khi chạy thuật toán: {e}")


    



