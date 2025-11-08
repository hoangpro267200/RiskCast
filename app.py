import streamlit as st
import pandas as pd
import numpy as np
import io
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

    
# ---------- PLACE THIS INSIDE YOUR STREAMLIT APP WHERE df_weights, df_company EXIST ----------
if st.button("🚀 Run FAHP + TOPSIS"):
    with st.spinner("Chạy FAHP → TOPSIS... Vui lòng chờ chút"):
        # 1) Chuẩn bị ma trận trọng số (df_weights) và ma trận công ty (df_company)
        try:
            # df_weights: có thể là ma trận vuông (pairwise) hoặc bảng [criterion, weight]
            W = None
            # nếu df_weights là ma trận vuông numeric
            mat = df_weights.copy()
            if mat.shape[0] == mat.shape[1] and np.all(np.isfinite(mat.select_dtypes(include=[np.number]).values)):
                # lấy phần số, đảm bảo theo thứ tự tiêu chí
                A = mat.values.astype(float)
                n = A.shape[0]
                # geometric mean method (approx)
                geo = np.prod(A, axis=1) ** (1.0 / n)
                w = geo / np.sum(geo)
                # nếu df_weights có index/cols tên tiêu chí thì dùng
                criteria = list(mat.index) if hasattr(mat, 'index') and len(mat.index) == n else [f"C{i+1}" for i in range(n)]
                W = pd.Series(w, index=criteria)
            else:
                # else thử coi có cột weight
                # tìm cột numeric
                numeric_cols = df_weights.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 1:
                    col = numeric_cols[0]
                    w_raw = df_weights[col].astype(float)
                    # nếu có tên tiêu chí ở cột đầu
                    if df_weights.shape[1] >= 2:
                        criteria = df_weights.iloc[:,0].astype(str).values
                        W = pd.Series(w_raw.values, index=criteria)
                    else:
                        W = pd.Series(w_raw.values, index=[f"C{i+1}" for i in range(len(w_raw))])
                    W = W / W.sum()
                else:
                    st.error("Không đọc được trọng số. Vui lòng kiểm tra file trọng số (FAHP).")
                    st.stop()

            # show weights
            st.subheader("📌 Trọng số (tính bằng geometric mean / FAHP-approx)")
            st.dataframe(W.rename("weight").to_frame())

            # 2) Chuẩn bị dữ liệu công ty (df_company) - chỉ lấy cột numeric tương ứng với tên tiêu chí
            # Nếu df_company có header tên tiêu chí giống W.index thì map trực tiếp
            df_num = df_company.copy()
            # chọn chỉ các cột numeric để TOPSIS (nếu có tên tiêu chí khớp với W)
            common = [c for c in df_num.columns if c in W.index]
            if len(common) == 0:
                # fallback: take numeric cols
                common = df_num.select_dtypes(include=[np.number]).columns.tolist()
                if len(common) == 0:
                    st.error("Không tìm thấy cột numeric trong sheet công ty. Kiểm tra file dữ liệu.")
                    st.stop()
                st.warning("Không tìm thấy tiêu chí khớp tên với trọng số; sử dụng tất cả cột số (numeric) trong sheet công ty.")
            X = df_num[common].astype(float).copy()
            X.index = df_num.index if df_num.index is not None else df_num.index

            # 3) Option: chọn tiêu chí là cost (giảm càng tốt)
            st.write("Nếu có tiêu chí *cost* (giảm tốt), chọn ở đây — nếu không, mặc định mọi tiêu chí là *benefit* (tăng tốt).")
            cost_cols = st.multiselect("Chọn các cột cost (chi phí)", options=common)

            # 4) Chuẩn hóa (vector normalization) và nhân trọng số
            # ensure order of weights matches columns
            W_for_X = []
            for c in common:
                if c in W.index:
                    W_for_X.append(W[c])
                else:
                    # nếu thiếu trọng số cho c thì gán trọng số bằng trung bình
                    W_for_X.append(1.0)
            W_for_X = np.array(W_for_X, dtype=float)
            # nếu bất kỳ weight là 1 (fallback) thì chuẩn hóa lại
            if W_for_X.sum() == 0:
                st.error("Tổng trọng số bằng 0, không thể tiếp tục.")
                st.stop()
            W_for_X = W_for_X / W_for_X.sum()

            # normalization
            norm = X.values / np.sqrt((X.values ** 2).sum(axis=0))
            # weighted normalized
            V = norm * W_for_X

            # 5) xác định PIS / NIS (ideal best/worst)
            # nếu benefit: ideal = max, if cost: ideal = min
            is_cost = np.array([c in cost_cols for c in common])
            ideal_best = np.max(V, axis=0).copy()
            ideal_worst = np.min(V, axis=0).copy()
            # but for cost columns invert
            for j, cost_flag in enumerate(is_cost):
                if cost_flag:
                    ideal_best[j] = np.min(V[:, j])
                    ideal_worst[j] = np.max(V[:, j])

            # 6) Khoảng cách đến ideal
            D_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
            D_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
            # tránh chia cho 0
            score = D_minus / (D_plus + D_minus + 1e-12)

            # 7) Kết quả: tạo dataframe
            result = pd.DataFrame({
                "company": X.index.astype(str),
                **{f"{common[j]}": X.iloc[:, j].values for j in range(len(common))},
                "score": score
            }).set_index("company")
            result["rank"] = result["score"].rank(ascending=False, method="min").astype(int)
            result = result.sort_values(["score"], ascending=False)

            st.subheader("🏁 Kết quả TOPSIS (score & ranking)")
            st.dataframe(result)

            # 8) Biểu đồ bar của score
            st.subheader("📊 Biểu đồ điểm (closeness score)")
            st.bar_chart(result["score"])

            # 9) Cho tải file Excel kết quả
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                result.to_excel(writer, sheet_name="TOPSIS_result")
                df_weights.to_excel(writer, sheet_name="weights_raw")
                df_company.to_excel(writer, sheet_name="company_raw")
            towrite.seek(0)
            st.download_button(label="⬇️ Tải file Excel kết quả", data=towrite, file_name="riskcast_topsis_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.success("Hoàn tất: FAHP-approx + TOPSIS đã chạy.")
        except Exception as e:
            st.error(f"Có lỗi khi chạy thuật toán: {e}")
            import traceback
            st.text(traceback.format_exc())
# ---------- END BLOCK ----------


