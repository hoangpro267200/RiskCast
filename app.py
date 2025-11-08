# app.py
import io
import math
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Page + Theme (gradient-ish via CSS)
# ---------------------------
st.set_page_config(page_title="RISKCAST Demo", layout="wide")
# minimal gradient background and card style
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #001f3f 40%, #002b5c 100%);
        color: #e6f0ff;
    }
    .block-container {
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        background-image: linear-gradient(90deg,#00c6ff,#7b2ff7);
        color: white;
        border: none;
        padding: 8px 18px;
    }
    .stDownloadButton>button {
        background-image: linear-gradient(90deg,#7b2ff7,#00c6ff);
        color: white;
        border: none;
        padding: 8px 18px;
    }
    .dataframe th {
        background: rgba(255,255,255,0.06) !important;
        color: #e6f0ff;
    }
    .css-1d391kg { /* streamlit header hack */
        color: #e6f0ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 DEMO RISKCAST")
st.markdown("**Chào Hoàng — Hệ thống FAHP (approx) + TOPSIS để hỗ trợ quyết định mua bảo hiểm vận tải.**")
st.write("Giao diện: *Gradient Blue / Neo Cyber*  •  Kết quả: *Bảng xếp hạng + biểu đồ*")

# ---------------------------
# Upload area
# ---------------------------
st.info("📁 Upload file Excel (.xlsx) chứa 2 sheet: 1) Ma trận trọng số (FAHP/pairwise) hoặc [criterion, weight]. 2) Dữ liệu công ty (hàng = công ty, cột = tiêu chí numeric).")
uploaded_file = st.file_uploader("📂 Upload file Excel (.xlsx)", type=["xlsx"])

if not uploaded_file:
    st.stop()

# read sheets
try:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
except Exception as e:
    st.error(f"Lỗi khi đọc file Excel: {e}")
    st.stop()

st.success(f"File có {len(sheet_names)} sheet: {', '.join(sheet_names)}")
# choose sheets
weight_sheet = st.selectbox("📌 Chọn sheet chứa trọng số (FAHP)", sheet_names, index=0)
company_sheet = st.selectbox("📌 Chọn sheet chứa dữ liệu công ty (TOPSIS)", sheet_names, index=min(1, len(sheet_names)-1))

# read selected
try:
    df_weights = pd.read_excel(uploaded_file, sheet_name=weight_sheet)
    df_company = pd.read_excel(uploaded_file, sheet_name=company_sheet)
except Exception as e:
    st.error(f"Lỗi khi đọc các sheet: {e}")
    st.stop()

st.subheader("🧾 Trọng số (sheet chọn)")
st.dataframe(df_weights, use_container_width=True)
st.subheader("🏢 Dữ liệu công ty (sheet chọn)")
st.dataframe(df_company, use_container_width=True)

# ---------------------------
# Helper: compute weights from df_weights (FAHP approx)
# Accepts:
# - pairwise numeric matrix (square)
# - table [criterion, weight] in two columns
# Return: pandas Series indexed by criteria names
# ---------------------------
def compute_weights_from_df(df_w):
    """
    Returns pandas Series of normalized weights (sum 1).
    If df_w is square numeric -> geometric mean (approx AHP)
    Else if df_w has two columns -> take second column as weights.
    """
    # drop fully-nan rows/cols
    df = df_w.copy()
    # If two columns and non-square: treat as [criterion, weight]
    if df.shape[1] == 2 and not (df.shape[0] == df.shape[1]):
        col0 = df.columns[0]
        col1 = df.columns[1]
        crit = df[col0].astype(str).str.strip().tolist()
        w = pd.to_numeric(df[col1], errors='coerce').fillna(0).astype(float)
        if w.sum() == 0:
            raise ValueError("Trọng số nhập vào (sheet) đều là 0 hoặc không hợp lệ.")
        w = w / w.sum()
        return pd.Series(w.values, index=crit)
    # If square numeric matrix
    vals = df.select_dtypes(include=[np.number])
    if df.shape[0] == df.shape[1] and vals.shape[0] == df.shape[0]:
        A = vals.values.astype(float)
        # avoid zeros and negatives in pairwise? allow >0
        if np.any(A <= 0):
            # If matrix includes zeros or negatives, we fallback to reading column weights if possible
            # Try to use a 'weight' column if exists
            for c in df.columns:
                if 'weight' in str(c).lower():
                    alt = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)
                    if alt.sum() > 0:
                        return pd.Series(alt/alt.sum(), index=df.index.astype(str))
            raise ValueError("Ma trận pairwise chứa giá trị <= 0; ma trận FAHP cần dương.")
        # geometric mean method
        geo = np.prod(A, axis=1) ** (1.0 / A.shape[0])
        w = geo / geo.sum()
        # try to preserve index names
        idx = df.index.astype(str) if hasattr(df, "index") else [f"C{i}" for i in range(len(w))]
        return pd.Series(w, index=idx)
    # else: try if there's a named 'weight' column
    for c in df.columns:
        if 'weight' in str(c).lower():
            alt = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)
            if alt.sum() > 0:
                return pd.Series(alt/alt.sum(), index=df.index.astype(str))
    raise ValueError("Không hiểu định dạng sheet trọng số. Hãy dùng ma trận vuông pairwise numeric hoặc table [criterion, weight].")

# ---------------------------
# TOPSIS implementation
# ---------------------------
def topsis(df_data, weights, benefit_cols=None, cost_cols=None):
    """
    df_data: pandas DataFrame; rows = alternatives (companies), columns = criteria numeric
    weights: pandas Series indexed by criteria; will be aligned by name
    benefit_cols / cost_cols: optional lists of columns. If none provided, assume all benefit.
    Returns: DataFrame with score and rank
    """
    # Align columns
    data = df_data.copy()
    # ensure numeric columns only (use selected columns from weights intersection)
    common = [c for c in data.columns if c in weights.index]
    if len(common) == 0:
        # fallback: if weights are unnamed, assume order
        common = list(data.select_dtypes(include=[np.number]).columns)
        weights = pd.Series(weights.values[:len(common)], index=common)
    else:
        weights = weights.loc[common]
    # prepare matrix
    M = data[common].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
    # 1) Normalization (vector normalization)
    denom = np.sqrt((M ** 2).sum(axis=0))
    denom[denom == 0] = 1
    R = M / denom
    # 2) Weighting
    W = weights.values.astype(float)
    if np.isclose(W.sum(), 0):
        raise ValueError("Tổng trọng số bằng 0.")
    W = W / W.sum()
    V = R * W
    # identify benefit/cost
    if benefit_cols is None and cost_cols is None:
        # assume all benefit
        benefit_mask = np.array([True] * V.shape[1])
    else:
        benefit_mask = np.array([col in (benefit_cols or []) or (col not in (cost_cols or [])) for col in common])
    # 3) ideal best/worst
    ideal_best = np.max(V, axis=0)  # for benefit
    ideal_worst = np.min(V, axis=0)
    # if any cost criterion given, invert for those positions
    if cost_cols:
        for i, col in enumerate(common):
            if col in cost_cols:
                ideal_best[i] = np.min(V[:, i])
                ideal_worst[i] = np.max(V[:, i])
    # 4) distances
    d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    # 5) score
    score = d_minus / (d_plus + d_minus + 1e-12)
    # results
    res_df = pd.DataFrame(index=data.index.astype(str) if data.index is not None else range(len(score)))
    res_df['score'] = score
    res_df['distance+'] = d_plus
    res_df['distance-'] = d_minus
    res_df = res_df.sort_values('score', ascending=False)
    res_df['rank'] = range(1, len(res_df) + 1)
    # attach used criteria values
    for i, col in enumerate(common):
        res_df[col] = data[col].values
    return res_df.reset_index().rename(columns={'index': 'company'})

# ---------------------------
# Run block
# ---------------------------
st.markdown("---")
st.markdown("### ⚙️ Chạy phân tích")

# Provide optional selection: which criteria are cost (lower better) vs benefit (higher better)
all_numeric_cols = df_company.select_dtypes(include=[np.number]).columns.tolist()
st.info("Nếu có tiêu chí *cost* (như chi phí, thời gian), chọn chúng để TOPSIS coi là *cost* (nhỏ tốt). Các tiêu chí còn lại được xem là *benefit* (lớn tốt).")
cost_cols = st.multiselect("Chọn cột cost (cost = nhỏ tốt)", all_numeric_cols, default=[])

# Run button
if st.button("Phân tích rủi ro ngay"):
    with st.spinner("Đang chạy FAHP → TOPSIS..."):
        try:
            # compute weights
            w_series = compute_weights_from_df(df_weights)
            st.success("✅ Trọng số đã tính xong.")
            # try to align with company columns; if names differ, show both and ask mapping
            numeric_cols = df_company.select_dtypes(include=[np.number]).columns.tolist()
            # if criteria names in weights not matching numeric cols, try fuzzy mapping by exact lower-case match
            if not set(w_series.index).issubset(set(numeric_cols)):
                # attempt a simple mapping: lowercase match
                map_matches = {}
                for crit in w_series.index:
                    lc = crit.strip().lower()
                    found = [c for c in numeric_cols if c.strip().lower() == lc]
                    if found:
                        map_matches[crit] = found[0]
                if map_matches and len(map_matches) > 0:
                    # remap
                    st.info("🔁 Một số tiêu chí đã tự động ghép tên tương ứng.")
                    new_idx = []
                    for crit in w_series.index:
                        if crit in map_matches:
                            new_idx.append(map_matches[crit])
                        else:
                            # if not found, skip
                            st.warning(f"Không tìm thấy cột tương ứng cho tiêu chí: {crit}. Tiêu chí này sẽ bị bỏ.")
                    # keep only those found
                    valid_pairs = [(map_matches[k], v) for k, v in w_series.items() if k in map_matches]
                    if len(valid_pairs) == 0:
                        raise ValueError("Không tìm thấy cột numeric tương ứng với trọng số. Hãy đặt tên tiêu chí trùng nhau hoặc chỉnh file.")
                    names, vals = zip(*valid_pairs)
                    w_series = pd.Series(list(vals), index=list(names))
                else:
                    # Ask user to map manually (simple UI)
                    st.error("Tên tiêu chí trong sheet trọng số không khớp với tên cột numeric trong dữ liệu công ty.")
                    st.info("Hãy đảm bảo tên tiêu chí trùng nhau hoặc đặt file theo format [criterion, weight].")
                    st.stop()
            # Now run TOPSIS
            result = topsis(df_company, w_series, benefit_cols=None, cost_cols=cost_cols)
            st.success("✅ TOPSIS hoàn tất.")
            # Display ranking table
            st.subheader("🏆 Bảng xếp hạng (TOPSIS score)")
            st.dataframe(result[['company', 'score', 'rank'] + [c for c in result.columns if c not in ['company','score','rank','distance+','distance-']]], use_container_width=True)
            # Chart: bar of scores (top N)
            st.subheader("📊 Biểu đồ điểm TOPSIS")
            fig, ax = plt.subplots(figsize=(10, 4))
            # plot top 10
            topn = result.head(10).sort_values('score', ascending=True)
            ax.barh(topn['company'], topn['score'])
            ax.set_xlabel("TOPSIS Score")
            ax.set_title("Top 10 companies by TOPSIS score")
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            st.pyplot(fig, clear_figure=True)
            # Show best option
            best = result.iloc[0]
            st.markdown(f"### 🥇 **Lựa chọn tốt nhất:** **{best['company']}** — Score = **{best['score']:.4f}**")
            # Download results as Excel
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                result.to_excel(writer, sheet_name="ranking", index=False)
                df_company.to_excel(writer, sheet_name="company_raw", index=False)
                w_series.rename("weight").to_frame().to_excel(writer, sheet_name="weights", index=True)
            towrite.seek(0)
            st.download_button(label="⬇️ Xuất báo cáo", data=towrite, file_name="riskcast_topsis_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Có lỗi khi chạy thuật toán: {e}")
            st.text(traceback.format_exc())


    



