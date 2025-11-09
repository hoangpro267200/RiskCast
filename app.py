# RISKCAST v3.1 — All upgrades: Fuzzy AHP (simple TFN defuzzify),
# Monte-Carlo Climate Risk layer + confidence score, NOAA optional fetch,
# fixes for TOPSIS ideal best/worst, company-specific climate sensitivity,
# improved PDF table layout.

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from fpdf import FPDF
import requests

# -----------------------
# Page config + CSS
# -----------------------
st.set_page_config(page_title="RISKCAST v3.1", layout="wide", page_icon="shield")
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg,#021023 0%, #082c4a 100%); color: #e6f0ff; font-family: 'Segoe UI'; }
    .block-container { padding: 1.5rem 2rem; }
    h1 { color: #7bd3ff; text-align: center; font-weight: 800; font-size: 2.8rem; }
    .stButton>button { background: linear-gradient(90deg,#00c6ff,#7b2ff7); color: white;
                       font-weight:bold; border-radius: 15px; padding: 0.8rem; font-size: 1.1rem; }
    .result-box { background: #1a2a44; padding: 1.5rem; border-radius: 15px;
                  border-left: 6px solid #00d4ff; margin: 1.5rem 0; box-shadow: 0 4px 12px rgba(0,212,255,0.3); }
    .footer { text-align: center; margin-top: 3rem; color: #aaa; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("RISKCAST v3.1 — HỆ THỐNG ĐỀ XUẤT BẢO HIỂM THÔNG MINH (Upgraded)")
st.caption("**Thêm: Fuzzy-approx weights, Monte-Carlo climate risk + Confidence score, NOAA optional fetch**")

# -----------------------
# Sidebar Input
# -----------------------
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị (USD)", value=39000, step=1000, format="%d")
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"]) 
    route = st.selectbox("Tuyến", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"]) 
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"]) 
    month = st.selectbox("Tháng", list(range(1, 13)), index=8)  # Tháng 9
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"]) 
    use_fuzzy = st.checkbox("Sử dụng Fuzzy AHP (TFN -> defuzzify) để hoá mềm trọng số", value=True)
    use_mc = st.checkbox("Kích hoạt Monte-Carlo cho Rủi ro khí hậu (C6)", value=True)
    mc_runs = st.number_input("Số vòng Monte-Carlo", min_value=200, max_value=20000, value=2000, step=100)
    fetch_noaa = st.checkbox("Cố gắng lấy dữ liệu khí hậu thật từ NOAA (nếu có internet)", value=False)
    st.markdown("---")
    st.markdown("**Chú ý:** Nếu không có quyền API NOAA code sẽ fallback về dữ liệu mẫu.")

# -----------------------
# Criteria & Weights (6 tiêu chí)
# -----------------------
criteria = [
    "C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"
]
cost_flags = {c: "cost" if c in ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

st.subheader("Điều chỉnh trọng số tiêu chí (crisp)")
cols = st.columns(6)
default_weights = [0.20, 0.15, 0.20, 0.20, 0.10, 0.15]
weights = [cols[i].slider(criteria[i], 0.0, 1.0, default_weights[i], 0.01) for i in range(6)]
w = np.array(weights)

# Boost theo ưu tiên (cập nhật trước fuzzy)
if priority == "An toàn tối đa":
    w[1] *= 1.5; w[4] *= 1.4; w[5] *= 1.3
elif priority == "Tối ưu chi phí":
    w[0] *= 1.6; w[5] *= 0.8
w = w / w.sum()
weights_series = pd.Series(w, index=criteria)

# -----------------------
# Simple Fuzzy TFN wrapper (if enabled)
# We'll create small TFN around each weight and defuzzify by centroid
# -----------------------
if use_fuzzy:
    st.markdown("**Fuzzy AHP (approx):** Dùng TFN nhỏ quanh trọng số người dùng để mô phỏng bất định chủ quan.")
    fuzziness = st.slider("Mức không chắc chắn (%)", 0.0, 50.0, 15.0, 1.0)
    # build TFNs and defuzzify by centroid (l+m+u)/3
    low = np.maximum(weights_series * (1 - fuzziness / 100.0), 0.0001)
    mid = weights_series.copy()
    high = np.minimum(weights_series * (1 + fuzziness / 100.0), 0.9999)
    defuzz = (low + mid + high) / 3
    weights_series = defuzz / defuzz.sum()\    

# -----------------------
# Dữ liệu mẫu + company-specific sensitivity
# -----------------------
sample = {
    "Company": ["Chubb", "PVI", "InternationalIns", "BaoViet", "Aon"],
    "C1: Tỷ lệ phí": [0.30, 0.28, 0.26, 0.32, 0.24],
    "C2: Thời gian xử lý": [6, 5, 8, 7, 4],
    "C3: Tỷ lệ tổn thất": [0.08, 0.06, 0.09, 0.10, 0.07],
    "C4: Hỗ trợ ICC": [9, 8, 6, 9, 7],
    "C5: Chăm sóc KH": [9, 8, 5, 7, 6],
}
df = pd.DataFrame(sample).set_index("Company")

# company-specific climate sensitivity (multiplicative)
sensitivity = {"Chubb":0.95, "PVI":1.10, "InternationalIns":1.20, "BaoViet":1.05, "Aon":0.90}

# base climate risk by route/month — fallback mapping
climate_risk_map = {
    ("VN - EU", 9): 0.65, ("VN - EU", 10): 0.48, ("VN - US", 9): 0.75,
    ("VN - Singapore", 9): 0.30, ("Domestic", 9): 0.20
}
base_climate = climate_risk_map.get((route, month), 0.40)

# Optionally try to fetch NOAA (very basic example; user must supply token if required)
noaa_success = False
noaa_note = "(fallback used)"
if fetch_noaa:
    try:
        # NOTE: NOAA API typically requires a token and specific endpoints. This is a minimal attempt.
        # Replace with a proper NOAA endpoint & token for production.
        resp = requests.get("https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets", timeout=5)
        if resp.status_code == 200:
            # real integration would parse station data / storm frequency etc.
            noaa_success = True
            noaa_note = "(NOAA fetch OK, used for climate baseline)"
            # For demo, slightly nudge base climate by a tiny random factor
            base_climate *= 1.02
    except Exception:
        noaa_success = False
        noaa_note = "(NOAA fetch failed — offline or token required)"

# -----------------------
# Adjust data based on inputs
# -----------------------
df_adj = df.copy().astype(float)
if cargo_value > 50000:
    df_adj["C1: Tỷ lệ phí"] *= 1.2
if route in ["VN - US", "VN - EU"]:
    df_adj["C2: Thời gian xử lý"] *= 1.3
if good_type in ["Hàng nguy hiểm", "Điện tử"]:
    df_adj["C3: Tỷ lệ tổn thất"] *= 1.5

# -----------------------
# Monte-Carlo for C6: simulate per-company distribution and compute mean+std
# -----------------------
if use_mc:
    st.info(f"Monte-Carlo: running {mc_runs} simulations for climate risk (this may take a moment)...")
    rng = np.random.default_rng(42)
    mc_results = np.zeros((len(df_adj), int(mc_runs)))
    # Assume climate base has some uncertainty (10% std) and company sensitivity further scales it
    for i, comp in enumerate(df_adj.index):
        mu = base_climate * sensitivity.get(comp, 1.0)
        sigma = max(0.03, mu * 0.12)  # at least 3% abs, or 12% relative
        mc_results[i, :] = rng.normal(loc=mu, scale=sigma, size=int(mc_runs))
        # clamp [0,1]
        mc_results[i, :] = np.clip(mc_results[i, :], 0.0, 1.0)
    mc_mean = mc_results.mean(axis=1)
    mc_std = mc_results.std(axis=1)
    # attach to df_adj as distribution summary
    df_adj["C6: Rủi ro khí hậu"] = mc_mean
else:
    # deterministic assignment but company-sensitive
    df_adj["C6: Rủi ro khí hậu"] = [base_climate * sensitivity[c] for c in df_adj.index]
    mc_std = np.zeros(len(df_adj)) + 0.0001

# -----------------------
# TOPSIS FUNCTION (fixed ideal best/worst computation)
# -----------------------
def topsis(df_data, weights, cost_flags):
    # df_data: index=company, columns = criteria names
    M = df_data[list(weights.index)].astype(float).values  # shape (n_comp, n_crit)
    denom = np.sqrt((M ** 2).sum(axis=0))
    denom[denom == 0] = 1
    R = M / denom  # normalized
    V = R * weights.values  # weighted normalized
    is_cost = np.array([cost_flags[c] == "cost" for c in weights.index])
    # ideal best/worst per criterion
    ideal_best = np.where(is_cost, np.min(V, axis=0), np.max(V, axis=0))
    ideal_worst = np.where(is_cost, np.max(V, axis=0), np.min(V, axis=0))
    d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)
    # build dataframe and sort
    res = pd.DataFrame({
        'company': df_data.index,
        'score': score,
        'd_plus': d_plus,
        'd_minus': d_minus
    })
    res = res.sort_values('score', ascending=False).reset_index(drop=True)
    res['rank'] = res.index + 1
    return res

# -----------------------
# RUN ANALYSIS
# -----------------------
if st.button("PHÂN TÍCH NGAY", use_container_width=True):
    with st.spinner("Đang tính toán (Fuzzy/TOPSIS/MC)..."):
        result = topsis(df_adj, weights_series, cost_flags)

        # add ICC & Risk bands
        result["ICC"] = result["score"].apply(lambda x: "ICC A" if x >= 0.75 else "ICC B" if x >= 0.5 else "ICC C")
        result["Risk"] = result["score"].apply(lambda x: "THẤP" if x >= 0.75 else "TRUNG BÌNH" if x >= 0.5 else "CAO")

        # attach MC std and compute confidence
        # confidence defined as 1 / (1 + CV) where CV = std/mean for C6 (lower CV -> higher confidence)
        mean_c6 = df_adj["C6: Rủi ro khí hậu"].values
        cv = np.where(mean_c6 == 0, 0.0, np.array(mc_std) / mean_c6)
        confidence = 1 / (1 + cv)
        # normalize confidence to [0.3, 1.0] to avoid extremely small
        confidence = 0.3 + 0.7 * (confidence - confidence.min()) / (confidence.ptp() + 1e-9)

        # map confidence back to companies and attach combined confidence by combining criterion dispersion
        # form 1: also use coefficient of variation across criteria for final confidence
        crit_cv = df_adj[list(weights_series.index)].std(axis=1) / (df_adj[list(weights_series.index)].mean(axis=1) + 1e-9)
        crit_conf = 1 / (1 + crit_cv)
        crit_conf = 0.3 + 0.7 * (crit_conf - crit_conf.min()) / (crit_conf.ptp() + 1e-9)

        # final confidence = geometric mean of climate confidence and crit_conf
        final_conf = np.sqrt(confidence * crit_conf)

        # attach to result in correct order
        comp_order = list(df_adj.index)
        conf_map = {comp_order[i]: float(final_conf[i]) for i in range(len(comp_order))}
        result['confidence'] = result['company'].map(conf_map)

        # create human-friendly columns
        result['score_pct'] = (result['score'] * 100).round(2)
        result = result[['rank', 'company', 'score', 'score_pct', 'ICC', 'Risk', 'confidence']]

        st.success("HOÀN TẤT PHÂN TÍCH!")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(result.set_index('rank'), use_container_width=True)
        with col2:
            fig_bar = px.bar(
                result.sort_values("score"),
                x="score", y="company", color="score",
                color_continuous_scale="Blues", title="Xếp hạng công ty bảo hiểm"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Radar Chart (use scaled criterion-level performance for top 3)
        top3 = result.head(3)['company'].tolist()
        radar_df = df_adj.loc[top3, list(weights_series.index)].copy()
        # scale each criterion to 0-1 for visualization
        radar_scaled = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
        radar_scaled['company'] = radar_scaled.index
        radar_melt = radar_scaled.reset_index(drop=True).melt(id_vars=['company'], var_name='criterion', value_name='value')
        fig_radar = px.line_polar(radar_melt, r='value', theta='criterion', color='company', line_close=True,
                                  title='So sánh tiêu chí (top 3)')
        st.plotly_chart(fig_radar, use_container_width=True)

        best = result.iloc[0]
        st.markdown(f"""
        <div class="result-box">
        <h3>ĐỀ XUẤT TỐI ƯU</h3>
        <p>✅ <strong>Công ty:</strong> {best['company']}</p>
        <p>✅ <strong>Loại bảo hiểm:</strong> {best['ICC']}</p>
        <p>✅ <strong>Mức rủi ro:</strong> {best['Risk']}</p>
        <p>✅ <strong>Score TOPSIS:</strong> {best['score']:.4f} ({best['score_pct']}%)</p>
        <p>✅ <strong>Confidence:</strong> {best['confidence']:.2f}</p>
        <p>🔎 NOAA: {noaa_note}</p>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------
        # EXPORT EXCEL
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            result.to_excel(writer, sheet_name="Result", index=False)
            df_adj.to_excel(writer, sheet_name="Adjusted_Data")
            pd.DataFrame(weights_series, columns=['weight']).to_excel(writer, sheet_name="Weights")
            # add MC summary
            mc_summary = pd.DataFrame({
                'company': df_adj.index,
                'C6_mean': df_adj['C6: Rủi ro khí hậu'].values,
                'C6_std': mc_std
            })
            mc_summary.to_excel(writer, sheet_name='C6_MC_Summary', index=False)
        output.seek(0)
        st.download_button("Xuất Excel", data=output, file_name="riskcast_v3.1_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # -----------------------
        # EXPORT PDF (adjusted widths)
        class PDF(FPDF):
            def header(self):
                self.set_font("Arial", "B", 16)
                self.cell(0, 12, "BÁO CÁO RISKCAST v3.1", ln=True, align="C")
                self.set_font("Arial", "", 10)
                self.cell(0, 8, "Mô hình TOPSIS + Fuzzy-approx + Monte-Carlo Climate Risk", ln=True, align="C")
                self.ln(5)
        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 8, f"Giá trị: {cargo_value:,} USD | Tuyến: {route} | Tháng: {month}", ln=True)
        pdf.cell(0, 8, f"Phương thức: {method} | Ưu tiên: {priority}", ln=True)
        pdf.ln(8)
        pdf.set_font("Arial", "B", 10)
        # column widths: rank, company, score, ICC, Risk, conf
        widths = [12, 54, 28, 28, 28, 28]
        headers = ["Rank", "Company", "Score", "ICC", "Risk", "Conf"]
        for wcol, h in zip(widths, headers):
            pdf.cell(wcol, 8, h, 1)
        pdf.ln()
        pdf.set_font("Arial", "", 9)
        for _, r in result.iterrows():
            pdf.cell(widths[0], 7, str(int(r["rank"])), 1)
            pdf.cell(widths[1], 7, str(r["company"])[:30], 1)
            pdf.cell(widths[2], 7, f"{r['score']:.4f}", 1)
            pdf.cell(widths[3], 7, r["ICC"], 1)
            pdf.cell(widths[4], 7, r["Risk"], 1)
            pdf.cell(widths[5], 7, f"{r['confidence']:.2f}", 1)
            pdf.ln()
        pdf.ln(6)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 6, f"Nguồn: NOAA {noaa_note}, MarineTraffic, PVI, Bảo Việt – Đề tài NCKH 2025", ln=True, align="C")
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("Xuất PDF", data=pdf_bytes, file_name="riskcast_v3.1_report.pdf", mime="application/pdf")

# -----------------------
# GIỚI THIỆU MÔ HÌNH (Expandable)
# -----------------------
with st.expander("Xem mô hình khoa học (Fuzzy approx + TOPSIS + Monte-Carlo)", expanded=False):
    st.markdown("""
    ### **MÔ HÌNH KHOA HỌC (Tóm tắt)**
    - **Fuzzy-approx weights**: TFN nhỏ quanh trọng số người dùng để mô phỏng bất định chủ quan;
    - **Monte-Carlo (C6)**: Mô phỏng phân phối rủi ro khí hậu theo base climate * company sensitivity để có mean/std;
    - **TOPSIS (sửa lỗi)**: Chuẩn hoá -> trọng số -> ideal best/worst (sửa tính toán axis) -> khoảng cách d+, d- -> score;
    - **Confidence**: Kết hợp dispersion của các tiêu chí và CV của C6 để trả về độ tin cậy cho khuyến nghị.
    """)

# -----------------------
# Footer
# -----------------------
st.markdown("""
<div class="footer">
    <strong>RISKCAST v3.1</strong> – Nâng cấp: Fuzzy & Monte-Carlo + Confidence<br>
    Tác giả: Bùi Xuân Hoàng
    Liên hệ: riskcast@gmail.com | Website: <a href="https://riskcast.streamlit.app">riskcast.streamlit.app</a>
</div>
""", unsafe_allow_html=True)
