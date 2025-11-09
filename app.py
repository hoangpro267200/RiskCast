# app.py
# RISKCAST v3.4 — Full, cleaned, export charts into PDF safely via fig.to_image()
# Requires: streamlit, plotly, numpy, pandas, fpdf2, kaleido
# pip install streamlit plotly numpy pandas fpdf2 kaleido

import io
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from fpdf import FPDF

# -------------------------
# Page config and style
# -------------------------
st.set_page_config(page_title="RISKCAST v3.4", layout="wide", page_icon="🛡️")
st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg,#00101F 0%, #0E2A47 100%); color: #E7F4FF; }
  h1 { color: #7bd3ff; text-align: center; }
  .footer { text-align:center; margin-top: 2rem; color:#aaa; font-size:0.9rem; }
  .small { font-size:12px; color:#9ac7ff; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ RISKCAST v3.4 — Hệ thống đề xuất bảo hiểm (Final)")

# -------------------------
# Sidebar — Inputs
# -------------------------
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị lô hàng (USD)", value=39000, step=1000, format="%d")
    good_type = st.selectbox("Loại hàng", ["Điện tử","Đông lạnh","Hàng khô","Hàng nguy hiểm","Khác"])
    route = st.selectbox("Tuyến vận chuyển", ["VN - EU","VN - US","VN - Singapore","VN - China","Domestic"])
    method = st.selectbox("Phương thức vận tải", ["Sea","Air","Truck"])
    month = st.selectbox("Tháng vận chuyển", list(range(1,13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa","Cân bằng","Tối ưu chi phí"])
    use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN → Defuzzify)", value=True)
    use_mc = st.checkbox("Bật Monte-Carlo rủi ro khí hậu", value=True)
    mc_runs = st.number_input("Số vòng Monte-Carlo", min_value=200, max_value=20000, value=2000, step=100)
    st.markdown("---")
    st.markdown("<div class='small'>Lưu ý: để xuất ảnh từ Plotly vào PDF cần package <code>kaleido</code>.</div>", unsafe_allow_html=True)

# -------------------------
# Criteria setup
# -------------------------
criteria = [
    "C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
    "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"
]

criteria_tooltip = {
    "C1: Tỷ lệ phí": "Phí bảo hiểm — càng thấp càng tốt.",
    "C2: Thời gian xử lý": "Tốc độ xử lý claim — càng nhanh càng tốt.",
    "C3: Tỷ lệ tổn thất": "Tỷ lệ từ chối/thiệt hại — càng thấp càng tốt.",
    "C4: Hỗ trợ ICC": "Phạm vi ICC (A/B/C) — càng rộng càng tốt.",
    "C5: Chăm sóc KH": "Dịch vụ hỗ trợ khách hàng — càng tốt càng an tâm.",
    "C6: Rủi ro khí hậu": "Rủi ro khí hậu/tuyến/tháng — càng thấp càng tốt."
}

cost_flags = {c: "cost" if c in ["C1: Tỷ lệ phí", "C3: Tỷ lệ tổn thất", "C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

# -------------------------
# Smart slider + lock + input + reset
# -------------------------
st.subheader("🎛️ Phân bổ trọng số (Smart Slider + Lock + Reset)")

# defaults
default_weights = np.array([0.20, 0.15, 0.20, 0.20, 0.10, 0.15], dtype=float)

if "weights" not in st.session_state:
    st.session_state.weights = default_weights.copy()
if "locked" not in st.session_state:
    st.session_state.locked = [False]*6

w = st.session_state.weights.copy()
locked = st.session_state.locked

# Reset control
if st.button("🔄 Reset trọng số về mặc định"):
    st.session_state.weights = default_weights.copy()
    st.session_state.locked = [False]*6
    st.experimental_rerun()

cols = st.columns(6)
for i in range(6):
    with cols[i]:
        st.markdown(f"**{criteria[i]}**  \n<small style='color:#86c6ff'>{criteria_tooltip[criteria[i]]}</small>", unsafe_allow_html=True)
        locked[i] = st.checkbox("🔒 Lock", value=locked[i], key=f"lock_{i}")
        inp = st.number_input("Nhập tỉ lệ", min_value=0.0, max_value=1.0, value=float(w[i]), step=0.01, key=f"inp_{i}")
        slid = st.slider("", min_value=0.0, max_value=1.0, value=float(inp), step=0.01, key=f"sl_{i}")

        if not locked[i]:
            diff = slid - w[i]
            w[i] = slid
            # adjust open indices proportionally
            idx_open = [j for j in range(6) if j != i and not locked[j]]
            if len(idx_open) > 0:
                remain = w[idx_open]
                total = remain.sum()
                # avoid divide by zero
                if total <= 0:
                    # distribute equally among open
                    w[idx_open] = (1.0 - w[i]) / len(idx_open)
                else:
                    w[idx_open] = remain * ((total - diff) / max(total, 1e-12))

# Final normalize for numeric stability
w = w / w.sum()
st.session_state.weights = w
st.session_state.locked = locked
weights_series = pd.Series(w, index=criteria)

# Realtime charts for weights
st.write("### Biểu đồ phân bố trọng số (Realtime)")
dfw = pd.DataFrame({"criterion": criteria, "weight": w})
fig_weights_bar = px.bar(dfw, x="weight", y="criterion", orientation="h", color="weight", color_continuous_scale="Blues")
fig_weights_radar = px.line_polar(dfw, r="weight", theta="criterion", line_close=True, title="Radar trọng số")

colA, colB = st.columns(2)
with colA:
    st.plotly_chart(fig_weights_bar, use_container_width=True)
with colB:
    st.plotly_chart(fig_weights_radar, use_container_width=True)

# -------------------------
# Example data (company matrix)
# -------------------------
sample = {
    "Company": ["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí": [0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý": [6,5,8,7,4],
    "C3: Tỷ lệ tổn thất": [0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC": [9,8,6,9,7],
    "C5: Chăm sóc KH": [9,8,5,7,6]
}
df = pd.DataFrame(sample).set_index("Company")

# sensitivity per company for climate
sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}
climate_map = {("VN - EU",9):0.65, ("VN - US",9):0.75, ("Domestic",9):0.20}
base_climate = climate_map.get((route, month), 0.40)

# small automatic adjustments
if cargo_value > 50000:
    df["C1: Tỷ lệ phí"] *= 1.2
if route in ["VN - US", "VN - EU"]:
    df["C2: Thời gian xử lý"] *= 1.3
if good_type in ["Điện tử", "Hàng nguy hiểm"]:
    df["C3: Tỷ lệ tổn thất"] *= 1.5

# -------------------------
# Monte-Carlo for C6
# -------------------------
if use_mc:
    rng = np.random.default_rng(42)
    mc_runs = int(mc_runs)
    mc_matrix = np.zeros((len(df), mc_runs))
    for i, comp in enumerate(df.index):
        mu = base_climate * sensitivity.get(comp, 1.0)
        sigma = max(0.03, mu * 0.12)
        mc_matrix[i, :] = np.clip(rng.normal(loc=mu, scale=sigma, size=mc_runs), 0.0, 1.0)
    df["C6: Rủi ro khí hậu"] = mc_matrix.mean(axis=1)
    mc_std = mc_matrix.std(axis=1)
else:
    df["C6: Rủi ro khí hậu"] = [base_climate * sensitivity[c] for c in df.index]
    mc_std = np.ones(len(df)) * 1e-4

# -------------------------
# Fuzzy approx for weights (simple TFN centroid)
# -------------------------
if use_fuzzy:
    fuzz_pct = 0.12  # 12% fuzziness default
    low = w * (1 - fuzz_pct)
    high = w * (1 + fuzz_pct)
    defuzz = (low + w + high) / 3.0
    defuzz = defuzz / defuzz.sum()
    weights_series = pd.Series(defuzz, index=criteria)
else:
    weights_series = pd.Series(w, index=criteria)

# -------------------------
# TOPSIS implementation
# -------------------------
def topsis(df_data, weights, cost_flags):
    # keep only columns in weights order
    M = df_data[list(weights.index)].astype(float).values  # shape (n_comp, n_crit)
    # normalization (vector normalization)
    denom = np.sqrt((M ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = M / denom
    V = R * weights.values  # weighted normalized matrix

    is_cost = np.array([cost_flags[c] == "cost" for c in weights.index])
    ideal_best = np.where(is_cost, np.min(V, axis=0), np.max(V, axis=0))
    ideal_worst = np.where(is_cost, np.max(V, axis=0), np.min(V, axis=0))

    d_plus = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)

    res = pd.DataFrame({
        "company": df_data.index,
        "score": score,
        "d_plus": d_plus,
        "d_minus": d_minus
    })
    res = res.sort_values(by="score", ascending=False).reset_index(drop=True)
    res["rank"] = res.index + 1
    return res

# -------------------------
# Run analysis & outputs
# -------------------------
if st.button("🚀 PHÂN TÍCH NGAY"):
    with st.spinner("Đang phân tích..."):
        result = topsis(df, weights_series, cost_flags)

        # friendly labels
        result["ICC"] = result["score"].apply(lambda x: "ICC A" if x >= 0.75 else "ICC B" if x >= 0.5 else "ICC C")
        result["Risk"] = result["score"].apply(lambda x: "THẤP" if x >= 0.75 else "TRUNG BÌNH" if x >= 0.5 else "CAO")

        # confidence from C6 MC CV (lower CV => higher confidence)
        mean_c6 = df["C6: Rủi ro khí hậu"].values
        cv = np.where(mean_c6 == 0, 0.0, mc_std / (mean_c6 + 1e-12))
        climate_conf = 1.0 / (1.0 + cv)
        # normalize to [0.3, 1.0]
        climate_conf = 0.3 + 0.7 * (climate_conf - climate_conf.min()) / (max(climate_conf.max() - climate_conf.min(), 1e-12))

        # criteria dispersion confidence (lower dispersion -> higher confidence)
        crit_cv = df[list(weights_series.index)].std(axis=1) / (df[list(weights_series.index)].mean(axis=1) + 1e-12)
        crit_conf = 1.0 / (1.0 + crit_cv)

        # final confidence geometric mean
        final_conf = np.sqrt(climate_conf * crit_conf.values)

        # attach confidence to result (map by company order)
        conf_map = {df.index[i]: float(final_conf[i]) for i in range(len(df))}
        result["confidence"] = result["company"].map(conf_map)

        result["score_pct"] = (result["score"] * 100).round(2)
        display = result[["rank", "company", "score", "score_pct", "ICC", "Risk", "confidence"]]

        st.success("✅ Hoàn tất phân tích")
        st.dataframe(display.set_index("rank"), use_container_width=True)

        # plotting ranking bar
        fig_bar = px.bar(result.sort_values("score"), x="score", y="company", orientation="h", color="score",
                         color_continuous_scale="Blues", title="Xếp hạng TOPSIS")
        st.plotly_chart(fig_bar, use_container_width=True)

        # plotting radar for top 3 companies (scaled values)
        top3 = result.head(3)["company"].tolist()
        radar_df = df.loc[top3, list(weights_series.index)].copy()
        radar_scaled = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-12)
        radar_scaled["company"] = radar_scaled.index
        radar_melt = radar_scaled.reset_index(drop=True).melt(id_vars=["company"], var_name="criterion", value_name="value")
        fig_radar = px.line_polar(radar_melt, r="value", theta="criterion", color="company", line_close=True, title="So sánh tiêu chí (Top 3)")
        st.plotly_chart(fig_radar, use_container_width=True)

        # -------------------------
        # Export to Excel
        # -------------------------
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            display.to_excel(writer, sheet_name="Result", index=False)
            df.to_excel(writer, sheet_name="Adjusted_Data")
            pd.DataFrame(weights_series, columns=["weight"]).to_excel(writer, sheet_name="Weights")
        output.seek(0)
        st.download_button("⬇️ Tải Excel (Kết quả)", data=output, file_name="riskcast_v34_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # -------------------------
        # Export to PDF with charts embedded (uses Plotly fig.to_image)
        # -------------------------
        try:
            # render figures to PNG bytes (kaleido must be installed)
            bar_png = fig_bar.to_image(format="png", width=1000, height=400, scale=1)
            radar_png = fig_radar.to_image(format="png", width=800, height=600, scale=1)
        except Exception as e:
            st.warning("Không thể xuất ảnh biểu đồ (thiếu kaleido?). Vui lòng cài 'kaleido'. Lỗi: " + str(e))
            bar_png = None
            radar_png = None

        # build pdf into bytes
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "RISKCAST v3.4 - Insurance Suggestion Report", ln=1, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, f"Route: {route}   Method: {method}   Month: {month}", ln=1)
        pdf.cell(0, 6, f"Cargo value: {cargo_value:,} USD   Priority: {priority}", ln=1)
        pdf.ln(4)

        # table: write top rows
        pdf.set_font("Arial", "B", 11)
        pdf.cell(15, 7, "Rank", 1)
        pdf.cell(55, 7, "Company", 1)
        pdf.cell(25, 7, "Score", 1)
        pdf.cell(25, 7, "ICC", 1)
        pdf.cell(30, 7, "Risk", 1)
        pdf.cell(25, 7, "Conf", 1)
        pdf.ln()
        pdf.set_font("Arial", "", 9)

        for _, row in result.iterrows():
            pdf.cell(15, 6, str(int(row["rank"])), 1)
            pdf.cell(55, 6, str(row["company"])[:30], 1)
            pdf.cell(25, 6, f"{row['score']:.4f}", 1)
            pdf.cell(25, 6, str(row["ICC"]), 1)
            pdf.cell(30, 6, str(row["Risk"]), 1)
            pdf.cell(25, 6, f"{row['confidence']:.2f}", 1)
            pdf.ln()

        pdf.ln(6)
        # embed bar chart PNG
        if bar_png:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(bar_png)
                tmp_bar = f.name
            try:
                pdf.cell(0, 6, "Ranking TOPSIS", ln=1)
                pdf.image(tmp_bar, x=10, w=190)
                pdf.ln(4)
            except Exception:
                pass

        # embed radar chart PNG
        if radar_png:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(radar_png)
                tmp_radar = f.name
            try:
                pdf.cell(0, 6, "Radar tiêu chí (Top 3)", ln=1)
                pdf.image(tmp_radar, x=20, w=170)
                pdf.ln(4)
            except Exception:
                pass

        # final footer
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 6, "Nguồn: Demo data. Mô hình RiskCast — Fuzzy AHP + Monte-Carlo + TOPSIS.", ln=1, align="C")

        # export pdf bytes to download
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button("⬇️ Tải PDF báo cáo", data=pdf_bytes, file_name="RISKCAST_v3.4_report.pdf", mime="application/pdf")

# -------------------------
# Footer
# -------------------------
st.markdown("<div class='footer'>RISKCAST v3.4 — Bùi Xuân Hoàng • AI Decision Support</div>", unsafe_allow_html=True)
