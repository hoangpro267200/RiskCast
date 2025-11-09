# app.py — RISKCAST NCKH FINAL (PDF nhiều) — Theme: Light Green
import io, os, math, warnings, tempfile
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime

# Optional ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

# ---------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------
def safe_ptp(a):
    """Return peak-to-peak but safe for scalars/zero-length arrays."""
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return 0.0
    return float(a.max() - a.min())

def save_fig_plotly(fig, path):
    """Try to save plotly to PNG via kaleido; fallback to static matplotlib snapshot."""
    try:
        fig.write_image(path, format="png", engine="kaleido")
        return True
    except Exception:
        # fallback: render png from static matplotlib by converting data roughly
        try:
            # attempt convert using static image produced from fig.to_image if possible
            img_bytes = fig.to_image(format="png")
            with open(path, "wb") as f:
                f.write(img_bytes)
            return True
        except Exception:
            return False

def save_fig_matplotlib(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True

def make_pdf_report(pdf_path, results_df, df_adj, charts_paths, params):
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    # cover
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "RISKCAST - NCKH REPORT", ln=True, align="C")
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Tác giả: {params.get('author','Hoang')}", ln=True)
    pdf.cell(0, 8, f"Ngày: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(6)
    pdf.multi_cell(0, 6, "Tóm tắt: Mô hình Fuzzy AHP + TOPSIS + Monte Carlo (C6) + VaR/CVaR + Confidence Score. Bản nộp cho NCKH.")
    pdf.ln(6)

    # Parameters
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Thông số chạy:", ln=True)
    pdf.set_font("Arial", "", 11)
    for k, v in params.items():
        pdf.cell(0, 6, f"- {k}: {v}", ln=True)
    pdf.ln(6)

    # Results table (first page)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Kết quả TOPSIS (tóm tắt)", ln=True)
    pdf.ln(2)
    # table header
    pdf.set_font("Arial", "B", 10)
    colw = [18, 55, 30, 30, 30]
    headers = ["Rank", "Company", "Score", "ICC", "Confidence"]
    for w, h in zip(colw, headers):
        pdf.cell(w, 7, h, border=1)
    pdf.ln()
    pdf.set_font("Arial", "", 10)
    for _, r in results_df.iterrows():
        pdf.cell(colw[0], 6, str(int(r['rank'])), border=1)
        pdf.cell(colw[1], 6, str(r['company'])[:25], border=1)
        pdf.cell(colw[2], 6, f"{r['score']:.4f}", border=1)
        pdf.cell(colw[3], 6, r['ICC'], border=1)
        pdf.cell(colw[4], 6, f"{r['confidence']:.2f}", border=1)
        pdf.ln()

    # Charts pages
    for p in charts_paths:
        if not os.path.exists(p):
            continue
        pdf.add_page()
        pdf.image(p, x=15, y=20, w=180)

    # Additional sheet: adjusted data (as simple table)
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Adjusted Data (sample)", ln=True)
    pdf.ln(2)
    pdf.set_font("Arial", "", 9)
    table = df_adj.reset_index().head(20)
    # header
    for col in table.columns:
        pdf.cell(32, 6, str(col)[:12], 1)
    pdf.ln()
    for _, row in table.iterrows():
        for col in table.columns:
            txt = str(row[col])[:12]
            pdf.cell(32, 6, txt, 1)
        pdf.ln()

    pdf.output(pdf_path)

# ---------------------------------------------------------
# UI / Styling (green theme)
# ---------------------------------------------------------
st.set_page_config(page_title="RISKCAST NCKH - Green", layout="wide", page_icon="🌱")
st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg,#083d12 0%, #0d2b15 100%); color: #eaf8ea; }
    .block-container{padding:1rem 2rem;}
    h1 { color: #e8fff0; text-align:center; font-weight:800; }
    .card { background: rgba(255,255,255,0.03); padding:12px; border-radius:12px; }
    .btn { background: linear-gradient(90deg,#7efc9d, #0ad17a); color: #072209; font-weight:bold; }
    </style>
""", unsafe_allow_html=True)

st.title("RISKCAST — NCKH Edition (Green Theme)")
st.write("Mục tiêu: Ổn định thuật toán + export PDF (nhiều trang) + giao diện chuyên nghiệp để nộp NCKH.")

# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị (USD)", value=35000, step=1000)
    good_type = st.selectbox("Loại hàng", ["Điện tử","Đông lạnh","Hàng khô","Hàng nguy hiểm","Khác"])
    route = st.selectbox("Tuyến", ["VN - EU","VN - US","VN - Singapore","Domestic"])
    method = st.selectbox("Phương thức", ["Sea","Air","Truck"])
    month = st.selectbox("Tháng", list(range(1,13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa","Cân bằng","Tối ưu chi phí"])

    st.markdown("---")
    st.header("Mô hình & Export")
    use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN)", value=True)
    use_arima = st.checkbox("Dùng ARIMA nếu available", value=False)
    use_var = st.checkbox("Tính VaR/CVaR (95%)", value=True)
    use_mc = st.checkbox("Chạy Monte Carlo cho C6", value=True)
    mc_runs = st.number_input("Số vòng Monte-Carlo", min_value=200, max_value=20000, value=2000, step=100)
    st.markdown("**PDF:** Nhiều trang (chi tiết)")
    st.markdown("---")

# -------------------------
# Criteria sliders + lock + numeric + auto-balance
# -------------------------
criteria = ["C1: Tỷ lệ phí","C2: Thời gian xử lý","C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC","C5: Chăm sóc KH","C6: Rủi ro khí hậu"]
n = len(criteria)
st.subheader("Phân bổ trọng số (Auto-balance & Lock)")
cols = st.columns([1]*n)
# maintain state for locks and numeric inputs
if "locks" not in st.session_state:
    st.session_state.locks = {c: False for c in criteria}
if "numbers" not in st.session_state:
    st.session_state.numbers = {c: 1.0/n for c in criteria}

# Reset button
if st.button("🔄 Reset trọng số về mặc định"):
    for c in criteria:
        st.session_state.locks[c] = False
        st.session_state.numbers[c] = 1.0/n

# show each control
for i, c in enumerate(criteria):
    with cols[i]:
        st.markdown(f"**{c}**")
        lock = st.checkbox("🔒 Lock", value=st.session_state.locks[c], key=f"lock_{i}")
        st.session_state.locks[c] = lock
        # numeric input
        val = st.number_input(f"Nhập tỉ lệ {c}", min_value=0.0, max_value=1.0, value=float(st.session_state.numbers[c]),
                              step=0.01, key=f"num_{i}")
        st.session_state.numbers[c] = val

# auto-normalize while keeping locked values fixed
locked = {k:v for k,v in st.session_state.locks.items() if v}
vals = st.session_state.numbers.copy()
total_locked = sum(vals[c] for c in locked)
free_keys = [k for k in criteria if k not in locked]
sum_free = sum(vals[k] for k in free_keys)
# if all locked or sum invalid, normalize all
if len(free_keys)==0 or (sum_free==0 and total_locked==0):
    # distribute equally
    for k in criteria:
        st.session_state.numbers[k] = 1.0/n
else:
    # scale free to sum to (1 - total_locked)
    target_free = max(0.0, 1.0 - total_locked)
    if sum_free <= 0:
        # distribute evenly
        for k in free_keys:
            st.session_state.numbers[k] = target_free / len(free_keys)
    else:
        scale = target_free / sum_free
        for k in free_keys:
            st.session_state.numbers[k] = vals[k] * scale

weights_series = pd.Series([st.session_state.numbers[c] for c in criteria], index=criteria)

st.markdown("**Realtime distribution**")
fig_pie = px.pie(names=weights_series.index, values=weights_series.values, color_discrete_sequence=px.colors.sequential.Plasma_r)
st.plotly_chart(fig_pie, use_container_width=True)

# Fuzzy handling
if use_fuzzy:
    fuzz_pct = st.slider("Biên độ bất định TFN (%)", 0, 50, 15)
    low = np.maximum(weights_series * (1 - fuzz_pct/100.0), 1e-6)
    high = np.minimum(weights_series * (1 + fuzz_pct/100.0), 0.9999)
    defuzz = (low + weights_series + high) / 3.0
    weights_series = defuzz / defuzz.sum()

# -------------------------
# Data (sample) & Monte Carlo for C6
# -------------------------
sample = {
    "Company": ["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí": [0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý": [6,5,8,7,4],
    "C3: Tỷ lệ tổn thất": [0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC": [9,8,6,9,7],
    "C5: Chăm sóc KH": [9,8,5,7,6]
}
df = pd.DataFrame(sample).set_index("Company").astype(float)
sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}
# base climate risk by route/month — simplified
base_map = {("VN - EU",9):0.65, ("VN - US",9):0.75, ("VN - Singapore",9):0.30, ("Domestic",9):0.20}
base_climate = base_map.get((route, month), 0.40)

df_adj = df.copy()
mc_mean = np.array([base_climate * sensitivity.get(c,1.0) for c in df_adj.index], dtype=float)
mc_std = np.zeros_like(mc_mean)

if use_mc:
    rng = np.random.default_rng(42)
    for i, comp in enumerate(df_adj.index):
        mu = base_climate * sensitivity.get(comp,1.0)
        sigma = max(0.03, mu * 0.12)
        sims = rng.normal(loc=mu, scale=sigma, size=mc_runs)
        sims = np.clip(sims, 0.0, 1.0)
        mc_mean[i] = sims.mean()
        mc_std[i] = sims.std()
df_adj["C6: Rủi ro khí hậu"] = mc_mean

# -------------------------
# TOPSIS
# -------------------------
def topsis(df_data, weights, cost_flags):
    M = df_data[list(weights.index)].values.astype(float)
    denom = np.sqrt((M**2).sum(axis=0))
    denom[denom==0] = 1.0
    R = M / denom
    V = R * weights.values
    is_cost = np.array([cost_flags[c]=="cost" for c in weights.index])
    ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
    ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))
    d_plus = np.sqrt(((V - ideal_best)**2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst)**2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)
    return score

cost_flags = {c: "cost" if c in ["C1: Tỷ lệ phí","C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

# -------------------------
# VaR & CVaR
# -------------------------
if use_var:
    st.subheader("VaR & CVaR (95%) — estimate từ C6 x CargoValue")
    losses = df_adj["C6: Rủi ro khí hậu"].values * cargo_value
    var95 = np.percentile(losses, 95)
    cvar95 = losses[losses >= var95].mean() if (losses >= var95).sum() > 0 else var95
    c1, c2 = st.columns(2)
    c1.metric("VaR 95%", f"${var95:,.0f}")
    c2.metric("CVaR 95%", f"${cvar95:,.0f}")

# -------------------------
# Run analysis button
# -------------------------
if st.button("🚀 PHÂN TÍCH & GỢI Ý (Export PDF nhiều trang)"):
    with st.spinner("Đang chạy phân tích..."):
        try:
            scores = topsis(df_adj, weights_series, cost_flags)
            res = pd.DataFrame({
                "company": df_adj.index,
                "score": scores,
            }).sort_values("score", ascending=False).reset_index(drop=True)
            res["rank"] = res.index + 1
            res["ICC"] = res["score"].apply(lambda x: "ICC A" if x>=0.75 else ("ICC B" if x>=0.5 else "ICC C"))

            # Confidence:
            cv_c6 = np.where(mc_mean==0, 0.0, mc_std / mc_mean)
            conf_c6 = 1.0 / (1.0 + cv_c6)
            # scale robustly
            ptp_c6 = safe_ptp(conf_c6)
            if ptp_c6 > 0:
                conf_c6_scaled = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / ptp_c6
            else:
                conf_c6_scaled = np.full_like(conf_c6, 0.65)

            crit_cv = df_adj.std(axis=1) / (df_adj.mean(axis=1) + 1e-9)
            conf_crit = 1.0 / (1.0 + crit_cv)
            ptp_crit = safe_ptp(conf_crit)
            if ptp_crit > 0:
                conf_crit_scaled = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / ptp_crit
            else:
                conf_crit_scaled = np.full_like(conf_crit, 0.65)

            final_conf = np.sqrt(conf_c6_scaled * conf_crit_scaled)
            # map by company order
            conf_map = {comp: float(final_conf[i]) for i, comp in enumerate(df_adj.index)}
            res["confidence"] = res["company"].map(conf_map)

            st.success("Phân tích hoàn tất!")
            st.dataframe(res.set_index("rank"))

            # Plots: bar and radar (plotly)
            fig_bar = px.bar(res.sort_values("score"), x="score", y="company", orientation="h",
                             color="score", color_continuous_scale=px.colors.sequential.Greens)
            st.plotly_chart(fig_bar, use_container_width=True)

            # radar for top3
            top3 = res.head(3)["company"].tolist()
            radar_df = df_adj.loc[top3, list(weights_series.index)]
            radar_scaled = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
            radar_melt = radar_scaled.reset_index().melt(id_vars=["Company"] if "Company" in radar_scaled.columns else None,
                                                         var_name="criterion", value_name="value")
            # simpler radar via plotly
            fig_radar = go.Figure()
            for comp in top3:
                vals = radar_scaled.loc[comp].values.tolist()
                vals.append(vals[0])
                fig_radar.add_trace(go.Scatterpolar(r=vals, theta=list(radar_scaled.columns)+[radar_scaled.columns[0]],
                                                    fill='toself', name=comp))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True,
                                    title="So sánh tiêu chí (Top 3)")
            st.plotly_chart(fig_radar, use_container_width=True)

            # Export: Excel
            out_xl = io.BytesIO()
            with pd.ExcelWriter(out_xl, engine="openpyxl") as writer:
                res.to_excel(writer, sheet_name="Result", index=False)
                df_adj.to_excel(writer, sheet_name="Adjusted_Data")
                pd.DataFrame(weights_series, columns=["weight"]).to_excel(writer, sheet_name="Weights")
            out_xl.seek(0)
            st.download_button("📥 Xuất Excel (Result)", data=out_xl.getvalue(), file_name="riskcast_result.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # Prepare charts as PNGs for PDF (use temp files)
            tmp_dir = tempfile.mkdtemp(prefix="riskcast_")
            charts = []
            # Bar
            bar_path = os.path.join(tmp_dir, "bar.png")
            try:
                save_fig_plotly(fig_bar, bar_path)
            except Exception:
                # fallback matplotlib
                fig, ax = plt.subplots(figsize=(8,4))
                ax.barh(res["company"], res["score"], color="#7efc9d")
                ax.set_xlabel("Score")
                plt.tight_layout()
                save_fig_matplotlib(fig, bar_path)
            charts.append(bar_path)

            # Radar
            radar_path = os.path.join(tmp_dir, "radar.png")
            try:
                save_fig_plotly(fig_radar, radar_path)
            except Exception:
                fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(6,6))
                # basic fallback
                angles = np.linspace(0, 2*np.pi, len(radar_scaled.columns), endpoint=False).tolist()
                for comp in top3:
                    vals = radar_scaled.loc[comp].values
                    vals = np.append(vals, vals[0])
                    ax.plot(np.append(angles, angles[0]), vals, label=comp)
                ax.legend(loc='upper right')
                save_fig_matplotlib(fig, radar_path)
            charts.append(radar_path)

            # final PDF path (in memory)
            pdf_buf = io.BytesIO()
            pdf_path = os.path.join(tmp_dir, "riskcast_report.pdf")
            params = {
                "cargo_value": cargo_value, "route": route, "month": month,
                "priority": priority, "fuzzy_pct": (fuzz_pct if use_fuzzy else 0),
                "mc_runs": mc_runs
            }
            make_pdf_report(pdf_path, res, df_adj, charts, params)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button("📄 Xuất PDF (Nhiều trang)", data=pdf_bytes, file_name="riskcast_report_full.pdf", mime="application/pdf")

        except Exception as e:
            st.error("Đã có lỗi khi chạy phân tích — mình log bên dưới (để dev fix nhanh).")
            st.exception(e)
