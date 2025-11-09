# app.py — RISKCAST v4.7 (fixed ptp & robust)
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# Optional ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

st.set_page_config(page_title="RISKCAST v4.7 — Green ESG", layout="wide", page_icon="🛡️")
st.markdown("""
<style>
  .stApp { background: linear-gradient(180deg,#0b3d0b 0%, #05320a 100%); color: #e9fbf0; font-family: 'Segoe UI', sans-serif; }
  h1 { color:#a3ff96; text-align:center; font-weight:800; }
  .card { background: rgba(255,255,255,0.03); padding:1rem; border-radius:10px; border:1px solid rgba(163,255,150,0.08); }
  .muted { color: #bfe8c6; font-size:0.95rem; }
  .small { font-size:0.85rem; color:#bfe8c6; }
  .result-box { background:#0f3d1f; padding:1rem; border-left:6px solid #3ef08a; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ RISKCAST v4.7 — Green ESG Insurance Advisor")
st.caption("ARIMA + MonteCarlo + VaR/CVaR + Fuzzy AHP + TOPSIS — Stable build")

# Sidebar inputs
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị lô hàng (USD)", min_value=1000, value=39000, step=1000)
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"])
    route = st.selectbox("Tuyến", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"])
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"])
    month = st.selectbox("Tháng (1-12)", list(range(1,13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

    st.markdown("---")
    st.header("Mô hình")
    use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN)", True)
    use_arima = st.checkbox("Dùng ARIMA để dự báo (nếu có)", True)
    use_var = st.checkbox("Tính VaR & CVaR", True)
    use_mc = st.checkbox("Chạy Monte Carlo cho C6", True)
    mc_runs = st.number_input("Số vòng Monte Carlo", min_value=500, max_value=10000, value=2000, step=500)

# Helpers
def auto_balance(weights, locked_flags):
    w = np.array(weights, dtype=float)
    locked = np.array(locked_flags, dtype=bool)
    locked_sum = w[locked].sum()
    free_idx = np.where(~locked)[0]
    if len(free_idx) == 0:
        if w.sum() == 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        return np.round(w, 6)
    remaining = max(0.0, 1.0 - locked_sum)
    free_vals = w[free_idx]
    if free_vals.sum() == 0:
        w[free_idx] = remaining / len(free_idx)
    else:
        w[free_idx] = free_vals / free_vals.sum() * remaining
    w = np.clip(w, 0.0, 1.0)
    diff = 1.0 - w.sum()
    if abs(diff) > 1e-8:
        idx = free_idx[0] if len(free_idx)>0 else 0
        w[idx] += diff
    return np.round(w, 6)

def defuzzify_centroid(low, mid, high):
    return (low + mid + high) / 3.0

def safe_plotly_to_png(fig):
    try:
        img_bytes = fig.to_image(format="png")
        return img_bytes
    except Exception:
        return None

@st.cache_data
def load_sample_data():
    months = list(range(1,13))
    base = {
        "VN - EU": [0.20,0.22,0.25,0.28,0.32,0.36,0.42,0.48,0.60,0.68,0.58,0.45],
        "VN - US": [0.30,0.33,0.36,0.40,0.45,0.50,0.56,0.62,0.75,0.72,0.60,0.52],
        "VN - Singapore": [0.15,0.16,0.18,0.20,0.22,0.26,0.30,0.32,0.35,0.34,0.28,0.25],
        "Domestic": [0.10,0.10,0.10,0.12,0.12,0.14,0.16,0.18,0.20,0.18,0.14,0.12],
        "VN - China": [0.18,0.19,0.21,0.24,0.26,0.30,0.34,0.36,0.40,0.38,0.32,0.28],
    }
    hist = pd.DataFrame({"month": months})
    for k,v in base.items():
        hist[k] = v
    rng = np.random.default_rng(123)
    losses = np.clip(rng.normal(loc=0.08, scale=0.02, size=2000), 0, 0.5)
    claims = pd.DataFrame({"loss_rate": losses})
    return hist, claims

historical, claims = load_sample_data()

criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

if "weights" not in st.session_state:
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15], dtype=float)
if "locked" not in st.session_state:
    st.session_state["locked"] = [False]*6

st.subheader("⚖️ Phân bổ trọng số tiêu chí (Lock & Auto-balance)")
cols = st.columns(6)
new_w = st.session_state["weights"].copy()
for i,c in enumerate(criteria):
    with cols[i]:
        st.markdown(f"**{c}**")
        st.checkbox("🔒 Lock", value=st.session_state["locked"][i], key=f"lock_{i}", on_change=None)
        val = st.number_input("Tỉ lệ", min_value=0.0, max_value=1.0, value=float(new_w[i]), step=0.01, key=f"w_in_{i}")
        new_w[i] = val
for i in range(6):
    st.session_state["locked"][i] = st.session_state.get(f"lock_{i}", False)

if st.button("🔄 Reset trọng số mặc định"):
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15], dtype=float)
    st.session_state["locked"] = [False]*6
else:
    st.session_state["weights"] = auto_balance(new_w, st.session_state["locked"])

weights_series = pd.Series(st.session_state["weights"], index=criteria)
fig_weights = px.pie(values=weights_series.values, names=weights_series.index, title="Phân bổ trọng số (Realtime)")
st.plotly_chart(fig_weights, use_container_width=True)

# sample companies
df = pd.DataFrame({
    "Company": ["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí": [0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý": [6,5,8,7,4],
    "C3: Tỷ lệ tổn thất": [0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC": [9,8,6,9,7],
    "C5: Chăm sóc KH": [9,8,5,7,6],
}).set_index("Company")

sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}
route_key = route
base_climate = float(historical.loc[historical['month']==month, route_key].iloc[0]) if month in historical['month'].values else 0.40

df_adj = df.copy().astype(float)
mc_mean = np.array([base_climate * sensitivity.get(c,1.0) for c in df_adj.index], dtype=float)
mc_std = np.zeros(len(df_adj), dtype=float)
if use_mc:
    rng = np.random.default_rng(2025)
    for i in range(len(df_adj)):
        mu = mc_mean[i]
        sigma = max(0.03, mu*0.12)
        sims = rng.normal(loc=mu, scale=sigma, size=int(mc_runs))
        sims = np.clip(sims, 0.0, 1.0)
        mc_mean[i] = sims.mean()
        mc_std[i] = sims.std()
df_adj["C6: Rủi ro khí hậu"] = mc_mean

if cargo_value > 50000:
    df_adj["C1: Tỷ lệ phí"] *= 1.1

def topsis(df_input, weight_series, cost_flags):
    M = df_input[list(weight_series.index)].values.astype(float)
    denom = np.sqrt((M**2).sum(axis=0))
    denom[denom==0] = 1.0
    R = M / denom
    w = np.array(weight_series.values, dtype=float)
    V = R * w
    is_cost = np.array([cost_flags[c]=="cost" for c in weight_series.index])
    ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
    ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))
    d_plus = np.sqrt(((V - ideal_best)**2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst)**2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)
    return score

cost_flags = {c: "cost" if c in ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

def compute_var_cvar(loss_rates, cargo_value, alpha=0.95):
    losses = np.array(loss_rates, dtype=float) * float(cargo_value)
    var = np.percentile(losses, alpha*100)
    tail = losses[losses >= var]
    cvar = tail.mean() if len(tail)>0 else var
    return float(var), float(cvar)

def forecast_route(route_key, months_ahead=3):
    series = historical[route_key].values if route_key in historical.columns else historical.iloc[:,1].values
    if use_arima and ARIMA_AVAILABLE:
        try:
            model = ARIMA(series, order=(1,1,1)).fit()
            fc = model.forecast(months_ahead)
            return np.asarray(series), np.asarray(fc)
        except Exception:
            pass
    last = np.array(series)
    avg = np.mean(last[-6:])
    trend = (last[-1] - last[-6]) / 6.0
    fc = np.array([max(0, last[-1] + (i+1)*trend) for i in range(months_ahead)])
    return last, fc

if st.button("🚀 PHÂN TÍCH & GỢI Ý"):
    with st.spinner("Đang chạy mô phỏng và tối ưu..."):
        weights = weights_series.copy()
        if use_fuzzy:
            f = float(st.sidebar.slider("Bất định TFN (%)", 0, 50, 15))
            low = np.maximum(weights*(1 - f/100.0), 1e-6)
            high = np.minimum(weights*(1 + f/100.0), 0.9999)
            defuz = defuzzify_centroid(low, weights, high)
            weights = pd.Series(defuz/defuz.sum(), index=weights.index)

        scores = topsis(df_adj, weights, cost_flags)
        results = pd.DataFrame({
            "company": df_adj.index,
            "score": scores,
            "C6_mean": mc_mean,
            "C6_std": mc_std
        }).sort_values("score", ascending=False).reset_index(drop=True)
        results["rank"] = results.index + 1
        results["recommend_icc"] = results["score"].apply(lambda x: "ICC A" if x>=0.75 else ("ICC B" if x>=0.5 else "ICC C"))

        # Confidence calc (robust)
        cv_c6 = np.where(results["C6_mean"].values==0, 0.0, results["C6_std"].values / (results["C6_mean"].values + 1e-9))
        conf_c6 = 1.0 / (1.0 + cv_c6)
        # use numpy.ptp for safety
        if np.ptp(conf_c6) > 0:
            conf_c6_scaled = 0.3 + 0.7*(conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + 1e-12)
        else:
            conf_c6_scaled = np.full_like(conf_c6, 0.65)

        crit_cv = df_adj.std(axis=1).values / (df_adj.mean(axis=1).values + 1e-9)
        conf_crit = 1.0 / (1.0 + crit_cv)
        if np.ptp(conf_crit) > 0:
            conf_crit_scaled = 0.3 + 0.7*(conf_crit - conf_crit.min()) / (np.ptp(conf_crit) + 1e-12)
        else:
            conf_crit_scaled = np.full_like(conf_crit, 0.65)

        conf_final = np.sqrt(conf_c6_scaled * conf_crit_scaled)
        order_map = {comp: conf_final[i] for i, comp in enumerate(df_adj.index)}
        results["confidence"] = results["company"].map(order_map).round(3)

        var95, cvar95 = (None, None)
        if use_var:
            var95, cvar95 = compute_var_cvar(results["C6_mean"].values, cargo_value, alpha=0.95)

        hist_series, fc = forecast_route(route)
        months_hist = list(range(1, len(hist_series)+1))
        months_fc = list(range(len(hist_series)+1, len(hist_series)+1+len(fc)))
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=months_hist, y=hist_series, mode='lines+markers', name='Lịch sử'))
        fig_forecast.add_trace(go.Scatter(x=months_fc, y=fc, mode='lines+markers', name='Dự báo', line=dict(color='lime')))
        fig_forecast.update_layout(title=f"Dự báo rủi ro: {route}", xaxis_title="Tháng index", yaxis_title="Rủi ro (0-1)")

        fig_topsis = px.bar(results.sort_values("score"), x="score", y="company", orientation='h', title="TOPSIS score (higher better)",
                            labels={"score":"Score","company":"Công ty"})

        st.success("Hoàn tất phân tích")
        left, right = st.columns((2,1))
        with left:
            st.subheader("Kết quả xếp hạng")
            st.table(results[["rank","company","score","confidence","recommend_icc"]].set_index("rank").round(3))
            st.markdown("<div class='result-box'><strong>ĐỀ XUẤT:</strong> {} — Score: {:.3f} — Confidence: {:.2f}</div>".format(
                results.iloc[0]["company"], results.iloc[0]["score"], results.iloc[0]["confidence"]
            ), unsafe_allow_html=True)
        with right:
            st.subheader("Tổng quan")
            st.metric("VaR 95%", f"${var95:,.0f}" if var95 is not None else "N/A")
            st.metric("CVaR 95%", f"${cvar95:,.0f}" if cvar95 is not None else "N/A")
            st.plotly_chart(fig_weights, use_container_width=True)

        st.plotly_chart(fig_topsis, use_container_width=True)
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Export Excel
        excel_out = io.BytesIO()
        with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Result", index=False)
            df_adj.to_excel(writer, sheet_name="Adjusted_Data")
            pd.DataFrame(weights).to_excel(writer, "Weights", index=True)
        excel_out.seek(0)
        st.download_button("⬇️ Xuất Excel (Kết quả)", excel_out, file_name="riskcast_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Export PDF (3 pages)
        pdf = FPDF(unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=12)
        try:
            pdf.add_font("DejaVu", "", fname="", uni=True)
            pdf.set_font("DejaVu", size=12)
        except Exception:
            pdf.set_font("Arial", size=12)

        pdf.add_page()
        pdf.set_font_size(16)
        pdf.cell(0, 8, "RISKCAST v4.7 — Executive Summary", ln=1)
        pdf.ln(2)
        pdf.set_font_size(10)
        pdf.cell(0, 6, f"Route: {route}    Month: {month}    Method: {method}", ln=1)
        pdf.cell(0, 6, f"Cargo value: ${cargo_value:,}    Priority: {priority}", ln=1)
        pdf.ln(4)
        pdf.set_font_size(11)
        pdf.multi_cell(0, 6, f"Recommended insurer: {results.iloc[0]['company']} ({results.iloc[0]['recommend_icc']})\nTOPSIS Score: {results.iloc[0]['score']:.4f}\nConfidence: {results.iloc[0]['confidence']:.2f}\nVaR95: ${var95:,.0f} | CVaR95: ${cvar95:,.0f}" if var95 is not None else "", align="L")
        pdf.ln(6)
        pdf.set_font_size(10)
        pdf.cell(40,6,"Rank",1); pdf.cell(50,6,"Company",1); pdf.cell(40,6,"Score",1); pdf.cell(35,6,"Confidence",1); pdf.ln()
        for idx, row in results.head(5).iterrows():
            pdf.cell(40,6,str(int(row["rank"])),1); pdf.cell(50,6,str(row["company"])[:20],1)
            pdf.cell(40,6,f"{row['score']:.4f}",1); pdf.cell(35,6,f"{row['confidence']:.2f}",1); pdf.ln()

        pdf.add_page()
        pdf.set_font_size(14)
        pdf.cell(0,8,"TOPSIS Scores", ln=1)
        img_bytes = safe_plotly_to_png(fig_topsis)
        if img_bytes:
            try:
                from PIL import Image
                im = Image.open(io.BytesIO(img_bytes))
                tmp = "tmp_topsis.png"
                im.save(tmp)
                pdf.image(tmp, x=15, w=180)
            except Exception:
                pdf.set_font_size(10)
                pdf.cell(0,6,"(Không thể xuất biểu đồ TOPSIS — missing PIL/kaleido)", ln=1)
        else:
            pdf.set_font_size(10)
            pdf.cell(0,6,"(Biểu đồ TOPSIS không thể xuất sang ảnh)", ln=1)
            pdf.ln(4)
            for idx,row in results.iterrows():
                pdf.cell(0,5,f"{int(row['rank'])}. {row['company']} — Score: {row['score']:.4f} — Conf: {row['confidence']:.2f}", ln=1)

        pdf.add_page()
        pdf.set_font_size(14)
        pdf.cell(0,8,"Forecast (ARIMA or fallback) & VaR", ln=1)
        img_bytes2 = safe_plotly_to_png(fig_forecast)
        if img_bytes2:
            try:
                from PIL import Image
                im2 = Image.open(io.BytesIO(img_bytes2))
                tmp2 = "tmp_forecast.png"
                im2.save(tmp2)
                pdf.image(tmp2, x=10, w=190)
            except Exception:
                pdf.set_font_size(10)
                pdf.cell(0,6,"(Không thể xuất biểu đồ Forecast — missing PIL/kaleido)", ln=1)
        else:
            pdf.set_font_size(10)
            pdf.cell(0,6,"(Biểu đồ Forecast không thể xuất)", ln=1)
        pdf.ln(6)
        if var95 is not None:
            pdf.set_font_size(11)
            pdf.cell(0,6,f"VaR 95%: ${var95:,.0f}", ln=1)
            pdf.cell(0,6,f"CVaR 95%: ${cvar95:,.0f}", ln=1)

        try:
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
        except Exception:
            pdf_bytes = pdf.output(dest="S").encode("utf-8", errors="ignore")
        st.download_button("⬇️ Xuất PDF báo cáo (3 trang)", data=pdf_bytes, file_name="RISKCAST_report.pdf", mime="application/pdf")

st.markdown("<br><div class='muted small'>RISKCAST v4.7 — Green ESG theme. Author: Bùi Xuân Hoàng.</div>", unsafe_allow_html=True)
