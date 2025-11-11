# =============================================================
# RISKCAST v4.9 — ESG Logistics Dashboard (PREMIUM BLUE UI)
# Author: Bùi Xuân Hoàng — UI + Chart Clarity Enhanced by Kai
# Fixed: Month 15 bug, Blurry text, PDF Unicode Error
# =============================================================
import io
import uuid
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import os
import tempfile

warnings.filterwarnings("ignore")

# ---------------- Optional libs ----------------
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# ---------------- PREMIUM UI (BLUE INSURANCE STYLE) ----------
st.set_page_config(page_title="RISKCAST v4.9 — ESG Insurance", layout="wide")

st.markdown("""
<style>
    * { opacity: 1 !important; }
    .stApp {
        background: linear-gradient(135deg, #d9e9ff 0%, #f4fbff 100%) !important;
        font-family: 'Segoe UI', sans-serif;
        color: #003060;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 2px solid #e6edf7;
        padding: 1rem;
    }
    section[data-testid="stSidebar"] label {
        color: #003060 !important;
        font-weight: 600 !important;
    }
    .block-container {
        background: rgba(255,255,255,1) !important;
        backdrop-filter: blur(6px);
        padding: 2rem !important;
        border-radius: 18px;
        box-shadow: 0px 6px 22px rgba(0,0,0,0.08);
        opacity: 1 !important;
    }
    h1, h2, h3 { color: #2A6FDB !important; font-weight: 700; }
    .stButton > button {
        background: #2A6FDB !important; color: #fff !important;
        border-radius: 8px; padding: 10px 24px; border: none;
        font-weight: 600; transition: all .25s ease-in-out;
    }
    .stButton > button:hover {
        background: #1e57b2 !important; transform: translateY(-2px);
    }
    .result-box {
        background: linear-gradient(90deg, #2A6FDB, #1e57b2);
        color: white !important; padding: 16px 24px; border-radius: 12px;
        font-weight: 700; text-align: center; font-size: 20px;
        margin: 15px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    table, .stDataFrame { font-size: 16px !important; font-weight: 500 !important; color: #003060 !important; }
    .stMetric { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- ENHANCE FIG + PNG EXPORT -----------------
def enhance_fig(fig, title=None, font_size=16, title_size=22):
    fig.update_layout(
        template="simple_white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=font_size, color="#003060"),
        title=dict(text=title or fig.layout.title.text, font=dict(size=title_size, color="#2A6FDB"), x=0.5, xanchor="center"),
        legend=dict(font=dict(size=font_size+2), bgcolor="rgba(255,255,255,0.95)", bordercolor="#e6edf7", borderwidth=1),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=70, r=70, t=100, b=70),
        hoverlabel=dict(font_size=font_size+2)
    )
    fig.update_xaxes(title_font=dict(size=font_size+4), tickfont=dict(size=font_size+2))
    fig.update_yaxes(title_font=dict(size=font_size+4), tickfont=dict(size=font_size+2))
    return fig

def fig_to_png_bytes(fig, width=1400, height=600, scale=3):
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except:
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.write_image(tmp.name, width=width, height=height, scale=scale)
                with open(tmp.name, "rb") as f:
                    data = f.read()
                os.unlink(tmp.name)
                return data
        except Exception as e:
            st.error(f"Lỗi xuất ảnh: {e}")
            return None

# ---------------- TITLE -----------------
st.title("RISKCAST v4.9 — ESG Logistics Dashboard")
st.caption("Fuzzy AHP + TOPSIS + Monte Carlo (C6) + VaR/CVaR + ARIMA (optional)")

# ================= Sidebar inputs =================
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
    use_arima = st.checkbox("Dùng ARIMA (nếu có)", True)
    use_mc = st.checkbox("Chạy Monte Carlo cho C6", True)
    use_var = st.checkbox("Tính VaR và CVaR", True)
    mc_runs = st.number_input("Số vòng Monte Carlo", 500, 10000, value=2000, step=500)

# ================= Helper functions =================
def auto_balance(weights, locked):
    w = np.array(weights, dtype=float)
    locked_flags = np.array(locked, dtype=bool)
    total_locked = w[locked_flags].sum()
    free_idx = np.where(~locked_flags)[0]
    if len(free_idx) == 0:
        return (w / w.sum()) if w.sum() != 0 else np.ones_like(w)/len(w)
    remaining = max(0.0, 1.0 - total_locked)
    free_sum = w[free_idx].sum()
    if free_sum == 0:
        w[free_idx] = remaining / len(free_idx)
    else:
        w[free_idx] = w[free_idx] / free_sum * remaining
    w = np.clip(w, 0.0, 1.0)
    diff = 1.0 - w.sum()
    if abs(diff) > 1e-8 and len(free_idx) > 0:
        w[free_idx[0]] += diff
    return np.round(w, 6)

def defuzzify_centroid(low, mid, high):
    return (low + mid + high) / 3.0

# ================= Sample data =================
@st.cache_data
def load_data():
    months = list(range(1,13))  # CHỈ 12 THÁNG
    base = {
        "VN - EU": [0.2,0.22,0.25,0.28,0.32,0.36,0.42,0.48,0.60,0.68,0.58,0.45],
        "VN - US": [0.3,0.33,0.36,0.4,0.45,0.5,0.56,0.62,0.75,0.72,0.6,0.52],
        "VN - Singapore": [0.15,0.16,0.18,0.2,0.22,0.26,0.30,0.32,0.35,0.34,0.28,0.25],
        "VN - China": [0.18,0.19,0.21,0.24,0.26,0.30,0.34,0.36,0.40,0.38,0.32,0.28],
        "Domestic": [0.1]*12
    }
    hist = pd.DataFrame({"month": months})
    for k, v in base.items():
        hist[k] = v
    rng = np.random.default_rng(123)
    losses = np.clip(rng.normal(loc=0.08, scale=0.02, size=2000), 0, 0.5)
    claims = pd.DataFrame({"loss_rate": losses})
    return hist, claims

historical, claims = load_data()

# ================= Criteria & Weights =================
criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

if "weights" not in st.session_state:
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15], dtype=float)
if "locked" not in st.session_state:
    st.session_state["locked"] = [False]*len(criteria)

st.subheader("Phân bổ trọng số (Realtime)")
cols = st.columns(len(criteria))
new_w = st.session_state["weights"].copy()
for i, c in enumerate(criteria):
    with cols[i]:
        st.markdown(f"**{c.split(':')[1]}**")
        st.checkbox("Lock", value=st.session_state["locked"][i], key=f"lock_{i}")
        val = st.number_input("", min_value=0.0, max_value=1.0, value=float(new_w[i]), step=0.01, key=f"w_{i}")
        new_w[i] = val

for i in range(len(criteria)):
    st.session_state["locked"][i] = st.session_state.get(f"lock_{i}", False)

if st.button("Reset trọng số mặc định"):
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15], dtype=float)
    st.session_state["locked"] = [False]*len(criteria)
else:
    st.session_state["weights"] = auto_balance(new_w, st.session_state["locked"])

weights = pd.Series(st.session_state["weights"], index=criteria)

# Pie chart
fig_weights = px.pie(values=weights.values, names=[c.split(':')[1] for c in weights.index],
                     title="Phân bổ trọng số", color_discrete_sequence=px.colors.sequential.Blues)
fig_weights = enhance_fig(fig_weights, font_size=14, title_size=18)
png_weights = fig_to_png_bytes(fig_weights, width=800, height=500)
if png_weights:
    st.image(png_weights, use_container_width=True)
else:
    st.plotly_chart(fig_weights, use_container_width=True)

# ================= Insurance Data =================
df = pd.DataFrame({
    "Company": ["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí": [0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý": [6,5,8,7,4],
    "C3: Tỷ lệ tổn thất": [0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC": [9,8,6,9,7],
    "C5: Chăm sóc KH": [9,8,5,7,6],
}).set_index("Company")

sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}
base_climate = float(historical.loc[historical["month"]==month, route].iloc[0])
df_adj = df.copy().astype(float)

# Monte Carlo
@st.cache_data
def monte_carlo_climate(base, sens_map, runs):
    rng = np.random.default_rng(2025)
    names = list(sens_map.keys())
    mu = np.array([base * sens_map[n] for n in names])
    sigma = np.maximum(0.03, mu * 0.12)
    sims = rng.normal(loc=mu, scale=sigma, size=(int(runs), len(names)))
    sims = np.clip(sims, 0.0, 1.0)
    return names, sims.mean(axis=0), sims.std(axis=0)

if use_mc:
    names_mc, mc_mean, mc_std = monte_carlo_climate(base_climate, sensitivity, mc_runs)
    order = [names_mc.index(n) for n in df_adj.index]
    mc_mean = mc_mean[order]; mc_std = mc_std[order]
else:
    mc_mean = np.zeros(len(df_adj)); mc_std = np.zeros(len(df_adj))

df_adj["C6: Rủi ro khí hậu"] = mc_mean
if cargo_value > 50000:
    df_adj["C1: Tỷ lệ phí"] *= 1.1

# TOPSIS
def topsis(df_input, weight_series, cost_flags):
    M = df_input[list(weight_series.index)].values.astype(float)
    denom = np.sqrt((M**2).sum(axis=0)); denom[denom == 0] = 1.0
    R = M / denom
    w = np.array(weight_series.values)
    V = R * w
    is_cost = np.array([cost_flags[c] == "cost" for c in weight_series.index])
    ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
    ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))
    d_plus = np.sqrt(((V - ideal_best)**2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst)**2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)
    return score

cost_flags = {c: ("cost" if c in ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu"] else "benefit") for c in criteria}

# VaR/CVaR
def compute_var_cvar(loss_rates, cargo_value, alpha=0.95):
    losses = np.array(loss_rates) * float(cargo_value)
    if losses.size == 0: return None, None
    var = np.percentile(losses, alpha * 100)
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size > 0 else float(var)
    return float(var), float(cvar)

# Forecast
def forecast_route(route_key, months_ahead=3):
    series = historical[route_key].values
    if use_arima and ARIMA_AVAILABLE:
        try:
            model = ARIMA(series, order=(1,1,1)).fit()
            fc = model.forecast(months_ahead)
            return np.asarray(series), np.asarray(fc)
        except: pass
    last = np_series[-1]
    trend = (last - series[-6]) / 6.0 if len(series) >= 6 else 0.0
    fc = np.array([max(0, last + (i+1)*trend) for i in range(months_ahead)])
    return series, fc

# ================= RUN ANALYSIS =================
if st.button("PHÂN TÍCH & GỢI Ý", type="primary"):
    with st.spinner("Đang chạy mô phỏng..."):
        w = pd.Series(st.session_state["weights"], index=criteria)
        if use_fuzzy:
            f = st.sidebar.slider("Bất định TFN (%)", 0, 50, 15)
            low = np.maximum(w * (1 - f/100.0), 1e-9)
            high = np.minimum(w * (1 + f/100.0), 0.9999)
            defuz = defuzzify_centroid(low, w.values, high)
            w = pd.Series(defuz / defuz.sum(), index=w.index)

        scores = topsis(df_adj, w, cost_flags)
        results = pd.DataFrame({
            "company": df_adj.index,
            "score": scores,
            "C6_mean": mc_mean,
            "C6_std": mc_std
        }).sort_values("score", ascending=False).reset_index(drop=True)
        results["rank"] = results.index + 1
        results["recommend_icc"] = results["score"].apply(lambda s: "ICC A" if s >= 0.75 else ("ICC B" if s >= 0.5 else "ICC C"))

        # Confidence
        eps = 1e-9
        cv_c6 = np.where(results["C6_mean"] == 0, 0.0, results["C6_std"] / (results["C6_mean"] + eps))
        conf_c6 = 1.0 / (1.0 + cv_c6)
        rng = np.ptp(conf_c6)
        conf_c6_scaled = (0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (rng + eps)) if rng > 0 else np.full_like(conf_c6, 0.65)
        conf_final = conf_c6_scaled
        results["confidence"] = np.round(conf_final, 3)

        var95, cvar95 = (compute_var_cvar(results["C6_mean"], cargo_value) if use_var else (None, None))
        hist_series, fc = forecast_route(route)
        months_hist = list(range(1, 13))  # CHỈ 12 THÁNG
        months_fc = list(range(13, 13 + len(fc)))

        # Biểu đồ
        fig_topsis = px.bar(results.sort_values("score"), x="score", y="company", orientation="h",
                            title="TOPSIS Score", text="score", color="score", color_continuous_scale="Blues")
        fig_topsis.update_traces(texttemplate="%{text:.3f}", textposition="outside",
                                 textfont=dict(size=18, family="Arial Black", color="black"))
        fig_topsis = enhance_fig(fig_topsis, title_size=24)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=months_hist, y=hist_series, mode="lines+markers", name="Lịch sử",
                                    line=dict(color="#2A6FDB", width=4), marker=dict(size=10)))
        fig_fc.add_trace(go.Scatter(x=months_fc, y=fc, mode="lines+markers", name="Dự báo",
                                    line=dict(color="#F4B000", width=4, dash="dot"), marker=dict(size=11, symbol="diamond")))
        fig_fc = enhance_fig(fig_fc, title=f"Dự báo rủi ro khí hậu: {route}", font_size=16, title_size=22)
        fig_fc.update_xaxes(title="Tháng", tickmode='array', tickvals=list(range(1,16)), ticktext=[str(i) for i in range(1,16)])
        fig_fc.update_yaxes(title="Mức rủi ro", range=[0,1])

        st.success("Hoàn tất phân tích")
        left, right = st.columns((2,1))
        with left:
            st.subheader("Kết quả xếp hạng TOPSIS")
            st.table(results[["rank", "company", "score", "confidence", "recommend_icc"]].set_index("rank"))
            st.markdown(f"<div class='result-box'>ĐỀ XUẤT: <b>{results.iloc[0]['company']}</b> — Score {results.iloc[0]['score']:.3f}</div>", unsafe_allow_html=True)
        with right:
            st.metric("VaR 95%", f"${var95:,.0f}" if var95 else "N/A")
            st.metric("CVaR 95%", f"${cvar95:,.0f}" if cvar95 else "N/A")

        # Hiển thị biểu đồ
        for fig, name in [(fig_topsis, "TOPSIS"), (fig_fc, "Forecast")]:
            png = fig_to_png_bytes(fig)
            if png:
                st.image(png, use_container_width=True)
            else:
                st.plotly_chart(fig, use_container_width=True)

        # Export Excel
        excel_out = io.BytesIO()
        with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Result", index=False)
            df_adj.to_excel(writer, sheet_name="Adjusted_Data")
            pd.DataFrame({"weight": w.values}, index=w.index).to_excel(writer, sheet_name="Weights")
        excel_out.seek(0)
        st.download_button("Xuất Excel", excel_out, "riskcast_result.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Export PDF (Unicode Safe)
        pdf = FPDF(unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        try:
            pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
            pdf.set_font("DejaVu", size=12)
        except:
            pdf.set_font("Arial", size=12)

        pdf.add_page()
        pdf.set_font_size(18); pdf.cell(0, 10, "RISKCAST v4.9 — Executive Report", ln=1, align='C')
        pdf.set_font_size(11); pdf.ln(5)
        pdf.cell(0, 6, f"Route: {route} | Month: {month} | Value: ${cargo_value:,}", ln=1)
        pdf.cell(0, 6, f"Recommended: {results.iloc[0]['company']} (ICC {results.iloc[0]['recommend_icc'][-1]})", ln=1)
        pdf.ln(5)
        for idx, row in results.head(5).iterrows():
            pdf.cell(0, 6, f"{int(row['rank'])}. {row['company']}: {row['score']:.3f} (Conf: {row['confidence']:.2f})", ln=1)

        # Add images
        for fig in [fig_topsis, fig_fc]:
            pdf.add_page()
            img_data = fig_to_png_bytes(fig, width=1800, height=800, scale=3)
            if img_data:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(img_data); tmp.close()
                pdf.image(tmp.name, x=10, w=190)
                os.unlink(tmp.name)

        try:
            pdf_bytes = pdf.output(dest="S")
        except:
            pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
        st.download_button("Xuất PDF (3 trang)", pdf_bytes, "RISKCAST_report.pdf", "application/pdf")

st.markdown("<br><div style='text-align:center;color:#666;font-size:14px;'>RISKCAST v4.9 — Fixed by Grok | UI Premium Blue</div>", unsafe_allow_html=True)
