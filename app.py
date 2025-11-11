# app.py — ("RISKCAST v4.8.1 — ESG Logistics Dashboard (UI Light)") — patched by Kai + Grok
# -------------------------------------------------------------------
# Mục đích:
# - Ứng dụng minh hoạ mô hình quyết định mua bảo hiểm vận tải quốc tế
# (Fuzzy AHP -> trọng số, TOPSIS -> xếp hạng, Monte Carlo cho C6,
# VaR/CVaR, tùy chọn ARIMA)
# - Phiên bản v4.8: tối ưu ổn định, tránh lỗi scalar/.ptp(), giảm xác suất
# lỗi duplicate element id trên Streamlit.
# - ĐÃ SỬA 100%: CHỮ BIỂU ĐỒ RÕ, TO, ĐẬM, ĐẸP TRÊN WEB & PDF
#
# Hướng dẫn:
# - Cài requirements từ requirements.txt (streamlit, pandas, numpy, plotly, ...)
# - Chạy: streamlit run app.py
# -------------------------------------------------------------------
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

warnings.filterwarnings("ignore")

# ---------------- optional libs ----------------
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

HAS_PIL = False
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# ---------------- page config + css --------------
st.markdown("""
<style>

    /* XÓA toàng hết opacity của mọi container Streamlit (fix bị mờ) */
    [data-testid="stAppViewContainer"],
    [data-testid="stVerticalBlock"],
    .element-container,
    .css-12w0qpk,
    .css-1d391kg,
    .css-1kyxreq,
    .css-1r6slb0,
    .css-1v0mbdj {
        opacity: 1 !important;
        background: transparent !important;
        filter: none !important;
    }

    /* Container chính (phần nội dung giữa) */
    .block-container {
        background: rgba(255, 255, 255, 0.97) !important;
        backdrop-filter: blur(0px) !important;
        border-radius: 16px;
        padding: 2rem 3rem;
        box-shadow: 0px 4px 25px rgba(0,0,0,0.07);
    }

    /* Nền tổng thể nhẹ như website Bảo Việt */
    .stApp {
        background: linear-gradient(180deg,#eef6ff 0%, #e6f2ff 100%) !important;
        color:#00224f !important;
        font-family:"Segoe UI", sans-serif;
    }

    /* chữ trong plotly legend */
    .legendtext {
        fill:#003f88 !important;
        font-weight:600 !important;
    }

    /* tiêu đề */
    h1, h2, h3 {
        color:#003f88 !important;
        font-weight:700 !important;
    }

</style>
""", unsafe_allow_html=True)


st.title("RISKCAST v4.8.1 — ESG Logistics Dashboard (UI Light)")
st.caption("Fuzzy AHP + TOPSIS + Monte Carlo (C6) + VaR/CVaR + (optional) ARIMA")

# ================= Sidebar inputs =================
with st.sidebar:
    st.header("Thông tin lô hàng")
    cargo_value = st.number_input("Giá trị lô hàng (USD)", min_value=1000, value=39000, step=1000, key="sid_cargo_value")
    good_type = st.selectbox("Loại hàng", ["Điện tử", "Đông lạnh", "Hàng khô", "Hàng nguy hiểm", "Khác"], key="sid_good_type")
    route = st.selectbox("Tuyến", ["VN - EU", "VN - US", "VN - Singapore", "VN - China", "Domestic"], key="sid_route")
    method = st.selectbox("Phương thức", ["Sea", "Air", "Truck"], key="sid_method")
    month = st.selectbox("Tháng (1-12)", list(range(1,13)), index=8, key="sid_month")
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"], key="sid_priority")
    st.markdown("---")
    st.header("Mô hình")
    use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN)", True, key="sid_use_fuzzy")
    use_arima = st.checkbox("Dùng ARIMA (nếu có)", True, key="sid_use_arima")
    use_mc = st.checkbox("Chạy Monte Carlo cho C6", True, key="sid_use_mc")
    use_var = st.checkbox("Tính VaR và CVaR", True, key="sid_use_var")
    mc_runs = st.number_input("Số vòng Monte Carlo", 500, 10000, value=2000, step=500, key="sid_mc_runs")

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
    if abs(diff) > 1e-8:
        w[free_idx[0]] += diff
    return np.round(w, 6)

def defuzzify_centroid(low, mid, high):
    return (low + mid + high) / 3.0

def fig_to_png_bytes(fig, width=1400, height=600, scale=3):
    """Chuyển Plotly fig thành PNG bytes (chất lượng cao)"""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except:
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.write_image(tmp.name, width=width, height=height, scale=scale)
                with open(tmp.name, "rb") as f:
                    data = f.read()
                os.unlink(tmp.name)
                return data
        except Exception as e:
            st.error(f"Lỗi xuất ảnh: {e}")
            return None

def enhance_fig(fig, title=None, font_size=14, title_size=18):
    """Tăng độ rõ nét chữ, tiêu đề, legend cho mọi biểu đồ"""
    fig.update_layout(
        template="simple_white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=font_size, color="#003060"),
        title=dict(
            text=title or fig.layout.title.text,
            font=dict(size=title_size, family="Segoe UI, Arial, sans-serif", color="#2A6FDB"),
            x=0.5, xanchor="center"
        ),
        legend=dict(
            font=dict(size=font_size),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e6edf7",
            borderwidth=1
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=90, b=60),
        hoverlabel=dict(font_size=font_size)
    )
    fig.update_xaxes(title_font=dict(size=font_size+2), tickfont=dict(size=font_size))
    fig.update_yaxes(title_font=dict(size=font_size+2), tickfont=dict(size=font_size))
    return fig

# ================= Sample data (demo) =================
@st.cache_data
def load_data():
    months = list(range(1,13))
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

# ================= Criteria & initial weights =================
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
    key_lock = f"lock_{i}_v8"
    key_w = f"w_{i}_v8"
    with cols[i]:
        st.markdown(f"**{c}**")
        st.checkbox("Lock", value=st.session_state["locked"][i], key=key_lock)
        val = st.number_input("Tỉ lệ", min_value=0.0, max_value=1.0, value=float(new_w[i]), step=0.01, key=key_w)
        new_w[i] = val

for i in range(len(criteria)):
    st.session_state["locked"][i] = st.session_state.get(f"lock_{i}_v8", False)

if st.button("Reset trọng số mặc định", key="reset_weights_v8"):
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15], dtype=float)
    st.session_state["locked"] = [False]*len(criteria)
else:
    st.session_state["weights"] = auto_balance(new_w, st.session_state["locked"])

weights = pd.Series(st.session_state["weights"], index=criteria)

# Biểu đồ Pie Weights (Realtime) - RÕ CHỮ
fig_weights = px.pie(
    values=weights.values,
    names=weights.index,
    title="Phân bổ trọng số (Realtime)",
    color_discrete_sequence=px.colors.sequential.Blues
)
fig_weights = enhance_fig(fig_weights, title="Phân bổ trọng số (Realtime)", font_size=14, title_size=18)

# Hiển thị bằng PNG chất lượng cao
png_weights = fig_to_png_bytes(fig_weights, width=800, height=500, scale=3)
if png_weights:
    st.image(png_weights, use_container_width=True)
else:
    st.plotly_chart(fig_weights, use_container_width=True, key="fig_weights_main")

# ================= Insurance companies demo =================
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
base_climate = float(historical.loc[historical["month"]==month, route_key].iloc[0]) if month in historical['month'].values else 0.4
df_adj = df.copy().astype(float)

# ---------------- Monte Carlo vectorized ----------------
@st.cache_data
def monte_carlo_climate(base, sens_map, runs):
    rng = np.random.default_rng(2025)
    names = list(sens_map.keys())
    mu = np.array([base * sens_map[n] for n in names], dtype=float)
    sigma = np.maximum(0.03, mu * 0.12)
    sims = rng.normal(loc=mu, scale=sigma, size=(int(runs), len(names)))
    sims = np.clip(sims, 0.0, 1.0)
    return names, sims.mean(axis=0), sims.std(axis=0)

if use_mc:
    names_mc, mc_mean, mc_std = monte_carlo_climate(base_climate, sensitivity, mc_runs)
    order = [names_mc.index(n) for n in df_adj.index]
    mc_mean = mc_mean[order]
    mc_std = mc_std[order]
else:
    mc_mean = np.zeros(len(df_adj), dtype=float)
    mc_std = np.zeros(len(df_adj), dtype=float)

df_adj["C6: Rủi ro khí hậu"] = mc_mean
if cargo_value > 50000:
    df_adj["C1: Tỷ lệ phí"] *= 1.1

# ================= TOPSIS =================
def topsis(df_input, weight_series, cost_flags):
    M = df_input[list(weight_series.index)].values.astype(float)
    denom = np.sqrt((M**2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = M / denom
    w = np.array(weight_series.values, dtype=float)
    V = R * w
    is_cost = np.array([cost_flags[c] == "cost" for c in weight_series.index])
    ideal_best = np.where(is_cost, V.min(axis=0), V.max(axis=0))
    ideal_worst = np.where(is_cost, V.max(axis=0), V.min(axis=0))
    d_plus = np.sqrt(((V - ideal_best)**2).sum(axis=1))
    d_minus = np.sqrt(((V - ideal_worst)**2).sum(axis=1))
    score = d_minus / (d_plus + d_minus + 1e-12)
    return score

cost_flags = {c: ("cost" if c in ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu"] else "benefit") for c in criteria}

# ================= VaR / CVaR =================
def compute_var_cvar(loss_rates, cargo_value, alpha=0.95):
    losses = np.array(loss_rates, dtype=float) * float(cargo_value)
    if losses.size == 0:
        return None, None
    var = np.percentile(losses, alpha * 100)
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size > 0 else float(var)
    return float(var), float(cvar)

# ================= Forecast (ARIMA fallback) =================
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
    trend = (last[-1] - last[-6]) / 6.0 if len(last) >= 6 else 0.0
    fc = np.array([max(0, last[-1] + (i+1)*trend) for i in range(months_ahead)])
    return last, fc

# ================= Main action =================
if st.button("PHÂN TÍCH & GỢI Ý", key="run_analysis_v8"):
    with st.spinner("Đang chạy mô phỏng và tối ưu..."):
        w = pd.Series(st.session_state["weights"], index=criteria)
        if use_fuzzy:
            f = st.sidebar.slider("Bất định TFN (%)", 0, 50, 15, key="sid_tfn_v8")
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

        eps = 1e-9
        cv_c6 = np.where(results["C6_mean"].values == 0, 0.0, results["C6_std"].values / (results["C6_mean"].values + eps))
        conf_c6 = 1.0 / (1.0 + cv_c6)
        conf_c6 = np.atleast_1d(conf_c6)
        rng = np.ptp(conf_c6)
        conf_c6_scaled = (0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (rng + eps)) if rng > 0 else np.full_like(conf_c6, 0.65)
        crit_cv = df_adj.std(axis=1).values / (df_adj.mean(axis=1).values + eps)
        conf_crit = np.atleast_1d(1.0 / (1.0 + crit_cv))
        rng2 = np.ptp(conf_crit)
        conf_crit_scaled = (0.3 + 0.7 * (conf_crit - conf_crit.min()) / (rng2 + eps)) if rng2 > 0 else np.full_like(conf_crit, 0.65)
        conf_final = np.sqrt(conf_c6_scaled * conf_crit_scaled)
        order_map = {comp: float(conf_final[i]) for i, comp in enumerate(df_adj.index)}
        results["confidence"] = results["company"].map(order_map).round(3)

        var95, cvar95 = (compute_var_cvar(results["C6_mean"].values, cargo_value, alpha=0.95) if use_var else (None, None))

        hist_series, fc = forecast_route(route)
        months_hist = list(range(1, len(hist_series) + 1))
        months_fc = list(range(len(hist_series) + 1, len(hist_series) + 1 + len(fc)))

        # Biểu đồ TOPSIS - CHỮ TO, ĐẬM, RÕ
        fig_topsis = px.bar(
            results.sort_values("score"),
            x="score", y="company", orientation="h",
            title="TOPSIS Score (cao hơn = tốt hơn)",
            text="score", color="score",
            color_continuous_scale="Blues"
        )
        fig_topsis.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            textfont=dict(size=18, color="black", family="Arial Black"),
            marker_line_width=2
        )
        fig_topsis = enhance_fig(fig_topsis, font_size=16, title_size=22)

        # Biểu đồ Forecast
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=months_hist, y=hist_series,
            mode="lines+markers", name="Lịch sử",
            line=dict(color="#2A6FDB", width=4),
            marker=dict(size=10)
        ))
        fig_fc.add_trace(go.Scatter(
            x=months_fc, y=fc,
            mode="lines+markers", name="Dự báo",
            line=dict(color="#F4B000", width=4, dash="dot"),
            marker=dict(size=11, symbol="diamond")
        ))
        fig_fc = enhance_fig(fig_fc, title=f"Dự báo rủi ro khí hậu: {route}", font_size=15, title_size=20)
        fig_fc.update_xaxes(title="Tháng", tickmode='linear', tickfont=dict(size=15))
        fig_fc.update_yaxes(title="Mức rủi ro (0-1)", range=[0, 1], tickfont=dict(size=15))

        # Pie chart right
        fig_weights_right = px.pie(
            values=w.values, names=w.index,
            title="Trọng số cuối cùng",
            color_discrete_sequence=px.colors.sequential.Greens
        )
        fig_weights_right = enhance_fig(fig_weights_right, title="Trọng số cuối cùng", font_size=13, title_size=16)

        st.success("Hoàn tất phân tích")
        left, right = st.columns((2,1))

        with left:
            st.subheader("Kết quả xếp hạng TOPSIS")
            st.table(results[["rank", "company", "score", "confidence", "recommend_icc"]].set_index("rank"))
            st.markdown(f"<div class='result-box'><b>ĐỀ XUẤT:</b> {results.iloc[0]['company']} — Score {results.iloc[0]['score']:.3f} — Confidence {results.iloc[0]['confidence']:.2f}</div>", unsafe_allow_html=True)

        with right:
            st.metric("VaR 95%", f"${var95:,.0f}" if var95 else "N/A")
            st.metric("CVaR 95%", f"${cvar95:,.0f}" if cvar95 else "N/A")
            png_right = fig_to_png_bytes(fig_weights_right, width=600, height=400, scale=3)
            if png_right:
                st.image(png_right, use_container_width=True)
            else:
                st.plotly_chart(fig_weights_right, use_container_width=True, key="fig_weights_right_v8")

        # Hiển thị biểu đồ chính bằng PNG
        png_topsis = fig_to_png_bytes(fig_topsis, width=1400, height=600, scale=3)
        if png_topsis:
            st.image(png_topsis, use_container_width=True)
        else:
            st.plotly_chart(fig_topsis, use_container_width=True, key="fig_topsis_v8")

        png_fc = fig_to_png_bytes(fig_fc, width=1400, height=600, scale=3)
        if png_fc:
            st.image(png_fc, use_container_width=True)
        else:
            st.plotly_chart(fig_fc, use_container_width=True, key="fig_fc_v8")

        # ---------------- Export Excel ----------------
        excel_out = io.BytesIO()
        with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Result", index=False)
            df_adj.to_excel(writer, sheet_name="Adjusted_Data")
            pd.DataFrame({"weight": w.values}, index=w.index).to_excel(writer, sheet_name="Weights")
        excel_out.seek(0)
        st.download_button("Xuất Excel (Kết quả)", excel_out, file_name="riskcast_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_excel_v8")

        # ---------------- Export PDF ----------------
        pdf = FPDF(unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=12)
        try:
            pdf.add_font("DejaVu", "", fname="DejaVuSans.ttf", uni=True)
            pdf.set_font("DejaVu", size=12)
        except Exception:
            pdf.set_font("Arial", size=12)

        pdf.add_page()
        pdf.set_font_size(16)
        pdf.cell(0, 8, "RISKCAST v4.8 — Executive Summary", ln=1)
        pdf.ln(2)
        pdf.set_font_size(10)
        pdf.cell(0, 6, f"Route: {route} Month: {month} Method: {method}", ln=1)
        pdf.cell(0, 6, f"Cargo value: ${cargo_value:,} Priority: {priority}", ln=1)
        pdf.ln(4)
        pdf.set_font_size(11)
        summary_text = f"Recommended insurer: {results.iloc[0]['company']} ({results.iloc[0]['recommend_icc']})\nTOPSIS Score: {results.iloc[0]['score']:.4f}\nConfidence: {results.iloc[0]['confidence']:.2f}"
        if var95 is not None:
            summary_text += f"\nVaR95: ${var95:,.0f} | CVaR95: ${cvar95:,.0f}"
        pdf.multi_cell(0, 6, summary_text, align="L")
        pdf.ln(6)
        pdf.set_font_size(10)
        pdf.cell(20,6,"Rank",1); pdf.cell(60,6,"Company",1); pdf.cell(40,6,"Score",1); pdf.cell(35,6,"Confidence",1); pdf.ln()
        for idx, row in results.head(5).iterrows():
            pdf.cell(20,6,str(int(row["rank"])),1); pdf.cell(60,6,str(row["company"])[:30],1)
            pdf.cell(40,6,f"{row['score']:.4f}",1); pdf.cell(35,6,f"{row['confidence']:.2f}",1); pdf.ln()

        pdf.add_page()
        pdf.set_font_size(14)
        pdf.cell(0,8,"TOPSIS Scores", ln=1)
        img_bytes = fig_to_png_bytes(fig_topsis, width=1400, height=600, scale=3)
        if img_bytes:
            tmp = f"tmp_{uuid.uuid4().hex}_topsis.png"
            with open(tmp, "wb") as f:
                f.write(img_bytes)
            pdf.image(tmp, x=15, w=180)
            os.remove(tmp)
        else:
            pdf.cell(0,6,"(Không thể xuất biểu đồ TOPSIS)", ln=1)

        pdf.add_page()
        pdf.set_font_size(14)
        pdf.cell(0,8,"Forecast (ARIMA or fallback) & VaR", ln=1)
        img_bytes2 = fig_to_png_bytes(fig_fc, width=1400, height=600, scale=3)
        if img_bytes2:
            tmp2 = f"tmp_{uuid.uuid4().hex}_forecast.png"
            with open(tmp2, "wb") as f:
                f.write(img_bytes2)
            pdf.image(tmp2, x=10, w=190)
            os.remove(tmp2)
        else:
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
        st.download_button("Xuất PDF báo cáo (3 trang)", data=pdf_bytes, file_name="RISKCAST_report.pdf", mime="application/pdf", key="dl_pdf_v8")

# ================= Footer =================
st.markdown("<br><div class='muted-small'>RISKCAST v4.8 — Full comment version. Author: Bùi Xuân Hoàng. UI & Chart Clarity Enhanced by Grok.</div>", unsafe_allow_html=True)
