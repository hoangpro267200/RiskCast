# app.py — RISKCAST v4.7 (fixed ptp & robust) — patched by Kai
# -------------------------------------------------------------------
# Mục đích:
#   - Ứng dụng Streamlit để minh hoạ và chạy mô hình ra quyết định mua
#     bảo hiểm vận tải quốc tế dựa trên: Fuzzy AHP, TOPSIS, Monte Carlo,
#     VaR & CVaR, và (tùy chọn) ARIMA cho dự báo chuỗi thời gian.
#   - Phiên bản này tối ưu ổn định, xử lý edge-case (scalar arrays),
#     export Excel/PDF an toàn, và chứa comment tiếng Việt phục vụ NCKH.
#
# Hướng dẫn ngắn:
#   - Cài dependencies trong requirements.txt (streamlit, numpy, pandas,
#     plotly, statsmodels, openpyxl, fpdf, Pillow nếu muốn xuất ảnh).
#   - Chạy: streamlit run app.py
#   - Trang bên trái là input (giá trị lô hàng, tuyến, phương thức, v.v.)
#   - Nhấn "PHÂN TÍCH & GỢI Ý" để chạy mô phỏng & nhận kết quả.
#
# Ghi chú NCKH (tài liệu kèm):
#   - Ý tưởng: kết hợp yếu tố định tính (Fuzzy AHP) -> trọng số tiêu chí,
#     và phương pháp nhiều tiêu chí (TOPSIS) -> xếp hạng nhà bảo hiểm.
#   - Độ bất định (TFN) cho phép mô phỏng độ không chắc chắn của trọng số.
#   - C6 (Rủi ro khí hậu) mô phỏng bằng Monte Carlo dựa trên hệ số nhạy
#     cảm (sensitivity) theo từng nhà bảo hiểm và tuyến hàng.
#   - VaR/CVaR tính trên tổn thất kỳ vọng = loss_rate * cargo_value.
#
# Tổ chức code:
#   - Phần đầu: import + config giao diện
#   - Helpers: auto_balance, defuzzify_centroid, try_plotly_to_png
#   - Dữ liệu mẫu (load_sample_data)
#   - GUI (sidebar + trọng số)
#   - Mô phỏng Monte Carlo (vectorized, cached)
#   - Hàm TOPSIS, VaR/CVaR, Forecast
#   - Phân tích & xuất kết quả
# -------------------------------------------------------------------

import io
import math
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")

# Optional ARIMA (nếu cài statsmodels)
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

# Optional image libs (Pillow) — dùng để xử lý ảnh trước khi chèn vào PDF
HAS_PIL = False
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False

# Page config + CSS (giao diện "Green ESG")
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

# -----------------------------
# Sidebar inputs (UI chính)
# -----------------------------
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

# -----------------------------
# Helpers (Hàm tiện ích)
# -----------------------------
def auto_balance(weights, locked_flags):
    """
    Cân bằng tự động trọng số khi một số tiêu chí bị lock.
    - weights: array-like (tiền chỉnh)
    - locked_flags: boolean list (True nếu đã lock)
    Trả về mảng trọng số chuẩn hoá (tổng = 1).
    """
    w = np.array(weights, dtype=float)
    locked = np.array(locked_flags, dtype=bool)
    locked_sum = w[locked].sum()
    free_idx = np.where(~locked)[0]
    # Nếu không còn tiêu chí tự do
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
    """
    Defuzzification theo centroid đơn giản cho TFN.
    low, mid, high có thể là mảng.
    """
    return (low + mid + high) / 3.0

def try_plotly_to_png(fig):
    """
    Thử nhiều cách xuất plotly figure sang bytes PNG.
    Trả về bytes hoặc None nếu thất bại.
    """
    # Cách 1: fig.to_image() (kaleido)
    try:
        return fig.to_image(format="png")
    except Exception:
        pass
    # Cách 2: write_image -> save tạm -> read
    try:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        path = tmp.name
        fig.write_image(path)
        tmp.close()
        with open(path, "rb") as f:
            data = f.read()
        os.remove(path)
        return data
    except Exception:
        return None

# -----------------------------
# Dữ liệu mẫu cho demo / NCKH
# -----------------------------
@st.cache_data
def load_sample_data():
    """
    Tạo dữ liệu lịch sử mẫu cho từng tuyến (thông số rủi ro cơ bản theo tháng)
    và một bộ mẫu tổn thất (loss_rate) để tính VaR.
    Trong nghiên cứu thực tế, thay thế bằng dữ liệu lịch sử.
    """
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

# -----------------------------
# Tiêu chí, trọng số khởi tạo
# -----------------------------
criteria = ["C1: Tỷ lệ phí", "C2: Thời gian xử lý", "C3: Tỷ lệ tổn thất",
            "C4: Hỗ trợ ICC", "C5: Chăm sóc KH", "C6: Rủi ro khí hậu"]

# Khởi tạo session_state để giữ trạng thái trọng số giữa các lần tương tác
if "weights" not in st.session_state:
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15], dtype=float)
if "locked" not in st.session_state:
    st.session_state["locked"] = [False]*6

# -----------------------------
# UI: phân bổ trọng số (lock + auto-balance)
# -----------------------------
st.subheader("⚖️ Phân bổ trọng số tiêu chí (Lock & Auto-balance)")
cols = st.columns(6)
new_w = st.session_state["weights"].copy()
for i,c in enumerate(criteria):
    with cols[i]:
        st.markdown(f"**{c}**")
        # checkbox lock: lưu vào session_state khóa riêng cho mỗi tiêu chí
        st.checkbox("🔒 Lock", value=st.session_state["locked"][i], key=f"lock_{i}", on_change=None)
        val = st.number_input("Tỉ lệ", min_value=0.0, max_value=1.0, value=float(new_w[i]), step=0.01, key=f"w_in_{i}")
        new_w[i] = val
# cập nhật flags locked
for i in range(6):
    st.session_state["locked"][i] = st.session_state.get(f"lock_{i}", False)

# Nút reset trọng số mặc định
if st.button("🔄 Reset trọng số mặc định"):
    st.session_state["weights"] = np.array([0.20,0.15,0.20,0.20,0.10,0.15], dtype=float)
    st.session_state["locked"] = [False]*6
else:
    st.session_state["weights"] = auto_balance(new_w, st.session_state["locked"])

weights_series = pd.Series(st.session_state["weights"], index=criteria)
fig_weights = px.pie(values=weights_series.values, names=weights_series.index, title="Phân bổ trọng số (Realtime)")
st.plotly_chart(fig_weights, use_container_width=True)

# -----------------------------
# Dữ liệu mẫu các công ty bảo hiểm
# -----------------------------
df = pd.DataFrame({
    "Company": ["Chubb","PVI","InternationalIns","BaoViet","Aon"],
    "C1: Tỷ lệ phí": [0.30,0.28,0.26,0.32,0.24],
    "C2: Thời gian xử lý": [6,5,8,7,4],
    "C3: Tỷ lệ tổn thất": [0.08,0.06,0.09,0.10,0.07],
    "C4: Hỗ trợ ICC": [9,8,6,9,7],
    "C5: Chăm sóc KH": [9,8,5,7,6],
}).set_index("Company")

# sensitivity: hệ số nhạy cảm với rủi ro khí hậu (các giá trị mẫu)
sensitivity = {"Chubb":0.95,"PVI":1.10,"InternationalIns":1.20,"BaoViet":1.05,"Aon":0.90}

route_key = route
# lấy giá trị cơ bản rủi ro theo tháng / tuyến (từ dữ liệu mẫu)
base_climate = float(historical.loc[historical['month']==month, route_key].iloc[0]) if month in historical['month'].values else 0.40

df_adj = df.copy().astype(float)

# -----------------------------
# Monte Carlo vectorized cho C6 (hiệu năng tốt)
# -----------------------------
@st.cache_data
def monte_carlo_climate(base_climate, sensitivity_map, mc_runs, rng_seed=2025):
    """
    Monte Carlo vectorized:
      - base_climate: giá trị rủi ro cơ bản của tuyến (0-1)
      - sensitivity_map: dict{company: multiplier}
      - mc_runs: số vòng mô phỏng
    Trả về: names, means, stds (mảng tương ứng)
    """
    rng = np.random.default_rng(rng_seed)
    names = list(sensitivity_map.keys())
    n = len(names)
    mu = np.array([base_climate * sensitivity_map.get(name,1.0) for name in names], dtype=float)
    sigma = np.maximum(0.03, mu * 0.12)
    sims = rng.normal(loc=mu, scale=sigma, size=(int(mc_runs), n))
    sims = np.clip(sims, 0.0, 1.0)
    means = sims.mean(axis=0)
    stds = sims.std(axis=0)
    return names, means, stds

if use_mc:
    names_mc, mc_mean, mc_std = monte_carlo_climate(base_climate, sensitivity, mc_runs)
    # đảm bảo thứ tự khớp với df_adj.index
    order = [names_mc.index(nm) for nm in df_adj.index]
    mc_mean = mc_mean[order]
    mc_std = mc_std[order]
else:
    mc_mean = np.zeros(len(df_adj), dtype=float)
    mc_std = np.zeros(len(df_adj), dtype=float)

# gán C6 (rủi ro khí hậu) vào bảng dữ liệu
df_adj["C6: Rủi ro khí hậu"] = mc_mean

# Điều chỉnh: nếu cargo_value quá lớn, tỷ lệ phí có thể tăng (ví dụ)
if cargo_value > 50000:
    df_adj["C1: Tỷ lệ phí"] *= 1.1

# -----------------------------
# Hàm TOPSIS (chuẩn)
# -----------------------------
def topsis(df_input, weight_series, cost_flags):
    """
    TOPSIS chuẩn:
      - df_input: DataFrame có các cột tương ứng với weight_series.index
      - weight_series: Series với index = tiêu chí, giá trị = trọng số (tổng=1)
      - cost_flags: dict tiêu chí -> "cost" hoặc "benefit"
    Trả về mảng score (0..1), càng lớn càng tốt.
    """
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

# Đánh dấu tiêu chí là cost/benefit (một số giả định minh hoạ)
cost_flags = {c: "cost" if c in ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu"] else "benefit" for c in criteria}

# -----------------------------
# VaR / CVaR (đơn giản)
# -----------------------------
def compute_var_cvar(loss_rates, cargo_value, alpha=0.95):
    """
    Tính VaR & CVaR đơn giản:
      - loss_rates: mảng tỷ lệ tổn thất (0..1)
      - cargo_value: giá trị hàng
      - alpha: confidence level (ví dụ 0.95)
    Trả về (VaR, CVaR) tính theo USD; None nếu không có dữ liệu.
    """
    eps = 1e-9
    losses = np.array(loss_rates, dtype=float) * float(cargo_value)
    if losses.size == 0:
        return None, None
    var = np.percentile(losses, alpha*100)
    # tail: phần vượt hoặc bằng VaR
    tail = losses[losses >= var - eps]
    cvar = float(tail.mean()) if len(tail)>0 else float(var)
    return float(var), float(cvar)

# -----------------------------
# Forecast route (ARIMA fallback)
# -----------------------------
def forecast_route(route_key, months_ahead=3):
    """
    Dự báo rủi ro theo tuyến:
      - Nếu ARIMA có sẵn, cố gắng fit ARIMA(1,1,1)
      - Nếu thất bại, fallback bằng trend tuyến tính đơn giản
    """
    series = historical[route_key].values if route_key in historical.columns else historical.iloc[:,1].values
    if use_arima and ARIMA_AVAILABLE:
        try:
            model = ARIMA(series, order=(1,1,1)).fit()
            fc = model.forecast(months_ahead)
            return np.asarray(series), np.asarray(fc)
        except Exception:
            pass
    last = np.array(series)
    avg = np.mean(last[-6:]) if len(last)>=6 else np.mean(last)
    trend = (last[-1] - last[-6]) / 6.0 if len(last)>=6 else 0.0
    fc = np.array([max(0, last[-1] + (i+1)*trend) for i in range(months_ahead)])
    return last, fc

# -----------------------------
# Nút chạy chính: Phân tích & Gợi ý
# -----------------------------
if st.button("🚀 PHÂN TÍCH & GỢI Ý"):
    with st.spinner("Đang chạy mô phỏng và tối ưu..."):
        # weights hiện thời (Series) — có thể bị defuzzify nếu dùng fuzzy
        weights = weights_series.copy()
        if use_fuzzy:
            # "Bất định TFN (%)": mức biến động cho TFN (ví dụ 15%)
            f = float(st.sidebar.slider("Bất định TFN (%)", 0, 50, 15))
            low = np.maximum(weights*(1 - f/100.0), 1e-9)
            high = np.minimum(weights*(1 + f/100.0), 0.9999)
            defuz = defuzzify_centroid(low, weights.values, high)
            weights = pd.Series(defuz/defuz.sum(), index=weights.index)

        # TOPSIS: tính score cho từng công ty
        scores = topsis(df_adj, weights, cost_flags)
        results = pd.DataFrame({
            "company": df_adj.index,
            "score": scores,
            "C6_mean": mc_mean,
            "C6_std": mc_std
        }).sort_values("score", ascending=False).reset_index(drop=True)
        results["rank"] = results.index + 1
        results["recommend_icc"] = results["score"].apply(lambda x: "ICC A" if x>=0.75 else ("ICC B" if x>=0.5 else "ICC C"))

        # -----------------------------
        # Confidence calc (robust)
        # Ghi chú NCKH: confidence = hàm kết hợp giữa độ ổn định của C6 và độ biến động giữa các tiêu chí
        # -----------------------------
        eps = 1e-9
        cv_c6 = np.where(results["C6_mean"].values == 0, 0.0,
                         results["C6_std"].values / (results["C6_mean"].values + eps))
        conf_c6 = 1.0 / (1.0 + cv_c6)

        # đảm bảo numpy array, xử lý trường hợp scalar
        conf_c6 = np.asarray(conf_c6)
        if conf_c6.ndim == 0:
            conf_c6 = np.full(len(results), float(conf_c6))

        if np.ptp(conf_c6) > 0:
            conf_c6_scaled = 0.3 + 0.7 * (conf_c6 - conf_c6.min()) / (np.ptp(conf_c6) + 1e-12)
        else:
            conf_c6_scaled = np.full_like(conf_c6, 0.65, dtype=float)

        # độ biến động giữa các tiêu chí (critic variability)
        crit_cv = df_adj.std(axis=1).values / (df_adj.mean(axis=1).values + eps)
        conf_crit = 1.0 / (1.0 + crit_cv)

        # ensure array
        conf_crit = np.asarray(conf_crit)
        if conf_crit.ndim == 0:
            conf_crit = np.full(len(results), float(conf_crit))

        if np.ptp(conf_crit) > 0:
            conf_crit_scaled = 0.3 + 0.7 * (conf_crit - conf_crit.min()) / (np.ptp(conf_crit) + 1e-12)
        else:
            conf_crit_scaled = np.full_like(conf_crit, 0.65, dtype=float)

        # confidence cuối cùng: geometric mean giữa 2 yếu tố (tương đối)
        conf_final = np.sqrt(conf_c6_scaled * conf_crit_scaled)
        order_map = {comp: float(conf_final[i]) for i, comp in enumerate(df_adj.index)}
        results["confidence"] = results["company"].map(order_map).round(3)

        # -----------------------------
        # VaR & CVaR (nếu bật)
        # -----------------------------
        var95, cvar95 = (None, None)
        if use_var:
            var95, cvar95 = compute_var_cvar(results["C6_mean"].values, cargo_value, alpha=0.95)

        # -----------------------------
        # Forecast cho tuyến + vẽ biểu đồ
        # -----------------------------
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

        # -----------------------------
        # Export Excel (cải tiến)
        # -----------------------------
        excel_out = io.BytesIO()
        with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
            results.to_excel(writer, sheet_name="Result", index=False)
            df_adj.to_excel(writer, sheet_name="Adjusted_Data")
            pd.DataFrame({"weight": weights.values}, index=weights.index).to_excel(writer, sheet_name="Weights")
        excel_out.seek(0)
        st.download_button("⬇️ Xuất Excel (Kết quả)", excel_out, file_name="riskcast_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # -----------------------------
        # Export PDF (3 trang) — xử lý ảnh an toàn
        # -----------------------------
        pdf = FPDF(unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=12)
        # Thử thêm font DejaVu nếu có ttf trong repo, nếu không fallback Arial
        try:
            pdf.add_font("DejaVu", "", fname="DejaVuSans.ttf", uni=True)
            pdf.set_font("DejaVu", size=12)
        except Exception:
            pdf.set_font("Arial", size=12)

        # Trang 1: Executive summary + bảng top 5
        pdf.add_page()
        pdf.set_font_size(16)
        pdf.cell(0, 8, "RISKCAST v4.7 — Executive Summary", ln=1)
        pdf.ln(2)
        pdf.set_font_size(10)
        pdf.cell(0, 6, f"Route: {route}    Month: {month}    Method: {method}", ln=1)
        pdf.cell(0, 6, f"Cargo value: ${cargo_value:,}    Priority: {priority}", ln=1)
        pdf.ln(4)
        pdf.set_font_size(11)
        summary_text = f"Recommended insurer: {results.iloc[0]['company']} ({results.iloc[0]['recommend_icc']})\nTOPSIS Score: {results.iloc[0]['score']:.4f}\nConfidence: {results.iloc[0]['confidence']:.2f}"
        if var95 is not None:
            summary_text += f"\nVaR95: ${var95:,.0f} | CVaR95: ${cvar95:,.0f}"
        pdf.multi_cell(0, 6, summary_text, align="L")
        pdf.ln(6)
        pdf.set_font_size(10)
        # table header
        pdf.cell(20,6,"Rank",1); pdf.cell(60,6,"Company",1); pdf.cell(40,6,"Score",1); pdf.cell(35,6,"Confidence",1); pdf.ln()
        for idx, row in results.head(5).iterrows():
            pdf.cell(20,6,str(int(row["rank"])),1); pdf.cell(60,6,str(row["company"])[:30],1)
            pdf.cell(40,6,f"{row['score']:.4f}",1); pdf.cell(35,6,f"{row['confidence']:.2f}",1); pdf.ln()

        # Trang 2: biểu đồ TOPSIS
        pdf.add_page()
        pdf.set_font_size(14)
        pdf.cell(0,8,"TOPSIS Scores", ln=1)
        img_bytes = try_plotly_to_png(fig_topsis)
        if img_bytes and HAS_PIL:
            try:
                im = Image.open(io.BytesIO(img_bytes))
                tmp = f"tmp_{uuid.uuid4().hex}_topsis.png"
                im.save(tmp)
                pdf.image(tmp, x=15, w=180)
            except Exception:
                pdf.set_font_size(10)
                pdf.cell(0,6,"(Không thể xuất biểu đồ TOPSIS — PIL export failed)", ln=1)
        elif img_bytes:
            # nếu không có PIL, thử ghi trực tiếp file tạm
            try:
                tmp = f"tmp_{uuid.uuid4().hex}_topsis.png"
                with open(tmp, "wb") as f:
                    f.write(img_bytes)
                pdf.image(tmp, x=15, w=180)
            except Exception:
                pdf.set_font_size(10)
                pdf.cell(0,6,"(Không thể xuất biểu đồ TOPSIS — image save failed)", ln=1)
        else:
            pdf.set_font_size(10)
            pdf.cell(0,6,"(Biểu đồ TOPSIS không thể xuất sang ảnh)", ln=1)
            pdf.ln(4)
            for idx,row in results.iterrows():
                pdf.cell(0,5,f"{int(row['rank'])}. {row['company']} — Score: {row['score']:.4f} — Conf: {row['confidence']:.2f}", ln=1)

        # Trang 3: Forecast & VaR
        pdf.add_page()
        pdf.set_font_size(14)
        pdf.cell(0,8,"Forecast (ARIMA or fallback) & VaR", ln=1)
        img_bytes2 = try_plotly_to_png(fig_forecast)
        if img_bytes2 and HAS_PIL:
            try:
                im2 = Image.open(io.BytesIO(img_bytes2))
                tmp2 = f"tmp_{uuid.uuid4().hex}_forecast.png"
                im2.save(tmp2)
                pdf.image(tmp2, x=10, w=190)
            except Exception:
                pdf.set_font_size(10)
                pdf.cell(0,6,"(Không thể xuất biểu đồ Forecast — PIL failed)", ln=1)
        elif img_bytes2:
            try:
                tmp2 = f"tmp_{uuid.uuid4().hex}_forecast.png"
                with open(tmp2, "wb") as f:
                    f.write(img_bytes2)
                pdf.image(tmp2, x=10, w=190)
            except Exception:
                pdf.set_font_size(10)
                pdf.cell(0,6,"(Không thể xuất biểu đồ Forecast — image save failed)", ln=1)
        else:
            pdf.set_font_size(10)
            pdf.cell(0,6,"(Biểu đồ Forecast không thể xuất)", ln=1)
        pdf.ln(6)
        if var95 is not None:
            pdf.set_font_size(11)
            pdf.cell(0,6,f"VaR 95%: ${var95:,.0f}", ln=1)
            pdf.cell(0,6,f"CVaR 95%: ${cvar95:,.0f}", ln=1)

        # Output PDF bytes
        try:
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
        except Exception:
            pdf_bytes = pdf.output(dest="S").encode("utf-8", errors="ignore")
        st.download_button("⬇️ Xuất PDF báo cáo (3 trang)", data=pdf_bytes, file_name="RISKCAST_report.pdf", mime="application/pdf")

# Footer / credit
st.markdown("<br><div class='muted small'>RISKCAST v4.7 — Green ESG theme. Author: Bùi Xuân Hoàng.</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Hướng phát triển / kiểm thử (gợi ý cho phần NCKH)
# -------------------------------------------------------------------
# 1) Nghiên cứu thực nghiệm:
#    - Thay 'historical' bằng dữ liệu thực tế lấy từ cảng, weather API, claim logs.
#    - So sánh kết quả model (top insurer) với lựa chọn thực tế, tính mức tiết kiệm phí.
# 2) Thử nghiệm sensitivity:
#    - Thay đổi 'f' (TFN uncertainty) và xem độ ổn định ranking.
# 3) Validation VaR/CVaR:
#    - Dùng dữ liệu tổn thất lịch sử để backtest VaR/CVaR (hit rate).
# 4) Tối ưu:
#    - Thêm calibration cho cost_flags (các tiêu chí cost/benefit nên dựa trên domain).
#    - Nâng cấp forecast bằng SARIMA hoặc Prophet nếu cần.
# -------------------------------------------------------------------
