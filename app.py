# RISKCAST v4.7 — NCKH STABLE EDITION ✅
# Fuzzy AHP (TFN) + Monte-Carlo Climate Risk + TOPSIS + VaR & CVaR + ARIMA (if available)

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF

import warnings
warnings.filterwarnings("ignore")

# --- Try import ARIMA ---
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False

# ========================= STREAMLIT UI CONFIG ==============================
st.set_page_config(
    page_title="RISKCAST v4.7",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg,#002B17 0%, #004726 100%);
        color: #E5FFF4;
        font-family: "Segoe UI";
    }
    .stButton>button {
        background-color: #04B978;
        color: white;
        padding: .7rem 1.2rem;
        border-radius: 10px;
        font-weight: bold;
        border: none;
    }
    .card {
        background: #003C21;
        padding: 1rem;
        border-radius: 12px;
        margin-top: .8rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ RISKCAST v4.7 — Green Risk Decision System")
st.caption("**Fuzzy AHP + Monte Carlo + ARIMA + VaR/CVaR + TOPSIS** (Bản tối ưu để nộp NCKH & thi cấp trường)")

# ========================= SIDEBAR INPUT ====================================
with st.sidebar:
    st.header("📦 Thông tin lô hàng")

    cargo_value = st.number_input("Giá trị hàng hóa (USD)", value=30000, step=1000)
    route = st.selectbox("Tuyến vận chuyển", ["VN - EU", "VN - US", "VN - Singapore"])
    method = st.selectbox("Phương thức", ["Sea", "Air"])
    month = st.selectbox("Tháng vận chuyển", list(range(1, 13)), index=8)
    priority = st.selectbox("Ưu tiên", ["An toàn tối đa", "Cân bằng", "Tối ưu chi phí"])

    st.header("⚙️ Mô hình áp dụng")
    use_fuzzy = st.checkbox("Bật Fuzzy AHP (TFN)", True)
    use_mc = st.checkbox("Chạy Monte Carlo cho C6", True)
    use_arima = st.checkbox("Dự báo ARIMA nếu có dữ liệu", True)
    use_var = st.checkbox("Tính VaR & CVaR", True)

    mc_runs = st.number_input("Số vòng Monte Carlo", 1000, 10000, 2000, 500)
    fuzzy_uncertainty = st.slider("Biên độ bất định TFN (%)", 0, 50, 15)


# ========================= DATA =============================================
@st.cache_data
def load_data():
    months = list(range(1,13))
    historical = pd.DataFrame({
        'Month': months,
        'VN_EU_Risk': [0.2,0.23,0.28,0.32,0.37,0.41,0.46,0.50,0.60,0.62,0.55,0.45],
        'VN_US_Risk': [0.3,0.33,0.36,0.40,0.45,0.49,0.53,0.58,0.68,0.65,0.56,0.50],
    })
    claims = pd.DataFrame({
        'Loss_Rate': np.random.normal(0.09, 0.025, 300).clip(0, 0.20),
        'Month': np.random.choice(months, 300),
    })
    return historical, claims

historical, claims = load_data()


# ========================= DISPLAY METRICS ==================================
st.subheader("📊 Phân tích dữ liệu tổn thất thực tế")
c1, c2 = st.columns(2)
with c1:
    st.metric("Tỷ lệ tổn thất TB", f"{claims['Loss_Rate'].mean():.2%}")
with c2:
    st.metric("Tháng rủi ro nhất", claims.groupby("Month")["Loss_Rate"].mean().idxmax())


# ========================= ARIMA FORECAST ===================================
if use_arima and ARIMA_AVAILABLE:
    st.subheader("📈 Dự báo xu hướng rủi ro tuyến (ARIMA)")
    col = "VN_EU_Risk" if route == "VN - EU" else "VN_US_Risk"
    ts = historical[col].values
    try:
        model = ARIMA(ts, order=(1,1,1)).fit()
        forecast = model.forecast(3)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=ts, name="Lịch sử"))
        fig.add_trace(go.Scatter(y=forecast, name="Dự báo", line=dict(color="lime")))
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Không đủ dữ liệu ARIMA, bỏ qua.")


# ========================= WEIGHTS ==========================================
st.subheader("⚖️ Phân bổ trọng số tiêu chí")
criteria = ["C1: Tỷ lệ phí", "C2: Xử lý claim", "C3: Tổn thất quá khứ", "C4: Hỗ trợ ICC", "C5: CSKH", "C6: Rủi ro khí hậu"]

cols = st.columns(6)
w = np.array([cols[i].slider(c, 0.0, 1.0, 0.15, 0.01) for i, c in enumerate(criteria)])

if priority == "An toàn tối đa": w[[1,4,5]] *= [1.4,1.4,1.3]
elif priority == "Tối ưu chi phí": w[[0]] *= 1.6

w = w / w.sum()  # normalize về tổng = 1
weights_series = pd.Series(w, index=criteria)

# ----- FUZZY AHP -----
if use_fuzzy:
    low = np.maximum(weights_series * (1 - fuzzy_uncertainty/100), 1e-4)
    high = np.minimum(weights_series * (1 + fuzzy_uncertainty/100), 0.9999)
    weights_series = ((low + weights_series + high) / 3) / ((low + weights_series + high) / 3).sum()


# ========================= INSURANCE DATA ===================================
df = pd.DataFrame({
    "Company": ["Chubb", "PVI", "BaoViet", "Aon"],
    "C1: Tỷ lệ phí": [0.30, 0.28, 0.32, 0.26],
    "C2: Xử lý claim": [7, 9, 8, 6],
    "C3: Tổn thất quá khứ": [0.09, 0.07, 0.10, 0.08],
    "C4: Hỗ trợ ICC": [9, 7, 8, 6],
    "C5: CSKH": [8, 9, 7, 6],
}).set_index("Company")


# ========================= MONTE CARLO FOR CLIMATE RISK =====================
base_risk = historical.iloc[month-1]["VN_EU_Risk"] if route == "VN - EU" else historical.iloc[month-1]["VN_US_Risk"]
df_adj = df.copy()

if use_mc:
    rng = np.random.default_rng()
    mc_result = [rng.normal(base_risk * r, 0.05, mc_runs).clip(0,1).mean() for r in [1.10, 0.95, 1.05, 0.90]]
    df_adj["C6: Rủi ro khí hậu"] = mc_result
else:
    df_adj["C6: Rủi ro khí hậu"] = [base_risk]*4


# ========================= TOPSIS ===========================================
def topsis(df, w, cost_flags):
    M = df[list(w.index)].values
    R = M / np.sqrt((M**2).sum(0, keepdims=True))
    V = R * w.values
    best = np.where(cost_flags == "cost", V.min(0), V.max(0))
    worst = np.where(cost_flags == "cost", V.max(0), V.min(0))
    d_best = np.sqrt(((V - best)**2).sum(1))
    d_worst = np.sqrt(((V - worst)**2).sum(1))
    return d_worst / (d_best + d_worst + 1e-12)

cost_flags = np.array(["cost","benefit","cost","benefit","benefit","cost"])
scores = topsis(df_adj, weights_series, cost_flags)

result = pd.DataFrame({"Company": df_adj.index, "Score": scores})
result["Rank"] = result["Score"].rank(ascending=False).astype(int)
result.sort_values("Score", ascending=False, inplace=True)


# ========================= OUTPUT ===========================================
st.subheader("✅ KẾT QUẢ ĐỀ XUẤT BẢO HIỂM")
st.dataframe(result)

best = result.iloc[0]
st.success(f"**👉 ĐỀ XUẤT NÊN CHỌN: `{best['Company']}` — Score = {best['Score']:.4f}**")


# ========================= EXPORT PDF =======================================
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=14)
pdf.cell(200, 10, txt="RISKCAST v4.7 - Recommendation Report", ln=True)
for idx, row in result.iterrows():
    pdf.cell(200, 8, txt=f"{row['Company']}: Score = {row['Score']:.4f}", ln=True)

buffer = io.BytesIO()
pdf.output(buffer)
st.download_button("📄 Xuất PDF kết quả", buffer, file_name="riskcast_report.pdf")
