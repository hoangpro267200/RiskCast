# risk_engine.py — Tách thuật toán: TOPSIS, Monte Carlo, VaR, Confidence
import numpy as np
import pandas as pd

class RiskEngine:
    def __init__(self, cargo_value, route, month, good_type, priority, weights, use_fuzzy=True, use_mc=True, mc_runs=2000):
        self.cargo_value = cargo_value
        self.route = route
        self.month = month
        self.good_type = good_type
        self.priority = priority
        self.weights = np.array(weights)
        self.use_fuzzy = use_fuzzy
        self.use_mc = use_mc
        self.mc_runs = mc_runs
        self._init_data()

    def _init_data(self):
        self.df = pd.DataFrame({
            "Company": ["Chubb", "PVI", "InternationalIns", "BaoViet", "Aon"],
            "C1: Tỷ lệ phí": [0.30, 0.28, 0.26, 0.32, 0.24],
            "C2: Thời gian xử lý": [6, 5, 8, 7, 4],
            "C3: Tỷ lệ tổn thất": [0.08, 0.06, 0.09, 0.10, 0.07],
            "C4: Hỗ trợ ICC": [9, 8, 6, 9, 7],
            "C5: Chăm sóc KH": [9, 8, 5, 7, 6],
        }).set_index("Company")

        self.sensitivity = {"Chubb":0.95, "PVI":1.10, "InternationalIns":1.20, "BaoViet":1.05, "Aon":0.90}
        self.base_risk = 0.65 if (self.route, self.month) == ("VN - EU", 9) else 0.4

        # Điều chỉnh theo input
        df = self.df.copy().astype(float)
        if self.cargo_value > 50000: df["C1: Tỷ lệ phí"] *= 1.2
        if self.route in ["VN - US", "VN - EU"]: df["C2: Thời gian xử lý"] *= 1.3
        if self.good_type in ["Hàng nguy hiểm", "Điện tử"]: df["C3: Tỷ lệ tổn thất"] *= 1.5

        # Monte Carlo
        mc_mean = np.array([self.base_risk * self.sensitivity.get(c, 1) for c in df.index])
        mc_std = np.zeros(len(df))
        if self.use_mc:
            rng = np.random.default_rng(42)
            for i in range(len(df)):
                mu, sigma = mc_mean[i], max(0.03, mc_mean[i]*0.12)
                sim = np.clip(rng.normal(mu, sigma, self.mc_runs), 0, 1)
                mc_mean[i], mc_std[i] = sim.mean(), sim.std()
        df["C6: Rủi ro khí hậu"] = mc_mean
        self.df_adj = df
        self.mc_std = mc_std

    def _fuzzy_weights(self):
        if not self.use_fuzzy: return self.weights
        f = 0.15
        low = np.maximum(self.weights * (1 - f), 1e-4)
        high = np.minimum(self.weights * (1 + f), 0.9999)
        return ((low + self.weights + high) / 3) / ((low + self.weights + high) / 3).sum()

    def run(self):
        w = self._fuzzy_weights()
        M = self.df_adj.values
        R = M / np.sqrt((M**2).sum(0, keepdims=True))
        V = R * w
        is_cost = np.isin(self.df_adj.columns, ["C1: Tỷ lệ phí", "C6: Rủi ro khí hậu"])
        best = np.where(is_cost, V.min(0), V.max(0))
        worst = np.where(is_cost, V.max(0), V.min(0))
        d_best = np.sqrt(((V - best)**2).sum(1))
        d_worst = np.sqrt(((V - worst)**2).sum(1))
        score = d_worst / (d_best + d_worst + 1e-12)

        result = pd.DataFrame({"company": self.df_adj.index, "score": score})
        result["score_pct"] = (result["score"] * 100).round(1)
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
        result["rank"] = result.index + 1
        result["ICC"] = result["score"].apply(lambda x: "A" if x >= 0.75 else "B" if x >= 0.5 else "C")
        result["Risk"] = result["score"].apply(lambda x: "THẤP" if x >= 0.75 else "TRUNG BÌNH" if x >= 0.5 else "CAO")

        # Confidence
        cv_c6 = np.where(self.df_adj["C6: Rủi ro khí hậu"] == 0, 0, self.mc_std / self.df_adj["C6: Rủi ro khí hậu"])
        conf_c6 = 1 / (1 + cv_c6)
        conf_c6_arr = np.array(conf_c6)
        ptp = conf_c6_arr.max() - conf_c6_arr.min() if len(conf_c6_arr) > 1 else 0
        conf_c6_scaled = np.where(ptp > 0, 0.3 + 0.7 * (conf_c6_arr - conf_c6_arr.min()) / ptp, 0.65)

        crit_cv = self.df_adj.std(axis=1) / (self.df_adj.mean(axis=1) + 1e-9)
        conf_crit = 1 / (1 + crit_cv)
        conf_crit_arr = np.array(conf_crit)
        ptp_crit = conf_crit_arr.max() - conf_crit_arr.min() if len(conf_crit_arr) > 1 else 0
        conf_crit_scaled = np.where(ptp_crit > 0, 0.3 + 0.7 * (conf_crit_arr - conf_crit_arr.min()) / ptp_crit, 0.65)

        result["confidence"] = np.sqrt(conf_c6_scaled * conf_crit_scaled).round(3)
        return result
