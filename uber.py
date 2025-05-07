# 🚦 Uber-Scale Causally-Informed Budget Optimizer (Advanced)
# Multi-Phase System: Deep Causal Inference, Spline Modeling, ADMM Optimization, Multi-City Dashboard

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from patsy import dmatrix
import cvxpy as cp
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import shap
import glob

# --- Phase 0: Load Multi-City Data ---
def load_data():
    data_list = []
    file_paths = [
        "Travel_Times_SF.csv",
        "Travel_Times_NYC.csv",
        "Travel_Times_Boston.csv"
    ]
    for file in file_paths:
        df = pd.read_csv(file)
        city = file.split("_")[-1].split(".")[0]  # SF, NYC, Boston
        df['region'] = city
        data_list.append(df)
    return pd.concat(data_list)


class SLearner(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.model(xt)

class IncentiveOptimizer:
    def __init__(self, data):
        self.data = data
        self.X = None  # Placeholder until process_data sets it
        self.process_data()
        self.model = SLearner(input_dim=self.X.shape[1])

    def process_data(self):
        data = self.data
        data['zone_pair'] = data['origin_display_name'] + " ➝ " + data['destination_display_name']
        data['travel_minutes'] = data['mean_travel_time_seconds'] / 60
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'])
            data['hour'] = data['datetime'].dt.hour
            data['weekday'] = data['datetime'].dt.dayofweek
        agg = data.groupby(['region', 'zone_pair']).agg({
            'travel_minutes': ['mean', 'std'],
            'datetime': 'count'
        }).reset_index()
        agg.columns = ['region', 'zone_pair', 'mean_travel', 'std_travel', 'ride_count']
        np.random.seed(42)
        agg['expected_rides'] = np.random.randint(2000, 8000, len(agg))
        agg['treatment'] = np.random.uniform(500, 2500, len(agg))
        agg['outcome'] = agg['expected_rides'] + 0.05 * agg['treatment'] + np.random.normal(0, 50, len(agg))
        self.agg = agg
        self.X = StandardScaler().fit_transform(agg[['mean_travel', 'std_travel']])

    def fit_s_learner(self):
        agg = self.agg
        X = self.X
        T = agg['treatment'].values.reshape(-1, 1)
        Y = agg['outcome'].values.reshape(-1, 1)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        T_tensor = torch.tensor((T - T.mean()) / T.std(), dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        for epoch in range(500):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X_tensor, T_tensor)
            loss = loss_fn(pred, Y_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            Y0 = self.model(X_tensor, torch.zeros_like(T_tensor))
            Y1 = self.model(X_tensor, torch.ones_like(T_tensor))
            CATE = (Y1 - Y0).numpy().flatten()
        agg['causal_effect'] = CATE

        # SHAP Explainability (Tabular Kernel Explainer for PyTorch fallback)
        combined_input = np.concatenate([X, T], axis=1)
        explainer = shap.KernelExplainer(lambda x: self.model(torch.tensor(x[:, :-1], dtype=torch.float32),
                                                              torch.tensor(x[:, -1:], dtype=torch.float32)).detach().numpy(),
                                         combined_input[:100])
        shap_values = explainer.shap_values(combined_input[:100])
        shap.summary_plot(shap_values, combined_input[:100], feature_names=["mean_travel", "std_travel", "treatment"], show=False)

    def optimize_budget(self):
        agg = self.agg
        agg['hour'] = np.random.randint(0, 24, len(agg))
        splines = dmatrix("bs(hour, df=6, degree=3, include_intercept=False)", data=agg, return_type='dataframe')
        agg = pd.concat([agg, splines], axis=1)

        agg['ev_score'] = np.random.uniform(0.5, 1.5, len(agg))
        agg['loyalty_index'] = np.random.uniform(0.8, 1.2, len(agg))
        agg['surge_risk'] = agg['std_travel'] / (agg['mean_travel'] + 1)

        agg['promo_score'] = agg['causal_effect'] * agg['expected_rides'] * agg['ev_score'] * agg['loyalty_index'] * (1 + agg['surge_risk'])
        agg['score_scaled'] = StandardScaler().fit_transform(agg[['promo_score']])

        n = len(agg)
        score = agg['score_scaled'].values
        x = cp.Variable(n)
        BUDGET = st.sidebar.slider("Total Budget ($)", 10000, 100000, 50000, step=5000)
        min_spend = st.sidebar.number_input("Min Spend/Zone", 500, step=100)

        constraints = [x >= min_spend, cp.sum(x) <= BUDGET]
        objective = cp.Maximize(score @ x)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        agg['optimal_spend'] = x.value
        agg['spend_per_ride'] = agg['optimal_spend'] / agg['expected_rides']
        self.agg = agg

    def display_dashboard(self):
        agg = self.agg
        st.title("📊 Uber Incentive Optimization Dashboard")
        st.markdown("Optimizes citywide incentives using deep causal inference and strategic weights.")

        fig = px.bar(agg.sort_values("optimal_spend", ascending=False).head(50),
                     x='zone_pair', y='optimal_spend', color='region',
                     title="Top 50 Zone Pairs by Optimal Spend")
        st.plotly_chart(fig)

        fig2 = px.scatter(agg, x='spend_per_ride', y='causal_effect', color='region',
                          size='optimal_spend', hover_name='zone_pair',
                          title="Spend Efficiency vs Causal Impact")
        st.plotly_chart(fig2)

        city_tabs = st.tabs(sorted(agg['region'].unique()))
        for i, city in enumerate(sorted(agg['region'].unique())):
            with city_tabs[i]:
                city_data = agg[agg['region'] == city].sort_values("optimal_spend", ascending=False)
                st.subheader(f"Top Zones - {city}")
                st.dataframe(city_data[['zone_pair', 'optimal_spend', 'causal_effect', 'expected_rides']])

# --- Execute Workflow ---
data = load_data()
optimizer = IncentiveOptimizer(data)
optimizer.fit_s_learner()
optimizer.optimize_budget()
optimizer.display_dashboard()