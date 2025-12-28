import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Drilling Optimization", layout="wide")
st.title("🛢️ AI & ML Integrated Drilling Optimization Application")

st.markdown("""
This application integrates petroleum engineering models with machine learning
to support real-time drilling optimization and problem detection.
""")

@st.cache_data
def load_data(file=None):
    if file is None:
        return pd.read_csv("sample_drilling_full.csv", parse_dates=["timestamp"])
    return pd.read_csv(file, parse_dates=["timestamp"])

uploaded = st.file_uploader("Upload drilling CSV", type=["csv"])
df = load_data(uploaded)

def compute_mse(WOB, Torque, RPM, ROP, bit_diam=8.5):
    WOB_lbf = WOB * 1000
    Torque_lbf_ft = Torque * 1000
    v = max(ROP/60, 1e-6)
    A = math.pi * bit_diam**2 / 4
    return (WOB_lbf/A) + (120*Torque_lbf_ft*RPM)/(A*v)

def warren_rop(WOB, RPM):
    return 0.05*(WOB**0.5)*(RPM**0.7)

def burgoyne_young_rop(WOB, RPM):
    return 0.1*(WOB**0.6)*(RPM**0.4)

df["MSE"] = df.apply(lambda r: compute_mse(r["WOB_klbf"], r["Torque_klbf_ft"], r["RPM"], r["ROP_ftph"]), axis=1)
df["BitWear"] = df["MSE"].rolling(30, min_periods=1).mean()

features = ["WOB_klbf","RPM","Torque_klbf_ft","ROP_ftph","MSE","BitWear"]
df["Problem"] = ((df["BitWear"] > df["BitWear"].quantile(0.97)) | (df["ROP_ftph"] < df["ROP_ftph"].quantile(0.03))).astype(int)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(df[features], df["Problem"])
df["Prediction"] = model.predict(df[features])

st.subheader("📊 Drilling Performance Dashboard")
st.line_chart(df.set_index("timestamp")[["MSE","BitWear"]])

st.subheader("⚙️ What-if Analysis")
with st.form("input_form"):
    WOB = st.number_input("WOB (klbf)", value=20.0)
    RPM = st.number_input("RPM", value=120.0)
    Torque = st.number_input("Torque (klbf-ft)", value=10.0)
    ROP = st.number_input("ROP (ft/hr)", value=25.0)
    submit = st.form_submit_button("Analyze")

if submit:
    st.metric("MSE", round(compute_mse(WOB,Torque,RPM,ROP),2))
    st.metric("B&Y ROP", round(burgoyne_young_rop(WOB,RPM),2))
    st.metric("Warren ROP", round(warren_rop(WOB,RPM),2))

st.download_button("Download Results", df.to_csv(index=False), "drilling_results.csv")
