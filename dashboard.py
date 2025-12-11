import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- 1. CONFIG PAGE ---
st.set_page_config(layout="wide", page_title="Munich Traffic AI Dashboard")
st.sidebar.image("Logo.png", caption="Powered by Cip's <3")  # Tambahkan logo jika ada
st.sidebar.header("Pengaturan Model AI")

# --- 2. INIT FUZZY SYSTEM (Cache) ---
@st.cache_resource
def init_fuzzy_system():
    # Universe
    flow = np.arange(0, 1001, 1)
    occ = np.arange(0, 1.01, 0.01)
    congestion = np.arange(0, 101, 1)

    # Membership Functions
    flow_low = fuzz.trimf(flow, [0, 0, 300])
    flow_medium = fuzz.trimf(flow, [200, 450, 700])
    flow_high = fuzz.trimf(flow, [600, 1000, 1000])

    occ_low = fuzz.trimf(occ, [0, 0, 0.3])
    occ_medium = fuzz.trimf(occ, [0.2, 0.5, 0.8])
    occ_high = fuzz.trimf(occ, [0.7, 1, 1])

    congestion_low = fuzz.trimf(congestion, [0, 0, 40])
    congestion_medium = fuzz.trimf(congestion, [30, 60, 90])
    congestion_high = fuzz.trimf(congestion, [80, 100, 100])

    # Control System Objects
    flow_ant = ctrl.Antecedent(flow, 'flow')
    occ_ant = ctrl.Antecedent(occ, 'occ')
    cong_con = ctrl.Consequent(congestion, 'congestion')

    flow_ant['low'] = flow_low; flow_ant['medium'] = flow_medium; flow_ant['high'] = flow_high
    occ_ant['low'] = occ_low; occ_ant['medium'] = occ_medium; occ_ant['high'] = occ_high
    cong_con['Lancar'] = congestion_low; cong_con['Padat Merayap'] = congestion_medium; cong_con['Macet'] = congestion_high

    # Rules
    rules = [
        ctrl.Rule(flow_ant['high'] & occ_ant['low'], cong_con['Lancar']),
        ctrl.Rule(flow_ant['medium'] & occ_ant['low'], cong_con['Lancar']),
        ctrl.Rule(flow_ant['low'] & occ_ant['low'], cong_con['Lancar']),
        ctrl.Rule(flow_ant['high'] & occ_ant['medium'], cong_con['Padat Merayap']),
        ctrl.Rule(flow_ant['medium'] & occ_ant['medium'], cong_con['Padat Merayap']),
        ctrl.Rule(flow_ant['low'] & occ_ant['medium'], cong_con['Macet']),
        ctrl.Rule(flow_ant['high'] & occ_ant['high'], cong_con['Padat Merayap']),
        ctrl.Rule(flow_ant['medium'] & occ_ant['high'], cong_con['Padat Merayap']),
        ctrl.Rule(flow_ant['low'] & occ_ant['high'], cong_con['Macet']),
    ]

    sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
    return sim, flow_ant, occ_ant, cong_con

sim, flow_ant, occ_ant, cong_con = init_fuzzy_system()

# --- 3. UI SIDEBAR ---
st.title("🚦 Sistem Prediksi Kemacetan Munich")

models_used = st.sidebar.selectbox(
    'Pilih Model AI', 
    ['XGBoost', 'Random Forest', 'Polynomial Regression', 'LightGBM', 'Supported Vector Regression (SVR)', "Extra Trees Regressor"]
)

# Mapping nama model ke kode API
model_map = {
    'XGBoost': 'xgboost',
    'Random Forest': 'rf',
    'Polynomial Regression': 'polynomial',
    'LightGBM': 'lightgbm',
    "Supported Vector Regression (SVR)": 'svr',
    "Extra Trees Regressor": 'et'
}
st.markdown("---")
col_side1, col_side2, blank = st.columns(3)
with col_side1:
    metode_input = st.radio("Metode Input:", ['Input Detik', 'Jam'])

with col_side2:
    if metode_input == 'Jam':
        jam, menit, detik = st.columns(3)
        with jam:
            hour = st.number_input("Jam (0-23)", 0, 23, 11)
        with menit:
            minute = st.number_input("Menit (0-59)", 0, 59, 59)
        with detik:
            second = st.number_input("Detik (0-59)", 0, 59, 59)
        interval = int(hour * 3600 + minute * 60 + second)
    else:
        interval = st.number_input("Detik (0-86400)", 0, 86400, 86400)
with blank:
    pass
# Load Data CSV
@st.cache_data
def get_data():
    try: return pd.read_csv("data/munich.csv")
    except: return None
df = get_data()

# --- 5. EXECUTION ---
if st.button("🚀 Prediksi Sekarang", type="primary"):
    API_URL = "http://127.0.0.1:8000/predict"
    
    payload = {
        "interval": interval, 
        "model_type": model_map[models_used]
    }
    
    try:
        with st.spinner(f'Menghitung prediksi menggunakan {models_used}...'):
            res = requests.post(API_URL, json=payload)
        
        if res.status_code == 200:
            data = res.json()
            p_flow = data['prediksi_flow']
            p_occ = data['prediksi_okupansi']
            presentase_occ = p_occ * 100
            status_txt = data['status_jalan']

            # --- A. TAMPILAN METRICS ---
            c1, c2, c3, c4, c5 = st.columns(5)
            # Warna status
            if "MACET" in status_txt.upper(): s_color = "red"
            elif "PADAT" in status_txt.upper(): s_color = "orange"
            else: s_color = "green"
            
            c1.markdown(f"### **Status**: :{s_color}[{status_txt}]")
            c2.metric("**Prediksi Flow**", f"{p_flow:.1f}")
            c3.metric("**Prediksi Occupancy**", f"{presentase_occ:.2f}%")
            c4.metric("**Input Interval:**", f"{interval}", "sec")
            c5.metric("**Model AI Digunakan:**", f"{models_used}")
            
            st.divider()

            # --- B. GRAFIK PREDIKSI UTAMA ---
            st.subheader("📈 Posisi Prediksi pada Data Historis")
            if df is not None:
                flow, occ = st.columns(2)
                with flow:
                    st.markdown("Flow Historis dengan Prediksi Saat Ini")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(data=df[::20], x='interval', y='flow', ax=ax, color='lightgrey', label='Data Historis')
                    ax.scatter([interval], [p_flow], color='red', s=100, zorder=2, label='Prediksi Saat Ini')
                    ax.axhline(350, color='orange', linestyle='--', alpha=0.5, label='Threshold Macet')
                    ax.set_ylabel("Flow")
                    ax.legend()
                    st.pyplot(fig)
                with occ:
                    st.markdown("Occupancy Historis dengan Prediksi Saat Ini")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(data=df[::20], x='interval', y='occ', ax=ax, color='lightgrey', label='Data Historis')
                    ax.scatter([interval], [p_occ], color='red', s=100, zorder=2, label='Prediksi Saat Ini')
                    ax.axhline(0.18, color='orange', linestyle='--', alpha=0.5, label='Threshold Macet')
                    ax.set_ylabel("Occupancy")
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.warning("File munich.csv tidak ditemukan.")

            st.divider()

            # C. Fuzzy Logic Explainer (LAYOUT TETAP)
            st.subheader("🧠 Analisis Fuzzy Logic")
            
            # Hitung Fuzzy Lokal
            sim.input['flow'] = p_flow
            sim.input['occ'] = p_occ
            sim.compute()
            output_congestion = sim.output['congestion']

            # Layout 2 Kolom: Kiri Grafik, Kanan Tabel
            col_grafik, col_tabel = st.columns([3, 2])
            
            with col_grafik:
                st.write("**Visualisasi Membership Function**")
                # Grafik Flow
                with st.expander("Buka Grafik Flow & Occ", expanded=True):
                    
                    # Grafik Flow
                    try:
                        flow_ant.view(sim=sim)   # 1. Suruh skfuzzy gambar
                        fig_flow = plt.gcf()     # 2. Tangkap gambar yang aktif (Get Current Figure)
                        st.pyplot(fig_flow)      # 3. Tampilkan di Streamlit
                        plt.clf()                # 4. Bersihkan memori agar grafik berikutnya tidak numpuk
                    except Exception as e:
                        st.warning(f"Grafik Flow gagal render: {e}")

                    # Grafik Occupancy
                    try:
                        occ_ant.view(sim=sim)
                        fig_occ = plt.gcf()      # Tangkap gambar
                        st.pyplot(fig_occ)
                        plt.clf()
                    except: pass

                # 2. Grafik Output (Keputusan)
                with st.expander("Buka Grafik Keputusan", expanded=True):
                    try:
                        cong_con.view(sim=sim)
                        fig_out = plt.gcf()      # Tangkap gambar
                        st.pyplot(fig_out)
                        plt.clf()
                    except Exception as e:
                        st.warning(f"Grafik Output kosong: {e}")

            with col_tabel:
                st.write("**Rule Base & Hasil**")
                st.metric("Skor Kemacetan (0-100)", f"{output_congestion:.2f}")
                
                rule_data = {
                    "Rule": ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"],
                    "Flow": ["High", "Med", "Low", "High", "Med", "Low", "High", "Med", "Low"],
                    "Occ": ["Low", "Low", "Low", "Med", "Med", "Med", "High", "High", "High"],
                    "Output": ["Lancar", "Lancar", "Lancar", "Padat", "Padat", "Macet", "Padat", "Padat", "Macet"]
                }
                st.dataframe(pd.DataFrame(rule_data), hide_index=True)
                 # Penjelasan Naratif
                st.info(
                    f"""
                    **Logika Keputusan:**
                    Sistem menerima input **Flow: {p_flow:.1f}** dan **Occ: {p_occ:.2f}**, dengan **Level keanggotaan: {output_congestion:.2f}**.
                
                    Berdasarkan grafik di sebelah kiri, sistem menggabungkan aturan yang relevan dan menghitung titik tengah (centroid) area biru.
                    """
                )
        else:
            st.error(f"Error Backend: {res.text}")

    except Exception as e:
        st.error(f"Koneksi Gagal: {e}. Pastikan 'main.py' berjalan.")