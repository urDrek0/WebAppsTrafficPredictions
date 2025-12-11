import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import joblib

# --- 1. CONFIG PAGE ---
st.set_page_config(layout="wide", page_title="Munich Traffic AI Dashboard")

# --- 2. FUNGSI LOAD SEMUA MODEL (PENGGANTI MAIN.PY) ---
@st.cache_resource
def load_all_models():
    models = {}
    
    def safe_load(path):
        try: return joblib.load(path)
        except: return None

    # 1. XGBoost
    models['xgboost_flow'] = safe_load('models/model_flow_xgb.pkl')
    models['xgboost_occ'] = safe_load('models/model_occ_xgb.pkl')

    # 2. SVR & Scalers
    models['svr_flow'] = safe_load('models/model_flow_svr.pkl')
    models['scaler_x_flow'] = safe_load('models/scaler_x_flow.pkl')
    models['scaler_y_flow'] = safe_load('models/scaler_y_flow.pkl')
    
    models['svr_occ'] = safe_load('models/model_occ_svr.pkl')
    models['scaler_x_occ'] = safe_load('models/scaler_x_occ.pkl')
    models['scaler_y_occ'] = safe_load('models/scaler_y_occ.pkl')

    # 3. Model Lainnya
    models['rf_flow'] = safe_load('models/model_flow_rf.pkl')
    models['rf_occ'] = safe_load('models/model_occ_rf.pkl')
    
    models['lgbm_flow'] = safe_load('models/model_flow_lgbm.pkl')
    models['lgbm_occ'] = safe_load('models/model_occ_lgbm.pkl')
    
    models['et_flow'] = safe_load('models/model_flow_et.pkl')
    models['et_occ'] = safe_load('models/model_occ_et.pkl')
    
    models['poly_flow'] = safe_load('models/model_flow_poly.pkl')
    models['poly_occ'] = safe_load('models/model_occ_poly.pkl')

    return models

# Load model ke memori aplikasi
loaded_models = load_all_models()

# --- 3. INIT FUZZY SYSTEM ---
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

# --- 4. UI SIDEBAR (UI ANDA TETAP) ---
st.sidebar.image("Logo.png", caption="Powered by Cip's <3") 
st.sidebar.header("Pengaturan Model AI")

st.title("🚦 Sistem Prediksi Kemacetan Munich")

models_used = st.sidebar.selectbox(
    'Pilih Model AI', 
    ['XGBoost', 'Random Forest', 'Polynomial Regression', 'LightGBM', 'Supported Vector Regression (SVR)', "Extra Trees Regressor"]
)

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
        interval = st.number_input("Detik (0-86400)", 0, 86400, 40000)
with blank:
    pass

# Load Data CSV
@st.cache_data
def get_data():
    try: return pd.read_csv("data/munich.csv")
    except: return None
df = get_data()

# --- 5. EKSEKUSI PREDIKSI (ALL-IN-ONE LOGIC) ---
if st.button("🚀 Prediksi Sekarang", type="primary"):
    
    # 1. Siapkan Input
    input_df = pd.DataFrame([[interval]], columns=['interval'])
    selected_code = model_map[models_used]
    p_flow, p_occ = 0.0, 0.0
    error_msg = None

    # 2. LOGIKA OTAK AI (Menggantikan requests.post)
    try:
        # A. SVR (Butuh Scaler)
        if selected_code == 'svr':
            if loaded_models['svr_flow'] is None: error_msg = "Model SVR belum dimuat!"
            else:
                # Flow Scaling
                sc_x_f = loaded_models['scaler_x_flow']
                sc_y_f = loaded_models['scaler_y_flow']
                p_flow = sc_y_f.inverse_transform(loaded_models['svr_flow'].predict(sc_x_f.transform(input_df)).reshape(-1, 1)).flatten()[0]
                
                # Occ Scaling
                sc_x_o = loaded_models['scaler_x_occ']
                sc_y_o = loaded_models['scaler_y_occ']
                p_occ = sc_y_o.inverse_transform(loaded_models['svr_occ'].predict(sc_x_o.transform(input_df)).reshape(-1, 1)).flatten()[0]

        # B. Random Forest
        elif selected_code == 'rf':
            if loaded_models['rf_flow'] is None: error_msg = "Model RF belum dimuat!"
            else:
                p_flow = loaded_models['rf_flow'].predict(input_df)[0]
                p_occ = loaded_models['rf_occ'].predict(input_df)[0]

        # C. LightGBM
        elif selected_code == 'lightgbm':
            if loaded_models['lgbm_flow'] is None: error_msg = "Model LGBM belum dimuat!"
            else:
                p_flow = loaded_models['lgbm_flow'].predict(input_df)[0]
                p_occ = loaded_models['lgbm_occ'].predict(input_df)[0]

        # D. Extra Trees
        elif selected_code == 'et':
            if loaded_models['et_flow'] is None: error_msg = "Model ET belum dimuat!"
            else:
                p_flow = loaded_models['et_flow'].predict(input_df)[0]
                p_occ = loaded_models['et_occ'].predict(input_df)[0]

        # E. Polynomial
        elif selected_code == 'polynomial':
            if loaded_models['poly_flow'] is None: error_msg = "Model Poly belum dimuat!"
            else:
                p_flow = loaded_models['poly_flow'].predict(input_df)[0]
                p_occ = loaded_models['poly_occ'].predict(input_df)[0]

        # F. Default (XGBoost)
        else:
            if loaded_models['xgboost_flow'] is None: error_msg = "Model XGBoost belum dimuat!"
            else:
                p_flow = loaded_models['xgboost_flow'].predict(input_df)[0]
                p_occ = loaded_models['xgboost_occ'].predict(input_df)[0]

        # 3. SETELAH PREDIKSI (UI ANDA)
        if error_msg:
            st.error(error_msg)
        else:
            # Bersihkan angka minus
            p_flow = max(0, float(p_flow))
            p_occ = max(0, float(p_occ))
            
            # Tentukan Status
            status_txt = "LANCAR"
            if p_occ > 0.18: status_txt = "MACET"
            elif p_occ > 0.10: status_txt = "PADAT MERAYAP"

            presentase_occ = p_occ * 100

            # --- A. TAMPILAN METRICS (UI ANDA) ---
            c1, c2, c3, c4, c5 = st.columns(5)
            
            if "MACET" in status_txt.upper(): s_color = "red"
            elif "PADAT" in status_txt.upper(): s_color = "orange"
            else: s_color = "green"
            
            c1.markdown(f"### **Status**: :{s_color}[{status_txt}]")
            c2.metric("**Prediksi Flow**", f"{p_flow:.1f}")
            c3.metric("**Prediksi Occupancy**", f"{presentase_occ:.2f}%")
            c4.metric("**Input Interval:**", f"{interval}", "sec")
            c5.metric("**Model AI Digunakan:**", f"{models_used}")
            
            st.divider()

            # --- B. GRAFIK PREDIKSI UTAMA (UI ANDA) ---
            st.subheader("📈 Posisi Prediksi pada Data Historis")
            if df is not None:
                flow_col, occ_col = st.columns(2)
                with flow_col:
                    st.markdown("Flow Historis dengan Prediksi Saat Ini")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.lineplot(data=df[::20], x='interval', y='flow', ax=ax, color='lightgrey', label='Data Historis')
                    ax.scatter([interval], [p_flow], color='red', s=100, zorder=2, label='Prediksi Saat Ini')
                    ax.axhline(350, color='orange', linestyle='--', alpha=0.5, label='Threshold Macet')
                    ax.set_ylabel("Flow")
                    ax.legend()
                    st.pyplot(fig)
                with occ_col:
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

            # --- C. FUZZY LOGIC EXPLAINER (UI ANDA) ---
            st.subheader("🧠 Analisis Fuzzy Logic")
            
            # Hitung Fuzzy Lokal
            sim.input['flow'] = min(p_flow, 1000)
            sim.input['occ'] = min(p_occ, 1.0)
            sim.compute()
            output_congestion = sim.output['congestion']

            col_grafik, col_tabel = st.columns([3, 2])
            
            with col_grafik:
                st.write("**Visualisasi Membership Function**")
                with st.expander("Buka Grafik Flow & Occ", expanded=True):
                    
                    try:
                        flow_ant.view(sim=sim)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except Exception as e:
                        st.warning(f"Grafik Flow gagal render: {e}")

                    try:
                        occ_ant.view(sim=sim)
                        st.pyplot(plt.gcf())
                        plt.clf()
                    except: pass

                with st.expander("Buka Grafik Keputusan", expanded=True):
                    try:
                        cong_con.view(sim=sim)
                        st.pyplot(plt.gcf())
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
                st.info(
                    f"""
                    **Logika Keputusan:**
                    Sistem menerima input **Flow: {p_flow:.1f}** dan **Occ: {p_occ:.2f}**, dengan **Level keanggotaan: {output_congestion:.2f}**.
                
                    Berdasarkan grafik di sebelah kiri, sistem menggabungkan aturan yang relevan dan menghitung titik tengah (centroid) area biru.
                    """
                )

    except Exception as e:
        st.error(f"Terjadi kesalahan pada sistem prediksi: {e}")
