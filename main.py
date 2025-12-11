from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Traffic Prediction API")

# --- GLOBAL VARIABLES ---
models = {}

# --- 1. LOAD SEMUA MODEL SAAT STARTUP ---
@app.on_event("startup")
def load_artifacts():
    print("⏳ Sedang memuat semua model...")
    try:
        # A. XGBOOST (Wajib)
        models['xgboost_flow'] = joblib.load('models/model_flow_xgb.pkl')
        models['xgboost_occ'] = joblib.load('models/model_occ_xgb.pkl')
        
        # B. RANDOM FOREST (Cek File)
        try:
            models['rf_flow'] = joblib.load('models/model_flow_rf.pkl')
            models['rf_occ'] = joblib.load('models/model_occ_rf.pkl')
            print("✅ Random Forest: OK")
        except: print("⚠️ Random Forest: File tidak ditemukan (SKIP)")

        # C. LIGHTGBM (Cek File)
        try:
            models['lgbm_flow'] = joblib.load('models/model_flow_lgbm.pkl')
            models['lgbm_occ'] = joblib.load('models/model_occ_lgbm.pkl')
            print("✅ LightGBM: OK")
        except: print("⚠️ LightGBM: File tidak ditemukan (SKIP)")

        # D. POLYNOMIAL (Cek File)
        try:
            models['poly_flow'] = joblib.load('models/model_flow_poly.pkl')
            models['poly_occ'] = joblib.load('models/model_occ_poly.pkl')
            print("✅ Polynomial: OK")
        except: print("⚠️ Polynomial: File tidak ditemukan (SKIP)")

        # E. SVR (Cek File & Scaler)
        try:
            models['svr_flow'] = joblib.load('models/model_flow_svr.pkl')
            models['scaler_x_flow'] = joblib.load('models/scaler_x_flow.pkl')
            models['scaler_y_flow'] = joblib.load('models/scaler_y_flow.pkl')
            
            models['svr_occ'] = joblib.load('models/model_occ_svr.pkl')
            models['scaler_x_occ'] = joblib.load('models/scaler_x_occ.pkl')
            models['scaler_y_occ'] = joblib.load('models/scaler_y_occ.pkl')
            print("✅ SVR: OK")
        except: print("⚠️ SVR: File/Scaler tidak lengkap (SKIP)")
        # F. EXTRA TREES (Baru)
        try:
            models['et_flow'] = joblib.load('models/model_flow_et.pkl')
            models['et_occ'] = joblib.load('models/model_occ_et.pkl')
            print("✅ Extra Trees: OK")
        except: print("⚠️ Extra Trees: File tidak ditemukan (SKIP)")

    except Exception as e:
        print(f"❌ Error Critical: {e}")

# --- INPUT SCHEMA ---
class TrafficInput(BaseModel):
    interval: float
    model_type: str = "xgboost" 

# --- STATUS HELPER ---
def determine_status(occ):
    if occ > 0.18: return "MACET PARAH"
    elif occ > 0.10: return "PADAT MERAYAP"
    else: return "LANCAR"

# --- ENDPOINT ---
@app.post("/predict")
def predict_traffic(data: TrafficInput):
    # Validasi
    if data.interval < 0 or data.interval > 86400:
        raise HTTPException(status_code=400, detail="Interval harus 0 - 86400.")

    input_df = pd.DataFrame([[data.interval]], columns=['interval'])
    pred_flow, pred_occ = 0.0, 0.0

    try:
        # === 1. LOGIKA SVR (SCALING) ===
        if data.model_type == "svr":
            if 'svr_flow' not in models: raise Exception("Model SVR belum dimuat/filenya hilang.")
            
            # Flow
            sc_x_f = models['scaler_x_flow']
            sc_y_f = models['scaler_y_flow']
            pred_flow = sc_y_f.inverse_transform(models['svr_flow'].predict(sc_x_f.transform(input_df)).reshape(-1, 1)).flatten()[0]

            # Occ
            sc_x_o = models['scaler_x_occ']
            sc_y_o = models['scaler_y_occ']
            pred_occ = sc_y_o.inverse_transform(models['svr_occ'].predict(sc_x_o.transform(input_df)).reshape(-1, 1)).flatten()[0]

        # === 2. LOGIKA LIGHTGBM ===
        elif data.model_type == "lgbm":
            if 'lgbm_flow' not in models: raise Exception("Model LightGBM belum dimuat/filenya hilang.")
            pred_flow = models['lgbm_flow'].predict(input_df)[0]
            pred_occ = models['lgbm_occ'].predict(input_df)[0]

        # === 3. LOGIKA RANDOM FOREST ===
        elif data.model_type == "rf":
            if 'rf_flow' not in models: raise Exception("Model Random Forest belum dimuat/filenya hilang.")
            pred_flow = models['rf_flow'].predict(input_df)[0]
            pred_occ = models['rf_occ'].predict(input_df)[0]

        # === 4. LOGIKA POLYNOMIAL ===
        elif data.model_type == "polynomial":
            if 'poly_flow' not in models: raise Exception("Model Polynomial belum dimuat/filenya hilang.")
            pred_flow = models['poly_flow'].predict(input_df)[0]
            pred_occ = models['poly_occ'].predict(input_df)[0]
        
        # === 5. LOGIKA EXTRA TREES ===
        elif data.model_type == "et":
            if 'et_flow' not in models: raise Exception("Model Extra Trees belum dimuat.")
            pred_flow = models['et_flow'].predict(input_df)[0]
            pred_occ = models['et_occ'].predict(input_df)[0]

        # === 6. DEFAULT (XGBOOST) ===
        else: 
            if 'xgboost_flow' not in models: raise Exception("Model XGBoost belum dimuat.")
            pred_flow = models['xgboost_flow'].predict(input_df)[0]
            pred_occ = models['xgboost_occ'].predict(input_df)[0]
        
        
        return {
            "model_used": data.model_type,
            "status_jalan": determine_status(max(0, pred_occ)),
            "prediksi_flow": max(0, float(pred_flow)),
            "prediksi_okupansi": max(0, float(pred_occ))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))