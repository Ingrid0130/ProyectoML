# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time

# -------------------------------------------------
# Configuración general
# -------------------------------------------------
st.set_page_config(
    page_title="Predicción de Tiempo de Recorrido",
    layout="centered"
)

st.title("Predicción de Tiempo Total de Recorrido")
st.markdown(
    "Ingrese los datos del recorrido para estimar el **tiempo total** usando Machine Learning."
)

st.divider()

# -------------------------------------------------
# Cargar modelo
# -------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_regresion_final.pkl")

model = cargar_modelo()

# -------------------------------------------------
# Sidebar - Inputs
# -------------------------------------------------
st.sidebar.header("📥 Datos de entrada")

vehicle_id = st.sidebar.text_input("ID del vehículo", value="1")
lap = st.sidebar.number_input("Número de vuelta (lap)", min_value=1, value=1)
average_speed = st.sidebar.number_input(
    "Velocidad promedio (km/h)", min_value=0.1, value=40.0
)

fecha = st.sidebar.date_input("📅 Fecha del recorrido", value=datetime.today())
hora = st.sidebar.time_input("⏰ Hora del recorrido", value=time(8, 0))

st.divider()

# -------------------------------------------------
# Botón de predicción
# -------------------------------------------------
if st.button("🔮 Predecir tiempo", use_container_width=True):

    # -------------------------------------------------
    # DataFrame base
    # -------------------------------------------------
    X = pd.DataFrame([{
        "vehicle_id_id": str(vehicle_id),
        "lap": lap,
        "average_speed": average_speed,
        "date": pd.to_datetime(fecha),
        "time": hora
    }])

    # -------------------------------------------------
    # Feature engineering (IGUAL AL TRAIN)
    # -------------------------------------------------
    X['vehicle_id_id'] = X['vehicle_id_id'].astype(str)

    # Fecha
    X['month'] = X['date'].dt.month
    X['day'] = X['date'].dt.day
    X.drop(columns=['date'], inplace=True)

    # Hora
    X['hour'] = hora.hour
    X['minute'] = hora.minute
    X.drop(columns=['time'], inplace=True)

    # -------------------------------------------------
    # Features logarítmicas (CLAVE)
    # -------------------------------------------------
    log_cols = ['lap', 'day', 'month', 'average_speed', 'hour']

    for col in log_cols:
        X[col + '_log'] = np.log1p(X[col])

    # -------------------------------------------------
    # Columnas esperadas por el modelo
    # -------------------------------------------------
    expected_cols = [
        'lap', 'day', 'month', 'average_speed', 'hour', 'minute',
        'lap_log', 'day_log', 'month_log', 'average_speed_log', 'hour_log',
        'vehicle_id_id'
    ]

    for c in expected_cols:
        if c not in X.columns:
            X[c] = 0

    X = X[expected_cols]

    # -------------------------------------------------
    # Predicción
    # -------------------------------------------------
    try:
        pred = model.predict(X)[0]

        st.success("Predicción realizada correctamente")

        col1, col2 = st.columns(2)
        col1.metric("⏱️ Tiempo estimado", f"{pred:.2f}")
        col2.metric("🚗 Velocidad promedio", f"{average_speed:.1f} km/h")

        st.caption("El valor corresponde al tiempo total estimado del recorrido.")

    except Exception as e:
        st.error(f"❌ Error al predecir: {e}")
