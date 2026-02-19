import streamlit as st
import gymnasium as gym
from src.env.gym_env import DrivingEnv
from stable_baselines3 import PPO
import time
import numpy as np

# Configuración de la página
st.set_page_config(page_title="TFM - IA Conducción Autónoma", layout="centered")

st.title("🏎️ Simulador de IA: Conducción Autónoma")
st.write("Esta aplicación muestra un modelo de Reinforcement Learning entrenado para navegar una pista procedimental.")

# --- BARRA LATERAL (CONFIGURACIÓN) ---
st.sidebar.header("Configuración")
velocidad_sim = st.sidebar.slider("Velocidad de simulación", 0.0, 0.1, 0.01, help="Menor es más rápido")
mostrar_sensores = st.sidebar.checkbox("Mostrar Sensores", value=True)

# --- CARGA DE ACTIVOS ---
@st.cache_resource
def load_assets():
    # Instanciamos con rgb_array para que funcione en la web
    env = DrivingEnv(render_mode="rgb_array")
    # Cargamos tu modelo final (asegúrate de que el nombre coincida)
    try:
        model = PPO.load("modelo_entrenado.zip")
        return env, model
    except:
        st.error("⚠️ No se encontró 'modelo_entrenado.zip'. ¡Asegúrate de subir tu modelo entrenado!")
        return env, None

env, model = load_assets()

# --- LÓGICA DE LA SIMULACIÓN ---
if model:
    if st.button('🏁 Iniciar Simulación'):
        obs, info = env.reset()
        
        # Este contenedor es donde se irá actualizando la imagen del coche
        placeholder = st.empty()
        
        progress_bar = st.progress(0)
        
        for step in range(1500):
            # La IA predice la acción basada en la observación
            action, _ = model.predict(obs, deterministic=True)
            
            # Aplicamos la acción al entorno
            obs, reward, terminated, truncated, info = env.step(action)
            
            # CAPTURAMOS EL RENDER (Esto devuelve el array que configuramos antes)
            frame = env.render(show_sensors=mostrar_sensores)
            
            if frame is not None:
                # Dibujamos el frame en la web
                placeholder.image(frame, channels="RGB", width="stretch")
            
            # Actualizamos barra de progreso
            progress_bar.progress(min(step / 1500, 1.0))
            
            if terminated or truncated:
                evento = info.get("event", "desconocido")
                if evento == "finish":
                    st.balloons()
                    st.success("¡META ALCANZADA! 🎉")
                else:
                    st.warning(f"Simulación terminada: {evento}")
                break
            
            # Pequeña pausa para que el ojo humano pueda seguir el movimiento
            time.sleep(velocidad_sim)

# --- SECCIÓN TÉCNICA (PIE DE PÁGINA) ---
st.divider()
st.subheader("Detalles del Proyecto")
st.info("""
**Tecnologías utilizadas:**
* **Motor:** Pygame (Physics Engine)
* **Entorno:** Gymnasium (Custom Environment)
* **Algoritmo:** PPO (Proximal Policy Optimization)
* **Detección:** Ray Casting & Teorema de la Curva de Jordan
""")