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
    env = DrivingEnv(render_mode="rgb_array")
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
        placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Variables para métricas
        start_time = time.time()
        recompensas = []
        max_vel_alcanzada = 0.0

        for step in range(1500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Guardar datos para el resumen final
            recompensas.append(reward)
            if env.car.velocity > max_vel_alcanzada:
                max_vel_alcanzada = env.car.velocity
            
            # --- MEJORA: Renderizado optimizado para la nube ---
            # Renderizamos 1 de cada 3 pasos para evitar saturar el servidor (KeyError)
            if step % 3 == 0:
                frame = env.render(show_sensors=mostrar_sensores)
                if frame is not None:
                    placeholder.image(frame, channels="RGB", width="stretch")
            
            # Sincronización de barra con los pasos reales del entorno
            progress_bar.progress(min(env.steps / 1500, 1.0))
            
            if terminated or truncated:
                # Al terminar, forzamos el dibujado del último frame (meta o choque)
                frame_final = env.render(show_sensors=mostrar_sensores)
                if frame_final is not None:
                    placeholder.image(frame_final, channels="RGB", width="stretch")

                evento = info.get("event", "desconocido")
                st.divider()
                
                # --- NOTIFICACIONES ---
                if evento == "finish":
                    st.toast('¡Objetivo completado!', icon='🏁')
                    st.success("✨ **RESULTADO: META ALCANZADA**")
                elif evento == "off_track":
                    st.toast('Colisión detectada', icon='💥')
                    st.error("💥 **RESULTADO: COLISIÓN**")
                elif evento == "timeout":
                    st.toast('Tiempo agotado', icon='⏳')
                    st.warning("⏳ **RESULTADO: TIEMPO AGOTADO**")

                # --- CUADRO DE MÉTRICAS SINCRONIZADO ---
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Pasos Reales", f"{env.steps}")
                col2.metric("Recompensa Media", f"{np.mean(recompensas):.2f}")
                col3.metric("Vel. Máxima", f"{max_vel_alcanzada:.1f} px/s")
                col4.metric("Tiempo Sim.", f"{time.time() - start_time:.1f}s")
                
                break
            
            # --- MEJORA: Sleep de seguridad ---
            # En la nube, menos de 0.03s suele romper la comunicación WebSocket
            time.sleep(max(velocidad_sim, 0.03))

# --- SECCIÓN TÉCNICA (PIE DE PÁGINA) ---
st.divider()
st.subheader("Detalles del Proyecto")
st.info("""
**Tecnologías utilizadas:**
* **Motor:** Pygame (Physics Engine)
* **Entorno:** Gymnasium (Custom Environment)
* **Algoritmo:** PPO (Proximal Policy Optimization)
* **Detección:** Ray Casting (7 sensores a 120°)
""")