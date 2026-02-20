# 1. Descripción general del proyecto

Este repositorio implementa un sistema de **conducción autónoma 2D** donde un agente aprende a conducir en una pista generada procedimentalmente usando **Aprendizaje por Refuerzo (Reinforcement Learning)**.

El objetivo del sistema es que el vehículo:
- avance por la carretera sin salirse,
- evite colisiones,
- y alcance la meta en un número limitado de pasos.

El problema que resuelve es la toma de decisiones de control continuo en un entorno dinámico: el agente debe elegir, en cada paso, valores de **aceleración** y **giro** para mantener estabilidad, progresar y terminar el circuito.

El enfoque se basa en un entorno compatible con **Gymnasium** (`DrivingEnv`) y en el algoritmo **PPO (Proximal Policy Optimization)** de Stable-Baselines3. En este proyecto, PPO optimiza una política neuronal (`MlpPolicy`) sobre un espacio de acciones continuo de 2 dimensiones (`throttle`, `steering`) y observaciones de 8 dimensiones (7 rayos de sensores + velocidad normalizada).

De forma resumida, el proceso de aprendizaje implementado en el código es:
1. Se genera una pista procedimental nueva al reiniciar episodio.
2. El agente observa distancias de ray casting y velocidad.
3. PPO propone una acción continua.
4. El entorno simula cinemática, detecta colisiones/meta/timeout y calcula recompensa.
5. PPO actualiza la política con los datos acumulados hasta mejorar el comportamiento.

# 2. Stack tecnológico utilizado

## Lenguaje principal
- **Python**.

## Librerías principales detectadas
- **Gymnasium (`gymnasium==1.2.3`)**: interfaz del entorno RL, espacios de acción/observación y ciclo `reset/step`.
- **Pygame (`pygame==2.6.1`)**: renderizado 2D, simulación visual, utilidades de fuente/superficie y soporte de ejecución interactiva.
- **Stable-Baselines3 (`stable_baselines3==2.7.1`)**: implementación de PPO usada para entrenamiento y evaluación.
- **PyTorch (`torch==2.10.0`)**: backend de cómputo para SB3.
- **Streamlit (`streamlit==1.54.0`)**: interfaz de evaluación visual del modelo entrenado.
- **NumPy (`numpy==2.2.6`)**: operaciones numéricas de observaciones y normalización.

## Dependencias adicionales declaradas
- `Shimmy==2.0.0`
- `pillow==12.1.0`
- `opencv-python-headless==4.13.0.92`

## Paquetes de sistema declarados
(archivo `packages.txt`, útiles en despliegues cloud para Pygame):
- `libsdl2-dev`
- `libsdl2-image-dev`
- `libsdl2-mixer-dev`
- `libsdl2-ttf-dev`
- `libfreetype6-dev`
- `libportmidi-dev`
- `libjpeg-dev`
- `xvfb`

# 3. Instalación y ejecución

## Requisitos previos
- Python 3.x (no se fija versión explícita en el repositorio).
- `pip`.
- Dependencias de sistema para SDL/Pygame (si aplica a tu entorno).

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

Si ejecutas en entorno Linux minimalista (por ejemplo, despliegue web), instala también paquetes del sistema equivalentes a `packages.txt`.

## Configuración necesaria
No hay archivo `.env` ni variables obligatorias definidas en el proyecto.

Solo debes considerar:
- Para evaluación con scripts y Streamlit, el repositorio incluye un modelo en la raíz: `modelo_entrenado.zip`.
- El script de entrenamiento guarda nuevos modelos en `models/` y logs en `logs/`.

## Cómo ejecutar el entorno (modo interactivo manual)

```bash
python -m src.scripts.run_interactive
```

Controles implementados:
- `W/S`: acelerar/frenar (marcha atrás).
- `A/D`: girar izquierda/derecha.
- `R`: generar nueva pista/episodio.
- `ESC`: salir.

## Cómo lanzar el entrenamiento PPO

```bash
python -m src.scripts.train_ppo
```

Comportamiento real del script:
- crea carpetas `models/` y `logs/` si no existen,
- entrena PPO con `total_timesteps=300000`,
- guarda modelo con nombre dinámico tipo `ppo_driving_car_v6_<timestamp>`,
- ejecuta una fase visual posterior de 5 episodios en `render_mode="human"`.

## Cómo evaluar un modelo entrenado

Evaluación visual de consola + ventana Pygame (usa `modelo_entrenado.zip`):

```bash
python -m src.scripts.eval_ppo
```

Smoke test del entorno con acciones aleatorias:

```bash
python -m src.scripts.check_env
```

## Cómo iniciar la aplicación Streamlit

```bash
streamlit run streamlit_app.py
```

La app:
- carga `DrivingEnv(render_mode="rgb_array")`,
- intenta cargar `modelo_entrenado.zip`,
- simula hasta 1500 pasos y muestra métricas (pasos, recompensa media, velocidad máxima, tiempo de simulación),
- permite activar/desactivar visualización de sensores.

# 4. Estructura del proyecto

```text
2d-autonomous-driving-rl/
├── README.md
├── requirements.txt
├── packages.txt
├── modelo_entrenado.zip
├── streamlit_app.py
└── src/
    ├── env/
    │   ├── __init__.py
    │   └── gym_env.py
    ├── render/
    │   ├── __init__.py
    │   └── pygame_renderer.py
    ├── sim/
    │   ├── __init__.py
    │   ├── car.py
    │   ├── sensors.py
    │   ├── state.py
    │   └── track.py
    └── scripts/
        ├── __init__.py
        ├── train_ppo.py
        ├── eval_ppo.py
        ├── run_interactive.py
        └── check_env.py
```

## Explicación de componentes clave

- **Entorno personalizado (`src/env/gym_env.py`)**
  - Define `DrivingEnv` (API Gymnasium).
  - Acción continua: `Box(shape=(2,))`.
  - Observación: `Box(shape=(8,))`.
  - Implementa `reset`, `step`, `render`, `close`.
  - Genera pista procedural al inicio de cada episodio y controla lógica de recompensa/terminación.

- **Implementación del agente (PPO)**
  - La política PPO se instancia en scripts de entrenamiento/evaluación usando SB3.
  - Se emplea `MlpPolicy` sobre observaciones vectoriales del entorno.

- **Script de entrenamiento (`src/scripts/train_ppo.py`)**
  - Configura hiperparámetros PPO.
  - Ejecuta aprendizaje.
  - Guarda modelo y muestra evaluación humana posterior.

- **Script de evaluación (`src/scripts/eval_ppo.py`)**
  - Carga `modelo_entrenado.zip`.
  - Corre episodios de evaluación con render humano.
  - Imprime motivo de finalización (`finish`, `off_track`, `timeout`).

- **Interfaz Streamlit (`streamlit_app.py`)**
  - Dashboard para reproducir el agente entrenado con métricas y visualización en tiempo real.

- **Carpeta/modelos guardados**
  - Existe un modelo preentrenado en la raíz: `modelo_entrenado.zip`.
  - El entrenamiento genera modelos adicionales en `models/` (creada en ejecución).

# 5. Funcionalidades principales

## Simulación del entorno 2D
- Cinemática del vehículo basada en aceleración, drag, límite de velocidad y orientación angular.
- Representación top-down con chasis, cabina y ruedas en Pygame.

## Generación procedimental de pista
- La pista se define por una **línea central sinusoidal** con parámetros aleatorios (amplitud y número de ondas), acotada por márgenes del mapa.
- Desde esa línea central se calculan bordes izquierdo/derecho con normales geométricas.

## Sistema de sensores por ray casting
- `SensorSuite` lanza **7 rayos** en un abanico de **120°** desde la parte frontal del coche.
- Cada rayo calcula la intersección más cercana con los bordes de la carretera.
- Las distancias se normalizan para construir la observación del agente.

## Implementación de PPO
- PPO se usa con `MlpPolicy` y parámetros explícitos en el script de entrenamiento (`learning_rate=0.0003`, `n_steps=2048`, `batch_size=64`).
- El agente aprende política de control continuo para velocidad y giro.

## Sistema de recompensas
La lógica de recompensa/fin de episodio del entorno incluye:
- **+150** al cruzar meta (`finish`).
- **-30** por salida de pista lejos de meta (`off_track`).
- **0.05** en “zona de gracia” (fuera de pista pero cerca de meta, dentro de una distancia basada en la longitud del coche).
- Penalización por baja velocidad (`-0.1`) o recompensa proporcional al avance cuando circula sobre pista.
- Truncamiento por límite de pasos (`1500`, evento `timeout`).

## Entrenamiento del agente
- Script dedicado para entrenamiento con PPO y guardado de modelos versionados.
- Soporte de logs para TensorBoard (`tensorboard_log="./logs/"`).

## Evaluación y visualización interactiva
- Evaluación local con Pygame (`eval_ppo.py`).
- Simulador manual interactivo (`run_interactive.py`) para inspección del entorno.
- Interfaz Streamlit (`streamlit_app.py`) con render `rgb_array`, control de velocidad de simulación y métricas de desempeño.
