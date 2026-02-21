# 2D Autonomous Driving with Reinforcement Learning (PPO)

## 1. Descripción general del proyecto

Este repositorio implementa un sistema de **conducción autónoma 2D** donde un agente aprende a conducir en una pista generada procedimentalmente usando **Aprendizaje por Refuerzo (Reinforcement Learning)**.

El objetivo del sistema es que el vehículo:
- Avance por la carretera sin salirse.
- Evite colisiones.
- Alcance la meta en un número limitado de pasos.

El problema que resuelve es la toma de decisiones de control continuo en un entorno dinámico: el agente debe elegir, en cada paso, valores de **aceleración** y **giro** para mantener estabilidad, progresar y terminar el circuito.

El enfoque se basa en un entorno compatible con **Gymnasium** (`DrivingEnv`) y en el algoritmo **PPO (Proximal Policy Optimization)** de Stable-Baselines3. En este proyecto, PPO optimiza una política neuronal (`MlpPolicy`) sobre un espacio de acciones continuo de 2 dimensiones (`throttle`, `steering`) y observaciones de 8 dimensiones (7 rayos de sensores + velocidad normalizada).

De forma resumida, el proceso de aprendizaje implementado en el código es:
1. Se genera una pista procedimental nueva al reiniciar episodio.
2. El agente observa distancias de ray casting y velocidad.
3. PPO propone una acción continua.
4. El entorno simula cinemática, detecta colisiones/meta/timeout y calcula recompensa.
5. PPO actualiza la política con los datos acumulados hasta mejorar el comportamiento.

## 2. Stack tecnológico utilizado

### Lenguaje principal
- **Python**.

### Librerías principales detectadas
- **Gymnasium (`gymnasium==1.2.3`)**: interfaz del entorno RL, espacios de acción/observación y ciclo `reset/step`.
- **Pygame (`pygame==2.6.1`)**: renderizado 2D, simulación visual, utilidades de fuente/superficie y soporte de ejecución interactiva.
- **Stable-Baselines3 (SB3) (`stable_baselines3==2.7.1`)**: implementación de PPO usada para entrenamiento y evaluación.
- **PyTorch (`torch==2.10.0`)**: backend de cómputo para SB3.
- **Streamlit (`streamlit==1.54.0`)**: interfaz de evaluación visual del modelo entrenado.
- **NumPy (`numpy==2.2.6`)**: operaciones numéricas de observaciones y normalización.

## 3. Instalación y ejecución

Clona el proyecto y crea un entorno virtual.

### Requisitos previos
- Python 3.10 o superior.
- Gestor de paquetes `pip`.

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

En sistemas Linux, puede ser necesario instalar las dependencias del sistema requeridas por Pygame, listadas en `packages.txt`.

### Modelos y almacenamiento
- Para evaluación con scripts y Streamlit, el repositorio incluye un modelo en la raíz: `modelo_entrenado.zip`.
- El script de entrenamiento guarda nuevos modelos en `models/` y logs en `logs/`.

### Cómo ejecutar el entorno (modo interactivo manual)

```bash
python -m src.scripts.run_interactive
```

Controles implementados:
- `W/S`: acelerar/frenar (marcha atrás).
- `A/D`: girar izquierda/derecha.
- `R`: generar nueva pista/episodio.
- `ESC`: salir.

### Cómo lanzar el entrenamiento PPO

```bash
python -m src.scripts.train_ppo
```

Comportamiento real del script:
- Crea carpetas `models/` y `logs/` si no existen.
- Entrena PPO con `total_timesteps=300000`.
- Guarda modelo con nombre dinámico tipo `ppo_driving_car_<version>_<timestamp>`.
- Ejecuta una fase visual posterior de 5 episodios en `render_mode="human"`.

### Cómo evaluar un modelo entrenado
Evaluación visual de consola + ventana Pygame (por defecto `modelo_entrenado.zip`):

```bash
python -m src.scripts.eval_ppo
```

### Despliegue en Streamlit Cloud

Este proyecto se encuentra desplegado en **Streamlit Cloud**, lo que permite acceder al simulador de conducción autónoma directamente desde el navegador, sin necesidad de instalación local.

**Acceso al simulador:**

https://2d-autonomous-driving-rl.streamlit.app/

A través de esta plataforma es posible:

- Visualizar la simulación del modelo PPO previamente entrenado.
- Analizar métricas de rendimiento en tiempo real.
- Observar el funcionamiento del sistema de sensores por ray casting.
- Evaluar el comportamiento autónomo del vehículo en el entorno 2D.

El despliegue en la nube facilita la demostración pública del proyecto y su evaluación remota sin requerir configuración adicional.

## 4. Estructura del proyecto

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

### Explicación de componentes clave

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
  - Guarda modelo y ejecuta una demostración visual post-entrenamiento.

- **Script de evaluación (`src/scripts/eval_ppo.py`)**
  - Carga `modelo_entrenado.zip`.
  - Corre episodios de evaluación con render humano.
  - Imprime motivo de finalización (`finish`, `off_track`, `timeout`).
  - Es posible evaluar cualquier modelo generado durante el entrenamiento (ubicados en la carpeta models/) modificando la variable `MODEL_PATH` en la parte superior del archivo `eval_ppo.py`.

- **Interfaz Streamlit (`streamlit_app.py`)**
  - Dashboard para reproducir el agente entrenado con métricas y visualización en tiempo real.

- **Persistencia de modelos**
  - **Modelo de referencia**: Se incluye un modelo preentrenado en la raíz del proyecto bajo el nombre `./modelo_entrenado.zip`.
  - **Directorios de salida**: El proceso de entrenamiento genera y almacena modelos adicionales en el directorio `./models/` (creado automáticamente durante la ejecución).

## 5. Funcionalidades principales

### Simulación del entorno 2D
- Cinemática del vehículo basada en aceleración, drag, límite de velocidad y orientación angular.
- Representación top-down con chasis, cabina y ruedas en Pygame.

### Generación procedimental de pista
- La pista se define por una **línea central sinusoidal** con parámetros aleatorios (amplitud y número de ondas), acotada por márgenes del mapa.
- Desde esa línea central se calculan bordes izquierdo/derecho con normales geométricas.

### Sistema de sensores por ray casting
- `SensorSuite` lanza **7 rayos** en un abanico de **120°** desde la parte frontal del coche.
- Cada rayo calcula la intersección más cercana con los bordes de la carretera.
- Las distancias se normalizan para construir la observación del agente.

### Implementación de PPO
- Se utiliza el algoritmo PPO (Proximal Policy Optimization) con una arquitectura de red neuronal `MlpPolicy` (Multi-layer Perceptron).
- Configuración de hiperparámetros:
  - `learning_rate=0.0003`: Tasa de aprendizaje moderada para asegurar una convergencia estable.
  - `n_steps=2048`: Tamaño de la ventana de experiencia recolectada antes de cada actualización.
  - `batch_size=64`: Número de muestras por cada paso de optimización del gradiente.
- El agente optimiza una política de control continuo, aprendiendo a modular de forma fluida la aceleración y el ángulo de giro.

### Sistema de recompensas
La lógica de recompensa/fin de episodio del entorno incluye:
- **+150** al cruzar meta (`finish`).
- **-30** por salida de pista lejos de meta (`off_track`).
- **0.05** en “zona de gracia” (fuera de pista pero cerca de meta, dentro de una distancia basada en la longitud del coche).
- Penalización por baja velocidad (`-0.1`) o recompensa proporcional al avance cuando circula sobre pista.
- Truncamiento por límite de pasos (`1500`, evento `timeout`).

### Entrenamiento del agente
- **Script de entrenamiento optimizado**: Permite el aprendizaje del agente con PPO y gestiona el versionado automático de los modelos guardados para evitar la pérdida de progresos.
- **Monitorización con TensorBoard**: El sistema genera logs detallados en el directorio `./logs/`, permitiendo visualizar en tiempo real métricas críticas como la evolución de la recompensa media, la pérdida de la red neuronal (loss) y la duración de los episodios.

### Visualización de métricas
Para analizar la evolución del aprendizaje en tiempo real, abre una terminal y ejecuta:

```bash
tensorboard --logdir ./logs/
```

Tras ejecutarlo, accede desde tu navegador a: http://localhost:6006/

#### Métricas clave a observar:

- `rollout/ep_rew_mean`: Indica el rendimiento global del agente. Representa la recompensa media acumulada por episodio, una tendencia ascendente es el principal indicador de que el agente está aprendiendo la tarea.

- `train/entropy_loss`: Refleja el nivel de exploración de la política. Una disminución progresiva indica que el agente reduce la aleatoriedad en sus acciones, pasando de una fase exploratoria inicial a una estrategia de conducción más definida y determinista.

- `train/explained_variance`: Mide la estabilidad y calidad del modelo. Cuanto más cerca esté de **1**, más predecibles y sólidos son los movimientos del agente, lo que significa que la red neuronal entiende correctamente las consecuencias de sus acciones.
