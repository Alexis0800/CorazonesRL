# Módulo 3: Arquitectura del Agente RL y Pipeline de Entrenamiento

## 1. Descripción General

Este módulo define la pila tecnológica, la topología de la red neuronal y el ciclo de entrenamiento. El objetivo es entrenar un agente capaz de maximizar su recompensa de suma cero en un entorno multi-agente con información imperfecta, utilizando técnicas de *Fictitious Self-Play*.

## 2. Stack Tecnológico

Para garantizar rendimiento y compatibilidad, el proyecto utilizará el siguiente stack en Python:

* **Entorno:** `Farama Gymnasium` (Para envolver el simulador del Módulo 1).
* **Entorno Multi-Agente:** `PettingZoo` (Estándar de la industria que extiende Gymnasium para juegos de varios jugadores y turnos secuenciales).
* **Framework de Deep Learning:** `PyTorch`.
* **Librería de RL:** `Stable-Baselines3` (SB3) o `Ray RLlib`. Se recomienda SB3 + SB3-Contrib (para incluir soporte nativo de *Action Masking*).

## 3. Selección de Algoritmo: PPO (Proximal Policy Optimization)

El algoritmo PPO es ideal para este entorno por su eficiencia de muestreo y su estabilidad en espacios de acción discretos.
La red neuronal aprenderá una política $\pi_\theta(a|s)$ que mapea el estado actual $s$ (el vector de 187 dimensiones) a una distribución de probabilidad sobre las acciones $a$ (las 52 cartas).

### 3.1 Topología de la Red Neuronal (Actor-Critic)

Se utilizará una arquitectura perceptrón multicapa (MLP) estándar con parámetros compartidos para el Actor (quien decide la jugada) y el Crítico (quien evalúa qué tan buena es la situación actual).

* **Capa de Entrada (Input Layer):** 187 neuronas (Corresponde al Vector de Observación).
* **Capas Ocultas (Hidden Layers):** * Capa 1: 256 neuronas (Activación ReLU).
  * Capa 2: 256 neuronas (Activación ReLU).
  * Capa 3: 128 neuronas (Activación ReLU).
* **Capa de Salida - Actor:** 52 neuronas (Activación Softmax). Devuelve las probabilidades de jugar cada carta. *Se aplica el Action Masking multiplicando por 0 las probabilidades de las cartas ilegales antes de la selección final.*
* **Capa de Salida - Crítico:** 1 neurona (Activación Lineal). Devuelve el "Valor" estimado del estado actual del juego.

## 4. Pipeline de Entrenamiento y Fictitious Self-Play

Para evitar el "Colapso de Política" y asegurar que el agente aprenda a jugar contra diferentes perfiles de riesgo, no se entrenará contra copias idénticas en tiempo real.

Se implementará un ciclo de **Auto-juego Ficticio**:

1. **Fase de Inicialización (Heurística):** El Agente V0 entrena jugando contra 3 bots basados en reglas estrictas (ej. "jugar la carta más baja legal") para que la red neuronal mapee rápidamente las reglas y castigos básicos.
2. **El Pool Histórico:** Una vez que el Agente supera a los bots, se inicia el Self-Play. Cada $N$ *steps* (ej. cada 1,000,000 de pasos), se guarda un "snapshot" (captura) del modelo en un directorio local (`/modelos_historicos/`).
3. **Selección de Oponentes:** Al iniciar una nueva partida en el entorno `PettingZoo`, el sistema asigna al Agente principal al Asiento 1. Para los Asientos 2, 3 y 4, el sistema carga aleatoriamente modelos del directorio histórico.
4. **Actualización:** Solo el Agente principal actualiza sus pesos (aprende) durante la partida. Los oponentes históricos actúan en modo inferencia pura (congelados).

## 5. Optimización de Hardware y Ejecución Local

Dado el volumen masivo de partidas necesarias, el entrenamiento debe ejecutarse localmente aprovechando el paralelismo.

* **Entornos Vectorizados (SubprocVecEnv):** En lugar de correr 1 partida a la vez, se instanciarán múltiples copias del entorno en paralelo (ej. 16 o 32 entornos simultáneos). Esto saturará los hilos del procesador de alto rendimiento para recolectar experiencias a máxima velocidad y almacenarlas rápidamente en la RAM antes de pasarlas a PyTorch para la actualización de la red.
* **Dispositivo de Cómputo:** Si bien PyTorch permite entrenar en GPU, para redes MLP pequeñas (256x256) atadas a simuladores lógicos pesados en CPU, el cuello de botella suele ser la simulación, no la red. Entrenar puramente con la CPU multi-hilo suele ofrecer el mejor rendimiento para este caso específico.
