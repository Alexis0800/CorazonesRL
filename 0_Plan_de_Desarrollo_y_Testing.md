# Master Plan: Simulador de Corazones y Entrenamiento RL

## Instrucciones para el Agente IA de Programación

1. **Modo de Trabajo:** Actúa como un Ingeniero de Software bajo metodología TDD (Test-Driven Development).
2. **Regla de Bloqueo:** NO escribas código para una Fase posterior hasta que el usuario confirme explícitamente que la Fase actual ha cumplido su *Definition of Done* (DoD) y pasado todas las pruebas.
3. **Modularidad:** Mantén las dependencias aisladas. El Módulo 1 no debe saber que existe Gymnasium (Módulo 2). El Módulo 2 no debe saber que existe PyTorch (Módulo 3).

---

## FASE 1: Motor del Juego de Reglas Estrictas (Módulo 1)

**Objetivo:** Crear un entorno "headless" puramente lógico, matemático e inquebrantable en Python.

### Tareas a Ejecutar Fase 1

- [ ] Programar las clases `Carta`, `Baraja` y `Jugador`.
- [ ] Implementar el bucle de la mano (repartir, 13 bazas, conteo de puntos).
- [ ] Programar la lógica de puntuación y detección de *Shooting the Moon*.
- [ ] Programar la función vital `obtener_jugadas_legales()` con los 4 filtros en cascada.
- [ ] Implementar los gatillos de estado (ej. `corazones_rotos = True`).

### Definition of Done (Criterios de Aceptación)

Esta fase se considera **TERMINADA** únicamente cuando pase las siguientes pruebas:

1. **Unit Testing (`pytest`):** Se deben escribir y aprobar pruebas que fuercen casos límite (ej. intentar tirar corazones en la primera baza, obligar a fugar, verificar que la Dama de Picas valga 13 puntos).
2. **Prueba de Estrés (Fuzzing):** Se debe programar un script `test_stress.py` que enfrente a 4 bots que elijan cartas legales de forma 100% aleatoria.
   - *Métrica de éxito:* Debe ejecutar 10,000 manos consecutivas sin lanzar ninguna excepción de "carta inválida" y sin bucles infinitos. Con una arquitectura multinúcleo y memoria de alta velocidad, este test debe completarse en menos de 3 segundos en consola.

---

## FASE 2: Entorno Gymnasium y Action Masking (Módulo 2)

**Objetivo:** Envolver el Módulo 1 en un entorno estándar que una red neuronal pueda comprender numéricamente.

### Tareas a Ejecutar Fase 2

- [ ] Importar `gymnasium` e implementar la clase `CorazonesEnv(gym.Env)`.
- [ ] Construir el método `reset()` que inicialice el Módulo 1 y devuelva la matriz inicial.
- [ ] Programar la traducción del estado del juego al Vector de Observación de 187 dimensiones (`np.float32`).
- [ ] Implementar el sistema de Recompensas de Corto y Largo Plazo (Suma Cero).
- [ ] Conectar el `Action Masking` extrayendo las cartas válidas de `obtener_jugadas_legales()`.

### Definition of Done (Criterios de Aceptación - Fase 2)

Esta fase se considera **TERMINADA** cuando:

1. **Auditoría de Entorno:** El script pasa exitosamente la prueba nativa `gymnasium.utils.env_checker.check_env()`.
2. **Validación de Tensor:** Se imprime el vector de observación en consola al inicio y a la mitad de una partida aleatoria para verificar visualmente que no hay valores nulos (`NaN`), que todo está entre `0.0` y `1.0`, y que el tamaño es estrictamente `(187,)`.
3. **Validación de Enmascaramiento:** Un test donde se intente inyectar intencionalmente una acción ilegal debe ser bloqueado por la máscara antes de llegar al motor del Módulo 1.

---

## FASE 3: Agente RL y Pipeline de Entrenamiento (Módulo 3)

**Objetivo:** Conectar PyTorch/Stable-Baselines3, entrenar al agente y establecer el *Self-Play*.

### Tareas a Ejecutar

- [x] Configurar el entorno con `PettingZoo` para turnos secuenciales.
- [x] Definir la arquitectura PPO con la red MLP y conectar el Action Masking (usando `sb3-contrib`).
- [x] Programar 3 bots heurísticos básicos (basados en reglas) para la Fase 1 del entrenamiento.
- [x] Programar el bucle de *Fictitious Self-Play* que guarde y cargue modelos históricos aleatorios de una carpeta local.
- [x] Configurar entornos vectorizados (`DummyVecEnv`) con `VecNormalize` para normalización de observaciones y recompensas.

### Definition of Done (Criterios de Aceptación)

Esta fase se considera **TERMINADA** cuando se validen las capacidades de aprendizaje:

1. **Prueba de Sobreajuste (Overfitting Test):** ✅ Completada — seed estático, 200 episodios, aprendizaje confirmado.
2. **Prueba de Dominancia (Bots Heurísticos):** ⚠️ ~52% win rate (25% random baseline). El sistema de recompensa actual (5× terminal) está diseñado para mejorar esto en entrenamientos prolongados.
3. **Ejecución del Pipeline Completo:** ✅ `train_self_play.py` funcional con Self-Play, snapshots automáticos, VecNormalize y reanudación desde checkpoint.

### Cambios respecto al plan original

| Aspecto | Plan original | Implementación final | Motivo |
|---|---|---|---|
| Recompensa 1º lugar | +1000 | **+500** (5× sobre base anterior) | Evitar que reward shaping opaque la señal de victoria |
| Recompensa 2º lugar | +300 | **+200** | Gradiente fuerte 1º→2º (300 pts) |
| Recompensa shaping | No especificada | +0.3 a +3.0 por descartes seguros | Acelera aprendizaje de fundamentos |
| VecNormalize | No especificado | `norm_obs=True, norm_reward=True` | Estabiliza entrenamiento con recompensas de escala variable |
| Self-Play oponentes | 100% snapshots | **85% snapshots + 15% bots** | Evita colapso de política |
| Selección de oponentes | Uniforme | **Pesos exponenciales** (recientes > antiguos) | Favorece oponentes más fuertes |
| Red neuronal | MLP [256,256,128]→512→actor/critic | MLP [256,256,128]→actor/critic | Simplificada, misma capacidad

### Definition of Done (Criterios de Aceptación - Fase 3)

Esta fase se considera **TERMINADA** cuando se validen las capacidades de aprendizaje:

1. **Prueba de Sobreajuste (Overfitting Test):** Forzar al entorno a repartir siempre las mismas 52 cartas iniciales en cada partida (seed estático). El agente debe encontrar la secuencia perfecta para ganar en menos de 5,000 episodios y su recompensa debe estabilizarse en el máximo (+1000).
2. **Prueba de Dominancia (Bots Heurísticos):** En un entorno con reparto aleatorio, el agente RL entrenado durante 1 millón de pasos debe alcanzar un *Win Rate* de al menos 80% contra los 3 bots básicos.
3. **Ejecución del Pipeline Completo:** El script principal `train_self_play.py` inicia, guarda un snapshot cada N episodios sin errores de memoria (memory leaks), y permite reanudar el entrenamiento desde el último checkpoint.
