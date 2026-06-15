# Módulo 2: Entorno de Aprendizaje (Gymnasium) y Representación del Estado

## 1. Descripción General

Este módulo envuelve el Motor de Simulación (Módulo 1) dentro del estándar de la industria `Farama Gymnasium`. Su objetivo es traducir el estado del juego a matrices matemáticas (Tensores) que una red neuronal pueda procesar, y aplicar el diseño de recompensas de suma cero.

## 2. Espacio de Observación (State Representation)

Para que la red neuronal pueda "ver" la partida, se elimina cualquier representación gráfica o de texto. El estado se transforma en un vector numérico (One-Hot Encoding) de **187 dimensiones** de tipo `np.float32`. Todos los valores están normalizados entre `0.0` y `1.0`.

El vector se compone de los siguientes bloques estratégicos:

### 2.1 Vector de Mano Actual (52 valores)

* Representa las cartas que el agente tiene actualmente para jugar.
* **Formato:** Índices del `0` al `51`. `1.0` significa que el agente posee la carta; `0.0` significa que no.

### 2.2 Vector de Mesa Actual (52 valores)

* Representa las cartas que han sido lanzadas en la **baza actual**.
* **Formato:** `1.0` si la carta está actualmente sobre la mesa; `0.0` si no.

### 2.3 Vector de Memoria / Cementerio (52 valores)

* Representa todas las cartas que ya fueron jugadas en bazas anteriores y están fuera de juego.

### 2.4 Vector de Vacíos Conocidos / Fugas (16 valores)

* *Crucial para detectar intentos de Shooting the Moon y planear estrategias de sangrado.* Registra qué jugadores han demostrado no tener cartas de un palo específico (Void) al no poder asistir a una baza.
* Son 4 bloques de 4 valores (Agente, Izquierda, Frente, Derecha). Cada bloque tiene 4 interruptores (Trébol, Diamante, Picas, Corazones).
* **Formato:** `1.0` si se sabe con certeza que el jugador ya NO tiene cartas de ese palo; `0.0` si aún podría tenerlas.
* *Ejemplo:* Si el Rival de la Derecha tira un Corazón cuando se pidieron Tréboles, su interruptor de Trébol cambia a `1.0` permanentemente por el resto de la mano.

### 2.5 Vector de Contexto y Metadatos Relativos (10 valores)

Proporciona el contexto global de la partida, mapeando los puntajes a los asientos físicos de la mesa para correlacionarlos con los Vacíos (Fugas).

* **Índices 172 a 175 (Puntajes Globales):** Puntaje acumulado histórico de los 4 asientos (Agente, Izquierda, Frente, Derecha). Normalizado de `0.0` a `1.0` (dividido entre 100).
* **Índices 176 a 179 (Puntos de la Mano Actual):** Puntos obtenidos en las bazas de la mano en curso (de 0 a 26) por los 4 asientos (Agente, Izquierda, Frente, Derecha). Normalizados entre `0.0` y `1.0`. *Si un rival empieza a acumular 10 o más puntos aquí, la IA cruzará este dato con el Vector de Vacíos para deducir un posible Pleno.*
* **Índice 180 (Corazones Rotos):** `1.0` si es True, `0.0` si es False.
* **Índice 181 (Posición en la Baza Actual):** Indica en qué orden le toca lanzar su carta a la IA en esta baza específica. `0.0` (Abre la baza), `0.33` (Segundo), `0.66` (Tercero), `1.0` (Último en cerrar).

### 2.6 Rastreador de la Dama de Picas (5 valores)

Este bloque es el núcleo táctico para estrategias avanzadas. Permite al agente diferenciar si un rival está intentando un "Shooting the Moon" o si simplemente se comió la carta de penalización por error. También le permite saber si ya es seguro "achicar" (ducking) o si la amenaza sigue en la baraja.

Consta de 5 interruptores mutuamente excluyentes (solo uno puede estar en `1.0` a la vez):

* **Índice 182:** `1.0` si la Dama de Picas sigue oculta (en juego).
* **Índice 183:** `1.0` si el Agente se comió la Dama de Picas en esta mano.
* **Índice 184:** `1.0` si el Rival de la Izquierda se comió la Dama.
* **Índice 185:** `1.0` si el Rival de Enfrente se comió la Dama.
* **Índice 186:** `1.0` si el Rival de la Derecha se comió la Dama.

## 3. Espacio de Acción (Action Space)

El espacio de acción es discreto (`Discrete(52)`). La red neuronal siempre devolverá 52 probabilidades (una por cada carta de la baraja).

### 3.1 Enmascaramiento de Acciones (Action Masking)

Para evitar que la IA elija jugadas que rompen las reglas, el entorno aplicará una máscara lógica en cada turno:

1. El entorno consulta la función `obtener_jugadas_legales()` del Módulo 1.
2. Las probabilidades de las cartas que devuelva la red neuronal y que NO estén en la lista legal, se multiplican por `0` (o se les aplica `-infinito` antes del Softmax).
3. La IA ejecuta la carta legal que haya obtenido la mayor probabilidad.

## 4. Diseño de Recompensas (Reward Structure)

El sistema utiliza una arquitectura de **Suma Cero (Zero-Sum Reward)** para los objetivos globales, garantizando que el agente siempre busque escalar en la clasificación incluso en escenarios perdidos, y castigos escalonados para las micro-decisiones.

### 4.1 Recompensas por Turno (Corto Plazo)

Se calculan inmediatamente después de que termina una baza de 4 cartas. En el caso del Pleno, se calcula al finalizar la 13ª baza.

* **+0.0:** Llevarse una baza limpia (Sin cartas de puntos). Fomenta el control neutral de la mesa.
* **-1.0:** Por cada carta de Corazones obtenida en la baza.
* **-13.0:** Por llevarse la Dama de Picas.
* **+50.0:** Recompensa crítica otorgada ÚNICAMENTE si el Agente logra el *Shooting the Moon* (26 puntos en una mano).
*(Nota: No hay penalización artificial a corto plazo por permitir que un rival haga el Pleno. El agente aprenderá a bloquearlo evaluando el impacto de recibir los +26 puntos del motor de juego sobre su recompensa final de Suma Cero).*

### 4.2 Recompensas de Fin de Partida (Largo Plazo)

Se otorgan una sola vez cuando el entorno llega a su estado `Terminal` (algún jugador cruza los 100 puntos históricos). Se basan en la posición final del agente respecto a los demás jugadores.

* **+1000.0:** Primer puesto (Victoria total).
* **+300.0:** Segundo puesto.
* **-300.0:** Tercer puesto.
* **-1000.0:** Cuarto puesto (Último lugar).

*Justificación de Diseño:* El gradiente fuerte entre el tercer y cuarto puesto obliga a la IA a seguir optimizando su juego defensivo hasta el final, evitando políticas de suicidio cuando detecta que no puede alcanzar el primer lugar.
