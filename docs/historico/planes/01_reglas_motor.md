# Módulo 1: Reglas del Juego y Motor de Simulación (Corazones)

## 1. Descripción General

Este módulo define la estructura lógica y matemática del simulador de Corazones (Headless Environment). No contiene dependencias de Inteligencia Artificial; su único propósito es simular partidas válidas a máxima velocidad, exponer el estado del juego y gestionar las transiciones de estado.

## 2. Componentes Básicos

### 2.1 Representación de las Cartas

Las cartas se representan internamente como objetos con dos valores enteros:

* **Palo (Suit):** `0` (Tréboles), `1` (Diamantes), `2` (Picas), `3` (Corazones).
* **Valor (Rank):** Del `2` al `14` (J=11, Q=12, K=13, As=14).
* **Unicidad:** La baraja consta de exactamente 52 cartas únicas. El motor debe garantizar por diseño que una carta jugada desaparece de las manos y que es matemáticamente imposible que existan cartas duplicadas en el estado del juego.

### 2.2 Puntuación de las Cartas

* Cualquier carta con Palo `3` (Corazones) = **1 punto**.
* Carta con Palo `2` y Valor `12` (Dama de Picas) = **13 puntos**.
* Resto de cartas = **0 puntos**.

### 2.3 Variables de Estado Globales

* `corazones_rotos`: Booleano (Inicia en `False`). Determina si es legal abrir una baza con corazones.

## 3. Ciclo de Vida del Juego

1. **Reparto:** Se barajan las 52 cartas y se reparten 13 a cada uno de los 4 jugadores.
2. **Intercambio (Fase Opcional V2.0):** Rotación de 3 cartas (Izquierda, Derecha, Frente, Nada). *Nota: Para la V1.0 de entrenamiento, se omitirá esta fase asumiendo la 4ª mano.*
3. **Bazas (Tricks):** 13 rondas consecutivas donde cada jugador tira una carta.
4. **Conteo:** Al final de las 13 rondas, se suman los puntos de las bazas. Si un jugador llega a 100 puntos totales acumulados en su historial, la partida finaliza.

## 4. Validación Lógica (Strict Rules Engine / Action Masking)

El método `obtener_jugadas_legales(mano_jugador, mesa, corazones_rotos, numero_baza)` debe aplicar los siguientes filtros en cascada para devolver solo el subconjunto de cartas legales:

* **Filtro 1 (La Salida Inicial):** Si es la baza #1 y la mesa está vacía, la ÚNICA jugada legal es el 2 de Tréboles (Palo `0`, Valor `2`).
* **Filtro 2 (Primera Baza Segura):** Durante la baza #1, ningún jugador puede jugar una carta que valga puntos (Corazones o Dama de Picas), a menos que no tenga otra opción legal.
* **Filtro 3 (Asistir al Palo / Voids):** Si ya hay una carta en la mesa, el jugador DEBE jugar una carta del mismo palo. Si no tiene cartas de ese palo en su mano (Void), está libre de esta restricción y puede jugar cualquier otra carta legal de su mano. *(Nota: Si lanza un corazón, esto activará el Gatillo de Romper Corazones descrito en la Sección 5).*
* **Filtro 4 (Liderar con Corazones):** Si un jugador va a abrir una baza (la mesa está vacía) y la variable global `corazones_rotos` es `False`, NO puede jugar Corazones. Excepción: Si toda su mano restante son solo Corazones.

*Nota de Seguridad:* Si el motor recibe una carta que no está en la lista de jugadas legales devuelta por este método, lanzará una excepción fatal (Crash) indicando un fallo en el modelo de la IA.

## 5. Ejecución de Jugadas y Cambios de Estado (Triggers)

Cada vez que un jugador lanza una carta a la mesa, el motor debe verificar las condiciones de estado:

* **Gatillo de Romper Corazones:** Si un jugador juega una carta de Corazones (Palo `3`) en una baza que NO fue iniciada con Corazones (aprovechando la regla de Void del Filtro 3), el motor debe cambiar permanentemente la variable global `corazones_rotos = True` para el resto de la mano. A partir de este momento, cualquier jugador puede abrir futuras bazas con Corazones.

## 6. Resolución de la Baza y Pleno (Shooting the Moon)

Quien tira la carta con el **Valor más alto del Palo de salida**, se lleva las 4 cartas.

Al final de las 13 bazas, se evalúa si un jugador acumuló los 26 puntos de la ronda:

* **Si NO hay Pleno:** Cada jugador suma los puntos de las cartas que se llevó a su puntaje histórico.
* **Si SÍ hay Pleno:** El jugador que hizo el Pleno suma `0` puntos. Los otros 3 jugadores suman `+26` puntos cada uno a su puntaje histórico.
