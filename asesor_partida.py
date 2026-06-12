"""
Asesor de partida completa para Corazones — con tracking de estado.

A diferencia de asesor_carta.py (stateless, una consulta por turno),
este script mantiene el estado completo de la partida entre bazas:
  - Tu mano (se actualiza automáticamente al jugar)
  - Cementerio (cartas de bazas anteriores)
  - Corazones rotos (detección automática)
  - Dama de Picas (tracking automático)
  - Vacíos detectados
  - Puntajes acumulados

Flujo de uso:
  1. Ingresás tus 13 cartas iniciales.
  2. El script determina quién lidera (automático si tenés 2♣).
  3. Para cada jugada en la baza:
     - Si es tu turno: el modelo recomienda la mejor carta.
     - Si es turno de otro: ingresás qué jugó.
  4. Al completar 4 cartas, la baza se resuelve automáticamente.
  5. Al terminar la mano (13 bazas), se aplica la puntuación.
  6. Podés seguir con otra mano o terminar la partida.

Uso:
    python asesor_partida.py
    python asesor_partida.py --modelo modelos_historicos/v2/modelo_final
    python asesor_partida.py --posicion 2  # Estás en posición 2 (tercero en jugar)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reutilizar utilidades de asesor_carta.py
from asesor_carta import (
    parsear_carta,
    parsear_mano,
    ids_a_nombres,
    compute_legales,
    construir_observacion_parcial,
    normalizar_observacion,
    cargar_modelo,
    NOMBRES_PALOS,
    _PALO_NOMBRE,
    _VALOR_NOMBRE,
    _NOMBRE_VALOR,
    _PALOS,
    _VALORES,
)

# ----------------------------------------------------------------
# EstadoPartida: tracking completo del juego
# ----------------------------------------------------------------


class EstadoPartida:
    """Mantiene el estado completo de una partida de Corazones.

    Trackea la mano del humano, la mesa actual, el cementerio,
    corazones rotos, Dama de Picas, vacíos, puntajes de la mano
    actual e históricos.

    No depende del motor ni de Gymnasium — es autocontenido
    para funcionar como asistente externo.

    Attributes:
        mano: Lista de IDs de cartas en mano del humano.
        agente_idx: Posición del humano en la mesa (0-3).
        mesa: Lista de tuplas (jugador_idx, carta_id) en la baza actual.
        cementerio: Lista de IDs de cartas ya jugadas en bazas anteriores.
        vacios: Lista de 4 sets con palos donde cada jugador hizo void.
        puntajes_historicos: Puntuación acumulada entre manos.
        puntos_mano: Puntos acumulados en la mano actual por jugador.
        corazones_rotos: True si ya se jugó un corazón.
        dama_picas_en: Índice del jugador que recibió la Q♠ (None si oculta).
        num_baza: Número de baza actual (1-13).
        jugador_inicial: Quién lidera la baza actual.
        palo_salida: Palo de la primera carta de la baza (None si mesa vacía).
        es_primera_baza: True durante la primera baza de la mano.
    """

    UMBRAL_FIN_PARTIDA: int = 100

    def __init__(
        self,
        mano_inicial: List[int],
        agente_idx: int = 0,
    ) -> None:
        """Inicializa el estado de partida.

        Args:
            mano_inicial: 13 IDs de carta (0-51), sin duplicados.
            agente_idx: Posición del humano (0-3).

        Raises:
            ValueError: Si la mano no tiene 13 cartas o tiene duplicados.
        """
        if len(mano_inicial) != 13:
            raise ValueError(
                f"La mano inicial debe tener 13 cartas, se recibieron {len(mano_inicial)}"
            )
        if len(set(mano_inicial)) != 13:
            raise ValueError("La mano inicial contiene cartas duplicadas")

        if not (0 <= agente_idx <= 3):
            raise ValueError(
                f"agente_idx debe estar entre 0 y 3, recibido {agente_idx}")

        self.mano: List[int] = sorted(mano_inicial)
        self.agente_idx: int = agente_idx

        # Estado por mano
        self.mesa: List[Tuple[int, int]] = []
        self.cementerio: List[int] = []
        self.vacios: List[set] = [set(), set(), set(), set()]
        self.corazones_rotos: bool = False
        self.dama_picas_en: Optional[int] = None
        self.num_baza: int = 0
        self.jugador_inicial: int = 0
        self.palo_salida: Optional[int] = None
        self.es_primera_baza: bool = True

        # Puntajes
        self.puntos_mano: List[int] = [0, 0, 0, 0]
        self.puntajes_historicos: List[int] = [0, 0, 0, 0]
        self._pleno_jugador: Optional[int] = None

    # ------------------------------------------------------------
    # Inicialización de mano
    # ------------------------------------------------------------

    def _tiene_2_treboles(self) -> bool:
        """True si el humano tiene el 2♣ (id=0)."""
        return 0 in self.mano

    def iniciar_mano(self, jugador_inicial: Optional[int] = None) -> None:
        """Inicia una nueva mano.

        Resetea el estado por mano (bazas, puntos, mesa, etc.).
        Si el humano tiene 2♣, determina automáticamente que lidera.

        Args:
            jugador_inicial: Quién lidera. Si es None y el humano
                no tiene 2♣, debe especificarse externamente.
        """
        self.mesa = []
        self.cementerio = []
        self.vacios = [set(), set(), set(), set()]
        self.corazones_rotos = False
        self.dama_picas_en = None
        self.num_baza = 1
        self.palo_salida = None
        self.es_primera_baza = True
        self.puntos_mano = [0, 0, 0, 0]
        self._pleno_jugador = None

        if jugador_inicial is not None:
            self.jugador_inicial = jugador_inicial
        elif self._tiene_2_treboles():
            self.jugador_inicial = self.agente_idx
        # Si no, se espera que el REPL lo pregunte

    # ------------------------------------------------------------
    # Propiedades calculadas
    # ------------------------------------------------------------

    @property
    def jugador_actual(self) -> int:
        """Quién debe jugar ahora (basado en jugador_inicial + mesa)."""
        return (self.jugador_inicial + len(self.mesa)) % 4

    def es_turno_humano(self) -> bool:
        """True si es el turno del humano."""
        return self.jugador_actual == self.agente_idx

    def baza_terminada(self) -> bool:
        """True si la baza tiene 4 cartas."""
        return len(self.mesa) == 4

    def mano_terminada(self) -> bool:
        """True si se jugaron las 13 bazas (mano vacía)."""
        return len(self.mano) == 0 and len(self.cementerio) == 52

    def juego_terminado(self) -> bool:
        """True si algún jugador alcanzó el umbral de fin de partida."""
        return any(p >= self.UMBRAL_FIN_PARTIDA for p in self.puntajes_historicos)

    def puede_jugar_carta(self, carta_id: int) -> bool:
        """True si la carta está en la mano del humano."""
        return carta_id in self.mano

    # ------------------------------------------------------------
    # Jugadas
    # ------------------------------------------------------------

    def registrar_jugada(self, jugador_idx: int, carta_id: int) -> None:
        """Registra una carta jugada por un jugador.

        Args:
            jugador_idx: Índice del jugador (0-3).
            carta_id: ID de la carta jugada (0-51).

        Raises:
            ValueError: Si el humano juega una carta que no tiene.
        """
        # Validar que el humano tenga la carta
        if jugador_idx == self.agente_idx and carta_id not in self.mano:
            raise ValueError(
                f"La carta {ids_a_nombres([carta_id])[0]} (id={carta_id}) "
                f"no está en tu mano."
            )

        palo = _PALOS[carta_id]

        # Establecer palo de salida si es la primera carta de la baza
        if not self.mesa:
            self.palo_salida = palo

        # Detectar void: si hay palo de salida y no se sigue
        if self.mesa and self.palo_salida is not None and palo != self.palo_salida:
            self.vacios[jugador_idx].add(self.palo_salida)

        # Remover de la mano del humano
        if jugador_idx == self.agente_idx:
            self.mano.remove(carta_id)

        # Agregar a la mesa
        self.mesa.append((jugador_idx, carta_id))

        # Gatillo: corazones rotos
        if palo == 3 and not self.corazones_rotos:
            self.corazones_rotos = True

    # ------------------------------------------------------------
    # Resolución de baza
    # ------------------------------------------------------------

    def resolver_baza(self) -> int:
        """Resuelve la baza actual: determina ganador, acumula puntos.

        Returns:
            Índice del jugador que ganó la baza.

        Raises:
            RuntimeError: Si la baza no tiene 4 cartas.
        """
        if len(self.mesa) != 4:
            raise RuntimeError(
                f"No se puede resolver una baza con {len(self.mesa)} cartas "
                f"(se necesitan 4)."
            )

        # Determinar ganador: carta más alta del palo de salida
        ganador_idx = self.mesa[0][0]
        carta_mas_alta = self.mesa[0][1]
        palo_ref = self.palo_salida if self.palo_salida is not None else _PALOS[
            self.mesa[0][1]]

        for jug_idx, cid in self.mesa[1:]:
            if _PALOS[cid] == palo_ref and _VALORES[cid] > _VALORES[carta_mas_alta]:
                ganador_idx = jug_idx
                carta_mas_alta = cid

        # Contar puntos y asignar al ganador
        puntos_baza = 0
        for _, cid in self.mesa:
            palo = _PALOS[cid]
            valor = _VALORES[cid]
            if palo == 3:  # corazón
                puntos_baza += 1
            elif palo == 2 and valor == 12:  # Q♠
                puntos_baza += 13
                self.dama_picas_en = ganador_idx

        self.puntos_mano[ganador_idx] += puntos_baza

        # Mover cartas al cementerio
        for _, cid in self.mesa:
            self.cementerio.append(cid)

        # Limpiar mesa y avanzar
        self.mesa = []
        self.palo_salida = None
        self.num_baza += 1
        self.jugador_inicial = ganador_idx
        self.es_primera_baza = False

        return ganador_idx

    # ------------------------------------------------------------
    # Jugadas legales para el humano
    # ------------------------------------------------------------

    def legales_humano(self) -> List[int]:
        """Calcula las jugadas legales para el humano en el estado actual.

        Returns:
            Lista de IDs de cartas legales. Vacía si no es su turno
            o no tiene cartas.
        """
        if not self.mano or not self.es_turno_humano():
            return []

        mesa_ids = [cid for _, cid in self.mesa]
        return compute_legales(
            mano=self.mano,
            mesa_ids=mesa_ids,
            corazones_rotos=self.corazones_rotos,
            es_primera_baza=self.es_primera_baza,
        )

    # ------------------------------------------------------------
    # Construcción de observación para el modelo
    # ------------------------------------------------------------

    def _construir_observacion(self) -> np.ndarray:
        """Construye el vector de 187 dimensiones para el modelo."""
        mesa_ids = [cid for _, cid in self.mesa]
        return construir_observacion_parcial(
            mano_ids=self.mano,
            mesa_ids=mesa_ids,
            cementerio_ids=self.cementerio,
            vacios_por_jugador=self.vacios,
            puntajes_historicos=self.puntajes_historicos,
            puntos_mano_actual=self.puntos_mano,
            corazones_rotos=self.corazones_rotos,
            dama_picas_en=self.dama_picas_en,
            agente_idx=self.agente_idx,
        )

    # ------------------------------------------------------------
    # Recomendación
    # ------------------------------------------------------------

    def recomendar(
        self,
        modelo,
        vecnorm_path: Optional[str] = None,
    ) -> Optional[int]:
        """Recomienda la mejor carta para jugar usando el modelo RL.

        Args:
            modelo: Modelo MaskablePPO cargado.
            vecnorm_path: Ruta al VecNormalize .pkl.

        Returns:
            ID de la carta recomendada, o None si no hay legales.
        """
        legales = self.legales_humano()
        if not legales:
            return None

        obs = self._construir_observacion()
        obs_norm = normalizar_observacion(obs, vecnorm_path)

        mask = np.zeros(52, dtype=np.bool_)
        for lid in legales:
            mask[lid] = True

        action, _ = modelo.predict(
            obs_norm, action_masks=mask, deterministic=True)
        action_int = int(action.item()) if hasattr(
            action, 'item') else int(action)

        # Garantizar que la acción es legal (fallback)
        if action_int not in legales:
            action_int = legales[0]

        return action_int

    # ------------------------------------------------------------
    # Finalización de mano
    # ------------------------------------------------------------

    def finalizar_mano(self) -> None:
        """Aplica la puntuación de la mano al histórico.

        Reglas:
          - Normal: cada jugador suma los puntos que acumuló en la mano.
          - Pleno (Shooting the Moon): si un jugador acumuló los 26 puntos,
            él suma 0 y los demás suman 26 cada uno.
        """
        puntos_finales = list(self.puntos_mano)

        # Detectar pleno
        for i in range(4):
            if self.puntos_mano[i] == 26:
                self._pleno_jugador = i
                puntos_finales = [26, 26, 26, 26]
                puntos_finales[i] = 0
                break

        for i in range(4):
            self.puntajes_historicos[i] += puntos_finales[i]


# ----------------------------------------------------------------
# Logger de partida (JSONL)
# ----------------------------------------------------------------

class LogPartida:
    """Guarda el historial completo de una partida en formato JSONL.

    Cada línea es un objeto JSON con el estado completo del juego
    en ese momento: timestamp, turno, mano, mesa, puntajes,
    recomendación del modelo, jugada elegida, resultado de baza, etc.

    Usar con: python asesor_partida.py --log partidas/mi_partida.jsonl
    """

    def __init__(self, ruta_log: str) -> None:
        self.ruta_log = ruta_log
        self._archivo = open(ruta_log, "w", encoding="utf-8")
        self._turno = 0
        print(f"📝 Log de partida: {ruta_log}")

    def _escribir(self, entrada: dict) -> None:
        entrada["turno"] = self._turno
        entrada["timestamp"] = datetime.now().isoformat()
        self._archivo.write(json.dumps(entrada, ensure_ascii=False) + "\n")
        self._archivo.flush()
        self._turno += 1

    def log_mano_inicial(self, estado: "EstadoPartida", num_mano: int) -> None:
        self._escribir({
            "evento": "inicio_mano",
            "num_mano": num_mano,
            "mano_inicial": ids_a_nombres(estado.mano),
            "mano_ids": estado.mano,
            "agente_idx": estado.agente_idx,
            "puntajes_historicos": estado.puntajes_historicos,
            "jugador_inicial": estado.jugador_inicial,
        })

    def log_recomendacion(
        self, estado: "EstadoPartida", legales: List[int],
        recomendada: Optional[int], observacion: np.ndarray, num_mano: int,
    ) -> None:
        self._escribir({
            "evento": "turno_humano",
            "num_mano": num_mano,
            "num_baza": estado.num_baza,
            "mano": ids_a_nombres(estado.mano),
            "mesa": [(j, ids_a_nombres([c])[0]) for j, c in estado.mesa],
            "corazones_rotos": estado.corazones_rotos,
            "dama_picas_en": estado.dama_picas_en,
            "puntajes_historicos": estado.puntajes_historicos,
            "puntos_mano": estado.puntos_mano,
            "legales": ids_a_nombres(legales),
            "legales_ids": legales,
            "recomendada": ids_a_nombres([recomendada])[0] if recomendada is not None else None,
            "recomendada_id": recomendada,
            "observacion": observacion.tolist(),
        })

    def log_jugada_humano(self, carta_id: int, estado: "EstadoPartida", num_mano: int) -> None:
        self._escribir({
            "evento": "jugada_humano",
            "num_mano": num_mano,
            "num_baza": estado.num_baza,
            "carta": ids_a_nombres([carta_id])[0],
            "carta_id": carta_id,
        })

    def log_jugada_rival(self, jugador_idx: int, carta_id: int,
                         estado: "EstadoPartida", num_mano: int) -> None:
        self._escribir({
            "evento": "jugada_rival",
            "num_mano": num_mano,
            "num_baza": estado.num_baza,
            "jugador": jugador_idx,
            "carta": ids_a_nombres([carta_id])[0],
            "carta_id": carta_id,
        })

    def log_resultado_baza(self, estado: "EstadoPartida", ganador: int,
                           puntos: int, num_mano: int) -> None:
        self._escribir({
            "evento": "resultado_baza",
            "num_mano": num_mano,
            "num_baza": estado.num_baza - 1,
            "ganador": ganador,
            "puntos_baza": puntos,
            "puntos_mano": estado.puntos_mano,
        })

    def log_fin_mano(self, estado: "EstadoPartida", num_mano: int) -> None:
        self._escribir({
            "evento": "fin_mano",
            "num_mano": num_mano,
            "puntos_mano_final": estado.puntos_mano,
            "puntajes_historicos": estado.puntajes_historicos,
            "pleno_jugador": estado._pleno_jugador,
        })

    def log_fin_partida(self, estado: "EstadoPartida") -> None:
        ranking = sorted(range(4), key=lambda i: estado.puntajes_historicos[i])
        self._escribir({
            "evento": "fin_partida",
            "puntajes_finales": estado.puntajes_historicos,
            "ranking": ranking,
            "posicion_humano": ranking.index(estado.agente_idx) + 1,
        })

    def cerrar(self) -> None:
        self._archivo.close()
        print(f"📝 Log guardado: {self.ruta_log}")


# ----------------------------------------------------------------
# REPL: loop interactivo de partida
# ----------------------------------------------------------------

# Símbolos para mostrar
_SIMBOLO_PALO = {0: "♣/c", 1: "♦/d", 2: "♠/s", 3: "♥/h"}
_PALO_LETRA = {0: "c",  1: "d",  2: "s",  3: "h"}
_NOMBRE_POSICION = {0: "1º (líder)", 1: "2º", 2: "3º", 3: "4º (último)"}


def _preguntar_si_no(texto: str) -> bool:
    """Pregunta sí/no."""
    resp = input(f"{texto} [s/n]: ").strip().lower()
    return resp in ("s", "si", "sí", "y", "yes")


def _pedir_entero(texto: str, minimo: int = 0, maximo: int = 200) -> int:
    """Pide un entero validado."""
    while True:
        try:
            val = int(input(f"{texto}: ").strip())
            if minimo <= val <= maximo:
                return val
            print(f"  ⚠️  Fuera de rango [{minimo}-{maximo}]")
        except ValueError:
            print("  ⚠️  Debe ser un número entero.")


def _pedir_carta(mensaje: str) -> int:
    """Pide una carta al usuario."""
    while True:
        try:
            texto = input(f"{mensaje}: ").strip()
            return parsear_carta(texto)
        except ValueError as e:
            print(f"  ⚠️  {e}")


def _mostrar_estado(estado: EstadoPartida) -> None:
    """Muestra el estado actual de la partida."""
    print()
    print("─" * 58)
    print(
        f"  🃏  BAZA {estado.num_baza} de 13  │  💔 {'SÍ' if estado.corazones_rotos else 'NO'}")
    print("─" * 58)

    # Puntajes
    print(f"  📊 Histórico: ", end="")
    for i in range(4):
        rol = "VOS" if i == estado.agente_idx else f"J{i}"
        print(f"[{rol}:{estado.puntajes_historicos[i]}]", end=" ")
    print()

    # Mesa
    if estado.mesa:
        print(f"  🃏 Mesa: ", end="")
        for j_idx, cid in estado.mesa:
            nombre = ids_a_nombres([cid])[0]
            rol = "VOS" if j_idx == estado.agente_idx else f"J{j_idx}"
            print(f"[{rol}:{nombre}]", end=" ")
        palo_str = _PALO_LETRA.get(
            estado.palo_salida, "?") if estado.palo_salida is not None else "?"
        print(f" (palo: {palo_str})")
    else:
        print(f"  🃏 Mesa: (vacía)")

    # Dama de Picas
    if estado.dama_picas_en is not None:
        rol = "VOS" if estado.dama_picas_en == estado.agente_idx else f"J{estado.dama_picas_en}"
        print(f"  👑 Q♠: {rol}")

    # Vacíos
    for i in range(4):
        if estado.vacios[i]:
            palos_str = "".join(_PALO_LETRA[p]
                                for p in sorted(estado.vacios[i]))
            rol = "VOS" if i == estado.agente_idx else f"J{i}"
            print(f"  📭 Void [{rol}]: {palos_str}")


def _mostrar_mano(estado: EstadoPartida) -> None:
    """Muestra la mano del humano agrupada por palo."""
    print(f"\n  🎴 TU MANO ({len(estado.mano)} cartas):")
    por_palo: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for cid in estado.mano:
        por_palo[_PALOS[cid]].append(cid)
    for palo in range(4):
        if por_palo[palo]:
            nombres = ids_a_nombres(
                sorted(por_palo[palo], key=lambda c: _VALORES[c]))
            print(f"     {_SIMBOLO_PALO[palo]}:  " +
                  "  ".join(f"{n:>4}" for n in nombres))
    print()


def _mostrar_legales(legales: List[int], recomendada: Optional[int] = None) -> None:
    """Muestra las jugadas legales agrupadas por palo."""
    print(f"  ✅ Jugadas legales ({len(legales)}):")
    por_palo: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for cid in legales:
        por_palo[_PALOS[cid]].append(cid)
    for palo in range(4):
        if por_palo[palo]:
            cartas = sorted(por_palo[palo], key=lambda c: _VALORES[c])
            linea = f"     {_SIMBOLO_PALO[palo]}: "
            for cid in cartas:
                nombre = ids_a_nombres([cid])[0]
                if cid == recomendada:
                    linea += f" ⭐{nombre}⭐"
                else:
                    linea += f" {nombre}"
            print(linea)


def _mostrar_resultado_baza(estado: EstadoPartida, ganador: int, puntos: int) -> None:
    """Muestra el resultado de la baza."""
    rol = "VOS" if ganador == estado.agente_idx else f"Jugador {ganador}"
    print(f"\n  🏆 Baza ganada por {rol}. +{puntos} pts.")
    print(f"  📊 Puntos esta mano: ", end="")
    for i in range(4):
        rol = "VOS" if i == estado.agente_idx else f"J{i}"
        print(f"[{rol}:{estado.puntos_mano[i]}]", end=" ")
    print()


def _mostrar_resultado_final(estado: EstadoPartida) -> None:
    """Muestra el resultado final de la partida."""
    print("\n" + "═" * 58)
    print("  🏁  PARTIDA TERMINADA  🏁")
    print("═" * 58)

    ranking = sorted(range(4), key=lambda i: estado.puntajes_historicos[i])
    posiciones = ["🥇 1º", "🥈 2º", "🥉 3º", "💀 4º"]
    for pos, jug_idx in enumerate(ranking):
        rol = "👤 VOS" if jug_idx == estado.agente_idx else f"🤖 J{jug_idx}"
        print(
            f"     {posiciones[pos]}: {rol} — {estado.puntajes_historicos[jug_idx]} pts")

    if ranking[0] == estado.agente_idx:
        print("\n  🎉 ¡GANASTE!")
    else:
        print(
            f"\n  📉 Quedaste en posición {ranking.index(estado.agente_idx) + 1}º.")


# ----------------------------------------------------------------
# Loop principal
# ----------------------------------------------------------------

def ejecutar_partida(
    modelo_path: str,
    vecnorm_path: Optional[str] = None,
    posicion: int = 0,
    log: Optional[LogPartida] = None,
) -> None:
    """Ejecuta el REPL interactivo de partida completa.

    Args:
        modelo_path: Ruta al modelo .zip.
        vecnorm_path: Ruta al VecNormalize .pkl.
        posicion: Posición del humano (0-3).
    """
    # Cargar modelo
    modelo, vecnorm_path = cargar_modelo(modelo_path, vecnorm_path)

    # --- INICIO: ingresar mano ---
    print("\n" + "═" * 58)
    print("  🃏  ASESOR DE PARTIDA — CORAZONES  🃏")
    print("═" * 58)
    print(f"  👤 Tu posición: {posicion} ({_NOMBRE_POSICION[posicion]})")
    print()
    print("  Ingresá tus 13 cartas iniciales.")
    print("  Formato: valor + letra de palo. Ej: Ah 2c Ks Qd 10h Jc")
    print("  Palos: c=Tréboles  d=Diamantes  s=Picas  h=Corazones")
    print("  (Escribí la letra después del valor, sin espacios: 2c, Ah, Ks, Qd)")
    print()

    while True:
        try:
            entrada = input("  🎴 Tu mano (13 cartas): ").strip()
            mano_ids = parsear_mano(entrada, esperadas=13)
            break
        except ValueError as e:
            print(f"  ⚠️  {e}")

    estado = EstadoPartida(mano_inicial=mano_ids, agente_idx=posicion)

    # --- Determinar quién lidera ---
    print()
    if estado._tiene_2_treboles():
        print(f"  ✅ Tenés el 2♣. Liderás la primera baza.")
        estado.iniciar_mano(jugador_inicial=posicion)
    else:
        print(f"  ❌ No tenés el 2♣.")
        lider = _pedir_entero("  ¿Quién lidera la primera baza? (0-3)", 0, 3)
        estado.iniciar_mano(jugador_inicial=lider)

    if log:
        log.log_mano_inicial(estado, num_mano=1)

    # --- LOOP PRINCIPAL: manos ---
    num_mano = 1
    while not estado.juego_terminado():
        print(f"\n{'═' * 58}")
        print(f"  🃏  MANO {num_mano}  🃏")
        print(f"{'═' * 58}")

        # --- LOOP: bazas ---
        while not estado.mano_terminada():
            _mostrar_estado(estado)
            _mostrar_mano(estado)

            # Jugar las 4 posiciones
            while not estado.baza_terminada():
                actual = estado.jugador_actual

                if estado.es_turno_humano():
                    # --- TURNO DEL HUMANO: recomendar ---
                    legales = estado.legales_humano()
                    recomendada = estado.recomendar(modelo, vecnorm_path)

                    if log:
                        obs_actual = estado._construir_observacion()
                        log.log_recomendacion(
                            estado, legales, recomendada, obs_actual, num_mano)

                    _mostrar_legales(legales, recomendada)

                    while True:
                        try:
                            eleccion = input(
                                f"\n  🎯 ¿Qué jugás? "
                                f"(Enter = recomendada {ids_a_nombres([recomendada])[0] if recomendada is not None else '?'}): "
                            ).strip()

                            if eleccion == "" and recomendada is not None:
                                carta_id = recomendada
                            elif eleccion.lower() in ("salir", "exit", "quit", "q"):
                                print("  👋 Partida cancelada.")
                                return
                            else:
                                carta_id = parsear_carta(eleccion)

                            if not estado.puede_jugar_carta(carta_id):
                                print(
                                    f"  ⚠️  {ids_a_nombres([carta_id])[0]} no está en tu mano.")
                                continue

                            if carta_id not in legales:
                                print(
                                    f"  ⚠️  {ids_a_nombres([carta_id])[0]} no es una jugada legal.")
                                continue

                            break
                        except ValueError as e:
                            print(f"  ⚠️  {e}")

                    estado.registrar_jugada(estado.agente_idx, carta_id)
                    print(f"  ✅ Jugaste {ids_a_nombres([carta_id])[0]}.")
                    if log:
                        log.log_jugada_humano(carta_id, estado, num_mano)

                else:
                    # --- TURNO DE OTRO JUGADOR: preguntar ---
                    print(f"\n  🔄 Turno del Jugador {actual}...")
                    while True:
                        try:
                            eleccion = input(
                                f"  ¿Qué jugó J{actual}?: ").strip()
                            if eleccion.lower() in ("salir", "exit", "quit", "q"):
                                print("  👋 Partida cancelada.")
                                return
                            carta_id = parsear_carta(eleccion)
                            break
                        except ValueError as e:
                            print(f"  ⚠️  {e}")

                    estado.registrar_jugada(actual, carta_id)

                    if log:
                        log.log_jugada_rival(
                            actual, carta_id, estado, num_mano)

            # --- RESOLVER BAZA ---
            puntos_antes = list(estado.puntos_mano)  # copia antes de resolver
            ganador = estado.resolver_baza()
            puntos_baza = sum(estado.puntos_mano) - sum(puntos_antes)
            _mostrar_resultado_baza(estado, ganador, puntos_baza)
            if log:
                log.log_resultado_baza(estado, ganador, puntos_baza, num_mano)

        # --- FIN DE MANO ---
        estado.finalizar_mano()

        if log:
            log.log_fin_mano(estado, num_mano)

        print(f"\n  📊 Puntuación después de la mano {num_mano}:")
        for i in range(4):
            rol = "VOS" if i == estado.agente_idx else f"J{i}"
            print(f"     [{rol}]: {estado.puntajes_historicos[i]} pts")

        if estado.juego_terminado():
            break

        # --- NUEVA MANO ---
        num_mano += 1
        print(f"\n  🔄 Nueva mano ({num_mano})...")
        print("  Ingresá tus NUEVAS 13 cartas:")

        while True:
            try:
                entrada = input("  🎴 Tu mano: ").strip()
                mano_ids = parsear_mano(entrada, esperadas=13)
                break
            except ValueError as e:
                print(f"  ⚠️  {e}")

        estado.mano = sorted(mano_ids)

        if estado._tiene_2_treboles():
            print(f"  ✅ Tenés el 2♣. Liderás.")
            estado.iniciar_mano(jugador_inicial=posicion)
        else:
            print(f"  ❌ No tenés el 2♣.")
            lider = _pedir_entero("  ¿Quién lidera? (0-3)", 0, 3)
            estado.iniciar_mano(jugador_inicial=lider)

        if log:
            log.log_mano_inicial(estado, num_mano)

    # --- FIN DE PARTIDA ---
    if log:
        log.log_fin_partida(estado)
        log.cerrar()

    _mostrar_resultado_final(estado)


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asesor de partida completa para Corazones — tracking stateful",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python asesor_partida.py
  python asesor_partida.py --modelo modelos_historicos/v2/modelo_final
  python asesor_partida.py --posicion 2
        """,
    )
    parser.add_argument(
        "--modelo", type=str,
        default="modelos_historicos/v5/snapshot_0007200000",
        help="Ruta al modelo .zip"
    )
    parser.add_argument(
        "--vecnorm", type=str, default=None,
        help="Ruta al VecNormalize .pkl (auto-detecta)"
    )
    parser.add_argument(
        "--posicion", type=int, default=0,
        help="Tu posición en la mesa (0-3, default=0)"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Ruta para guardar el log de la partida en formato JSONL"
    )

    args = parser.parse_args()

    # Auto-detectar VecNormalize
    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        # Buscar en orden: per-snapshot, v5 global, v2 legacy
        ruta_limpia = args.modelo.replace(".zip", "")
        for candidato in [
            ruta_limpia + "_vecnorm.pkl",
            "vecnormalize/v5/v5_vecnorm.pkl",
            "vecnormalize/v5/v5_vecnorm_final.pkl",
            "vecnormalize/v2_vecnorm_final.pkl",
            "vecnormalize/vecnorm.pkl",
        ]:
            if os.path.exists(candidato):
                vecnorm_path = candidato
                break

    # Crear logger si se solicitó
    log = None
    if args.log:
        log = LogPartida(args.log)

    ejecutar_partida(
        modelo_path=args.modelo,
        vecnorm_path=vecnorm_path,
        posicion=args.posicion,
        log=log,
    )


if __name__ == "__main__":
    main()
