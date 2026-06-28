"""
Motor del juego de Corazones (Headless Environment).

Implementa el ciclo de vida completo de una mano: reparto, bazas,
validación estricta de jugadas legales con 4 filtros en cascada,
gatillos de estado (corazones_rotos), resolución de bazas, detección
de Pleno (Shooting the Moon) y cálculo de puntuación.

Utiliza arrays precomputados (indexados por carta.id) para maximizar
el rendimiento de obtener_jugadas_legales() en el hot path.

No tiene dependencias de IA ni de Gymnasium.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple
from src.dominio.carta import Carta, _PALOS, _PUNTOS, _ES_CORAZON
from src.dominio.baraja import Baraja
from src.dominio.jugador import Jugador


class MotorCorazones:
    """Motor headless del juego de Corazones con validación estricta de reglas."""

    def __init__(self) -> None:
        self.jugadores: List[Jugador] = [
            Jugador("Norte"), Jugador("Este"), Jugador(
                "Sur"), Jugador("Oeste"),
        ]
        self.baraja: Baraja = Baraja()
        self.corazones_rotos: bool = False
        self.numero_baza: int = 0
        # Número de mano dentro de la PARTIDA (para la rotación del pase).
        self.numero_mano: int = 0
        self.mesa: List[Tuple[int, Carta]] = []
        self.palo_de_salida: Optional[int] = None
        self.indice_jugador_inicial: int = 0
        self._mano_activa: bool = False

    def repartir(self) -> None:
        """Baraja y reparte 13 cartas a cada jugador. Reinicia el estado de la mano."""
        self.baraja = Baraja()
        self.baraja.repartir(self.jugadores)
        for jug in self.jugadores:
            jug.bazas_ganadas = []
        self.corazones_rotos = False
        self.numero_baza = 1
        self.numero_mano += 1
        self.mesa = []
        self.palo_de_salida = None
        self._mano_activa = True
        self._fijar_jugador_inicial()

    def _fijar_jugador_inicial(self) -> None:
        """Fija el jugador inicial como el portador del 2♣ (tras reparto o pase)."""
        for i, jug in enumerate(self.jugadores):
            if any(c.es_dos_de_treboles for c in jug.mano):
                self.indice_jugador_inicial = i
                return
        raise RuntimeError(
            "Ningún jugador tiene el 2 de Tréboles.")

    # ------------------------------------------------------------------
    # El Pase (passing) — rotación izquierda/derecha/enfrente/sin pase
    # ------------------------------------------------------------------

    def direccion_pase(self) -> Optional[str]:
        """Dirección del pase para la mano actual según la rotación estándar.

        Ciclo por número de mano: 1→izquierda, 2→derecha, 3→enfrente, 4→sin pase.
        Retorna None si esta mano no tiene pase.
        """
        if self.numero_mano <= 0:
            return None
        fase = (self.numero_mano - 1) % 4
        return {0: "izquierda", 1: "derecha", 2: "enfrente", 3: None}[fase]

    # Offset de asiento del receptor según dirección (orden de juego = +1 = izquierda).
    _OFFSET_PASE = {"izquierda": 1, "derecha": 3, "enfrente": 2}

    def receptor_pase(self, jugador_idx: int) -> Optional[int]:
        """Índice del jugador que recibe el pase de `jugador_idx` (o None si no hay pase)."""
        direccion = self.direccion_pase()
        if direccion is None:
            return None
        return (jugador_idx + self._OFFSET_PASE[direccion]) % 4

    def ejecutar_pase(self, selecciones: dict) -> None:
        """Ejecuta el intercambio de 3 cartas y recalcula el portador del 2♣.

        Args:
            selecciones: {jugador_idx: [3 Cartas]} — las cartas que cada jugador pasa.

        El intercambio es simultáneo. Tras él, el jugador inicial se recalcula
        porque el 2♣ pudo haber cambiado de manos.
        """
        direccion = self.direccion_pase()
        if direccion is None:
            return  # mano sin pase
        offset = self._OFFSET_PASE[direccion]

        # 1) Retirar las cartas salientes de cada mano (simultáneo).
        for idx, cartas in selecciones.items():
            if len(cartas) != 3:
                raise ValueError(f"El jugador {idx} debe pasar exactamente 3 cartas.")
            for c in cartas:
                self.jugadores[idx].mano.remove(c)

        # 2) Entregar a los receptores.
        for idx, cartas in selecciones.items():
            receptor = (idx + offset) % 4
            self.jugadores[receptor].mano.extend(cartas)

        # 3) El portador del 2♣ pudo cambiar → recalcular quién abre.
        self._fijar_jugador_inicial()

    def obtener_jugador_actual(self) -> int:
        """Retorna el índice del jugador que debe jugar en este momento."""
        return (self.indice_jugador_inicial + len(self.mesa)) % 4

    def obtener_jugadas_legales(self, jugador_idx: int) -> List[Carta]:
        """Determina el subconjunto de cartas legales para un jugador.

        Aplica 4 filtros en cascada según las reglas estrictas de Corazones:
            1. Salida inicial: solo 2♣ en la primera baza con mesa vacía.
            2. Seguir el palo: si hay cartas en la mesa, debe asistir al palo de salida.
            3. Primera baza segura: en la baza 1 no se pueden jugar cartas con puntos.
            4. Liderar corazones: no se puede abrir con corazones si no están rotos.
        """
        mano = self.jugadores[jugador_idx].mano
        if not mano:
            return []

        # Filtro 1: Salida Inicial (2 de Tréboles obligatorio)
        if self.numero_baza == 1 and not self.mesa:
            for c in mano:
                if c.id == 0:
                    return [c]
            return list(mano)

        # Filtro 3: Asistir al Palo
        palo_salida = self.palo_de_salida
        if self.mesa and palo_salida is not None:
            mismo_palo = [c for c in mano if _PALOS[c.id] == palo_salida]
            legales = mismo_palo if mismo_palo else list(mano)
        else:
            legales = list(mano)

        # Filtro 2: Primera Baza Segura
        if self.numero_baza == 1:
            sin_puntos = [c for c in legales if _PUNTOS[c.id] == 0]
            if sin_puntos:
                legales = sin_puntos

        # Filtro 4: Liderar con Corazones
        if not self.mesa and not self.corazones_rotos:
            sin_corazones = [c for c in legales if not _ES_CORAZON[c.id]]
            if sin_corazones:
                legales = sin_corazones

        return legales

    def jugar_carta(self, jugador_idx: int, carta: Carta) -> None:
        """Valida y ejecuta la jugada de una carta, actualizando el estado."""
        legales = self.obtener_jugadas_legales(jugador_idx)
        if not any(c == carta for c in legales):
            raise ValueError(
                f"Jugada ilegal: {carta} no está en las jugadas legales "
                f"para el jugador {self.jugadores[jugador_idx].nombre}."
            )
        if not self.mesa:
            self.palo_de_salida = carta.palo
        self.jugadores[jugador_idx].jugar_carta(carta)
        self.mesa.append((jugador_idx, carta))
        if carta.es_corazon and not self.corazones_rotos:
            self.corazones_rotos = True

    def resolver_baza(self) -> int:
        """Determina el ganador de la baza actual y le asigna las cartas."""
        if len(self.mesa) != 4:
            raise RuntimeError(
                f"No se puede resolver una baza con {len(self.mesa)} cartas.")
        ganador_idx = self.mesa[0][0]
        carta_mas_alta = self.mesa[0][1]
        for jugador_idx, carta in self.mesa[1:]:
            if carta.palo == self.palo_de_salida and carta.valor > carta_mas_alta.valor:
                carta_mas_alta = carta
                ganador_idx = jugador_idx
        cartas_baza = [c for _, c in self.mesa]
        self.jugadores[ganador_idx].bazas_ganadas.extend(cartas_baza)
        self.indice_jugador_inicial = ganador_idx
        self.mesa = []
        self.palo_de_salida = None
        self.numero_baza += 1
        return ganador_idx

    def calcular_puntuacion_mano(self) -> List[int]:
        """Calcula la puntuación de la mano actual aplicando regla de Pleno."""
        puntos_crudos = [j.contar_puntos_bazas() for j in self.jugadores]
        for i, pts in enumerate(puntos_crudos):
            if pts == 26:
                resultado = [26] * 4
                resultado[i] = 0
                return resultado
        return puntos_crudos

    def aplicar_puntuacion(self) -> List[int]:
        """Calcula y aplica la puntuación de la mano a los historiales."""
        puntuaciones = self.calcular_puntuacion_mano()
        for jugador, pts in zip(self.jugadores, puntuaciones):
            jugador.sumar_puntos(pts)
        return puntuaciones

    # ------------------------------------------------------------------
    # Nivel de partida (múltiples manos hasta el límite de puntos)
    # ------------------------------------------------------------------

    LIMITE_PARTIDA: int = 100

    def nueva_partida(self) -> None:
        """Inicia una partida completa: reinicia el marcador acumulado y reparte.

        A diferencia de `repartir()` (que solo prepara una mano nueva conservando
        el marcador histórico), esto pone a cero la puntuación de los 4 jugadores.
        Usar al comienzo de cada episodio de partida completa.
        """
        for jug in self.jugadores:
            jug.puntuacion_historica = 0
        self.numero_mano = 0  # repartir() lo pone en 1 (primera mano de la partida)
        self.repartir()

    def puntuaciones_historicas(self) -> List[int]:
        """Marcador acumulado de la partida (una entrada por jugador)."""
        return [j.puntuacion_historica for j in self.jugadores]

    def partida_terminada(self, limite: Optional[int] = None) -> bool:
        """True si algún jugador alcanzó/superó el límite (fin de partida)."""
        tope = self.LIMITE_PARTIDA if limite is None else limite
        return any(j.puntuacion_historica >= tope for j in self.jugadores)

    def ranking_partida(self) -> List[int]:
        """Índices de jugadores ordenados de menor a mayor puntuación.

        El primer elemento es el ganador (menor puntuación). Los empates se
        resuelven de forma estable por índice de jugador.
        """
        return sorted(range(4), key=lambda i: self.jugadores[i].puntuacion_historica)

    def jugar_mano(self, selector_cartas: Optional[Callable] = None) -> List[int]:
        """Ejecuta una mano completa de 13 bazas."""
        self.repartir()
        for _ in range(13):
            for _ in range(4):
                idx = self.obtener_jugador_actual()
                legales = self.obtener_jugadas_legales(idx)
                if selector_cartas is not None:
                    carta = selector_cartas(self, idx, legales)
                else:
                    carta = random.choice(legales)
                self.jugar_carta(idx, carta)
            self.resolver_baza()
        self._mano_activa = False
        return self.aplicar_puntuacion()


__all__ = ["MotorCorazones"]
