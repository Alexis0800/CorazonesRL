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
from src.carta import Carta, _PALOS, _PUNTOS, _ES_CORAZON, _ES_DOS_TREBOL
from src.baraja import Baraja
from src.jugador import Jugador


class MotorCorazones:
    """Motor headless del juego de Corazones con validación estricta de reglas."""

    def __init__(self) -> None:
        self.jugadores: List[Jugador] = [
            Jugador("Norte"),
            Jugador("Este"),
            Jugador("Sur"),
            Jugador("Oeste"),
        ]
        self.baraja: Baraja = Baraja()
        self.corazones_rotos: bool = False
        self.numero_baza: int = 0
        self.mesa: List[Tuple[int, Carta]] = []
        self.palo_de_salida: Optional[int] = None
        self.indice_jugador_inicial: int = 0
        self._mano_activa: bool = False

    # ----------------------------------------------------------------
    # Inicialización y reparto
    # ----------------------------------------------------------------

    def repartir(self) -> None:
        """Baraja y reparte 13 cartas a cada jugador. Reinicia el estado de la mano.

        Postcondición:
            - Las 52 cartas están distribuidas (13 por jugador).
            - corazones_rotos = False.
            - numero_baza = 1.
            - mesa vacía.
            - El jugador con el 2♣ es el que abre la primera baza.
        """
        self.baraja = Baraja()
        self.baraja.repartir(self.jugadores)
        self.corazones_rotos = False
        self.numero_baza = 1
        self.mesa = []
        self.palo_de_salida = None
        self._mano_activa = True

        # Buscar quién tiene el 2 de Tréboles para que abra
        for i, jug in enumerate(self.jugadores):
            if any(c.es_dos_de_treboles for c in jug.mano):
                self.indice_jugador_inicial = i
                return

        raise RuntimeError(
            "Ningún jugador tiene el 2 de Tréboles tras el reparto.")

    # ----------------------------------------------------------------
    # Consulta de estado
    # ----------------------------------------------------------------

    def obtener_jugador_actual(self) -> int:
        """Retorna el índice del jugador que debe jugar en este momento.

        Returns:
            Índice (0-3) del jugador al que le toca jugar.
        """
        return (self.indice_jugador_inicial + len(self.mesa)) % 4

    # ----------------------------------------------------------------
    # Validación de jugadas legales (Action Masking lógico)
    # Utiliza arrays precomputados indexados por carta.id para máximo
    # rendimiento en el hot path.
    # ----------------------------------------------------------------

    def obtener_jugadas_legales(self, jugador_idx: int) -> List[Carta]:
        """Determina el subconjunto de cartas legales para un jugador.

        Aplica 4 filtros en cascada según las reglas estrictas de Corazones:
            1. Salida inicial: solo 2♣ en la primera baza con mesa vacía.
            2. Seguir el palo: si hay cartas en la mesa, debe asistir al palo de salida.
            3. Primera baza segura: en la baza 1 no se pueden jugar cartas con puntos.
            4. Liderar corazones: no se puede abrir con corazones si no están rotos.

        Args:
            jugador_idx: Índice del jugador (0-3).

        Returns:
            Lista de cartas que el jugador puede jugar legalmente.
            Nunca retorna una lista vacía.
        """
        mano = self.jugadores[jugador_idx].mano
        if not mano:
            return []

        # --- Filtro 1: Salida Inicial (2 de Tréboles obligatorio) ---
        if self.numero_baza == 1 and not self.mesa:
            # Carta(id=0) es el 2♣
            for c in mano:
                if c.id == 0:
                    return [c]
            return list(mano)  # no debería ocurrir, pero por seguridad

        # --- Filtro 3: Asistir al Palo (seguir el palo de salida) ---
        palo_salida = self.palo_de_salida
        if self.mesa and palo_salida is not None:
            # Filtrar usando array precomputado _PALOS
            mismo_palo = [c for c in mano if _PALOS[c.id] == palo_salida]
            if mismo_palo:
                legales = mismo_palo
            else:
                legales = list(mano)  # void
        else:
            legales = list(mano)

        # --- Filtro 2: Primera Baza Segura (sin puntos en baza 1) ---
        if self.numero_baza == 1:
            sin_puntos = [c for c in legales if _PUNTOS[c.id] == 0]
            if sin_puntos:
                legales = sin_puntos

        # --- Filtro 4: Liderar con Corazones ---
        if not self.mesa and not self.corazones_rotos:
            sin_corazones = [c for c in legales if not _ES_CORAZON[c.id]]
            if sin_corazones:
                legales = sin_corazones

        return legales

    # ----------------------------------------------------------------
    # Ejecución de jugadas
    # ----------------------------------------------------------------

    def jugar_carta(self, jugador_idx: int, carta: Carta) -> None:
        """Valida y ejecuta la jugada de una carta, actualizando el estado.

        Args:
            jugador_idx: Índice del jugador que juega (0-3).
            carta: Carta a jugar.

        Raises:
            ValueError: Si la carta no está en la lista de jugadas legales.
        """
        legales = self.obtener_jugadas_legales(jugador_idx)
        if not any(c == carta for c in legales):
            raise ValueError(
                f"Jugada ilegal: {carta} no está en las jugadas legales {legales} "
                f"para el jugador {self.jugadores[jugador_idx].nombre}."
            )

        # Establecer palo de salida si es la primera carta de la baza
        if not self.mesa:
            self.palo_de_salida = carta.palo

        # Remover la carta de la mano del jugador
        self.jugadores[jugador_idx].jugar_carta(carta)
        self.mesa.append((jugador_idx, carta))

        # --- Gatillo: Romper Corazones ---
        if carta.es_corazon and not self.corazones_rotos:
            self.corazones_rotos = True

    # ----------------------------------------------------------------
    # Resolución de baza
    # ----------------------------------------------------------------

    def resolver_baza(self) -> int:
        """Determina el ganador de la baza actual y le asigna las cartas.

        El ganador es quien jugó la carta más alta del palo de salida.
        Las cartas de otros palos no compiten.

        Returns:
            Índice del jugador que ganó la baza.

        Raises:
            RuntimeError: Si la baza no tiene 4 cartas.
        """
        if len(self.mesa) != 4:
            raise RuntimeError(
                f"No se puede resolver una baza con {len(self.mesa)} cartas."
            )

        ganador_idx = self.mesa[0][0]
        carta_mas_alta = self.mesa[0][1]

        for jugador_idx, carta in self.mesa[1:]:
            if carta.palo == self.palo_de_salida and carta.valor > carta_mas_alta.valor:
                carta_mas_alta = carta
                ganador_idx = jugador_idx

        # Asignar las cartas de la baza al ganador
        cartas_baza = [c for _, c in self.mesa]
        self.jugadores[ganador_idx].bazas_ganadas.extend(cartas_baza)

        # Preparar para la siguiente baza
        self.indice_jugador_inicial = ganador_idx
        self.mesa = []
        self.palo_de_salida = None
        self.numero_baza += 1

        return ganador_idx

    # ----------------------------------------------------------------
    # Puntuación
    # ----------------------------------------------------------------

    def calcular_puntuacion_mano(self) -> List[int]:
        """Calcula la puntuación de la mano actual aplicando regla de Pleno.

        Si un jugador acumuló los 26 puntos (Shooting the Moon / Pleno),
        ese jugador suma 0 y los otros 3 suman 26 cada uno.
        En caso contrario, cada jugador suma los puntos de sus bazas.

        Returns:
            Lista de 4 enteros con los puntos de esta mano para cada jugador.
        """
        puntos_crudos = [j.contar_puntos_bazas() for j in self.jugadores]

        # Verificar Pleno (Shooting the Moon)
        for i, pts in enumerate(puntos_crudos):
            if pts == 26:
                # Pleno detectado: el jugador i suma 0, los demás 26
                resultado = [26] * 4
                resultado[i] = 0
                return resultado

        return puntos_crudos

    def aplicar_puntuacion(self) -> List[int]:
        """Calcula y aplica la puntuación de la mano a los historiales.

        Returns:
            Lista de 4 enteros con los puntos aplicados a cada jugador.
        """
        puntuaciones = self.calcular_puntuacion_mano()
        for jugador, pts in zip(self.jugadores, puntuaciones):
            jugador.sumar_puntos(pts)
        return puntuaciones

    # ----------------------------------------------------------------
    # Ciclo completo de mano
    # ----------------------------------------------------------------

    def jugar_mano(
        self,
        selector_cartas: Optional[Callable] = None,
    ) -> List[int]:
        """Ejecuta una mano completa de 13 bazas.

        Args:
            selector_cartas: Callable opcional que recibe (motor, jugador_idx, legales)
                y retorna la Carta a jugar. Si es None, se usa selección aleatoria.

        Returns:
            Lista de 4 enteros con los puntos de esta mano para cada jugador.
        """
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
