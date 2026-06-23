"""
Tests unitarios para features de liderazgo en ObservacionBuilderV3.

El PIMC identificó "LIDERAR mal" como el error #1 (25-39% óptimo).
Estas features ayudan al agente a decidir correctamente cuando lidera.

Features nuevas [250:260]:
    [250] soy_lider — 1.0 si la mesa está vacía (agente lidera esta baza)
    [251] lidero_picas_forzado — 1.0 si debo liderar y solo tengo ♠
    [252:256] maxima_absoluta_palo — 1.0 si tengo la carta más alta viva de ese palo
    [256:260] puedo_quemar_palo — 1.0 si liderar mi máxima del palo me da baza limpia

Estrategia TDD:
    RED   → Escribir tests primero (deben fallar).
    GREEN → Implementar features mínimas para que pasen.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_V10
from src.v3.observacion import ObservacionBuilderV3, DIM_V3

_TREBOL, _DIAMANTE, _PICA, _CORAZON = 0, 1, 2, 3


# ─── Helpers ──────────────────────────────────────────────────────────────

def _carta(palo: int, valor: int) -> Carta:
    """Crea una carta por palo y valor."""
    return Carta(palo, valor)


def _crear_motor(
    cartas_j0: list[Carta],
    cartas_j1: list[Carta] | None = None,
    cartas_j2: list[Carta] | None = None,
    cartas_j3: list[Carta] | None = None,
    *,
    baza: int = 1,
    mesa_vacia: bool = True,
) -> MotorCorazones:
    """Crea motor con manos predefinidas."""
    m = MotorCorazones()
    m.jugadores[0].mano = list(cartas_j0)
    m.jugadores[1].mano = list(cartas_j1 or [])
    m.jugadores[2].mano = list(cartas_j2 or [])
    m.jugadores[3].mano = list(cartas_j3 or [])
    m.numero_baza = baza
    m.indice_jugador_inicial = 0
    m._mano_activa = True
    if mesa_vacia:
        m.mesa = []
        m.palo_de_salida = None
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Tests: soy_lider [250]
# ═══════════════════════════════════════════════════════════════════════════


class TestSoyLider:
    """Feature [250]: indica si el agente es quien lidera la baza."""

    def test_soy_lider_mesa_vacia(self) -> None:
        """Cuando la mesa está vacía, soy_lider = 1.0."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(0, 2)], baza=1, mesa_vacia=True)

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert obs[250] == 1.0, (
            f"soy_lider debe ser 1.0 con mesa vacía, es {obs[250]}"
        )

    def test_no_soy_lider_con_cartas_en_mesa(self) -> None:
        """Cuando ya hay cartas en la mesa, soy_lider = 0.0."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(0, 3)], baza=3, mesa_vacia=False)
        m.mesa = [(1, _carta(0, 2))]
        m.palo_de_salida = 0

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert obs[250] == 0.0, (
            f"soy_lider debe ser 0.0 con mesa ocupada, es {obs[250]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: lidero_picas_forzado [251]
# ═══════════════════════════════════════════════════════════════════════════


class TestLideroPicasForzado:
    """Feature [251]: 1.0 si el agente debe liderar y solo tiene ♠."""

    def test_lidero_picas_forzado_cuando_solo_tengo_picas(self) -> None:
        """Si debo liderar y solo tengo ♠ → 1.0."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(2, 5), _carta(2, 7)], baza=1)

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert obs[251] == 1.0, (
            f"lidero_picas_forzado debe ser 1.0 con solo ♠, es {obs[251]}"
        )

    def test_no_forzado_si_tengo_otro_palo(self) -> None:
        """Si tengo ♠ y otro palo → 0.0 (no estoy forzado)."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(2, 5), _carta(0, 3)], baza=1)

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert obs[251] == 0.0, (
            f"lidero_picas_forzado debe ser 0.0 con otro palo, es {obs[251]}"
        )

    def test_no_forzado_si_no_soy_lider(self) -> None:
        """Si no soy líder, lidero_picas_forzado = 0.0 aunque solo tenga ♠."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(2, 5)], baza=3, mesa_vacia=False)
        m.mesa = [(1, _carta(0, 2))]
        m.palo_de_salida = 0

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert obs[251] == 0.0, (
            f"lidero_picas_forzado debe ser 0.0 si no lidero, es {obs[251]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: maxima_absoluta_palo [252:256]
# ═══════════════════════════════════════════════════════════════════════════


class TestMaximaAbsolutaPalo:
    """Features [252:256]: 1.0 si tengo la carta más alta viva de ese palo."""

    def test_tengo_maxima_de_trebol(self) -> None:
        """Con A♣ en mano y sin cartas jugadas → soy máxima de ♣."""
        builder = ObservacionBuilderV3()
        # A♣ = palo 0, valor 14
        m = _crear_motor([_carta(0, 14), _carta(1, 3)])

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert obs[252] == 1.0, (
            f"maxima_absoluta ♣ debe ser 1.0 con A♣, es {obs[252]}"
        )

    def test_no_soy_maxima_de_picas(self) -> None:
        """Con 10♠ en mano, no soy máxima (A♠ en rival)."""
        builder = ObservacionBuilderV3()
        m = _crear_motor(
            [_carta(2, 10), _carta(1, 3)],  # J0: 10♠, 3♦
            [_carta(2, 14)],                 # J1: A♠ (más alta)
        )

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert obs[254] == 0.0, (
            f"maxima_absoluta ♠ debe ser 0.0 con A♠ en rival, es {obs[254]}"
        )

    def test_maxima_por_descarte(self) -> None:
        """Si A♣ fue jugada y tengo K♣ → ahora soy máxima de ♣."""
        builder = ObservacionBuilderV3()
        # A♣ ya jugada (en bazas de J1), K♣ en mano de J0
        m = _crear_motor([_carta(0, 13), _carta(1, 3)])  # K♣, 3♦
        m.jugadores[1].bazas_ganadas = [_carta(0, 14)]  # A♣ ya jugada

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert obs[252] == 1.0, (
            f"maxima_absoluta ♣ debe ser 1.0 con K♣ y A♣ jugada, es {obs[252]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: puedo_quemar_palo [256:260]
# ═══════════════════════════════════════════════════════════════════════════


class TestPuedoQuemarPalo:
    """Features [256:260]: 1.0 si liderar mi máxima me da baza limpia (0 pts)."""

    def test_quemar_trebol_con_maxima(self) -> None:
        """Con A♣ y sin Q♠ activa → puedo quemar ♣ (baza limpia)."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(0, 14), _carta(1, 3)])

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=1,  # Q♠ capturada → ♠ libre
        )

        assert obs[256] == 1.0, (
            f"puedo_quemar ♣ debe ser 1.0 con A♣, es {obs[256]}"
        )

    def test_no_quemar_sin_maxima(self) -> None:
        """Sin la máxima del palo (rival tiene A♣) → no puedo garantizar baza limpia."""
        builder = ObservacionBuilderV3()
        m = _crear_motor(
            [_carta(0, 5), _carta(1, 3)],  # J0: 5♣, 3♦
            [_carta(0, 14)],                 # J1: A♣ (más alta)
        )

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert obs[256] == 0.0, (
            f"puedo_quemar ♣ debe ser 0.0 sin máxima, es {obs[256]}"
        )

    def test_no_quemar_picas_si_qs_activa(self) -> None:
        """Con máxima de ♠ pero Q♠ activa → no puedo quemar ♠ (peligro)."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(2, 14), _carta(1, 3)])  # A♠, 3♦

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,  # Q♠ activa
        )

        assert obs[258] == 0.0, (
            f"puedo_quemar ♠ debe ser 0.0 con Q♠ activa, es {obs[258]}"
        )

    def test_quemar_picas_si_qs_capturada(self) -> None:
        """Con máxima de ♠ y Q♠ ya capturada → sí puedo quemar ♠."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(2, 14), _carta(1, 3)])

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=0,  # Yo mismo capturé Q♠
        )

        assert obs[258] == 1.0, (
            f"puedo_quemar ♠ debe ser 1.0 con Q♠ capturada, es {obs[258]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Tests: Integración
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegracionLiderazgo:
    """Verifica que las features de liderazgo no rompan nada existente."""

    def test_dims_correctas(self) -> None:
        """El vector debe tener exactamente 260 dimensiones."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(0, 2)])

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        assert len(obs) == 265, (
            f"Observación debe tener 265 dims, tiene {len(obs)}"
        )

    def test_features_base_intactas(self) -> None:
        """Las features base [0:250] no deben alterarse."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(0, 2)])

        obs = builder.construir(
            m, agente_idx=0,
            vacios=[set() for _ in range(4)],
            puntuacion_historica=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            dama_picas_en=None,
        )

        # Verificar mano one-hot: J0 tiene 2♣ → id=0 → obs[0] = 1.0
        assert obs[0] == 1.0, (
            f"Base [0] (mano one-hot) debe ser 1.0, es {obs[0]}"
        )

    def test_construir_desde_motor_rellena_ceros(self) -> None:
        """construir_desde_motor debe dejar [250:260] en 0."""
        builder = ObservacionBuilderV3()
        m = _crear_motor([_carta(0, 2)])

        obs = builder.construir_desde_motor(m, 0)

        assert np.all(obs[250:260] == 0.0), (
            f"construir_desde_motor debe dejar [250:260] en 0, "
            f"valores: {obs[250:260]}"
        )
