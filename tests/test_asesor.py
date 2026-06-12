"""
Tests unitarios para asesor_carta.py — parseo de cartas, cómputo de legales,
construcción de observación parcial para el modelo.

No depende de PyTorch ni de stable-baselines3.
"""

from asesor_carta import (
    parsear_carta,
    parsear_mano,
    cartas_a_ids,
    ids_a_nombres,
    _PALO_NOMBRE,
    _VALOR_NOMBRE,
    compute_legales,
    construir_observacion_parcial,
    NOMBRES_PALOS,
)
import pytest
import numpy as np
import sys
import os

# Asegurar que src/ está en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================
# parsear_carta
# ================================================================

class TestParsearCarta:
    """Pruebas para el parseo de nombres de carta a ID."""

    def test_parsear_as_corazones(self) -> None:
        assert parsear_carta("A♥") == 51  # palo 3, valor 14

    def test_parsear_dos_treboles(self) -> None:
        assert parsear_carta("2♣") == 0  # palo 0, valor 2

    def test_parsear_dama_picas(self) -> None:
        assert parsear_carta("Q♠") == 36  # palo 2, valor 12

    def test_parsear_rey_diamantes(self) -> None:
        assert parsear_carta("K♦") == 24  # palo 1, valor 13

    def test_parsear_diez_corazones(self) -> None:
        assert parsear_carta("10♥") == 3 * 13 + (10 - 2)  # 47

    def test_parsear_jota_treboles(self) -> None:
        assert parsear_carta("J♣") == 11 - 2  # 9

    def test_parsear_sin_simbolo(self) -> None:
        """Debe lanzar ValueError si falta el símbolo de palo."""
        with pytest.raises(ValueError, match="Símbolo de palo desconocido"):
            parsear_carta("A")

    def test_parsear_palo_invalido(self) -> None:
        with pytest.raises(ValueError, match="Símbolo de palo desconocido"):
            parsear_carta("A★")

    def test_parsear_valor_invalido(self) -> None:
        with pytest.raises(ValueError, match="Valor desconocido"):
            parsear_carta("1♥")

    def test_parsear_vacio(self) -> None:
        with pytest.raises(ValueError, match="vacía"):
            parsear_carta("")


class TestParsearMano:
    """Pruebas para parsear una mano completa de cartas."""

    def test_parsear_13_cartas(self) -> None:
        mano = "2♣ 3♣ 4♣ 5♣ 6♣ 7♣ 8♣ 9♣ 10♣ J♣ Q♣ K♣ A♣"
        ids = parsear_mano(mano)
        assert len(ids) == 13
        assert sorted(ids) == list(range(13))  # todos tréboles

    def test_parsear_con_comas(self) -> None:
        mano = "2♣,3♣,4♣,5♣,6♣,7♣,8♣,9♣,10♣,J♣,Q♣,K♣,A♣"
        ids = parsear_mano(mano, sep=",")
        assert len(ids) == 13

    def test_parsear_cantidad_invalida(self) -> None:
        with pytest.raises(ValueError, match="13 cartas"):
            parsear_mano("2♣ 3♣ 4♣")


class TestCartasAIds:
    """Pruebas para conversión de lista de nombres a IDs."""

    def test_lista_vacia(self) -> None:
        assert cartas_a_ids([]) == []

    def test_varias_cartas(self) -> None:
        assert cartas_a_ids(["A♥", "2♣"]) == [51, 0]


class TestIdsANombres:
    """Pruebas para conversión de IDs a nombres legibles."""

    def test_ids_validos(self) -> None:
        nombres = ids_a_nombres([0, 13, 26, 39])
        assert nombres == ["2c", "2d", "2s", "2h"]

    def test_lista_vacia(self) -> None:
        assert ids_a_nombres([]) == []


# ================================================================
# compute_legales
# ================================================================

class TestComputeLegales:
    """Pruebas para el cómputo de jugadas legales."""

    def test_liderar_sin_corazones_rotos(self) -> None:
        """Al liderar sin corazones rotos, no se puede jugar corazones
        a menos que sea la única opción."""
        mano = [0, 1, 2, 13, 39]  # 2♣ 3♣ 4♣ 2♦ 2♥
        legales = compute_legales(
            mano=mano,
            mesa_ids=[],
            corazones_rotos=False,
            es_primera_baza=False,
        )
        # 2♥ (39) NO debería ser legal
        assert 39 not in legales
        # Cartas de trébol y diamante sí son legales
        for cid in [0, 1, 2, 13]:
            assert cid in legales

    def test_liderar_con_corazones_rotos(self) -> None:
        """Con corazones rotos, cualquier carta de la mano es legal."""
        mano = [0, 1, 2, 13, 39]
        legales = compute_legales(
            mano=mano,
            mesa_ids=[],
            corazones_rotos=True,
            es_primera_baza=False,
        )
        assert sorted(legales) == sorted(mano)

    def test_liderar_solo_corazones(self) -> None:
        """Si solo quedan corazones en la mano, se pueden liderar."""
        mano = [39, 40]  # 2♥ 3♥
        legales = compute_legales(
            mano=mano,
            mesa_ids=[],
            corazones_rotos=False,
            es_primera_baza=False,
        )
        assert sorted(legales) == sorted(mano)

    def test_seguir_palo_con_cartas_del_palo(self) -> None:
        """Al seguir el palo, solo se pueden jugar cartas de ese palo."""
        mano = [0, 1, 2, 13, 39]  # 2♣ 3♣ 4♣ 2♦ 2♥
        legales = compute_legales(
            mano=mano,
            mesa_ids=[26],  # 2♠ liderada
            corazones_rotos=True,
            es_primera_baza=False,
        )
        # No hay picas en la mano → todas las cartas son legales (void)
        assert sorted(legales) == sorted(mano)

    def test_seguir_palo_con_cartas_del_palo_disponibles(self) -> None:
        """Al seguir el palo con cartas disponibles, solo ese palo."""
        mano = [0, 1, 2, 13, 39]  # 2♣ 3♣ 4♣ 2♦ 2♥
        legales = compute_legales(
            mano=mano,
            mesa_ids=[0],  # 2♣ liderada
            corazones_rotos=True,
            es_primera_baza=False,
        )
        # Solo tréboles en mano: 0, 1, 2
        assert sorted(legales) == [0, 1, 2]

    def test_primera_baza_no_corazones(self) -> None:
        """En la primera baza, no se pueden jugar corazones ni dama de picas."""
        mano = [0, 36, 39]  # 2♣ Q♠ 2♥
        legales = compute_legales(
            mano=mano,
            mesa_ids=[],
            corazones_rotos=False,
            es_primera_baza=True,
        )
        # Solo 2♣ es legal en la primera baza (sin corazones ni Q♠)
        assert legales == [0]

    def test_mesa_vacia_es_liderar(self) -> None:
        """Mesa vacía = liderar."""
        mano = list(range(13))
        legales = compute_legales(
            mano=mano,
            mesa_ids=[],
            corazones_rotos=False,
            es_primera_baza=False,
        )
        # Sin corazones rotos y sin corazones en mano, todos legales
        assert sorted(legales) == sorted(mano)


# ================================================================
# construir_observacion_parcial
# ================================================================

class TestConstruirObservacionParcial:
    """Pruebas para la construcción del vector de observación."""

    def test_shape_correcto(self) -> None:
        obs = construir_observacion_parcial(
            mano_ids=list(range(13)),  # todos tréboles
            mesa_ids=[],
            cementerio_ids=[],
            vacios_por_jugador=[set(), set(), set(), set()],
            puntajes_historicos=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            corazones_rotos=False,
            dama_picas_en=None,
            agente_idx=0,
        )
        assert obs.shape == (190,)
        assert obs.dtype == np.float32

    def test_mano_en_observacion(self) -> None:
        obs = construir_observacion_parcial(
            mano_ids=[0, 1, 2],
            mesa_ids=[],
            cementerio_ids=[],
            vacios_por_jugador=[set(), set(), set(), set()],
            puntajes_historicos=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            corazones_rotos=False,
            dama_picas_en=None,
            agente_idx=0,
        )
        assert obs[0] == 1.0  # carta 0
        assert obs[1] == 1.0  # carta 1
        assert obs[2] == 1.0  # carta 2
        assert obs[3] == 0.0  # carta 3 no en mano

    def test_mesa_en_observacion(self) -> None:
        obs = construir_observacion_parcial(
            mano_ids=[0],
            mesa_ids=[13],  # 2♦ en mesa
            cementerio_ids=[],
            vacios_por_jugador=[set(), set(), set(), set()],
            puntajes_historicos=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            corazones_rotos=False,
            dama_picas_en=None,
            agente_idx=0,
        )
        assert obs[52 + 13] == 1.0  # mesa slot para 2♦

    def test_corazones_rotos_en_observacion(self) -> None:
        obs = construir_observacion_parcial(
            mano_ids=[0],
            mesa_ids=[],
            cementerio_ids=[],
            vacios_por_jugador=[set(), set(), set(), set()],
            puntajes_historicos=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            corazones_rotos=True,
            dama_picas_en=None,
            agente_idx=0,
        )
        assert obs[180] == 1.0

    def test_dama_picas_oculta(self) -> None:
        obs = construir_observacion_parcial(
            mano_ids=[0],
            mesa_ids=[],
            cementerio_ids=[],
            vacios_por_jugador=[set(), set(), set(), set()],
            puntajes_historicos=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            corazones_rotos=False,
            dama_picas_en=None,
            agente_idx=0,
        )
        assert obs[182] == 1.0  # estado "oculta"

    def test_puntajes_historicos_normalizados(self) -> None:
        obs = construir_observacion_parcial(
            mano_ids=[0],
            mesa_ids=[],
            cementerio_ids=[],
            vacios_por_jugador=[set(), set(), set(), set()],
            puntajes_historicos=[50, 0, 25, 75],
            puntos_mano_actual=[0, 0, 0, 0],
            corazones_rotos=False,
            dama_picas_en=None,
            agente_idx=0,
        )
        # agente_idx=0 → relativo a sí mismo
        assert obs[172] == pytest.approx(0.5)  # propio puntaje
        assert obs[173] == 0.0  # jugador 1
        assert obs[174] == pytest.approx(0.25)  # jugador 2
        assert obs[175] == pytest.approx(0.75)  # jugador 3

    def test_vacios_relativos(self) -> None:
        """Vacíos se almacenan relativos al agente."""
        obs = construir_observacion_parcial(
            mano_ids=[0],
            mesa_ids=[],
            cementerio_ids=[],
            vacios_por_jugador=[
                set(),       # yo (sin vacíos)
                {0},         # jugador 1 vacío en tréboles
                {3},         # jugador 2 vacío en corazones
                {2},         # jugador 3 vacío en picas
            ],
            puntajes_historicos=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            corazones_rotos=False,
            dama_picas_en=None,
            agente_idx=0,
        )
        # jugador 1 → rel=1, palo=0 → idx=156+1*4+0=160
        assert obs[160] == 1.0
        # jugador 2 → rel=2, palo=3 → idx=156+2*4+3=167
        assert obs[167] == 1.0
        # jugador 3 → rel=3, palo=2 → idx=156+3*4+2=170
        assert obs[170] == 1.0


# ================================================================
# smoke test: integración sin modelo
# ================================================================

class TestFlujoCompletoSinModelo:
    """Prueba de integración del flujo completo sin cargar modelo real."""

    def test_flujo_recomendacion_sin_modelo(self) -> None:
        """Verifica que el flujo no crashea incluso sin modelo."""
        mano = "2♣ 3♣ 4♣ 5♣ 6♣ 7♣ 8♠ 9♠ 10♠ J♠ Q♠ K♠ A♠"
        ids = parsear_mano(mano)
        assert len(ids) == 13

        mesa_ids: list = []
        legales = compute_legales(
            mano=ids,
            mesa_ids=mesa_ids,
            corazones_rotos=False,
            es_primera_baza=True,
        )
        assert len(legales) >= 1

        obs = construir_observacion_parcial(
            mano_ids=ids,
            mesa_ids=mesa_ids,
            cementerio_ids=[],
            vacios_por_jugador=[set(), set(), set(), set()],
            puntajes_historicos=[0, 0, 0, 0],
            puntos_mano_actual=[0, 0, 0, 0],
            corazones_rotos=False,
            dama_picas_en=None,
            agente_idx=0,
        )
        assert obs.shape == (190,)
        assert not np.any(np.isnan(obs))

        mask = np.zeros(52, dtype=np.bool_)
        for lid in legales:
            mask[lid] = True
        assert np.sum(mask) == len(legales)
