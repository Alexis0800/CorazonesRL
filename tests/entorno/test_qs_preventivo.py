"""
Tests TDD para la penalización preventiva de Q♠ (v2_1).

Verifica que:
  - recompensa_qs_preventivo penaliza jugar Q♠ cuando hay alternativas seguras.
  - No penaliza cuando Q♠ es la única carta jugable del palo.
  - No penaliza cuando Q♠ ya fue capturada.
  - Se integra correctamente con el resto de señales.
"""

from __future__ import annotations

import pytest

from src.dominio.carta import Carta
from src.v2_1.recompensas import (
    CalculadoraRecompensasV21,
    RewardConfigV21,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _carta(valor: int, palo: int) -> Carta:
    for c in Carta._TODAS:
        if c.valor == valor and c.palo == palo:
            return c
    raise ValueError(f"No encontrada: valor={valor}, palo={palo}")


TREBOL = 0
DIAMANTE = 1
PICA = 2
CORAZON = 3

_2_TREBOL = _carta(2, TREBOL)
_3_TREBOL = _carta(3, TREBOL)
_10_TREBOL = _carta(10, TREBOL)
_2_DIAMANTE = _carta(2, DIAMANTE)
_2_PICAS = _carta(2, PICA)
_5_PICAS = _carta(5, PICA)
_9_PICAS = _carta(9, PICA)
_10_PICAS = _carta(10, PICA)
_J_PICAS = _carta(11, PICA)
_Q_ESPADAS = _carta(12, PICA)
_K_ESPADAS = _carta(13, PICA)
_A_ESPADAS = _carta(14, PICA)
_2_CORAZON = _carta(2, CORAZON)
_5_CORAZON = _carta(5, CORAZON)
_10_CORAZON = _carta(10, CORAZON)
_A_CORAZON = _carta(14, CORAZON)


# ============================================================
# Clase 1 — RewardConfigV21 tiene nueva constante
# ============================================================

class TestRewardConfigV21QSPreventivo:
    """Verifica que RewardConfigV21 incluya la penalización preventiva."""

    def test_constante_existe(self) -> None:
        """REWARD_QS_PREVENTIVO debe existir en RewardConfigV21."""
        cfg = RewardConfigV21()
        assert hasattr(cfg, "REWARD_QS_PREVENTIVO"), (
            "Falta REWARD_QS_PREVENTIVO en RewardConfigV21"
        )

    def test_valor_negativo_razonable(self) -> None:
        """La penalización preventiva debe ser negativa pero no tan fuerte
        como capturar Q♠ (-13). Sugerido: -5.0."""
        cfg = RewardConfigV21()
        valor = cfg.REWARD_QS_PREVENTIVO
        assert valor < 0.0, (
            f"REWARD_QS_PREVENTIVO debe ser negativo, es {valor}"
        )
        assert valor > -13.0, (
            f"REWARD_QS_PREVENTIVO no debe exceder REWARD_DAMA_PICAS (-13.0), "
            f"es {valor}"
        )


# ============================================================
# Clase 2 — recompensa_qs_preventivo (nuevo método)
# ============================================================

class TestRecompensaQSPreventivo:
    """Verifica el comportamiento de recompensa_qs_preventivo()."""

    @pytest.fixture
    def calc(self) -> CalculadoraRecompensasV21:
        return CalculadoraRecompensasV21()

    # --- Casos donde DEBE penalizar ---

    def test_penaliza_jugar_qs_teniendo_alternativa_en_palo(
        self, calc,
    ) -> None:
        """Caso: agente juega Q♠ liderando, pero tiene 2♠ y 5♠ en mano.
        Debe penalizar porque podía jugar una carta más segura."""
        reward = calc.recompensa_qs_preventivo(
            carta_jugada=_Q_ESPADAS,
            cartas_del_palo=[_Q_ESPADAS, _2_PICAS, _5_PICAS],
            qs_activa=True,
        )
        assert reward < 0.0, (
            f"Jugar Q♠ con alternativas seguras debe penalizarse. "
            f"Obtenido: {reward}"
        )

    def test_penaliza_jugar_qs_siguiendo_palo_con_baja(
        self, calc,
    ) -> None:
        """Caso: agente sigue palo de ♠ con Q♠, pero tiene 2♠ disponible.
        Debe penalizar porque podía jugar una carta más baja."""
        reward = calc.recompensa_qs_preventivo(
            carta_jugada=_Q_ESPADAS,
            cartas_del_palo=[_Q_ESPADAS, _2_PICAS, _9_PICAS],
            qs_activa=True,
        )
        assert reward < 0.0, (
            f"Seguir con Q♠ teniendo 2♠ debe penalizarse. "
            f"Obtenido: {reward}"
        )

    def test_penaliza_jugar_qs_siguiendo_palo_con_alta_no_q(
        self, calc,
    ) -> None:
        """Caso: agente sigue palo de ♠ con Q♠, pero tiene K♠ disponible.
        Debe penalizar porque K♠ es más seguro que Q♠ (si Q♠ está en mi mano,
        K♠ no puede capturar Q♠)."""
        reward = calc.recompensa_qs_preventivo(
            carta_jugada=_Q_ESPADAS,
            cartas_del_palo=[_Q_ESPADAS, _K_ESPADAS, _2_PICAS],
            qs_activa=True,
        )
        assert reward < 0.0, (
            f"Jugar Q♠ teniendo K♠ alternativo debe penalizarse. "
            f"Obtenido: {reward}"
        )

    # --- Casos donde NO debe penalizar ---

    def test_no_penaliza_si_qs_no_es_la_jugada(
        self, calc,
    ) -> None:
        """Jugar una carta que no es Q♠ nunca acciona esta penalización."""
        reward = calc.recompensa_qs_preventivo(
            carta_jugada=_5_PICAS,
            cartas_del_palo=[_Q_ESPADAS, _5_PICAS, _2_PICAS],
            qs_activa=True,
        )
        assert reward == 0.0, (
            f"Jugar no-Q♠ no debe penalizar. Obtenido: {reward}"
        )

    def test_no_penaliza_si_qs_es_la_unica_del_palo(
        self, calc,
    ) -> None:
        """Si Q♠ es la única carta de ♠ en mano, está forzado a jugarla.
        No debe penalizar."""
        reward = calc.recompensa_qs_preventivo(
            carta_jugada=_Q_ESPADAS,
            cartas_del_palo=[_Q_ESPADAS],
            qs_activa=True,
        )
        assert reward == 0.0, (
            f"Jugar Q♠ como única carta del palo no debe penalizar. "
            f"Obtenido: {reward}"
        )

    def test_no_penaliza_si_qs_ya_fue_capturada(
        self, calc,
    ) -> None:
        """Si Q♠ ya está en el cementerio (qs_activa=False), no hay
        penalización preventiva (porque no hay Q♠ que proteger)."""
        reward = calc.recompensa_qs_preventivo(
            carta_jugada=_Q_ESPADAS,
            cartas_del_palo=[_Q_ESPADAS, _2_PICAS],
            qs_activa=False,
        )
        assert reward == 0.0, (
            f"Q♠ inactiva no debe generar penalización. Obtenido: {reward}"
        )

    def test_no_penaliza_si_no_hay_cartas_del_palo(
        self, calc,
    ) -> None:
        """Edge case: cartas_del_palo vacío (no debería ocurrir en práctica,
        pero el método debe ser robusto)."""
        reward = calc.recompensa_qs_preventivo(
            carta_jugada=_Q_ESPADAS,
            cartas_del_palo=[],
            qs_activa=True,
        )
        # No debe crashear. El valor puede ser 0 o negativo.
        assert isinstance(reward, float)


# ============================================================
# Clase 3 — Integración: Q♠ preventivo + dump no compiten
# ============================================================

class TestQSPreventivoVsDump:
    """Verifica que recompensa_qs_preventivo y recompensa_qs_dump
    no compitan entre sí."""

    @pytest.fixture
    def calc(self) -> CalculadoraRecompensasV21:
        return CalculadoraRecompensasV21()

    def test_dump_y_preventivo_son_independientes(
        self, calc,
    ) -> None:
        """qs_dump premia descartar Q♠ sobre rival (REWARD_QS_DUMP = +12).
        qs_preventivo penaliza jugar Q♠ teniendo alternativas (negativo).
        Son señales independientes: una decisión puede recibir ambas
        (ej: liderar Q♠ cuando eres void en el palo de salida)."""
        # Verificar que existen como métodos separados
        assert hasattr(calc, "recompensa_qs_dump")
        assert hasattr(calc, "recompensa_qs_preventivo")

        # Caso: liderar Q♠ cuando tienes 2♠ también
        # - qs_dump: palo_salida=None, ganador!=agente → +12
        #   (el código considera liderar Q♠ como "dump" si otro gana)
        # - qs_preventivo: tienes alternativa → -5
        # Son señales independientes y opuestas: dump premia el resultado
        # (rival captura Q♠), preventivo penaliza la decisión (tenías alternativa)
        dump_r = calc.recompensa_qs_dump(
            agente_idx=0,
            carta_jugada_agente=_Q_ESPADAS,
            palo_salida=None,
            ganador=1,
        )
        prev_r = calc.recompensa_qs_preventivo(
            carta_jugada=_Q_ESPADAS,
            cartas_del_palo=[_Q_ESPADAS, _2_PICAS],
            qs_activa=True,
        )
        # Ambas señales se activan: dump premia el resultado (+12),
        # preventivo penaliza la decisión (-5)
        assert dump_r > 0.0, f"dump debe premiar: {dump_r}"
        assert prev_r < 0.0, f"preventivo debe penalizar: {prev_r}"
