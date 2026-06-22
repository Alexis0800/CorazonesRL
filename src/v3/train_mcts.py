"""
MCTS-guided training para v3 — Buffer BC + Oráculo PIMC.

Basado en el diagnóstico PIMC:
  - 35% del error en bazas ≥8 (donde la enumeración exacta es viable)
  - 46% de decisiones subóptimas por risk_assessment deficiente

Estrategia (Expert Iteration — ExIt):
  1. Durante el entrenamiento, cuando la baza ≥ 10 y el número de
     mundos posibles ≤ 100K, ejecutar PIMC exacto como oráculo.
  2. Guardar (observación, acción_óptima) en un buffer circular.
  3. Periódicamente, calcular BC loss sobre el buffer y combinarlo
     con la PPO loss para guiar la política hacia decisiones óptimas.

Delega en el módulo `src.mcts.analisis` para la lógica pesada de PIMC.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.mcts.analisis import pimc_exacto

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Buffer circular de experiencias del oráculo
# ──────────────────────────────────────────────────────────────

class MCTSBuffer:
    """Buffer circular para almacenar (obs, action) del oráculo PIMC.

    Propiedades:
    - Tamaño fijo (FIFO circular cuando se excede la capacidad).
    - Almacena observaciones como np.float32 y acciones como np.int64.
    - sample(n) retorna batch aleatorio sin reposición (dentro del batch).
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._obs: List[np.ndarray] = []
        self._actions: List[int] = []
        self._write_pos = 0

    def add(self, obs: np.ndarray, action: int) -> None:
        """Añade un par (obs, action) al buffer.

        Si el buffer está lleno, sobrescribe el más antiguo (FIFO circular).

        Args:
            obs: Vector de observación (250-d, float32).
            action: Acción óptima (índice 0-51).
        """
        obs_arr = np.asarray(obs, dtype=np.float32)

        if len(self._obs) < self._capacity:
            self._obs.append(obs_arr)
            self._actions.append(action)
        else:
            self._obs[self._write_pos] = obs_arr
            self._actions[self._write_pos] = action
            self._write_pos = (self._write_pos + 1) % self._capacity

    def sample(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Muestrea n pares (obs, action) aleatoriamente sin reposición.

        Si n > len(buffer), retorna todo el buffer.

        Args:
            n: Número de muestras a obtener.

        Returns:
            Tuple de:
            - observations: (batch, obs_dim) float32
            - actions: (batch,) int64
        """
        size = min(n, len(self._obs))
        indices = np.random.default_rng().choice(
            len(self._obs), size=size, replace=False
        )
        obs_batch = np.stack([self._obs[i] for i in indices])
        act_batch = np.array([self._actions[i]
                             for i in indices], dtype=np.int64)
        return obs_batch, act_batch

    def clear(self) -> None:
        """Vacía el buffer."""
        self._obs.clear()
        self._actions.clear()
        self._write_pos = 0

    def is_empty(self) -> bool:
        """True si el buffer no tiene elementos."""
        return len(self._obs) == 0

    def __len__(self) -> int:
        return len(self._obs)

    def __repr__(self) -> str:
        return f"MCTSBuffer(capacity={self._capacity}, size={len(self._obs)})"


# ──────────────────────────────────────────────────────────────
# Oráculo PIMC
# ──────────────────────────────────────────────────────────────

def evaluar_con_oraculo(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    num_mundos: int = 100,
    rng: Optional[np.random.Generator] = None,
    buffer: Optional[MCTSBuffer] = None,
    obs: Optional[np.ndarray] = None,
) -> Tuple[Carta, Dict[int, float]]:
    """Evalúa cartas legales con PIMC y opcionalmente guarda en buffer BC.

    Delega en `pimc_exacto()` para la decisión (exacta cuando viable,
    sampling cuando no). Si se proporciona buffer y obs, guarda la
    acción óptima para entrenamiento BC futuro.

    Args:
        motor: Estado actual (NO se modifica).
        agente_idx: Jugador que decide.
        legales: Cartas legales.
        num_mundos: Mundos de sampling si la enumeración no es viable.
        rng: Generador aleatorio.
        buffer: Buffer MCTS opcional para guardar (obs, acción_óptima).
        obs: Vector de observación correspondiente al estado.

    Returns:
        Tuple de:
        - Carta óptima (menor puntuación esperada)
        - Dict[carta_id → puntuación esperada]
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(legales) <= 1:
        mejor = legales[0]
        scores = {mejor.id: 0.0}
    else:
        try:
            mejor, scores, exacto = pimc_exacto(
                motor,
                agente_idx,
                legales,
                rng=rng,
                fallback_mundos=num_mundos,
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "PIMC falló (%s) — usando fallback determinista con bot evasivo",
                exc,
            )
            # Fallback determinista: usar la carta con menor valor numérico
            mejor_conservador = min(legales, key=lambda c: c.valor)
            scores = {c.id: float(c.valor) for c in legales}
            mejor = mejor_conservador
            exacto = False

        if exacto:
            logger.debug(
                "PIMC exacto: %d mundos, mejor=%s score=%.2f",
                len(scores), mejor.id, scores[mejor.id],
            )
        else:
            logger.debug(
                "PIMC sampling: %d mundos, mejor=%s score=%.2f",
                num_mundos, mejor.id, scores[mejor.id],
            )

    # Guardar en buffer BC si está disponible
    if buffer is not None and obs is not None:
        buffer.add(obs, mejor.id)

    return mejor, scores


def evaluar_y_guardar_batch(
    motor: MotorCorazones,
    agente_idx: int,
    legales: List[Carta],
    buffer: MCTSBuffer,
    obs: np.ndarray,
    num_mundos: int = 100,
    rng: Optional[np.random.Generator] = None,
    forzar: bool = False,
) -> Optional[Carta]:
    """Wrapper para entrenamiento: evalúa con oráculo y guarda.

    Solo ejecuta el oráculo si:
    - forzar=True, o
    - La baza es ≥ 10 y hay ≤ 100K mundos (condición de viabilidad).

    Args:
        motor: Estado actual.
        agente_idx: Jugador que decide.
        legales: Cartas legales.
        buffer: Buffer donde guardar.
        obs: Observación actual.
        num_mundos: Mundos de sampling.
        rng: Generador aleatorio.
        forzar: Si True, ejecuta siempre (útil para testing).

    Returns:
        Carta óptima si se ejecutó el oráculo, None si se saltó.
    """
    from src.mcts.analisis import _num_mundos_posibles

    if not forzar:
        if motor.numero_baza < 10:
            return None
        n = _num_mundos_posibles(motor, agente_idx)
        if n > 100_000:
            return None

    mejor, _ = evaluar_con_oraculo(
        motor, agente_idx, legales,
        num_mundos=num_mundos,
        rng=rng,
        buffer=buffer,
        obs=obs,
    )
    return mejor


def calcular_bc_loss(
    policy_fn: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    buffer: MCTSBuffer,
    batch_size: int = 256,
) -> float:
    """Calcula la Behavioral Cloning loss sobre un batch del buffer.

    Args:
        policy_fn: Función (obs) → (logits, values). Logits shape (batch, 52).
        buffer: Buffer con experiencias del oráculo.
        batch_size: Tamaño del batch.

    Returns:
        Cross-entropy loss promedio entre las predicciones y las
        acciones del oráculo.
    """
    if buffer.is_empty():
        return 0.0

    obs_batch, act_batch = buffer.sample(batch_size)

    # Pasar al dispositivo correcto (asumimos CPU por ahora)
    import torch
    obs_tensor = torch.from_numpy(obs_batch)
    act_tensor = torch.from_numpy(act_batch).long()

    logits, _ = policy_fn(obs_tensor)
    loss = torch.nn.functional.cross_entropy(logits, act_tensor)

    return float(loss.item())


__all__ = [
    "MCTSBuffer",
    "evaluar_con_oraculo",
    "evaluar_y_guardar_batch",
    "calcular_bc_loss",
]
