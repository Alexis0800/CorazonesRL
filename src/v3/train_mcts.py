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
            obs: Vector de observación (DIM_V3, float32).
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
    num_mundos: int = 50,
    rng: Optional[np.random.Generator] = None,
    forzar: bool = False,
) -> Optional[Carta]:
    """Wrapper para entrenamiento: evalúa con oráculo (sampling PIMC) y guarda.

    Usa SIEMPRE sampling (determinización) en vez de enumeración exacta,
    para mantener el costo computacional acotado (~50 mundos → ~1s por
    llamada, vs ~12min con enumeración exacta a 34K mundos en baza 10).

    Solo ejecuta el oráculo si:
    - forzar=True, o
    - La baza es ≥ 10.

    Args:
        motor: Estado actual.
        agente_idx: Jugador que decide.
        legales: Cartas legales.
        buffer: Buffer donde guardar.
        obs: Observación actual.
        num_mundos: Mundos de sampling (default 50, balance velocidad/calidad).
        rng: Generador aleatorio.
        forzar: Si True, ejecuta siempre (útil para testing).

    Returns:
        Carta óptima si se ejecutó el oráculo, None si se saltó.
    """
    from src.mcts.pimc import _puntaje_esperado_por_carta

    if not forzar:
        if motor.numero_baza < 10:
            return None

    if rng is None:
        rng = np.random.default_rng()

    scores = _puntaje_esperado_por_carta(
        motor, agente_idx, legales,
        num_mundos=num_mundos,
        rng=rng,
    )
    mejor = min(legales, key=lambda c: scores[c.id])
    buffer.add(obs, mejor.id)
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


def entrenar_bc_epoch(
    model,  # MaskablePPO
    buffer: MCTSBuffer,
    batch_size: int = 256,
    lr: float = 5e-4,
    max_batches: Optional[int] = None,
    vecnorm: Optional[Any] = None,
) -> float:
    """Ejecuta un epoch de Behavioral Cloning fine-tuning sobre el buffer.

    Itera el buffer completo una vez (o hasta max_batches) usando un
    optimizador Adam temporal con cross-entropy loss sobre las acciones
    del oráculo. No modifica el optimizer del modelo PPO.

    Args:
        model: Modelo MaskablePPO entrenado.
        buffer: Buffer con experiencias del oráculo.
        batch_size: Tamaño de batch.
        lr: Learning rate para el optimizador BC.
        max_batches: Límite de batches (None = buffer completo).
        vecnorm: VecNormalize para normalizar observaciones crudas
                 antes de pasarlas a la política (opcional pero
                 recomendado si el modelo se entrenó con VecNormalize).

    Returns:
        Pérdida promedio del epoch.
    """
    import torch

    if buffer.is_empty():
        return 0.0

    n_total = len(buffer)
    n_batches = (n_total + batch_size - 1) // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    device = model.device if hasattr(model, 'device') else 'cpu'
    policy = model.policy.to(device)

    # Optimizador temporal solo para este fine-tuning
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_total = 0.0

    indices = np.random.default_rng().permutation(n_total)
    obs_raw = np.stack([buffer._obs[i] for i in indices])
    act_arr = np.array([buffer._actions[i]
                       for i in indices], dtype=np.int64)

    # Normalizar si hay VecNormalize (las obs del buffer son crudas)
    if vecnorm is not None:
        obs_raw = vecnorm.normalize_obs(obs_raw)

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_total)
        obs_batch = torch.from_numpy(obs_raw[start:end]).to(device)
        act_batch = torch.from_numpy(act_arr[start:end]).to(device)

        optimizer.zero_grad()

        # SB3 ActorCriticPolicy: extract_features → policy_net → action_net
        # El mlp_extractor.policy_net reduce features_dim → last_net_arch_dim
        # (ej. Transformer: 256 → 128), necesario para que action_net(128→52)
        # reciba la dimensión correcta.
        features = policy.extract_features(obs_batch)
        if (policy.mlp_extractor is not None
                and policy.mlp_extractor.policy_net is not None):
            latent_pi = policy.mlp_extractor.policy_net(features)
        else:
            latent_pi = features
        logits = policy.action_net(latent_pi)

        loss = torch.nn.functional.cross_entropy(logits, act_batch)
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    avg_loss = loss_total / n_batches if n_batches > 0 else 0.0
    return avg_loss


def entrenar_bc_dataset(
    model,
    dataset_path: str,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> List[float]:
    """Pre-entrena el modelo con un dataset offline de (obs, all_scores).

    Carga el dataset generado por _generar_dataset.py (v2).
    all_scores es una matriz (N, 52) donde cada fila tiene el score
    esperado de CADA carta (o -1 si no es legal en ese estado).
    La accion optima es argmax(all_scores).

    Args:
        model: Modelo MaskablePPO.
        dataset_path: Ruta al archivo .npz del dataset.
        epochs: Numero de epochs completos.
        batch_size: Tamaño de batch.
        lr: Learning rate.

    Returns:
        Lista de perdidas por epoch.
    """
    import torch

    data = np.load(dataset_path)
    obs_all = data["observations"]
    scores_all = data["all_scores"]  # (N, 52) — score esperado, menor=mejor

    # Accion optima = argmin de scores validos (excluyendo -1 = no disponible)
    masked_scores = np.where(scores_all >= 0, scores_all, np.inf)
    act_all = np.argmin(masked_scores, axis=1).astype(np.int64)

    device = model.device if hasattr(model, 'device') else 'cpu'
    policy = model.policy.to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    n_total = len(obs_all)
    epoch_losses = []

    for epoch in range(epochs):
        indices = np.random.default_rng().permutation(n_total)
        loss_total = 0.0
        n_batches = 0

        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)
            batch_idx = indices[start:end]

            obs_batch = torch.from_numpy(obs_all[batch_idx]).to(device)
            act_batch = torch.from_numpy(act_all[batch_idx]).to(device)

            optimizer.zero_grad()
            features = policy.extract_features(obs_batch)
            if (policy.mlp_extractor is not None
                    and policy.mlp_extractor.policy_net is not None):
                latent_pi = policy.mlp_extractor.policy_net(features)
            else:
                latent_pi = features
            logits = policy.action_net(latent_pi)

            loss = torch.nn.functional.cross_entropy(logits, act_batch)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            n_batches += 1

        avg_loss = loss_total / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        logger.info("BC dataset epoch %d/%d: loss=%.4f",
                    epoch + 1, epochs, avg_loss)

    return epoch_losses


__all__ = [
    "MCTSBuffer",
    "evaluar_con_oraculo",
    "evaluar_y_guardar_batch",
    "calcular_bc_loss",
    "entrenar_bc_epoch",
    "entrenar_bc_dataset",
]
