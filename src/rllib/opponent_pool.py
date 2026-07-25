"""
Pool de oponentes para self-play en Hearts — curriculum de 5 fases.

  Fase 0  (0–5%):   3 bots simples — bootstrap, aprender reglas básicas.
  Fase 1  (5–15%):  2 bots simples + 1 BotExperto — introducir oponente duro.
  Fase 2 (15–40%):  1 BotExperto + 2 snapshots — mezcla experto + self-play.
  Fase 3 (40–70%):  1 BotExperto (ANCLA) + 2 snapshots del pool completo.
  Fase 4 (70–100%): 1 BotExperto (ANCLA) + 2 snapshots recientes — presión máxima.

BotExperto se mantiene como ANCLA fija en todas las fases ≥1. Lección de
v10_lstm: el self-play PURO (fases 3-4 sin ancla) provoca regresión vs
oponentes de distribución distinta (el agente olvida el juego robusto y se
sobre-especializa en su propio estilo). Un ancla de 1 experto en cada mesa
preserva la robustez sin renunciar al refinamiento por self-play.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import numpy as np

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.agentes.bot_experto import BotExperto
from src.agentes.heuristicos import bot_conservador, bot_agresivo, bot_evasivo
from src.entorno.dimensiones import DIM_ENTORNO, NUM_CARTAS
from src.entorno.observacion import ObservacionBuilder

PolicyFn = Callable[[MotorCorazones, int, List[Carta]], Carta]

_BOTS_SIMPLES: List[PolicyFn] = [bot_conservador, bot_agresivo, bot_evasivo]


class SnapshotPolicy:
    """Wrapper serializable que adapta pesos de política a (motor, idx, legales) -> Carta.

    Almacena los pesos como dict de numpy arrays (picklable) en lugar del objeto
    Policy de Ray (que contiene _thread.RLock y no es serializable entre workers).
    El modelo PyTorch se reconstruye de forma lazy en cada worker.
    """

    def __init__(self, policy, obs_dim: int = DIM_ENTORNO, temperatura: Optional[float] = None):
        self._weights: dict = policy.get_weights()
        self._obs_dim = obs_dim
        self._obs_builder = ObservacionBuilder(dim=obs_dim)
        self._model = None
        # temperatura None = argmax determinista (snapshots, default). Un float
        # (p.ej. 1.0) = MUESTREA del softmax(logits/T): usado por el clon humano
        # para ser estocástico/diverso como un humano real y no explotable por
        # una única línea de juego memorizada.
        self._temperatura = temperatura
        # Estado LSTM por jugador (idx -> [h, c]) y marcador previo por jugador,
        # para arrastrar la memoria a lo largo de la partida y resetearla al
        # inicio de cada partida. Dict por idx para el caso de que el mismo
        # snapshot ocupe varios asientos a la vez.
        self._lstm_state: dict = {}
        self._prev_scores_sum: dict = {}

    @classmethod
    def from_weights(cls, weights: dict, obs_dim: int = DIM_ENTORNO,
                     temperatura: Optional[float] = None) -> "SnapshotPolicy":
        """Crea un SnapshotPolicy directamente desde un dict de pesos numpy."""
        obj = object.__new__(cls)
        obj._weights = weights
        obj._obs_dim = obs_dim
        obj._obs_builder = ObservacionBuilder(dim=obs_dim)
        obj._model = None
        obj._lstm_state = {}
        obj._prev_scores_sum = {}
        obj._temperatura = temperatura
        return obj

    def __getstate__(self):
        return {"weights": self._weights, "obs_dim": self._obs_dim,
                "temperatura": self._temperatura}

    def __setstate__(self, state):
        self._weights = state["weights"]
        self._obs_dim = state["obs_dim"]
        self._obs_builder = ObservacionBuilder(dim=self._obs_dim)
        self._model = None
        self._lstm_state = {}
        self._prev_scores_sum = {}
        self._temperatura = state.get("temperatura")

    def _get_model(self):
        """Construye el modelo PyTorch desde los pesos la primera vez (lazy).

        Detecta automáticamente si los pesos son de HeartsLSTMModel o HeartsActionMaskModel
        inspeccionando las claves del state_dict.
        """
        if self._model is not None:
            return self._model

        import torch
        from src.rllib.model import HeartsActionMaskModel, HeartsLSTMModel
        from gymnasium import spaces

        obs_space = spaces.Dict({
            "obs": spaces.Box(0.0, 1.0, shape=(self._obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0.0, 1.0, shape=(NUM_CARTAS,), dtype=np.float32),
        })
        action_space = spaces.Discrete(NUM_CARTAS)

        is_lstm = any("_lstm_layer" in k for k in self._weights.keys())

        if is_lstm:
            lstm_hidden = self._weights["_lstm_layer.weight_hh_l0"].shape[1]
            model_config = {
                "lstm_cell_size": lstm_hidden,
                "max_seq_len": 13,
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            }
            model = HeartsLSTMModel(obs_space, action_space, NUM_CARTAS, model_config, "snapshot")
        else:
            model_config = {"fcnet_hiddens": [512, 512, 256], "fcnet_activation": "relu", "vf_share_layers": False}
            model = HeartsActionMaskModel(obs_space, action_space, NUM_CARTAS, model_config, "snapshot")

        torch_state = {k: torch.tensor(v) for k, v in self._weights.items()}
        model.load_state_dict(torch_state, strict=True)
        model.eval()
        self._model = model
        return model

    def __call__(
        self,
        motor: MotorCorazones,
        idx: int,
        legales: List[Carta],
        obs_vec: Optional[np.ndarray] = None,
    ) -> Carta:
        import torch

        # Si el env provee la observación COMPLETA (perspectiva de idx), usarla;
        # si no (contextos sin env, p.ej. elo legacy), caer a la obs mínima.
        if obs_vec is None:
            obs_vec = self._obs_builder.construir_desde_motor(motor, jugador_idx=idx)
        mask = np.zeros(NUM_CARTAS, dtype=np.float32)
        for c in legales:
            mask[c.id] = 1.0

        model = self._get_model()
        with torch.no_grad():
            obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            input_dict = {"obs": {"obs": obs_t, "action_mask": mask_t}}

            initial = model.get_initial_state()
            es_recurrente = bool(initial)
            if es_recurrente:
                # Arrastrar el estado LSTM a lo largo de la PARTIDA (igual que en
                # entrenamiento, donde el episodio es una partida completa). Se
                # resetea al inicio de una partida nueva, detectado porque el
                # marcador acumulado baja (de ~100 a 0) o es la primera llamada.
                scores_sum = sum(j.puntuacion_historica for j in motor.jugadores)
                prev = self._prev_scores_sum.get(idx)
                if idx not in self._lstm_state or prev is None or scores_sum < prev:
                    self._lstm_state[idx] = [s.unsqueeze(0) for s in initial]
                self._prev_scores_sum[idx] = scores_sum
                state = self._lstm_state[idx]
            else:
                state = []

            logits, new_state = model.forward(input_dict, state, None)
            if es_recurrente:
                self._lstm_state[idx] = new_state  # arrastrar al siguiente step
            if self._temperatura is not None:
                # Muestreo estocástico: softmax(logits/T) sobre las legales (los
                # ilegales ya llevan -1e9 por la máscara → prob ~0).
                probs = torch.softmax(logits / self._temperatura, dim=1)
                action = int(torch.multinomial(probs, 1).item())
            else:
                action = int(logits.argmax(dim=1).item())

        carta = Carta._TODAS[action]
        if carta not in legales:
            carta = random.choice(legales)
        return carta

    def pasar(self, motor: MotorCorazones, idx: int) -> List[Carta]:
        """Selecciona 3 cartas a pasar con la POLÍTICA NEURONAL (v10c).

        Replica las 3 sub-decisiones que hace el agente en el env durante la fase
        de pase: construye la obs de pase (fase_pase=1, dirección, nº seleccionadas;
        vacíos vacíos porque es inicio de mano) y corre el modelo eligiendo el
        argmax de las cartas aún no seleccionadas. Así los snapshots pasan igual
        que el agente → self-play 100% consistente.
        """
        import torch

        model = self._get_model()
        direccion = motor.direccion_pase()
        dir_norm = {"izquierda": 0.33, "derecha": 0.66,
                    "enfrente": 1.0}.get(direccion, 0.0)
        scores = [j.puntuacion_historica for j in motor.jugadores]
        mano = list(motor.jugadores[idx].mano)
        seleccion: List[Carta] = []

        initial = model.get_initial_state()
        es_recurrente = bool(initial)
        if es_recurrente:
            st = self._lstm_state.get(idx)
            if st is None:
                st = [s.unsqueeze(0) for s in initial]
        else:
            st = []

        for k in range(3):
            obs_vec = self._obs_builder.construir(
                motor=motor, agente_idx=idx,
                vacios=[set() for _ in range(4)],
                puntuacion_historica=scores,
                puntos_mano_actual=[0, 0, 0, 0],
                dama_picas_en=None,
                fase_pase=1.0, direccion_pase=dir_norm, n_pase_seleccionadas=k,
            )
            mask = np.zeros(NUM_CARTAS, dtype=np.float32)
            for c in mano:
                if c not in seleccion:
                    mask[c.id] = 1.0
            with torch.no_grad():
                o = torch.as_tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
                m = torch.as_tensor(mask, dtype=torch.float32).unsqueeze(0)
                logits, new_st = model.forward(
                    {"obs": {"obs": o, "action_mask": m}}, st, None)
                if es_recurrente:
                    st = new_st
                action = int(logits.argmax(dim=1).item())
            carta = Carta._TODAS[action]
            seleccionables = [c for c in mano if c not in seleccion]
            if carta not in seleccionables:
                carta = seleccionables[0]
            seleccion.append(carta)

        if es_recurrente:
            self._lstm_state[idx] = st
        return seleccion


class OpponentPool:
    """Pool de oponentes para self-play con bots y snapshots históricos.

    Usa 5 fases (cortes en progress 0.05/0.15/0.40/0.70):
      - Fase 0 (0–5%):    3 bots simples para aprender las reglas básicas.
      - Fase 1 (5–15%):   2 bots simples + 1 oponente duro (BotExperto, o
                          arquetipo/clon humano al azar si pool_diverso/humano_bc).
      - Fase 2 (15–40%):  1 oponente duro + 2 snapshots. Mezcla + self-play.
      - Fase 3 (40–70%):  self-play. Sin ancla = 3 snapshots del pool completo
                          (como v10); con `anclar_experto=True` = 1 oponente duro
                          + 2 snapshots.
      - Fase 4 (70–100%): presión máxima con los snapshots recientes (últimos 10).
                          Con `anclar_experto=True` mantiene 1 oponente duro de ancla.

    El ancla-experto (fases 3–4) solo aplica si se construyó con
    `anclar_experto=True`; por defecto (False) esas fases son self-play puro.
    Las degradaciones por pool insuficiente (p. ej. `len(snapshots) < 2`) pueden
    mantener una fase temprana aunque `progress` ya haya avanzado.

    La factory retornada acepta `agente_idx` como parámetro para soportar
    rotación multi-posición (el agente puede entrenar desde cualquier asiento).

    Uso:
        pool = OpponentPool(snapshot_dir="models/v_rllib/snapshots")
        factory = pool.make_factory(progress=0.3)
        # En el env:
        opponents = factory(agente_idx=2)  # dict {0: fn, 1: fn, 3: fn}
    """

    def __init__(
        self,
        agente_idx: int = 0,
        snapshot_dir: Optional[str] = None,
        max_snapshots: int = 50,
        obs_dim: int = DIM_ENTORNO,
        anclar_experto: bool = False,
        pool_diverso: bool = False,
        humano_bc_path: Optional[str] = None,
        prob_humano: float = 0.5,
        mesa_humana: float = 0.0,
        temp_humano: Optional[float] = 1.0,
        lunero_garantizado: bool = False,
    ):
        self._agente_idx = agente_idx
        self._snapshot_dir = snapshot_dir
        self._max_snapshots = max_snapshots
        self._obs_dim = obs_dim
        # Si True, mantiene 1 BotExperto como ANCLA en fases 3-4 (evita la
        # regresión del self-play puro). Default False = self-play puro como v10.
        self._anclar_experto = anclar_experto
        # Si True, el oponente "duro" de cada fase es un arquetipo humano al azar
        # (experto/castigador/lunatico/atacante) en vez de siempre BotExperto.
        # Hace el self-play robusto a juego variado (generaliza mejor a humanos).
        self._pool_diverso = pool_diverso
        # Opponent de IMITACIÓN HUMANA (BC sobre partidas reales, ver
        # scripts/entrenar_bc_humano.py). Si se da, cubre una fracción `prob_humano`
        # de los slots de oponente "duro": mete el ESTILO HUMANO REAL en el pool,
        # que es lo que la burbuja de self-play (79% vs bots, 24% vs humanos) no
        # tiene. Ver docs/auditoria_moon_2026-07-20.md.
        self._prob_humano = prob_humano
        # mesa_humana: fracción de MESAS completas dominadas por el clon humano
        # (2 de 3 asientos = clon + 1 ancla de robustez). El fine-tune fallido
        # demostró que la exposición por-slot (prob_humano) se diluye a ~1/6;
        # esta es la palanca de exposición REAL al meta humano.
        self._mesa_humana = mesa_humana
        # temp_humano: temperatura de muestreo del clon (None = argmax). 1.0 =
        # estocástico como un humano; evita que el agente memorice una única
        # línea de explotación contra un clon determinista.
        self._temp_humano = temp_humano
        # lunero_garantizado: en fases 2-4 uno de los slots es SIEMPRE un
        # OponenteLunar (ModoLunar + BotExperto). Los humanos coronan lunas
        # 2.5%/mano por rival, mayormente comprometiéndose MID-MANO; BotLunatico
        # (compromiso desde el pase) no da esa presión realista.
        self._lunero_garantizado = lunero_garantizado
        self._humano_bc_pesos = None
        if humano_bc_path:
            _d = np.load(humano_bc_path)
            self._humano_bc_pesos = {k: _d[k] for k in _d.files}
        self._snapshots: List[SnapshotPolicy] = []

    def add_snapshot(self, policy) -> None:
        """Añade una nueva política snapshot al pool (FIFO si excede el máximo)."""
        snap = SnapshotPolicy(policy, obs_dim=self._obs_dim)
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)

    def make_factory(self, progress: float) -> Callable:
        """Devuelve un opponent_factory serializable para la fase actual.

        El closure captura una copia de los snapshots disponibles en este momento.
        Llamar periódicamente para incorporar nuevos snapshots al pool de rivales.

        Args:
            progress: fracción de entrenamiento completada (0.0 – 1.0).

        Returns:
            Callable(agente_idx: int) -> dict[int, PolicyFn]
        """
        snapshots = list(self._snapshots)
        # Snapshots recientes: últimos 10 (fase 4 solo usa estos para máxima presión)
        snapshots_recientes = snapshots[-10:] if len(snapshots) >= 10 else snapshots
        anclar = self._anclar_experto  # capturar local (closure serializable)
        diverso = self._pool_diverso
        humano_pesos = self._humano_bc_pesos  # dict numpy picklable (o None)
        prob_humano = self._prob_humano
        mesa_humana = self._mesa_humana
        temp_humano = self._temp_humano
        obs_dim_local = self._obs_dim
        lunero = self._lunero_garantizado

        def _clon_humano():
            return SnapshotPolicy.from_weights(
                humano_pesos, obs_dim=obs_dim_local, temperatura=temp_humano)

        def _bot_dificil():
            """Oponente 'duro': bot de IMITACIÓN HUMANA (si está disponible), o
            BotExperto/arquetipo al azar.

            Con `humano_bc_path`, una fracción `prob_humano` de los slots duros es
            el clon del estilo humano real (val top-1 ~0.71) -- es el oponente más
            parecido a los humanos que enfrentamos (campeón 46% win-rate vs él, vs
            24% real, vs 92% bots simples). Rompe la burbuja de self-play que hace
            que el campeón le gane a los bots pero no a los humanos. Ver
            docs/auditoria_moon_2026-07-20.md.

            BotLunatico con el doble de peso que los demás. Justificación
            histórica: el backtest de regret mostraba las manos con pozo como
            concentradoras de error; la auditoría posterior demostró que ese
            regret elevado es BRECHA ESTRUCTURAL (BotExperto igual de elevado),
            no defecto entrenable. El 2x se mantiene porque la exposición a
            intentos de pozo sigue siendo el único estímulo del pool para la
            defensa temprana, y los humanos lunean 2.52%/mano cada uno (ver
            docs/auditoria_moon_2026-07-20.md §OFENSIVA).
            """
            if humano_pesos is not None and random.random() < prob_humano:
                return _clon_humano()
            if diverso:
                from src.agentes.bot_castigador import BotCastigador
                from src.agentes.bot_lunatico import BotLunatico
                from src.agentes.bot_atacante_lider import BotAtacanteLider
                arquetipos = [BotExperto, BotCastigador, BotLunatico, BotAtacanteLider]
                pesos = [1, 1, 2, 1]
                return random.choices(arquetipos, weights=pesos, k=1)[0]()
            return BotExperto()

        def _bot_ancla_no_humano():
            """Ancla de robustez para las mesas humanas: heurístico, nunca el clon."""
            if diverso:
                from src.agentes.bot_castigador import BotCastigador
                from src.agentes.bot_lunatico import BotLunatico
                from src.agentes.bot_atacante_lider import BotAtacanteLider
                arquetipos = [BotExperto, BotCastigador, BotLunatico, BotAtacanteLider]
                return random.choices(arquetipos, weights=[1, 1, 2, 1], k=1)[0]()
            return BotExperto()

        def _factory(agente_idx: int = 0) -> Dict[int, PolicyFn]:
            opp_indices = [i for i in range(4) if i != agente_idx]
            random.shuffle(opp_indices)
            fns: Dict[int, PolicyFn] = {}

            # MESA HUMANA (prioridad sobre las fases): 2 clones humanos + 1 ancla.
            # Es la exposición mayoritaria al meta humano que la vía por-slot no
            # logra (ver docs/auditoria_moon_2026-07-20.md, post-mortem fine-tune).
            # SOLO en fases tardías (progress>=0.40): v11 la tuvo desde el paso 0
            # y diluyó el bootstrap (autopsia en docs/decision_desde_cero_2026-07-25.md).
            if humano_pesos is not None and progress >= 0.40 and random.random() < mesa_humana:
                fns[opp_indices[0]] = _clon_humano()
                fns[opp_indices[1]] = _clon_humano()
                fns[opp_indices[2]] = _bot_ancla_no_humano()
                return fns

            if progress < 0.05 or len(snapshots) == 0:
                # Fase 0: bootstrap con 3 bots simples
                for idx in opp_indices:
                    fns[idx] = random.choice(_BOTS_SIMPLES)

            elif progress < 0.15 or len(snapshots) < 1:
                # Fase 1: 2 bots simples + 1 oponente duro (diverso si pool_diverso)
                fns[opp_indices[0]] = random.choice(_BOTS_SIMPLES)
                fns[opp_indices[1]] = random.choice(_BOTS_SIMPLES)
                fns[opp_indices[2]] = _bot_dificil()

            elif progress < 0.40 or len(snapshots) < 2:
                # Fase 2: 1 oponente duro + 2 snapshots — mezcla + self-play
                fns[opp_indices[0]] = _bot_dificil()
                fns[opp_indices[1]] = random.choice(snapshots)
                fns[opp_indices[2]] = random.choice(snapshots)

            elif progress < 0.70:
                # Fase 3: self-play. Con ancla = 1 oponente duro + 2 snapshots;
                # sin ancla = 3 snapshots del pool completo (como v10).
                if anclar:
                    fns[opp_indices[0]] = _bot_dificil()
                    fns[opp_indices[1]] = random.choice(snapshots)
                    fns[opp_indices[2]] = random.choice(snapshots)
                else:
                    for idx in opp_indices:
                        fns[idx] = random.choice(snapshots)

            else:
                # Fase 4: presión máxima con snapshots recientes (+ ancla opcional).
                pool = snapshots_recientes if len(snapshots_recientes) >= 1 else snapshots
                if anclar:
                    fns[opp_indices[0]] = _bot_dificil()
                    fns[opp_indices[1]] = random.choice(pool)
                    fns[opp_indices[2]] = random.choice(pool)
                else:
                    for idx in opp_indices:
                        fns[idx] = random.choice(pool)

            # Lunero garantizado (fases 2-4): 1 slot SIEMPRE es OponenteLunar.
            # progress>=0.15 y >=1 snapshot ⇔ estamos en fase 2, 3 o 4 (las
            # degradaciones por pool insuficiente caen en fases 0-1, sin lunero).
            if lunero and progress >= 0.15 and len(snapshots) >= 1:
                from src.agentes.oponente_lunar import OponenteLunar
                # Slot [1], no [0]: el [0] es el oponente DURO (ancla experto /
                # clon) — pisarlo dejaba la mesa sin ancla y anulaba prob-humano
                # (bug del Run A). En todo branch donde este guard puede disparar
                # (progress>=0.15, >=1 snapshot), opp_indices[1] es un snapshot.
                fns[opp_indices[1]] = OponenteLunar()

            return fns

        return _factory
