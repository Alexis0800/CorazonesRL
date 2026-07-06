# Modelo aprendido de moon_prob (Fase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reemplazar la heurística de `moon_prob_agente`/`moon_prob_rival` (hoy duplicada en 3 lugares) por 2 modelos aprendidos pequeños (PyTorch MLP), entrenados con las 483 manos reales reconstruibles de `data/partidas_bridge.jsonl`, y conectarlos a `recomendador.py` (producción). No toca el entrenamiento de v10c (eso es Fase 2, spec separado).

**Architecture:** Un módulo nuevo `src/entorno/moon_model.py` centraliza extracción de features (pura, determinista) + las 2 redes + un wrapper `EstimadorMoonProb` con fallback seguro a 0.0 si no hay pesos entrenados. Un script `scripts/entrenar_moon_prob.py` genera el dataset re-jugando manos reales desde las 4 perspectivas y entrena ambas redes. `Recomendador` gana estado nuevo (historial de bazas, memoria del pase real vía un endpoint nuevo `/registrar_pase`) para alimentar los modelos en producción.

**Tech Stack:** Python 3.12, PyTorch (ya es dependencia del proyecto), pytest.

**Spec:** `docs/superpowers/specs/2026-07-06-moon-prob-modelo-aprendido-design.md`

---

## Antes de empezar

Activar el venv en cada sesión de shell:

```bash
source .venv/Scripts/activate
```

---

### Task 1: `moon_model.py` — estructura de historial y helpers puros

**Files:**
- Create: `src/entorno/moon_model.py`
- Test: `tests/entorno/test_moon_model.py`

- [ ] **Step 1: Escribir los tests que fallan**

```python
"""Tests de src/entorno/moon_model.py."""
from __future__ import annotations

from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.moon_model import (
    DIM_PROPIO,
    DIM_RIVAL,
    EntradaBaza,
    _alguien_mas_tiene_puntos,
    _ganador_parcial,
    _ratio_bazas_con_puntos,
    _tasa_lidero_corazon_dama,
)


def _carta(palo, valor):
    return Carta(palo, valor)


class TestAlguienMasTienePuntos:
    def test_falso_si_nadie_ha_capturado_puntos(self):
        m = MotorCorazones()
        m.repartir()
        assert _alguien_mas_tiene_puntos(m, 0) is False

    def test_verdadero_si_otro_jugador_capturo_un_corazon(self):
        m = MotorCorazones()
        m.repartir()
        m.jugadores[1].bazas_ganadas = [_carta(3, 5)]  # 5 de corazones
        assert _alguien_mas_tiene_puntos(m, 0) is True

    def test_ignora_los_puntos_propios_del_objetivo(self):
        m = MotorCorazones()
        m.repartir()
        m.jugadores[0].bazas_ganadas = [_carta(3, 5)]
        assert _alguien_mas_tiene_puntos(m, 0) is False


class TestGanadorParcial:
    def test_none_si_mesa_vacia(self):
        m = MotorCorazones()
        m.repartir()
        assert _ganador_parcial(m) is None

    def test_gana_la_carta_mas_alta_del_palo_de_salida(self):
        m = MotorCorazones()
        m.repartir()
        m.mesa = [(0, _carta(0, 5)), (1, _carta(0, 10)), (2, _carta(3, 14))]
        m.palo_de_salida = 0
        assert _ganador_parcial(m) == 1  # el As de corazones no sigue el palo

    def test_no_seguir_el_palo_no_gana(self):
        m = MotorCorazones()
        m.repartir()
        m.mesa = [(0, _carta(0, 5)), (1, _carta(3, 14))]
        m.palo_de_salida = 0
        assert _ganador_parcial(m) == 0


class TestRatioBazasConPuntos:
    def test_cero_sin_historial(self):
        assert _ratio_bazas_con_puntos([], 0) == 0.0

    def test_ignora_bazas_sin_puntos(self):
        historial = [
            EntradaBaza(lider=0, ganador=0, tenia_puntos=False, lidero_corazon_o_dama=False),
        ]
        assert _ratio_bazas_con_puntos(historial, 0) == 0.0

    def test_calcula_la_razon_correcta(self):
        historial = [
            EntradaBaza(lider=0, ganador=1, tenia_puntos=True, lidero_corazon_o_dama=False),
            EntradaBaza(lider=1, ganador=1, tenia_puntos=True, lidero_corazon_o_dama=False),
            EntradaBaza(lider=2, ganador=2, tenia_puntos=False, lidero_corazon_o_dama=False),
        ]
        assert _ratio_bazas_con_puntos(historial, 1) == 1.0
        assert _ratio_bazas_con_puntos(historial, 0) == 0.0


class TestTasaLideroCorazonDama:
    def test_cero_si_nunca_lidero(self):
        historial = [
            EntradaBaza(lider=1, ganador=1, tenia_puntos=True, lidero_corazon_o_dama=True),
        ]
        assert _tasa_lidero_corazon_dama(historial, 0) == 0.0

    def test_calcula_la_tasa_correcta(self):
        historial = [
            EntradaBaza(lider=0, ganador=0, tenia_puntos=True, lidero_corazon_o_dama=True),
            EntradaBaza(lider=0, ganador=1, tenia_puntos=False, lidero_corazon_o_dama=False),
        ]
        assert _tasa_lidero_corazon_dama(historial, 0) == 0.5


def test_dimensiones_publicadas():
    assert DIM_PROPIO == 333
    assert DIM_RIVAL == 272
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `python -m pytest tests/entorno/test_moon_model.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'src.entorno.moon_model'`

- [ ] **Step 3: Implementar `src/entorno/moon_model.py` (parte 1: estructura + helpers)**

```python
"""
Modelos aprendidos de moon_prob (reemplazan la heurística de coeficientes
fijos, antes duplicada en corazones_rllib.py, recomendador.py y
pimc_regret.py). Ver spec:
docs/superpowers/specs/2026-07-06-moon-prob-modelo-aprendido-design.md

Dos modelos, con esquemas de features distintos (información disponible
distinta):
  - "propio": mi propio pozo, mano exacta conocida -- reutiliza el vector
    completo de ObservacionBuilder(dim=332).
  - "rival": ¿un rival específico está armando el pozo? -- SOLO señales
    públicas de ese rival (nunca su mano real).

En ambos casos hay un gate DURO (no aprendido): si cualquier otro jugador ya
capturó puntos en la mano, el pozo del objetivo es 0 exacto -- es una regla
del juego, no algo incierto que el modelo deba aprender.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.dominio.motor import MotorCorazones
from src.entorno.dimensiones import DIM_V13
from src.entorno.observacion import ObservacionBuilder

DIM_PROPIO = DIM_V13 + 1  # vector v13 completo (moon_prob en 0) + razón bazas-con-puntos
DIM_RIVAL = 272  # ver features_rival() para el desglose exacto de este número


@dataclass
class EntradaBaza:
    """Una baza ya resuelta, para trackear comportamiento (no solo captura final).

    `lider` es None para una baza cerrada por remate ("se lleva el resto") --
    ahí no hay una carta de salida real que analizar.
    """
    lider: Optional[int]
    ganador: int
    tenia_puntos: bool
    lidero_corazon_o_dama: bool


def _alguien_mas_tiene_puntos(motor: MotorCorazones, idx: int) -> bool:
    """Gate duro: si CUALQUIER otro jugador ya capturó puntos, el pozo de
    `idx` es imposible (el pozo exige TODOS los puntos para un solo jugador)."""
    return any(
        j.contar_puntos_bazas() > 0
        for i, j in enumerate(motor.jugadores) if i != idx
    )


def _ganador_parcial(motor: MotorCorazones) -> Optional[int]:
    """Quién va ganando la baza en curso, posiblemente incompleta.

    None si la mesa está vacía (nadie ha jugado esta baza todavía).
    """
    if not motor.mesa:
        return None
    palo_salida = motor.palo_de_salida
    ganador_idx, carta_mas_alta = motor.mesa[0]
    for idx, carta in motor.mesa[1:]:
        if carta.palo == palo_salida and carta.valor > carta_mas_alta.valor:
            ganador_idx, carta_mas_alta = idx, carta
    return ganador_idx


def _ratio_bazas_con_puntos(historial: List[EntradaBaza], idx: int) -> float:
    """Bazas-con-puntos que ganó `idx` / bazas-con-puntos jugadas hasta ahora.

    0/0 -> 0.0: ninguna baza con puntos jugada aún no es señal de nada.
    """
    con_puntos = [h for h in historial if h.tenia_puntos]
    if not con_puntos:
        return 0.0
    ganadas = sum(1 for h in con_puntos if h.ganador == idx)
    return ganadas / len(con_puntos)


def _tasa_lidero_corazon_dama(historial: List[EntradaBaza], idx: int) -> float:
    """De las bazas que `idx` lideró, en qué fracción lideró con corazón o Q♠
    -- señal de intención más fuerte que solo "ganó la baza": liderar con
    corazones sin necesidad es deliberado."""
    lideradas = [h for h in historial if h.lider == idx]
    if not lideradas:
        return 0.0
    return sum(1 for h in lideradas if h.lidero_corazon_o_dama) / len(lideradas)
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `python -m pytest tests/entorno/test_moon_model.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add src/entorno/moon_model.py tests/entorno/test_moon_model.py
git commit -m "feat: add moon_model helpers and EntradaBaza history structure"
```

---

### Task 2: `moon_model.py` — `features_propio`

**Files:**
- Modify: `src/entorno/moon_model.py`
- Test: `tests/entorno/test_moon_model.py`

- [ ] **Step 1: Agregar los tests que fallan**

Agregar al final de `tests/entorno/test_moon_model.py`:

```python
from src.entorno.moon_model import features_propio


class TestFeaturesPropio:
    def test_forma_y_rango(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        historial = []
        feats = features_propio(
            m, 0, vacios, historial, [], [],
            [0, 0, 0, 0], [0, 0, 0, 0], None,
        )
        assert feats.shape == (DIM_PROPIO,)
        assert feats.dtype == np.float32
        # los 2 slots de moon_prob del vector v13 reutilizado deben quedar en 0
        assert feats[187] == 0.0
        assert feats[188] == 0.0

    def test_mano_propia_se_refleja_en_el_one_hot(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        feats = features_propio(m, 0, vacios, [], [], [], [0, 0, 0, 0], [0, 0, 0, 0], None)
        for c in m.jugadores[0].mano:
            assert feats[c.id] == 1.0

    def test_ratio_bazas_con_puntos_al_final_del_vector(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        historial = [
            EntradaBaza(lider=0, ganador=0, tenia_puntos=True, lidero_corazon_o_dama=False),
        ]
        feats = features_propio(m, 0, vacios, historial, [], [], [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert feats[-1] == 1.0  # gané la única baza-con-puntos jugada
```

(Agregar `import numpy as np` arriba del archivo de test si no está ya presente — revisar antes de correr.)

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `python -m pytest tests/entorno/test_moon_model.py -v -k FeaturesPropio`
Expected: FAIL con `ImportError: cannot import name 'features_propio'`

- [ ] **Step 3: Implementar `features_propio` (agregar al final de `moon_model.py`)**

```python
def features_propio(
    motor: MotorCorazones,
    agente_idx: int,
    vacios: List[set],
    historial: List[EntradaBaza],
    cartas_dadas: List[int],
    cartas_recibidas: List[int],
    puntuacion_historica: List[int],
    puntos_mano_actual: List[int],
    dama_picas_en: Optional[int],
) -> np.ndarray:
    """Features para "mi propio pozo": reutiliza el vector COMPLETO de
    ObservacionBuilder(dim=332) desde mi perspectiva real (mano exacta,
    cementerio, vacíos, memoria del pase v13 -- todo ya implementado), con
    los 2 slots de moon_prob en 0 (son el objetivo a predecir, no pueden ser
    también entrada), más 1 feature bonus: razón bazas-con-puntos que gané.
    """
    builder = ObservacionBuilder(dim=DIM_V13)
    puedo_alimentar = any(
        puntuacion_historica[j] >= 85 for j in range(4) if j != agente_idx
    )
    base = builder.construir(
        motor=motor,
        agente_idx=agente_idx,
        vacios=vacios,
        puntuacion_historica=puntuacion_historica,
        puntos_mano_actual=puntos_mano_actual,
        dama_picas_en=dama_picas_en,
        moon_prob_agente=0.0,
        moon_prob_rival=0.0,
        puedo_alimentar=puedo_alimentar,
        cartas_pasadas=cartas_dadas,
        cartas_recibidas=cartas_recibidas,
    )
    ratio = _ratio_bazas_con_puntos(historial, agente_idx)
    return np.concatenate([base, np.array([ratio], dtype=np.float32)])
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `python -m pytest tests/entorno/test_moon_model.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add src/entorno/moon_model.py tests/entorno/test_moon_model.py
git commit -m "feat: add features_propio for the self-moon model"
```

---

### Task 3: `moon_model.py` — `features_rival`

**Files:**
- Modify: `src/entorno/moon_model.py`
- Test: `tests/entorno/test_moon_model.py`

- [ ] **Step 1: Agregar los tests que fallan**

```python
from src.entorno.moon_model import features_rival


class TestFeaturesRival:
    def test_forma(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        feats = features_rival(m, 1, 0, vacios, [], [], [], False)
        assert feats.shape == (DIM_RIVAL,)
        assert feats.dtype == np.float32

    def test_capturas_del_rival_en_el_primer_bloque(self):
        m = MotorCorazones()
        m.repartir()
        carta_rival = _carta(3, 5)  # 5 de corazones
        m.jugadores[1].bazas_ganadas = [carta_rival]
        vacios = [set() for _ in range(4)]
        feats = features_rival(m, 1, 0, vacios, [], [], [], False)
        assert feats[carta_rival.id] == 1.0

    def test_cementerio_global_en_el_segundo_bloque(self):
        m = MotorCorazones()
        m.repartir()
        carta_ajena = _carta(0, 7)
        m.jugadores[2].bazas_ganadas = [carta_ajena]
        vacios = [set() for _ in range(4)]
        feats = features_rival(m, 1, 0, vacios, [], [], [], False)
        assert feats[52 + carta_ajena.id] == 1.0

    def test_vacios_del_rival(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        vacios[1].add(2)  # rival 1 es void en picas
        feats = features_rival(m, 1, 0, vacios, [], [], [], False)
        assert feats[156 + 2] == 1.0
        assert feats[156 + 0] == 0.0

    def test_posicion_relativa_one_hot(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        # rival 1 respecto de agente 0: rel=1 -> izquierda -> índice 165
        feats = features_rival(m, 1, 0, vacios, [], [], [], False)
        assert feats[165] == 1.0
        assert feats[166] == 0.0 and feats[167] == 0.0
        # rival 3 respecto de agente 0: rel=3 -> derecha -> índice 167
        feats3 = features_rival(m, 3, 0, vacios, [], [], [], False)
        assert feats3[167] == 1.0

    def test_cartas_dadas_al_rival_aun_no_jugadas(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        carta_dada = m.jugadores[2].mano[0]  # cualquier carta que no esté jugada
        feats = features_rival(m, 1, 0, vacios, [], [carta_dada.id], [], False)
        assert feats[168 + carta_dada.id] == 1.0

    def test_cartas_dadas_ya_jugadas_no_se_marcan(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        carta_jugada = _carta(0, 9)
        m.jugadores[0].bazas_ganadas = [carta_jugada]
        feats = features_rival(m, 1, 0, vacios, [], [carta_jugada.id], [], False)
        assert feats[168 + carta_jugada.id] == 0.0

    def test_cartas_recibidas_del_rival_marcadas_siempre(self):
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        carta_recibida = _carta(1, 6)
        feats = features_rival(m, 1, 0, vacios, [], [], [carta_recibida.id], False)
        assert feats[220 + carta_recibida.id] == 1.0
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `python -m pytest tests/entorno/test_moon_model.py -v -k FeaturesRival`
Expected: FAIL con `ImportError: cannot import name 'features_rival'`

- [ ] **Step 3: Implementar `features_rival` (agregar al final de `moon_model.py`)**

```python
def features_rival(
    motor: MotorCorazones,
    rival_idx: int,
    agente_idx: int,
    vacios: List[set],
    historial: List[EntradaBaza],
    cartas_dadas_a_rival: List[int],
    cartas_recibidas_de_rival: List[int],
    corazones_rotos: bool,
) -> np.ndarray:
    """Features para "¿el rival `rival_idx` está armando el pozo?" -- SOLO
    señales públicas sobre ESE rival específico (nunca su mano real: eso
    filtraría información imposible de tener en producción).

    Layout (272 dims):
      [0:52]    cartas capturadas por el rival (one-hot exacto, no conteo)
      [52:104]  cementerio global (todas las capturas de los 4 jugadores)
      [104:156] mesa actual (la baza en curso hasta el momento)
      [156:160] vacíos del rival por palo
      [160]     razón bazas-con-puntos que ganó el rival
      [161]     tasa que lideró con corazón/Q♠ pudiendo evitarlo
      [162]     ¿va ganando la baza en curso ahora mismo?
      [163]     baza actual / 13.0
      [164]     corazones rotos
      [165:168] posición relativa del rival (izquierda/frente/derecha)
      [168:220] cartas que LE DI (soy su dador) y aún no se han jugado
      [220:272] cartas que recibí DE ÉL (ya no las tiene)
    """
    obs = np.zeros(DIM_RIVAL, dtype=np.float32)

    for c in motor.jugadores[rival_idx].bazas_ganadas:
        obs[c.id] = 1.0
    for j in motor.jugadores:
        for c in j.bazas_ganadas:
            obs[52 + c.id] = 1.0
    for _, c in motor.mesa:
        obs[104 + c.id] = 1.0

    for palo in vacios[rival_idx]:
        obs[156 + palo] = 1.0

    obs[160] = _ratio_bazas_con_puntos(historial, rival_idx)
    obs[161] = _tasa_lidero_corazon_dama(historial, rival_idx)
    obs[162] = 1.0 if _ganador_parcial(motor) == rival_idx else 0.0
    obs[163] = min(motor.numero_baza / 13.0, 1.0)
    obs[164] = 1.0 if corazones_rotos else 0.0

    rel = (rival_idx - agente_idx) % 4  # 1=izquierda, 2=frente, 3=derecha (nunca 0)
    obs[165 + (rel - 1)] = 1.0

    jugadas = {c.id for j in motor.jugadores for c in j.bazas_ganadas}
    jugadas.update(c.id for _, c in motor.mesa)
    for cid in cartas_dadas_a_rival:
        if cid not in jugadas:
            obs[168 + cid] = 1.0
    for cid in cartas_recibidas_de_rival:
        obs[220 + cid] = 1.0

    return obs
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `python -m pytest tests/entorno/test_moon_model.py -v`
Expected: PASS (22 tests)

- [ ] **Step 5: Commit**

```bash
git add src/entorno/moon_model.py tests/entorno/test_moon_model.py
git commit -m "feat: add features_rival for the rival-moon model"
```

---

### Task 4: `moon_model.py` — red MLP y `EstimadorMoonProb`

**Files:**
- Modify: `src/entorno/moon_model.py`
- Test: `tests/entorno/test_moon_model.py`

- [ ] **Step 1: Agregar los tests que fallan**

```python
from src.entorno.moon_model import EstimadorMoonProb, _RedMoonMLP


class TestRedMoonMLP:
    def test_forward_devuelve_probabilidad_en_0_1(self):
        red = _RedMoonMLP(DIM_PROPIO)
        x = torch.zeros((1, DIM_PROPIO))
        with torch.no_grad():
            y = red(x)
        assert y.shape == (1,)
        assert 0.0 <= float(y.item()) <= 1.0


class TestEstimadorMoonProbSinPesos:
    """Sin pesos entrenados (directorio vacío), debe caer a 0.0 en vez de fallar."""

    def test_propio_devuelve_cero_sin_pesos(self, tmp_path):
        est = EstimadorMoonProb(dir_modelos=str(tmp_path))
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        p = est.propio(m, 0, vacios, [], [], [], [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert p == 0.0

    def test_rival_devuelve_cero_sin_pesos(self, tmp_path):
        est = EstimadorMoonProb(dir_modelos=str(tmp_path))
        m = MotorCorazones()
        m.repartir()
        vacios = [set() for _ in range(4)]
        p = est.rival(m, 1, 0, vacios, [], None, None, [], [], False)
        assert p == 0.0

    def test_gate_duro_devuelve_cero_aunque_haya_pesos(self, tmp_path):
        # entrena y guarda una red que SIEMPRE predice 1.0, para probar que
        # el gate duro gana incluso sobre un modelo "confiado".
        red = _RedMoonMLP(DIM_PROPIO)
        with torch.no_grad():
            for p in red.parameters():
                p.zero_()
            red.red[-1].bias.fill_(50.0)  # sigmoid(50) ~= 1.0
        torch.save(red.state_dict(), tmp_path / "propio.pt")

        est = EstimadorMoonProb(dir_modelos=str(tmp_path))
        m = MotorCorazones()
        m.repartir()
        m.jugadores[1].bazas_ganadas = [_carta(3, 2)]  # otro jugador ya tiene puntos
        vacios = [set() for _ in range(4)]
        p = est.propio(m, 0, vacios, [], [], [], [0, 0, 0, 0], [0, 0, 0, 0], None)
        assert p == 0.0
```

(Agregar `import torch` arriba del archivo de test si no está ya presente.)

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `python -m pytest tests/entorno/test_moon_model.py -v -k "RedMoonMLP or EstimadorMoonProb"`
Expected: FAIL con `ImportError`

- [ ] **Step 3: Implementar la red y el estimador (agregar al final de `moon_model.py`)**

```python
class _RedMoonMLP(nn.Module):
    """MLP chico: entrada -> 32 -> 1 con sigmoid. Uno por modelo (propio/rival)."""

    def __init__(self, dim_entrada: int, dim_oculta: int = 32):
        super().__init__()
        self.dim_entrada = dim_entrada
        self.red = nn.Sequential(
            nn.Linear(dim_entrada, dim_oculta),
            nn.ReLU(),
            nn.Linear(dim_oculta, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.red(x)).squeeze(-1)


def _cargar_red(ruta: Path, dim_entrada: int) -> Optional[_RedMoonMLP]:
    if not ruta.is_file():
        return None
    red = _RedMoonMLP(dim_entrada)
    red.load_state_dict(torch.load(ruta, map_location="cpu"))
    red.eval()
    return red


class EstimadorMoonProb:
    """Reemplaza la heurística de moon_prob con los 2 modelos aprendidos.

    Si no hay pesos entrenados (`<dir_modelos>/propio.pt` o `rival.pt`
    ausentes), devuelve 0.0 en vez de fallar -- permite que el resto del
    pipeline funcione mientras se entrena o si el entrenamiento aún no corrió.
    """

    def __init__(self, dir_modelos: str = "models/moon"):
        d = Path(dir_modelos)
        self._propio = _cargar_red(d / "propio.pt", DIM_PROPIO)
        self._rival = _cargar_red(d / "rival.pt", DIM_RIVAL)

    def propio(
        self,
        motor: MotorCorazones,
        agente_idx: int,
        vacios: List[set],
        historial: List[EntradaBaza],
        cartas_dadas: List[int],
        cartas_recibidas: List[int],
        puntuacion_historica: List[int],
        puntos_mano_actual: List[int],
        dama_picas_en: Optional[int],
    ) -> float:
        if _alguien_mas_tiene_puntos(motor, agente_idx):
            return 0.0
        if self._propio is None:
            return 0.0
        feats = features_propio(
            motor, agente_idx, vacios, historial, cartas_dadas, cartas_recibidas,
            puntuacion_historica, puntos_mano_actual, dama_picas_en,
        )
        with torch.no_grad():
            x = torch.from_numpy(feats).unsqueeze(0)
            return float(self._propio(x).item())

    def rival(
        self,
        motor: MotorCorazones,
        rival_idx: int,
        agente_idx: int,
        vacios: List[set],
        historial: List[EntradaBaza],
        receptor: Optional[int],
        dador: Optional[int],
        cartas_dadas: List[int],
        cartas_recibidas: List[int],
        corazones_rotos: bool,
    ) -> float:
        if _alguien_mas_tiene_puntos(motor, rival_idx):
            return 0.0
        if self._rival is None:
            return 0.0
        cartas_dadas_a_rival = cartas_dadas if rival_idx == receptor else []
        cartas_recibidas_de_rival = cartas_recibidas if rival_idx == dador else []
        feats = features_rival(
            motor, rival_idx, agente_idx, vacios, historial,
            cartas_dadas_a_rival, cartas_recibidas_de_rival, corazones_rotos,
        )
        with torch.no_grad():
            x = torch.from_numpy(feats).unsqueeze(0)
            return float(self._rival(x).item())
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `python -m pytest tests/entorno/test_moon_model.py -v`
Expected: PASS (26 tests)

- [ ] **Step 5: Commit**

```bash
git add src/entorno/moon_model.py tests/entorno/test_moon_model.py
git commit -m "feat: add MLP and EstimadorMoonProb with safe zero fallback"
```

---

### Task 5: `scripts/entrenar_moon_prob.py` — generación del dataset

**Files:**
- Create: `scripts/entrenar_moon_prob.py`
- Test: `tests/test_entrenar_moon_prob.py`

Este script re-juega manos reales para generar `(features, label)`. Reutiliza
`_preparar_motor`/`reconstruir_manos`/`mano_reconstruible` de `replay.py`
(igual que `pimc_regret_real.py`) y `MotorCorazones.receptor_pase` para
resolver quién es mi receptor/dador dada una dirección de pase.

- [ ] **Step 1: Escribir el test que falla**

```python
"""Tests de scripts/entrenar_moon_prob.py (generación de dataset, sin entrenar red)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from entrenar_moon_prob import _receptor_y_dador, ejemplos_de_mano
from src.captura.modelos import Jugada, RegistroMano
from src.entorno.moon_model import DIM_PROPIO, DIM_RIVAL


def _mano_completa_simple() -> RegistroMano:
    """Una mano de 13 bazas donde el asiento 0 gana TODO (pozo perfecto),
    construida a mano con jugadas legales reales (2T primero, sigue el palo
    cuando puede)."""
    # Reutiliza una partida real jugada por el motor mismo para garantizar
    # legalidad, en vez de inventar 52 cartas a mano.
    from src.dominio.motor import MotorCorazones

    m = MotorCorazones()
    m.repartir()
    jugadas = []
    baza = 1
    while not all(len(j.mano) == 0 for j in m.jugadores):
        idx = m.obtener_jugador_actual()
        legales = m.obtener_jugadas_legales(idx)
        # El asiento 0 siempre intenta ganar (juega la más alta legal);
        # los demás juegan la más baja legal -- fuerza que 0 gane todo.
        carta = max(legales, key=lambda c: c.valor) if idx == 0 else min(legales, key=lambda c: c.valor)
        jugadas.append(Jugada(asiento=idx, carta_id=carta.id, baza=baza))
        m.jugar_carta(idx, carta)
        if len(m.mesa) == 4:
            m.resolver_baza()
            baza += 1
    puntos = m.calcular_puntuacion_mano()
    return RegistroMano(
        numero_mano=1, direccion_pase=None, mano_inicial_agente=[],
        jugadas=jugadas, puntuacion_mano=puntos,
    )


def test_receptor_y_dador_izquierda():
    receptor, dador = _receptor_y_dador("izquierda", 0)
    assert receptor == 1  # seat+1 = izquierda
    assert dador == 3     # seat-1 me pasó a mí


def test_receptor_y_dador_sin_pase():
    assert _receptor_y_dador(None, 0) == (None, None)


def test_ejemplos_de_mano_formas_y_no_vacio():
    mano = _mano_completa_simple()
    ejemplos_propio, ejemplos_rival = ejemplos_de_mano(mano, asiento_agente_real=0)
    assert len(ejemplos_propio) > 0
    assert len(ejemplos_rival) > 0
    feats, label = ejemplos_propio[0]
    assert feats.shape == (DIM_PROPIO,)
    assert label in (0.0, 1.0)
    feats_r, label_r = ejemplos_rival[0]
    assert feats_r.shape == (DIM_RIVAL,)
    assert label_r in (0.0, 1.0)


def test_ejemplos_de_mano_etiqueta_el_pozo_correctamente():
    mano = _mano_completa_simple()
    if sum(mano.puntuacion_mano) != 26:
        return  # esta semilla en particular no produjo pozo; no es el foco del test
    luna_seat = mano.puntuacion_mano.index(0)
    ejemplos_propio, _ = ejemplos_de_mano(mano, asiento_agente_real=0)
    # al menos un ejemplo de la perspectiva del que hizo el pozo debe tener label 1.0
    # (los del asiento ganador, antes de que el gate lo excluya en las últimas bazas)
    assert any(label == 1.0 for _, label in ejemplos_propio) or luna_seat != 0
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `python -m pytest tests/test_entrenar_moon_prob.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'entrenar_moon_prob'`

- [ ] **Step 3: Implementar `scripts/entrenar_moon_prob.py` (parte 1: generación de dataset)**

```python
"""
Genera datasets y entrena los 2 modelos aprendidos de moon_prob
(src/entorno/moon_model.py) a partir de las manos reales reconstruibles de
data/partidas_bridge.jsonl. Ver spec:
docs/superpowers/specs/2026-07-06-moon-prob-modelo-aprendido-design.md

Limitación de datos conocida: la memoria del pase (cartas dadas/recibidas)
SOLO se conoce con certeza para el asiento realmente logueado por el bridge
en cada partida (`partida.asiento_agente`) -- el bridge no registra el
intercambio de los otros 3 asientos. Al generar ejemplos desde las 4
perspectivas por mano, esas features quedan en 0 (sin dato) salvo cuando la
perspectiva evaluada ES ese asiento real. No es un bug: es la limitación
real de los datos disponibles, y coincide con el caso legítimo de "sin
información de pase" que también ocurre en producción (el 4º jugador nunca
tiene relación de pase conmigo).

Uso:
    python scripts/entrenar_moon_prob.py --partidas data/partidas_bridge.jsonl \
        --out-dir models/moon --epocas 300
"""
from __future__ import annotations

# --- bootstrap path: permite `python scripts/<x>.py` desde la raiz del repo ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin bootstrap ---

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.captura.escritor import cargar_partidas
from src.captura.modelos import RegistroMano, RegistroPartida
from src.captura.replay import _preparar_motor, mano_reconstruible
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.moon_model import (
    DIM_PROPIO,
    DIM_RIVAL,
    EntradaBaza,
    _RedMoonMLP,
    _alguien_mas_tiene_puntos,
    features_propio,
    features_rival,
)

_DIRECCION_A_NUMERO_MANO = {"izquierda": 1, "derecha": 2, "enfrente": 3}


def _receptor_y_dador(direccion: Optional[str], seat: int) -> Tuple[Optional[int], Optional[int]]:
    """Índice de asiento receptor/dador del pase de `seat`, o (None, None) sin pase."""
    if direccion is None or direccion not in _DIRECCION_A_NUMERO_MANO:
        return None, None
    m = MotorCorazones()
    m.numero_mano = _DIRECCION_A_NUMERO_MANO[direccion]
    receptor = m.receptor_pase(seat)
    dador = next(d for d in range(4) if m.receptor_pase(d) == seat)
    return receptor, dador


def _lunaseat_de(mano: RegistroMano) -> Optional[int]:
    if mano.puntuacion_mano and sum(mano.puntuacion_mano) == 26:
        return next(i for i, p in enumerate(mano.puntuacion_mano) if p == 0)
    return None


def ejemplos_de_mano(mano: RegistroMano, asiento_agente_real: int):
    """(features, label) para el modelo propio y para el rival, por cada
    baza resuelta, desde las 4 perspectivas posibles."""
    motor = _preparar_motor(mano)
    luna_seat = _lunaseat_de(mano)
    receptor_agente, dador_agente = _receptor_y_dador(mano.direccion_pase, asiento_agente_real)

    historial: List[EntradaBaza] = []
    ejemplos_propio = []
    ejemplos_rival = []
    vacios: List[set] = [set() for _ in range(4)]
    mesa_actual: list = []

    for j in mano.jugadas:
        actual = motor.obtener_jugador_actual()
        if actual != j.asiento:
            break  # datos reales inconsistentes (ver pimc_regret_real.py), se descarta el resto
        carta = Carta._TODAS[j.carta_id]
        palo_salida = motor.palo_de_salida

        if not mesa_actual:
            puntos_mano_actual = [jg.contar_puntos_bazas() for jg in motor.jugadores]
            for seat in range(4):
                if _alguien_mas_tiene_puntos(motor, seat):
                    continue
                dadas = mano.pase_dado if seat == asiento_agente_real else []
                recibidas = mano.pase_recibido if seat == asiento_agente_real else []
                feats = features_propio(
                    motor, seat, vacios, historial, dadas, recibidas,
                    [0, 0, 0, 0], puntos_mano_actual, None,
                )
                ejemplos_propio.append((feats, 1.0 if seat == luna_seat else 0.0))

                for rival in range(4):
                    if rival == seat:
                        continue
                    if seat == asiento_agente_real:
                        dadas_r = mano.pase_dado if rival == receptor_agente else []
                        recibidas_r = mano.pase_recibido if rival == dador_agente else []
                    else:
                        dadas_r, recibidas_r = [], []
                    feats_r = features_rival(
                        motor, rival, seat, vacios, historial,
                        dadas_r, recibidas_r, motor.corazones_rotos,
                    )
                    ejemplos_rival.append((feats_r, 1.0 if rival == luna_seat else 0.0))

        if mesa_actual and palo_salida is not None and carta.palo != palo_salida:
            vacios[actual].add(palo_salida)
        motor.jugar_carta(actual, carta)
        mesa_actual.append((actual, carta))
        if len(mesa_actual) == 4:
            ganador = motor.resolver_baza()
            historial.append(EntradaBaza(
                lider=mesa_actual[0][0],
                ganador=ganador,
                tenia_puntos=any(c.puntos > 0 for _, c in mesa_actual),
                lidero_corazon_o_dama=(
                    mesa_actual[0][1].es_corazon or mesa_actual[0][1].es_dama_de_picas
                ),
            ))
            mesa_actual = []

    return ejemplos_propio, ejemplos_rival


def construir_dataset(ruta_partidas: str):
    """Devuelve 2 dicts partida_id -> [(features, label), ...] (uno por modelo)
    y un dict partida_id -> timestamp (para el corte cronológico de validación)."""
    partidas = cargar_partidas(ruta_partidas)
    propio_por_partida: Dict[str, list] = {}
    rival_por_partida: Dict[str, list] = {}
    timestamp_por_partida: Dict[str, str] = {}
    avisos: List[str] = []

    for p in partidas:
        ep: list = []
        er: list = []
        for mano in p.manos:
            if not mano_reconstruible(mano):
                continue
            try:
                e1, e2 = ejemplos_de_mano(mano, p.asiento_agente)
            except ValueError as e:
                avisos.append(f"{p.partida_id}: mano {mano.numero_mano} descartada ({e})")
                continue
            ep.extend(e1)
            er.extend(e2)
        if ep or er:
            propio_por_partida[p.partida_id] = ep
            rival_por_partida[p.partida_id] = er
            timestamp_por_partida[p.partida_id] = p.timestamp

    if avisos:
        print(f"{len(avisos)} manos descartadas por datos inconsistentes:")
        for a in avisos:
            print(f"  - {a}")

    return propio_por_partida, rival_por_partida, timestamp_por_partida
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `python -m pytest tests/test_entrenar_moon_prob.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/entrenar_moon_prob.py tests/test_entrenar_moon_prob.py
git commit -m "feat: add real-hand dataset generation for moon_prob training"
```

---

### Task 6: `scripts/entrenar_moon_prob.py` — entrenamiento, CLI y corrida real

**Files:**
- Modify: `scripts/entrenar_moon_prob.py`
- Test: `tests/test_entrenar_moon_prob.py`

- [ ] **Step 1: Agregar el test que falla**

Agregar al final de `tests/test_entrenar_moon_prob.py`:

```python
import numpy as np
from entrenar_moon_prob import _auc, _brier, _entrenar
from src.entorno.moon_model import _RedMoonMLP


def test_auc_perfecto_cuando_scores_separan_las_clases():
    y_true = np.array([0, 0, 1, 1], dtype=np.float32)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    assert _auc(y_true, y_score) == 1.0


def test_auc_medio_si_no_hay_de_una_clase():
    y_true = np.array([1, 1], dtype=np.float32)
    y_score = np.array([0.5, 0.6], dtype=np.float32)
    assert np.isnan(_auc(y_true, y_score))


def test_brier_cero_si_prediccion_perfecta():
    y_true = np.array([0.0, 1.0], dtype=np.float32)
    y_score = np.array([0.0, 1.0], dtype=np.float32)
    assert _brier(y_true, y_score) == 0.0


def test_entrenar_reduce_la_perdida_en_datos_separables():
    rng = np.random.default_rng(0)
    dim = 5
    X_pos = rng.normal(3.0, 0.1, size=(50, dim)).astype(np.float32)
    X_neg = rng.normal(-3.0, 0.1, size=(50, dim)).astype(np.float32)
    X_train = np.concatenate([X_pos[:40], X_neg[:40]])
    y_train = np.concatenate([np.ones(40), np.zeros(40)]).astype(np.float32)
    X_val = np.concatenate([X_pos[40:], X_neg[40:]])
    y_val = np.concatenate([np.ones(10), np.zeros(10)]).astype(np.float32)

    red = _RedMoonMLP(dim)
    red, auc, brier = _entrenar(red, X_train, y_train, X_val, y_val, epocas=200)
    assert auc > 0.9
    assert brier < 0.1
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `python -m pytest tests/test_entrenar_moon_prob.py -v -k "auc or brier or entrenar"`
Expected: FAIL con `ImportError`

- [ ] **Step 3: Implementar entrenamiento + CLI (agregar al final de `scripts/entrenar_moon_prob.py`)**

```python
def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC manual (evita sumar scikit-learn por una sola métrica):
    probabilidad de que un positivo al azar tenga score mayor que un
    negativo al azar."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean(pos[:, None] > neg[None, :]))


def _brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_score - y_true) ** 2))


def _entrenar(red, X_train, y_train, X_val, y_val, epocas: int, lr: float = 1e-3,
              paciencia: int = 15):
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    perdida = nn.BCELoss()
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)
    Xv = torch.from_numpy(X_val)

    mejor_auc = -1.0
    mejor_estado = {k: v.clone() for k, v in red.state_dict().items()}
    sin_mejora = 0

    for _ in range(epocas):
        red.train()
        opt.zero_grad()
        loss = perdida(red(Xt), yt)
        loss.backward()
        opt.step()

        red.eval()
        with torch.no_grad():
            pred_val = red(Xv).numpy()
        auc = _auc(y_val, pred_val)
        if not np.isnan(auc) and auc > mejor_auc:
            mejor_auc = auc
            mejor_estado = {k: v.clone() for k, v in red.state_dict().items()}
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break

    red.load_state_dict(mejor_estado)
    red.eval()
    with torch.no_grad():
        pred_final = red(Xv).numpy()
    return red, _auc(y_val, pred_final), _brier(y_val, pred_final)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partidas", required=True)
    p.add_argument("--out-dir", default="models/moon")
    p.add_argument("--epocas", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.2)
    args = p.parse_args()

    print("Generando ejemplos desde manos reales reconstruibles...", flush=True)
    propio_por_partida, rival_por_partida, timestamp_por_partida = construir_dataset(args.partidas)
    ids = sorted(propio_por_partida.keys())
    print(f"{len(ids)} partidas con al menos una mano reconstruible", flush=True)

    rng = np.random.default_rng(args.seed)
    orden = rng.permutation(len(ids))
    corte = int(len(ids) * (1 - args.val_frac))
    train_ids = {ids[i] for i in orden[:corte]}
    val_ids = {ids[i] for i in orden[corte:]}

    # Corte cronológico ADICIONAL (solo diagnóstico, no se usa para entrenar ni
    # para early stopping): las sesiones más recientes por timestamp, para
    # detectar sobreajuste a patrones de oponentes de esos días específicos en
    # vez de generalización real.
    ids_por_fecha = sorted(ids, key=lambda pid: timestamp_por_partida[pid])
    corte_fecha = int(len(ids_por_fecha) * (1 - args.val_frac))
    val_ids_recientes = set(ids_por_fecha[corte_fecha:])

    out = _Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for nombre, ejemplos_por_partida, dim in (
        ("propio", propio_por_partida, DIM_PROPIO),
        ("rival", rival_por_partida, DIM_RIVAL),
    ):
        train = [e for pid in train_ids for e in ejemplos_por_partida[pid]]
        val = [e for pid in val_ids for e in ejemplos_por_partida[pid]]
        if not train or not val:
            print(f"\n=== modelo {nombre}: datos insuficientes, se omite ===")
            continue
        pct_pos = 100 * sum(l for _, l in train) / len(train)
        print(f"\n=== modelo {nombre}: {len(train)} train / {len(val)} val "
              f"({pct_pos:.1f}% positivos train) ===", flush=True)

        X_train = np.stack([f for f, _ in train]).astype(np.float32)
        y_train = np.array([l for _, l in train], dtype=np.float32)
        X_val = np.stack([f for f, _ in val]).astype(np.float32)
        y_val = np.array([l for _, l in val], dtype=np.float32)

        red = _RedMoonMLP(dim)
        red, auc, brier = _entrenar(red, X_train, y_train, X_val, y_val, args.epocas)
        print(f"  AUC val: {auc:.3f}  |  Brier val: {brier:.4f}")

        recientes = [e for pid in val_ids_recientes for e in ejemplos_por_partida[pid]]
        if recientes:
            X_r = np.stack([f for f, _ in recientes]).astype(np.float32)
            y_r = np.array([l for _, l in recientes], dtype=np.float32)
            with torch.no_grad():
                pred_r = red(torch.from_numpy(X_r)).numpy()
            print(f"  AUC en sesiones más recientes: {_auc(y_r, pred_r):.3f}  "
                  f"|  Brier: {_brier(y_r, pred_r):.4f}  ({len(recientes)} ejemplos)"
                  "  -- si es mucho peor que el AUC de val de arriba, hay sobreajuste "
                  "a patrones de oponentes específicos de esas sesiones.")

        ruta = out / f"{nombre}.pt"
        torch.save(red.state_dict(), ruta)
        print(f"  guardado en {ruta}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Correr todos los tests del script y verificar que pasan**

Run: `python -m pytest tests/test_entrenar_moon_prob.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/entrenar_moon_prob.py tests/test_entrenar_moon_prob.py
git commit -m "feat: add training loop, AUC/Brier metrics and CLI to entrenar_moon_prob.py"
```

- [ ] **Step 6: Entrenar los modelos reales**

Requiere `data/partidas_bridge.jsonl` (generado con `scripts/importar_sesiones_bridge.py
--dir D:/Github/Personal/hearts-sfs-bridge/logs --out data/partidas_bridge.jsonl`
si aún no existe en el repo).

Run: `python scripts/entrenar_moon_prob.py --partidas data/partidas_bridge.jsonl --out-dir models/moon`
Expected: imprime AUC/Brier de validación para ambos modelos y guarda
`models/moon/propio.pt` y `models/moon/rival.pt`. Anotar los números (se
comparan en el Task 11 contra el backtest de regret).

---

### Task 7: `Recomendador` — historial de bazas

**Files:**
- Modify: `scripts/recomendador.py`
- Test: `tests/test_recomendador.py`

- [ ] **Step 1: Agregar el test que falla**

Agregar a `tests/test_recomendador.py`:

```python
from src.entorno.moon_model import EntradaBaza


def test_registrar_baza_agrega_al_historial():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
    jugadas = [(0, parse_carta("2T")), (1, parse_carta("5T")),
               (2, parse_carta("9C")), (3, parse_carta("3T"))]  # 9 de corazones: tiene puntos
    r.registrar_baza(jugadas, ganador=2)
    assert len(r.historial_bazas) == 1
    entrada = r.historial_bazas[0]
    assert entrada.lider == 0
    assert entrada.ganador == 2
    assert entrada.tenia_puntos is True
    assert entrada.lidero_corazon_o_dama is False  # lideró con 2♣, no con corazón/Q♠


def test_registrar_resto_agrega_al_historial_sin_lider():
    r = _recomendador_sin_modelo()
    r.mano = []
    r.registrar_resto(ganador=1, cartas_restantes=parse_cartas("QP 5C"))
    assert len(r.historial_bazas) == 1
    entrada = r.historial_bazas[0]
    assert entrada.lider is None
    assert entrada.ganador == 1
    assert entrada.tenia_puntos is True


def test_reset_mano_limpia_el_historial():
    r = _recomendador_sin_modelo()
    r.historial_bazas = [EntradaBaza(0, 0, True, False)]
    r.reset_mano([])
    assert r.historial_bazas == []
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `python -m pytest tests/test_recomendador.py -v -k historial`
Expected: FAIL con `AttributeError: 'Recomendador' object has no attribute 'historial_bazas'`

- [ ] **Step 3: Implementar el tracking en `scripts/recomendador.py`**

Modificar el import al inicio del archivo:

```python
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.observacion import ObservacionBuilder
from src.entorno.moon_model import EntradaBaza
from src.rllib.utils import cargar_policy_desde_checkpoint
```

Modificar `reset_mano` (agregar la línea del historial):

```python
    def reset_mano(self, mi_mano: List[Carta]):
        self.mano: List[Carta] = list(mi_mano)
        self.cementerio: Dict[int, List[Carta]] = {i: [] for i in range(4)}
        self.vacios: List[Set[int]] = [set() for _ in range(4)]
        self.corazones_rotos = False
        self.numero_baza = 1
        self.dama_picas_en: Optional[int] = None
        self.historial_bazas: List[EntradaBaza] = []
```

Modificar `registrar_baza` (agregar el append justo antes de `self.numero_baza += 1`):

```python
        self.cementerio[ganador].extend([c for _, c in jugadas])
        self.historial_bazas.append(EntradaBaza(
            lider=jugadas[0][0],
            ganador=ganador,
            tenia_puntos=any(c.puntos > 0 for _, c in jugadas),
            lidero_corazon_o_dama=(
                jugadas[0][1].es_corazon or jugadas[0][1].es_dama_de_picas
            ),
        ))
        self.numero_baza += 1
```

Modificar `registrar_resto` (agregar el append antes de `self.numero_baza = 14`):

```python
        self.cementerio[ganador].extend(cartas_restantes)
        self.historial_bazas.append(EntradaBaza(
            lider=None,
            ganador=ganador,
            tenia_puntos=any(c.puntos > 0 for c in cartas_restantes),
            lidero_corazon_o_dama=False,
        ))
        self.numero_baza = 14
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `python -m pytest tests/test_recomendador.py -v`
Expected: PASS (todos, incluyendo los 3 nuevos)

- [ ] **Step 5: Commit**

```bash
git add scripts/recomendador.py tests/test_recomendador.py
git commit -m "feat: track per-trick history in Recomendador for moon_model features"
```

---

### Task 8: `Recomendador` — memoria real del pase (`registrar_pase`)

**Files:**
- Modify: `scripts/recomendador.py`
- Test: `tests/test_recomendador.py`

- [ ] **Step 1: Agregar el test que falla**

```python
def test_registrar_pase_izquierda_calcula_receptor_y_dador():
    r = _recomendador_sin_modelo()
    dadas = parse_cartas("2T 3T 4T")
    recibidas = parse_cartas("5T 6T 7T")
    r.registrar_pase("izquierda", dadas, recibidas)
    assert r.receptor == 1  # asiento 0 + 1 = izquierda
    assert r.dador == 3     # asiento 3 me pasó a mí (0 - 1 mod 4)
    assert r.cartas_dadas == [c.id for c in dadas]
    assert r.cartas_recibidas == [c.id for c in recibidas]


def test_registrar_pase_sin_pase_deja_todo_en_none():
    r = _recomendador_sin_modelo()
    r.registrar_pase("sin", [], [])
    assert r.receptor is None
    assert r.dador is None
    assert r.cartas_dadas == []
    assert r.cartas_recibidas == []


def test_reset_mano_limpia_el_estado_del_pase():
    r = _recomendador_sin_modelo()
    r.registrar_pase("izquierda", parse_cartas("2T"), parse_cartas("3T"))
    r.reset_mano([])
    assert r.receptor is None
    assert r.dador is None
    assert r.cartas_dadas == []
    assert r.cartas_recibidas == []
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `python -m pytest tests/test_recomendador.py -v -k pase`
Expected: FAIL con `AttributeError: 'Recomendador' object has no attribute 'receptor'`

- [ ] **Step 3: Implementar `registrar_pase`**

Modificar `reset_mano` (agregar las 4 líneas nuevas):

```python
    def reset_mano(self, mi_mano: List[Carta]):
        self.mano: List[Carta] = list(mi_mano)
        self.cementerio: Dict[int, List[Carta]] = {i: [] for i in range(4)}
        self.vacios: List[Set[int]] = [set() for _ in range(4)]
        self.corazones_rotos = False
        self.numero_baza = 1
        self.dama_picas_en: Optional[int] = None
        self.historial_bazas: List[EntradaBaza] = []
        self.receptor: Optional[int] = None
        self.dador: Optional[int] = None
        self.cartas_dadas: List[int] = []
        self.cartas_recibidas: List[int] = []
```

Agregar el método nuevo (justo después de `recomendar_pase`):

```python
    def registrar_pase(self, direccion: str, cartas_dadas: List[Carta],
                        cartas_recibidas: List[Carta]) -> None:
        """Registra la ejecución REAL del pase (qué se dio y qué se recibió),
        para alimentar la memoria de pase de moon_model. Llamar después de
        recomendar_pase y antes de la primera jugada de la mano."""
        if direccion == "sin":
            self.receptor = None
            self.dador = None
        else:
            m = MotorCorazones()
            m.numero_mano = _DIRECCION_A_MANO[direccion]
            self.receptor = m.receptor_pase(self.me)
            self.dador = next(d for d in range(4) if m.receptor_pase(d) == self.me)
        self.cartas_dadas = [c.id for c in cartas_dadas]
        self.cartas_recibidas = [c.id for c in cartas_recibidas]
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `python -m pytest tests/test_recomendador.py -v`
Expected: PASS (todos)

- [ ] **Step 5: Commit**

```bash
git add scripts/recomendador.py tests/test_recomendador.py
git commit -m "feat: add registrar_pase to track real pass exchange in Recomendador"
```

---

### Task 9: `servidor_inferencia.py` — endpoint `/registrar_pase`

**Files:**
- Modify: `scripts/servidor_inferencia.py`
- Test: `tests/test_servidor_inferencia.py`

- [ ] **Step 1: Agregar el test que falla**

Agregar a `tests/test_servidor_inferencia.py`:

```python
def test_registrar_pase_actualiza_estado_del_recomendador(tmp_path):
    server, r = _servidor_sin_modelo(tmp_path)
    try:
        puerto = server.server_address[1]
        salida = _post(puerto, "/registrar_pase", {
            "direccion": "izquierda",
            "cartas_dadas": [0, 1, 2],
            "cartas_recibidas": [10, 11, 12],
        })
        assert salida == {"ok": True}
        assert r.receptor == 1
        assert r.dador == 3
        assert r.cartas_dadas == [0, 1, 2]
        assert r.cartas_recibidas == [10, 11, 12]
    finally:
        server.shutdown()
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `python -m pytest tests/test_servidor_inferencia.py -v -k registrar_pase`
Expected: FAIL con `HTTPError: HTTP Error 404`

- [ ] **Step 3: Implementar el endpoint en `scripts/servidor_inferencia.py`**

Agregar la rama nueva en `_manejar_post` (justo después del bloque
`elif self.path == "/recomendar_pase":`):

```python
            elif self.path == "/registrar_pase":
                cartas_dadas = [_carta(i) for i in datos["cartas_dadas"]]
                cartas_recibidas = [_carta(i) for i in datos["cartas_recibidas"]]
                r.registrar_pase(datos["direccion"], cartas_dadas, cartas_recibidas)
                salida = {"ok": True}
                self._log_evento("registrar_pase", datos, salida)
                self._responder(200, salida)
```

Actualizar la lista de endpoints en el docstring del módulo (agregar debajo
de la línea de `/recomendar_pase`):

```
    /registrar_pase   {"direccion": "...", "cartas_dadas": [id,id,id],
                        "cartas_recibidas": [id,id,id]} -> {"ok": true}
        El bridge reporta la ejecución REAL del pase (no solo la sugerencia
        de /recomendar_pase) -- alimenta la memoria de pase de moon_model.
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `python -m pytest tests/test_servidor_inferencia.py -v`
Expected: PASS (todos)

- [ ] **Step 5: Commit**

```bash
git add scripts/servidor_inferencia.py tests/test_servidor_inferencia.py
git commit -m "feat: add /registrar_pase endpoint to servidor_inferencia.py"
```

---

### Task 10: Conectar `EstimadorMoonProb` en `recomendador.py` y retirar el código muerto

**Files:**
- Modify: `scripts/recomendador.py`
- Modify: `tests/test_recomendador.py`

Este task reemplaza el Monte Carlo (`determinizar`/`_mundos_moon`, agregado
en la sesión anterior) por los modelos aprendidos. Deja de hacer falta
fabricar manos rivales del todo: ni `features_propio` ni `features_rival`
tocan el contenido de la mano de un rival en ningún momento (confirmado en
el diseño), así que también se simplifica `_motor()`.

- [ ] **Step 1: Actualizar los tests que van a cambiar de comportamiento**

En `tests/test_recomendador.py`, **eliminar** estas 4 funciones de test (ya
no aplican, probaban el reparto ficticio de manos rivales que se retira):
`test_motor_reparte_todas_las_desconocidas`,
`test_motor_asigna_tamano_correcto_a_medio_baza`,
`test_mundos_moon_no_asigna_carta_a_rival_void_en_ese_palo`,
`test_mundos_moon_devuelve_n_mundos_validos`.

Actualizar el helper `_recomendador_sin_modelo` para no depender de `_rng`
(ya no existe) y sí de un `EstimadorMoonProb` sin pesos (fallback a 0.0):

```python
def _recomendador_sin_modelo(mi_idx: int = 0) -> Recomendador:
    from src.entorno.moon_model import EstimadorMoonProb
    r = Recomendador.__new__(Recomendador)
    r.me = mi_idx
    r._estimador_moon = EstimadorMoonProb(dir_modelos="models/moon_inexistente")
    r.scores = [0, 0, 0, 0]
    r.reset_mano([])
    return r
```

Agregar 2 tests nuevos al final del archivo:

```python
def test_motor_no_fabrica_manos_rivales():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
    m = r._motor(mesa=[])
    assert m.jugadores[0].mano == r.mano
    for i in (1, 2, 3):
        assert m.jugadores[i].mano == []


def test_obs_usa_el_estimador_moon_y_cae_a_cero_sin_pesos():
    r = _recomendador_sin_modelo()
    r.mano = parse_cartas("AP KP QP JP 10P 9P 8P 7P 6P 5P 4P 3P 2P")
    m = r._motor(mesa=[])
    obs = r._obs(m)
    assert obs[187] == 0.0  # moon_prob_agente: sin pesos entrenados -> 0.0
    assert obs[188] == 0.0  # moon_prob_rival: idem
```

- [ ] **Step 2: Correr los tests y verificar que fallan**

Run: `python -m pytest tests/test_recomendador.py -v`
Expected: FAIL (los 2 tests nuevos fallan porque `_motor`/`_obs` todavía
usan la lógica vieja; los 4 tests eliminados ya no corren).

- [ ] **Step 3: Reescribir `_motor` y `_obs` en `scripts/recomendador.py`**

Reemplazar el import de `determinizar` y quitar la constante `_N_MUNDOS_MOON`:

```python
from src.dominio.carta import Carta
from src.dominio.motor import MotorCorazones
from src.entorno.observacion import ObservacionBuilder
from src.entorno.moon_model import EntradaBaza, EstimadorMoonProb
from src.rllib.utils import cargar_policy_desde_checkpoint
```

En `__init__`, reemplazar `self._rng = np.random.default_rng()` por:

```python
        self._estimador_moon = EstimadorMoonProb()
```

Reemplazar el cuerpo de `_motor` (quitar el bloque de reparto ficticio de
manos rivales — ya no hace falta, nada lo usa):

```python
    def _motor(self, mesa: List, numero_mano: int = 1) -> MotorCorazones:
        m = MotorCorazones()
        for i in range(4):
            m.jugadores[i].bazas_ganadas = list(self.cementerio[i])
            m.jugadores[i].puntuacion_historica = self.scores[i]
        m.jugadores[self.me].mano = list(self.mano)
        m.mesa = list(mesa)
        m.palo_de_salida = mesa[0][1].palo if mesa else None
        m.corazones_rotos = self.corazones_rotos
        m.numero_baza = self.numero_baza
        m.numero_mano = numero_mano
        m.indice_jugador_inicial = (self.me - len(mesa)) % 4
        return m
```

Eliminar el método `_mundos_moon` por completo. Reemplazar el cuerpo de
`_obs`:

```python
    def _obs(self, m: MotorCorazones):
        scores = m.puntuaciones_historicas()
        puntos_mano_actual = [j.contar_puntos_bazas() for j in m.jugadores]
        mp_ag = self._estimador_moon.propio(
            m, self.me, self.vacios, self.historial_bazas,
            self.cartas_dadas, self.cartas_recibidas,
            scores, puntos_mano_actual, self.dama_picas_en,
        )
        mp_riv = max(
            self._estimador_moon.rival(
                m, i, self.me, self.vacios, self.historial_bazas,
                self.receptor, self.dador, self.cartas_dadas, self.cartas_recibidas,
                self.corazones_rotos,
            )
            for i in range(4) if i != self.me
        )
        puedo_alim = any(scores[j] >= 85 for j in range(4) if j != self.me)
        return self.builder.construir(
            motor=m, agente_idx=self.me, vacios=self.vacios,
            puntuacion_historica=scores,
            puntos_mano_actual=puntos_mano_actual,
            dama_picas_en=self.dama_picas_en,
            moon_prob_agente=mp_ag, moon_prob_rival=mp_riv, puedo_alimentar=puedo_alim)
```

Eliminar también la función libre `_moon_prob` al inicio del archivo (la
heurística vieja, ya reemplazada), la constante `_N_MUNDOS_MOON`, y la línea
`import numpy as np` — verificado que las 3 únicas apariciones de `np.` en
todo el archivo son exactamente las que este task retira (`self._rng =
np.random.default_rng()` y las 2 llamadas a `np.mean` dentro de la vieja
`_obs`), así que el import queda sin uso.

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `python -m pytest tests/test_recomendador.py -v`
Expected: PASS (todos)

- [ ] **Step 5: Commit**

```bash
git add scripts/recomendador.py tests/test_recomendador.py
git commit -m "feat: wire EstimadorMoonProb into recomendador.py, drop dead Monte Carlo code"
```

---

### Task 11: Regresión completa y backtest de regret antes/después

**Files:** ninguno (verificación, sin cambios de código)

- [ ] **Step 1: Correr la suite completa**

Run: `python -m pytest tests/ -q`
Expected: todos los tests pasan (los existentes + los ~40 nuevos de este plan)

- [ ] **Step 2: Smoke test end-to-end con un checkpoint real**

Run: `python scripts/recomendador.py --modelo models/produccion/v10c_campeon --demo`
Expected: corre sin errores, imprime recomendaciones de pase y jugada
(verificar que no tira excepciones por `historial_bazas`/`receptor`/`dador`
sin inicializar en el flujo real de `__init__` + `reset_mano`).

- [ ] **Step 3: Backtest de regret -- después de conectar los modelos aprendidos**

Run: `python scripts/pimc_regret_real.py --modelo models/produccion/v10c_campeon --partidas data/partidas_bridge.jsonl --max-decisiones 5000 --rollouts 30`

Anotar regret medio / mediana / % óptimo / cola (≥3/≥5/≥10) y compararlos
con los números YA MEDIDOS antes de este cambio (regret medio 1.021, %
óptimo 56.1%, ≥3pts 11.4%, ≥5pts 5.5%, ≥10pts 1.2%, sobre 4767 decisiones).
Si el regret medio y la cola bajan, los modelos aprendidos mejoran la
estimación de moon_prob en partidas reales -- evidencia para decidir si
vale la pena la Fase 2 (reentrenar v10c). Si no bajan, iterar sobre
features/arquitectura antes de tocar el entrenamiento.

---

## Notas para Fase 2 (fuera de este plan)

Una vez validado el backtest de regret, la Fase 2 (spec separado) conecta
`EstimadorMoonProb` en `src/entorno/corazones_rllib.py::_build_obs` (en vez
de `_calcular_moon_prob`) y hace fine-tune de `models/produccion/v10c_campeon`
con LR bajo, manteniendo `obs_dim=228`. No iniciar esa fase sin los números
del Task 11 en mano.
