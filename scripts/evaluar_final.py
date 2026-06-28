"""
Evaluación final post-entrenamiento (v10 — partida completa).

Carga el último snapshot de un modelo y produce un informe alineado con el
objetivo real (ganar la partida / quedar top-2), más un diagnóstico con
recomendaciones concretas de qué ajustar.

No requiere Ray (lee los pesos directamente del checkpoint).

Uso:
    python evaluar_final.py --dir models/v10
    python evaluar_final.py --dir models/v10 --snapshot models/v10/snapshots/snapshot_000010000000 --partidas 200

Informe:
  1. Win-rate / top-2 / puesto medio en PARTIDAS COMPLETAS vs cada tipo de bot.
  2. Chequeo de simetría self-play (vs 3 copias de sí mismo ≈ 25% win → sano).
  3. Diagnóstico de logs (entropía, anti-farming, tendencia).
  4. Recomendaciones priorizadas de ajuste.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

# Forzar UTF-8 en stdout para no romper en consolas Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.entorno.dimensiones import DIM_ENTORNO
from src.rllib.eval_bots import evaluar_vs_bots, _eval_model_vs_factory
from src.rllib.utils import cargar_policy_desde_checkpoint, listar_snapshots

console = Console()


def _leer_jsonl(path: str, tipo: Optional[str] = None) -> List[dict]:
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                d = json.loads(linea)
            except json.JSONDecodeError:
                continue
            if tipo is None or d.get("tipo") == tipo:
                out.append(d)
    return out


def _simetria_self_play(snap, n_partidas: int, obs_dim: int,
                        con_pase: bool = False) -> dict:
    """El agente juega contra 3 copias de sí mismo (vía env, obs completa). En un
    juego simétrico, un agente sano debería ganar ≈25% de las partidas.
    Desviaciones grandes sugieren sesgo de posición o un bug en el env/obs."""
    model = snap._get_model()

    def factory(ai=0):
        return {i: snap for i in range(4) if i != ai}

    res = _eval_model_vs_factory(model, factory, n_partidas, obs_dim,
                                 agente_idx=0, con_pase=con_pase)
    puestos = [r["puesto"] for r in res]
    win = sum(p == 1 for p in puestos) / len(puestos)
    top2 = sum(p <= 2 for p in puestos) / len(puestos)
    return {"win_rate": round(win, 4), "top2_rate": round(top2, 4)}


def _diagnostico(model_dir: str, bot_metrics: dict, simetria: dict) -> List[str]:
    """Genera recomendaciones priorizadas a partir de métricas y logs."""
    recs: List[str] = []
    log_dir = os.path.join(model_dir, "logs")
    metricas = _leer_jsonl(os.path.join(log_dir, "eval_log.jsonl"), "metricas")

    # Entropía final
    entropies = [m.get("entropy") for m in metricas
                 if isinstance(m.get("entropy"), (int, float))]
    if entropies:
        ent = entropies[-1]
        if ent < 0.4:
            recs.append(
                f"[P1] Entropía final baja ({ent:.3f}): la política colapsó. "
                "Subir entropy_coeff (p.ej. 0.05→0.1) o reducir num_epochs.")
        elif ent > 1.5:
            recs.append(
                f"[P2] Entropía alta al final ({ent:.3f}): no terminó de converger. "
                "Entrenar más pasos o bajar lr_end.")

    # Anti-farming: reward_medio vs r_terminal_medio
    pares = [(m.get("reward_medio"), m.get("r_terminal_medio")) for m in metricas
             if isinstance(m.get("reward_medio"), (int, float))
             and isinstance(m.get("r_terminal_medio"), (int, float))]
    if pares:
        rm, rt = pares[-1]
        if abs(rm - rt) > 0.5:
            recs.append(
                f"[P1] Brecha reward-terminal ({abs(rm-rt):.3f}): el shaping PBRS "
                "podría estar dominando. Bajar PHI_LAMBDA (0.5→0.3).")

    # Rendimiento vs experto (señal absoluta clave). Objetivo: top-2 consistente.
    top2_exp = bot_metrics.get("top2_rate_vs_experto", 0.0)
    win_exp = bot_metrics.get("win_rate_vs_experto", 0.0)
    if win_exp < 0.25:
        recs.append(
            f"[P1] Pierde contra BotExperto (win-rate {win_exp:.2f} < 0.25 = azar). "
            "El agente aún no domina. Más pasos, revisar curriculum o LR.")
    if top2_exp < 0.5:
        recs.append(
            f"[P2] Top-2 vs experto {top2_exp:.2f} < 0.5: por debajo del objetivo de "
            "consistencia top-2. Considerar más self-play o ajustar R_terminal (2º).")
    elif top2_exp >= 0.6 and win_exp >= 0.3:
        recs.append(
            f"[OK] Supera el objetivo top-2 vs experto (top2={top2_exp:.2f}, "
            f"win={win_exp:.2f}). Listo para v10b (añadir pase).")

    # Simetría self-play
    sim = simetria["win_rate"]
    if abs(sim - 0.25) > 0.1:
        recs.append(
            f"[P2] Self-play asimétrico (win-rate {sim:.2f} vs 0.25 esperado): "
            "posible sesgo de posición o feature inconsistente. Revisar random_position.")

    # Moon
    moon = bot_metrics.get("moon_rate", 0.0)
    if moon < 0.005:
        recs.append(
            "[P3] El agente casi nunca hace el pozo (moon_rate≈0). No es un fallo, "
            "pero podría faltar incentivo/contexto para intentarlo o defenderlo.")

    if not recs:
        recs.append("[OK] Sin banderas rojas. Métricas dentro de rangos esperados.")
    return recs


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluación final Hearts v10")
    p.add_argument("--dir", required=True, help="Directorio del modelo (p.ej. models/v10)")
    p.add_argument("--snapshot", default=None,
                   help="Checkpoint específico. Por defecto: el más reciente.")
    p.add_argument("--partidas", type=int, default=100,
                   help="Partidas por escenario (default 100)")
    p.add_argument("--obs-dim", type=int, default=DIM_ENTORNO)
    p.add_argument("--con-pase", action="store_true",
                   help="Evaluar con fase de pase (modelos v10b, obs 228).")
    args = p.parse_args()
    if args.con_pase:
        from src.entorno.dimensiones import DIM_V12
        args.obs_dim = max(args.obs_dim, DIM_V12)

    snapshot_dir = os.path.join(args.dir, "snapshots")
    if args.snapshot:
        ckpt = args.snapshot
    else:
        snaps = listar_snapshots(snapshot_dir)
        if not snaps:
            console.print(f"[red]No hay snapshots en {snapshot_dir}[/red]")
            return
        ckpt = snaps[-1][1]

    console.print(f"[bold]Cargando snapshot:[/bold] {ckpt}")
    snap = cargar_policy_desde_checkpoint(ckpt)

    console.print(f"[bold]Jugando {args.partidas} partidas completas por escenario...[/bold]")
    bot_metrics = evaluar_vs_bots(snap, obs_dim=args.obs_dim,
                                  n_partidas=args.partidas, con_pase=args.con_pase)
    simetria = _simetria_self_play(snap, max(args.partidas // 2, 20),
                                   args.obs_dim, con_pase=args.con_pase)

    # --- Tabla de resultados vs bots ---
    t = Table(title="Partidas completas vs bots fijos", expand=True)
    t.add_column("Escenario"); t.add_column("win-rate", justify="right")
    t.add_column("top-2", justify="right"); t.add_column("puesto medio", justify="right")
    escenarios = [k[len("win_rate_vs_"):] for k in bot_metrics
                  if k.startswith("win_rate_vs_")]
    for esc in escenarios:
        t.add_row(esc,
                  f"{bot_metrics[f'win_rate_vs_{esc}']:.3f}",
                  f"{bot_metrics[f'top2_rate_vs_{esc}']:.3f}",
                  f"{bot_metrics[f'puesto_medio_vs_{esc}']:.3f}")
    t.add_row("[bold]GLOBAL[/bold]",
              f"{bot_metrics['win_rate']:.3f}",
              f"{bot_metrics['top2_rate']:.3f}",
              f"{bot_metrics['puesto_medio']:.3f}")
    t.add_row("[dim]self-play (3 copias)[/dim]",
              f"{simetria['win_rate']:.3f}", f"{simetria['top2_rate']:.3f}", "—")
    t.caption = (f"moon_rate={bot_metrics['moon_rate']:.4f}  "
                 f"manos/partida={bot_metrics['manos_por_partida']:.1f}  "
                 f"(referencia: win-rate de azar en 4 jugadores = 0.25)")
    console.print(t)

    # --- Diagnóstico y recomendaciones ---
    recs = _diagnostico(args.dir, bot_metrics, simetria)
    console.print(Panel("\n".join(recs),
                        title="Diagnóstico y recomendaciones",
                        border_style="cyan"))


if __name__ == "__main__":
    main()
