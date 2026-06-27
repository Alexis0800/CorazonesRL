"""
Monitor de avance del entrenamiento (v10 — partida completa).

Lee los logs de un directorio de modelo y muestra la progresión de las métricas
clave: win-rate, top-2, puesto medio, reward, entropía, y la evaluación absoluta
contra bots. Detecta de un vistazo si el entrenamiento mejora o está estancado.

Uso:
    python monitorear.py --dir models/v10
    python monitorear.py --dir models/v10 --watch          # refresco en vivo
    python monitorear.py --dir models/v10 --watch --intervalo 30

Métricas (de eval_log.jsonl, durante el entrenamiento):
    win_rate / top2_rate / puesto_medio  — calidad relativa al pool de self-play
    reward_medio / r_terminal_medio       — chequeo anti-farming (deben ir juntos)
    entropy                               — salud de exploración (alerta si <0.4)

Métricas (de bot_eval_log.jsonl, evaluación absoluta vs bots fijos):
    win_rate / top2_rate / puesto_medio vs experto — señal absoluta de progreso
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

# Forzar UTF-8 en stdout (sparklines/flechas) para no romper en consolas Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Group

console = Console()

_SPARK = "▁▂▃▄▅▆▇█"


def _leer_jsonl(path: str, tipo: Optional[str] = None) -> List[dict]:
    if not os.path.isfile(path):
        return []
    filas = []
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
                filas.append(d)
    return filas


def _sparkline(valores: List[float], ancho: int = 40) -> str:
    vals = [v for v in valores if isinstance(v, (int, float)) and v == v]
    if not vals:
        return "(sin datos)"
    vals = vals[-ancho:]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return _SPARK[len(_SPARK) // 2] * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / (hi - lo) * (len(_SPARK) - 1))
        out.append(_SPARK[idx])
    return "".join(out)


def _fmt(v, dec: int = 3) -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "—"
    return f"{v:.{dec}f}"


def _tendencia(valores: List[float], n: int = 10) -> str:
    vals = [v for v in valores if isinstance(v, (int, float)) and v == v]
    if len(vals) < 2 * n:
        return "—"
    ini = sum(vals[-2 * n:-n]) / n
    fin = sum(vals[-n:]) / n
    delta = fin - ini
    flecha = "↑" if delta > 1e-4 else ("↓" if delta < -1e-4 else "→")
    return f"{flecha} {delta:+.3f}"


def _construir_vista(model_dir: str) -> Group:
    log_dir = os.path.join(model_dir, "logs")
    metricas = _leer_jsonl(os.path.join(log_dir, "eval_log.jsonl"), "metricas")
    alertas = _leer_jsonl(os.path.join(log_dir, "eval_log.jsonl"), "alerta")
    bot_eval = _leer_jsonl(os.path.join(log_dir, "bot_eval_log.jsonl"))

    paso = metricas[-1].get("paso", 0) if metricas else 0
    cab = f"[bold]Modelo:[/bold] {model_dir}   [bold]Paso:[/bold] {paso:,}   " \
          f"[bold]Iteraciones logueadas:[/bold] {len(metricas)}"

    # --- Tabla de métricas de entrenamiento (relativas al pool) ---
    t = Table(title="Entrenamiento (relativo al pool de self-play)", expand=True)
    t.add_column("Métrica"); t.add_column("Último", justify="right")
    t.add_column("Tendencia", justify="right"); t.add_column("Sparkline (reciente)")

    def serie(clave):
        return [m.get(clave) for m in metricas]

    for clave, label, dec in [
        ("win_rate", "win-rate", 3),
        ("top2_rate", "top-2 rate", 3),
        ("puesto_medio", "puesto medio (1=mejor)", 3),
        ("reward_medio", "reward medio", 3),
        ("r_terminal_medio", "R_terminal medio", 3),
        ("entropy", "entropía", 3),
        ("manos_por_partida", "manos/partida", 1),
    ]:
        s = serie(clave)
        t.add_row(label, _fmt(s[-1] if s else None, dec),
                  _tendencia(s), _sparkline(s))

    # --- Tabla de evaluación absoluta vs bots ---
    tb = Table(title="Evaluación absoluta (vs bots fijos)", expand=True)
    tb.add_column("Escenario"); tb.add_column("win-rate", justify="right")
    tb.add_column("top-2", justify="right"); tb.add_column("puesto medio", justify="right")
    if bot_eval:
        ult = bot_eval[-1]
        for esc in ["evasivo", "conservador", "agresivo", "experto"]:
            tb.add_row(
                esc,
                _fmt(ult.get(f"win_rate_vs_{esc}")),
                _fmt(ult.get(f"top2_rate_vs_{esc}")),
                _fmt(ult.get(f"puesto_medio_vs_{esc}")),
            )
        tb.add_row("[bold]GLOBAL[/bold]", _fmt(ult.get("win_rate")),
                   _fmt(ult.get("top2_rate")), _fmt(ult.get("puesto_medio")))
        tb.caption = f"moon_rate={_fmt(ult.get('moon_rate'),4)}  " \
                     f"manos/partida={_fmt(ult.get('manos_por_partida'),1)}  " \
                     f"(paso {ult.get('paso',0):,})"
    else:
        tb.add_row("(sin bot_eval todavía)", "—", "—", "—")

    paneles = [Panel(cab, border_style="blue"), t, tb]

    if alertas:
        ultimas = alertas[-5:]
        txt = "\n".join(f"[yellow]⚠[/yellow] paso {a.get('paso',0):,}: {a.get('mensaje','')}"
                        for a in ultimas)
        paneles.append(Panel(txt, title="Alertas recientes", border_style="yellow"))

    return Group(*paneles)


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor de entrenamiento Hearts v10")
    p.add_argument("--dir", required=True, help="Directorio del modelo (p.ej. models/v10)")
    p.add_argument("--watch", action="store_true", help="Refresco en vivo")
    p.add_argument("--intervalo", type=int, default=20, help="Segundos entre refrescos")
    args = p.parse_args()

    if not args.watch:
        console.print(_construir_vista(args.dir))
        return

    with Live(_construir_vista(args.dir), console=console, refresh_per_second=1) as live:
        while True:
            time.sleep(args.intervalo)
            live.update(_construir_vista(args.dir))


if __name__ == "__main__":
    main()
