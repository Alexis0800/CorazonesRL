"""
Calibrador interactivo de TODAS las regiones del juego Hearts (es).

Muestra un frame con todas las regiones superpuestas en colores. Permite
seleccionar cuál editar con teclas numéricas o click, y arrastrar/resize
con el mouse. Guarda todo en regiones.json.

Uso:
  python scripts/calibrar_todo.py
  python scripts/calibrar_todo.py --frame ruta/a/frame.png --max-height 800

Controles:
  1-9,c,p = seleccionar región (ver lista abajo)
  Tab   = siguiente región
  Click dentro de región = seleccionarla
  Arrastrar esquinas = resize  |  Arrastrar centro = mover
  s = guardar   q = salir   h = ayuda

Regiones:
  [1] Banner
  [2] Mano agente
  [3] Mesa arriba    [4] Mesa izquierda   [5] Mesa derecha   [6] Mesa abajo
  [7] Marcador arr.  [8] Marcador izq.    [9] Marcador der.  [0] Marcador abajo
  [p] Zona pases     [c] Confirmar pase
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

REGIONES_JSON = Path("calibracion/hearts_app/regiones.json")

REGIONES_META: List[dict] = [
    {"id": "1", "key": "banner",             "label": "Banner",
        "color": (0, 255, 255)},
    {"id": "2", "key": "mano",
        "label": "Mano agente",          "color": (255, 0, 0)},
    {"id": "3", "key": "mesa.arriba",
        "label": "Mesa arriba",          "color": (0, 255, 0)},
    {"id": "4", "key": "mesa.izquierda",
        "label": "Mesa izquierda",       "color": (0, 255, 0)},
    {"id": "5", "key": "mesa.derecha",
        "label": "Mesa derecha",         "color": (0, 255, 0)},
    {"id": "6", "key": "mesa.abajo",
        "label": "Mesa abajo (agente)",  "color": (0, 255, 0)},
    {"id": "7", "key": "marcador.arriba",
        "label": "Marcador arriba",      "color": (255, 128, 0)},
    {"id": "8", "key": "marcador.izquierda",
        "label": "Marcador izquierda",   "color": (255, 128, 0)},
    {"id": "9", "key": "marcador.derecha",
        "label": "Marcador derecha",     "color": (255, 128, 0)},
    {"id": "0", "key": "marcador.abajo",
        "label": "Marcador abajo",       "color": (255, 128, 0)},
    {"id": "p", "key": "pases",
        "label": "Zona pases",           "color": (255, 255, 0)},
    {"id": "c", "key": "confirmar",
        "label": "Confirmar pase",       "color": (255, 0, 255)},
]

# ─── JSON helpers ───


def cargar() -> dict:
    return json.loads(REGIONES_JSON.read_text(encoding="utf-8"))


def guardar(reg: dict):
    txt = json.dumps(reg, indent=2, ensure_ascii=False) + "\n"
    REGIONES_JSON.write_text(txt, encoding="utf-8")


def _get_frac(reg: dict, key: str) -> Optional[List[float]]:
    if "." in key:
        parent, child = key.split(".", 1)
        d = reg.get(parent)
        return d.get(child) if isinstance(d, dict) else None
    return reg.get(key)


def _set_frac(reg: dict, key: str, val: List[float]):
    if "." in key:
        parent, child = key.split(".", 1)
        reg.setdefault(parent, {})[child] = val
    else:
        reg[key] = val


# ─── Coord helpers ───

def frac_a_px(r: List[float], W: int, H: int) -> Tuple[int, int, int, int]:
    return (int(r[0] * W), int(r[1] * H),
            int(r[0] * W) + int(r[2] * W), int(r[1] * H) + int(r[3] * H))


def px_a_frac(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> List[float]:
    return [round(x1 / W, 4), round(y1 / H, 4),
            round((x2 - x1) / W, 4), round((y2 - y1) / H, 4)]


# ─── Estado ───

DRAG_NONE, DRAG_CENTER, DRAG_TL, DRAG_BR = 0, 1, 2, 3

st: dict = {}


def select(idx: int):
    if 0 <= idx < len(REGIONES_META):
        st["sel"] = idx
        st["drag"] = DRAG_NONE


def redraw():
    img = st["base"].copy()
    W, H = st["img_w"], st["img_h"]
    s = st["scale"]
    sel = st["sel"]

    # Dibujar todas las regiones
    for i, meta in enumerate(REGIONES_META):
        f = st["fracs"][i]
        if f is None:
            continue
        x1, y1, x2, y2 = frac_a_px(f, W, H)
        c = meta["color"]
        active = (i == sel)

        ov = img.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), c, -1)
        cv2.addWeighted(ov, 0.30 if active else 0.10, img,
                        1 - (0.30 if active else 0.10), 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2 if active else 1)

        if active:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(img, (cx - 12, cy), (cx + 12, cy), (255, 255, 255), 1)
            cv2.line(img, (cx, cy - 12), (cx, cy + 12), (255, 255, 255), 1)

    # Panel inferior
    ov = img.copy()
    cv2.rectangle(ov, (0, H - 75), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, img, 0.35, 0, img)

    meta = REGIONES_META[sel]
    f = st["fracs"][sel]
    lines = [
        f"[{meta['id']}] {meta['label']}  →  {f}",
        "Tab=siguiente  Arrastra=editar  s=guardar  q=salir  h=ayuda",
    ]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, H - 60 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Ayuda
    if st.get("help"):
        hlp = [
            "1-9,p,c = seleccionar  |  Tab = siguiente  |  Click = seleccionar",
            "Arrastrar esquina = resize  |  Arrastrar centro = mover",
            "s = guardar    q = salir    h = cerrar ayuda",
        ]
        bw = 520
        bh = len(hlp) * 22 + 18
        lx = W // 2 - bw // 2
        cv2.rectangle(img, (lx, 15), (lx + bw, 15 + bh), (35, 35, 35), -1)
        cv2.rectangle(img, (lx, 15), (lx + bw, 15 + bh), (180, 180, 180), 1)
        for i, line in enumerate(hlp):
            cv2.putText(img, line, (lx + 15, 43 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    display = cv2.resize(img, (int(W * s), int(H * s)),
                         interpolation=cv2.INTER_AREA)
    cv2.imshow("Calibrar Regiones", display)


# ─── Mouse ───

def mouse_cb(event, x, y, flags, param):
    s = st["scale"]
    ox, oy = int(x / s), int(y / s)
    sel = st["sel"]
    W, H = st["img_w"], st["img_h"]

    if event == cv2.EVENT_LBUTTONDOWN:
        # ¿Click en alguna región? → seleccionar
        for i in range(len(REGIONES_META)):
            f = st["fracs"][i]
            if f is None:
                continue
            x1, y1, x2, y2 = frac_a_px(f, W, H)
            if x1 <= ox <= x2 and y1 <= oy <= y2:
                if i != sel:
                    select(i)
                    redraw()
                # Iniciar drag en la región (sea nueva o ya seleccionada)
                margin = int(25 / s)
                if abs(ox - x1) < margin and abs(oy - y1) < margin:
                    st["drag"] = DRAG_TL
                elif abs(ox - x2) < margin and abs(oy - y2) < margin:
                    st["drag"] = DRAG_BR
                elif x1 < ox < x2 and y1 < oy < y2:
                    st["drag"] = DRAG_CENTER
                    st["dox"] = ox - (x1 + x2) // 2
                    st["doy"] = oy - (y1 + y2) // 2
                return

        # Click fuera de toda región → intentar editar la seleccionada igual
        f = st["fracs"][sel]
        if f is not None:
            x1, y1, x2, y2 = frac_a_px(f, W, H)
            margin = int(25 / s)
            if abs(ox - x1) < margin and abs(oy - y1) < margin:
                st["drag"] = DRAG_TL
            elif abs(ox - x2) < margin and abs(oy - y2) < margin:
                st["drag"] = DRAG_BR
            elif x1 < ox < x2 and y1 < oy < y2:
                st["drag"] = DRAG_CENTER
                st["dox"] = ox - (x1 + x2) // 2
                st["doy"] = oy - (y1 + y2) // 2

    elif event == cv2.EVENT_MOUSEMOVE and st.get("drag") not in (None, DRAG_NONE):
        f = st["fracs"][sel]
        if f is None:
            return
        x1, y1, x2, y2 = frac_a_px(f, W, H)
        if st["drag"] == DRAG_TL:
            x1, y1 = max(0, ox), max(0, oy)
        elif st["drag"] == DRAG_BR:
            x2, y2 = min(W, ox), min(H, oy)
        elif st["drag"] == DRAG_CENTER:
            dw, dh = x2 - x1, y2 - y1
            nx, ny = ox - st["dox"], oy - st["doy"]
            x1 = max(0, min(W - dw, nx - dw // 2))
            y1 = max(0, min(H - dh, ny - dh // 2))
            x2, y2 = x1 + dw, y1 + dh
        st["fracs"][sel] = px_a_frac(x1, y1, x2, y2, W, H)
        redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        st["drag"] = DRAG_NONE


# ─── Main ───

def main():
    p = argparse.ArgumentParser(
        description="Calibrador visual de TODAS las regiones")
    p.add_argument(
        "--frame", default="calibracion/captura1.png")
    p.add_argument("--max-height", type=int, default=900)
    args = p.parse_args()

    fp = Path(args.frame)
    if not fp.exists():
        print(f"[!] No existe: {fp}")
        return

    img = cv2.imread(str(fp))
    if img is None:
        print(f"[!] No se pudo leer: {fp}")
        return

    H, W = img.shape[:2]
    scale = min(1.0, args.max_height / H)

    print(
        f"Frame: {W}x{H}  Display: {int(W*scale)}x{int(H*scale)} ({scale:.0%})")
    print()

    reg = cargar()
    fracs = []
    for meta in REGIONES_META:
        v = _get_frac(reg, meta["key"])
        fracs.append(v)
        print(f"  [{meta['id']}] {meta['label']:22s} {v}")

    st["base"] = img.copy()
    st["img_w"], st["img_h"] = W, H
    st["scale"] = scale
    st["fracs"] = fracs
    st["sel"] = 0
    st["drag"] = DRAG_NONE
    st["help"] = True

    cv2.namedWindow("Calibrar Regiones")
    cv2.setMouseCallback("Calibrar Regiones", mouse_cb)
    redraw()

    N = len(REGIONES_META)

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q") or key == 27:
            print("Salir sin guardar.")
            break

        if key == ord("s"):
            for i, meta in enumerate(REGIONES_META):
                f = st["fracs"][i]
                if f is not None:
                    _set_frac(reg, meta["key"], f)
            guardar(reg)
            print("✅ Guardado en regiones.json")
            break

        if key == ord("h"):
            st["help"] = not st.get("help")
            redraw()

        if key == 9:  # Tab
            select((st["sel"] + 1) % N)
            redraw()

        if ord("0") <= key <= ord("9"):
            idx = 9 if key == ord("0") else (key - ord("1"))
            if idx < N:
                select(idx)
                redraw()

        if key in (ord("p"), ord("P")):
            select(10)  # pases es idx 10
            redraw()

        if key in (ord("c"), ord("C")):
            select(11)  # confirmar es idx 11
            redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
