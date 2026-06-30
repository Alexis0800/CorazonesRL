"""
Diagnóstico de SOLO la mano: captura en vivo por ADB y analiza la detección
de las 13 cartas. Sin modelo, sin pase, sin tocar nada.

Guarda una imagen de debug con cada carta y su crop. Ideal para iterar rápido
sobre el reconocimiento de mano sin el pipeline completo.

Uso:
  python scripts/diagnostico_mano.py --serial R52W90431FP --debug debug_mano
  python scripts/diagnostico_mano.py --frame calibracion/hearts_app/captura_real.png
"""
from __future__ import annotations
import argparse
import cv2
import numpy as np

# --- bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# --- fin ---
try:
    _sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _dibujar_mano(img_bgr, reg, cartas, out_path: str) -> None:
    """Dibuja la mano entera con las cajas de cada carta y su crop."""
    from src.captura.modelos import carta_a_str

    H, W = img_bgr.shape[:2]
    mx, my = reg.mano[0], reg.mano[1]
    mw, mh = reg.mano[2], reg.mano[3]
    x0, y0 = int(mx * W), int(my * H)

    vis = img_bgr.copy()
    # rectángulo de la región mano
    cv2.rectangle(vis, (x0, y0), (int((mx + mw) * W), int((my + mh) * H)),
                  (255, 255, 0), 2)

    for c in cartas:
        if c.carta_id is None:
            continue
        cx, cy = int(c.centro[0]), int(c.centro[1])
        cv2.circle(vis, (cx, cy), 12, (0, 255, 0), 2)
        label = carta_a_str(c.carta_id)
        cv2.putText(vis, label, (cx + 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imwrite(out_path, vis)
    print(f"   💾 overlay guardado: {out_path}")


def _dibujar_bloques(img_bgr, reg, out_path: str) -> None:
    """Dibuja las regiones de los bloques (filas) de cartas."""
    from src.captura.vision_cartas import _filas_de_cartas

    H, W = img_bgr.shape[:2]
    mx, my = reg.mano[0], reg.mano[1]
    mw, mh = reg.mano[2], reg.mano[3]
    x0, y0 = int(mx * W), int(my * H)
    roi = img_bgr[int(my * H):int((my + mh) * H),
                  int(mx * W):int((mx + mw) * W)]

    vis = roi.copy()
    filas = sorted(_filas_de_cartas(roi, 0.01),
                   key=lambda b: (b[1] // 50, b[0]))
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    for i, (fx, fy, fw, fh) in enumerate(filas):
        color = colors[i % len(colors)]
        cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), color, 2)
        cv2.putText(vis, f"bloque {i}", (fx + 5, fy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(out_path, vis)
    print(f"   💾 bloques guardados: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Diagnóstico de SOLO la mano — ADB o frame fijo.")
    p.add_argument("--serial", default=None,
                   help="Serial ADB del dispositivo.")
    p.add_argument("--adb", default="adb")
    p.add_argument("--frame", default=None, help="Frame fijo (sin ADB).")
    p.add_argument(
        "--regiones", default="calibracion/hearts_app/regiones.json")
    p.add_argument("--completas", default="calibracion/hearts_app/cartas_completas",
                   help="Naipes completos del sprite (para la mano).")
    p.add_argument("--debug", default=None,
                   help="Directorio donde guardar imágenes de debug (opcional).")
    args = p.parse_args()

    from src.captura.vision_hearts import Regiones, leer_mano_posiciones
    from src.captura.vision_cartas import ReconocedorPlantilla
    from src.captura.modelos import carta_a_str

    reg = Regiones.cargar(args.regiones)
    rec = ReconocedorPlantilla(args.completas)

    # ── captura de imagen ──
    if args.frame:
        img = cv2.imread(args.frame, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"No se pudo leer: {args.frame}")
        print(f"Frame fijo: {args.frame}  ({img.shape[1]}x{img.shape[0]})")
    elif args.serial:
        from src.captura.adb import ClienteADB
        cli = ClienteADB(adb=args.adb, serial=args.serial)
        img = cli.captura()
        print(f"Captura ADB: {img.shape[1]}x{img.shape[0]}")
    else:
        raise SystemExit("Necesito --serial (ADB) o --frame (archivo).")

    debug_dir = _Path(args.debug) if args.debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "00_captura.png"), img)
        _dibujar_bloques(img, reg, str(debug_dir / "01_bloques.png"))

    # ── leer mano ──
    import time
    t0 = time.perf_counter()
    cartas = leer_mano_posiciones(img, reg, rec)
    dt = time.perf_counter() - t0

    print(f"\nMano leída en {dt:.2f}s:")
    mano_ids = [c.carta_id for c in cartas]
    n_ok = sum(1 for cid in mano_ids if cid is not None)
    print(f"  {n_ok}/{len(cartas)} reconocidas")

    # Tabla detallada
    print(f"\n{'#':>2s}  {'carta':>5s}  {'método':>8s}  {'centro':>12s}")
    print("-" * 36)
    for i, c in enumerate(cartas):
        cid_str = carta_a_str(c.carta_id) if c.carta_id is not None else "??"
        centro_str = f"({c.centro[0]:.0f},{c.centro[1]:.0f})"
        print(f"{i:2d}  {cid_str:>5s}  {c.metodo or '--':>8s}  {centro_str:>12s}")

    # Línea compacta (como diagnostico_captura.py)
    print("\n  " + " ".join(carta_a_str(c) if c is not None else "??"
                            for c in mano_ids))

    # ── debug overlay ──
    if debug_dir:
        _dibujar_mano(img, reg, cartas, str(debug_dir / "02_overlay.png"))

    # ── crops individuales (ventana ESTRECHA: 1 sola carta) ──
    if debug_dir:
        crops_dir = debug_dir / "crops"
        crops_dir.mkdir(exist_ok=True)
        H, W = img.shape[:2]
        for i, c in enumerate(cartas):
            cid_str = carta_a_str(
                c.carta_id) if c.carta_id is not None else "??"
            cx, cy = int(c.centro[0]), int(c.centro[1])
            # Ancho: porcion visible + 30% margen (sin invadir la carta de al lado)
            vis = max(c.visible_w, 10)
            half_w = int(vis * 0.65)
            # Alto: la carta entera + 10% margen vertical
            half_h = int(
                c.card_h * 0.60) if c.card_h else int(vis / 0.774 * 0.60)
            x1 = max(0, cx - half_w)
            x2 = min(W, cx + half_w)
            y1 = max(0, cy - half_h)
            y2 = min(H, cy + half_h)
            if x2 > x1 and y2 > y1:
                crop = img[y1:y2, x1:x2]
                fn = crops_dir / f"{i:02d}_{cid_str}.png"
                cv2.imwrite(str(fn), crop)


if __name__ == "__main__":
    main()
