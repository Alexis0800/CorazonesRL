"""
Genera un overlay visual que mapea las secciones del layout XML (room.xml)
sobre un screenshot real, con zonas coloreadas y etiquetas.

Uso:
  python scripts/overlay_xml_ui.py
  python scripts/overlay_xml_ui.py --frame ruta/a/frame.png
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# ─── Constantes ───
REF_W, REF_H = 1080, 1728
REGIONES_JSON = Path("calibracion/hearts_app/regiones.json")
FRAME_DEFAULT = Path("calibracion/captura.png")
OUTPUT = Path("calibracion/hearts_app/_overlay_XML_UI.png")

# Colores BGR para cada zona
COLOR_BANNER = (0, 255, 255)          # amarillo
COLOR_MANO = (255, 0, 0)              # azul
COLOR_MESA = (0, 255, 0)              # verde
COLOR_MARCADOR = (255, 128, 0)        # naranja
COLOR_CONFIRMAR = (255, 0, 255)       # magenta
COLOR_JUGADORES = (128, 255, 255)     # amarillo claro
COLOR_CHAT = (255, 128, 128)          # azul claro
COLOR_BOTONES = (128, 128, 255)       # rosa
COLOR_SCROLL = (0, 128, 128)          # teal
COLOR_INCLUDE = (128, 0, 128)         # púrpura oscuro


def fraccion_a_pixeles(r: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convierte [x, y, w, h] en fracciones a píxeles [x1, y1, x2, y2]."""
    x = int(r[0] * img_w)
    y = int(r[1] * img_h)
    w = int(r[2] * img_w)
    h = int(r[3] * img_h)
    return (x, y, x + w, y + h)


def draw_labeled_rect(
    img: np.ndarray,
    xy1: tuple[int, int],
    xy2: tuple[int, int],
    color: tuple[int, int, int],
    label: str,
    alpha: float = 0.3,
):
    """Dibuja un rectángulo con fill semitransparente y etiqueta centrada."""
    overlay = img.copy()
    cv2.rectangle(overlay, xy1, xy2, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Borde
    cv2.rectangle(img, xy1, xy2, color, 2)

    # Etiqueta centrada
    cx = (xy1[0] + xy2[0]) // 2
    cy = (xy1[1] + xy2[1]) // 2
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # Si la caja es muy pequeña, poner label arriba
    box_w = xy2[0] - xy1[0]
    box_h = xy2[1] - xy1[1]
    if box_w < tw + 10 or box_h < th + 10:
        tx = xy1[0]
        ty = xy1[1] - 5
        align = "left"
    else:
        tx = cx - tw // 2
        ty = cy + th // 2
        align = "center"

    # Fondo del texto
    cv2.rectangle(
        img,
        (tx - 3, ty - th - 3),
        (tx + tw + 3, ty + 3),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
    )


def draw_legend(img: np.ndarray, items: list[tuple[str, tuple[int, int, int]]]):
    """Draw legend in top-left corner."""
    x0, y0 = 15, 15
    for i, (label, color) in enumerate(items):
        y = y0 + i * 28
        cv2.rectangle(img, (x0, y), (x0 + 20, y + 18), color, -1)
        cv2.rectangle(img, (x0, y), (x0 + 20, y + 18), (255, 255, 255), 1)
        cv2.putText(
            img, label, (x0 + 28, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )


def main():
    frame_path = FRAME_DEFAULT
    if not frame_path.exists():
        print(
            f"[!] No se encuentra {frame_path}. Creando canvas sintético 1080×1728...")
        img = np.zeros((REF_H, REF_W, 3), dtype=np.uint8)
        img[:, :] = (30, 30, 30)  # gris oscuro
    else:
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"[!] No se pudo leer {frame_path}")
            return

    h, w = img.shape[:2]
    print(f"Imagen: {w}×{h} (ref: {REF_W}×{REF_H})")

    # Cargar regiones
    with open(REGIONES_JSON) as f:
        reg = json.load(f)

    # ─── 1. BANNER ───
    x1, y1, x2, y2 = fraccion_a_pixeles(reg["banner"], w, h)
    draw_labeled_rect(img, (x1, y1), (x2, y2), COLOR_BANNER,
                      "BANNER\n(fase / dirección / turno)")

    # ─── 2. MANO (HorizontalScrollView) ───
    x1, y1, x2, y2 = fraccion_a_pixeles(reg["mano"], w, h)
    draw_labeled_rect(img, (x1, y1), (x2, y2), COLOR_MANO,
                      "MANO DEL AGENTE\nHorizontalScrollView + LinearLayout\n13 cartas")

    # ─── 3. MESA (4 cartas en cruz) ───
    for idx, (key, rr) in enumerate(reg["mesa"].items()):
        x1, y1, x2, y2 = fraccion_a_pixeles(rr, w, h)
        label = f"MESA [{idx}]\n{key}"
        draw_labeled_rect(img, (x1, y1), (x2, y2), COLOR_MESA, label)

    # ─── 4. MARCADOR (puntuaciones) ───
    for key, rr in reg["marcador"].items():
        x1, y1, x2, y2 = fraccion_a_pixeles(rr, w, h)
        draw_labeled_rect(img, (x1, y1), (x2, y2), COLOR_MARCADOR,
                          f"MARCADOR\n{key} (EmojiTextView)")

    # ─── 5. CONFIRMAR (región actual, puede estar mal) ───
    x1, y1, x2, y2 = fraccion_a_pixeles(reg["confirmar"], w, h)
    draw_labeled_rect(img, (x1, y1), (x2, y2), COLOR_CONFIRMAR,
                      "CONFIRMAR PASE (?)", alpha=0.2)

    # ─── 6. CHAT + ENTRADA ───
    # Inferido: debajo de la mesa, encima de la mano
    # ~ y=0.64 a 0.73, centrado horizontal
    chat_x1 = int(0.10 * w)
    chat_x2 = int(0.90 * w)
    chat_y1 = int(0.60 * h)
    chat_y2 = int(0.72 * h)
    draw_labeled_rect(img, (chat_x1, chat_y1), (chat_x2, chat_y2), COLOR_CHAT,
                      "CHAT + EditText\n(Zona de mensajes)\n[inferido del XML]", alpha=0.2)

    # ─── 7. BOTONES DE ACCIÓN ───
    # Inferido: dos filas de botones entre chat y mano
    btn_y1 = int(0.72 * h)
    btn_y2 = int(0.745 * h)  # justo arriba de la mano
    draw_labeled_rect(img, (int(0.05 * w), btn_y1), (int(0.95 * w), btn_y2), COLOR_BOTONES,
                      "BOTONES ACCIÓN (5-9 Views+Buttons)\n[inferido del XML]", alpha=0.25)

    # ─── 8. MANOS RIVALES (HorizontalScrollViews adicionales) ───
    # Arriba: mano del rival de enfrente
    draw_labeled_rect(img, (int(0.10 * w), int(0.28 * h)),
                      (int(0.90 * w), int(0.34 * h)),
                      COLOR_SCROLL, "MANO RIVAL (arriba)\nHorizontalScrollView\n[inferido XML]", alpha=0.2)
    # Izquierda: probablemente vertical
    draw_labeled_rect(img, (int(0.01 * w), int(0.40 * h)),
                      (int(0.10 * w), int(0.60 * h)),
                      COLOR_SCROLL, "RIVAL\nizq.", alpha=0.2)
    # Derecha
    draw_labeled_rect(img, (int(0.90 * w), int(0.40 * h)),
                      (int(0.99 * w), int(0.60 * h)),
                      COLOR_SCROLL, "RIVAL\nder.", alpha=0.2)

    # ─── 9. <include> inferior ───
    # El include al final del FrameLayout raíz — probablemente un banner/toast
    x1, y1, x2, y2 = fraccion_a_pixeles(reg["banner"], w, h)
    draw_labeled_rect(
        img, (x1, y1 + 20), (x2, y2 + 40), COLOR_INCLUDE,
        "<include> (layout extra)\n[fin del FrameLayout raíz]", alpha=0.15,
    )

    # ─── LEYENDA ───
    legend_items = [
        ("Banner", COLOR_BANNER),
        ("Mano agente", COLOR_MANO),
        ("Mesa (4 cartas)", COLOR_MESA),
        ("Marcador/Players", COLOR_MARCADOR),
        ("Confirmar (actual)", COLOR_CONFIRMAR),
        ("Chat (inferido)", COLOR_CHAT),
        ("Botones (inferido)", COLOR_BOTONES),
        ("Manos rivales (inferido)", COLOR_SCROLL),
        ("<include>", COLOR_INCLUDE),
    ]
    draw_legend(img, legend_items)

    # ─── GUARDAR ───
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), img)
    print(f"[✓] Overlay guardado en: {OUTPUT}")

    # También mostrar las regiones calibradas vs inferidas en texto
    print("\n--- REGIONES CALIBRADAS (regiones.json) ---")
    for k in ["banner", "mano", "confirmar", "mesa", "marcador"]:
        v = reg[k]
        if isinstance(v, list):
            print(f"  {k}: {v}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")

    print("\n--- REGIONES INFERIDAS DEL XML ---")
    print(f"  chat:            [0.10, 0.60, 0.80, 0.12]")
    print(f"  botones_accion:  [0.05, 0.72, 0.90, 0.025]")
    print(f"  mano_rival_arr:  [0.10, 0.28, 0.80, 0.06]")
    print(f"  mano_rival_izq:  [0.01, 0.40, 0.09, 0.20]")
    print(f"  mano_rival_der:  [0.90, 0.40, 0.09, 0.20]")


if __name__ == "__main__":
    main()
