"""One-shot: genera cartas_mitad/ recortando la mitad izquierda de cada PNG
de cartas_completas/. Las plantillas resultantes capturan rango + pips de
cuerpo (suficiente para distinguir palo) — para cartas APILADAS en la mano.
"""
import cv2
from pathlib import Path

SRC = Path("calibracion/hearts_app/cartas_completas")
DST = Path("calibracion/hearts_app/cartas_mitad")
WF, HF = 0.45, 0.95  # fracción de ancho/alto a conservar

DST.mkdir(parents=True, exist_ok=True)
n = 0
for f in sorted(SRC.glob("*.png")):
    img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    H, W = img.shape[:2]
    mitad = img[0: int(HF * H), 0: int(WF * W)]
    cv2.imwrite(str(DST / f.name), mitad)
    n += 1

print(f"{n} plantillas mitad generadas en {DST}")
