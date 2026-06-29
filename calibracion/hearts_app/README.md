# Calibración de la app Hearts (es) para captura por visión

Assets para leer la pantalla de la app de Corazones (1080×1728) sin OCR de
sistema (solo OpenCV). Todo lo aprende del video; las regiones están en
fracciones, así que toleran otra resolución.

---

## Origen de los datos

### Desde el APK (extracción estática)

| Recurso | Ubicación | Uso |
|---|---|---|
| **Layouts XML** (`room.xml`, `activity.xml`, …) | `../Corazones/recursos/layouts/` | Estructura jerárquica de la UI. Decodificados con `scripts/layout_tree.py` para entender posiciones relativas de banner, mesa, marcador, mano, botones. **No se usan en runtime.** |
| **Sprite de naipes** (`cards_0.png`) | `../Corazones/recursos/assets/` | Arte fuente de las 52 cartas. Se recortó con `scripts/cartas_desde_sprite.py` → `cartas_completas/`. Usado en runtime por `ReconocedorPlantilla` (matchTemplate). |
| **Íconos y drawables** | `../Corazones/recursos/drawable/` | Botones, indicadores. Usados en la fase exploratoria para inferir regiones; **no se usan en runtime.** |

### Desde capturas ADB (calibración manual)

| Recurso | Ubicación | Cómo se obtuvo |
|---|---|---|
| **`regiones.json`** | `./regiones.json` | Calibrado a mano con `scripts/calibrar_todo.py` sobre `../captura.png` (screenshot 1600×2560 del teléfono). 12 regiones: banner, mano, mesa×4, marcador×4, pases, confirmar. |
| **Plantillas de banner** | `banners/` | Recortes etiquetados a mano desde frames de video/ADB. Se descubren nuevos con `scripts/agrupar_banners.py` → `banners_descubiertos/`, luego se etiquetan y copian aquí. |
| **Plantillas de carta (esquina)** | `cartas/` | Recortes de la esquina de cada carta en la mesa, etiquetados a mano. Se usa en el reconocedor híbrido de mesa. Se descubren con `scripts/agrupar_cartas.py` → `cartas_descubiertas/`. |
| **Naipes completos** | `cartas_completas/` | Generados **automáticamente** desde el sprite del APK con `scripts/cartas_desde_sprite.py`. **Versionados** — no requieren recalibración. |
| **Screenshot de referencia** | `../captura.png` | Captura 1600×2560 del teléfono real con la app en mano inicial. Se usa como fondo del calibrador. |

---

## Contenido del directorio

```
calibracion/
├── captura.png                  ← Screenshot de referencia para el calibrador
├── hearts_app/
│   ├── regiones.json            ← 12 regiones calibradas (fracciones 0..1)
│   ├── banners/                 ← Plantillas de banner etiquetadas (runtime)
│   ├── banners_descubiertos/    ← Salida cruda de agrupar_banners.py (regenerable)
│   ├── cartas/                  ← Plantillas de esquina de carta (runtime, mesa)
│   ├── cartas_completas/        ← 52 naipes del sprite APK (runtime, mano)
│   ├── cartas_descubiertas/     ← Salida cruda de agrupar_cartas.py (regenerable)
│   └── README.md                ← Este archivo
└── Corazones/recursos/          ← Assets extraídos del APK (documentación)
    ├── layouts/                 ← XMLs de layout (room.xml, activity.xml, …)
    ├── drawable/                ← Íconos y botones
    ├── assets/                  ← Sprites, fuentes
    └── …
```

---

## Regiones calibradas (`regiones.json`)

12 regiones en fracciones `[x, y, w, h]` sobre referencia 1080×1728:

| # | Clave | Descripción | Uso |
|---|---|---|---|
| 1 | `banner` | Barra de texto superior | Clasificación de fase/turno |
| 2 | `mano` | Mano del agente (13 cartas abajo) | Lectura de cartas propias |
| 3–6 | `mesa.{arriba,izquierda,derecha,abajo}` | 4 posiciones de carta en cruz | Lectura de bazas |
| 7–0 | `marcador.{arriba,izquierda,derecha,abajo}` | Puntuaciones de cada jugador | No se usan (el motor recalcula) |
| p | `pases` | Zona donde aparecen las 3 cartas seleccionadas/recibidas | Auto-pase: leer cartas recibidas |
| c | `confirmar` | Botón círculo-check de confirmar pase | Auto-pase: tap para confirmar |

Para recalibrar: `python scripts/calibrar_todo.py`

---

## Cómo se reconoce una carta

**Mesa (carta suelta, entera)** — híbrido en `Reconocedor`:

1. **Color** del índice → rojo (♥♦) o negro (♠♣).
2. **Rango** por plantilla del glifo de esquina (cubre los 13).
3. **Palo** por la **forma del pip de cuerpo**: ♥/♦ por nº de lóbulos arriba;
   ♣/♠ por *solidez* (el trébol tiene huecos).

**Mano (13 cartas solapadas en abanico)** — `ReconocedorPlantilla` +
`vision_hearts.leer_mano`: como el arte de la app es **idéntico al sprite**, se
usa `matchTemplate` (correlación normalizada) contra los naipes completos. Claves
que lo llevan al **100%**: (a) la ventana de búsqueda absorbe el desajuste de
pocos píxeles que rompe la firma de esquina; (b) **multiescala** (`_ESCALAS`)
para la fila "levantada" del pase, que se renderiza a otro tamaño; (c)
**posiciones uniformes** (el paso del abanico es constante → se interpolan entre
la primera y la última carta, corrigiendo el ruido de localización en ♣); (d) el
**palo por bloque** (la app agrupa la mano por palo) desde su carta más a la
derecha (entera), y el **rango por carta**.

Si la confianza es baja → `None` (mejor no emitir una carta equivocada). La
validez final de una captura la da el motor en el replay.

---

## Flujo de trabajo

### 1. Calibrar regiones
```bash
python scripts/calibrar_todo.py
# Teclas 1-9,0,p,c = seleccionar región | arrastrar = editar | s = guardar
```
Esto guarda en `regiones.json`. El overlay de verificación:
```bash
python scripts/overlay_xml_ui.py
# → calibracion/hearts_app/_overlay_XML_UI.png
```

### 2. Validar offline sobre el video (sin móvil)
```bash
python scripts/capturar_visual.py --fuente carpeta \
    --carpeta "videos/fotogramas" --solo-validas \
    --salida datos/capturas/video.jsonl
```
`--solo-validas` descarta las manos que el motor no puede reconstruir → el JSONL
nunca queda corrupto.

### 3. Capturar en vivo por ADB (tú juegas; el ADB solo observa)
```bash
python scripts/capturar_visual.py --fuente adb --serial <SERIAL> \
    --salida datos/capturas/sesion.jsonl --solo-validas
```
Requiere `adb` en el PATH y depuración USB. Poll por defecto 0.4 s (varios polls
por turno → la confirmación temporal funciona).

### 4. Extender / corregir plantillas (recomendado en el dispositivo real)
```bash
# banners nuevos (p.ej. "Pasar 3 cartas a la derecha", mano sin pase):
python scripts/agrupar_banners.py --frames <carpeta_frames>
#   → revisar banners_descubiertos/, copiar el bueno a banners/<tag>__<i>.png

# cartas nuevas / dudosas:
python scripts/agrupar_cartas.py --frames <carpeta_frames>
#   → etiquetar y copiar a cartas/<carta>__<i>.png
```

---

## Probar rápido

```bash
# un frame → banner + mesa + mano (13 cartas):
python scripts/diagnostico_captura.py --frame calibracion/captura.png --overlay
# en vivo por ADB (solo observa):
python scripts/monitor_vivo.py --serial <SERIAL> --poll 0.5
```

---

## Estado y pendientes

- **Mano inicial 13/13 perfecto** en la captura real del teléfono (1600×2560):
  `matchTemplate` multiescala contra el sprite. El **banner generaliza** entre
  dispositivos sin recalibrar.
- Sólido (en el video 1080×1728): clasificación de banner (374/391; el resto son
  popups), lectura de carta de **mesa** (rango+color+palo), máquina de estados con
  orden de baza correcto (líder del 2♣ en la baza 1), validación por replay.
- **Auto-pase** (tap por ADB con el modelo) implementado: `scripts/auto_pase.py`
  + `src/captura/auto_pase.py`. Lee la mano con `leer_mano_posiciones` (id +
  punto de toque), pide las 3 cartas al modelo, las toca **releyendo la mano
  antes de cada toque** (las seleccionadas suben sobre el banner y los bloques se
  re-flujen ♣♦♠♥), confirma y deduce las recibidas por diferencia
  (`mano_nueva − (mano_vieja − pasadas)`). Probarlo en seco: `python
  scripts/auto_pase.py --modelo models/produccion/v10c_campeon --frame
  calibracion/captura.png --seco`.
- Pendiente: aplicar el mismo `ReconocedorPlantilla` (matchTemplate del sprite) a
  la **mesa** y validar con frames en-juego del teléfono; popups de centro
  (concesión "se llevará el resto" → `manos_restantes`, "Pulsa para continuar",
  tabla de fin de mano); **auto-juego** de las bazas (no solo el pase).
