# Calibración de la app Hearts (es) para captura por visión

Assets para leer la pantalla de la app de Corazones (1080×1728) sin OCR de
sistema (solo OpenCV). Todo lo aprende del video; las regiones están en
fracciones, así que toleran otra resolución.

## Contenido

- `regiones.json` — cajas (fracciones `[x,y,w,h]`) de: `banner`, 4 `marcador`,
  4 posiciones de `mesa` en cruz, y `mano`. Verificar/ajustar con
  `scripts/calibrar_regiones.py`.
- `banners/` — plantillas de banner etiquetadas `<tag>__<i>.png`
  (`pase_izquierda`, `turno_arriba`, `baza_izquierda`, `vacio`, …). El tag mapea
  a fase/posición en `src/captura/vision_hearts._BANNER_SEMANTICA`.
- `cartas/` — plantillas de **esquina** etiquetadas `<carta>__<i>.png` (formato
  `carta_a_str`, p.ej. `QP`, `10C`). Las usa el reconocedor de **mesa** (híbrido).
- `cartas_completas/` — los **52 naipes enteros** del sprite de la APK
  (`<carta>.png`, p.ej. `AP.png`). Las usa `ReconocedorPlantilla` (matchTemplate)
  para leer la **mano**. Se generan con `scripts/cartas_desde_sprite.py` desde el
  sprite (`Corazones/recursos/assets/cards_0.png`, gitignored). **Versionadas.**
- `banners_descubiertos/`, `cartas_descubiertas/` — salida cruda del agrupado
  (regenerable; no es necesaria para producción).

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

## Flujo de trabajo

### 1. Validar offline sobre el video (sin móvil)
```bash
python scripts/capturar_visual.py --fuente carpeta \
    --carpeta "videos/fotogramas" --solo-validas \
    --salida datos/capturas/video.jsonl
```
`--solo-validas` descarta las manos que el motor no puede reconstruir → el JSONL
nunca queda corrupto.

### 2. Capturar en vivo por ADB (tú juegas; el ADB solo observa)
```bash
python scripts/capturar_visual.py --fuente adb --serial <SERIAL> \
    --salida datos/capturas/sesion.jsonl --solo-validas
```
Requiere `adb` en el PATH y depuración USB. Poll por defecto 0.4 s (varios polls
por turno → la confirmación temporal funciona).

### 3. Extender / corregir plantillas (recomendado en el dispositivo real)
```bash
# banners nuevos (p.ej. "Pasar 3 cartas a la derecha", mano sin pase):
python scripts/agrupar_banners.py --frames <carpeta_frames>
#   -> revisar banners_descubiertos/, copiar el bueno a banners/<tag>__<i>.png

# cartas nuevas / dudosas:
python scripts/agrupar_cartas.py --frames <carpeta_frames>
#   -> etiquetar y copiar a cartas/<carta>__<i>.png
```

## Probar rápido

```bash
# un frame -> banner + mesa + mano (13 cartas):
python scripts/diagnostico_captura.py --frame calibracion/hearts_app/captura_real.png --overlay
# en vivo por ADB (solo observa):
python scripts/monitor_vivo.py --serial <SERIAL> --poll 0.5
```

## Estado y pendientes

- **Mano inicial 13/13 perfecto** en la captura real del teléfono (1600×2560):
  `matchTemplate` multiescala contra el sprite. El **banner generaliza** entre
  dispositivos sin recalibrar.
- Sólido (en el video 1080×1728): clasificación de banner (374/391; el resto son
  popups), lectura de carta de **mesa** (rango+color+palo), máquina de estados con
  orden de baza correcto (líder del 2♣ en la baza 1), validación por replay.
- Pendiente: aplicar el mismo `ReconocedorPlantilla` (matchTemplate del sprite) a
  la **mesa** y validar con frames en-juego del teléfono; leer el **pase recibido**;
  popups de centro (concesión "se llevará el resto" → `manos_restantes`, "Pulsa
  para continuar", tabla de fin de mano); **auto-juego** (tap por ADB con el modelo).
