# =============================================================================
#
#   PAVESCAN — EVALUACIÓN DE BACHES EN PAVIMENTO CON INTELIGENCIA ARTIFICIAL
#
#   Universidad Nacional San Cristóbal de Huamanga (UNSCH)
#   Facultad de Ingeniería de Minas · Escuela Profesional de Ingeniería Civil
#   Tesis de Grado · Huamanga, Ayacucho, Perú · 2026
#
#   Base normativa: Manual de Carreteras: Mantenimiento o Conservación Vial
#   MTC — RD N° 08-2014-MTC/14 / RD N° 05-2016-MTC/14
#
#   VERSIÓN ENFOCADA AL MANUAL MTC — PERÚ
#   ──────────────────────────────────────
#   Este archivo implementa el pipeline completo de detección y evaluación
#   de baches según la normativa peruana vigente:
#
#   · Cap. 4 — Tabla 4-8: Clasificación Baches (Huecos) — Falla 7
#       Gravedad 1: Diámetro < 0.20 m
#       Gravedad 2: Diámetro entre 0.20 y 0.50 m
#       Gravedad 3: Diámetro > 0.50 m
#
#   · Cap. 400 — Técnicas de conservación de pavimentos flexibles
#       Sec.410: Parchado Superficial (profundidad < 50 mm)
#       Sec.415: Parchado Profundo    (profundidad > 50 mm)
#       Sec.460/465: Recapeo / Reconstrucción
#
#   · Cap. 3 — Niveles de servicio:
#       Porcentaje máximo de área con baches = 0% en TODA vía
#       Todo bache detectado requiere intervención inmediata
#
#   DEPENDENCIAS (instalar con pip):
#       pip install numpy opencv-python torch transformers
#
#   CONFIGURACIÓN DEL MODELO:
#       Coloca la carpeta de pesos fine-tuneados en ./modelo_segformer
#       (debe contener config.json, *.safetensors y preprocessor_config.json)
#
#   EJECUCIÓN:
#       python pavescan_segformer_mtc.py
#
# =============================================================================

import numpy as np
import cv2
import math
import os

# torch y transformers se cargan de forma diferida en _intentar_cargar_segformer()
# para que el pipeline funcione sin ellos (MODO C: umbral adaptativo)

# ── Módulo de evaluación normativa MTC ───────────────────────────────────────
# pavescan_fuzzy_mtc.py implementa el sistema de lógica difusa Mamdani
# basado EXCLUSIVAMENTE en el Manual de Carreteras MTC:
#   · RD N° 08-2014-MTC/14 / RD N° 05-2016-MTC/14
#   · Cap. 3: Niveles de servicio (baches = 0% en toda vía)
#   · Cap. 4: Catálogo de deterioros (3 niveles de gravedad)
#   · Sec. 410: Parchado Superficial (profundidad < 50 mm)
#   · Sec. 415: Parchado Profundo    (profundidad > 50 mm)
#   · Sec. 460: Recapeo/Reconstrucción (daño extenso)
try:
    from pavescan_fuzzy_mtc import (
        evaluar_bache_hibrido      as _mtc_evaluar,
        guardar_resultado_json_mtc as _mtc_guardar_json,
    )
    _MTC_MODULO_OK = True
except ImportError:
    _MTC_MODULO_OK = False
    print("[AVISO] pavescan_fuzzy_mtc.py no encontrado. "
          "Colócalo en la misma carpeta.")


# =============================================================================
# PASO 1 — CONVERSIÓN A ESCALA DE GRISES
# Manual MTC: preprocesamiento de imagen para detección de fallas superficiales
# =============================================================================

def paso1_escala_de_grises(imagen):
    """
    ¿Por qué convertir a grises?
    ─────────────────────────────
    El color no aporta información relevante para detectar baches:
    lo que importa es la INTENSIDAD (qué tan oscuro es un píxel).
    Reducir de 3 canales (R,G,B) a 1 canal simplifica el cómputo
    sin perder información de detección.

    ¿Cómo se hace?
    ──────────────
    Se usa la fórmula de luminancia ponderada estándar ITU-R BT.601:

        Y = 0.299·R  +  0.587·G  +  0.114·B

    Los pesos NO son iguales porque el ojo humano percibe el verde con
    mayor intensidad.

    Parámetro de entrada:
        imagen → array de forma (alto, ancho, 3), valores 0–255

    Valor de retorno:
        array de forma (alto, ancho), valores 0–255
    """

    if imagen.ndim == 2:
        return imagen

    R = imagen[:, :, 0].astype(np.float32)
    G = imagen[:, :, 1].astype(np.float32)
    B = imagen[:, :, 2].astype(np.float32)

    # Fórmula de luminancia ponderada
    Y = 0.299 * R + 0.587 * G + 0.114 * B

    return np.clip(Y, 0, 255).astype(np.uint8)


# =============================================================================
# PASO 2 — REDIMENSIONAMIENTO A 512 × 512 PÍXELES
# Requerimiento del modelo SegFormer para procesamiento uniforme
# =============================================================================

def paso2_redimensionar(imagen, tamano=512):
    """
    ¿Por qué 512×512?
    ──────────────────
    El modelo SegFormer-B2 requiere imágenes de exactamente 512×512 píxeles.
    Sus capas internas asumen ese tamaño fijo.

    ¿Qué interpolación se usa?
    ──────────────────────────
    - INTER_CUBIC  → si la imagen es más pequeña que 512 (se amplía)
    - INTER_AREA   → si la imagen es más grande que 512 (se reduce)

    Parámetro de entrada:
        imagen → array de cualquier tamaño
        tamano → tamaño destino (512 por defecto)

    Valor de retorno:
        array de forma (512, 512)
    """

    alto, ancho = imagen.shape[:2]
    if alto == tamano and ancho == tamano:
        return imagen
    metodo = cv2.INTER_CUBIC if (alto < tamano or ancho < tamano) else cv2.INTER_AREA
    return cv2.resize(imagen, (tamano, tamano), interpolation=metodo)


# =============================================================================
# PASO 3 — CONVERSIÓN PÍXEL → TENSOR (normalización)
# Preprocesamiento requerido por el modelo SegFormer
# =============================================================================

def paso3_pixel_a_tensor(imagen):
    """
    ¿Qué es un tensor aquí?
    ────────────────────────
    Las redes neuronales trabajan con números decimales en rangos pequeños,
    no con enteros de 0 a 255.

    ¿Por qué normalizar a [0, 1]?
    ──────────────────────────────
    Dividir entre 255 pone todos los valores en [0.0, 1.0]:
    - Evita que píxeles muy brillantes dominen el aprendizaje
    - Es el estándar de PyTorch y HuggingFace para imágenes

    Parámetro de entrada:
        imagen → array uint8 (512, 512) en grises

    Valor de retorno:
        tensor → array float32 (1, 512, 512), valores en [0.0, 1.0]
    """

    tensor = imagen.astype(np.float32) / 255.0

    if tensor.ndim == 3:
        tensor = np.transpose(tensor, (2, 0, 1))
    elif tensor.ndim == 2:
        tensor = tensor[np.newaxis, :, :]

    return tensor


# =============================================================================
# PASO 4 — ETIQUETADO SEMÁNTICO (segmentación de baches)
# Detección automática de zonas oscuras en pavimento
# =============================================================================

def paso4_segmentar_bache(imagen_gris):
    """
    ¿Qué hace esta función?
    ───────────────────────
    Detecta automáticamente qué píxeles pertenecen a un bache.
    Los baches se detectan por ser zonas más oscuras que el pavimento
    circundante.

    Proceso en 4 etapas:
    ─────────────────────
    1. Suavizado gaussiano → reduce ruido de textura del asfalto
    2. Umbral adaptativo   → detecta regiones más oscuras que su entorno
    3. Cierre morfológico  → rellena huecos dentro del bache detectado
    4. Apertura morfológica → elimina puntos aislados (manchas, sombras)

    Parámetro de entrada:
        imagen_gris → array (512, 512) uint8

    Valor de retorno:
        mascara    → array (512, 512) uint8 — 255 donde hay bache, 0 en el resto
        contornos  → lista de arrays con las coordenadas del borde del bache
    """

    # ETAPA 1: Suavizado gaussiano
    suavizada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

    # ETAPA 2: Umbral adaptativo gaussiano
    # blockSize=31 → analiza ventanas de 31×31 píxeles
    # C=10         → bache si 10 unidades más oscuro que su media local
    mascara = cv2.adaptiveThreshold(
        suavizada, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=10
    )

    # ETAPA 3: Cierre morfológico — rellena huecos dentro del bache
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel_cierre)

    # ETAPA 4: Apertura morfológica — elimina puntos aislados
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_apertura)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    AREA_MINIMA = 500   # píxeles²
    contornos_validos = [c for c in contornos if cv2.contourArea(c) >= AREA_MINIMA]

    mascara_final = np.zeros_like(mascara)
    cv2.drawContours(mascara_final, contornos_validos, -1, 255, thickness=cv2.FILLED)

    return mascara_final, contornos_validos


# =============================================================================
# PASO 5 — AUMENTO DE DATOS (referencia — ya aplicado en entrenamiento)
# El aumento se realizó en Roboflow antes del entrenamiento en Google Colab
# =============================================================================

def paso5_aumentar_datos(imagen, mascara=None):
    """
    NOTA — REDUNDANCIA ELIMINADA
    ─────────────────────────────
    El aumento de datos (flip horizontal + rotaciones) ya fue aplicado
    en Roboflow durante la preparación del dataset y está incorporado
    en los pesos del modelo entrenado.

    Esta función devuelve únicamente la imagen original sin transformaciones.
    Se conserva en el pipeline para mantener compatibilidad con el flujo.

    Parámetro de entrada:
        imagen  → array (512, 512) uint8
        mascara → array (512, 512) uint8 (opcional)

    Valor de retorno:
        lista con 1 dict: solo la imagen original
    """

    return [{
        'imagen':  imagen.copy(),
        'mascara': mascara.copy() if mascara is not None else None,
        'tipo':    'original'
    }]


# =============================================================================
# PASO 6 — MODELO SEGFORMER (inferencia / predicción)
# SegFormer-B2 fine-tuneado con dataset de baches peruanos
# =============================================================================

# =============================================================================
# CONFIGURACIÓN DEL MODELO SEGFORMER
# =============================================================================
#
# MODOS DE OPERACIÓN — se seleccionan automáticamente en este orden:
#
#   MODO A — Modelo local (pesos propios entrenados en Google Colab):
#       Crea la carpeta ./modelo_segformer/ con tu config.json,
#       model.safetensors y preprocessor_config.json.
#
#   MODO B — Modelo en HuggingFace Hub:
#       Cambia RUTA_MODELO a un repo-id, por ejemplo:
#           RUTA_MODELO = "usuario/modelo"
#       El modelo se descarga automáticamente la primera vez.
#
#   MODO C — Umbral adaptativo (RESPALDO sin modelo disponible):
#       Si ninguno de los anteriores está disponible, el pipeline usa
#       el umbral adaptativo gaussiano del paso4_segmentar_bache().
#       No requiere torch ni GPU.
#
# =============================================================================

RUTA_MODELO = os.environ.get(
    "SEGFORMER_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_segformer")
)
CLASE_BACHE = int(os.environ.get("SEGFORMER_POTHOLE_CLASS", "1"))

# Cache del modelo (carga diferida — solo cuando se necesita)
_segformer_cache = {"processor": None, "model": None, "device": None,
                    "disponible": None}


def _intentar_cargar_segformer():
    """
    Intenta cargar el modelo SegFormer. Devuelve True si tuvo éxito.
    Si falla (carpeta no existe, torch no disponible, etc.) devuelve False
    y el pipeline continúa con umbral adaptativo como respaldo.
    """
    if _segformer_cache["disponible"] is not None:
        return _segformer_cache["disponible"]

    es_local = os.path.isdir(RUTA_MODELO)
    es_hub   = not os.sep in RUTA_MODELO and "/" in RUTA_MODELO

    if not es_local and not es_hub:
        print(f"     [SegFormer] Modo C: carpeta '{RUTA_MODELO}' no encontrada.")
        print(f"     [SegFormer] Usando umbral adaptativo como predictor.")
        print(f"     [SegFormer] Para activar el modelo real:")
        print(f"       · Coloca tus pesos en '{RUTA_MODELO}/' "
              f"(config.json + model.safetensors + preprocessor_config.json)")
        _segformer_cache["disponible"] = False
        return False

    try:
        import torch as _torch
        from transformers import (SegformerForSemanticSegmentation as _Seg,
                                  SegformerImageProcessor as _Proc)

        device = "cuda" if _torch.cuda.is_available() else "cpu"
        ruta   = RUTA_MODELO
        print(f"     [SegFormer] Cargando modelo desde: '{ruta}'  (device={device})")

        processor = _Proc.from_pretrained(ruta)
        model     = _Seg.from_pretrained(ruta)
        model.to(device)
        model.eval()

        _segformer_cache["processor"]   = processor
        _segformer_cache["model"]       = model
        _segformer_cache["device"]      = device
        _segformer_cache["disponible"]  = True

        print(f"     [SegFormer] Modelo listo ✓  "
              f"Clases: {model.config.num_labels} | Clase bache = {CLASE_BACHE}")
        return True

    except Exception as e:
        print(f"     [SegFormer] No se pudo cargar el modelo: {e}")
        print(f"     [SegFormer] Continuando con umbral adaptativo.")
        _segformer_cache["disponible"] = False
        return False


def aplicar_clahe(imagen_rgb):
    """
    Realza el contraste local en zonas de sombra sin sobre-exponer las
    zonas ya iluminadas (CLAHE — Contrast Limited Adaptive Histogram
    Equalization). Se usa para ayudar al modelo a distinguir un bache
    real oculto en sombra fuerte de la sombra misma.

    CRÍTICO: esta misma función debe aplicarse EXACTAMENTE igual durante
    el entrenamiento (en PavescanDataset, en Colab) y en la inferencia
    (aquí) — si difieren, el modelo ve una distribución de imagen
    distinta a la que aprendió y el desempeño empeora en vez de mejorar.

    Parámetro:
        imagen_rgb → array uint8 (H, W, 3) en formato RGB

    Valor de retorno:
        array uint8 (H, W, 3) RGB con contraste realzado
    """
    lab = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_mejorado = clahe.apply(l)
    lab_mejorado = cv2.merge([l_mejorado, a, b])
    return cv2.cvtColor(lab_mejorado, cv2.COLOR_LAB2RGB)


def paso6_segformer_predecir(imagen_gris_512, imagen_rgb_512=None):
    """
    Ejecuta la predicción de segmentación de baches con SegFormer.

    ── MODO A/B: SegFormer fine-tuneado ──
      1. Usa imagen RGB real (si está disponible) o convierte gris → RGB
      2. Normaliza con SegformerImageProcessor (media/std de ImageNet)
      3. Inferencia → logits (1, clases, H/4, W/4)
      4. Interpolación bilineal → 512×512
      5. Argmax por píxel → clase ganadora
      6. Máscara binaria: clase==CLASE_BACHE → 255, resto → 0

    ── MODO C: Umbral adaptativo gaussiano (respaldo sin modelo) ──
      Detección por intensidad usando ventana local de 31×31 px y morfología.

    Parámetros de entrada:
        imagen_gris_512 → array uint8 (512, 512) — para MODO C
        imagen_rgb_512  → array uint8 (512, 512, 3) — para MODO A/B (opcional)

    Valor de retorno:
        mascara_pred → array uint8 (512, 512) — 255=bache, 0=fondo
    """

    modelo_ok = _intentar_cargar_segformer()

    if modelo_ok:
        # ── MODO A/B: inferencia con SegFormer real ───────────────────────────
        import torch as _torch

        processor = _segformer_cache["processor"]
        model     = _segformer_cache["model"]
        device    = _segformer_cache["device"]

        # Usar imagen RGB real si está disponible (mejor calidad de predicción)
        # Si no, convertir grises → RGB como respaldo
        if imagen_rgb_512 is not None:
            if imagen_rgb_512.shape[:2] != (512, 512):
                imagen_rgb = cv2.resize(imagen_rgb_512, (512, 512),
                                        interpolation=cv2.INTER_AREA)
            else:
                imagen_rgb = imagen_rgb_512
        else:
            imagen_rgb = cv2.cvtColor(imagen_gris_512, cv2.COLOR_GRAY2RGB)

        # NOTA: CLAHE se probó como técnica de mejora (realce de contraste en
        # sombras) pero se descartó tras comparar métricas reales — empeoró
        # Recall/F1/IoU frente al modelo sin CLAHE. Se deja la función
        # aplicar_clahe() definida por si se retoma en el futuro, pero NO
        # se aplica aquí porque el modelo actual fue entrenado sin ella
        # (aplicarla causaría un desajuste entrenamiento/inferencia).
        # imagen_rgb = aplicar_clahe(imagen_rgb)

        # Preprocesamiento y normalización con valores ImageNet
        entradas = processor(images=imagen_rgb, return_tensors="pt")
        entradas = {k: v.to(device) for k, v in entradas.items()}

        # Inferencia sin gradientes
        with _torch.no_grad():
            salidas = model(**entradas)

        # Interpolar logits de 128×128 a 512×512
        logits_full = _torch.nn.functional.interpolate(
            salidas.logits, size=(512, 512),
            mode="bilinear", align_corners=False
        )

        # Argmax → clase por píxel → máscara binaria
        pred_classes = logits_full.argmax(dim=1).squeeze(0).cpu().numpy()
        mascara_pred = np.where(pred_classes == CLASE_BACHE, 255, 0).astype(np.uint8)

    else:
        # ── MODO C: umbral adaptativo (respaldo sin modelo) ───────────────────
        mascara_pred, _ = paso4_segmentar_bache(imagen_gris_512)

    return mascara_pred


# =============================================================================
# PASO 7 — MÉTRICAS DE EVALUACIÓN
# Evaluación del rendimiento del modelo SegFormer
# =============================================================================

def paso7_calcular_metricas(prediccion, ground_truth):
    """
    ¿Qué miden estas métricas?
    ───────────────────────────
    Comparan píxel a píxel la predicción del modelo con la máscara real:

        TP (True Positive)  → píxel predicho BACHE que SÍ es bache
        FP (False Positive) → píxel predicho BACHE que NO es bache
        FN (False Negative) → píxel predicho FONDO que SÍ era bache
        TN (True Negative)  → píxel predicho FONDO que NO es bache

    Métricas calculadas:

    ┌─────────────────────────────────────────────────────────────────┐
    │  Recall    = TP / (TP + FN)                                    │
    │  Pregunta: de todos los baches reales, ¿cuántos detecté?      │
    │                                                                 │
    │  Precisión = TP / (TP + FP)                                    │
    │  Pregunta: de todo lo que marqué como bache, ¿cuánto era?     │
    │                                                                 │
    │  F1 Score  = 2 × P × R / (P + R)                              │
    │  Media armónica de Recall y Precisión. Balance entre ambas.   │
    │                                                                 │
    │  IoU       = TP / (TP + FP + FN)                              │
    │  Intersección / Unión. Mide qué tan bien se superponen        │
    │  la predicción y el ground truth. Alta IoU = buen contorno.   │
    └─────────────────────────────────────────────────────────────────┘

    Parámetros de entrada:
        prediccion   → array (H, W) — 0 o valor > 0
        ground_truth → array (H, W) — 0 o valor > 0

    Valor de retorno:
        dict con recall, precision, f1, iou, accuracy (valores en %)
    """

    pred = prediccion > 0
    real = ground_truth > 0

    TP = np.logical_and(pred,  real ).sum()
    FP = np.logical_and(pred,  ~real).sum()
    FN = np.logical_and(~pred, real ).sum()
    TN = np.logical_and(~pred, ~real).sum()

    eps = 1e-8

    recall    = TP / (TP + FN + eps)
    precision = TP / (TP + FP + eps)
    f1        = 2 * (precision * recall) / (precision + recall + eps)
    iou       = TP / (TP + FP + FN + eps)
    accuracy  = (TP + TN) / (TP + FP + FN + TN + eps)

    return {
        'recall':    round(float(recall    * 100), 2),
        'precision': round(float(precision * 100), 2),
        'f1':        round(float(f1        * 100), 2),
        'iou':       round(float(iou       * 100), 2),
        'accuracy':  round(float(accuracy  * 100), 2),
        'TP': int(TP), 'FP': int(FP),
        'FN': int(FN), 'TN': int(TN),
    }


# =============================================================================
# PASO 8 — NUBE DE PUNTOS 3D REAL (sensor RealSense D435i)
# Construye la nube de puntos a partir del archivo .npy de profundidad
# capturado en campo — YA NO SE SIMULA NINGÚN DATO.
# Especificaciones del sensor (Manual MTC — Sección 3.1):
#   · Altura de montaje: 1.25 metros
#   · FOV horizontal: 87° × 58° vertical
#   · Precisión: < 2% del error a 2 metros
# =============================================================================

def paso8_nube_puntos_desde_realsense(ruta_npy, mascara_bache=None,
                                       fx=420.0, fy=420.0, cx=424.0, cy=240.0):
    """
    Construye la nube de puntos 3D a partir de un archivo .npy de
    profundidad capturado con la Intel RealSense D435i en campo.

    ¿Qué es una nube de puntos?
    ────────────────────────────
    Es un conjunto de puntos en el espacio 3D, cada uno con coordenadas
    (X, Y, Z), obtenidos a partir de la profundidad real medida por el
    sensor en cada píxel — sin ningún dato simulado o inventado.

    ¿Cómo funciona la conversión píxel → 3D?
    ──────────────────────────────────────────
    El .npy contiene profundidad en milímetros (uint16) por cada píxel,
    en la resolución nativa del sensor (ej. 1280×720). Se usa el modelo
    de cámara pinhole:

        Z = profundidad_mm / 1000        (a metros)
        X = (u - cx) · Z / fx
        Y = (v - cy) · Z / fy

    Parámetros:
        ruta_npy       → ruta al archivo depth_*.npy real de campo
        mascara_bache  → array (H,W) booleano/uint8 — si se provee, solo
                         construye la nube dentro de esa región. Si es
                         None, construye la nube del frame completo
                         (necesario para que RANSAC encuentre el plano
                         real del pavimento).
        fx, fy, cx, cy → parámetros intrínsecos de la D435i (para 848×480;
                         se reescalan proporcionalmente si la captura es
                         de otra resolución, ej. 1280×720)

    Valor de retorno:
        array (N, 3) con columnas [X, Y, Z] en metros, o array vacío (0,3)
        si no hay ningún píxel válido tras el filtrado de rango.
    """
    if not os.path.isfile(ruta_npy):
        raise FileNotFoundError(f"[RealSense] No se encontró el archivo: '{ruta_npy}'")

    depth_mm = np.load(ruta_npy)   # uint16, milímetros, forma (H, W)
    h, w = depth_mm.shape

    # Reescalar intrínsecos si la captura no es 848×480 (D435i resolución variable)
    escala_x = w / 848.0
    escala_y = h / 480.0
    fx_r, fy_r = fx * escala_x, fy * escala_y
    cx_r, cy_r = cx * escala_x, cy * escala_y

    v_idx, u_idx = np.indices((h, w))

    # La RealSense marca píxeles sin lectura válida con 0, y algunos drivers
    # usan 65535 como centinela de saturación/fuera de rango. El rango físico
    # real del D435i es ~0.1–10 m, así que descartamos todo fuera de eso.
    RANGO_MIN_MM, RANGO_MAX_MM = 100, 10000
    valido_rango = (depth_mm >= RANGO_MIN_MM) & (depth_mm <= RANGO_MAX_MM)

    if mascara_bache is not None:
        if mascara_bache.shape != (h, w):
            mascara_bache = cv2.resize(
                mascara_bache.astype(np.uint8), (w, h),
                interpolation=cv2.INTER_NEAREST).astype(bool)
        valido = valido_rango & mascara_bache.astype(bool)
    else:
        valido = valido_rango

    Z = depth_mm[valido].astype(np.float32) / 1000.0    # mm → m
    U = u_idx[valido].astype(np.float32)
    V = v_idx[valido].astype(np.float32)

    X = (U - cx_r) * Z / fx_r
    Y = (V - cy_r) * Z / fy_r

    puntos = np.column_stack([X, Y, Z]).astype(np.float32)
    return puntos


# =============================================================================
# PASO 9 — RANSAC: Detectar el plano del pavimento
# Algoritmo para identificar el plano de referencia del pavimento
# Ecuación del plano: A·x + B·y + C·z + D = 0
# =============================================================================

def paso9_ransac(puntos, umbral=0.015, n_iteraciones=100):
    """
    ¿Por qué necesitamos RANSAC?
    ─────────────────────────────
    La nube de puntos tiene dos tipos de datos:
        - INLIERS: puntos del pavimento plano (la mayoría)
        - OUTLIERS: puntos del bache + reflexiones erróneas

    RANSAC encuentra el plano que mejor representa el pavimento
    ignorando los outliers.

    ¿Cómo funciona?
    ────────────────
    Repite N veces:
        1. Elegir 3 puntos al azar
        2. Calcular el plano que pasa por esos 3 puntos
        3. Contar cuántos puntos están a menos de `umbral` del plano
        4. Guardar el plano con más puntos dentro del umbral

    El plano se define por:  A·x + B·y + C·z + D = 0

    Parámetros:
        puntos        → array (N, 3) de la nube de puntos
        umbral        → distancia máxima al plano para ser inlier (metros)
        n_iteraciones → cuántas veces probar combinaciones aleatorias

    Valor de retorno:
        normal_plano     → vector (3,) perpendicular al pavimento
        mascara_inlier   → array booleano (N,) — True para puntos del pavimento
        puntos_alineados → nube rotada para que el pavimento sea horizontal
    """

    N = len(puntos)
    mejor_inliers  = np.zeros(N, dtype=bool)
    mejor_normal   = np.array([0.0, 0.0, 1.0])
    mejor_D        = 0.0
    np.random.seed(0)

    for _ in range(n_iteraciones):

        indices = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = puntos[indices]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norma  = np.linalg.norm(normal)

        if norma < 1e-8:
            continue

        normal = normal / norma
        D = -np.dot(normal, p1)

        distancias = np.abs(puntos @ normal + D)
        inliers    = distancias < umbral

        if inliers.sum() > mejor_inliers.sum():
            mejor_inliers = inliers
            mejor_normal  = normal
            mejor_D       = D

    # Alinear la nube: rotar el plano para que sea paralelo a XY
    eje_z = np.array([0.0, 0.0, 1.0])
    eje_rot = np.cross(mejor_normal, eje_z)
    sin_a   = np.linalg.norm(eje_rot)

    if sin_a > 1e-8:
        eje_rot /= sin_a
        cos_a = np.dot(mejor_normal, eje_z)
        angulo = np.arctan2(sin_a, cos_a)

        K = np.array([
            [0,          -eje_rot[2],  eje_rot[1]],
            [eje_rot[2],  0,          -eje_rot[0]],
            [-eje_rot[1], eje_rot[0],  0         ]
        ])
        R = np.eye(3) + np.sin(angulo) * K + (1 - np.cos(angulo)) * (K @ K)
        puntos_alineados = (R @ puntos.T).T
    else:
        puntos_alineados = puntos.copy()

    return mejor_normal, mejor_inliers, puntos_alineados.astype(np.float32)


def paso9b_filtrar_outliers(puntos, k_vecinos=10, factor=2.0):
    """
    Filtrado estadístico de outliers por distancia a vecinos.

    Para cada punto, calcula su distancia media a sus k vecinos más cercanos.
    Si esa distancia es mucho mayor que el promedio global, el punto es outlier.

    Parámetros:
        puntos    → array (N, 3) de la nube alineada
        k_vecinos → cuántos vecinos considerar
        factor    → umbral en desviaciones estándar (2.0 = ±2σ)

    Valor de retorno:
        puntos filtrados sin los outliers
    """

    N = len(puntos)
    if N < k_vecinos + 1:
        return puntos

    distancias_medias = np.zeros(N)

    for i in range(N):
        diffs = puntos - puntos[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists_ordenadas = np.sort(dists)[1:k_vecinos + 1]
        distancias_medias[i] = np.mean(dists_ordenadas)

    media = np.mean(distancias_medias)
    sigma = np.std(distancias_medias)

    umbral_estadistico = media + factor * sigma
    inliers = distancias_medias <= umbral_estadistico

    return puntos[inliers]


# =============================================================================
# PASO 10 — CONVERSIÓN 2D → 3D Y ESTIMACIÓN DE DIMENSIONES
# Medición de diámetro (Tabla 4-8 MTC) y profundidad (Sec.410/415 MTC)
# =============================================================================

def paso10_pixel_a_3d(u, v, altura_sensor=1.25, fx=420.0, fy=420.0,
                       cx=424.0, cy=240.0):
    """
    Convierte coordenadas de píxel (2D) a coordenadas del mundo (3D).

    Modelo de cámara pinhole:
        Z = altura_sensor   (conocida — el sensor está a 1.25 m del suelo)
        X = (u - cx) · Z / fx
        Y = (v - cy) · Z / fy

    Parámetros:
        u, v           → coordenadas del píxel (columna, fila)
        altura_sensor  → distancia sensor-suelo = Z conocida (1.25 m MTC)
        fx, fy         → distancias focales del RealSense D435i (~420 px)
        cx, cy         → centro óptico (~424, 240 para resolución 848×480)

    Valor de retorno:
        (X, Y, Z) en metros
    """

    Z = altura_sensor
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return float(X), float(Y), float(Z)


def paso10b_estimar_diametro(contorno, altura_sensor=1.25, fov_h=87.0,
                              res_ancho=512, res_alto=512, fov_v=58.0):
    """
    Estimación del diámetro del bache según Tabla 4-8 del Manual MTC.

    El Manual MTC clasifica baches por DIÁMETRO:
        Gravedad 1: Diámetro < 0.20 m  (< 20 cm)
        Gravedad 2: Diámetro 0.20-0.50 m (20-50 cm)
        Gravedad 3: Diámetro > 0.50 m  (> 50 cm)

    Proceso:
        1. Transformada de distancia para el mayor círculo inscrito
        2. Escala promediada horizontal + vertical
        3. Conversión a centímetros
        4. Validación de rangos según MTC

    IMPORTANTE — resolución del contorno vs. resolución del sensor:
        `contorno` viene siempre de la máscara ya redimensionada a 512×512
        (paso2_redimensionar), NO de la imagen nativa de la RealSense
        (848×480). Antes esta función asumía por defecto res_ancho=848 y
        res_alto=480 aunque el contorno real vivía en espacio 512×512, lo
        que inflaba el diámetro estimado en ~29% (la escala metros/píxel
        se calculaba dividiendo el mismo ancho real de campo de visión
        entre menos píxeles de los que realmente tiene la máscara).
        Por eso los valores por defecto ahora son 512×512.

    Parámetros:
        contorno      → array de puntos del borde del bache
        altura_sensor → metros desde el sensor hasta el suelo (1.25 m MTC)
        fov_h         → FOV horizontal del D435i en grados (87°)
        res_ancho     → resolución horizontal (848 px para D435i)
        res_alto      → resolución vertical   (480 px para D435i)
        fov_v         → FOV vertical del D435i en grados (58°)

    Valor de retorno:
        diámetro estimado en centímetros
    """

    if contorno is None or len(contorno) < 3:
        return 0.0

    area = cv2.contourArea(contorno)
    if area < 100:
        print(f"     [Diámetro] Contorno descartado: área={area:.0f} px² < 100 (ruido)")
        return 0.0

    if len(contorno) < 20:
        print(f"     [Diámetro] ⚠ Contorno con solo {len(contorno)} puntos — diámetro aproximado")

    # Transformada de distancia para el mayor círculo inscrito
    bbox = cv2.boundingRect(contorno)
    x, y, w, h = bbox
    margen = 10
    mascara_local  = np.zeros((h + 2*margen, w + 2*margen), dtype=np.uint8)
    contorno_local = contorno - np.array([[[x - margen, y - margen]]])
    cv2.drawContours(mascara_local, [contorno_local], -1, 255, cv2.FILLED)
    dist_transform = cv2.distanceTransform(mascara_local, cv2.DIST_L2, 5)
    radio_pixeles  = float(dist_transform.max())

    # Escala horizontal y vertical promediada
    theta_h = math.radians(fov_h)
    theta_v = math.radians(fov_v)
    W_real  = 2 * altura_sensor * math.tan(theta_h / 2)
    H_real  = 2 * altura_sensor * math.tan(theta_v / 2)
    H_scale = W_real / res_ancho
    V_scale = H_real / res_alto
    escala  = (H_scale + V_scale) / 2

    # Diámetro en cm
    diametro_m  = (2 * radio_pixeles) * escala
    diametro_cm = diametro_m * 100

    # Validar rangos según Manual MTC Tabla 4-8
    if diametro_cm < 5:
        print(f"     [Diámetro MTC] ⚠ Valor muy bajo ({diametro_cm:.1f} cm) — posible ruido")
    elif diametro_cm < 20:
        print(f"     [Diámetro MTC] ✓ {diametro_cm:.2f} cm → Gravedad 1 (D < 0.20 m) — Tabla 4-8 MTC")
    elif diametro_cm <= 50:
        print(f"     [Diámetro MTC] ✓ {diametro_cm:.2f} cm → Gravedad 2 (0.20-0.50 m) — Tabla 4-8 MTC")
    else:
        print(f"     [Diámetro MTC] ✓ {diametro_cm:.2f} cm → Gravedad 3 (D > 0.50 m) — Tabla 4-8 MTC")

    return round(diametro_cm, 2)


def paso10c_estimar_profundidad(puntos_bache):
    """
    Estimación de la profundidad del bache.

    Referencia MTC:
        Sec.410: Profundidad < 50 mm → Parchado Superficial
        Sec.415: Profundidad > 50 mm → Parchado Profundo

    Fórmula:
        Profundidad = Z_punto_más_profundo − Z_superficie

    Donde:
        Z_superficie     → mediana del percentil inferior (nivel del pavimento)
        Z_punto_más_profundo → percentil 95 dentro del bache

    Parámetro:
        puntos_bache → array (N, 3) solo con los puntos del área del bache

    Valor de retorno:
        profundidad en milímetros
    """

    if len(puntos_bache) < 5:
        return 0.0

    z_valores = puntos_bache[:, 2]

    z_inf          = z_valores[z_valores <= np.percentile(z_valores, 40)]
    z_superficie   = float(np.median(z_inf)) if len(z_inf) > 0 \
                     else float(np.percentile(z_valores, 25))

    z_mas_profundo = float(np.percentile(z_valores, 95))

    if z_mas_profundo <= z_superficie:
        print(f"     [Profundidad MTC] ⚠ No se detectó depresión significativa")
        return 0.0

    profundidad_m  = z_mas_profundo - z_superficie
    profundidad_mm = max(0.0, profundidad_m * 1000)

    # Validar umbral normativo MTC Sec.410/415
    if profundidad_mm < 50:
        print(f"     [Profundidad MTC] ✓ {profundidad_mm:.2f} mm → Sec.410 Parchado Superficial")
    else:
        print(f"     [Profundidad MTC] ✓ {profundidad_mm:.2f} mm → Sec.415 Parchado Profundo")

    return round(profundidad_mm, 2)


# =============================================================================
# PASO 11 — SISTEMA DE LÓGICA DIFUSA MAMDANI — NORMATIVA MTC
# Base: Manual de Carreteras RD N° 08-2014-MTC/14
# Variables de entrada: Diámetro (Tabla 4-8) y Profundidad (Sec.410/415)
# Variables de salida:  Nivel de Gravedad (1/2/3) y Técnica de Conservación
# =============================================================================

def _funcion_membresia_trapezoidal(x, a, b, c, d):
    """
    Función de membresía trapezoidal para lógica difusa.

    Forma del trapecio:
        ___________
       /           \\
      /             \\
    a  b           c  d

    - Entre a y b: membresía sube de 0 a 1
    - Entre b y c: membresía = 1 (meseta superior)
    - Entre c y d: membresía baja de 1 a 0
    - Fuera de [a,d]: membresía = 0

    Parámetros:
        x       → valor de entrada (normalizado [0,1])
        a,b,c,d → parámetros del trapecio

    Valor de retorno:
        grado de membresía en [0, 1]

    NOTA: la meseta (b<=x<=c) se comprueba antes que "fuera de rango"
    (x<=a o x>=d) — ver la explicación completa en membresia_trapezoidal()
    de pavescan_fuzzy_mtc.py. Sin este orden, una entrada recortada
    (clamp) exactamente a 0.0 o 1.0 en una categoría abierta (b==a o
    c==d) devolvía 0.0 en vez de 1.0, apagando la categoría entera.
    """

    if b <= x <= c:
        return 1.0
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    elif c < x < d:
        return (d - x) / (d - c) if d != c else 1.0
    return 0.0


def paso11_logica_difusa(diametro_cm, profundidad_mm):
    """
    Sistema de lógica difusa Mamdani basado en el Manual MTC.

    Variables de entrada (Manual MTC):
    ─────────────────────────────────
    Diámetro (Tabla 4-8 — Falla 7 Baches):
        PEQUEÑO:  Diámetro < 0.20 m
        MEDIANO:  Diámetro 0.20-0.50 m
        GRANDE:   Diámetro > 0.50 m

    Profundidad (Sec.410 y Sec.415):
        SUPERFICIAL: Profundidad < 50 mm → Parchado Superficial
        PROFUNDA:    Profundidad > 50 mm → Parchado Profundo

    Variables de salida (Cap.400 MTC):
    ───────────────────────────────────
    Gravedad:   GRAVEDAD_1 / GRAVEDAD_2 / GRAVEDAD_3
    Técnica:    PARCHADO_SUPERFICIAL (Sec.410) /
                PARCHADO_PROFUNDO    (Sec.415)
    Ambas técnicas son CONSERVACION RUTINARIA según Cap.400 MTC.

    Etapas del sistema Mamdani:
        1. Fuzzificación  → normalizar entradas y calcular membresías
        2. Evaluación     → aplicar reglas difusas (AND = mínimo)
        3. Agregación     → combinar reglas (OR = máximo)
        4. Defuzzificación → convertir resultado a número real (centroide)

    Parámetros:
        diametro_cm    → diámetro estimado del bache (cm)
        profundidad_mm → profundidad estimada del bache (mm)

    Valor de retorno:
        dict con gravedad, tecnica, valores numéricos y membresías
    """

    if _MTC_MODULO_OK:
        # Usar módulo MTC completo con todas las referencias normativas
        return _mtc_evaluar(diametro_cm, profundidad_mm)

    # ── Fallback: lógica difusa local ────────────────────────────────────────
    # Normalización MTC: diámetro máx=100cm, profundidad máx=150mm
    d = max(0.0, min(1.0, diametro_cm    / 100.0))
    p = max(0.0, min(1.0, profundidad_mm / 150.0))

    # Membresías de DIÁMETRO — Tabla 4-8 MTC
    mem_diam = {
        'PEQUENIO': _funcion_membresia_trapezoidal(d, 0.00, 0.00, 0.17, 0.23),
        'MEDIANO':  _funcion_membresia_trapezoidal(d, 0.17, 0.23, 0.47, 0.53),
        'GRANDE':   _funcion_membresia_trapezoidal(d, 0.47, 0.53, 1.00, 1.00),
    }

    # Membresías de PROFUNDIDAD — Sec.410/415 MTC
    mem_prof = {
        'SUPERFICIAL': _funcion_membresia_trapezoidal(p, 0.00, 0.00, 0.267, 0.400),
        'PROFUNDA':    _funcion_membresia_trapezoidal(p, 0.267, 0.400, 1.00, 1.00),
    }

    # Activaciones de salida — solo Sec.410 y Sec.415 (ambas RUTINARIA)
    act_gravedad = {'GRAVEDAD_1': 0.0, 'GRAVEDAD_2': 0.0, 'GRAVEDAD_3': 0.0}
    act_tecnica  = {
        'PARCHADO_SUPERFICIAL': 0.0,
        'PARCHADO_PROFUNDO':    0.0,
    }

    # Base de reglas — Manual MTC Cap.400
    # Diámetro (Tabla 4-8) → Gravedad
    # Profundidad (Sec.410/415) → Técnica (SIEMPRE Conservación Rutinaria)
    reglas = [
        ('SUPERFICIAL', 'PEQUENIO', 'GRAVEDAD_1', 'PARCHADO_SUPERFICIAL'),
        ('SUPERFICIAL', 'MEDIANO',  'GRAVEDAD_2', 'PARCHADO_SUPERFICIAL'),
        ('SUPERFICIAL', 'GRANDE',   'GRAVEDAD_3', 'PARCHADO_SUPERFICIAL'),
        ('PROFUNDA',    'PEQUENIO', 'GRAVEDAD_1', 'PARCHADO_PROFUNDO'),
        ('PROFUNDA',    'MEDIANO',  'GRAVEDAD_2', 'PARCHADO_PROFUNDO'),
        ('PROFUNDA',    'GRANDE',   'GRAVEDAD_3', 'PARCHADO_PROFUNDO'),
    ]

    for prof_cat, diam_cat, gravedad, tecnica in reglas:
        activacion = min(mem_prof[prof_cat], mem_diam[diam_cat])
        if activacion > 0:
            act_gravedad[gravedad] = max(act_gravedad[gravedad], activacion)
            act_tecnica[tecnica]   = max(act_tecnica[tecnica],   activacion)

    # Defuzzificación — centroide
    # Solo dos técnicas: Sec.410 y Sec.415 (ambas Conservación Rutinaria)
    valores_gravedad = {'GRAVEDAD_1': 0.20, 'GRAVEDAD_2': 0.55, 'GRAVEDAD_3': 0.90}
    valores_tecnica  = {
        'PARCHADO_SUPERFICIAL': 0.35,
        'PARCHADO_PROFUNDO':    0.75,
    }

    def defuzzificar(activaciones, valores_crisp):
        num = sum(activaciones[k] * valores_crisp[k]
                  for k in activaciones if activaciones[k] > 0)
        den = sum(activaciones[k]
                  for k in activaciones if activaciones[k] > 0)
        if den < 1e-8:
            return 0.0, list(activaciones.keys())[0]
        return round(num / den, 4), max(activaciones, key=activaciones.get)

    valor_grav, etiq_grav = defuzzificar(act_gravedad, valores_gravedad)
    valor_tec,  etiq_tec  = defuzzificar(act_tecnica,  valores_tecnica)

    # Nombre legible de la técnica
    nombres_tec = {
        'PARCHADO_SUPERFICIAL': 'Parchado Superficial en Calzada',
        'PARCHADO_PROFUNDO':    'Parchado Profundo en Calzada',
    }
    secciones_tec = {
        'PARCHADO_SUPERFICIAL': 'Sec.410 — Manual MTC RD N 08-2014-MTC/14',
        'PARCHADO_PROFUNDO':    'Sec.415 — Manual MTC RD N 08-2014-MTC/14',
    }

    return {
        'diametro_cm':        round(diametro_cm, 2),
        'profundidad_mm':     round(profundidad_mm, 2),
        'gravedad':           etiq_grav,
        'valor_gravedad':     valor_grav,
        'tecnica':            etiq_tec,
        'nombre_tecnica':     nombres_tec.get(etiq_tec, etiq_tec),
        'seccion_mtc':        secciones_tec.get(etiq_tec, 'Cap.400 MTC'),
        'valor_tecnica':      valor_tec,
        'tipo_conservacion':  'CONSERVACION_RUTINARIA',
        'nombre_conservacion':'Conservacion Rutinaria',
        'capitulo_mtc':       'Cap.400 — Manual MTC RD N 08-2014-MTC/14',
        'urgencia':           etiq_grav,       # compatibilidad GUI
        'valor_urgencia':     valor_grav,
        'recomendacion':      etiq_tec,
        'valor_rec':          valor_tec,
        'membresias_diam':    {k: round(v, 3) for k, v in mem_diam.items()},
        'membresias_prof':    {k: round(v, 3) for k, v in mem_prof.items()},
        'base_normativa':     'RD N° 08-2014-MTC/14 / RD N° 05-2016-MTC/14',
    }


# =============================================================================
# CARGA DE IMÁGENES REALES DESDE EL DATASET ROBOFLOW
# =============================================================================

RUTA_DATASET = os.environ.get(
    "PAVESCAN_DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
)

def _listar_pares(split="test"):
    """
    Devuelve lista de tuplas (ruta_imagen, ruta_mascara) para el split indicado.
    Solo incluye pares donde AMBOS archivos (.jpg y _mask.png) existen.
    """
    carpeta = os.path.join(RUTA_DATASET, split)
    if not os.path.isdir(carpeta):
        raise FileNotFoundError(
            f"\n[Dataset] No se encontró la carpeta: '{carpeta}'\n"
            f"  ► Estructura esperada: {RUTA_DATASET}/train/, /valid/, /test/\n"
        )

    pares = []
    for nombre in sorted(os.listdir(carpeta)):
        if not nombre.endswith(".jpg"):
            continue
        base         = nombre[:-4]
        ruta_img     = os.path.join(carpeta, nombre)
        ruta_mascara = os.path.join(carpeta, base + "_mask.png")
        if os.path.isfile(ruta_mascara):
            pares.append((ruta_img, ruta_mascara))

    if not pares:
        raise FileNotFoundError(
            f"\n[Dataset] No se encontraron pares imagen/_mask.png en '{carpeta}'.\n"
            f"  ► Formato Roboflow: <nombre>.jpg + <nombre>_mask.png\n"
        )

    return pares


def cargar_imagen_dataset(ruta_imagen, ruta_mascara=None):
    """
    Carga un par imagen+máscara del dataset Roboflow.

    Retorna:
        img_rgb    → array uint8 (512, 512, 3)
        mascara_gt → array uint8 (512, 512) — 255 donde hay BACHE, 0 en fondo
    """
    img_bgr = cv2.imdecode(
        np.fromfile(ruta_imagen, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if img_bgr is None:
        raise FileNotFoundError(f"[Dataset] No se pudo leer: '{ruta_imagen}'")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if img_rgb.shape[:2] != (512, 512):
        img_rgb = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)

    mascara_gt = None
    if ruta_mascara is not None and os.path.isfile(ruta_mascara):
        from PIL import Image as _PIL_Image
        mask_raw   = np.array(_PIL_Image.open(ruta_mascara))
        mascara_gt = np.where(mask_raw == 1, 255, 0).astype(np.uint8)
# Redimensionar máscara al mismo tamaño que la imagen (512×512)
    if mascara_gt.shape != (512, 512):
        mascara_gt = cv2.resize(mascara_gt, (512, 512),
                            interpolation=cv2.INTER_NEAREST)

    return img_rgb, mascara_gt


def _seleccionar_imagen_dataset(split="test", indice=None):
    """Selecciona una imagen del dataset para procesar."""
    pares = _listar_pares(split)

    if indice is None:
        import random
        indice = random.randint(0, len(pares) - 1)
    else:
        indice = max(0, min(indice, len(pares) - 1))

    ruta_img, ruta_mask = pares[indice]
    img_rgb, mascara_gt = cargar_imagen_dataset(ruta_img, ruta_mask)

    return img_rgb, mascara_gt, ruta_img, ruta_mask, len(pares)


# =============================================================================
# PIPELINE COMPLETO — Integración de los pasos
# Evaluación de baches según Manual MTC RD N° 08-2014-MTC/14
# =============================================================================

def ejecutar_pipeline(diametro_manual=None, profundidad_manual=None,
                      split="test", indice_imagen=None,
                      modo_normativo="MTC",
                      ruta_imagen_manual=None,
                      ruta_depth_npy=None):
    """
    Ejecuta el pipeline completo de evaluación de baches según el Manual MTC.

    Filosofía de esta versión: NINGÚN dato se inventa o simula.
        - La imagen viene del dataset real o de una captura real de campo.
        - El ground truth solo existe si hay una máscara real disponible
          (dataset anotado o archivo "_mask.png" hermano); si no existe,
          simplemente no se compara contra nada — no se fabrica.
        - La detección de "¿hay bache?" depende únicamente de lo que el
          SegFormer realmente predice.
        - La profundidad solo se calcula si hay un .npy real de la
          RealSense (o un valor medido a mano por el usuario); si no hay
          ninguno de los dos, el programa lo dice claramente en vez de
          inventar un número.

    Parámetros opcionales:
        diametro_manual    → valor de diámetro medido a mano (cm), se usa
                              solo como referencia informativa; el diámetro
                              real siempre se calcula del contorno predicho
        profundidad_manual → profundidad medida a mano por el usuario (mm),
                              p. ej. con cinta métrica en campo — se usa
                              como dato real cuando no hay .npy disponible
        split              → 'train', 'valid' o 'test'
        indice_imagen      → índice de la imagen (None = aleatorio)
        ruta_imagen_manual → ruta a una imagen elegida por el usuario
        ruta_depth_npy     → ruta al .npy real de profundidad de la
                              RealSense D435i, correspondiente a la misma
                              captura que ruta_imagen_manual

    Valor de retorno:
        dict con los resultados de cada paso y la evaluación MTC
    """

    print("\n" + "═"*60)
    print("  PAVESCAN AI — Evaluación de Baches en Pavimento")
    print("  UNSCH · Ingeniería Civil · Huamanga, Ayacucho, Perú")
    print("  Manual MTC RD N° 08-2014-MTC/14 / RD N° 05-2016-MTC/14")
    print("═"*60)

    # ── PASO 1: Cargar imagen ─────────────────────────────────────────────────
    mascara_gt_real = None
    tiene_gt_valido = False
    print(f"\n[01] Cargando imagen...")

    # Si se seleccionó una imagen manualmente desde la GUI, usarla directamente
    if ruta_imagen_manual is not None and os.path.isfile(ruta_imagen_manual):
        img_bgr = cv2.imdecode(
            np.fromfile(ruta_imagen_manual, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"No se pudo leer la imagen: '{ruta_imagen_manual}'")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        nombre_archivo = os.path.basename(ruta_imagen_manual)
        print(f"     ✓ Imagen manual: {nombre_archivo}")
        print(f"     Resolución: {img_rgb.shape[1]}×{img_rgb.shape[0]} px")

        # Buscar máscara hermana real (mismo patrón Roboflow: <nombre>_mask.png)
        # — solo se usa si REALMENTE existe; no se fabrica ninguna.
        base_sin_ext, _ext = os.path.splitext(ruta_imagen_manual)
        ruta_mask_hermana = base_sin_ext + "_mask.png"
        if os.path.isfile(ruta_mask_hermana):
            from PIL import Image as _PIL_Image_local
            mask_raw = np.array(_PIL_Image_local.open(ruta_mask_hermana))
            mascara_gt_real = np.where(mask_raw == 1, 255, 0).astype(np.uint8)
            if mascara_gt_real.shape != (512, 512):
                mascara_gt_real = cv2.resize(
                    mascara_gt_real, (512, 512), interpolation=cv2.INTER_NEAREST)
            tiene_gt_valido = True
            tiene_bache_gt = mascara_gt_real.any()
            print(f"     Máscara GT: {'CON bache ✓' if tiene_bache_gt else 'sin bache (negativo)'} "
                  f"(encontrada: {os.path.basename(ruta_mask_hermana)})")
        else:
            print(f"     Máscara GT: no disponible — imagen de campo nueva, "
                  f"modo DETECCIÓN (no evaluación).")
    else:
        # Sin imagen manual: cargar una real del dataset anotado
        img_rgb, mascara_gt_real, ruta_img, ruta_mask, total = \
            _seleccionar_imagen_dataset(split=split, indice=indice_imagen)
        nombre_archivo = os.path.basename(ruta_img)
        tiene_gt_valido = True   # toda imagen del dataset tiene máscara conocida
        tiene_bache_gt  = mascara_gt_real is not None and mascara_gt_real.any()
        print(f"     ✓ Imagen: {nombre_archivo}")
        print(f"     Resolución: {img_rgb.shape[1]}×{img_rgb.shape[0]} px")
        print(f"     Máscara GT: {'CON bache ✓' if tiene_bache_gt else 'sin bache (negativo)'}")
        print(f"     Total pares en '{split}': {total}")

    # ── PASO 2: Escala de grises ──────────────────────────────────────────────
    print("[02] Conversión a escala de grises (Y = 0.299R + 0.587G + 0.114B)...")
    img_gris = paso1_escala_de_grises(img_rgb)
    print(f"     Rango de luminancia: [{img_gris.min()}, {img_gris.max()}]")

    # ── PASO 3: Resize 512×512 ────────────────────────────────────────────────
    print("[03] Redimensionando a 512×512 px...")
    img_512 = paso2_redimensionar(img_gris, 512)
    print(f"     Forma resultante: {img_512.shape}")

    # ── PASO 4: Píxel → Tensor ────────────────────────────────────────────────
    print("[04] Convirtiendo a tensor Float32 [0, 1]...")
    tensor = paso3_pixel_a_tensor(img_512)
    print(f"     Tensor: forma={tensor.shape}, min={tensor.min():.4f}, max={tensor.max():.4f}")

    # ── PASO 5: Ground truth ──────────────────────────────────────────────────
    print("[05] Obteniendo ground truth (máscara de referencia)...")
    if tiene_gt_valido:
        mascara_gt = mascara_gt_real
        contornos_gt, _ = cv2.findContours(
            mascara_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        px_bache_gt = int((mascara_gt > 0).sum())
        print(f"     Fuente: máscara real (dataset anotado / Roboflow PNG mask)")
        print(f"     Bache en GT: {px_bache_gt:,} px  ({px_bache_gt/(512*512)*100:.1f}% imagen)")
        print(f"     Contornos: {len(contornos_gt)}")
    else:
        # Imagen de campo nueva sin anotación previa: NO se inventa un
        # pseudo-ground-truth con umbral clásico. La detección real se
        # basa exclusivamente en la predicción del SegFormer (Paso 6).
        mascara_gt = None
        contornos_gt = []
        print(f"     Sin ground truth disponible — modo DETECCIÓN de campo.")
        print(f"     La presencia de bache se determina por la predicción")
        print(f"     real del SegFormer-B2 (Paso 6), no por un umbral clásico.")

    # ── PASO 6: SegFormer inferencia ──────────────────────────────────────────
    print("[06] Inferencia SegFormer-B2 (modelo fine-tuneado)...")
    print(f"     · Ruta modelo: {RUTA_MODELO}")
    print(f"     · Clase bache id: {CLASE_BACHE}")
    img_rgb_512 = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA) \
                  if img_rgb.shape[:2] != (512, 512) else img_rgb
    mascara_pred = paso6_segformer_predecir(img_512, imagen_rgb_512=img_rgb_512)

    # Área mínima para considerar un contorno como bache real (no ruido).
    # Un modelo poco entrenado suele generar manchas aisladas de pocos
    # píxeles — se descartan aquí en vez de tratarlas como bache real.
    AREA_MINIMA_DETECCION = 150
    contorno_principal = None
    contornos_pred, _ = cv2.findContours(
        mascara_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_pred_validos = [
        c for c in contornos_pred if cv2.contourArea(c) >= AREA_MINIMA_DETECCION]
    if contornos_pred_validos:
        contorno_principal = max(contornos_pred_validos, key=cv2.contourArea)
    px_pred = int((mascara_pred > 0).sum())
    print(f"     Predicción: {px_pred:,} px como bache "
          f"({len(contornos_pred)} contorno(s), "
          f"{len(contornos_pred_validos)} válido(s) ≥{AREA_MINIMA_DETECCION}px)")

    # ── PASO 7: Aumento de datos (referencia informativa) ─────────────────────
    print("[07] Aumento de datos (aplicado en Roboflow antes del entrenamiento)...")
    print(f"     Roboflow aplicó: flip horizontal + rotación 180° (×3 imágenes por original)")

    # ── PASO 8: Métricas ──────────────────────────────────────────────────────
    if tiene_gt_valido:
        print("[08] Calculando métricas SegFormer vs Ground Truth...")
        metricas = paso7_calcular_metricas(mascara_pred, mascara_gt)
        print(f"     Recall:    {metricas['recall']:>6.2f}%  (TP={metricas['TP']:,}, FN={metricas['FN']:,})")
        print(f"     Precisión: {metricas['precision']:>6.2f}%  (TP={metricas['TP']:,}, FP={metricas['FP']:,})")
        print(f"     F1 Score:  {metricas['f1']:>6.2f}%")
        print(f"     IoU:       {metricas['iou']:>6.2f}%")
    else:
        print("[08] Sin ground truth — métricas de validación no aplican "
              "(modo detección de campo, no evaluación).")
        metricas = {'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'iou': 0.0,
                    'accuracy': 0.0, 'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

    # ── DETECCIÓN — basada siempre en la predicción real, nunca en el GT ─────
    tiene_bache = contorno_principal is not None and px_pred > AREA_MINIMA_DETECCION
    print(f"     [Detección] → {'BACHE DETECTADO' if tiene_bache else 'sin bache (ruido descartado)'}")

    if not tiene_bache:
        print("\n" + "─"*60)
        print("  RESULTADO: NO SE DETECTÓ NINGÚN BACHE EN ESTA IMAGEN")
        print("  El SegFormer no encontró una región con confianza")
        print("  suficiente. No se calculan dimensiones ni gravedad MTC")
        print("  porque no hay bache real que evaluar.")
        print("─"*60 + "\n")

        resultado_vacio = {
            'diametro_cm': 0.0, 'profundidad_mm': 0.0,
            'gravedad': 'SIN_BACHE',
            'descripcion_gravedad': 'No se detectó ningún bache en la imagen evaluada.',
            'tecnica': 'NINGUNA', 'nombre_tecnica': 'No aplica — sin bache detectado',
            'seccion_mtc': '', 'tipo_conservacion': '', 'nombre_conservacion': '',
            'urgencia': 'SIN_BACHE', 'valor_urgencia': 0.0,
            'recomendacion': 'NINGUNA', 'valor_rec': 0.0,
            'membresias_diam': {}, 'membresias_prof': {},
            'base_normativa': 'RD N° 08-2014-MTC/14 / RD N° 05-2016-MTC/14',
        }
        return {
            'metricas':            metricas,
            'diametro_cm':         0.0,
            'profundidad_mm':      0.0,
            'resultado_mtc':       resultado_vacio,
            'tiene_bache':         False,
            'mascara_pred':        mascara_pred,
            'mascara_gt':          mascara_gt,
            'tiene_ground_truth':  tiene_gt_valido,
            'tiene_profundidad_real': False,
            'fuente_profundidad':  None,
            'imagen_rgb':          img_rgb_512,
            'nombre_archivo':      nombre_archivo,
        }

    # ── PASO 9: DIÁMETRO — siempre real, del contorno predicho ────────────────
    print("[09] Estimación de diámetro (Tabla 4-8 MTC)...")
    diametro_cm = paso10b_estimar_diametro(contorno_principal)
    if diametro_manual:
        print(f"     · Diámetro estimado (SegFormer): {diametro_cm:.2f} cm")
        print(f"     · Diámetro medido manualmente:   {diametro_manual:.2f} cm  (referencia)")

    # ── PASO 10: PROFUNDIDAD — solo con datos reales (.npy o medición manual) ─
    profundidad_mm = None
    fuente_profundidad = None

    if ruta_depth_npy is not None and os.path.isfile(ruta_depth_npy):
        print("[10] Profundidad real — nube de puntos 3D (RealSense D435i)...")
        try:
            # Nube COMPLETA del frame (sin máscara) para que RANSAC encuentre
            # el plano real del pavimento circundante.
            nube_completa = paso8_nube_puntos_desde_realsense(ruta_depth_npy)
            if len(nube_completa) < 50:
                raise ValueError(f"muy pocos puntos válidos en el frame ({len(nube_completa)})")

            normal, inliers, _alineados = paso9_ransac(nube_completa)
            z_pavimento = float(np.median(nube_completa[inliers, 2]))
            print(f"     Plano pavimento (RANSAC): Z={z_pavimento:.4f} m "
                  f"| Inliers: {inliers.sum()}/{len(nube_completa)}")

            # Nube SOLO del bache (usando la máscara real predicha) para
            # medir qué tan lejos está su fondo respecto al pavimento.
            nube_bache = paso8_nube_puntos_desde_realsense(
                ruta_depth_npy, mascara_bache=mascara_pred)
            if len(nube_bache) < 10:
                raise ValueError(f"muy pocos puntos dentro del bache ({len(nube_bache)})")

            z_fondo = float(np.percentile(nube_bache[:, 2], 90))
            profundidad_mm = max(0.0, (z_fondo - z_pavimento) * 1000.0)
            fuente_profundidad = 'realsense'
            print(f"     Fondo del bache (P90 Z): {z_fondo:.4f} m")
            print(f"     Profundidad real:        {profundidad_mm:.2f} mm")
        except Exception as e_npy:
            print(f"     ⚠ No se pudo calcular profundidad real desde el .npy: {e_npy}")
            profundidad_mm = None

    if profundidad_mm is None and profundidad_manual:
        # Medición manual real del usuario (ej. con cinta métrica en campo)
        # — es un dato real ingresado por la persona, no un invento del programa.
        profundidad_mm = float(profundidad_manual)
        fuente_profundidad = 'manual'
        print(f"[10] Profundidad medida manualmente por el usuario: {profundidad_mm:.2f} mm")

    if profundidad_mm is None:
        print("[10] Sin datos reales de profundidad (no hay .npy de la RealSense")
        print("     ni medición manual). No se puede calcular gravedad ni técnica")
        print("     MTC — solo se reporta el diámetro detectado.")

    print(f"     Diámetro:    {diametro_cm:.2f} cm")
    print(f"     Profundidad: {profundidad_mm:.2f} mm" if profundidad_mm is not None
          else f"     Profundidad: NO DISPONIBLE")

    # ── PASO 11: Evaluación MTC (solo si hay profundidad real) ───────────────
    if profundidad_mm is None:
        print(f"[11] Evaluación MTC omitida — falta profundidad real.")
        resultado_fuzzy = {
            'diametro_cm': diametro_cm, 'profundidad_mm': None,
            'gravedad': 'SIN_PROFUNDIDAD',
            'descripcion_gravedad': (
                'Bache detectado y diámetro estimado, pero no se puede '
                'clasificar gravedad ni técnica MTC sin datos reales de '
                'profundidad (conecta la RealSense D435i o ingresa la '
                'medición manual).'),
            'tecnica': 'NINGUNA', 'nombre_tecnica': 'Requiere datos de profundidad',
            'seccion_mtc': '', 'tipo_conservacion': '', 'nombre_conservacion': '',
            'urgencia': 'SIN_PROFUNDIDAD', 'valor_urgencia': 0.0,
            'recomendacion': 'NINGUNA', 'valor_rec': 0.0,
            'membresias_diam': {}, 'membresias_prof': {},
            'base_normativa': 'RD N° 08-2014-MTC/14 / RD N° 05-2016-MTC/14',
        }
        gravedad, tecnica, tipo_cons, seccion = 'SIN_PROFUNDIDAD', 'N/A', '', ''
    else:
        print(f"[11] Evaluación según Manual MTC RD N° 08-2014-MTC/14...")
        resultado_fuzzy = paso11_logica_difusa(diametro_cm, profundidad_mm)

        gravedad  = resultado_fuzzy.get('gravedad', '')
        tecnica   = resultado_fuzzy.get('nombre_tecnica',
                    resultado_fuzzy.get('tecnica', ''))
        seccion   = resultado_fuzzy.get('seccion_mtc', 'Sec.410/415 Cap.400')
        tipo_cons = resultado_fuzzy.get('nombre_conservacion', '')
        capitulo  = resultado_fuzzy.get('capitulo_mtc', 'Cap.400 MTC')

        print(f"     Falla:               Deterioro/Falla 7 Baches (Huecos) — Tabla 4-8 MTC")
        print(f"     Diámetro:            {diametro_cm:.2f} cm = {diametro_cm/100:.3f} m")
        print(f"     Profundidad:         {profundidad_mm:.2f} mm  (fuente: {fuente_profundidad})")
        print(f"     Nivel gravedad:      {gravedad}")
        print(f"     Tipo conservación:   {tipo_cons}  ({capitulo})")
        print(f"     Técnica MTC:         {tecnica}")
        print(f"     Sección MTC:         {seccion}")
        print(f"     Referencia:          RD N° 08-2014-MTC/14 / RD N° 05-2016-MTC/14")

        if _MTC_MODULO_OK:
            try:
                _mtc_guardar_json(resultado_fuzzy)
            except Exception:
                pass

    print(f"\n{'═'*60}")
    print(f"  PAVESCAN — UNSCH · Evaluación de Baches")
    print(f"  Universidad Nacional San Cristóbal de Huamanga")
    print(f"{'─'*60}")
    print(f"  Falla 7: Baches (Huecos) — Tabla 4-8 MTC")
    print(f"{'─'*60}")
    print(f"  Diámetro:            {diametro_cm:.2f} cm  ({diametro_cm/100:.3f} m)")
    print(f"  Profundidad:         "
          f"{profundidad_mm:.2f} mm" if profundidad_mm is not None else "  Profundidad:         NO DISPONIBLE")
    print(f"  Gravedad MTC:        {gravedad}")
    if tipo_cons:
        print(f"  Tipo conservación:   {tipo_cons}")
    print(f"  Técnica MTC:         {tecnica}")
    print(f"  Referencia:          Tabla 4-8, Cap.4 MTC | {seccion}")
    print(f"  Norma:               RD N° 08-2014-MTC/14 / RD N° 05-2016-MTC/14")
    print(f"{'═'*60}\n")

    return {
        'metricas':               metricas,
        'diametro_cm':            diametro_cm,
        'profundidad_mm':         profundidad_mm if profundidad_mm is not None else 0.0,
        'resultado_mtc':          resultado_fuzzy,
        'tiene_bache':            tiene_bache,
        'mascara_pred':           mascara_pred,
        'mascara_gt':             mascara_gt,
        'tiene_ground_truth':     tiene_gt_valido,
        'tiene_profundidad_real': profundidad_mm is not None,
        'imagen_rgb':             img_rgb_512,
        'nombre_archivo':         nombre_archivo,
        'fuente_profundidad':     fuente_profundidad,
    }



def evaluar_dataset_completo(split="test"):
    """
    Evalúa todas las imágenes del split especificado y calcula
    métricas globales acumuladas.

    Parámetro:
        split → 'train', 'valid' o 'test'

    Valor de retorno:
        dict con métricas globales y resultados por imagen
    """

    pares = _listar_pares(split)
    print(f"\n{'═'*60}")
    print(f"  PAVESCAN — Evaluación Dataset Completo (split='{split}')")
    print(f"  Total imágenes: {len(pares)}")
    print(f"{'═'*60}")

    TP_total = FP_total = FN_total = TN_total = 0
    resultados_por_imagen = []

    for i, (ruta_img, ruta_mask) in enumerate(pares, 1):
        nombre = os.path.basename(ruta_img)
        img_rgb, mascara_gt = cargar_imagen_dataset(ruta_img, ruta_mask)

        if mascara_gt is None:
            continue

        img_gris     = paso1_escala_de_grises(img_rgb)
        img_512      = paso2_redimensionar(img_gris)
        img_rgb_512  = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA) \
                       if img_rgb.shape[:2] != (512, 512) else img_rgb
        mascara_pred = paso6_segformer_predecir(img_512, imagen_rgb_512=img_rgb_512)
        m            = paso7_calcular_metricas(mascara_pred, mascara_gt)

        TP_total += m['TP']
        FP_total += m['FP']
        FN_total += m['FN']
        TN_total += m['TN']

        tiene_bache = (mascara_gt > 0).any()
        estado = "BACHE" if tiene_bache else "negativo"
        print(f"  [{i:3d}/{len(pares)}] {nombre[:45]:<45} "
              f"IoU={m['iou']:5.1f}%  ({estado})")
        resultados_por_imagen.append({'archivo': nombre, **m,
                                      'tiene_bache': tiene_bache})

    eps = 1e-8
    recall_g    = TP_total / (TP_total + FN_total + eps) * 100
    precision_g = TP_total / (TP_total + FP_total + eps) * 100
    f1_g        = 2 * precision_g * recall_g / (precision_g + recall_g + eps)
    iou_g       = TP_total / (TP_total + FP_total + FN_total + eps) * 100

    print(f"\n{'─'*60}")
    print(f"  MÉTRICAS GLOBALES (acumulado sobre {len(pares)} imágenes)")
    print(f"{'─'*60}")
    print(f"  Recall:    {recall_g:6.2f}%")
    print(f"  Precisión: {precision_g:6.2f}%")
    print(f"  F1 Score:  {f1_g:6.2f}%")
    print(f"  IoU:       {iou_g:6.2f}%")
    print(f"{'═'*60}\n")

    return {
        'recall':      round(recall_g, 2),
        'precision':   round(precision_g, 2),
        'f1':          round(f1_g, 2),
        'iou':         round(iou_g, 2),
        'por_imagen':  resultados_por_imagen,
        'n_imagenes':  len(pares),
    }


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    import sys
    if "--evaluar" in sys.argv:
        evaluar_dataset_completo("test")
    else:
        # Ejecutar pipeline con imagen del split test
        resultados = ejecutar_pipeline(split="test")

        # Prueba con casos de referencia MTC — Tabla 4-8
        print("\nPRUEBA CON CASOS DE REFERENCIA MTC (Tabla 4-8):")
        print("─" * 55)

        casos_mtc = [
            (12,  25.0, "Gravedad 1 superficial  (D<0.20m, P<50mm)"),
            (12,  65.0, "Gravedad 1 profundo     (D<0.20m, P>50mm)"),
            (35,  30.0, "Gravedad 2 superficial  (D=0.35m, P<50mm)"),
            (35,  75.0, "Gravedad 2 profundo     (D=0.35m, P>50mm)"),
            (60,  30.0, "Gravedad 3 superficial  (D>0.50m, P<50mm)"),
            (60, 100.0, "Gravedad 3 profundo     (D>0.50m, P>50mm)"),
        ]

        for diam, prof, desc in casos_mtc:
            r = paso11_logica_difusa(diam, prof)
            gravedad = r.get('gravedad', r.get('urgencia', ''))
            tecnica  = r.get('tecnica',  r.get('recomendacion', ''))
            print(f"  {desc:<42} →  {gravedad} / {tecnica}")

        print("\n✅ Evaluación MTC completada.")
        print("   Referencia: Manual de Carreteras MTC RD N° 08-2014-MTC/14")
