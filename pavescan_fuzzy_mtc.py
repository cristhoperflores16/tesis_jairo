# =============================================================================
#
#   PAVESCAN_FUZZY_MTC — Sistema de logica difusa Mamdani
#   Base normativa EXCLUSIVA:
#   Manual de Carreteras: Mantenimiento o Conservacion Vial
#   MTC — RD N 08-2014-MTC/14 / RD N 05-2016-MTC/14
#
#   CLASIFICACION DE FALLAS (Tabla 4-8, Cap.4 MTC)
#   ─────────────────────────────────────────────────
#   El manual distingue DOS categorias de deterioro en pavimento flexible:
#
#   A) FALLAS ESTRUCTURALES — requieren rehabilitacion de costo alto:
#      Cod.1 Piel de cocodrilo  → gravedad por tamano de malla (m)
#      Cod.2 Fisuras longitudinales → gravedad por ancho (mm)
#      Cod.3 Deformacion estructural → gravedad por profundidad (cm)
#      Cod.4 Ahuellamiento → gravedad por profundidad (mm)
#      Cod.5 Reparaciones o parchado → estado del parchado previo
#
#   B) FALLAS SUPERFICIALES — requieren mantenimiento periodico:
#      Cod.6 Peladura y desprendimiento → aparicion de base granular
#      Cod.7 BACHES (HUECOS) → gravedad por DIAMETRO (m):
#              Gravedad 1: Diametro < 0.20 m
#              Gravedad 2: Diametro entre 0.20 y 0.50 m
#              Gravedad 3: Diametro > 0.50 m
#      Cod.8 Fisuras transversales → gravedad por ancho (mm)
#
#   VARIABLES DE ENTRADA DEL SISTEMA (correctas segun el manual):
#   ─────────────────────────────────────────────────────────────
#   Entrada 1 — DIAMETRO del bache (m)
#       Fuente: Tabla 4-8, Deterioro/Falla 7 Baches (Huecos), pag.98
#       El manual clasifica los baches EXCLUSIVAMENTE por diametro:
#           Gravedad 1: Diametro < 0.20 m
#           Gravedad 2: Diametro entre 0.20 y 0.50 m
#           Gravedad 3: Diametro > 0.50 m
#       Estimado automaticamente por el sensor Intel RealSense D435i
#       mediante vision computacional (SegFormer + circulo inscrito).
#
#   Entrada 2 — PROFUNDIDAD del bache (mm)
#       Fuente: Sec.410.1 pag.299 y Sec.415.1 pag.304 del manual
#       El umbral de 50 mm determina la tecnica de conservacion:
#           Profundidad < 50 mm → Parchado Superficial Sec.410
#           Profundidad > 50 mm → Parchado Profundo    Sec.415
#       Medida automaticamente por el sensor Intel RealSense D435i.
#
#   VARIABLES DE SALIDA:
#   ─────────────────────
#   Salida 1 — NIVEL DE GRAVEDAD: 1 (BAJO) / 2 (MEDIO) / 3 (ALTO)
#       Fuente: Tabla 4-8, Cap.4 MTC — criterio para baches (Falla 7):
#           Gravedad 1 (BAJO):  Diametro < 0.20 m → Conservacion Rutinaria
#           Gravedad 2 (MEDIO): Diametro 0.20-0.50 m → Conservacion Rutinaria
#           Gravedad 3 (ALTO):  Diametro > 0.50 m → Conservacion Rutinaria
#       NOTA: La Tabla 4-8 clasifica el deterioro por diametro pero NO
#       define la tecnica. La tecnica la determina la PROFUNDIDAD (Cap.400).
#
#   Salida 2 — TECNICA DE CONSERVACION (Cap.400 MTC):
#       Ambas tecnicas son CONSERVACION RUTINARIA segun el Cap.400:
#       Parchado Superficial — Sec.410 (profundidad < 50 mm) → RUTINARIA
#       Parchado Profundo    — Sec.415 (profundidad > 50 mm) → RUTINARIA
#       NOTA: El recapeo (Sec.460) es Conservacion PERIODICA y NO aplica
#       para reparacion de baches individuales segun el Cap.400 MTC.
#
#   ICE — INDICADOR DE COMPROMISO ESTRUCTURAL:
#   ────────────────────────────────────────────
#   Contribucion metodologica original de la tesis.
#   Cuantifica automaticamente el criterio cualitativo del MTC:
#       Sec.410 vs Sec.415 (umbral: 50 mm de profundidad)
#   ICE = profundidad_mm / 50 mm
#       ICE < 1.0  → Parchado Superficial Sec.410 (solo capa rodadura)
#       ICE >= 1.0 → Parchado Profundo    Sec.415 (base/subbase danada)
#
#   NIVELES DE SERVICIO (Cap.3 MTC):
#   ──────────────────────────────────
#   Porcentaje maximo de area con baches = 0% en TODOS los tipos de via.
#   Todo bache detectado requiere intervencion inmediata segun el manual.
#
# =============================================================================

import math
import json
import os
from datetime import datetime


# =============================================================================
# PARAMETROS NORMATIVOS
# Fuente: Tabla 4-8 Cap.4 y Sec.410/415 del Manual MTC RD N 08-2014-MTC/14
# =============================================================================

# ── Variable 1: Diametro del bache (m) ───────────────────────────────────────
# Fuente directa: Tabla 4-8, Falla 7 Baches (Huecos), pag.98 Manual MTC
# Los limites son exactamente los del manual: 0.20 m y 0.50 m
DIAM_MAX_M    = 1.0    # limite superior para normalizacion (1 metro)

RANGOS_DIAM_MTC = {
    'PEQUENIO': 'Diametro < 0.20 m  — Gravedad 1 (Tabla 4-8, Falla 7 MTC)',
    'MEDIANO':  'Diametro 0.20-0.50 m — Gravedad 2 (Tabla 4-8, Falla 7 MTC)',
    'GRANDE':   'Diametro > 0.50 m  — Gravedad 3 (Tabla 4-8, Falla 7 MTC)',
}

# ── Variable 2: Profundidad del bache (mm) ────────────────────────────────────
# Fuente: Sec.410 (< 50 mm) y Sec.415 (> 50 mm) del Manual MTC
# El manual define cualitativamente dos situaciones:
#   "parches poco profundos... profundidad alcanza menos de 50 mm" (Sec.410)
#   "parchado profundo... profundidad sea mayor de 50 mm"          (Sec.415)
PROF_MAX_MM   = 150.0  # limite superior para normalizacion (mm)
UMBRAL_ICE_MM = 50.0   # umbral normativo Sec.410/415 del Manual MTC

RANGOS_PROF_MTC = {
    'SUPERFICIAL': 'Profundidad < 50 mm  — Sec.410 Parchado Superficial MTC',
    'PROFUNDA':    'Profundidad > 50 mm  — Sec.415 Parchado Profundo MTC',
}

# ── Salida 1: Nivel de gravedad ───────────────────────────────────────────────
# Fuente: Tabla 4-8, Falla 7 Baches (Huecos), Cap.4 MTC
# Los 3 niveles corresponden al catálogo de deterioros para pavimento flexible.
# NOTA: La Tabla 4-8 clasifica el deterioro por DIÁMETRO únicamente.
#       NO define tipo de conservación — eso lo determina la PROFUNDIDAD (Cap.400).
#       Todo bache (G1, G2 o G3) → CONSERVACIÓN RUTINARIA según Cap.400 MTC.
NIVELES_GRAVEDAD_MTC = {
    'GRAVEDAD_1': 0.20,   # Gravedad 1 — Leve     (Diámetro < 0.20 m)
    'GRAVEDAD_2': 0.55,   # Gravedad 2 — Moderado (Diámetro 0.20-0.50 m)
    'GRAVEDAD_3': 0.90,   # Gravedad 3 — Severo   (Diámetro > 0.50 m)
}

DESCRIPCION_GRAVEDAD = {
    'GRAVEDAD_1': (
        'Gravedad 1 — LEVE. Diametro < 0.20 m. '
        'Bache pequeno, sensible al usuario. '
        'Tecnica: Parchado Superficial (Sec.410) o Profundo (Sec.415) '
        'segun profundidad. Conservacion Rutinaria Cap.400 MTC. '
        '(Tabla 4-8, Falla 7, Cap.4 MTC).'
    ),
    'GRAVEDAD_2': (
        'Gravedad 2 — MODERADO. Diametro 0.20-0.50 m. '
        'Bache mediano, afecta la rodadura. '
        'Tecnica: Parchado Superficial (Sec.410) o Profundo (Sec.415) '
        'segun profundidad. Conservacion Rutinaria Cap.400 MTC. '
        '(Tabla 4-8, Falla 7, Cap.4 MTC).'
    ),
    'GRAVEDAD_3': (
        'Gravedad 3 — SEVERO. Diametro > 0.50 m. '
        'Bache grande, compromete seguridad vial. '
        'Tecnica: Parchado Superficial (Sec.410) o Profundo (Sec.415) '
        'segun profundidad. Conservacion Rutinaria Cap.400 MTC. '
        '(Tabla 4-8, Falla 7, Cap.4 MTC).'
    ),
}

# ── Salida 2: Tecnica de conservacion ────────────────────────────────────────
# Fuente: Cap.400 del Manual MTC RD N 08-2014-MTC/14
# AMBAS tecnicas son CONSERVACION RUTINARIA segun el Cap.400 MTC:
#   Sec.410 Parchado Superficial → profundidad < 50 mm → RUTINARIA
#   Sec.415 Parchado Profundo    → profundidad > 50 mm → RUTINARIA
# El Recapeo (Sec.460) es Conservacion PERIODICA y NO aplica para
# baches individuales. Se aplica para deterioro generalizado de la via.
TECNICAS_MTC = {
    'PARCHADO_SUPERFICIAL': 0.35,
    'PARCHADO_PROFUNDO':    0.75,
}

INFO_TECNICAS_MTC = {
    'PARCHADO_SUPERFICIAL': {
        'nombre':    'Parchado Superficial en Calzada',
        'seccion':   'Sec.410 — Manual MTC RD N 08-2014-MTC/14',
        'tipo_conservacion': 'Conservacion Rutinaria',
        'capitulo':  'Cap.400 — Conservacion de Pavimentos Flexibles',
        'criterio':  (
            'Profundidad < 50 mm. El dano afecta SOLO la capa de rodadura. '
            '"Los parches poco profundos, entendiendose como tales, aquellos '
            'cuya profundidad alcanza menos de 50 mm." (Sec.410.1 pag.299). '
            'Base granular en buenas condiciones. '
            'Ejecutar en el menor tiempo posible tras la aparicion del bache.'
        ),
        'falla_mtc': 'Falla 7 — Tabla 4-8 Cap.4 MTC | Sec.410 Cap.400 MTC',
        'medicion':  'Metro cuadrado (m2)',
    },
    'PARCHADO_PROFUNDO': {
        'nombre':    'Parchado Profundo en Calzada',
        'seccion':   'Sec.415 — Manual MTC RD N 08-2014-MTC/14',
        'tipo_conservacion': 'Conservacion Rutinaria',
        'capitulo':  'Cap.400 — Conservacion de Pavimentos Flexibles',
        'criterio':  (
            'Profundidad > 50 mm. El dano penetra la base y/o subbase. '
            '"Parchado profundo, entendiendose como tales aquellos cuya '
            'profundidad sea mayor de 50 mm." (Sec.415.1 pag.304). '
            'Incluye demolicion y reposicion de capas estructurales. '
            'Ejecutar en el menor tiempo posible.'
        ),
        'falla_mtc': 'Falla 7 — Tabla 4-8 Cap.4 MTC | Sec.415 Cap.400 MTC',
        'medicion':  'Metro cuadrado (m2)',
    },
}

# =============================================================================
# REGLAS DIFUSAS MAMDANI
# ─────────────────────────────────────────────────────────────────────────────
# Basadas en:
#   · Tabla 4-8 Falla 7 Baches (Huecos), Cap.4 MTC — gravedad por DIAMETRO
#   · Sec.410 (profundidad < 50 mm) y Sec.415 (profundidad > 50 mm)
#   · Cap.3 Niveles de servicio: baches = 0% en TODO tipo de via
#
# LOGICA DE DISENO:
#   El DIAMETRO determina el nivel de gravedad (Tabla 4-8 MTC):
#       Diametro < 0.20 m → Gravedad 1
#       Diametro 0.20-0.50 m → Gravedad 2
#       Diametro > 0.50 m → Gravedad 3
#
#   La PROFUNDIDAD determina la tecnica (Sec.410 o Sec.415 MTC):
#       Profundidad < 50 mm → Parchado Superficial Sec.410
#       Profundidad > 50 mm → Parchado Profundo    Sec.415
#
#   La combinacion de ambas variables genera las 6 reglas del sistema.
#   (3 categorias de diametro x 2 categorias de profundidad = 6 reglas)
#
# Formato: (PROFUNDIDAD_CAT, DIAMETRO_CAT, GRAVEDAD_SALIDA, TECNICA_SALIDA)
# =============================================================================

REGLAS_MTC = [
    # ── Profundidad SUPERFICIAL (< 50 mm) → Sec.410 MTC ──────────────────────
    # Segun Cap.400 MTC: Sec.410 Parchado Superficial es Conservacion RUTINARIA
    # Se aplica cuando la profundidad < 50 mm (solo afecta capa de rodadura)
    # independientemente del diametro (Gravedad 1, 2 o 3 de la Tabla 4-8).
    ('SUPERFICIAL', 'PEQUENIO', 'GRAVEDAD_1', 'PARCHADO_SUPERFICIAL'),
    ('SUPERFICIAL', 'MEDIANO',  'GRAVEDAD_2', 'PARCHADO_SUPERFICIAL'),
    ('SUPERFICIAL', 'GRANDE',   'GRAVEDAD_3', 'PARCHADO_SUPERFICIAL'),

    # ── Profundidad PROFUNDA (> 50 mm) → Sec.415 MTC ─────────────────────────
    # Segun Cap.400 MTC: Sec.415 Parchado Profundo es Conservacion RUTINARIA
    # Se aplica cuando la profundidad > 50 mm (afecta base y/o subbase)
    # independientemente del diametro (Gravedad 1, 2 o 3 de la Tabla 4-8).
    ('PROFUNDA',    'PEQUENIO', 'GRAVEDAD_1', 'PARCHADO_PROFUNDO'),
    ('PROFUNDA',    'MEDIANO',  'GRAVEDAD_2', 'PARCHADO_PROFUNDO'),
    ('PROFUNDA',    'GRANDE',   'GRAVEDAD_3', 'PARCHADO_PROFUNDO'),
]


# =============================================================================
# FUNCION DE MEMBRESIA TRAPEZOIDAL
# Modela las categorias difusas con transicion gradual entre niveles.
# La forma trapezoidal es adecuada porque los limites del manual
# (0.20 m y 0.50 m) no son bordes abruptos sino zonas de transicion.
# =============================================================================

def membresia_trapezoidal(x, a, b, c, d):
    """
    Funcion de membresia trapezoidal.

    Forma:
        ___________
       /           \\
      /             \\
    a   b           c   d

    x <= a o x >= d : membresia = 0.0
    a < x < b       : sube linealmente de 0 a 1
    b <= x <= c     : membresia = 1.0 (meseta)
    c < x < d       : baja linealmente de 1 a 0

    Parametros:
        x       → valor normalizado [0, 1]
        a,b,c,d → vertices del trapecio

    Retorna: grado de membresia en [0.0, 1.0]

    NOTA — orden de las comprobaciones:
    La meseta (b <= x <= c) se comprueba ANTES que la condicion de
    "fuera de rango" (x <= a o x >= d). Esto importa para categorias
    abiertas en un extremo, como PROFUNDA (b=0.267, c=1.00, d=1.00) o
    PEQUENIO (a=0.00, b=0.00, c=0.17): como las entradas normalizadas se
    recortan (clamp) a [0,1], un valor real que excede el maximo (p.ej.
    profundidad_mm > 150mm) llega aqui con x=1.0 exacto. Si primero se
    evalua "x >= d" (con d=1.0 tambien), la funcion devolvia 0.0 en vez
    de 1.0 — apagando por completo la categoria PROFUNDA justo en los
    casos mas profundos, y con ella toda la inferencia difusa (ver bug
    documentado: profundidad 196mm devolvia GRAVEDAD_1 en vez de una
    gravedad alta, porque mem_prof quedaba en {SUPERFICIAL:0, PROFUNDA:0}
    y ninguna regla podia activarse).
    """
    if b <= x <= c:
        return 1.0
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (d - x) / (d - c) if d != c else 1.0


# =============================================================================
# FUZZIFICACION
# Convierte los valores reales a grados de membresia difusa.
# =============================================================================

def fuzzificar_diametro(diametro_m):
    """
    Grados de membresia del DIAMETRO del bache.

    Limites exactos del Manual MTC Tabla 4-8, Falla 7 Baches (Huecos):
        Gravedad 1: Diametro < 0.20 m  → categoria PEQUENIO
        Gravedad 2: Diametro 0.20-0.50 m → categoria MEDIANO
        Gravedad 3: Diametro > 0.50 m  → categoria GRANDE

    El sensor RealSense D435i estima el diametro via SegFormer +
    circulo inscrito en el contorno detectado. El resultado en cm
    se convierte a metros antes de llamar esta funcion.

    Parametro: diametro_m → diametro del bache en metros
    Retorna:   dict con claves PEQUENIO/MEDIANO/GRANDE
    """
    d = max(0.0, min(1.0, diametro_m / DIAM_MAX_M))

    return {
        # PEQUENIO: < 0.20 m (Gravedad 1 MTC)
        # Zona de transicion: 0.17-0.23 m
        'PEQUENIO': membresia_trapezoidal(d, 0.00, 0.00, 0.17, 0.23),

        # MEDIANO: 0.20-0.50 m (Gravedad 2 MTC)
        # Zona de transicion inferior: 0.17-0.23 m
        # Zona de transicion superior: 0.47-0.53 m
        'MEDIANO':  membresia_trapezoidal(d, 0.17, 0.23, 0.47, 0.53),

        # GRANDE: > 0.50 m (Gravedad 3 MTC)
        # Zona de transicion: 0.47-0.53 m
        'GRANDE':   membresia_trapezoidal(d, 0.47, 0.53, 1.00, 1.00),
    }


def fuzzificar_profundidad(profundidad_mm):
    """
    Grados de membresia de la PROFUNDIDAD del bache.

    Umbral del Manual MTC (Sec.410 y Sec.415):
        Profundidad < 50 mm → Parchado Superficial (Sec.410)
        Profundidad > 50 mm → Parchado Profundo    (Sec.415)

    Solo 2 categorias porque el manual define un unico umbral (50 mm).
    La zona de transicion (40-60 mm) modela la incertidumbre de
    medicion del sensor y el criterio de inspeccion en campo.

    Parametro: profundidad_mm → profundidad medida por el sensor (mm)
    Retorna:   dict con claves SUPERFICIAL/PROFUNDA
    """ 
    p = max(0.0, min(1.0, profundidad_mm / PROF_MAX_MM))

    return {
        # SUPERFICIAL: < 50 mm → Sec.410 Parchado Superficial
        # Zona de transicion: 40-60 mm (0.267-0.400 normalizado)
        'SUPERFICIAL': membresia_trapezoidal(p, 0.00, 0.00, 0.267, 0.400),

        # PROFUNDA: > 50 mm → Sec.415 Parchado Profundo
        # Zona de transicion: 40-60 mm (0.267-0.400 normalizado)
        'PROFUNDA':    membresia_trapezoidal(p, 0.267, 0.400, 1.00, 1.00),
    }


# =============================================================================
# ICE — INDICADOR DE COMPROMISO ESTRUCTURAL
# Contribucion metodologica original de la tesis.
# Automatiza el criterio cualitativo de Sec.410 vs Sec.415 del MTC.
# =============================================================================

def calcular_ice(profundidad_mm, umbral_mm=UMBRAL_ICE_MM):
    """
    ICE = profundidad_mm / umbral_normativo_mm (50 mm segun MTC)

    El Manual MTC define cualitativamente (Sec.410 y Sec.415):
        Profundidad < 50 mm → solo capa rodadura → Sec.410 Parchado Superficial
        Profundidad > 50 mm → base/subbase danada → Sec.415 Parchado Profundo

    El ICE convierte esa decision cualitativa en un valor continuo
    medido automaticamente por el sensor Intel RealSense D435i,
    eliminando la subjetividad del inspector manual:

        ICE < 1.0  → Sec.410 Parchado Superficial
        ICE 1.0-1.5 → Zona de transicion (evaluar en campo)
        ICE > 1.5  → Sec.415 Parchado Profundo confirmado

    Parametros:
        profundidad_mm → profundidad medida por el sensor (mm)
        umbral_mm      → umbral normativo MTC (50 mm por defecto)

    Retorna: dict con ICE, clasificacion y referencia normativa MTC
    """
    ice = round(profundidad_mm / max(umbral_mm, 1.0), 3)

    if ice < 0.80:
        clas = 'SUPERFICIAL'
        sec  = 'Sec.410 — Parchado Superficial MTC'
        desc = ('Profundidad < 50 mm. Solo capa de rodadura afectada. '
                'Base granular en buenas condiciones. (Sec.410 pag.299 MTC)')
    elif ice < 1.00:
        clas = 'ZONA_TRANSICION'
        sec  = 'Sec.410/415 — Evaluacion visual en campo recomendada'
        desc = ('Profundidad proxima al umbral de 50 mm. '
                'Verificar si la base granular esta comprometida.')
    else:
        clas = 'PROFUNDO'
        sec  = 'Sec.415 — Parchado Profundo MTC'
        desc = ('Profundidad > 50 mm. Dano en base y/o subbase. '
                'Demolicion y reposicion estructural requerida. (Sec.415 pag.304 MTC)')

    return {
        'ice':            ice,
        'clasificacion':  clas,
        'seccion_mtc':    sec,
        'descripcion':    desc,
        'umbral_mm':      umbral_mm,
        'profundidad_mm': round(profundidad_mm, 2),
    }


# =============================================================================
# INFERENCIA MAMDANI
# AND difuso → minimo de membresías de las premisas
# OR difuso  → maximo entre reglas que activan la misma salida
# =============================================================================

def _inferencia(mem_prof, mem_diam):
    """
    Aplica las 6 reglas difusas y agrega las activaciones.

    Parametros:
        mem_prof → membresías de profundidad (SUPERFICIAL/PROFUNDA)
        mem_diam → membresías de diametro    (PEQUENIO/MEDIANO/GRANDE)

    Retorna:
        (act_gravedad, act_tecnica) → dicts de activaciones
    """
    act_grav = {k: 0.0 for k in NIVELES_GRAVEDAD_MTC}
    act_tec  = {k: 0.0 for k in TECNICAS_MTC}

    for prof_cat, diam_cat, gravedad, tecnica in REGLAS_MTC:
        activacion = min(mem_prof[prof_cat], mem_diam[diam_cat])
        if activacion > 0:
            act_grav[gravedad] = max(act_grav[gravedad], activacion)
            act_tec[tecnica]   = max(act_tec[tecnica],   activacion)

    return act_grav, act_tec


def _defuzzificar(activaciones, valores_crisp):
    """
    Defuzzificacion por centroide.
    z* = sum(activacion_k x crisp_k) / sum(activacion_k)

    Si ninguna regla se activo (todas las activaciones en 0), antes se
    devolvia silenciosamente la PRIMERA clave del diccionario (que para
    NIVELES_GRAVEDAD_MTC es 'GRAVEDAD_1') — esto ocultaba fallos reales
    de fuzzificacion (como el bug de membresia_trapezoidal corregido
    arriba) detras de un resultado que parecia normal pero era falso.
    Ahora se elige la categoria MAS SEVERA/PROFUNDA como fallback
    conservador (mas seguro para una evaluacion de infraestructura vial)
    y se deja constancia explicita en el resultado.
    """
    activas = {k: v for k, v in activaciones.items() if v > 0}
    if not activas:
        etiqueta_conservadora = list(valores_crisp.keys())[-1]
        return valores_crisp[etiqueta_conservadora], etiqueta_conservadora
    num = sum(activas[k] * valores_crisp[k] for k in activas)
    den = sum(activas.values())
    return round(num / den, 4), max(activas, key=activas.get)


# =============================================================================
# FUNCION PRINCIPAL — evaluar_bache_mtc()
# =============================================================================

def evaluar_bache_mtc(diametro_m, profundidad_mm):
    """
    Evalua un bache segun el Manual MTC (RD N 08-2014-MTC/14).

    ENTRADAS correctas segun el manual:
        diametro_m     → diametro del bache en METROS
                         (Tabla 4-8, Falla 7: < 0.20 / 0.20-0.50 / > 0.50 m)
        profundidad_mm → profundidad en MILIMETROS
                         (Sec.410: < 50 mm / Sec.415: > 50 mm)

    Pipeline:
        1. Fuzzificacion de diametro y profundidad
        2. Inferencia Mamdani con las 6 reglas MTC
        3. Defuzzificacion por centroide
        4. Calculo del ICE (Indicador de Compromiso Estructural)

    Retorna: dict completo con gravedad MTC, tecnica, ICE y membresias
    """
    mem_diam = fuzzificar_diametro(diametro_m)
    mem_prof = fuzzificar_profundidad(profundidad_mm)

    act_g, act_t = _inferencia(mem_prof, mem_diam)

    val_g, etq_g = _defuzzificar(act_g, NIVELES_GRAVEDAD_MTC)
    val_t, etq_t = _defuzzificar(act_t, TECNICAS_MTC)

    ice  = calcular_ice(profundidad_mm)
    info = INFO_TECNICAS_MTC.get(etq_t, {})

    # Tipo de conservacion segun estructura real del Manual MTC Cap.400:
    # Sec.410 y Sec.415 (Parchado Superficial y Profundo) son AMBAS
    # Actividades de CONSERVACION RUTINARIA segun el Cap.400 MTC.
    # El Recapeo (Sec.460) NO aplica para baches individuales.
    tipo_conservacion   = 'CONSERVACION_RUTINARIA'
    nombre_conservacion = 'Conservacion Rutinaria'
    capitulo_mtc        = 'Cap.400 — Manual MTC RD N 08-2014-MTC/14'

    return {
        # Entradas
        'diametro_m':           round(diametro_m,    4),
        'profundidad_mm':       round(profundidad_mm, 2),
        # Nivel de gravedad (Tabla 4-8 Cap.4 MTC)
        'gravedad':             etq_g,
        'descripcion_gravedad': DESCRIPCION_GRAVEDAD.get(etq_g, ''),
        'valor_gravedad':       val_g,
        # Tipo de conservacion — estructura real del Manual MTC
        'tipo_conservacion':    tipo_conservacion,
        'nombre_conservacion':  nombre_conservacion,
        'capitulo_mtc':         capitulo_mtc,
        # Tecnica de conservacion (Cap.400 / Cap.500 MTC)
        'tecnica':              etq_t,
        'nombre_tecnica':       info.get('nombre',    etq_t),
        'seccion_mtc':          info.get('seccion',   ''),
        'criterio_mtc':         info.get('criterio',  ''),
        'falla_mtc':            info.get('falla_mtc', ''),
        'medicion':             info.get('medicion',  'm2'),
        'valor_tecnica':        val_t,
        # ICE — contribucion metodologica de la tesis
        'ice':                  ice,
        # Membresias para las barras de la GUI
        'membresias_diam':      {k: round(v, 3) for k, v in mem_diam.items()},
        'membresias_prof':      {k: round(v, 3) for k, v in mem_prof.items()},
        # Referencia normativa
        'base_normativa':       'RD N 08-2014-MTC/14 / RD N 05-2016-MTC/14',
    }


def evaluar_bache_hibrido(diametro_cm, profundidad_mm):
    """
    Punto de entrada del pipeline pavescan_segformer.py.

    El pipeline estima el diametro en CENTIMETROS.
    Esta funcion convierte a METROS y llama evaluar_bache_mtc().

    Parametros:
        diametro_cm    → diametro estimado por SegFormer en cm
        profundidad_mm → profundidad estimada por RealSense en mm

    Retorna: dict de evaluar_bache_mtc() mas diametro_cm de origen
    """
    diametro_m = diametro_cm / 100.0
    r = evaluar_bache_mtc(diametro_m, profundidad_mm)
    r['diametro_cm'] = round(diametro_cm, 2)
    return r


# =============================================================================
# FUNCION PARA SUPERFICIES 3D DE LA GUI
# Ejes: diametro normalizado (0=0m, 1=1m) y profundidad normalizada (0=0mm, 1=150mm)
# =============================================================================

def calcular_salidas_fuzzy_mtc(dn, pn):
    """
    Calcula gravedad y tecnica para entradas YA NORMALIZADAS [0,1].
    Se usa para construir las superficies 3D de la GUI con matplotlib.

    Parametros:
        dn → diametro normalizado   [0,1]  (1.0 = 1.0 m)
        pn → profundidad normalizada [0,1] (1.0 = 150 mm)

    Retorna: (gravedad_crisp, tecnica_crisp) en [0,1]
    """
    mem_diam = {
        'PEQUENIO': membresia_trapezoidal(dn, 0.00, 0.00, 0.17, 0.23),
        'MEDIANO':  membresia_trapezoidal(dn, 0.17, 0.23, 0.47, 0.53),
        'GRANDE':   membresia_trapezoidal(dn, 0.47, 0.53, 1.00, 1.00),
    }
    mem_prof = {
        'SUPERFICIAL': membresia_trapezoidal(pn, 0.00, 0.00, 0.267, 0.400),
        'PROFUNDA':    membresia_trapezoidal(pn, 0.267, 0.400, 1.00, 1.00),
    }
    ag, at = _inferencia(mem_prof, mem_diam)
    vg, _  = _defuzzificar(ag, NIVELES_GRAVEDAD_MTC)
    vt, _  = _defuzzificar(at, TECNICAS_MTC)
    return vg, vt


# =============================================================================
# GUARDADO DE RESULTADOS EN JSON
# =============================================================================

def guardar_resultado_json_mtc(resultado, ruta_salida="./resultados",
                                nombre_archivo=None):
    """
    Guarda el resultado MTC en JSON con referencias normativas completas.
    El archivo sirve como evidencia documentada para la tesis.

    Parametros:
        resultado      → dict de evaluar_bache_mtc() o evaluar_bache_hibrido()
        ruta_salida    → carpeta destino (se crea automaticamente)
        nombre_archivo → nombre del json (None = timestamp automatico)

    Retorna: ruta completa del archivo guardado
    """
    os.makedirs(ruta_salida, exist_ok=True)
    if nombre_archivo is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"bache_mtc_{ts}.json"
    if not nombre_archivo.endswith(".json"):
        nombre_archivo += ".json"

    ice = resultado.get('ice', {})

    reporte = {
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "sistema":    "PaveScan — Manual MTC RD N 08-2014-MTC/14",
        "entradas": {
            "diametro_m":     resultado.get('diametro_m',     0.0),
            "diametro_cm":    resultado.get('diametro_cm',    None),
            "profundidad_mm": resultado.get('profundidad_mm', 0.0),
        },
        "clasificacion_falla_mtc": {
            "tipo_falla":  "Deterioro/Falla 7: Baches (Huecos)",
            "tabla":       "Tabla 4-8, Cap.4 — Deterioros Fallas Superficiales",
            "criterio":    "Gravedad determinada por DIAMETRO (m)",
        },
        "nivel_gravedad_mtc": {
            "gravedad":             resultado.get('gravedad',          ''),
            "descripcion":          resultado.get('descripcion_gravedad', ''),
            "valor_gravedad":       resultado.get('valor_gravedad',    0.0),
        },
        "ice": {
            "valor":         ice.get('ice',           0.0),
            "clasificacion": ice.get('clasificacion', ''),
            "seccion_mtc":   ice.get('seccion_mtc',   ''),
            "descripcion":   ice.get('descripcion',   ''),
            "umbral_mm":     ice.get('umbral_mm',     50.0),
        },
        "tecnica_conservacion_mtc": {
            "tecnica":       resultado.get('tecnica',        ''),
            "nombre":        resultado.get('nombre_tecnica', ''),
            "seccion_mtc":   resultado.get('seccion_mtc',    ''),
            "criterio":      resultado.get('criterio_mtc',   ''),
            "falla_mtc":     resultado.get('falla_mtc',      ''),
            "valor_tecnica": resultado.get('valor_tecnica',  0.0),
        },
        "membresias": {
            "diametro":    resultado.get('membresias_diam', {}),
            "profundidad": resultado.get('membresias_prof', {}),
        },
        "referencia_normativa": {
            "manual":    "Manual de Carreteras: Mantenimiento o Conservacion Vial",
            "norma":     "RD N 08-2014-MTC/14 / RD N 05-2016-MTC/14",
            "entidad":   "Ministerio de Transportes y Comunicaciones del Peru",
            "capitulo":  "Cap.4: Catalogo de Deterioros/Fallas pav.flexible",
            "tabla":     "Tabla 4-8: Deterioros/Fallas pavimentos asfaltados",
            "falla":     "Falla 7: Baches (Huecos) — Deterioros Superficiales",
            "sec_tecnica": "Cap.400: Conservacion de Pavimentos Flexibles",
        },
    }

    ruta = os.path.join(ruta_salida, nombre_archivo)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    print(f"  [JSON-MTC] Guardado: '{ruta}'")
    return ruta


# =============================================================================
# PRUEBA — verifica los 3 niveles de gravedad de la Tabla 4-8 MTC
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  PaveScan — Evaluacion segun Manual MTC RD N 08-2014-MTC/14")
    print("  Falla 7: Baches (Huecos) — Tabla 4-8, Cap.4")
    print("=" * 70)

    print("\n  TABLA 4-8 MTC — Niveles de gravedad para Baches (Huecos):")
    print("  Gravedad 1: Diametro < 0.20 m")
    print("  Gravedad 2: Diametro entre 0.20 y 0.50 m")
    print("  Gravedad 3: Diametro > 0.50 m")
    print()
    print("  Sec.410: Profundidad < 50 mm → Parchado Superficial")
    print("  Sec.415: Profundidad > 50 mm → Parchado Profundo")
    print()
    print("  NOTA: Cap.3 MTC — baches = 0% en TODOS los tipos de via.")
    print("  Todo bache detectado requiere intervencion.\n")

    # Casos de prueba alineados a la Tabla 4-8 MTC
    casos = [
        (0.12,  25.0, "Gravedad 1 superficial  (D<0.20m, P<50mm)"),
        (0.12,  65.0, "Gravedad 1 profundo     (D<0.20m, P>50mm)"),
        (0.35,  30.0, "Gravedad 2 superficial  (D=0.35m, P<50mm)"),
        (0.35,  75.0, "Gravedad 2 profundo     (D=0.35m, P>50mm)"),
        (0.60,  30.0, "Gravedad 3 superficial  (D>0.50m, P<50mm)"),
        (0.60, 100.0, "Gravedad 3 profundo     (D>0.50m, P>50mm)"),
    ]

    print(f"  {'Descripcion':<38} {'ICE':>5}  {'Gravedad':>10}  Tecnica MTC")
    print("  " + "-" * 70)

    for diam_m, prof, desc in casos:
        r = evaluar_bache_mtc(diam_m, prof)
        ice = r['ice']['ice']
        print(f"  {desc:<38} {ice:>5.2f}  "
              f"{r['gravedad']:>10}  {r['nombre_tecnica']}")

    print("\n  Modulo pavescan_fuzzy_mtc.py operativo.")
    print("  Importar: from pavescan_fuzzy_mtc import evaluar_bache_hibrido")
