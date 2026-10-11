# =============================================================================
#
#   PAVESCAN GUI — Interfaz gráfica para evaluación de baches
#   Base normativa: Manual de Carreteras MTC RD N°08-2014-MTC/14
#   ─────────────────────────────────────────────────────────────
#   Este archivo es el LANZADOR VISUAL del pipeline pavescan_comentado.py.
#   Al ejecutarlo, abre una ventana profesional que:
#
#     1. Permite configurar todos los parámetros del sensor y del bache
#     2. Ejecuta pavescan_comentado.ejecutar_pipeline() en un hilo separado
#     3. Muestra en tiempo real el log del pipeline
#     4. Genera las MISMAS 3 visualizaciones 3D que pavescan_3d.html:
#           (a) Superficie difusa de Urgencia
#           (b) Superficie difusa de Recomendación
#           (c) Geometría 3D del bache predicho
#     5. Muestra métricas, membresías difusas e información detallada
#
#   DEPENDENCIAS:
#       pip install numpy opencv-python matplotlib
#
#   EJECUCIÓN:
#       python pavescan_gui.py
#       (pavescan_comentado.py debe estar en la misma carpeta)
#
# =============================================================================

# ── Importaciones estándar ────────────────────────────────────────────────────
import sys
import os
import math
import threading
import importlib
import traceback

# ── NumPy: operaciones matriciales para las superficies 3D ──────────────────
import numpy as np

# ── Tkinter: GUI nativa de Python (incluida por defecto) ────────────────────
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Matplotlib: motor de renderizado 3D (Axes3D + Surface) ──────────────────
import matplotlib
matplotlib.use('TkAgg')                          # Backend compatible con tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 — activa proyección 3D
from matplotlib import cm as colormap_lib
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap

# ── OpenCV: procesamiento de imágenes (importado vía el pipeline) ────────────
import cv2

# ── PIL: para cargar el logo institucional UNSCH ──────────────────────────────
from PIL import Image as _PIL_Image, ImageTk as _PIL_ImageTk

# ── Módulo de evaluación normativa MTC ───────────────────────────────────────
# Base normativa EXCLUSIVA: Manual MTC RD N°08-2014-MTC/14
# Cap.3: Niveles servicio | Cap.4: Deterioros | Sec.410/415/460: Técnicas
try:
    from pavescan_fuzzy_mtc import (
        evaluar_bache_hibrido      as _mtc_evaluar,
        calcular_salidas_fuzzy_mtc as _mtc_superficie,
    )
    _MTC_DISPONIBLE = True
except ImportError:
    _MTC_DISPONIBLE = False
    print("[AVISO] pavescan_fuzzy_mtc.py no encontrado.")


# =============================================================================
# PALETA DE COLORES Y ESTILOS DE LA INTERFAZ
# Replica el esquema visual de pavescan_3d.html (dark technical)
# =============================================================================

# Colores principales
COLOR_BG        = '#06080c'   # fondo principal (casi negro)
COLOR_BG1       = '#0c0f16'   # fondo de paneles
COLOR_BG2       = '#12161f'   # fondo de cards
COLOR_BG3       = '#181d28'   # fondo de headers
COLOR_BORDER    = '#1e2535'   # bordes sutiles
COLOR_CYAN      = '#00e5ff'   # acento principal
COLOR_LIME      = '#b8ff57'   # acento secundario
COLOR_AMBER     = '#ffb347'   # acento cálido
COLOR_RED       = '#ff5252'   # urgencia alta
COLOR_VIOLET    = '#c084fc'   # recomendación
COLOR_TEXT_0    = '#eef2ff'   # texto principal
COLOR_TEXT_1    = '#8899bb'   # texto secundario
COLOR_TEXT_2    = '#4a5a72'   # texto terciario

# Fuentes
FONT_TITLE  = ('Courier New', 13, 'bold')
FONT_MONO   = ('Courier New', 10)
FONT_MONO_S = ('Courier New', 9)
FONT_LABEL  = ('Courier New', 9, 'bold')
FONT_BIG    = ('Courier New', 22, 'bold')
FONT_MED    = ('Courier New', 14, 'bold')

# Mapa de colores para las superficies 3D (replica palette 'cool' del HTML)
CMAP_SURFACE  = colormap_lib.get_cmap('cool')     # superficie difusa
CMAP_POTHOLE  = colormap_lib.get_cmap('plasma')   # geometría del bache

# ── Umbrales de categoría MTC (idénticos a los usados en el label flotante) ──
#   Gravedad:  <0.25 → G1 (leve) | 0.25-0.65 → G2 (moderado) | >0.65 → G3 (severo)
#   Técnica:   <0.55 → Sec.410 Parchado Superficial | >0.55 → Sec.415 Parchado Profundo
UMBRAL_GRAVEDAD = [0.0, 0.25, 0.65, 1.0]
UMBRAL_TECNICA  = [0.0, 0.55, 1.0]

# Paletas discretas — reutilizan los acentos ya definidos en la interfaz,
# así el color de la superficie 3D coincide siempre con el color del label
# flotante y con las barras de membresía de la pestaña Resultados.
CMAP_GRAVEDAD_DISCRETO = ListedColormap([COLOR_LIME, COLOR_AMBER, COLOR_RED])
NORM_GRAVEDAD_DISCRETO = BoundaryNorm(UMBRAL_GRAVEDAD, CMAP_GRAVEDAD_DISCRETO.N)

CMAP_TECNICA_DISCRETO = ListedColormap([COLOR_CYAN, COLOR_VIOLET])
NORM_TECNICA_DISCRETO = BoundaryNorm(UMBRAL_TECNICA, CMAP_TECNICA_DISCRETO.N)


# =============================================================================
# LÓGICA DIFUSA EMBEBIDA
# Replica exactamente las funciones de pavescan_comentado.py
# para calcular las superficies 3D sin depender del módulo externo.
# =============================================================================

def _trap(x, a, b, c, d):
    """Función de membresía trapezoidal (ver pavescan_segformer_mtc.py paso 11).

    La meseta (b<=x<=c) se comprueba antes que "fuera de rango" (x<=a o
    x>=d) — necesario para categorías abiertas (b==a o c==d), donde un
    valor recortado (clamp) exactamente a 0.0 o 1.0 debe caer en la
    meseta, no en cero. Ver la nota completa en membresia_trapezoidal()
    de pavescan_fuzzy_mtc.py.
    """
    if b <= x <= c: return 1.0
    if x <= a or x >= d: return 0.0
    if x < b: return (x-a)/(b-a) if b!=a else 1.0
    return (d-x)/(d-c) if d!=c else 1.0


def calcular_salidas_fuzzy(dn, pn):
    """
    Evalúa el sistema Mamdani para un par (diámetro_normalizado, prof_normalizada).
    Devuelve (urgencia_crisp, recomendacion_crisp) en [0, 1].
    Esta función se vectoriza para construir las superficies 3D.
    """
    mD = {
        'MINIMAL': _trap(dn,0,.00,.08,.12), 'LOW':  _trap(dn,.08,.12,.18,.22),
        'MEDIUM':  _trap(dn,.18,.22,.43,.47),'MODERATE':_trap(dn,.43,.47,.73,.77),
        'HIGH':    _trap(dn,.73,.77,1.,1.)
    }
    mP = {
        'MINIMAL': _trap(pn,0,.00,.15,.20), 'LOW':  _trap(pn,.15,.20,.30,.36),
        'MEDIUM':  _trap(pn,.30,.36,.47,.53),'MODERATE':_trap(pn,.47,.53,.63,.70),
        'HIGH':    _trap(pn,.63,.70,1.,1.)
    }

    # Activaciones de salida
    uA = {'NO_DAMAGE':0.,'LOW':0.,'MEDIUM':0.,'HIGH':0.}
    rA = {'ROUTINE_MAINTENANCE':0.,'CRACK_SEALING':0.,'CRACK_FELLING':0.,
          'PATCHING':0.,'SEAL_IRRIGATION':0.,'RECONSTRUCTION':0.}

    # Reglas difusas — Manual MTC Tabla 4-8 adaptadas a 6 categorías MTC
    REGLAS = [
        ('MINIMAL','MINIMAL','NO_DAMAGE','ROUTINE_MAINTENANCE'),
        ('MINIMAL','LOW','NO_DAMAGE','ROUTINE_MAINTENANCE'),
        ('MINIMAL','MEDIUM','NO_DAMAGE','ROUTINE_MAINTENANCE'),
        ('MINIMAL','MODERATE','LOW','ROUTINE_MAINTENANCE'),
        ('MINIMAL','HIGH','LOW','ROUTINE_MAINTENANCE'),
        ('LOW','MINIMAL','LOW','CRACK_SEALING'),
        ('LOW','LOW','LOW','CRACK_SEALING'),
        ('LOW','MEDIUM','LOW','CRACK_FELLING'),
        ('LOW','MODERATE','MEDIUM','CRACK_FELLING'),
        ('LOW','HIGH','MEDIUM','PATCHING'),
        ('MEDIUM','MINIMAL','MEDIUM','CRACK_SEALING'),
        ('MEDIUM','LOW','MEDIUM','PATCHING'),
        ('MEDIUM','MEDIUM','MEDIUM','PATCHING'),
        ('MEDIUM','MODERATE','HIGH','PATCHING'),
        ('MEDIUM','HIGH','HIGH','PATCHING'),
        ('MODERATE','MINIMAL','MEDIUM','CRACK_SEALING'),
        ('MODERATE','LOW','MEDIUM','PATCHING'),
        ('MODERATE','MEDIUM','HIGH','PATCHING'),
        ('MODERATE','MODERATE','HIGH','SEAL_IRRIGATION'),
        ('MODERATE','HIGH','HIGH','SEAL_IRRIGATION'),
        ('HIGH','MINIMAL','HIGH','RECONSTRUCTION'),
        ('HIGH','LOW','HIGH','RECONSTRUCTION'),
        ('HIGH','MEDIUM','HIGH','RECONSTRUCTION'),
        ('HIGH','MODERATE','HIGH','RECONSTRUCTION'),
        ('HIGH','HIGH','HIGH','RECONSTRUCTION'),
    ]

    for pc, dc, ug, rc in REGLAS:
        a = min(mP.get(pc, 0.), mD.get(dc, 0.))
        if a > 0:
            uA[ug] = max(uA[ug], a)
            rA[rc] = max(rA[rc], a)

    # Defuzzificación por centroide (método Mamdani)
    vU = {'NO_DAMAGE':0.,'LOW':.3,'MEDIUM':.6,'HIGH':.9}
    vR = {'ROUTINE_MAINTENANCE':.05,'CRACK_SEALING':.2,'CRACK_FELLING':.35,
          'PATCHING':.55,'SEAL_IRRIGATION':.75,'RECONSTRUCTION':.95}

    def defuzz(acts, vals):
        n = sum(acts[k]*vals[k] for k in acts if acts[k]>0)
        d = sum(acts[k]         for k in acts if acts[k]>0)
        return n/d if d>1e-8 else 0.

    return defuzz(uA, vU), defuzz(rA, vR)


def construir_superficies_fuzzy(N=20):
    """
    Pre-calcula las dos superficies difusas en una malla N×N.
    Ejes en valores reales MTC (Opción B):
      X = Diámetro en cm   (0-100 cm  — Tabla 4-8 MTC)
      Y = Profundidad en mm (0-150 mm  — Sec.410/415 MTC)
      Z = Gravedad/Técnica normalizada [0, 1]
    """
    diam_arr = np.linspace(0, 100, N)   # 0-100 cm (Tabla 4-8 MTC)
    prof_arr = np.linspace(0, 150, N)   # 0-150 mm (Sec.410/415 MTC)
    # indexing='ij': DIAM varía en filas (eje X), PROF en columnas (eje Y)
    # Garantiza que ambos gráficos tengan el mismo origen (0,0) en la misma esquina
    DIAM, PROF = np.meshgrid(diam_arr, prof_arr, indexing='ij')

    Z_urgencia = np.zeros((N, N))
    Z_rec      = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            dn = DIAM[i,j] / 100.0
            pn = PROF[i,j] / 150.0
            u, r = calcular_salidas_fuzzy(dn, pn)
            Z_urgencia[i,j] = u
            Z_rec[i,j]      = r

    return DIAM, PROF, Z_urgencia, Z_rec


def construir_geometria_bache(diametro_cm, profundidad_mm, nr=40, na=60):
    """
    Genera la malla 3D de la geometría del bache.
    Forma: paraboloide circular (el perfil real de un bache).

    Replica _buildGeometry() de la clase Pothole3D en pavescan_3d.html.

    Parámetros:
        diametro_cm   → diámetro del bache en centímetros
        profundidad_mm → profundidad en milímetros
        nr, na        → resolución radial y angular

    Retorna:
        X, Y, Z → arrays 2D para plot_surface()  (en metros)
        C       → array 2D de color por profundidad [0,1]
    """
    # Profundidad normalizada relativa al radio
    radio_m   = (diametro_cm / 100) / 2
    prof_norm = (profundidad_mm / 1000) / max(radio_m, 0.001)

    # Coordenadas polares del bache (r de 0 a 1, ángulo de 0 a 2π)
    r_arr = np.linspace(0, 1, nr)
    a_arr = np.linspace(0, 2*math.pi, na, endpoint=False)
    R, A  = np.meshgrid(r_arr, a_arr)

    # Perfil paraboloide: z = -prof × (1 - r²)^0.9
    # A r=0 (centro): z = -profundidad máxima
    # A r=1 (borde):  z = 0 (nivel del pavimento)
    irregularidad = 0.04 * np.sin(5*A + 1) * np.sin(3*A + 2) * (1 - R)
    Z_bache = -prof_norm * np.power(np.clip(1 - R, 0, 1), 1.8) + irregularidad*(1-R)

    # Convertir a metros centrado en (0,0)
    X = R * np.cos(A) * radio_m
    Y = R * np.sin(A) * radio_m
    Z = Z_bache * radio_m   # escalar a metros reales

    # Color: 0 en el borde (rojo) → 1 en el centro (amarillo/blanco)
    C = 1 - R

    # ── Pavimento exterior (anillos de r=1 a r=2.5 × radio) ─────────────────
    r_pav  = np.linspace(1, 2.5, 20)
    a_pav  = np.linspace(0, 2*math.pi, na, endpoint=False)
    Rp, Ap = np.meshgrid(r_pav, a_pav)

    Xp = Rp * np.cos(Ap) * radio_m
    Yp = Rp * np.sin(Ap) * radio_m
    Zp = np.zeros_like(Xp) + np.random.normal(0, 0.0005, Xp.shape)
    Cp = np.full_like(Xp, -0.1)   # color gris para el pavimento

    return X, Y, Z, C, Xp, Yp, Zp, Cp


# =============================================================================
# CLASE PRINCIPAL DE LA INTERFAZ GRÁFICA
# =============================================================================

class PaveScanGUI:
    """
    Ventana principal de PaveScan GUI.

    Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │  TOPBAR — título, badge, estado                             │
    ├──────────┬──────────────────────────────────────────────────┤
    │ PANEL    │  NOTEBOOK con 3 pestañas:                        │
    │ IZQUIERDO│   [Pipeline] [Visualizaciones 3D] [Resultados]   │
    │          │                                                   │
    │ • Params │  Pestaña Pipeline:                               │
    │ • Sliders│    • Log en tiempo real del pipeline             │
    │ • Fuzzy  │    • Imágenes del proceso (gris, segmentación)   │
    │ • Botones│                                                   │
    │          │  Pestaña 3D:                                     │
    │          │    • (a) Superficie Urgencia                     │
    │          │    • (b) Superficie Recomendación                │
    │          │    • (c) Geometría 3D del bache                  │
    │          │                                                   │
    │          │  Pestaña Resultados:                             │
    │          │    • Métricas, membresías, tabla de casos        │
    └──────────┴──────────────────────────────────────────────────┘
    """

    def __init__(self, root):
        self.root = root
        self.root.title("PAVESCAN — UNSCH · Evaluación de Baches con IA")
        self.root.configure(bg=COLOR_BG)
        self.root.geometry("1400x860")
        self.root.minsize(1100, 700)

        # Mantener la decoración nativa del SO (sin overrideredirect)
        # para que la ventana aparezca en la barra de tareas y sea
        # recuperable al cambiar de aplicación.

        # Variables de estado
        self.pipeline_activo   = False
        self.resultados        = None       # última ejecución del pipeline
        self.superficies_cache = None       # cache de superficies fuzzy
        self.hilo_pipeline     = None

        # Control de debounce: evita redibujar en cada tick del slider.
        # _debounce_id guarda el ID del after() pendiente; se cancela si
        # llega un nuevo evento antes de que expire el delay.
        # _hilo_superficie_activo es un flag que el hilo de cálculo revisa
        # para saber si debe abortar (si ya hay uno más nuevo en camino).
        self._debounce_id            = None
        self._hilo_superficie        = None
        self._hilo_superficie_activo = False

        # Variables de control (tkinter)
        self.var_diametro   = tk.DoubleVar(value=40.0)
        self.var_profundidad= tk.DoubleVar(value=35.0)
        self.var_altura     = tk.DoubleVar(value=1.25)
        self.var_fov        = tk.DoubleVar(value=87.0)
        self.var_resolucion = tk.IntVar(value=20)  # resolución reducida para mayor fluidez
        self.var_paleta     = tk.StringVar(value='cool')
        self.var_imagen     = tk.StringVar(value='')
        self.var_depth_npy  = tk.StringVar(value='')   # .npy REAL de la RealSense
        self.var_modo_fuzzy = tk.StringVar(value='MTC')    # MTC=Manual MTC (siempre activo)

        self._construir_ui()
        self._aplicar_estilos_ttk()

        # Pre-calcular superficies al iniciar (en hilo para no bloquear UI)
        threading.Thread(target=self._precalcular_superficies, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN DE LA INTERFAZ
    # ─────────────────────────────────────────────────────────────────────────

    def _construir_ui(self):
        """Construye todos los widgets de la ventana."""
        self._construir_topbar()
        self._construir_layout_principal()

    def _construir_topbar(self):
        """
        Barra superior institucional UNSCH.
        Incluye logo, datos académicos, estado del pipeline y controles.
        """
        import os as _os

        # ── Franja guinda institucional UNSCH ─────────────────────────────────
        tk.Frame(self.root, bg='#6B0F1A', height=4).pack(fill='x', side='top')

        bar = tk.Frame(self.root, bg=COLOR_BG3, height=68)
        bar.pack(fill='x', side='top')
        bar.pack_propagate(False)

        # ── Botones de control (derecha — siempre primero en tkinter) ─────────
        fr_ctrl = tk.Frame(bar, bg=COLOR_BG3)
        fr_ctrl.pack(side='right', padx=(0, 6), pady=0)

        def _btn(parent, txt, fg, hover, cmd):
            b = tk.Button(parent, text=txt, bg=COLOR_BG3, fg=fg,
                          activebackground=hover, activeforeground=fg,
                          font=('Courier New', 13, 'bold'),
                          relief='flat', bd=0, padx=12, pady=0,
                          cursor='hand2', command=cmd,
                          highlightthickness=0)
            b.pack(side='left')
            b.bind('<Enter>', lambda e: b.config(bg=hover))
            b.bind('<Leave>', lambda e: b.config(bg=COLOR_BG3))
            return b

        _btn(fr_ctrl, '✕', '#ff5252', '#4a1010', self.root.destroy)
        self.btn_max = _btn(fr_ctrl, '□', COLOR_TEXT_1, COLOR_BG2,
                            self._toggle_maximizar)
        _btn(fr_ctrl, '−', COLOR_TEXT_1, COLOR_BG2,
             lambda: self.root.iconify())

        tk.Frame(bar, bg=COLOR_BORDER, width=1).pack(
            side='right', fill='y', pady=8, padx=4)

        # ── Estado del pipeline (derecha) ─────────────────────────────────────
        self.lbl_estado = tk.Label(bar, text='● LISTO', bg=COLOR_BG3,
                                   fg=COLOR_LIME, font=FONT_MONO)
        self.lbl_estado.pack(side='right', padx=10)

        tk.Frame(bar, bg=COLOR_BORDER, width=1).pack(
            side='right', fill='y', pady=8, padx=2)

        # ── Info normativa + académica (derecha) ──────────────────────────────
        fr_info = tk.Frame(bar, bg=COLOR_BG3)
        fr_info.pack(side='right', padx=12)
        tk.Label(fr_info, text='Manual MTC · RD N° 08-2014-MTC/14',
                 bg=COLOR_BG3, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)).pack(anchor='e')
        tk.Label(fr_info, text='Fac. Ingeniería de Minas · Escuela Civil',
                 bg=COLOR_BG3, fg='#C0392B',
                 font=('Courier New', 7, 'bold')).pack(anchor='e')
        tk.Label(fr_info, text='Huamanga · Ayacucho · Perú · 2025',
                 bg=COLOR_BG3, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)).pack(anchor='e')

        # ── Logo UNSCH (izquierda) ────────────────────────────────────────────
        ruta_logo = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), 'unsch.jpg')
        try:
            logo_pil = _PIL_Image.open(ruta_logo)
            logo_pil = logo_pil.resize((50, 66), _PIL_Image.LANCZOS)
            self._logo_tk = _PIL_ImageTk.PhotoImage(logo_pil)
            tk.Label(bar, image=self._logo_tk, bg=COLOR_BG3,
                     padx=8).pack(side='left', pady=1)
        except Exception:
            tk.Label(bar, text='⚑', bg=COLOR_BG3,
                     fg='#8B1C2E', font=('Arial', 22),
                     padx=10).pack(side='left', pady=4)

        # ── Títulos del sistema (izquierda) ───────────────────────────────────
        titulos = tk.Frame(bar, bg=COLOR_BG3)
        titulos.pack(side='left', pady=4)

        # Nombre del sistema
        tk.Label(titulos, text='PAVESCAN AI',
                 bg=COLOR_BG3, fg=COLOR_TEXT_0,
                 font=('Courier New', 15, 'bold')).pack(anchor='w')

        # Subtítulo técnico
        tk.Label(titulos,
                 text='Sistema de Detección y Evaluación de Baches · '
                      'SegFormer B2 + Lógica Difusa Mamdani + RealSense D435i',
                 bg=COLOR_BG3, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)).pack(anchor='w')

        # Identificación institucional UNSCH
        tk.Label(titulos,
                 text='Universidad Nacional San Cristóbal de Huamanga · '
                      'Tesis de Grado · Ingeniería Civil',
                 bg=COLOR_BG3, fg='#C0392B',
                 font=('Courier New', 7, 'bold')).pack(anchor='w')

        # ── Separador inferior guinda ─────────────────────────────────────────
        tk.Frame(self.root, bg='#6B0F1A', height=2).pack(fill='x')

    def _toggle_maximizar(self):
        """Alterna entre ventana normal y maximizada usando el gestor nativo."""
        try:
            # Windows / Linux con Tk >= 8.5
            estado = self.root.state()
            if estado == 'zoomed':
                self.root.state('normal')
                self.btn_max.config(text='□')
            else:
                self.root.state('zoomed')
                self.btn_max.config(text='❐')
        except Exception:
            # Fallback universal: geometry manual
            if getattr(self, '_maximizado', False):
                geo = getattr(self, '_geo_restaurar', '1400x860')
                self.root.geometry(geo)
                self.btn_max.config(text='□')
                self._maximizado = False
            else:
                self._geo_restaurar = self.root.geometry()
                sw = self.root.winfo_screenwidth()
                sh = self.root.winfo_screenheight()
                self.root.geometry(f'{sw}x{sh}+0+0')
                self.btn_max.config(text='❐')
                self._maximizado = True

    def _construir_layout_principal(self):
        """Divide la ventana en panel izquierdo (controles) y área principal."""
        contenedor = tk.Frame(self.root, bg=COLOR_BG)
        contenedor.pack(fill='both', expand=True)

        # ── Panel izquierdo: controles ────────────────────────────────────────
        self.panel_izq = tk.Frame(contenedor, bg=COLOR_BG1, width=290)
        self.panel_izq.pack(side='left', fill='y')
        self.panel_izq.pack_propagate(False)

        # Separador vertical
        tk.Frame(contenedor, bg=COLOR_BORDER, width=1).pack(side='left', fill='y')

        # ── Área principal: notebook con pestañas ─────────────────────────────
        self.area_principal = tk.Frame(contenedor, bg=COLOR_BG)
        self.area_principal.pack(side='left', fill='both', expand=True)

        self._construir_panel_izquierdo()
        self._construir_notebook()

    def _construir_panel_izquierdo(self):
        """Panel de controles: parámetros, sliders, botones."""
        panel = self.panel_izq

        # Scroll interno
        canvas_scroll = tk.Canvas(panel, bg=COLOR_BG1, highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient='vertical',
                                  command=canvas_scroll.yview)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        canvas_scroll.pack(fill='both', expand=True)

        frame_scroll = tk.Frame(canvas_scroll, bg=COLOR_BG1)
        canvas_scroll.create_window((0, 0), window=frame_scroll, anchor='nw')
        frame_scroll.bind('<Configure>',
            lambda e: canvas_scroll.configure(
                scrollregion=canvas_scroll.bbox('all')))
        canvas_scroll.bind('<MouseWheel>',
            lambda e: canvas_scroll.yview_scroll(-1*(e.delta//120), 'units'))

        p = frame_scroll   # alias corto

        # ── SECCIÓN: Imagen de entrada ────────────────────────────────────────
        self._seccion(p, '📸  IMAGEN DE ENTRADA')

        frame_img = tk.Frame(p, bg=COLOR_BG1)
        frame_img.pack(fill='x', padx=12, pady=(0, 8))

        self.lbl_imagen = tk.Label(
            frame_img, text='[imagen sintética]',
            bg=COLOR_BG2, fg=COLOR_TEXT_2, font=FONT_MONO_S,
            relief='flat', padx=8, pady=6, anchor='w', width=22)
        self.lbl_imagen.pack(fill='x', pady=(0, 4))

        tk.Button(frame_img, text='Seleccionar imagen...',
                  command=self._seleccionar_imagen,
                  bg=COLOR_BG3, fg=COLOR_TEXT_1, font=FONT_MONO_S,
                  relief='flat', padx=6, pady=4,
                  activebackground=COLOR_BORDER, cursor='hand2'
                  ).pack(fill='x')

        # ── SECCIÓN: Sensor ───────────────────────────────────────────────────
        self._seccion(p, '📡  SENSOR REALSENSE D435i')

        self._slider_help(p, 'Altura de captura',
                     self.var_altura, 1.0, 1.5, '{:.2f} m',
                     color=COLOR_CYAN,
                     ayuda='Distancia vertical del sensor al suelo.\n'
                           'Usado en la conversión píxel→metros.\n'
                           'Valor típico de montaje en vehículo: 1.0–1.5 m.\n'
                           'Cámara: Intel RealSense D435i.')

        self._slider_help(p, 'FOV Horizontal',
                     self.var_fov, 60, 110, '{:.0f}°',
                     color=COLOR_CYAN,
                     ayuda='Campo de visión horizontal de la cámara.\n'
                           'Especificación real del RealSense D435i: 87°.\n'
                           'Determina cuántos cm cubre cada píxel.\n'
                           'Cambia solo si usas otra cámara.')

        # ── SECCIÓN: Bache manual ─────────────────────────────────────────────
        self._seccion(p, '📐  MEDIDAS DEL BACHE')

        self._slider_entry(p, 'Diámetro',
                     self.var_diametro, 5, 100, '{:.1f}', 'cm',
                     color=COLOR_AMBER,
                     command=self._actualizar_3d_en_vivo,
                     ayuda='Diámetro estimado del bache en centímetros.\n'
                           'En modo automático lo calcula el pipeline\n'
                           'con visión 3D. En modo manual,\n'
                           'ingresa el valor medido en campo.\n'
                           'MTC Tabla 4-8: G1<20cm, G2=20-50cm, G3>50cm.')

        self._slider_entry(p, 'Profundidad',
                     self.var_profundidad, 5, 75, '{:.1f}', 'mm',
                     color=COLOR_AMBER,
                     command=self._actualizar_3d_en_vivo,
                     ayuda='Profundidad máxima del bache en milímetros.\n'
                           'El pipeline la obtiene con la nube de puntos\n'
                           '3D del sensor RealSense D435i.\n'
                           'MTC Sec.410: <50mm Parchado Superficial.\n'
                           'MTC Sec.415: >50mm Parchado Profundo.')

        # ── SECCIÓN: Visualización ────────────────────────────────────────────
        self._seccion(p, '🎨  VISUALIZACIÓN 3D')

        self._slider_help(p, 'Resolución malla',
                     self.var_resolucion, 10, 50, '{:.0f}',
                     color=COLOR_VIOLET,
                     ayuda='Puntos N de la grilla N×N de las superficies\n'
                           'difusas (a) y (b). Más alto = más suave,\n'
                           'pero más tiempo de cálculo.\n'
                           'Recomendado: 25–35. Máximo útil: 45.')

        # Selector de paleta
        fr_pal = tk.Frame(p, bg=COLOR_BG1)
        fr_pal.pack(fill='x', padx=12, pady=4)
        tk.Label(fr_pal, text='Paleta de color', bg=COLOR_BG1,
                 fg=COLOR_TEXT_2, font=FONT_MONO_S).pack(anchor='w')
        fr_btns = tk.Frame(fr_pal, bg=COLOR_BG1)
        fr_btns.pack(fill='x', pady=4)
        for nombre, cmap in [('COOL','cool'), ('PLASMA','plasma'), ('VIRIDIS','viridis')]:
            tk.Button(fr_btns, text=nombre, bg=COLOR_BG3, fg=COLOR_TEXT_1,
                      font=('Courier New', 8), relief='flat', padx=6, pady=3,
                      command=lambda c=cmap: self._cambiar_paleta(c),
                      cursor='hand2'
                      ).pack(side='left', padx=2)

        # steps_labels vacío (mantenemos compatibilidad con _marcar_paso)
        self.steps_labels = []

        # ── INDICADOR DE NORMATIVA MTC ─────────────────────────────────────────
        fr_norma = tk.Frame(p, bg=COLOR_BG2)
        fr_norma.pack(fill='x', padx=12, pady=(8, 2))
        tk.Label(fr_norma, text='BASE NORMATIVA',
                 bg=COLOR_BG2, fg=COLOR_TEXT_2,
                 font=('Courier New', 8, 'bold')
                 ).pack(anchor='w', padx=10, pady=(6, 2))
        tk.Label(fr_norma,
                 text='  Manual de Carreteras MTC',
                 bg=COLOR_BG2, fg=COLOR_CYAN,
                 font=('Courier New', 8, 'bold')
                 ).pack(anchor='w', padx=10)
        tk.Label(fr_norma,
                 text='  RD N 08-2014-MTC/14',
                 bg=COLOR_BG2, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)
                 ).pack(anchor='w', padx=10)
        tk.Label(fr_norma,
                 text='  Cap.3: Niveles de Servicio',
                 bg=COLOR_BG2, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)
                 ).pack(anchor='w', padx=10)
        tk.Label(fr_norma,
                 text='  Cap.4: Catalogo de Deterioros',
                 bg=COLOR_BG2, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)
                 ).pack(anchor='w', padx=10)
        tk.Label(fr_norma,
                 text='  Sec.410: Parchado Superficial (<50mm)',
                 bg=COLOR_BG2, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)
                 ).pack(anchor='w', padx=10)
        tk.Label(fr_norma,
                 text='  Sec.415: Parchado Profundo (>50mm)',
                 bg=COLOR_BG2, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)
                 ).pack(anchor='w', padx=10, pady=(0,6))
        if not _MTC_DISPONIBLE:
            tk.Label(fr_norma,
                     text='  AVISO: pavescan_fuzzy_mtc.py no encontrado',
                     bg=COLOR_BG2, fg=COLOR_RED,
                     font=('Courier New', 7)).pack(anchor='w', padx=10, pady=(0,4))

        # ── BOTONES PRINCIPALES ───────────────────────────────────────────────
        fr_btn = tk.Frame(p, bg=COLOR_BG1)
        fr_btn.pack(fill='x', padx=12, pady=14)

        self.btn_ejecutar = tk.Button(
            fr_btn, text='▶  EJECUTAR PIPELINE',
            command=self._ejecutar_pipeline,
            bg=COLOR_CYAN, fg='#000000',
            font=('Courier New', 11, 'bold'),
            relief='flat', pady=10, cursor='hand2',
            activebackground='#33eeff')
        self.btn_ejecutar.pack(fill='x', pady=(0, 6))

        tk.Button(
            fr_btn, text='↺  Ejemplo casos MTC (Tabla 4-8)',
            command=self._cargar_ejemplo,
            bg=COLOR_BG3, fg=COLOR_TEXT_1,
            font=FONT_MONO_S, relief='flat', pady=7, cursor='hand2',
            activebackground=COLOR_BORDER
        ).pack(fill='x', pady=(0, 4))

        tk.Button(
            fr_btn, text='🔄  Redibujar gráficas 3D',
            command=self._redibujar_todo,
            bg=COLOR_BG3, fg=COLOR_TEXT_1,
            font=FONT_MONO_S, relief='flat', pady=7, cursor='hand2',
            activebackground=COLOR_BORDER
        ).pack(fill='x')

    def _seccion(self, parent, titulo):
        """Encabezado de sección con línea decorativa."""
        fr = tk.Frame(parent, bg=COLOR_BG1)
        fr.pack(fill='x', padx=10, pady=(14, 6))
        tk.Label(fr, text=titulo, bg=COLOR_BG1, fg=COLOR_TEXT_2,
                 font=('Courier New', 8, 'bold')).pack(anchor='w')
        tk.Frame(parent, bg=COLOR_BORDER, height=1).pack(
            fill='x', padx=10, pady=(0, 4))

    def _slider(self, parent, etiqueta, variable, minv, maxv,
                formato='{:.1f}', color=COLOR_CYAN, command=None):
        """Slider estilizado con etiqueta y valor actual."""
        fr = tk.Frame(parent, bg=COLOR_BG1)
        fr.pack(fill='x', padx=12, pady=3)

        fr_top = tk.Frame(fr, bg=COLOR_BG1)
        fr_top.pack(fill='x')
        tk.Label(fr_top, text=etiqueta, bg=COLOR_BG1,
                 fg=COLOR_TEXT_1, font=FONT_MONO_S).pack(side='left')
        lbl_val = tk.Label(fr_top, text=formato.format(variable.get()),
                           bg=COLOR_BG2, fg=color, font=FONT_MONO_S,
                           padx=6, pady=1)
        lbl_val.pack(side='right')

        def on_change(val):
            lbl_val.config(text=formato.format(float(val)))
            if command:
                command()

        scale = tk.Scale(
            fr, variable=variable, from_=minv, to=maxv,
            orient='horizontal', resolution=(maxv-minv)/200,
            bg=COLOR_BG1, fg=color, troughcolor=COLOR_BG3,
            highlightthickness=0, bd=0, sliderrelief='flat',
            sliderlength=14, width=4, showvalue=False,
            command=on_change)
        scale.pack(fill='x', pady=(2, 0))

    # ── Sistema de tooltips ───────────────────────────────────────────────────

    def _tooltip(self, widget, texto):
        """
        Adjunta un tooltip emergente a cualquier widget.
        Aparece 500 ms después de que el mouse entra y desaparece al salir.
        """
        tip_win = [None]

        def _mostrar(event):
            if tip_win[0]:
                return
            # Posición justo debajo del widget
            x = widget.winfo_rootx() + 4
            y = widget.winfo_rooty() + widget.winfo_height() + 4

            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)   # sin barra de título
            tw.wm_geometry(f'+{x}+{y}')
            tw.configure(bg=COLOR_BORDER)

            # Marco interior con padding
            fr = tk.Frame(tw, bg='#0d1520', bd=0)
            fr.pack(padx=1, pady=1)
            tk.Label(fr, text=texto,
                     bg='#0d1520', fg=COLOR_TEXT_0,
                     font=('Courier New', 8),
                     justify='left', padx=10, pady=7,
                     wraplength=260).pack()
            tip_win[0] = tw

        def _ocultar(event):
            if tip_win[0]:
                tip_win[0].destroy()
                tip_win[0] = None

        widget.bind('<Enter>', lambda e: widget.after(500, lambda: _mostrar(e)
                                         if widget.winfo_exists() else None))
        widget.bind('<Leave>', _ocultar)
        widget.bind('<Button-1>', _ocultar)

    def _slider_help(self, parent, etiqueta, variable, minv, maxv,
                     formato='{:.1f}', color=COLOR_CYAN, command=None,
                     ayuda=''):
        """
        Slider estilizado + botón ? con tooltip de ayuda.
        Versión para parámetros de solo-slider (sensor, resolución).
        """
        fr = tk.Frame(parent, bg=COLOR_BG1)
        fr.pack(fill='x', padx=12, pady=3)

        fr_top = tk.Frame(fr, bg=COLOR_BG1)
        fr_top.pack(fill='x')

        # Botón de ayuda
        btn_help = tk.Label(fr_top, text=' ? ', bg=COLOR_BG3,
                            fg=COLOR_TEXT_2, font=('Courier New', 8, 'bold'),
                            cursor='hand2', padx=2, pady=0,
                            relief='flat')
        btn_help.pack(side='right', padx=(2, 0))
        if ayuda:
            self._tooltip(btn_help, ayuda)

        lbl_val = tk.Label(fr_top,
                           text=formato.format(variable.get()),
                           bg=COLOR_BG2, fg=color, font=FONT_MONO_S,
                           padx=6, pady=1)
        lbl_val.pack(side='right', padx=(0, 4))

        tk.Label(fr_top, text=etiqueta, bg=COLOR_BG1,
                 fg=COLOR_TEXT_1, font=FONT_MONO_S).pack(side='left')

        def on_change(val):
            lbl_val.config(text=formato.format(float(val)))
            if command:
                command()

        scale = tk.Scale(
            fr, variable=variable, from_=minv, to=maxv,
            orient='horizontal', resolution=(maxv - minv) / 200,
            bg=COLOR_BG1, fg=color, troughcolor=COLOR_BG3,
            highlightthickness=0, bd=0, sliderrelief='flat',
            sliderlength=14, width=4, showvalue=False,
            command=on_change)
        scale.pack(fill='x', pady=(2, 0))

    def _slider_entry(self, parent, etiqueta, variable, minv, maxv,
                      fmt_num='{:.1f}', unidad='', color=COLOR_AMBER,
                      command=None, ayuda=''):
        """
        Slider + campo de texto para ingreso manual + botón ?.
        Versión para Diámetro y Profundidad: permite escribir el valor exacto
        además de arrastrar el slider.

        Flujo de actualización:
          • Slider → actualiza Entry y variable
          • Entry  → valida rango, actualiza slider y variable, lanza command
        """
        fr = tk.Frame(parent, bg=COLOR_BG1)
        fr.pack(fill='x', padx=12, pady=4)

        # ── Fila superior: etiqueta + unidad + botón ? ────────────────────────
        fr_top = tk.Frame(fr, bg=COLOR_BG1)
        fr_top.pack(fill='x')

        tk.Label(fr_top, text=etiqueta, bg=COLOR_BG1,
                 fg=COLOR_TEXT_1, font=FONT_MONO_S).pack(side='left')

        btn_help = tk.Label(fr_top, text=' ? ', bg=COLOR_BG3,
                            fg=COLOR_TEXT_2, font=('Courier New', 8, 'bold'),
                            cursor='hand2', padx=2, relief='flat')
        btn_help.pack(side='right', padx=(2, 0))
        if ayuda:
            self._tooltip(btn_help, ayuda)

        tk.Label(fr_top, text=unidad, bg=COLOR_BG1,
                 fg=COLOR_TEXT_2, font=FONT_MONO_S).pack(side='right', padx=(0, 4))

        # ── Fila media: campo Entry de ingreso manual ─────────────────────────
        fr_entry = tk.Frame(fr, bg=COLOR_BG1)
        fr_entry.pack(fill='x', pady=(3, 2))

        # StringVar conectada al Entry (independiente del DoubleVar del slider)
        sv = tk.StringVar(value=fmt_num.format(variable.get()))

        entry = tk.Entry(
            fr_entry, textvariable=sv,
            bg=COLOR_BG2, fg=color,
            insertbackground=color,
            selectbackground=COLOR_BORDER,
            font=('Courier New', 11, 'bold'),
            relief='flat', bd=0,
            highlightthickness=1,
            highlightbackground=COLOR_BORDER,
            highlightcolor=color,
            width=8, justify='center')
        entry.pack(side='left', padx=(0, 6), ipady=4)

        # Indicador de rango
        tk.Label(fr_entry,
                 text=f'({minv}–{maxv} {unidad})',
                 bg=COLOR_BG1, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)).pack(side='left')

        # ── Fila inferior: slider ─────────────────────────────────────────────
        def _desde_slider(val):
            """Slider movido → actualiza Entry."""
            sv.set(fmt_num.format(float(val)))
            if command:
                command()

        scale = tk.Scale(
            fr, variable=variable, from_=minv, to=maxv,
            orient='horizontal', resolution=(maxv - minv) / 200,
            bg=COLOR_BG1, fg=color, troughcolor=COLOR_BG3,
            highlightthickness=0, bd=0, sliderrelief='flat',
            sliderlength=14, width=4, showvalue=False,
            command=_desde_slider)
        scale.pack(fill='x', pady=(0, 2))

        def _aplicar_entry(*_):
            """Entry editado → valida, actualiza slider y dispara command."""
            raw = sv.get().strip().replace(',', '.')
            try:
                val = float(raw)
            except ValueError:
                # Valor no numérico: restaurar al valor actual
                sv.set(fmt_num.format(variable.get()))
                entry.config(highlightbackground='#ff5252')
                entry.after(800, lambda: entry.config(
                    highlightbackground=COLOR_BORDER))
                return

            # Clampear al rango permitido
            val = max(minv, min(maxv, val))
            sv.set(fmt_num.format(val))
            variable.set(val)
            entry.config(highlightbackground=color)
            if command:
                command()

        # Aplicar al presionar Enter o al perder el foco
        entry.bind('<Return>',    _aplicar_entry)
        entry.bind('<FocusOut>',  _aplicar_entry)
        # Resaltar al entrar
        entry.bind('<FocusIn>',
                   lambda e: entry.config(highlightbackground=color))

    def _construir_notebook(self):
        """Crea el notebook (pestañas) con las 3 vistas principales."""
        style = ttk.Style()
        style.configure('Dark.TNotebook', background=COLOR_BG,
                        borderwidth=0)
        style.configure('Dark.TNotebook.Tab',
                        background=COLOR_BG3, foreground=COLOR_TEXT_2,
                        font=('Courier New', 9, 'bold'),
                        padding=[14, 6])
        style.map('Dark.TNotebook.Tab',
                  background=[('selected', COLOR_BG2)],
                  foreground=[('selected', COLOR_CYAN)])

        self.notebook = ttk.Notebook(self.area_principal, style='Dark.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=0, pady=0)

        # ── Pestaña 1: Pipeline ───────────────────────────────────────────────
        self.tab_pipeline = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab_pipeline, text='  Pipeline  ')
        self._construir_tab_pipeline()

        # ── Pestaña 2: Visualizaciones 3D ────────────────────────────────────
        self.tab_3d = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab_3d, text='  Visualizaciones 3D  ')
        self._construir_tab_3d()

        # ── Pestaña 3: Resultados ─────────────────────────────────────────────
        self.tab_res = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(self.tab_res, text='  Resultados  ')
        self._construir_tab_resultados()

    # ─────────────────────────────────────────────────────────────────────────
    # PESTAÑA 1 — PIPELINE (log + imágenes del proceso)
    # ─────────────────────────────────────────────────────────────────────────

    def _construir_tab_pipeline(self):
        """Log en tiempo real + previsualizaciones de imágenes."""
        tab = self.tab_pipeline

        # ── Sección: título de la visualización ──────────────────────────────
        fr_viz_title = tk.Frame(tab, bg=COLOR_BG)
        fr_viz_title.pack(fill='x', padx=14, pady=(12, 4))
        tk.Label(fr_viz_title, text='—', bg=COLOR_BG,
                 fg=COLOR_CYAN, font=('Courier New', 9)).pack(side='left', padx=(0,8))
        tk.Label(fr_viz_title, text='VISUALIZACIÓN DEL PIPELINE DE IMAGEN',
                 bg=COLOR_BG, fg=COLOR_TEXT_2,
                 font=('Courier New', 9, 'bold')).pack(side='left')

        # ── Figura con 3 paneles de imagen al estilo de la captura ──────────
        fr_imgs = tk.Frame(tab, bg=COLOR_BG)
        fr_imgs.pack(fill='x', padx=14, pady=(0, 8))

        self.fig_imgs = plt.figure(figsize=(13, 3.8))
        self.fig_imgs.patch.set_facecolor(COLOR_BG)

        # 3 subplots lado a lado con margen entre ellos (simula cards)
        self.axes_imgs = []
        for col in range(3):
            ax = self.fig_imgs.add_axes(
                [col/3 + 0.008, 0.0, 1/3 - 0.016, 1.0])
            ax.set_facecolor(COLOR_BG1)
            ax.axis('off')
            self.axes_imgs.append(ax)

        # Estado inicial con placeholder estilizado por panel
        TITULOS_INIT = [
            ('01  ORIGINAL RGB',              'ENTRADA',      COLOR_CYAN),
            ('02-03  ESCALA DE GRISES 512x512','Y=0.299R+0.587G+0.114B', COLOR_AMBER),
            ('05  SEGMENTACION SEMANTICA',     'SEGFORMER-B2', COLOR_LIME),
        ]
        for ax, (titulo, badge, badge_color) in zip(self.axes_imgs, TITULOS_INIT):
            ax.set_facecolor(COLOR_BG1)
            ax.axhspan(0.90, 1.0, color=COLOR_BG3, zorder=5)
            ax.text(0.02, 0.952, titulo, transform=ax.transAxes,
                    color=COLOR_TEXT_1, fontsize=7.5, fontfamily='monospace',
                    fontweight='bold', va='center', zorder=6, clip_on=True)
            ax.text(0.98, 0.952, badge, transform=ax.transAxes,
                    color=badge_color, fontsize=6.5, fontfamily='monospace',
                    va='center', ha='right', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.25', fc=COLOR_BG,
                              ec=badge_color, linewidth=0.8, alpha=0.85))
            ax.text(0.5, 0.52, '--', transform=ax.transAxes,
                    color=COLOR_TEXT_2, fontsize=28, ha='center', va='center')
            ax.text(0.5, 0.36, 'Ejecuta el pipeline para ver la imagen',
                    transform=ax.transAxes, color=COLOR_TEXT_2,
                    fontsize=7, fontfamily='monospace', ha='center', va='center')
            for spine in ax.spines.values():
                spine.set_edgecolor(COLOR_BORDER)
                spine.set_linewidth(0.8)
                spine.set_visible(True)

        self.canvas_imgs = FigureCanvasTkAgg(self.fig_imgs, master=fr_imgs)
        self.canvas_imgs.get_tk_widget().pack(fill='x')

        # ── Separador ─────────────────────────────────────────────────────────
        tk.Frame(tab, bg=COLOR_BORDER, height=1).pack(fill='x', padx=14)

        # ── Log del pipeline ──────────────────────────────────────────────────
        fr_log = tk.Frame(tab, bg=COLOR_BG)
        fr_log.pack(fill='both', expand=True, padx=14, pady=8)

        fr_log_head = tk.Frame(fr_log, bg=COLOR_BG)
        fr_log_head.pack(fill='x', pady=(0, 6))
        tk.Label(fr_log_head, text='LOG DEL PIPELINE',
                 bg=COLOR_BG, fg=COLOR_TEXT_2,
                 font=('Courier New', 9, 'bold')).pack(side='left')
        tk.Button(fr_log_head, text='Limpiar', command=self._limpiar_log,
                  bg=COLOR_BG3, fg=COLOR_TEXT_2, font=('Courier New', 8),
                  relief='flat', padx=6, cursor='hand2').pack(side='right')

        # Text widget con scrollbar para el log
        fr_text = tk.Frame(fr_log, bg=COLOR_BG)
        fr_text.pack(fill='both', expand=True)

        self.txt_log = tk.Text(
            fr_text,
            bg=COLOR_BG1, fg=COLOR_TEXT_1,
            font=('Courier New', 10),
            relief='flat', padx=10, pady=8,
            wrap='word', state='disabled',
            insertbackground=COLOR_CYAN,
            selectbackground=COLOR_BORDER)
        self.txt_log.pack(side='left', fill='both', expand=True)

        scroll_log = ttk.Scrollbar(fr_text, command=self.txt_log.yview)
        scroll_log.pack(side='right', fill='y')
        self.txt_log.config(yscrollcommand=scroll_log.set)

        # Tags de color para el log
        self.txt_log.tag_configure('step', foreground=COLOR_CYAN)
        self.txt_log.tag_configure('ok',   foreground=COLOR_LIME)
        self.txt_log.tag_configure('info', foreground=COLOR_TEXT_1)
        self.txt_log.tag_configure('warn', foreground=COLOR_AMBER)
        self.txt_log.tag_configure('err',  foreground=COLOR_RED)
        self.txt_log.tag_configure('sep',  foreground=COLOR_TEXT_2)

        self._log('Sistema iniciado. Pulsa ▶ EJECUTAR PIPELINE.', 'info')
        self._log('─' * 52, 'sep')

    # ─────────────────────────────────────────────────────────────────────────
    # PESTAÑA 2 — VISUALIZACIONES 3D
    # ─────────────────────────────────────────────────────────────────────────

    def _construir_tab_3d(self):
        """
        Dos gráficos 3D en matplotlib — Manual MTC:
          (a) Superficie difusa de Gravedad  → Tabla 4-8 Cap.4 MTC
          (b) Superficie difusa de Técnica   → Sec.410/415 Cap.400 MTC

        Mouse interactivo:
          - Arrastrar con botón izquierdo → rotar la vista
          - Rueda del mouse              → zoom
          - Label flotante               → muestra Diámetro y Profundidad
            del punto apuntado en tiempo real
        """
        tab = self.tab_3d

        # ── Frame principal — ocupa todo el tab ───────────────────────────────
        fr_top = tk.Frame(tab, bg=COLOR_BG)
        fr_top.pack(fill='both', expand=True, padx=10, pady=(10, 10))

        # ── Figura con 2 subplots — tamaño ampliado al eliminar panel (c) ────
        self.fig_fuzzy = plt.figure(figsize=(13, 7.5))
        self.fig_fuzzy.patch.set_facecolor(COLOR_BG)

        self.ax_urg = self.fig_fuzzy.add_subplot(121, projection='3d')
        self.ax_rec = self.fig_fuzzy.add_subplot(122, projection='3d')

        self._estilizar_ax3d(self.ax_urg,
                             '(a) Nivel de Gravedad — Tabla 4-8 MTC Cap.4',
                             'DIÁMETRO (cm)', 'PROFUNDIDAD (mm)', 'GRAVEDAD MTC')
        self._estilizar_ax3d(self.ax_rec,
                             '(b) Técnica de Conservación — Cap.400 MTC',
                             'DIÁMETRO (cm)', 'PROFUNDIDAD (mm)', 'TÉCNICA MTC')

        self.fig_fuzzy.tight_layout(pad=2.0)
        self.canvas_fuzzy = FigureCanvasTkAgg(self.fig_fuzzy, master=fr_top)
        self.canvas_fuzzy.get_tk_widget().pack(fill='both', expand=True)

        # ── Label flotante para mouse interactivo ─────────────────────────────
        self._cursor_lbl_fuzzy = tk.Label(
            fr_top, text='',
            bg='#0a1520', fg=COLOR_CYAN,
            font=('Courier New', 9, 'bold'),
            relief='flat', padx=10, pady=5,
            bd=0, highlightthickness=1,
            highlightbackground=COLOR_CYAN)

        # ── Instrucciones de uso del mouse ────────────────────────────────────
        fr_instruc = tk.Frame(tab, bg=COLOR_BG3)
        fr_instruc.pack(fill='x', padx=10, pady=(0, 6))
        tk.Label(fr_instruc,
                 text='  🖱  Arrastrar → rotar   |   Rueda → zoom   |   '
                      'Mover → ver valores Diámetro / Profundidad / Gravedad',
                 bg=COLOR_BG3, fg=COLOR_TEXT_2,
                 font=('Courier New', 8)).pack(side='left', pady=4)

        # ── Evento mouse: label flotante con valores difusos ──────────────────
        def _on_mouse_move_fuzzy(event):
            """
            Muestra un label flotante con los valores difusos
            del punto apuntado por el mouse en la superficie.
            """
            ax = event.inaxes
            if ax not in (self.ax_urg, self.ax_rec):
                self._cursor_lbl_fuzzy.place_forget()
                return

            x2d, y2d = event.xdata, event.ydata
            if x2d is None or y2d is None:
                self._cursor_lbl_fuzzy.place_forget()
                return

            # Los ejes están en valores reales MTC (cm y mm)
            diam_real = max(0.0, min(100.0, float(x2d)))   # cm
            prof_real = max(0.0, min(150.0, float(y2d)))   # mm
            dn = diam_real / 100.0
            pn = prof_real / 150.0

            # Calcular valores difusos en ese punto
            u_val, r_val = calcular_salidas_fuzzy(dn, pn)

            # Clasificar gravedad MTC
            if u_val < 0.25:
                grav_txt = 'GRAVEDAD 1'
                grav_col = COLOR_LIME
            elif u_val < 0.65:
                grav_txt = 'GRAVEDAD 2'
                grav_col = COLOR_AMBER
            else:
                grav_txt = 'GRAVEDAD 3'
                grav_col = COLOR_RED

            # Clasificar técnica MTC — solo Sec.410 y Sec.415 (ambas Rutinaria)
            if r_val < 0.55:
                tec_txt = 'Parchado Superficial Sec.410 (Rutinaria)'
            else:
                tec_txt = 'Parchado Profundo Sec.415 (Rutinaria)'

            self._cursor_lbl_fuzzy.config(
                text=f'  Ø {diam_real:.1f} cm  |  ↕ {prof_real:.1f} mm'
                     f'  →  {grav_txt}  |  {tec_txt}  ',
                fg=grav_col)

            # Posicionar label flotante siguiendo el mouse
            widget = self.canvas_fuzzy.get_tk_widget()
            fig_h  = self.fig_fuzzy.get_figheight() * self.fig_fuzzy.dpi
            px = int(event.x) + 15
            py = int(fig_h - event.y) + 10
            w_w = widget.winfo_width()
            if px + 520 > w_w:
                px = int(event.x) - 530
            self._cursor_lbl_fuzzy.place(
                in_=widget, x=max(0, px), y=max(0, py))

        def _on_leave_fuzzy(event):
            self._cursor_lbl_fuzzy.place_forget()

        self.canvas_fuzzy.mpl_connect('motion_notify_event', _on_mouse_move_fuzzy)
        self.canvas_fuzzy.mpl_connect('figure_leave_event', _on_leave_fuzzy)

        # ── Variables dummy para compatibilidad con código que usa ax_pot ─────
        # (algunos métodos llaman a _dibujar_pothole — se mantiene silencioso)
        self.fig_pothole  = self.fig_fuzzy
        self.ax_pot       = self.ax_urg
        self.canvas_pothole = self.canvas_fuzzy
        self._cursor_lbl  = self._cursor_lbl_fuzzy
        self._cursor_point_pot = None
        self._cursor_lines_pot = []

        # ── Dibujar estado inicial ────────────────────────────────────────────
        self._dibujar_superficies_fuzzy()

    def _estilizar_ax3d(self, ax, titulo, xlabel, ylabel, zlabel):
        """Aplica el estilo oscuro a un Axes3D."""
        ax.set_facecolor(COLOR_BG1)
        ax.set_title(titulo, color=COLOR_TEXT_1, fontsize=8.5,
                     fontfamily='monospace', pad=8)
        ax.set_xlabel(xlabel, color=COLOR_TEXT_2,
                      fontsize=7, fontfamily='monospace', labelpad=4)
        ax.set_ylabel(ylabel, color=COLOR_TEXT_2,
                      fontsize=7, fontfamily='monospace', labelpad=4)
        ax.set_zlabel(zlabel, color=COLOR_TEXT_2,
                      fontsize=7, fontfamily='monospace', labelpad=4)

        # Colores de los paneles del fondo
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(COLOR_BORDER)
        ax.yaxis.pane.set_edgecolor(COLOR_BORDER)
        ax.zaxis.pane.set_edgecolor(COLOR_BORDER)

        # Grid y ticks
        ax.grid(True, color=COLOR_BORDER, linewidth=0.4, alpha=0.5)
        ax.tick_params(colors=COLOR_TEXT_2, labelsize=6, pad=2)
        for spine in ax.spines.values():
            spine.set_edgecolor(COLOR_BORDER)

    # ─────────────────────────────────────────────────────────────────────────
    # PESTAÑA 3 — RESULTADOS
    # ─────────────────────────────────────────────────────────────────────────

    def _construir_tab_resultados(self):
        """Panel de resultados: métricas, membresías y casos de referencia MTC."""
        tab = self.tab_res

        # Canvas scrollable
        canvas_r = tk.Canvas(tab, bg=COLOR_BG, highlightthickness=0)
        scroll_r = ttk.Scrollbar(tab, orient='vertical',
                                  command=canvas_r.yview)
        canvas_r.configure(yscrollcommand=scroll_r.set)
        scroll_r.pack(side='right', fill='y')
        canvas_r.pack(fill='both', expand=True)
        fr = tk.Frame(canvas_r, bg=COLOR_BG)
        canvas_r.create_window((0,0), window=fr, anchor='nw')
        fr.bind('<Configure>',
            lambda e: canvas_r.configure(scrollregion=canvas_r.bbox('all')))

        # ── MÉTRICAS ──────────────────────────────────────────────────────────
        self._titulo_seccion(fr, 'MÉTRICAS DEL MODELO  (Ecuaciones 2–5)')

        fr_met = tk.Frame(fr, bg=COLOR_BG)
        fr_met.pack(fill='x', padx=14, pady=(0, 12))

        self.tiles_metricas = {}
        config_met = [
            ('recall',    'Recall',    COLOR_LIME,   'TP/(TP+FN)  Ec.2'),
            ('precision', 'Precisión', COLOR_CYAN,   'TP/(TP+FP)  Ec.3'),
            ('f1',        'F1 Score',  COLOR_AMBER,  '2PR/(P+R)   Ec.4'),
            ('iou',       'IoU',       COLOR_VIOLET, 'A∩B/A∪B     Ec.5'),
        ]
        for i, (key, nombre, color, sub) in enumerate(config_met):
            tile = self._tile_metrica(fr_met, nombre, '—', color, sub)
            tile.grid(row=0, column=i, padx=5, pady=4, sticky='nsew')
            fr_met.columnconfigure(i, weight=1)
            self.tiles_metricas[key] = tile.lbl_val

        # ── PARÁMETROS DEL BACHE ──────────────────────────────────────────────
        self._titulo_seccion(fr, 'PARÁMETROS DEL BACHE  (Ecuaciones 9–14)')

        fr_kv = tk.Frame(fr, bg=COLOR_BG1, relief='flat')
        fr_kv.pack(fill='x', padx=14, pady=(0, 12))

        self.kvs = {}
        kv_items = [
            ('diametro_cm',    'Diámetro estimado',     '—', COLOR_CYAN,   'cm'),
            ('profundidad_mm', 'Profundidad estimada',  '—', COLOR_AMBER,  'mm'),
            ('area_cm2',       'Área estimada  π·r²',   '—', COLOR_LIME,   'cm²'),
            ('vol_cm3',        'Volumen  π·r²·h/3',     '—', COLOR_VIOLET, 'cm³'),
            ('urgencia',       'Urgencia difusa',        '—', COLOR_RED,    ''),
            ('recomendacion',  'Técnica recomendada',   '—', COLOR_AMBER,  ''),
        ]
        for i, (key, nombre, val, color, unidad) in enumerate(kv_items):
            self._kv_row(fr_kv, nombre, val, color, unidad, i)
            self.kvs[key] = (fr_kv, i, color, unidad)

        # ── MEMBRESÍAS DIFUSAS ────────────────────────────────────────────────
        self._titulo_seccion(fr, 'MEMBRESÍAS DIFUSAS  (Sistema Mamdani — Tesis PAVESCAN)')

        fr_fuzz = tk.Frame(fr, bg=COLOR_BG)
        fr_fuzz.pack(fill='x', padx=14, pady=(0, 12))

        fr_prof_fuzz = tk.Frame(fr_fuzz, bg=COLOR_BG1)
        fr_prof_fuzz.pack(side='left', fill='x', expand=True, padx=(0,4))
        tk.Label(fr_prof_fuzz, text='Profundidad (mm)',
                 bg=COLOR_BG1, fg=COLOR_TEXT_2, font=FONT_MONO_S
                 ).pack(anchor='w', padx=8, pady=4)
        self.bars_prof = self._barras_membresia(fr_prof_fuzz)

        fr_diam_fuzz = tk.Frame(fr_fuzz, bg=COLOR_BG1)
        fr_diam_fuzz.pack(side='left', fill='x', expand=True, padx=(4,0))
        tk.Label(fr_diam_fuzz, text='Diámetro (cm)',
                 bg=COLOR_BG1, fg=COLOR_TEXT_2, font=FONT_MONO_S
                 ).pack(anchor='w', padx=8, pady=4)
        self.bars_diam = self._barras_membresia(fr_diam_fuzz)

        # ── TABLA 6 DEL ARTÍCULO ──────────────────────────────────────────────
        self._titulo_seccion(fr, 'CASOS DE REFERENCIA  (Manual MTC — Tabla 4-8)')
        self._tabla_casos(fr)

    def _titulo_seccion(self, parent, texto):
        """Encabezado de sección en la pestaña de resultados."""
        fr = tk.Frame(parent, bg=COLOR_BG)
        fr.pack(fill='x', padx=14, pady=(14, 4))
        tk.Label(fr, text=texto, bg=COLOR_BG, fg=COLOR_TEXT_2,
                 font=('Courier New', 9, 'bold')).pack(side='left')
        tk.Frame(fr, bg=COLOR_BORDER, height=1).pack(
            side='left', fill='x', expand=True, padx=(8, 0), pady=6)

    def _tile_metrica(self, parent, nombre, valor, color, sub):
        """Tarjeta de métrica con barra de progreso."""
        tile = tk.Frame(parent, bg=COLOR_BG2, relief='flat')
        tile.lbl_val = None   # se asigna abajo

        # Barra superior de color
        tk.Frame(tile, bg=color, height=2).pack(fill='x')

        tk.Label(tile, text=nombre, bg=COLOR_BG2, fg=COLOR_TEXT_2,
                 font=FONT_MONO_S).pack(anchor='w', padx=10, pady=(6,0))
        tk.Label(tile, text=sub, bg=COLOR_BG2, fg=COLOR_TEXT_2,
                 font=('Courier New', 7)).pack(anchor='w', padx=10)

        lbl = tk.Label(tile, text='—', bg=COLOR_BG2, fg=color,
                       font=('Courier New', 26, 'bold'))
        lbl.pack(anchor='w', padx=10, pady=4)
        tile.lbl_val = lbl

        # Barra de progreso
        fr_bar = tk.Frame(tile, bg=COLOR_BG3, height=3)
        fr_bar.pack(fill='x', padx=10, pady=(0,8))
        bar_fill = tk.Frame(fr_bar, bg=color, height=3)
        bar_fill.place(x=0, y=0, relwidth=0, relheight=1)
        tile.bar_fill = bar_fill

        return tile

    def _kv_row(self, parent, nombre, valor, color, unidad, idx):
        """Fila clave-valor estilizada."""
        bg = COLOR_BG2 if idx % 2 == 0 else COLOR_BG1
        fr = tk.Frame(parent, bg=bg)
        fr.pack(fill='x', pady=0)
        tk.Label(fr, text=nombre, bg=bg, fg=COLOR_TEXT_1,
                 font=FONT_MONO_S, width=26, anchor='w',
                 padx=12, pady=5).pack(side='left')
        lbl = tk.Label(fr, text=f'{valor} {unidad}'.strip(),
                       bg=bg, fg=color, font=FONT_MONO_S, padx=12)
        lbl.pack(side='right')
        fr.lbl = lbl
        return fr

    def _barras_membresia(self, parent):
        """Crea las barras de membresía difusa para 5 categorías."""
        categorias  = ['MINIMAL', 'LOW', 'MEDIUM', 'MODERATE', 'HIGH']
        colores_bar = ['#b8ff57', '#00e5ff', '#ffb347', '#ff8c42', '#ff5252']
        barras = {}

        for cat, col in zip(categorias, colores_bar):
            fr = tk.Frame(parent, bg=COLOR_BG1)
            fr.pack(fill='x', padx=8, pady=2)
            tk.Label(fr, text=f'{cat:10s}', bg=COLOR_BG1,
                     fg=COLOR_TEXT_2, font=('Courier New', 8), width=10
                     ).pack(side='left')
            track = tk.Frame(fr, bg=COLOR_BG3, height=6)
            track.pack(side='left', fill='x', expand=True, padx=4)
            fill = tk.Frame(track, bg=col, height=6)
            fill.place(x=0, y=0, relwidth=0, relheight=1)
            lbl_pct = tk.Label(fr, text='0%', bg=COLOR_BG1,
                               fg=COLOR_TEXT_2, font=('Courier New', 8), width=4)
            lbl_pct.pack(side='right')
            barras[cat] = (fill, lbl_pct)

        return barras

    def _tabla_casos(self, parent):
        """Tabla de casos de referencia MTC — Tabla 4-8 + Sec.410/415."""
        fr = tk.Frame(parent, bg=COLOR_BG1)
        fr.pack(fill='x', padx=14, pady=(0, 20))

        headers = ['Caso', 'Diámetro', 'Profundidad',
                   'Gravedad MTC', 'Técnica MTC', 'Sección', 'Tipo']
        widths   = [5, 11, 12, 12, 26, 10, 12]
        for j, (h, w) in enumerate(zip(headers, widths)):
            tk.Label(fr, text=h, bg=COLOR_BG3, fg=COLOR_TEXT_2,
                     font=('Courier New', 8, 'bold'),
                     width=w, relief='flat', pady=4
                     ).grid(row=0, column=j, padx=1, pady=1, sticky='nsew')

        # Casos de referencia — Tabla 4-8 MTC + Cap.400 Sec.410/415
        # Formato: (num, diam_cm, prof_mm, gravedad, tecnica, seccion, tipo)
        datos = [
            (1, 12,  25.0, 'GRAVEDAD_1', 'Parchado Superficial', 'Sec.410', 'Rutinaria'),
            (2, 12,  65.0, 'GRAVEDAD_1', 'Parchado Profundo',    'Sec.415', 'Rutinaria'),
            (3, 35,  30.0, 'GRAVEDAD_2', 'Parchado Superficial', 'Sec.410', 'Rutinaria'),
            (4, 35,  75.0, 'GRAVEDAD_2', 'Parchado Profundo',    'Sec.415', 'Rutinaria'),
            (5, 60,  30.0, 'GRAVEDAD_3', 'Parchado Superficial', 'Sec.410', 'Rutinaria'),
            (6, 60, 100.0, 'GRAVEDAD_3', 'Parchado Profundo',    'Sec.415', 'Rutinaria'),
        ]

        GRAV_COLOR = {
            'GRAVEDAD_1': COLOR_LIME,
            'GRAVEDAD_2': COLOR_AMBER,
            'GRAVEDAD_3': COLOR_RED,
        }

        for i, (num, diam, prof, grav, tec, sec, tipo) in enumerate(datos):
            bg      = COLOR_BG2 if i % 2 == 0 else COLOR_BG1
            col_g   = GRAV_COLOR.get(grav, COLOR_TEXT_1)
            valores = [str(num), f'{diam} cm', f'{prof} mm',
                       grav, tec, sec, tipo]
            for j, (v, w) in enumerate(zip(valores, widths)):
                color_txt = col_g if j == 3 else COLOR_LIME if j == 6 else COLOR_TEXT_1
                tk.Label(fr, text=v, bg=bg, fg=color_txt,
                         font=('Courier New', 8), width=w,
                         pady=4).grid(row=i+1, column=j, padx=1, pady=1,
                                      sticky='nsew')

        for j in range(len(headers)):
            fr.columnconfigure(j, weight=1)

    def _decodificar_urgencia(self, val):
        if val < 0.15: return 'NO_DAMAGE'
        if val < 0.45: return 'LOW'
        if val < 0.75: return 'MEDIUM'
        return 'HIGH'

    def _decodificar_rec(self, val):
        if val < 0.12: return 'ROUTINE_MAINTENANCE'
        if val < 0.27: return 'CRACK_SEALING'
        if val < 0.45: return 'CRACK_FELLING'
        if val < 0.65: return 'PATCHING'
        if val < 0.85: return 'SEAL_IRRIGATION'
        return 'RECONSTRUCTION'

    # ─────────────────────────────────────────────────────────────────────────
    # DIBUJADO DE SUPERFICIES 3D
    # ─────────────────────────────────────────────────────────────────────────

    def _actualizar_barras_mtc(self, r_mtc: dict):
        """
        Actualiza las barras de membresía con las categorías del sistema MTC.
        Profundidad: SUPERFICIAL / PROFUNDA  (Sec.410/415)
        Diámetro:    PEQUENIO / MEDIANO / GRANDE  (Tabla 4-8)
        """
        # Membresías de profundidad MTC
        mem_prof_mtc = r_mtc.get('membresias_prof', {})
        mapa_prof = {
            'SUPERFICIAL': 'MINIMAL',
            'PROFUNDA':    'HIGH',
        }
        for cat in self.bars_prof:
            fill, lbl = self.bars_prof[cat]
            fill.place(relwidth=0)
            lbl.config(text='0%')
        for cat_mtc, cat_barra in mapa_prof.items():
            val = mem_prof_mtc.get(cat_mtc, 0.0)
            if cat_barra in self.bars_prof:
                fill, lbl = self.bars_prof[cat_barra]
                fill.place(relwidth=float(val))
                lbl.config(text=f'{int(val*100)}%')

        # Membresías de diámetro MTC — Tabla 4-8
        mem_diam_mtc = r_mtc.get('membresias_diam', {})
        mapa_diam = {
            'PEQUENIO': 'MINIMAL',
            'MEDIANO':  'MEDIUM',
            'GRANDE':   'HIGH',
        }
        for cat in self.bars_diam:
            fill, lbl = self.bars_diam[cat]
            fill.place(relwidth=0)
            lbl.config(text='0%')
        for cat_mtc, cat_barra in mapa_diam.items():
            val = mem_diam_mtc.get(cat_mtc, 0.0)
            if cat_barra in self.bars_diam:
                fill, lbl = self.bars_diam[cat_barra]
                fill.place(relwidth=float(val))
                lbl.config(text=f'{int(val*100)}%')

    def _precalcular_superficies(self):
        """Calcula las superficies fuzzy en background al iniciar la app."""
        N = max(10, self.var_resolucion.get())
        self.superficies_cache = construir_superficies_fuzzy(N)
        self.root.after(0, self._dibujar_superficies_fuzzy)

    def _dibujar_superficies_fuzzy(self):
        """
        Dibuja las superficies difusas (a) y (b) exactamente como
        pavescan_3d.html: DN y PN en X e Y, valor fuzzy en Z,
        coloreado por altura con el colormap seleccionado.
        """
        if self.superficies_cache is None:
            return

        DN, PN, Z_urg, Z_rec = self.superficies_cache
        cmap = colormap_lib.get_cmap(self.var_paleta.get())

        # Marcador en valores reales MTC
        diam_cur = self.var_diametro.get()      # cm reales
        prof_cur = self.var_profundidad.get()   # mm reales
        dn_cur   = max(0.0, min(1.0, diam_cur / 100.0))
        pn_cur   = max(0.0, min(1.0, prof_cur / 150.0))
        u_cur, r_cur = calcular_salidas_fuzzy(dn_cur, pn_cur)

        # Límites compartidos para ambos gráficos — valores reales MTC
        XLIM = (0, 100)    # Diámetro    eje X: 0-100 cm  (Tabla 4-8 MTC)
        YLIM = (0, 150)    # Profundidad eje Y: 0-150 mm (Sec.410/415 MTC)
        ZLIM = (0, 1)
        XTICKS = [0, 20, 50, 100]   # límites Gravedad 1/2/3 Tabla 4-8
        YTICKS = [0, 50, 100, 150]  # umbral Sec.410 y Sec.415
        ZTICKS = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Recortar malla al rango exacto para evitar expansión automática
        import numpy as _np2
        mask_x = (DN[0,:] >= 0) & (DN[0,:] <= 100)
        mask_y = (PN[:,0] >= 0) & (PN[:,0] <= 150)
        DN_  = DN[_np2.ix_(mask_y, mask_x)]
        PN_  = PN[_np2.ix_(mask_y, mask_x)]
        Zu_  = Z_urg[_np2.ix_(mask_y, mask_x)]
        Zr_  = Z_rec[_np2.ix_(mask_y, mask_x)]

        # Quitar colorbars previas (si existen) para no acumularlas en cada redibujo
        for attr in ('_cbar_urg', '_cbar_rec'):
            cb = getattr(self, attr, None)
            if cb is not None:
                try:
                    cb.remove()
                except Exception:
                    pass
                setattr(self, attr, None)

        # ── Superficie (a): Gravedad ──────────────────────────────────────────
        self.ax_urg.cla()
        self._estilizar_ax3d(self.ax_urg,
                             '(a) Nivel de Gravedad — Tabla 4-8 MTC Cap.4',
                             'DIÁMETRO (cm)', 'PROFUNDIDAD (mm)', 'GRAVEDAD MTC')
        # Fijar límites ANTES de graficar para que matplotlib los respete
        self.ax_urg.set_xlim3d(*XLIM)
        self.ax_urg.set_ylim3d(*YLIM)
        self.ax_urg.set_zlim3d(*ZLIM)
        surf_urg = self.ax_urg.plot_surface(
            DN_, PN_, Zu_,
            cmap=cmap, alpha=0.92,
            rstride=1, cstride=1,
            linewidth=0.25, edgecolor=(1, 1, 1, 0.15),
            antialiased=True)
        # "Sombra" proyectada en el piso (z=0): mismas regiones categóricas
        # que ves en el label flotante, en colores planos y sólidos — esto
        # da una vista cenital clara de dónde empieza cada Gravedad.
        self.ax_urg.contourf(
            DN_, PN_, Zu_,
            levels=UMBRAL_GRAVEDAD, cmap=CMAP_GRAVEDAD_DISCRETO,
            norm=NORM_GRAVEDAD_DISCRETO, zdir='z', offset=ZLIM[0],
            alpha=0.55, antialiased=True)
        # Líneas blancas marcando el límite exacto entre niveles
        self.ax_urg.contour(
            DN_, PN_, Zu_,
            levels=UMBRAL_GRAVEDAD[1:-1], colors='white',
            linewidths=1.1, zdir='z', offset=ZLIM[0])
        self.ax_urg.scatter([diam_cur], [prof_cur], [u_cur + 0.04],
                            color='white', s=80, zorder=10, depthshade=False,
                            edgecolor='black', linewidth=0.6)
        # Fijar límites también DESPUÉS por seguridad
        self.ax_urg.set_xlim3d(*XLIM)
        self.ax_urg.set_ylim3d(*YLIM)
        self.ax_urg.set_zlim3d(*ZLIM)
        self.ax_urg.set_xticks(XTICKS)
        self.ax_urg.set_yticks(YTICKS)
        self.ax_urg.set_zticks(ZTICKS)
        self.ax_urg.view_init(elev=25, azim=-45)

        cbar_u = self.fig_fuzzy.colorbar(
            colormap_lib.ScalarMappable(norm=NORM_GRAVEDAD_DISCRETO,
                                         cmap=CMAP_GRAVEDAD_DISCRETO),
            ax=self.ax_urg, shrink=0.55, pad=0.08, aspect=14)
        cbar_u.set_ticks([0.125, 0.45, 0.825])
        cbar_u.set_ticklabels(['GRAVEDAD 1\nLeve', 'GRAVEDAD 2\nModerado',
                               'GRAVEDAD 3\nSevero'])
        cbar_u.ax.tick_params(colors=COLOR_TEXT_2, labelsize=6.5)
        cbar_u.outline.set_edgecolor(COLOR_BORDER)
        self._cbar_urg = cbar_u

        # ── Superficie (b): Técnica ───────────────────────────────────────────
        self.ax_rec.cla()
        self._estilizar_ax3d(self.ax_rec,
                             '(b) Técnica de Conservación — Cap.400 MTC',
                             'DIÁMETRO (cm)', 'PROFUNDIDAD (mm)', 'TÉCNICA MTC')
        # Fijar límites ANTES de graficar — exactamente iguales a (a)
        self.ax_rec.set_xlim3d(*XLIM)
        self.ax_rec.set_ylim3d(*YLIM)
        self.ax_rec.set_zlim3d(*ZLIM)
        self.ax_rec.plot_surface(
            DN_, PN_, Zr_,
            cmap=cmap, alpha=0.92,
            rstride=1, cstride=1,
            linewidth=0.25, edgecolor=(1, 1, 1, 0.15),
            antialiased=True)
        self.ax_rec.contourf(
            DN_, PN_, Zr_,
            levels=UMBRAL_TECNICA, cmap=CMAP_TECNICA_DISCRETO,
            norm=NORM_TECNICA_DISCRETO, zdir='z', offset=ZLIM[0],
            alpha=0.55, antialiased=True)
        self.ax_rec.contour(
            DN_, PN_, Zr_,
            levels=UMBRAL_TECNICA[1:-1], colors='white',
            linewidths=1.1, zdir='z', offset=ZLIM[0])
        self.ax_rec.scatter([diam_cur], [prof_cur], [r_cur + 0.04],
                            color='white', s=80, zorder=10, depthshade=False,
                            edgecolor='black', linewidth=0.6)
        # Fijar límites también DESPUÉS por seguridad
        self.ax_rec.set_xlim3d(*XLIM)
        self.ax_rec.set_ylim3d(*YLIM)
        self.ax_rec.set_zlim3d(*ZLIM)
        self.ax_rec.set_xticks(XTICKS)
        self.ax_rec.set_yticks(YTICKS)
        self.ax_rec.set_zticks(ZTICKS)
        self.ax_rec.view_init(elev=25, azim=-45)

        cbar_r = self.fig_fuzzy.colorbar(
            colormap_lib.ScalarMappable(norm=NORM_TECNICA_DISCRETO,
                                         cmap=CMAP_TECNICA_DISCRETO),
            ax=self.ax_rec, shrink=0.55, pad=0.08, aspect=14)
        cbar_r.set_ticks([0.275, 0.775])
        cbar_r.set_ticklabels(['Sec.410\nSuperficial', 'Sec.415\nProfundo'])
        cbar_r.ax.tick_params(colors=COLOR_TEXT_2, labelsize=6.5)
        cbar_r.outline.set_edgecolor(COLOR_BORDER)
        self._cbar_rec = cbar_r

        self.fig_fuzzy.tight_layout(pad=1.5)
        self.canvas_fuzzy.draw()

    def _dibujar_pothole(self):
        """
        Panel (c) eliminado — ahora los gráficos (a) y (b) ocupan
        todo el espacio de la pestaña 3D.
        Este método se conserva por compatibilidad pero no dibuja nada.
        """
        return  # Panel (c) desactivado


    def _ejecutar_pipeline(self):
        """
        Lanza pavescan_comentado.ejecutar_pipeline() en un hilo separado
        para no bloquear la interfaz mientras se procesa.
        """
        if self.pipeline_activo:
            return

        self.pipeline_activo = True
        self.btn_ejecutar.config(text='⏳  PROCESANDO...', bg='#1a3a3a',
                                  fg=COLOR_CYAN, state='disabled')
        self.lbl_estado.config(text='● PROCESANDO', fg=COLOR_CYAN)
        self._resetear_pasos()

        # Limpiar el log de la corrida anterior — cada ejecución empieza
        # con la consola en blanco, para no confundir resultados de
        # análisis distintos acumulados en pantalla.
        self._limpiar_log()

        # Cambiar a pestaña Pipeline para ver el log en tiempo real
        self.notebook.select(0)

        self._log('═' * 52, 'sep')
        self._log('EJECUTANDO PIPELINE — pavescan_segformer_mtc.py', 'step')
        self._log('═' * 52, 'sep')

        # Lanzar en hilo daemon (muere si se cierra la ventana)
        self.hilo_pipeline = threading.Thread(
            target=self._hilo_pipeline,
            daemon=True)
        self.hilo_pipeline.start()

    def _on_cambio_modo(self):
        """
        Actualiza las etiquetas de las superficies 3D según el sistema MTC.
        """
        self.ax_urg.set_title(
            '(a) Nivel de Gravedad — Tabla 4-8 MTC Cap.4',
            color=COLOR_TEXT_1, fontsize=8.5, fontfamily='monospace')
        self.ax_rec.set_title(
            '(b) Técnica de Conservación — Cap.400 MTC',
            color=COLOR_TEXT_1, fontsize=8.5, fontfamily='monospace')
        self.ax_urg.set_xlabel('DIÁMETRO (cm)',    color=COLOR_TEXT_2)
        self.ax_rec.set_xlabel('DIÁMETRO (cm)',    color=COLOR_TEXT_2)
        self.ax_urg.set_ylabel('PROFUNDIDAD (mm)', color=COLOR_TEXT_2)
        self.ax_rec.set_ylabel('PROFUNDIDAD (mm)', color=COLOR_TEXT_2)
        self._log('Sistema MTC activado — RD N°08-2014-MTC/14', 'ok')
        self._log('  Diámetro (Tabla 4-8)  |  Profundidad (Sec.410/415)', 'info')
        self._log('  Base normativa: Manual de Carreteras MTC', 'info')

        # Regenerar superficies con el modo actual
        self._redibujar_todo()

    def _hilo_pipeline(self):
        """
        Se ejecuta en un hilo separado.
        Importa pavescan_comentado, llama a ejecutar_pipeline(),
        y actualiza la UI mediante root.after() (thread-safe).
        """
        try:
            # ── Importar (o recargar) pavescan_segformer ──────────────────────
            self._log_hilo('Importando pavescan_segformer_mtc.py...', 'info')
            try:
                import pavescan_segformer_mtc as pc
                importlib.reload(pc)               # asegurar versión fresca
            except ImportError as e:
                self._log_hilo(f'ERROR al importar: {e}', 'err')
                self._log_hilo('Asegúrate de que pavescan_segformer_mtc.py', 'warn')
                self._log_hilo('esté en la misma carpeta que este script.', 'warn')
                self._finalizar_pipeline(None)
                return

            self._log_hilo('Módulo cargado correctamente.', 'ok')

            # ── Preparar parámetros ───────────────────────────────────────────
            diam_in  = self.var_diametro.get()
            prof_in  = self.var_profundidad.get()
            ruta_img = self.var_imagen.get()   # imagen seleccionada manualmente

            self._log_hilo(f'Diámetro configurado: {diam_in:.1f} cm', 'info')
            self._log_hilo(f'Profundidad configurada: {prof_in:.1f} mm', 'info')
            if ruta_img and os.path.exists(ruta_img):
                self._log_hilo(f'Imagen seleccionada: {os.path.basename(ruta_img)}', 'ok')
            self._log_hilo('─' * 52, 'sep')

            # ── Ejecutar cada paso con actualización visual ────────────────────
            pasos_fn = [
                (0,  'Escala de grises  Ec.1', None),
                (1,  'Resize 512×512 px', None),
                (2,  'Píxel → Tensor [0,1]', None),
                (3,  'Etiquetado semántico', None),
                (4,  'SegFormer-B2', None),
                (5,  'Aumento de datos ×3', None),
                (6,  'Recall · Prec · F1 · IoU', None),
                (7,  'Nube de puntos 3D', None),
                (8,  'RANSAC · Ec.6', None),
                (9,  '2D → 3D · Ec.7-8', None),
                (10, 'Diámetro y profundidad', None),
                (11, 'Lógica difusa Mamdani', None),
            ]

            # Ejecutar el pipeline completo (captura prints internos)
            import io, contextlib

            # Redirigir stdout para capturar los prints del pipeline
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                # Simular progreso de pasos
                for idx, nombre, _ in pasos_fn:
                    self.root.after(0, self._marcar_paso, idx, 'active')
                    import time; time.sleep(0.08)

                # Llamada real al pipeline
                ruta_npy_real = self.var_depth_npy.get()
                resultado = pc.ejecutar_pipeline(
                    diametro_manual=diam_in,
                    profundidad_manual=prof_in,
                    modo_normativo=self.var_modo_fuzzy.get(),
                    ruta_imagen_manual=ruta_img if ruta_img and os.path.exists(ruta_img) else None,
                    ruta_depth_npy=ruta_npy_real if ruta_npy_real and os.path.exists(ruta_npy_real) else None
                )

                # Marcar todos como completados
                for idx, _, _ in pasos_fn:
                    self.root.after(0, self._marcar_paso, idx, 'done')

            # Enviar las líneas del print al log
            for linea in buffer.getvalue().splitlines():
                if linea.strip():
                    tipo = ('step' if linea.startswith('[') else
                            'ok'   if 'Recall' in linea or 'IoU' in linea else
                            'info')
                    self._log_hilo(linea, tipo)

            self._finalizar_pipeline(resultado)

        except Exception as e:
            self._log_hilo(f'ERROR INESPERADO: {e}', 'err')
            self._log_hilo(traceback.format_exc(), 'err')
            self._finalizar_pipeline(None)

    def _finalizar_pipeline(self, resultado):
        """Llamado desde el hilo al terminar. Actualiza la UI."""
        self.resultados = resultado
        self.root.after(0, self._actualizar_ui_con_resultados, resultado)

    def _actualizar_ui_con_resultados(self, resultado):
        """
        Actualiza todos los paneles con los resultados del pipeline.
        Se ejecuta en el hilo principal (tkinter thread-safe).
        """
        self.pipeline_activo = False
        self.btn_ejecutar.config(text='▶  EJECUTAR PIPELINE',
                                  bg=COLOR_CYAN, fg='#000000',
                                  state='normal')

        if resultado is None:
            self.lbl_estado.config(text='● ERROR', fg=COLOR_RED)
            self._log('Pipeline finalizado con errores.', 'err')
            return

        self.lbl_estado.config(text='● COMPLETADO', fg=COLOR_LIME)
        self._log('─' * 52, 'sep')
        self._log('PIPELINE COMPLETADO EXITOSAMENTE ✓', 'ok')
        self._log('─' * 52, 'sep')

        # ── Actualizar métricas ───────────────────────────────────────────────
        tiene_gt = resultado.get('tiene_ground_truth', True)
        m = resultado.get('metricas', {}) or {}
        pares = [('recall','recall'), ('precision','precision'),
                 ('f1','f1'), ('iou','iou')]
        for key_tile, key_m in pares:
            if key_tile in self.tiles_metricas:
                tile_lbl = self.tiles_metricas[key_tile]
                tile_fr = tile_lbl.master
                if tiene_gt:
                    val = m.get(key_m, 0)
                    tile_lbl.config(text=f'{val:.1f}%')
                    if hasattr(tile_fr, 'bar_fill'):
                        tile_fr.bar_fill.place(relwidth=val/100)
                else:
                    # Sin ground truth real (imagen de campo nueva): las
                    # métricas de validación no aplican — mostrar 0.0% sería
                    # engañoso, así que se indica claramente "N/A".
                    tile_lbl.config(text='N/A')
                    if hasattr(tile_fr, 'bar_fill'):
                        tile_fr.bar_fill.place(relwidth=0)
        if not tiene_gt:
            self._log('Sin ground truth — modo detección de campo '
                      '(métricas de validación no aplican).', 'info')

        # ── Actualizar KV de parámetros ───────────────────────────────────────
        diam        = resultado.get('diametro_cm', 0)
        prof        = resultado.get('profundidad_mm', 0)
        tiene_bache = resultado.get('tiene_bache', True)
        tiene_prof  = resultado.get('tiene_profundidad_real', True)
        fuente_prof = resultado.get('fuente_profundidad', None)
        fz          = resultado.get('resultado_mtc', {}) or {}
        area = math.pi * (diam/2)**2
        vol  = math.pi * (diam/2)**2 * (prof/10) / 3

        # Actualizar filas KV en la pestaña de resultados
        self._actualizar_kv('diametro_cm', f'{diam:.2f}', 'cm')
        if tiene_bache and tiene_prof:
            self._actualizar_kv('profundidad_mm', f'{prof:.2f}', 'mm')
            self._actualizar_kv('area_cm2', f'{area:.1f}', 'cm²')
            self._actualizar_kv('vol_cm3',  f'{vol:.1f}',  'cm³')
        else:
            self._actualizar_kv('profundidad_mm', 'sin datos reales', '')
            self._actualizar_kv('area_cm2', f'{area:.1f}', 'cm²')
            self._actualizar_kv('vol_cm3',  'N/A', '')

        # Mostrar resultados en terminología del Manual MTC
        if not tiene_bache:
            self._actualizar_kv('urgencia',      'NO ES UN BACHE', '')
            self._actualizar_kv('recomendacion', 'Sin acción requerida', '')
        elif not tiene_prof:
            self._actualizar_kv('urgencia',      'Bache detectado — falta profundidad', '')
            self._actualizar_kv('recomendacion', 'Conecta la RealSense o ingresa medición manual', '')
        else:
            gravedad_mtc = fz.get('gravedad', '')
            tecnica_mtc  = fz.get('nombre_tecnica', fz.get('tecnica', ''))
            self._actualizar_kv('urgencia',      f'Gravedad {gravedad_mtc} (MTC Cap.4)', '')
            self._actualizar_kv('recomendacion', f'{tecnica_mtc}', '')

        # ── Resultados MTC ────────────────────────────────────────────────────
        if not tiene_bache:
            self._log('─' * 52, 'sep')
            self._log('RESULTADO: NO SE DETECTÓ NINGÚN BACHE EN LA IMAGEN', 'warn')
            self._log('  El SegFormer no encontró una región de bache con', 'info')
            self._log('  confianza suficiente. No se calcula gravedad MTC.', 'info')
            self._log('─' * 52, 'sep')
        elif not tiene_prof:
            self._log('─' * 52, 'sep')
            self._log('BACHE DETECTADO, PERO SIN DATOS REALES DE PROFUNDIDAD', 'warn')
            self._log('  Selecciona una foto con su .npy real de la RealSense,', 'info')
            self._log('  o ingresa una medición manual en el slider de', 'info')
            self._log('  Profundidad, para clasificar gravedad y técnica MTC.', 'info')
            self._log('─' * 52, 'sep')
        elif self.var_modo_fuzzy.get() == 'MTC' and _MTC_DISPONIBLE:
            try:
                gravedad = fz.get('gravedad', '')
                tec      = fz.get('nombre_tecnica', fz.get('tecnica', ''))
                sec      = fz.get('seccion_mtc', 'Cap.400 MTC').split('—')[0].strip()
                tipo_con = fz.get('nombre_conservacion', '')
                ice      = fz.get('ice', {}).get('ice', 0.0) if isinstance(fz.get('ice'), dict) else 0.0
                ice_cls  = fz.get('ice', {}).get('clasificacion', '') if isinstance(fz.get('ice'), dict) else ''

                area_m2 = math.pi * (diam / 200) ** 2  # estimado desde diámetro

                self._log('── EVALUACIÓN MTC ──', 'step')
                self._log(f'  Área estimada: {area_m2:.4f} m²'
                          f'  |  Profundidad: {prof:.1f} mm (fuente: {fuente_prof})', 'info')
                self._log(f'  Gravedad:          {gravedad}', 'ok')
                if tipo_con:
                    self._log(f'  Tipo conservación: {tipo_con}', 'ok')
                self._log(f'  Técnica MTC:       {tec}', 'ok')
                self._log(f'  Sección:           {sec}', 'info')
                self._log(f'  Norma: RD N°08-2014-MTC/14', 'info')

                # Actualizar barras de membresía con categorías MTC (reales)
                if _MTC_DISPONIBLE:
                    r_mtc_barras = _mtc_evaluar(diam, prof)
                    self._actualizar_barras_mtc(r_mtc_barras)
            except Exception as _e_mtc:
                self._log(f'  [MTC] Error en evaluación: {_e_mtc}', 'warn')

        # NOTA: las barras de membresía ya se actualizaron arriba con
        # _actualizar_barras_mtc(), que traduce las categorías reales del
        # Manual MTC (PEQUENIO/MEDIANO/GRANDE, SUPERFICIAL/PROFUNDA) a las
        # barras MINIMAL/LOW/MEDIUM/MODERATE/HIGH de la interfaz.
        # Antes, aquí se volvía a llamar a _actualizar_barras() usando
        # fz.get('membresias_prof'/'membresias_diam') directamente — pero
        # esas claves vienen con nombres MTC (SUPERFICIAL/PROFUNDA, etc.)
        # que no existen en self.bars_prof/self.bars_diam (que usan
        # MINIMAL/LOW/MEDIUM/MODERATE/HIGH), así que .get(cat, 0.0) siempre
        # devolvía 0.0 y sobrescribía las barras correctas con 0% justo
        # después de haberlas llenado bien. Se elimina esa llamada duplicada.

        # ── Actualizar imágenes en pestaña Pipeline ───────────────────────────
        self._generar_imagenes_pipeline()

        # ── Redibujar gráficas 3D ────────────────────────────────────────────
        # Primero dibujar el bache (rápido — no requiere cálculo fuzzy)
        self._dibujar_pothole()

        # Calcular superficies fuzzy en background para no bloquear la UI.
        # Usamos _redibujar_todo() que ya incluye control de hilos y flag.
        self._redibujar_todo()

        # Cambiar a pestaña 3D después de que el bache (rápido) esté listo
        self.root.after(500, lambda: self.notebook.select(1))

    def _actualizar_kv(self, key, valor, unidad):
        """Actualiza el texto de una fila KV en la pestaña de resultados."""
        if key in self.kvs:
            fr, idx, color, _ = self.kvs[key]
            # Buscar el label en la fila correspondiente
            hijos = fr.winfo_children()
            if idx < len(hijos):
                fila = hijos[idx]
                for widget in fila.winfo_children():
                    if isinstance(widget, tk.Label) and widget.cget('anchor') == 'e' or \
                       hasattr(fila, 'lbl'):
                        pass
                # Más simple: reconfigurar el último label de la fila
                labels = [w for w in fila.winfo_children()
                          if isinstance(w, tk.Label)]
                if labels:
                    labels[-1].config(text=f'{valor} {unidad}'.strip(),
                                      fg=color)

    def _actualizar_barras(self, barras, membresias):
        """Actualiza las barras de membresía difusa."""
        for cat, (fill, lbl_pct) in barras.items():
            val = membresias.get(cat, 0.0)
            fill.place(relwidth=float(val))
            lbl_pct.config(text=f'{int(val*100)}%')

    def _generar_imagenes_pipeline(self):
        """
        Genera los 3 paneles de imagen del pipeline con estilo profesional.

        Cada panel tiene:
          - Encabezado oscuro con título y badge de color
          - La imagen del proceso (original / grises / segmentación)
          - Metadatos superpuestos en la esquina inferior izquierda
            (fecha, coordenadas GPS, altitud, velocidad — como en la captura)
          - Barra inferior con estadísticas (dimensiones, rango tensor, IoU)
          - Contornos del bache dibujados sobre la imagen de segmentación

        Si el usuario cargó una imagen real, la usa; si no, genera una sintética.
        """
        try:
            import datetime as _dt

            # ── Generar o cargar imagen fuente ────────────────────────────────
            ruta = self.var_imagen.get()
            if ruta and os.path.exists(ruta):
                # np.fromfile soporta rutas con caracteres especiales (Ñ, tildes)
                # cv2.imread falla en Windows con estos caracteres
                img_bgr = cv2.imdecode(
                    np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if img_bgr is None:
                    raise ValueError(f"No se pudo leer: {ruta}")
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # Escalar para que no sea demasiado grande
                h_orig, w_orig = img.shape[:2]
                max_dim = 640
                if h_orig > max_dim or w_orig > max_dim:
                    scale = min(max_dim/h_orig, max_dim/w_orig)
                    img = cv2.resize(img, (int(w_orig*scale), int(h_orig*scale)))
                es_real = True
            elif self.resultados is not None and 'imagen_rgb' in self.resultados:
                # Sin selección manual: usar EXACTAMENTE la misma imagen que
                # ejecutar_pipeline ya analizó (antes se elegía una imagen
                # NUEVA al azar aquí, causando que el panel visual mostrara
                # una foto distinta a la que realmente se evaluó en el log).
                img = self.resultados['imagen_rgb']
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                es_real = True
                h_orig, w_orig = img.shape[:2]
            else:
                # Aún no se ejecutó ningún pipeline — no hay imagen real que
                # mostrar todavía; se toma una del dataset solo para no dejar
                # el panel vacío antes del primer análisis.
                import pavescan_segformer_mtc as _pc_mod
                img, _mask_gt_gui, _ruta_img_gui, _ruta_mask_gui, _total_gui = \
                    _pc_mod._seleccionar_imagen_dataset(split="test")
                es_real = True
                h_orig, w_orig = img.shape[:2]

            h_orig, w_orig = img.shape[:2]

            # ── Paso 1: Escala de grises (Y = 0.299R + 0.587G + 0.114B) ──────
            R = img[:,:,0].astype(np.float32)
            G = img[:,:,1].astype(np.float32)
            B = img[:,:,2].astype(np.float32)
            gray = np.clip(0.299*R + 0.587*G + 0.114*B, 0, 255).astype(np.uint8)
            t_min = float(gray.min()) / 255
            t_max = float(gray.max()) / 255

            # ── Paso 2: Resize a 512×512 ──────────────────────────────────────
            gray_512 = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_CUBIC)

            # ── Paso 4-5: Segmentación — SIEMPRE la predicción real del SegFormer ──
            # Ya no se dibuja ningún contorno de respaldo (umbral gaussiano)
            # cuando el modelo predice 0 px — mostrar algo ahí sería sugerir
            # visualmente una detección que en realidad no existe.
            if self.resultados is not None and 'mascara_pred' in self.resultados:
                mask = self.resultados['mascara_pred']
                if mask.shape != (gray_512.shape[0], gray_512.shape[1]):
                    mask = cv2.resize(mask, (gray_512.shape[1], gray_512.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                if int((mask > 0).sum()) > 0:
                    badge_seg = 'SEGFORMER-B2'
                    badge_col = COLOR_LIME
                else:
                    badge_seg = 'SEGFORMER-B2 (sin detección)'
                    badge_col = COLOR_TEXT_2
            else:
                mask = np.zeros_like(gray_512)
                badge_seg = 'SEGFORMER-B2 (sin ejecutar)'
                badge_col = COLOR_TEXT_2

            # Métricas de segmentación
            px_detectados = int((mask > 0).sum())
            total_px      = mask.shape[0] * mask.shape[1]

            # IoU real si hay ground truth disponible
            if (self.resultados is not None and
                    'metricas' in self.resultados and
                    self.resultados['metricas']['iou'] > 0):
                iou_val  = self.resultados['metricas']['iou']
                iou_txt  = f'IoU: {iou_val:.1f}%'
            else:
                iou_txt  = f'{px_detectados:,} px bache'

            # Contornos del bache detectado por SegFormer
            contornos, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contornos = [c for c in contornos if cv2.contourArea(c) > 300]

            # ── Construir overlay de segmentación con máscara SegFormer ──────
            # Base: imagen RGB original redimensionada a 512×512
            if img.shape[:2] != (512, 512):
                img_overlay = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            else:
                img_overlay = img.copy()
            overlay = img_overlay.copy()
            # Tinte cian translúcido sobre los píxeles del bache
            tinte = overlay.copy()
            tinte[mask > 0] = [0, 180, 210]
            overlay = cv2.addWeighted(overlay, 0.55, tinte, 0.45, 0)
            # Contornos: línea cian brillante
            cv2.drawContours(overlay, contornos, -1, (0, 229, 255), 2)
            # Contorno exterior grueso lima (borde principal del bache)
            if contornos:
                mayor = max(contornos, key=cv2.contourArea)
                cv2.drawContours(overlay, [mayor], -1, (0, 229, 200), 3)

            # ── Fecha real de la ejecución (ya no se inventan GPS/altitud) ────
            ahora = _dt.datetime.now()
            fecha_str  = ahora.strftime('%d %b. %Y %I:%M:%S %p')


            # ── Dibujar los 3 paneles ─────────────────────────────────────────
            diam = self.var_diametro.get()
            prof = self.var_profundidad.get()

            configs = [
                # (ax, imagen, titulo, badge, badge_color, usar_cmap, pie_izq, pie_der)
                (self.axes_imgs[0],
                 img,
                 '01  ORIGINAL RGB',
                 'ENTRADA', COLOR_CYAN,
                 False,
                 f'{w_orig}x{h_orig} px',
                 f'{w_orig*h_orig//1000}K pixels'),

                (self.axes_imgs[1],
                 gray_512,
                 '02-03  ESCALA DE GRISES 512x512',
                 'Y=0.299R+0.587G+0.114B', COLOR_AMBER,
                 True,
                 f'Tensor Float32 (1x512x512)',
                 f'[{t_min:.3f}, {t_max:.3f}]'),

                (self.axes_imgs[2],
                 overlay,
                 '05  SEGMENTACION SEMANTICA',
                 badge_seg, badge_col,
                 False,
                 f'{px_detectados:,} px detectados',
                 iou_txt),
            ]

            for ax, data, titulo, badge, badge_c, usar_cmap, pie_izq, pie_der in configs:
                ax.cla()
                ax.set_facecolor(COLOR_BG1)

                # ── Imagen central ────────────────────────────────────────────
                if usar_cmap:
                    ax.imshow(data, cmap='gray', aspect='auto',
                              extent=[0,1,0,1], zorder=1)
                else:
                    ax.imshow(data, aspect='auto',
                              extent=[0,1,0,1], zorder=1)

                # ── Encabezado oscuro con título y badge ──────────────────────
                ax.axhspan(0.905, 1.0, color=COLOR_BG3, alpha=0.92, zorder=4)
                ax.text(0.012, 0.951, titulo,
                        transform=ax.transAxes,
                        color=COLOR_TEXT_1, fontsize=7,
                        fontfamily='monospace', fontweight='bold',
                        va='center', zorder=5, clip_on=True)
                ax.text(0.988, 0.951, badge,
                        transform=ax.transAxes,
                        color=badge_c, fontsize=6.5, fontfamily='monospace',
                        va='center', ha='right', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.28', fc=COLOR_BG,
                                  ec=badge_c, linewidth=0.8, alpha=0.9))

                # ── Metadatos reales en la esquina inferior izquierda ─────────
                # Fondo semitransparente negro para legibilidad
                ax.axhspan(0.0, 0.13, xmin=0.0, xmax=0.72,
                           color='#000000', alpha=0.55, zorder=3)
                nombre_mostrar = (self.resultados or {}).get('nombre_archivo', '')
                meta_lines = [fecha_str] + ([nombre_mostrar] if nombre_mostrar else [])
                for k, linea in enumerate(meta_lines):
                    ax.text(0.012, 0.11 - k * 0.055, linea,
                            transform=ax.transAxes,
                            color='white', fontsize=5.8,
                            fontfamily='monospace',
                            va='top', zorder=6)

                # ── Barra inferior con estadísticas ───────────────────────────
                ax.axhspan(0.0, 0.06, color=COLOR_BG3, alpha=0.90, zorder=4)
                ax.text(0.012, 0.03, pie_izq,
                        transform=ax.transAxes,
                        color=COLOR_TEXT_2, fontsize=6,
                        fontfamily='monospace', va='center', zorder=5)
                ax.text(0.988, 0.03, pie_der,
                        transform=ax.transAxes,
                        color=badge_c, fontsize=6,
                        fontfamily='monospace', va='center',
                        ha='right', zorder=5)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')

                # Borde del card
                for spine in ax.spines.values():
                    spine.set_edgecolor(COLOR_BORDER)
                    spine.set_linewidth(0.8)
                    spine.set_visible(True)

            self.canvas_imgs.draw()
            self._log(
                f'Imagenes actualizadas: {w_orig}x{h_orig} px | '
                f'bache: {px_detectados:,} px | {iou_txt}',
                'ok')

        except Exception as e:
            self._log(f'Error generando imagenes: {e}', 'warn')
            import traceback as _tb
            self._log(_tb.format_exc(), 'warn')

    def _actualizar_3d_en_vivo(self):
        """
        Llamada en cada tick del slider de diámetro/profundidad.

        PROBLEMA ORIGINAL: se redibujaba en cada evento del slider, lo que
        saturaba matplotlib con cientos de llamadas simultáneas, generando
        el bucle infinito de 'Recalculando superficies con N=28...' que se
        veía en el log.

        SOLUCIÓN — debounce de 300 ms:
        Si el usuario sigue moviendo el slider, cancelamos el after() anterior
        y programamos uno nuevo. Solo cuando el usuario PARA de mover (durante
        300 ms) se ejecuta el redibujo. Esto convierte N eventos en 1 llamada.
        """
        # Cancelar cualquier redibujo pendiente anterior
        if self._debounce_id is not None:
            self.root.after_cancel(self._debounce_id)
            self._debounce_id = None

        # Programar redibujo diferido — solo se ejecuta si no llega otro evento
        self._debounce_id = self.root.after(300, self._redibujar_bache_solo)

    def _redibujar_bache_solo(self):
        """
        Redibujo ligero: solo la geometría del bache (c).
        No recalcula las superficies fuzzy (a) y (b) — eso es más costoso.
        Solo actualiza el marcador de posición actual en las superficies.
        Llamado tras el debounce del slider.
        """
        self._debounce_id = None
        try:
            self._dibujar_pothole()
            # Actualizar marcador en superficies solo si ya están calculadas
            if self.superficies_cache is not None:
                self._dibujar_superficies_fuzzy()
        except Exception:
            pass  # silenciar errores de redibujo parcial

    def _redibujar_todo(self):
        """
        Recalcula las superficies fuzzy (a) y (b) con la resolución actual.
        Solo se llama desde el botón '🔄 Redibujar' o al cambiar paleta.
        NUNCA se llama desde el slider — eso lo maneja _actualizar_3d_en_vivo.

        CORRECCIÓN: el hilo anterior se invalida mediante el flag
        _hilo_superficie_activo antes de lanzar uno nuevo, evitando que
        múltiples hilos acumulen trabajo en background.
        """
        # Invalidar hilo anterior si sigue corriendo
        self._hilo_superficie_activo = False

        N = max(10, self.var_resolucion.get())
        self._log(f'Recalculando superficies N={N}×{N}...', 'info')

        # Activar nuevo hilo
        self._hilo_superficie_activo = True
        token = True   # copia local del flag en el momento del lanzamiento

        def _worker():
            # El hilo verifica el flag antes de cada operación costosa.
            # Si llega otro _redibujar_todo, _hilo_superficie_activo se vuelve
            # False y este hilo aborta silenciosamente.
            if not self._hilo_superficie_activo:
                return
            # Superficies 3D usando sistema MTC (Manual RD N 08-2014-MTC/14)
            if _MTC_DISPONIBLE:
                _np = __import__('numpy')
                diam_arr = _np.linspace(0, 100, N)   # 0-100 cm MTC
                prof_arr = _np.linspace(0, 150, N)   # 0-150 mm MTC
                DN, PN = _np.meshgrid(diam_arr, prof_arr, indexing='ij')
                Z_urg = _np.zeros((N, N))
                Z_rec = _np.zeros((N, N))
                for _i in range(N):
                    for _j in range(N):
                        _dn = DN[_i,_j] / 100.0
                        _pn = PN[_i,_j] / 150.0
                        _u, _r = _mtc_superficie(_dn, _pn)
                        Z_urg[_i,_j] = _u
                        Z_rec[_i,_j] = _r
                resultado = (DN, PN, Z_urg, Z_rec)
            else:
                resultado = construir_superficies_fuzzy(N)
            if not self._hilo_superficie_activo:
                return   # fue superado por otro hilo más reciente
            self.superficies_cache = resultado
            self.root.after(0, self._dibujar_superficies_fuzzy)
            self.root.after(0, self._dibujar_pothole)

        self._hilo_superficie = threading.Thread(target=_worker, daemon=True)
        self._hilo_superficie.start()

    def _cambiar_paleta(self, cmap):
        """Cambia la paleta y redibuja. Solo lanza 1 hilo aunque se pulse varias veces."""
        self.var_paleta.set(cmap)
        # Invalidar hilo anterior y lanzar uno nuevo
        self._hilo_superficie_activo = False
        self.root.after(50, self._redibujar_todo)

    def _marcar_paso(self, idx, estado):
        """Actualiza el indicador visual de un paso del pipeline."""
        if idx >= len(self.steps_labels):
            return
        dot, lbl = self.steps_labels[idx]
        if estado == 'active':
            dot.config(text='◉', fg=COLOR_CYAN)
            lbl.config(fg=COLOR_CYAN)
        elif estado == 'done':
            dot.config(text='✓', fg=COLOR_LIME)
            lbl.config(fg=COLOR_TEXT_2)
        else:
            dot.config(text='○', fg=COLOR_TEXT_2)
            lbl.config(fg=COLOR_TEXT_2)

    def _resetear_pasos(self):
        for i in range(len(self.steps_labels)):
            self._marcar_paso(i, '')

    def _log(self, mensaje, tipo='info'):
        """Escribe una línea en el log (hilo principal)."""
        self.txt_log.config(state='normal')
        import datetime
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        self.txt_log.insert('end', f'[{ts}] {mensaje}\n', tipo)
        self.txt_log.config(state='disabled')
        self.txt_log.see('end')

    def _log_hilo(self, mensaje, tipo='info'):
        """Versión thread-safe del log (para usarse desde hilos)."""
        self.root.after(0, self._log, mensaje, tipo)

    def _limpiar_log(self):
        self.txt_log.config(state='normal')
        self.txt_log.delete('1.0', 'end')
        self.txt_log.config(state='disabled')

    def _seleccionar_imagen(self):
        ruta = filedialog.askopenfilename(
            title='Seleccionar imagen de pavimento',
            filetypes=[('Imágenes', '*.jpg *.jpeg *.png *.bmp *.webp'),
                       ('Todos', '*.*')])
        if ruta:
            nombre = os.path.basename(ruta)
            self.var_imagen.set(ruta)
            self.lbl_imagen.config(text=nombre)
            self._log(f'Imagen seleccionada: {nombre}', 'ok')

            # ── Auto-detectar el .npy REAL de profundidad correspondiente ─────
            # El script de captura de campo guarda tríos con el mismo timestamp:
            #   color_20260718_103245.png / depth_20260718_103245.png /
            #   depth_20260718_103245.npy
            # Ya no hay nube de puntos simulada como respaldo: si no se
            # encuentra el .npy real, la profundidad simplemente no estará
            # disponible (se avisa claramente, no se inventa nada).
            self.var_depth_npy.set('')
            carpeta   = os.path.dirname(ruta)
            base, _   = os.path.splitext(nombre)
            if base.startswith('color_'):
                candidato = os.path.join(
                    carpeta, base.replace('color_', 'depth_', 1) + '.npy')
                if os.path.isfile(candidato):
                    self.var_depth_npy.set(candidato)
                    self._log(f'  ↳ Profundidad real detectada: '
                              f'{os.path.basename(candidato)}', 'ok')
                else:
                    self._log('  ↳ Sin .npy de profundidad real asociado — '
                              'no se podrá calcular gravedad MTC, solo detección '
                              'y diámetro.', 'warn')
            else:
                self._log('  ↳ Nombre no sigue el patrón color_*.png — '
                          'sin profundidad real disponible.', 'warn')

    def _cargar_ejemplo(self):
        """Carga valores de referencia MTC — Gravedad 2 típico."""
        self.var_diametro.set(35.0)
        self.var_profundidad.set(40.0)
        self.lbl_imagen.config(text='Caso MTC · Gravedad 2 · Tabla 4-8')
        self._log('Ejemplo MTC cargado: Gravedad 2 (D=35cm, P=40mm)', 'ok')
        self._log('Referencia: Tabla 4-8, Falla 7 — Manual MTC', 'info')
        self._ejecutar_pipeline()

    def _aplicar_estilos_ttk(self):
        """Estilos globales para widgets ttk."""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Vertical.TScrollbar',
                        background=COLOR_BG3, troughcolor=COLOR_BG,
                        arrowcolor=COLOR_TEXT_2, borderwidth=0)
        style.configure('TProgressbar',
                        background=COLOR_CYAN,
                        troughcolor=COLOR_BG3, borderwidth=0)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

def main():
    """Crea la ventana principal y lanza el loop de eventos."""
    root = tk.Tk()

    # Centrar ventana en pantalla
    ancho, alto = 1400, 860
    root.update_idletasks()
    x = (root.winfo_screenwidth()  - ancho) // 2
    y = (root.winfo_screenheight() - alto)  // 2
    root.geometry(f'{ancho}x{alto}+{x}+{y}')

    # Configurar tema oscuro en macOS
    try:
        root.tk.call('tk', 'scaling', 1.2)
    except Exception:
        pass

    app = PaveScanGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
