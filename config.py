"""
Configuración central para el procesador de facturas LayoutLMv3
"""
import os
from pathlib import Path


class Config:
    """Configuración global del sistema"""

    # ==================== RUTAS ====================
    PROJECT_ROOT = Path(__file__).parent
    OUTPUT_DIR = PROJECT_ROOT / "output"

    # ==================== OCR y EXTRACCIÓN ====================
    # DPI para conversión de PDF a imagen
    DPI = 300

    # NOTA: Ahora se usa PaddleOCR en lugar de Tesseract
    # PaddleOCR configura idioma automáticamente en el constructor
    # Estas configuraciones se mantienen para compatibilidad legacy
    TESSERACT_LANG = "spa"  # Español (legacy - no usado)
    TESSERACT_CONFIG = "--oem 3 --psm 6"  # legacy - no usado

    # ==================== MATCHING ====================
    # Umbral mínimo de similitud para fuzzy matching (0-100)
    MATCH_THRESHOLD = 85

    # Umbral de confianza mínimo para considerar un match válido
    CONFIDENCE_THRESHOLD = 0.75

    # ==================== DETECCIÓN DE TABLA ====================
    # Tolerancia vertical para agrupar palabras en la misma fila (píxeles)
    ROW_TOLERANCE = 10

    # Tolerancia horizontal para agrupar palabras en la misma columna (píxeles)
    COL_TOLERANCE = 15

    # ==================== TOKENIZACIÓN ====================
    # Máximo de tokens por chunk (LayoutLMv3 acepta 512, dejamos margen)
    MAX_TOKENS = 384

    # Overlap entre chunks (tokens)
    TOKEN_OVERLAP = 128

    # Estimación de tokens por palabra (promedio)
    TOKENS_PER_WORD = 2

    # ==================== COORDENADAS ====================
    # Escala para normalización de coordenadas (estándar LayoutLMv3)
    BBOX_SCALE = 1000

    # ==================== ORIENTACIÓN ====================
    # Ángulos de rotación posibles
    ROTATION_ANGLES = [0, 90, 180, 270]

    # Número de muestras de página para detectar orientación (si doc tiene muchas páginas)
    ORIENTATION_SAMPLE_SIZE = 1

    # ==================== PROCESAMIENTO ====================
    # Número de workers para procesamiento paralelo
    NUM_WORKERS = 4

    # Intervalo de checkpoint (guardar progreso cada N documentos)
    CHECKPOINT_INTERVAL = 1000

    # ==================== VALIDACIÓN ====================
    # Match rate mínimo aceptable (0.0 - 1.0)
    MIN_MATCH_RATE = 0.85

    # Tasa de éxito mínima esperada del procesamiento (0.0 - 1.0)
    MIN_SUCCESS_RATE = 0.95

    # ==================== LOGGING ====================
    # Nivel de logging
    LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Formato de logging
    LOG_FORMAT = "[%(levelname)s] %(message)s"

    # ==================== CAMPOS DE FACTURA ====================
    # Campos del header (nivel documento)
    HEADER_FIELDS = [
        "tipo_documento",
        "serie_completa",
        "fecha_emision",
        "fecha_vencimiento",
        "emisor_ruc",
        "emisor_razon_social",
        "emisor_nombre_comercial",
        "emisor_domicilio_fiscal",
        "emisor_direccion",
        "emisor_urbanizacion",
        "emisor_departamento",
        "emisor_provincia",
        "emisor_distrito",
        "emisor_ubigeo",
        "emisor_pais",
        "emisor_codigo_local",
        "emisor_telefono",
        "emisor_email",
        "receptor_tipo_doc",
        "receptor_numero_doc",
        "receptor_razon_social",
        "receptor_nombre_comercial",
        "receptor_direccion",
        "receptor_departamento",
        "receptor_provincia",
        "receptor_distrito",
        "receptor_ubigeo",
        "receptor_pais",
        "receptor_email",
        "moneda",
        "tipo_cambio",
        "subtotal",
        "descuento_global",
        "recargo_global",
        "total_operaciones_gravadas",
        "total_operaciones_inafectas",
        "total_operaciones_exoneradas",
        "total_operaciones_gratuitas",
        "sumatoria_igv",
        "sumatoria_isc",
        "sumatoria_otros_tributos",
        "total_descuentos",
        "total_cargos",
        "redondeo",
        "igv",
        "icbper",
        "importe_total",
        "valor_venta",
        "precio_venta",
        "anticipo_aplicado",
        "total_percepcion",
        "total_retencion",
        "total_detraccion",
        "monto_neto_pendiente_pago",
        "forma_pago",
        "medio_pago",
        "condicion_pago",
        "numero_cuenta_bancaria",
        "banco",
        "codigo_cuenta_interbancaria",
        "moneda_cuenta",
        "detraccion_aplicable",
        "detraccion_porcentaje",
        "detraccion_monto",
        "detraccion_numero_cuenta",
        "orden_compra",
        "guia_remision",
        "observaciones",
        "nota_adicional",
        "terminos_condiciones",
        "regimen_percepcion",
        "tasa_percepcion",
        "monto_percepcion",
        "regimen_retencion",
        "tasa_retencion",
        "monto_retencion",
        "tipo_operacion",
        "codigo_operacion",
        "motivo_traslado",
        "modalidad_traslado",
        "fecha_inicio_traslado",
        "peso_bruto_total",
        "unidad_medida_peso",
        "numero_bultos",
        "tipo_documento_relacionado",
        "numero_documento_relacionado",
        "motivo_nota",
        "sustento_nota",
        "exportacion_codigo_tipo",
        "exportacion_valor_fob",
        "exportacion_valor_flete",
        "exportacion_valor_seguro",
        "exportacion_gastos",
        "exportacion_otros_cargos",
        "hash_cpe",
        "firma_digital",
        "codigo_qr",
        "representacion_impresa"
    ]

    # Campos de items (nivel línea)
    ITEM_FIELDS = [
        "item",
        "codigo",
        "codigo_producto_sunat",
        "descripcion",
        "cantidad",
        "unidad_medida",
        "precio_unitario",
        "valor_unitario",
        "descuento",
        "recargo",
        "valor_venta",
        "igv",
        "isc",
        "icbper",
        "otros_tributos",
        "precio_venta_unitario",
        "afectacion_igv",
        "tipo_precio"
    ]

    # ==================== MÉTODOS AUXILIARES ====================
    @classmethod
    def ensure_output_dir(cls):
        """Crea el directorio de salida si no existe"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_output_path(cls, filename: str) -> Path:
        """Retorna la ruta completa de un archivo de salida"""
        cls.ensure_output_dir()
        return cls.OUTPUT_DIR / filename
