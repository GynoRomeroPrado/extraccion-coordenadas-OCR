"""
Extractor OCR usando PaddleOCR

Extrae texto y coordenadas de PDFs escaneados (imágenes)
utilizando PaddleOCR (más rápido y preciso que Tesseract).
"""
import logging
from typing import List, Dict
from paddleocr import PaddleOCR
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path
import numpy as np

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    Extractor OCR usando PaddleOCR

    Convierte páginas PDF a imágenes y aplica OCR para extraer
    texto con coordenadas y confianza.

    PaddleOCR ventajas sobre Tesseract:
    - Mejor soporte para español
    - Mayor precisión en facturas
    - No requiere archivos de datos externos
    - Más rápido en CPU y GPU
    """

    def __init__(self, dpi: int = 300, use_gpu: bool = False):
        """
        Args:
            dpi: Resolución para conversión de PDF a imagen
            use_gpu: Usar GPU para acelerar OCR (requiere paddlepaddle-gpu)
        """
        self.dpi = dpi
        self.bbox_scale = Config.BBOX_SCALE
        self.use_gpu = use_gpu

        # Inicializar PaddleOCR
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Detectar orientación de texto
                lang='es',           # Español
                use_gpu=use_gpu,
                show_log=False       # No mostrar logs de PaddleOCR
            )
            logger.info(f"PaddleOCR inicializado (GPU: {use_gpu})")
        except Exception as e:
            logger.error(f"Error al inicializar PaddleOCR: {e}")
            raise

    def extract(self, pdf_path: str, page_num: int = 0) -> List[Dict]:
        """
        Extrae palabras con coordenadas usando PaddleOCR

        Args:
            pdf_path: Ruta al archivo PDF
            page_num: Número de página a extraer

        Returns:
            Lista de diccionarios con formato:
            [
                {
                    "text": "palabra",
                    "bbox": [x0, y0, x1, y1],  # Normalizado 0-1000
                    "confidence": 0.95,
                    "page": 0
                },
                ...
            ]

        Raises:
            FileNotFoundError: Si el PDF no existe
            ValueError: Si la página no existe
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        try:
            # Convertir página a imagen
            image = self._pdf_page_to_image(pdf_path, page_num)

            # Obtener dimensiones de la imagen
            img_width, img_height = image.size

            # Convertir PIL Image a numpy array (requerido por PaddleOCR)
            img_array = np.array(image)

            # Aplicar OCR con PaddleOCR
            ocr_result = self.ocr.ocr(img_array, cls=True)

            # Formatear palabras
            result = []

            if ocr_result is None or len(ocr_result) == 0:
                logger.warning(f"PaddleOCR no encontró texto en página {page_num}")
                return result

            # ocr_result[0] es una lista de líneas detectadas
            for line in ocr_result[0]:
                if line is None:
                    continue

                # Cada línea es: [bbox_poligonal, (text, confidence)]
                bbox_poly = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (texto, confidence)

                text = text_info[0].strip()
                confidence = float(text_info[1])

                if not text:
                    continue

                # Convertir bbox poligonal a rectangular
                # Extraer todas las coordenadas X e Y
                x_coords = [point[0] for point in bbox_poly]
                y_coords = [point[1] for point in bbox_poly]

                # Bbox rectangular: [x_min, y_min, x_max, y_max]
                x0 = min(x_coords)
                y0 = min(y_coords)
                x1 = max(x_coords)
                y1 = max(y_coords)

                # Normalizar coordenadas a escala 0-1000
                bbox = self._normalize_bbox(
                    [x0, y0, x1, y1],
                    img_width,
                    img_height
                )

                result.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": confidence,
                    "page": page_num
                })

            logger.debug(f"Extraídas {len(result)} palabras de página {page_num} (PaddleOCR)")
            return result

        except Exception as e:
            logger.error(f"Error al extraer con PaddleOCR de {pdf_path}: {e}")
            raise

    def extract_all_pages(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """
        Extrae todas las páginas del PDF usando OCR

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Diccionario {página: [palabras], ...}
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()

            result = {}
            for page_num in range(num_pages):
                words = self.extract(pdf_path, page_num)
                result[page_num] = words

            logger.info(f"Extraídas {num_pages} páginas de {pdf_path.name} (OCR)")
            return result

        except Exception as e:
            logger.error(f"Error al extraer todas las páginas con OCR de {pdf_path}: {e}")
            raise

    def _pdf_page_to_image(self, pdf_path: str, page_num: int) -> Image.Image:
        """
        Convierte una página de PDF a imagen PIL

        Args:
            pdf_path: Ruta al archivo PDF
            page_num: Número de página

        Returns:
            Imagen PIL
        """
        try:
            doc = fitz.open(pdf_path)

            if page_num >= len(doc):
                raise ValueError(
                    f"Página {page_num} no existe (total: {len(doc)} páginas)"
                )

            page = doc[page_num]

            # Convertir a imagen con DPI especificado
            # Calcular zoom para DPI deseado (72 DPI es el default)
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convertir a PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            doc.close()

            return img

        except Exception as e:
            logger.error(f"Error al convertir página {page_num} a imagen: {e}")
            raise

    def _normalize_bbox(
        self,
        bbox: List[float],
        img_width: float,
        img_height: float
    ) -> List[int]:
        """
        Normaliza coordenadas a escala 0-1000 (estándar LayoutLMv3)

        Args:
            bbox: Coordenadas originales [x0, y0, x1, y1]
            img_width: Ancho de la imagen
            img_height: Alto de la imagen

        Returns:
            Coordenadas normalizadas [x0, y0, x1, y1]
        """
        x0, y0, x1, y1 = bbox

        # Normalizar a escala 0-1000
        x0_norm = int((x0 / img_width) * self.bbox_scale)
        y0_norm = int((y0 / img_height) * self.bbox_scale)
        x1_norm = int((x1 / img_width) * self.bbox_scale)
        y1_norm = int((y1 / img_height) * self.bbox_scale)

        # Asegurar que las coordenadas estén en el rango válido
        x0_norm = max(0, min(self.bbox_scale, x0_norm))
        y0_norm = max(0, min(self.bbox_scale, y0_norm))
        x1_norm = max(0, min(self.bbox_scale, x1_norm))
        y1_norm = max(0, min(self.bbox_scale, y1_norm))

        return [x0_norm, y0_norm, x1_norm, y1_norm]
