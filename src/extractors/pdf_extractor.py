"""
Extractor de texto nativo de PDFs

Extrae texto y coordenadas directamente del PDF cuando este contiene
texto embebido (no es una imagen escaneada).
"""
import logging
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from pathlib import Path

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extractor de texto nativo de PDFs

    Utiliza PyMuPDF para extraer texto con coordenadas directamente
    del PDF cuando este contiene información de texto embebida.
    """

    def __init__(self):
        """Inicializa el extractor"""
        self.bbox_scale = Config.BBOX_SCALE

    def is_native_pdf(self, pdf_path: str, page_num: int = 0) -> bool:
        """
        Detecta si el PDF contiene texto nativo o es imagen escaneada

        Args:
            pdf_path: Ruta al archivo PDF
            page_num: Número de página a analizar

        Returns:
            True si es PDF nativo, False si es imagen escaneada
        """
        try:
            doc = fitz.open(pdf_path)

            if page_num >= len(doc):
                page_num = 0

            page = doc[page_num]

            # Extraer texto
            text = page.get_text()

            # Contar palabras extraídas
            word_count = len(text.split())

            doc.close()

            # Si tiene más de 10 palabras, consideramos que es nativo
            is_native = word_count > 10

            logger.debug(
                f"PDF {'nativo' if is_native else 'escaneado'} "
                f"({word_count} palabras en página {page_num})"
            )

            return is_native

        except Exception as e:
            logger.error(f"Error al detectar tipo de PDF: {e}")
            return False

    def extract(self, pdf_path: str, page_num: int = 0) -> List[Dict]:
        """
        Extrae palabras con coordenadas de una página

        Args:
            pdf_path: Ruta al archivo PDF
            page_num: Número de página a extraer

        Returns:
            Lista de diccionarios con formato:
            [
                {
                    "text": "palabra",
                    "bbox": [x0, y0, x1, y1],  # Normalizado 0-1000
                    "confidence": 1.0,  # PDFs nativos tienen confianza máxima
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
            doc = fitz.open(pdf_path)

            if len(doc) == 0:
                raise ValueError(f"El PDF no tiene páginas: {pdf_path}")

            if page_num >= len(doc):
                raise ValueError(
                    f"Página {page_num} no existe (total: {len(doc)} páginas)"
                )

            page = doc[page_num]

            # Extraer palabras con coordenadas
            words = page.get_text("words")  # Retorna (x0, y0, x1, y1, "palabra", block_no, line_no, word_no)

            # Obtener dimensiones de la página para normalización
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            # Formatear palabras
            result = []
            for word in words:
                x0, y0, x1, y1 = word[:4]
                text = word[4]

                # Normalizar coordenadas a escala 0-1000
                bbox = self._normalize_bbox(
                    [x0, y0, x1, y1],
                    page_width,
                    page_height
                )

                result.append({
                    "text": text,
                    "bbox": bbox,
                    "confidence": 1.0,  # PDF nativo tiene confianza máxima
                    "page": page_num
                })

            doc.close()

            logger.debug(f"Extraídas {len(result)} palabras de página {page_num}")
            return result

        except Exception as e:
            logger.error(f"Error al extraer texto de {pdf_path}: {e}")
            raise

    def extract_all_pages(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """
        Extrae todas las páginas del PDF

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

            logger.info(f"Extraídas {num_pages} páginas de {pdf_path.name}")
            return result

        except Exception as e:
            logger.error(f"Error al extraer todas las páginas de {pdf_path}: {e}")
            raise

    def _normalize_bbox(
        self,
        bbox: List[float],
        page_width: float,
        page_height: float
    ) -> List[int]:
        """
        Normaliza coordenadas a escala 0-1000 (estándar LayoutLMv3)

        Args:
            bbox: Coordenadas originales [x0, y0, x1, y1]
            page_width: Ancho de la página
            page_height: Alto de la página

        Returns:
            Coordenadas normalizadas [x0, y0, x1, y1]
        """
        x0, y0, x1, y1 = bbox

        # Normalizar a escala 0-1000
        x0_norm = int((x0 / page_width) * self.bbox_scale)
        y0_norm = int((y0 / page_height) * self.bbox_scale)
        x1_norm = int((x1 / page_width) * self.bbox_scale)
        y1_norm = int((y1 / page_height) * self.bbox_scale)

        # Asegurar que las coordenadas estén en el rango válido
        x0_norm = max(0, min(self.bbox_scale, x0_norm))
        y0_norm = max(0, min(self.bbox_scale, y0_norm))
        x1_norm = max(0, min(self.bbox_scale, x1_norm))
        y1_norm = max(0, min(self.bbox_scale, y1_norm))

        return [x0_norm, y0_norm, x1_norm, y1_norm]

    def get_page_count(self, pdf_path: str) -> int:
        """
        Obtiene el número de páginas del PDF

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Número de páginas
        """
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
            return num_pages
        except Exception as e:
            logger.error(f"Error al obtener número de páginas de {pdf_path}: {e}")
            return 0
