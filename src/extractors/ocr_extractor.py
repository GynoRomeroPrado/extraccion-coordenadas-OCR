"""
Extractor OCR usando Tesseract

Extrae texto y coordenadas de PDFs escaneados (imágenes)
utilizando Tesseract OCR.
"""
import logging
from typing import List, Dict
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path
import pandas as pd

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    Extractor OCR usando Tesseract

    Convierte páginas PDF a imágenes y aplica OCR para extraer
    texto con coordenadas y confianza.
    """

    def __init__(self, dpi: int = 300):
        """
        Args:
            dpi: Resolución para conversión de PDF a imagen
        """
        self.dpi = dpi
        self.bbox_scale = Config.BBOX_SCALE
        self.tesseract_lang = Config.TESSERACT_LANG
        self.tesseract_config = Config.TESSERACT_CONFIG

    def extract(self, pdf_path: str, page_num: int = 0) -> List[Dict]:
        """
        Extrae palabras con coordenadas usando OCR

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

            # Aplicar OCR
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.tesseract_lang,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DATAFRAME
            )

            # Filtrar filas con texto
            ocr_data = ocr_data[ocr_data['text'].notna()]
            ocr_data = ocr_data[ocr_data['text'].str.strip() != '']

            # Obtener dimensiones de la imagen
            img_width, img_height = image.size

            # Formatear palabras
            result = []
            for _, row in ocr_data.iterrows():
                text = str(row['text']).strip()
                if not text:
                    continue

                # Coordenadas originales
                x0 = row['left']
                y0 = row['top']
                width = row['width']
                height = row['height']
                x1 = x0 + width
                y1 = y0 + height

                # Confianza
                confidence = row['conf'] / 100.0 if row['conf'] >= 0 else 0.0

                # Normalizar coordenadas
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

            logger.debug(f"Extraídas {len(result)} palabras de página {page_num} (OCR)")
            return result

        except Exception as e:
            logger.error(f"Error al extraer con OCR de {pdf_path}: {e}")
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
