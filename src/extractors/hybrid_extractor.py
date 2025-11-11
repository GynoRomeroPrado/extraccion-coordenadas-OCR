"""
Extractor híbrido que combina extracción nativa + OCR

Estrategia inteligente:
1. Detectar tipo de PDF (nativo vs rasterizado)
2. Si es nativo: usar PyMuPDF
3. Si es rasterizado: usar Tesseract OCR
4. Retornar palabras con coordenadas normalizadas
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path

from .pdf_extractor import PDFExtractor
from .ocr_extractor import OCRExtractor
from ..preprocessor.orientation_detector import OrientationDetector
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class HybridExtractor:
    """
    Extractor híbrido que selecciona automáticamente el mejor método

    Combina extracción nativa (PyMuPDF) y OCR (Tesseract) según el
    tipo de PDF detectado.
    """

    def __init__(self, dpi: int = 300, auto_rotate: bool = True):
        """
        Args:
            dpi: Resolución para conversión de PDF a imagen (si se usa OCR)
            auto_rotate: Si True, detecta y corrige orientación automáticamente
        """
        self.dpi = dpi
        self.auto_rotate = auto_rotate

        # Inicializar extractores
        self.pdf_extractor = PDFExtractor()
        self.ocr_extractor = OCRExtractor(dpi=dpi)
        self.orientation_detector = OrientationDetector(dpi=dpi)

    def extract(
        self,
        pdf_path: str,
        page_num: int = 0,
        force_method: Optional[str] = None
    ) -> List[Dict]:
        """
        Extrae palabras con coordenadas de una página

        Args:
            pdf_path: Ruta al archivo PDF
            page_num: Número de página a extraer
            force_method: Forzar método ('native' o 'ocr'), None para auto

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
            ValueError: Si force_method es inválido
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        # Validar force_method
        if force_method not in [None, 'native', 'ocr']:
            raise ValueError(f"force_method inválido: {force_method}")

        try:
            # Detectar y corregir orientación si está habilitado
            working_pdf = str(pdf_path)
            rotation_angle = 0

            if self.auto_rotate:
                rotation_angle = self.orientation_detector.detect(working_pdf, page_num)
                if rotation_angle != 0:
                    logger.info(f"Rotando PDF {rotation_angle}° para corregir orientación")
                    working_pdf = self.orientation_detector.rotate_if_needed(
                        working_pdf,
                        rotation_angle
                    )

            # Determinar método de extracción
            if force_method == 'native':
                method = 'native'
            elif force_method == 'ocr':
                method = 'ocr'
            else:
                # Auto-detectar
                is_native = self.pdf_extractor.is_native_pdf(working_pdf, page_num)
                method = 'native' if is_native else 'ocr'

            logger.info(f"Usando método: {method}")

            # Extraer según método
            if method == 'native':
                words = self.pdf_extractor.extract(working_pdf, page_num)
            else:
                words = self.ocr_extractor.extract(working_pdf, page_num)

            # Agregar metadatos
            for word in words:
                word['extraction_method'] = method
                word['rotation_corrected'] = rotation_angle

            logger.info(
                f"Extraídas {len(words)} palabras de {pdf_path.name} "
                f"(página {page_num}, método: {method})"
            )

            return words

        except Exception as e:
            logger.error(f"Error al extraer de {pdf_path}: {e}")
            raise

    def extract_all_pages(
        self,
        pdf_path: str,
        force_method: Optional[str] = None
    ) -> Dict[int, List[Dict]]:
        """
        Extrae todas las páginas del PDF

        Args:
            pdf_path: Ruta al archivo PDF
            force_method: Forzar método ('native' o 'ocr'), None para auto

        Returns:
            Diccionario {página: [palabras], ...}
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        try:
            # Obtener número de páginas
            num_pages = self.pdf_extractor.get_page_count(str(pdf_path))

            # Detectar orientación en primera página si auto_rotate está habilitado
            working_pdf = str(pdf_path)
            rotation_angle = 0

            if self.auto_rotate:
                rotation_angle = self.orientation_detector.detect(working_pdf, 0)
                if rotation_angle != 0:
                    logger.info(f"Rotando PDF {rotation_angle}° para corregir orientación")
                    working_pdf = self.orientation_detector.rotate_if_needed(
                        working_pdf,
                        rotation_angle
                    )

            # Determinar método de extracción (en primera página)
            if force_method == 'native':
                method = 'native'
            elif force_method == 'ocr':
                method = 'ocr'
            else:
                # Auto-detectar
                is_native = self.pdf_extractor.is_native_pdf(working_pdf, 0)
                method = 'native' if is_native else 'ocr'

            logger.info(f"Usando método: {method} para {num_pages} páginas")

            # Extraer todas las páginas
            result = {}
            for page_num in range(num_pages):
                if method == 'native':
                    words = self.pdf_extractor.extract(working_pdf, page_num)
                else:
                    words = self.ocr_extractor.extract(working_pdf, page_num)

                # Agregar metadatos
                for word in words:
                    word['extraction_method'] = method
                    word['rotation_corrected'] = rotation_angle

                result[page_num] = words

                logger.debug(
                    f"Página {page_num + 1}/{num_pages}: {len(words)} palabras extraídas"
                )

            logger.info(
                f"Extraídas {num_pages} páginas de {pdf_path.name} (método: {method})"
            )

            return result

        except Exception as e:
            logger.error(f"Error al extraer todas las páginas de {pdf_path}: {e}")
            raise

    def get_extraction_stats(
        self,
        pdf_path: str,
        words_by_page: Dict[int, List[Dict]]
    ) -> Dict:
        """
        Calcula estadísticas de extracción

        Args:
            pdf_path: Ruta al PDF
            words_by_page: Resultado de extract_all_pages

        Returns:
            Diccionario con estadísticas:
            {
                "num_pages": 3,
                "total_words": 1234,
                "words_per_page": [450, 400, 384],
                "avg_confidence": 0.92,
                "method": "native",
                "rotation_angle": 0
            }
        """
        total_words = sum(len(words) for words in words_by_page.values())
        words_per_page = [len(words) for words in words_by_page.values()]

        # Calcular confianza promedio
        all_confidences = []
        for words in words_by_page.values():
            all_confidences.extend([w['confidence'] for w in words])

        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

        # Obtener método y rotación (del primer word)
        method = "unknown"
        rotation_angle = 0
        if words_by_page:
            first_page_words = words_by_page[0]
            if first_page_words:
                method = first_page_words[0].get('extraction_method', 'unknown')
                rotation_angle = first_page_words[0].get('rotation_corrected', 0)

        return {
            "pdf_name": Path(pdf_path).name,
            "num_pages": len(words_by_page),
            "total_words": total_words,
            "words_per_page": words_per_page,
            "avg_words_per_page": sum(words_per_page) / len(words_per_page) if words_per_page else 0,
            "avg_confidence": round(avg_confidence, 3),
            "method": method,
            "rotation_angle": rotation_angle
        }
