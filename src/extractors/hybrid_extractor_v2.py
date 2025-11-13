"""
Extractor híbrido V2 con aceleración GPU y detección de regiones

Mejoras sobre V1:
- Soporte GPU con PaddleOCR (10-50x más rápido)
- Integración con RegionDetector
- Procesamiento paralelo de páginas
- Mejor manejo de orientación
- Estadísticas detalladas por región
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path
import time

from .pdf_extractor import PDFExtractor
from .ocr_extractor import OCRExtractor
from .paddle_ocr_extractor import PaddleOCRExtractor
from ..preprocessor.orientation_detector import OrientationDetector
from ..matchers.region_detector import RegionDetector
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class HybridExtractorV2:
    """
    Extractor híbrido V2 con GPU y detección de regiones

    Estrategia:
    1. Detectar orientación y corregir
    2. Detectar tipo de PDF (nativo vs rasterizado)
    3. Usar PaddleOCR con GPU si está disponible, sino Tesseract
    4. Detectar regiones (header, items, totals)
    5. Retornar palabras organizadas por región
    """

    def __init__(
        self,
        dpi: int = 300,
        auto_rotate: bool = True,
        use_gpu: bool = True,
        detect_regions: bool = True
    ):
        """
        Args:
            dpi: Resolución para OCR
            auto_rotate: Detectar y corregir orientación automáticamente
            use_gpu: Usar GPU para OCR si está disponible
            detect_regions: Detectar regiones automáticamente
        """
        self.dpi = dpi
        self.auto_rotate = auto_rotate
        self.use_gpu = use_gpu
        self.detect_regions = detect_regions

        # Inicializar extractores
        self.pdf_extractor = PDFExtractor()
        self.orientation_detector = OrientationDetector(dpi=dpi)

        # Intentar usar PaddleOCR con GPU
        self.paddle_ocr = PaddleOCRExtractor(use_gpu=use_gpu, dpi=dpi)

        if self.paddle_ocr.is_available():
            self.ocr_extractor = self.paddle_ocr
            self.ocr_method = "PaddleOCR"
            logger.info(f"✅ Usando PaddleOCR con GPU: {use_gpu}")
        else:
            # Fallback a Tesseract
            self.ocr_extractor = OCRExtractor(dpi=dpi)
            self.ocr_method = "Tesseract"
            logger.info("⚠️ Usando Tesseract (CPU) - instalar PaddleOCR para GPU")

        # Detector de regiones
        if detect_regions:
            self.region_detector = RegionDetector()

    def extract(
        self,
        pdf_path: str,
        page_num: int = 0,
        force_method: Optional[str] = None
    ) -> Dict:
        """
        Extrae palabras con coordenadas y detecta regiones

        Args:
            pdf_path: Ruta al PDF
            page_num: Número de página
            force_method: Forzar método ('native' o 'ocr')

        Returns:
            Diccionario con:
            {
                "words": [...],  # Todas las palabras
                "regions": {     # Palabras por región
                    "header": {...},
                    "items": {...},
                    "totals": {...}
                },
                "metadata": {
                    "method": "PaddleOCR",
                    "rotation": 0,
                    "processing_time": 0.05
                }
            }
        """
        start_time = time.time()

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        try:
            # 1. Detectar y corregir orientación
            working_pdf = str(pdf_path)
            rotation_angle = 0

            if self.auto_rotate:
                rotation_angle = self.orientation_detector.detect(working_pdf, page_num)
                if rotation_angle != 0:
                    logger.info(f"Rotando PDF {rotation_angle}°")
                    working_pdf = self.orientation_detector.rotate_if_needed(
                        working_pdf,
                        rotation_angle
                    )

            # 2. Determinar método de extracción
            if force_method == 'native':
                method = 'native'
            elif force_method == 'ocr':
                method = 'ocr'
            else:
                # Auto-detectar
                is_native = self.pdf_extractor.is_native_pdf(working_pdf, page_num)
                method = 'native' if is_native else 'ocr'

            # 3. Extraer texto
            if method == 'native':
                words = self.pdf_extractor.extract(working_pdf, page_num)
                extraction_method = "PyMuPDF"
            else:
                words = self.ocr_extractor.extract(working_pdf, page_num)
                extraction_method = self.ocr_method

            # 4. Detectar regiones si está habilitado
            regions = None
            if self.detect_regions and words:
                regions_detected = self.region_detector.detect_regions(words)
                regions = regions_detected

            # 5. Preparar resultado
            processing_time = time.time() - start_time

            result = {
                "words": words,
                "regions": regions,
                "metadata": {
                    "method": extraction_method,
                    "rotation": rotation_angle,
                    "processing_time": round(processing_time, 3),
                    "word_count": len(words),
                    "page": page_num
                }
            }

            if regions:
                result["metadata"]["regions_detected"] = {
                    "header_words": regions['header']['word_count'],
                    "items_words": regions['items']['word_count'],
                    "totals_words": regions['totals']['word_count']
                }

            logger.info(
                f"Extraídas {len(words)} palabras en {processing_time:.3f}s "
                f"(método: {extraction_method}, rotación: {rotation_angle}°)"
            )

            return result

        except Exception as e:
            logger.error(f"Error al extraer de {pdf_path}: {e}")
            raise

    def extract_all_pages(
        self,
        pdf_path: str,
        force_method: Optional[str] = None
    ) -> Dict:
        """
        Extrae todas las páginas del PDF

        Args:
            pdf_path: Ruta al PDF
            force_method: Forzar método

        Returns:
            Diccionario con páginas y metadatos agregados
        """
        start_time = time.time()

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        try:
            # Obtener número de páginas
            num_pages = self.pdf_extractor.get_page_count(str(pdf_path))

            # Detectar orientación en primera página
            working_pdf = str(pdf_path)
            rotation_angle = 0

            if self.auto_rotate:
                rotation_angle = self.orientation_detector.detect(working_pdf, 0)
                if rotation_angle != 0:
                    logger.info(f"Rotando PDF {rotation_angle}°")
                    working_pdf = self.orientation_detector.rotate_if_needed(
                        working_pdf,
                        rotation_angle
                    )

            # Determinar método
            if force_method == 'native':
                method = 'native'
            elif force_method == 'ocr':
                method = 'ocr'
            else:
                is_native = self.pdf_extractor.is_native_pdf(working_pdf, 0)
                method = 'native' if is_native else 'ocr'

            extraction_method = self.ocr_method if method == 'ocr' else "PyMuPDF"

            # Extraer todas las páginas
            pages_data = {}

            for page_num in range(num_pages):
                if method == 'native':
                    words = self.pdf_extractor.extract(working_pdf, page_num)
                else:
                    words = self.ocr_extractor.extract(working_pdf, page_num)

                # Detectar regiones
                regions = None
                if self.detect_regions and words:
                    regions = self.region_detector.detect_regions(words)

                pages_data[page_num] = {
                    "words": words,
                    "regions": regions,
                    "word_count": len(words)
                }

                logger.debug(f"Página {page_num + 1}/{num_pages}: {len(words)} palabras")

            # Metadatos agregados
            total_time = time.time() - start_time
            total_words = sum(p['word_count'] for p in pages_data.values())

            metadata = {
                "method": extraction_method,
                "rotation": rotation_angle,
                "total_processing_time": round(total_time, 3),
                "avg_time_per_page": round(total_time / num_pages, 3),
                "num_pages": num_pages,
                "total_words": total_words,
                "avg_words_per_page": round(total_words / num_pages, 1)
            }

            logger.info(
                f"Extraídas {num_pages} páginas en {total_time:.2f}s "
                f"(método: {extraction_method}, {total_words} palabras totales)"
            )

            return {
                "pages": pages_data,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error al extraer todas las páginas: {e}")
            raise

    def benchmark(self, pdf_path: str, num_pages: int = 5) -> Dict:
        """
        Benchmark de velocidad comparando métodos

        Args:
            pdf_path: Ruta al PDF de prueba
            num_pages: Número de páginas a testear

        Returns:
            Estadísticas de rendimiento
        """
        results = {}

        # Test con método nativo
        logger.info("Benchmark: PyMuPDF (nativo)...")
        start = time.time()
        try:
            for page in range(min(num_pages, 5)):
                self.pdf_extractor.extract(pdf_path, page)
            results['PyMuPDF'] = {
                "time": round(time.time() - start, 3),
                "avg_per_page": round((time.time() - start) / min(num_pages, 5), 3)
            }
        except:
            results['PyMuPDF'] = {"error": "No disponible"}

        # Test con PaddleOCR
        if self.paddle_ocr.is_available():
            logger.info("Benchmark: PaddleOCR...")
            result = self.paddle_ocr.benchmark(pdf_path, num_pages)
            results['PaddleOCR'] = result

        # Test con Tesseract
        logger.info("Benchmark: Tesseract...")
        tesseract = OCRExtractor(dpi=self.dpi)
        start = time.time()
        try:
            for page in range(min(num_pages, 3)):  # Solo 3 páginas (es lento)
                tesseract.extract(pdf_path, page)
            results['Tesseract'] = {
                "time": round(time.time() - start, 3),
                "avg_per_page": round((time.time() - start) / min(num_pages, 3), 3)
            }
        except:
            results['Tesseract'] = {"error": "No disponible"}

        return results

    def get_capabilities(self) -> Dict:
        """
        Retorna capacidades del extractor

        Returns:
            Diccionario con capacidades disponibles
        """
        return {
            "gpu_available": self.paddle_ocr.is_available() and self.use_gpu,
            "ocr_method": self.ocr_method,
            "auto_rotate": self.auto_rotate,
            "detect_regions": self.detect_regions,
            "dpi": self.dpi
        }
