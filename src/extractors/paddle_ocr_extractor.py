"""
Extractor OCR con aceleración GPU usando PaddleOCR

PaddleOCR es significativamente más rápido que Tesseract y soporta GPU.
Velocidad: ~10-50x más rápido con GPU vs Tesseract CPU
"""
import logging
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class PaddleOCRExtractor:
    """
    Extractor OCR con GPU usando PaddleOCR

    Ventajas sobre Tesseract:
    - 10-50x más rápido con GPU
    - Mejor precisión en español
    - Detección automática de orientación
    - Retorna coordenadas más precisas
    """

    def __init__(self, use_gpu: bool = True, dpi: int = 300):
        """
        Args:
            use_gpu: Usar GPU si está disponible (default: True)
            dpi: Resolución para conversión de PDF a imagen
        """
        self.dpi = dpi
        self.bbox_scale = Config.BBOX_SCALE
        self.use_gpu = use_gpu

        # Importar PaddleOCR (lazy import)
        try:
            from paddleocr import PaddleOCR

            # Inicializar PaddleOCR con GPU
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Detección de ángulo
                lang='es',  # Español
                use_gpu=use_gpu,
                show_log=False,
                det_model_dir=None,  # Usar modelo por defecto
                rec_model_dir=None,
                cls_model_dir=None
            )

            self.available = True
            logger.info(f"✅ PaddleOCR inicializado (GPU: {use_gpu})")

        except ImportError:
            logger.warning(
                "⚠️ PaddleOCR no está instalado. "
                "Instalar con: pip install paddlepaddle-gpu paddleocr"
            )
            self.ocr = None
            self.available = False

        except Exception as e:
            logger.warning(f"⚠️ Error al inicializar PaddleOCR: {e}")
            self.ocr = None
            self.available = False

    def is_available(self) -> bool:
        """Retorna True si PaddleOCR está disponible"""
        return self.available

    def extract(self, pdf_path: str, page_num: int = 0) -> List[Dict]:
        """
        Extrae palabras con coordenadas usando PaddleOCR con GPU

        Args:
            pdf_path: Ruta al archivo PDF
            page_num: Número de página a extraer

        Returns:
            Lista de diccionarios con palabras y coordenadas
        """
        if not self.available:
            raise RuntimeError("PaddleOCR no está disponible")

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        try:
            # Convertir página a imagen
            image = self._pdf_page_to_image(pdf_path, page_num)

            # Convertir PIL a numpy array
            img_array = np.array(image)

            # Aplicar OCR con GPU
            result = self.ocr.ocr(img_array, cls=True)

            # Obtener dimensiones de la imagen
            img_height, img_width = img_array.shape[:2]

            # Formatear resultados
            words = []

            if result and result[0]:
                for line in result[0]:
                    # PaddleOCR retorna: [bbox, (text, confidence)]
                    bbox_points = line[0]  # 4 puntos [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]

                    # Convertir bbox de 4 puntos a formato [x0, y0, x1, y1]
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]

                    x0 = min(x_coords)
                    y0 = min(y_coords)
                    x1 = max(x_coords)
                    y1 = max(y_coords)

                    # Normalizar coordenadas
                    bbox = self._normalize_bbox(
                        [x0, y0, x1, y1],
                        img_width,
                        img_height
                    )

                    # Dividir línea en palabras individuales
                    text_words = text.split()

                    if len(text_words) == 1:
                        # Una sola palabra
                        words.append({
                            "text": text,
                            "bbox": bbox,
                            "confidence": confidence,
                            "page": page_num
                        })
                    else:
                        # Múltiples palabras - dividir bbox proporcionalmente
                        bbox_width = x1 - x0
                        total_chars = len(text)
                        current_x = x0

                        for word in text_words:
                            word_chars = len(word)
                            word_width = (word_chars / total_chars) * bbox_width

                            word_bbox = self._normalize_bbox(
                                [current_x, y0, current_x + word_width, y1],
                                img_width,
                                img_height
                            )

                            words.append({
                                "text": word,
                                "bbox": word_bbox,
                                "confidence": confidence,
                                "page": page_num
                            })

                            current_x += word_width + (bbox_width * 0.02)  # Espacio entre palabras

            logger.debug(f"PaddleOCR extrajo {len(words)} palabras de página {page_num}")
            return words

        except Exception as e:
            logger.error(f"Error al extraer con PaddleOCR: {e}")
            raise

    def extract_all_pages(self, pdf_path: str) -> Dict[int, List[Dict]]:
        """
        Extrae todas las páginas del PDF usando PaddleOCR con GPU

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Diccionario {página: [palabras], ...}
        """
        if not self.available:
            raise RuntimeError("PaddleOCR no está disponible")

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

            logger.info(
                f"PaddleOCR extrajo {num_pages} páginas de {pdf_path.name} "
                f"(GPU: {self.use_gpu})"
            )
            return result

        except Exception as e:
            logger.error(f"Error al extraer todas las páginas: {e}")
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
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convertir a PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            doc.close()

            return img

        except Exception as e:
            logger.error(f"Error al convertir página a imagen: {e}")
            raise

    def _normalize_bbox(
        self,
        bbox: List[float],
        img_width: float,
        img_height: float
    ) -> List[int]:
        """
        Normaliza coordenadas a escala 0-1000

        Args:
            bbox: Coordenadas originales [x0, y0, x1, y1]
            img_width: Ancho de la imagen
            img_height: Alto de la imagen

        Returns:
            Coordenadas normalizadas
        """
        x0, y0, x1, y1 = bbox

        # Normalizar a escala 0-1000
        x0_norm = int((x0 / img_width) * self.bbox_scale)
        y0_norm = int((y0 / img_height) * self.bbox_scale)
        x1_norm = int((x1 / img_width) * self.bbox_scale)
        y1_norm = int((y1 / img_height) * self.bbox_scale)

        # Asegurar rango válido
        x0_norm = max(0, min(self.bbox_scale, x0_norm))
        y0_norm = max(0, min(self.bbox_scale, y0_norm))
        x1_norm = max(0, min(self.bbox_scale, x1_norm))
        y1_norm = max(0, min(self.bbox_scale, y1_norm))

        return [x0_norm, y0_norm, x1_norm, y1_norm]

    def benchmark(self, pdf_path: str, num_pages: int = 5) -> Dict:
        """
        Benchmark de velocidad

        Args:
            pdf_path: Ruta al PDF de prueba
            num_pages: Número de páginas a procesar

        Returns:
            Estadísticas de velocidad
        """
        import time

        if not self.available:
            return {"error": "PaddleOCR no disponible"}

        try:
            start_time = time.time()

            result = {}
            for page_num in range(min(num_pages, 5)):
                words = self.extract(pdf_path, page_num)
                result[page_num] = len(words)

            total_time = time.time() - start_time
            avg_time = total_time / len(result)

            return {
                "total_time": round(total_time, 2),
                "avg_time_per_page": round(avg_time, 2),
                "pages_processed": len(result),
                "words_extracted": sum(result.values()),
                "gpu_enabled": self.use_gpu
            }

        except Exception as e:
            return {"error": str(e)}
