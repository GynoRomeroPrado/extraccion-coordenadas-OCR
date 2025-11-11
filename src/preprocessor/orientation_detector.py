"""
Detector de orientación de documentos PDF

Detecta la orientación correcta del PDF (0°, 90°, 180°, 270°)
probando OCR en las 4 orientaciones y seleccionando la de mayor confianza.
"""
import logging
from typing import Tuple, Optional
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from pathlib import Path
import tempfile

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class OrientationDetector:
    """
    Detecta la orientación correcta del PDF

    Algoritmo:
    1. Convertir página a imagen
    2. Probar OCR en las 4 orientaciones (0°, 90°, 180°, 270°)
    3. Calcular confianza de cada orientación
    4. Seleccionar la de mayor confianza
    5. Rotar PDF si es necesario
    """

    def __init__(self, dpi: int = 300):
        """
        Args:
            dpi: Resolución para conversión de PDF a imagen
        """
        self.dpi = dpi
        self.angles = Config.ROTATION_ANGLES

    def detect(self, pdf_path: str, page_num: int = 0) -> int:
        """
        Detecta el ángulo de rotación necesario para corregir la orientación

        Args:
            pdf_path: Ruta al archivo PDF
            page_num: Número de página a analizar (default: primera página)

        Returns:
            Ángulo de rotación necesario (0, 90, 180, 270)

        Raises:
            FileNotFoundError: Si el PDF no existe
            ValueError: Si el PDF no tiene páginas o page_num es inválido
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        try:
            # Abrir PDF
            doc = fitz.open(pdf_path)

            if len(doc) == 0:
                raise ValueError(f"El PDF no tiene páginas: {pdf_path}")

            if page_num >= len(doc):
                logger.warning(
                    f"page_num {page_num} excede número de páginas ({len(doc)}). "
                    f"Usando página 0."
                )
                page_num = 0

            # Obtener página
            page = doc[page_num]

            # Convertir a imagen
            pix = page.get_pixmap(dpi=self.dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            doc.close()

            # Probar todas las orientaciones
            best_angle = self._find_best_orientation(img)

            logger.info(f"Orientación detectada: {best_angle}°")
            return best_angle

        except Exception as e:
            logger.error(f"Error al detectar orientación de {pdf_path}: {e}")
            raise

    def _find_best_orientation(self, image: Image.Image) -> int:
        """
        Prueba OCR en todas las orientaciones y retorna la mejor

        Args:
            image: Imagen PIL de la página

        Returns:
            Ángulo de rotación necesario (0, 90, 180, 270)
        """
        confidences = {}

        for angle in self.angles:
            # Rotar imagen
            rotated = image.rotate(-angle, expand=True)  # Negativo porque PIL rota en sentido antihorario

            # Ejecutar OCR con información de confianza
            try:
                osd = pytesseract.image_to_osd(rotated, config='--psm 0', lang=Config.TESSERACT_LANG)

                # Extraer confianza del output
                confidence = self._parse_osd_confidence(osd)
                confidences[angle] = confidence

                logger.debug(f"Ángulo {angle}°: confianza = {confidence:.2f}")

            except Exception as e:
                logger.warning(f"Error en OCR para ángulo {angle}°: {e}")
                confidences[angle] = 0.0

        # Seleccionar ángulo con mayor confianza
        if not confidences:
            logger.warning("No se pudo calcular confianza para ningún ángulo. Asumiendo 0°.")
            return 0

        best_angle = max(confidences.items(), key=lambda x: x[1])[0]
        best_confidence = confidences[best_angle]

        logger.debug(f"Mejor orientación: {best_angle}° (confianza: {best_confidence:.2f})")

        return best_angle

    def _parse_osd_confidence(self, osd_output: str) -> float:
        """
        Extrae la confianza del output de Tesseract OSD

        Args:
            osd_output: Texto de salida de image_to_osd

        Returns:
            Confianza como float (0.0 - 100.0)
        """
        try:
            # Buscar línea con "Orientation confidence"
            for line in osd_output.split('\n'):
                if 'Orientation confidence' in line or 'Rotate' in line:
                    # Extraer número
                    parts = line.split(':')
                    if len(parts) > 1:
                        confidence = float(parts[1].strip())
                        return confidence

            # Si no se encuentra, intentar con método alternativo
            # usando pytesseract.image_to_data
            return 50.0  # Default medio

        except Exception as e:
            logger.warning(f"Error al parsear confianza OSD: {e}")
            return 0.0

    def rotate_if_needed(
        self,
        pdf_path: str,
        angle: int,
        output_path: Optional[str] = None
    ) -> str:
        """
        Rota el PDF si es necesario

        Args:
            pdf_path: Ruta al PDF original
            angle: Ángulo de rotación necesario (0, 90, 180, 270)
            output_path: Ruta de salida (opcional, si no se provee se crea temp)

        Returns:
            Ruta del PDF corregido (original si angle=0, nuevo si rotado)

        Raises:
            FileNotFoundError: Si el PDF no existe
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        # Si no necesita rotación, retornar original
        if angle == 0:
            logger.debug(f"PDF {pdf_path.name} no necesita rotación")
            return str(pdf_path)

        try:
            # Abrir PDF
            doc = fitz.open(pdf_path)

            # Rotar todas las páginas
            for page in doc:
                page.set_rotation(angle)

            # Determinar ruta de salida
            if output_path is None:
                # Crear archivo temporal
                temp_dir = tempfile.gettempdir()
                output_path = Path(temp_dir) / f"{pdf_path.stem}_rotated_{angle}.pdf"
            else:
                output_path = Path(output_path)

            # Guardar PDF rotado
            doc.save(str(output_path))
            doc.close()

            logger.info(f"PDF rotado {angle}° guardado en: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Error al rotar PDF {pdf_path}: {e}")
            raise

    def detect_and_rotate(
        self,
        pdf_path: str,
        page_num: int = 0,
        output_path: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Detecta orientación y rota el PDF en un solo paso

        Args:
            pdf_path: Ruta al PDF original
            page_num: Página a analizar para detectar orientación
            output_path: Ruta de salida (opcional)

        Returns:
            Tupla (ruta_pdf_corregido, ángulo_rotación)
        """
        angle = self.detect(pdf_path, page_num)
        corrected_path = self.rotate_if_needed(pdf_path, angle, output_path)
        return corrected_path, angle
