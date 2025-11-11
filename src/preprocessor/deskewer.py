"""
Corrector de inclinación (deskewing) de documentos

Detecta y corrige inclinación leve del documento escaneado
para mejorar la precisión del OCR.
"""
import logging
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class Deskewer:
    """
    Corrector de inclinación de documentos

    Algoritmo:
    1. Detectar líneas horizontales/verticales usando Hough Transform
    2. Calcular ángulo de inclinación promedio
    3. Rotar imagen para corregir
    4. Recortar bordes negros resultantes
    """

    def __init__(self, max_angle: float = 10.0):
        """
        Args:
            max_angle: Máximo ángulo de corrección en grados (default: 10°)
        """
        self.max_angle = max_angle

    def detect_skew(self, image: np.ndarray) -> float:
        """
        Detecta el ángulo de inclinación de la imagen

        Args:
            image: Imagen como array numpy (BGR o grayscale)

        Returns:
            Ángulo de inclinación en grados (-max_angle a +max_angle)
            Positivo = rotado en sentido horario
            Negativo = rotado en sentido antihorario
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Aplicar threshold binario
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Detectar bordes con Canny
            edges = cv2.Canny(binary, 50, 150, apertureSize=3)

            # Detectar líneas con Hough Transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10
            )

            if lines is None:
                logger.debug("No se detectaron líneas para calcular skew")
                return 0.0

            # Calcular ángulos de las líneas
            angles = []

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calcular ángulo
                if x2 - x1 != 0:
                    angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))

                    # Filtrar líneas cercanas a horizontal o vertical
                    # (las que nos interesan para detectar skew)
                    if abs(angle) < self.max_angle:
                        angles.append(angle)

            if not angles:
                logger.debug("No se encontraron líneas con ángulo válido")
                return 0.0

            # Calcular ángulo mediano (más robusto que promedio)
            skew_angle = np.median(angles)

            logger.debug(f"Ángulo de inclinación detectado: {skew_angle:.2f}°")

            return skew_angle

        except Exception as e:
            logger.error(f"Error al detectar skew: {e}")
            return 0.0

    def deskew(
        self,
        image: np.ndarray,
        angle: Optional[float] = None,
        background_color: tuple = (255, 255, 255)
    ) -> np.ndarray:
        """
        Corrige la inclinación de la imagen

        Args:
            image: Imagen como array numpy
            angle: Ángulo de corrección (si None, se detecta automáticamente)
            background_color: Color de fondo para áreas vacías (default: blanco)

        Returns:
            Imagen corregida
        """
        try:
            # Detectar ángulo si no se proporciona
            if angle is None:
                angle = self.detect_skew(image)

            # Si el ángulo es muy pequeño, no rotar
            if abs(angle) < 0.1:
                logger.debug("Ángulo muy pequeño, no se requiere corrección")
                return image

            # Obtener dimensiones
            height, width = image.shape[:2]

            # Calcular matriz de rotación
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Calcular nuevas dimensiones para que quepa toda la imagen rotada
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])

            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))

            # Ajustar la matriz de rotación para las nuevas dimensiones
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]

            # Aplicar rotación
            deskewed = cv2.warpAffine(
                image,
                rotation_matrix,
                (new_width, new_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=background_color
            )

            # Recortar bordes (crop al contenido)
            deskewed = self._crop_borders(deskewed, background_color)

            logger.info(f"Imagen corregida con ángulo: {angle:.2f}°")

            return deskewed

        except Exception as e:
            logger.error(f"Error al corregir inclinación: {e}")
            return image

    def deskew_pil(
        self,
        image: Image.Image,
        angle: Optional[float] = None
    ) -> Image.Image:
        """
        Corrige inclinación de una imagen PIL

        Args:
            image: Imagen PIL
            angle: Ángulo de corrección (si None, se detecta automáticamente)

        Returns:
            Imagen PIL corregida
        """
        # Convertir PIL a numpy
        np_image = np.array(image)

        # Convertir RGB a BGR si es necesario
        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        # Aplicar deskew
        deskewed = self.deskew(np_image, angle)

        # Convertir de vuelta a PIL
        if len(deskewed.shape) == 3:
            deskewed = cv2.cvtColor(deskewed, cv2.COLOR_BGR2RGB)

        return Image.fromarray(deskewed)

    def _crop_borders(
        self,
        image: np.ndarray,
        background_color: tuple,
        margin: int = 10
    ) -> np.ndarray:
        """
        Recorta bordes vacíos de la imagen

        Args:
            image: Imagen
            background_color: Color del fondo
            margin: Margen a dejar (píxeles)

        Returns:
            Imagen recortada
        """
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Threshold para detectar contenido
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

            # Encontrar contornos del contenido
            coords = cv2.findNonZero(binary)

            if coords is None:
                # Si no hay contenido, retornar imagen original
                return image

            # Calcular bounding box del contenido
            x, y, w, h = cv2.boundingRect(coords)

            # Aplicar margen
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)

            # Recortar
            cropped = image[y:y+h, x:x+w]

            return cropped

        except Exception as e:
            logger.error(f"Error al recortar bordes: {e}")
            return image

    def process_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa imagen para mejorar OCR (deskew + mejoras)

        Args:
            image: Imagen original

        Returns:
            Imagen preprocesada
        """
        # Detectar y corregir inclinación
        deskewed = self.deskew(image)

        return deskewed
