"""
Mejorador de calidad de imagen para OCR

Aplica técnicas de mejora de imagen para optimizar la precisión del OCR:
- Ajuste de contraste y brillo
- Reducción de ruido
- Binarización adaptativa
- Sharpen
"""
import logging
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image, ImageEnhance

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Mejorador de calidad de imagen para OCR

    Aplica múltiples técnicas de procesamiento de imagen para
    mejorar la legibilidad y precisión del OCR.
    """

    def __init__(self):
        """Inicializa el mejorador"""
        pass

    def enhance(
        self,
        image: np.ndarray,
        denoise: bool = True,
        adjust_contrast: bool = True,
        sharpen: bool = True,
        binarize: bool = False
    ) -> np.ndarray:
        """
        Aplica mejoras a la imagen

        Args:
            image: Imagen como array numpy
            denoise: Aplicar reducción de ruido
            adjust_contrast: Ajustar contraste automáticamente
            sharpen: Aplicar sharpening
            binarize: Aplicar binarización adaptativa

        Returns:
            Imagen mejorada
        """
        result = image.copy()

        try:
            # 1. Reducción de ruido
            if denoise:
                result = self.reduce_noise(result)

            # 2. Ajuste de contraste
            if adjust_contrast:
                result = self.adjust_contrast(result)

            # 3. Sharpen
            if sharpen:
                result = self.sharpen(result)

            # 4. Binarización (opcional, último paso)
            if binarize:
                result = self.adaptive_binarize(result)

            logger.debug("Imagen mejorada para OCR")

        except Exception as e:
            logger.error(f"Error al mejorar imagen: {e}")
            return image

        return result

    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Reduce ruido de la imagen

        Args:
            image: Imagen

        Returns:
            Imagen sin ruido
        """
        try:
            # Usar fastNlMeansDenoisingColored para imágenes color
            if len(image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(
                    image,
                    None,
                    h=10,  # Filter strength para luminancia
                    hColor=10,  # Filter strength para color
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            else:
                # fastNlMeansDenoising para escala de grises
                denoised = cv2.fastNlMeansDenoising(
                    image,
                    None,
                    h=10,
                    templateWindowSize=7,
                    searchWindowSize=21
                )

            return denoised

        except Exception as e:
            logger.error(f"Error al reducir ruido: {e}")
            return image

    def adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Ajusta contraste automáticamente usando CLAHE

        Args:
            image: Imagen

        Returns:
            Imagen con contraste ajustado
        """
        try:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(image.shape) == 3:
                # Convertir a LAB y aplicar CLAHE en canal L
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)

                lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # Aplicar CLAHE directamente en escala de grises
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)

            return enhanced

        except Exception as e:
            logger.error(f"Error al ajustar contraste: {e}")
            return image

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica sharpening a la imagen

        Args:
            image: Imagen

        Returns:
            Imagen con sharpening
        """
        try:
            # Kernel de sharpening
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])

            sharpened = cv2.filter2D(image, -1, kernel)

            return sharpened

        except Exception as e:
            logger.error(f"Error al aplicar sharpening: {e}")
            return image

    def adaptive_binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica binarización adaptativa

        Args:
            image: Imagen

        Returns:
            Imagen binarizada
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Binarización adaptativa
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )

            return binary

        except Exception as e:
            logger.error(f"Error en binarización adaptativa: {e}")
            return image

    def enhance_pil(
        self,
        image: Image.Image,
        brightness: float = 1.0,
        contrast: float = 1.5,
        sharpness: float = 1.5
    ) -> Image.Image:
        """
        Mejora imagen PIL usando métodos de PIL

        Args:
            image: Imagen PIL
            brightness: Factor de brillo (1.0 = sin cambio)
            contrast: Factor de contraste (1.0 = sin cambio)
            sharpness: Factor de sharpness (1.0 = sin cambio)

        Returns:
            Imagen PIL mejorada
        """
        try:
            result = image

            # Ajustar brillo
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(result)
                result = enhancer.enhance(brightness)

            # Ajustar contraste
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(result)
                result = enhancer.enhance(contrast)

            # Ajustar sharpness
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(result)
                result = enhancer.enhance(sharpness)

            return result

        except Exception as e:
            logger.error(f"Error al mejorar imagen PIL: {e}")
            return image

    def auto_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica mejoras automáticas óptimas para OCR

        Args:
            image: Imagen original

        Returns:
            Imagen optimizada para OCR
        """
        # Pipeline óptimo para OCR
        enhanced = self.enhance(
            image,
            denoise=True,
            adjust_contrast=True,
            sharpen=True,
            binarize=False  # Dejar que Tesseract maneje la binarización
        )

        return enhanced

    def upscale(
        self,
        image: np.ndarray,
        scale_factor: float = 2.0,
        interpolation: int = cv2.INTER_CUBIC
    ) -> np.ndarray:
        """
        Aumenta la resolución de la imagen

        Útil para imágenes de baja resolución antes de OCR

        Args:
            image: Imagen
            scale_factor: Factor de escala (2.0 = doble tamaño)
            interpolation: Método de interpolación

        Returns:
            Imagen escalada
        """
        try:
            height, width = image.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            upscaled = cv2.resize(
                image,
                (new_width, new_height),
                interpolation=interpolation
            )

            logger.debug(f"Imagen escalada de {width}x{height} a {new_width}x{new_height}")

            return upscaled

        except Exception as e:
            logger.error(f"Error al escalar imagen: {e}")
            return image

    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Elimina sombras de la imagen

        Args:
            image: Imagen

        Returns:
            Imagen sin sombras
        """
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Dilatar para encontrar background
            dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))

            # Blur para suavizar
            blurred = cv2.medianBlur(dilated, 21)

            # Dividir imagen original por background
            result = 255 - cv2.absdiff(gray, blurred)

            # Normalizar
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

            return result

        except Exception as e:
            logger.error(f"Error al eliminar sombras: {e}")
            return image

    def detect_low_quality(self, image: np.ndarray) -> bool:
        """
        Detecta si la imagen es de baja calidad

        Args:
            image: Imagen

        Returns:
            True si es de baja calidad
        """
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Calcular varianza del Laplacian (medida de blur)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Si la varianza es baja, la imagen está borrosa
            is_low_quality = laplacian_var < 100

            if is_low_quality:
                logger.warning(f"Imagen de baja calidad detectada (laplacian_var: {laplacian_var:.2f})")

            return is_low_quality

        except Exception as e:
            logger.error(f"Error al detectar calidad: {e}")
            return False
