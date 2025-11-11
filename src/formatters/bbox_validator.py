"""
Validador de coordenadas y bounding boxes

Valida y corrige coordenadas para asegurar calidad del dataset LayoutLMv3.
"""
import logging
from typing import List, Dict, Optional, Tuple
import re

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class BBoxValidator:
    """
    Validador de bounding boxes

    Funciones:
    - Detectar coordenadas inválidas ([0,0])
    - Expandir bboxes demasiado pequeños
    - Detectar y resolver duplicados
    - Validar dimensiones mínimas
    """

    def __init__(
        self,
        min_width: int = 5,
        min_height: int = 5,
        avg_char_width: int = 8,
        avg_char_height: int = 12
    ):
        """
        Args:
            min_width: Ancho mínimo de bbox (píxeles normalizados)
            min_height: Alto mínimo de bbox (píxeles normalizados)
            avg_char_width: Ancho promedio por carácter (para estimación)
            avg_char_height: Alto promedio por carácter (para estimación)
        """
        self.min_width = min_width
        self.min_height = min_height
        self.avg_char_width = avg_char_width
        self.avg_char_height = avg_char_height

    def is_valid_bbox(self, bbox: List[int]) -> bool:
        """
        Verifica si un bbox es válido

        Args:
            bbox: Coordenadas [x0, y0, x1, y1]

        Returns:
            True si es válido
        """
        if not bbox or len(bbox) != 4:
            return False

        x0, y0, x1, y1 = bbox

        # Verificar que no sea [0,0]
        if x0 == 0 and y0 == 0 and x1 == 0 and y1 == 0:
            return False

        # Verificar que x1 > x0 y y1 > y0
        if x1 <= x0 or y1 <= y0:
            return False

        # Verificar dimensiones mínimas
        width = x1 - x0
        height = y1 - y0

        if width < self.min_width or height < self.min_height:
            return False

        # Verificar rango válido (0-1000)
        if not all(0 <= coord <= 1000 for coord in bbox):
            return False

        return True

    def is_suspiciously_small(self, bbox: List[int], text: str) -> bool:
        """
        Detecta si un bbox es sospechosamente pequeño para el texto

        Args:
            bbox: Coordenadas [x0, y0, x1, y1]
            text: Texto que debería contener

        Returns:
            True si es sospechosamente pequeño
        """
        if not text or not bbox:
            return False

        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0

        # Estimar dimensiones esperadas basadas en el texto
        text_len = len(text)
        expected_width = text_len * self.avg_char_width
        expected_height = self.avg_char_height

        # Si el ancho es < 50% del esperado, es sospechoso
        if width < expected_width * 0.5:
            logger.debug(
                f"Bbox sospechosamente pequeño: '{text}' ({text_len} chars) "
                f"tiene width={width} (esperado ~{expected_width})"
            )
            return True

        # Si el alto es < 50% del esperado, es sospechoso
        if height < expected_height * 0.5:
            return True

        return False

    def expand_bbox(self, bbox: List[int], text: str) -> List[int]:
        """
        Expande un bbox basándose en el texto

        Args:
            bbox: Coordenadas originales [x0, y0, x1, y1]
            text: Texto contenido

        Returns:
            Bbox expandido
        """
        if not text:
            return bbox

        x0, y0, x1, y1 = bbox
        text_len = len(text)

        # Calcular dimensiones esperadas
        expected_width = max(text_len * self.avg_char_width, self.min_width)
        expected_height = max(self.avg_char_height, self.min_height)

        # Expandir desde el centro
        current_width = x1 - x0
        current_height = y1 - y0

        if current_width < expected_width:
            # Expandir horizontalmente
            expansion = (expected_width - current_width) // 2
            x0 = max(0, x0 - expansion)
            x1 = min(1000, x1 + expansion)

        if current_height < expected_height:
            # Expandir verticalmente
            expansion = (expected_height - current_height) // 2
            y0 = max(0, y0 - expansion)
            y1 = min(1000, y1 + expansion)

        logger.debug(
            f"Expandido bbox para '{text[:20]}...': "
            f"[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}] → [{x0},{y0},{x1},{y1}]"
        )

        return [x0, y0, x1, y1]

    def fix_bbox(self, bbox: List[int], text: str) -> List[int]:
        """
        Corrige un bbox inválido o sospechoso

        Args:
            bbox: Coordenadas originales
            text: Texto contenido

        Returns:
            Bbox corregido
        """
        # Si es sospechosamente pequeño, expandir
        if self.is_suspiciously_small(bbox, text):
            return self.expand_bbox(bbox, text)

        # Si tiene dimensiones mínimas pero válidas, dejar como está
        x0, y0, x1, y1 = bbox
        width = x1 - x0
        height = y1 - y0

        # Asegurar dimensiones mínimas
        if width < self.min_width:
            x1 = x0 + self.min_width
            x1 = min(1000, x1)

        if height < self.min_height:
            y1 = y0 + self.min_height
            y1 = min(1000, y1)

        return [x0, y0, x1, y1]

    def detect_duplicates(self, elements: List[Dict]) -> Dict[Tuple, List[int]]:
        """
        Detecta elementos con bbox duplicados

        Args:
            elements: Lista de elementos formateados

        Returns:
            Diccionario {bbox_tuple: [índices_de_elementos]}
        """
        bbox_map = {}

        for i, elem in enumerate(elements):
            bbox = tuple(elem.get('box', []))
            if bbox not in bbox_map:
                bbox_map[bbox] = []
            bbox_map[bbox].append(i)

        # Filtrar solo los que tienen duplicados
        duplicates = {bbox: indices for bbox, indices in bbox_map.items() if len(indices) > 1}

        if duplicates:
            logger.warning(f"Detectados {len(duplicates)} bboxes duplicados")

        return duplicates

    def resolve_duplicates(
        self,
        elements: List[Dict],
        strategy: str = "keep_best"
    ) -> List[Dict]:
        """
        Resuelve duplicados de bbox

        Args:
            elements: Lista de elementos
            strategy: Estrategia de resolución:
                - "keep_best": Mantener solo el de mayor confidence
                - "expand": Expandir ligeramente cada bbox para diferenciarlos
                - "remove_all": Eliminar todos los duplicados

        Returns:
            Lista de elementos sin duplicados
        """
        duplicates = self.detect_duplicates(elements)

        if not duplicates:
            return elements

        # Marcar elementos a mantener
        keep_indices = set(range(len(elements)))

        for bbox, indices in duplicates.items():
            if strategy == "keep_best":
                # Mantener solo el de mayor confidence
                best_idx = max(
                    indices,
                    key=lambda i: elements[i].get('confidence', 0)
                )
                # Eliminar los demás
                for idx in indices:
                    if idx != best_idx:
                        keep_indices.discard(idx)

                logger.debug(
                    f"Duplicados en bbox {bbox}: manteniendo elemento {best_idx} "
                    f"(confidence: {elements[best_idx].get('confidence', 0):.2f})"
                )

            elif strategy == "expand":
                # Expandir cada bbox ligeramente
                for offset, idx in enumerate(indices):
                    elem = elements[idx]
                    x0, y0, x1, y1 = elem['box']

                    # Desplazar ligeramente cada duplicado
                    shift = offset * 2  # 2 píxeles por duplicado
                    elem['box'] = [
                        min(1000, x0 + shift),
                        y0,
                        min(1000, x1 + shift),
                        y1
                    ]

            elif strategy == "remove_all":
                # Eliminar todos los duplicados
                for idx in indices:
                    keep_indices.discard(idx)

        # Retornar solo elementos válidos
        result = [elements[i] for i in sorted(keep_indices)]

        removed = len(elements) - len(result)
        if removed > 0:
            logger.info(f"Eliminados {removed} elementos duplicados")

        return result

    def validate_words_consistency(self, element: Dict) -> bool:
        """
        Valida que las palabras sean consistentes con el texto y bbox

        Args:
            element: Elemento formateado

        Returns:
            True si es consistente
        """
        text = element.get('text', '')
        words = element.get('words', [])
        main_bbox = element.get('box', [])

        if not text or not words:
            return True

        # Verificar que el número de palabras coincida
        text_words = text.split()
        if len(text_words) != len(words):
            logger.warning(
                f"Inconsistencia: texto tiene {len(text_words)} palabras "
                f"pero 'words' tiene {len(words)}"
            )
            return False

        # Verificar que la suma de anchos de palabras no exceda el bbox total
        if len(main_bbox) == 4:
            main_width = main_bbox[2] - main_bbox[0]
            words_total_width = sum(
                w['box'][2] - w['box'][0] for w in words if 'box' in w
            )

            if words_total_width > main_width * 1.5:  # 50% de tolerancia
                logger.warning(
                    f"Inconsistencia: suma de anchos de palabras ({words_total_width}) "
                    f"excede bbox principal ({main_width})"
                )
                return False

        return True

    def get_quality_metrics(self, elements: List[Dict]) -> Dict:
        """
        Calcula métricas de calidad de los elementos

        Args:
            elements: Lista de elementos

        Returns:
            Diccionario con métricas
        """
        total = len(elements)
        if total == 0:
            return {
                "total_elements": 0,
                "valid_bboxes": 0,
                "invalid_bboxes": 0,
                "suspiciously_small": 0,
                "zero_confidence": 0,
                "duplicates": 0,
                "inconsistent_words": 0
            }

        invalid = 0
        suspicious = 0
        zero_conf = 0
        inconsistent = 0

        for elem in elements:
            bbox = elem.get('box', [])
            text = elem.get('text', '')
            confidence = elem.get('confidence', 0)

            if not self.is_valid_bbox(bbox):
                invalid += 1

            if self.is_suspiciously_small(bbox, text):
                suspicious += 1

            if confidence < 0.01:
                zero_conf += 1

            if not self.validate_words_consistency(elem):
                inconsistent += 1

        duplicates = self.detect_duplicates(elements)

        return {
            "total_elements": total,
            "valid_bboxes": total - invalid,
            "invalid_bboxes": invalid,
            "suspiciously_small": suspicious,
            "zero_confidence": zero_conf,
            "duplicate_groups": len(duplicates),
            "total_duplicates": sum(len(v) - 1 for v in duplicates.values()),
            "inconsistent_words": inconsistent,
            "quality_score": (total - invalid - zero_conf) / total if total > 0 else 0
        }
