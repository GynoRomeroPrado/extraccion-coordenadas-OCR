"""
Formateador mejorado para LayoutLMv3 con validaciones robustas

Versión 2 con:
- Filtrado de campos con confidence < 0.5
- Validación y corrección de bboxes
- Resolución de duplicados
- Eliminación de metadata
- Métricas de calidad
"""
import logging
from typing import List, Dict, Any, Optional
import json

from config import Config
from .bbox_validator import BBoxValidator

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class LayoutLMv3FormatterV2:
    """
    Formateador mejorado para LayoutLMv3 con validaciones robustas
    """

    # Campos de metadata que NO deben estar en el output final
    METADATA_FIELDS = {
        'augmentation', 'is_augmented', 'extraction_method',
        'rotation_corrected', 'match_score', 'word_count'
    }

    def __init__(
        self,
        min_confidence: float = 0.5,
        resolve_duplicates: bool = True,
        expand_small_bboxes: bool = True,
        duplicate_strategy: str = "keep_best"
    ):
        """
        Args:
            min_confidence: Confidence mínimo para incluir un campo (default: 0.5)
            resolve_duplicates: Si True, resuelve bboxes duplicados
            expand_small_bboxes: Si True, expande bboxes sospechosamente pequeños
            duplicate_strategy: "keep_best", "expand", o "remove_all"
        """
        self.id_counter = 0
        self.min_confidence = min_confidence
        self.resolve_duplicates = resolve_duplicates
        self.expand_small_bboxes = expand_small_bboxes
        self.duplicate_strategy = duplicate_strategy

        # Inicializar validador de bbox
        self.bbox_validator = BBoxValidator()

        # Estadísticas de procesamiento
        self.stats = {
            'total_fields': 0,
            'filtered_low_confidence': 0,
            'filtered_invalid_bbox': 0,
            'expanded_bboxes': 0,
            'resolved_duplicates': 0,
            'removed_metadata_fields': 0
        }

    def format_field(
        self,
        field_name: str,
        match_result: Optional[Dict],
        label: str = "answer"
    ) -> Optional[Dict]:
        """
        Formatea un campo individual con validaciones

        Args:
            field_name: Nombre del campo
            match_result: Resultado del matching
            label: Tipo de label

        Returns:
            Diccionario formateado o None si no pasa validación
        """
        if match_result is None:
            return None

        self.stats['total_fields'] += 1

        # Extraer información
        text = match_result.get('text', '')
        bbox = match_result.get('bbox', [0, 0, 0, 0])
        confidence = match_result.get('confidence', 0.0)
        page = match_result.get('page', 0)

        # VALIDACIÓN 1: Filtrar confidence bajo
        if confidence < self.min_confidence:
            logger.debug(
                f"Filtrado por confidence bajo: '{field_name}' "
                f"(confidence: {confidence:.2f} < {self.min_confidence})"
            )
            self.stats['filtered_low_confidence'] += 1
            return None

        # VALIDACIÓN 2: Verificar bbox válido
        if not self.bbox_validator.is_valid_bbox(bbox):
            logger.debug(
                f"Filtrado por bbox inválido: '{field_name}' "
                f"bbox: {bbox}"
            )
            self.stats['filtered_invalid_bbox'] += 1
            return None

        # VALIDACIÓN 3: Expandir si es necesario
        if self.expand_small_bboxes and self.bbox_validator.is_suspiciously_small(bbox, text):
            original_bbox = bbox.copy()
            bbox = self.bbox_validator.expand_bbox(bbox, text)
            logger.debug(
                f"Expandido bbox para '{field_name}': {original_bbox} → {bbox}"
            )
            self.stats['expanded_bboxes'] += 1

        # VALIDACIÓN 4: Corregir bbox si necesario
        bbox = self.bbox_validator.fix_bbox(bbox, text)

        # Dividir texto en palabras
        words = self._split_into_words(text, bbox)

        # Crear elemento limpio (sin metadata)
        element = {
            "id": self._get_next_id(),
            "text": text,
            "box": bbox,
            "label": label,
            "field_name": field_name,
            "words": words,
            "page": page,
            "confidence": round(confidence, 3)
        }

        return element

    def format_header(
        self,
        header_data: Dict[str, Any],
        pdf_words: List[Dict],
        matcher
    ) -> List[Dict]:
        """
        Formatea campos del header con filtrado

        Args:
            header_data: Campos del header
            pdf_words: Palabras del PDF
            matcher: Instancia de FuzzyMatcher

        Returns:
            Lista de elementos formateados y validados
        """
        formatted_fields = []

        for field_name, field_value in header_data.items():
            # Saltar campos de metadata
            if field_name in self.METADATA_FIELDS:
                self.stats['removed_metadata_fields'] += 1
                continue

            if field_value is None or field_value == "":
                continue

            # Buscar campo en el PDF
            if isinstance(field_value, (int, float)):
                match_result = matcher.match_numeric(field_value, pdf_words, field_name)
            elif isinstance(field_value, str) and len(field_value.split()) > 1:
                match_result = matcher.match_multiword(field_value, pdf_words, field_name)
            else:
                match_result = matcher.match(field_value, pdf_words, field_name)

            # Formatear campo
            formatted = self.format_field(field_name, match_result, label="answer")

            if formatted:
                formatted_fields.append(formatted)

        logger.info(
            f"Formateados {len(formatted_fields)} campos del header "
            f"(filtrados: {self.stats['filtered_low_confidence']} por confidence, "
            f"{self.stats['filtered_invalid_bbox']} por bbox inválido)"
        )

        return formatted_fields

    def format_items(
        self,
        items: List[Dict],
        pdf_words_by_page: Dict[int, List[Dict]],
        matcher,
        table_detector=None
    ) -> List[Dict]:
        """
        Formatea items con validaciones

        Args:
            items: Lista de items
            pdf_words_by_page: Palabras por página
            matcher: FuzzyMatcher
            table_detector: Opcional

        Returns:
            Lista de elementos formateados
        """
        if not items:
            return []

        formatted_items = []

        # Combinar todas las palabras
        all_pdf_words = []
        for page_words in pdf_words_by_page.values():
            all_pdf_words.extend(page_words)

        # Procesar cada item
        for item in items:
            for field_name, field_value in item.items():
                # Saltar metadata
                if field_name in self.METADATA_FIELDS:
                    self.stats['removed_metadata_fields'] += 1
                    continue

                if field_value is None or field_value == "":
                    continue

                # Buscar campo
                if isinstance(field_value, (int, float)):
                    match_result = matcher.match_numeric(field_value, all_pdf_words, f"item.{field_name}")
                elif isinstance(field_value, str) and len(field_value.split()) > 1:
                    match_result = matcher.match_multiword(field_value, all_pdf_words, f"item.{field_name}")
                else:
                    match_result = matcher.match(field_value, all_pdf_words, f"item.{field_name}")

                # Formatear campo
                formatted = self.format_field(f"item.{field_name}", match_result, label="answer")

                if formatted:
                    formatted_items.append(formatted)

        logger.info(f"Formateados {len(formatted_items)} campos de items")

        return formatted_items

    def create_document(
        self,
        elements: List[Dict],
        sort_by_position: bool = True,
        apply_validations: bool = True
    ) -> Dict:
        """
        Crea documento completo con validaciones

        Args:
            elements: Lista de elementos
            sort_by_position: Ordenar por posición
            apply_validations: Aplicar validaciones adicionales

        Returns:
            Documento formateado
        """
        if apply_validations:
            # Resolver duplicados si está habilitado
            if self.resolve_duplicates:
                original_count = len(elements)
                elements = self.bbox_validator.resolve_duplicates(
                    elements,
                    strategy=self.duplicate_strategy
                )
                removed = original_count - len(elements)
                if removed > 0:
                    self.stats['resolved_duplicates'] += removed
                    logger.info(f"Resueltos {removed} elementos duplicados")

            # Validar consistencia de palabras
            valid_elements = []
            for elem in elements:
                if self.bbox_validator.validate_words_consistency(elem):
                    valid_elements.append(elem)
                else:
                    logger.warning(f"Elemento con inconsistencia eliminado: {elem.get('field_name')}")

            elements = valid_elements

        # Ordenar si se solicita
        if sort_by_position:
            elements = self._sort_by_reading_order(elements)

        return {
            "form": elements
        }

    def _split_into_words(self, text: str, bbox: List[int]) -> List[Dict]:
        """
        Divide texto en palabras con bboxes aproximados

        Args:
            text: Texto completo
            bbox: Bbox completo

        Returns:
            Lista de palabras con bboxes
        """
        words = text.split()
        if len(words) == 1:
            return [{"text": text, "box": bbox}]

        # Dividir bbox proporcionalmente
        x0, y0, x1, y1 = bbox
        total_width = x1 - x0
        char_count = len(text)

        result = []
        current_x = x0

        for word in words:
            # Estimar ancho de la palabra
            word_width = max(int((len(word) / char_count) * total_width), 5)  # Mínimo 5 píxeles

            word_bbox = [
                current_x,
                y0,
                min(current_x + word_width, x1),
                y1
            ]

            result.append({
                "text": word,
                "box": word_bbox
            })

            current_x += word_width + max(int(total_width * 0.02), 2)  # Espacio entre palabras (mínimo 2px)

        return result

    def _sort_by_reading_order(self, elements: List[Dict]) -> List[Dict]:
        """
        Ordena elementos por orden de lectura

        Args:
            elements: Lista de elementos

        Returns:
            Lista ordenada
        """
        def get_position_key(elem):
            bbox = elem.get('box', [0, 0, 0, 0])
            page = elem.get('page', 0)
            return (page, bbox[1], bbox[0])

        return sorted(elements, key=get_position_key)

    def _get_next_id(self) -> int:
        """Retorna siguiente ID"""
        current_id = self.id_counter
        self.id_counter += 1
        return current_id

    def reset_id_counter(self):
        """Resetea el contador de IDs"""
        self.id_counter = 0

    def reset_stats(self):
        """Resetea estadísticas"""
        self.stats = {
            'total_fields': 0,
            'filtered_low_confidence': 0,
            'filtered_invalid_bbox': 0,
            'expanded_bboxes': 0,
            'resolved_duplicates': 0,
            'removed_metadata_fields': 0
        }

    def get_stats(self) -> Dict:
        """Retorna estadísticas de procesamiento"""
        return self.stats.copy()

    def get_quality_metrics(self, document: Dict) -> Dict:
        """
        Calcula métricas de calidad del documento

        Args:
            document: Documento formateado

        Returns:
            Métricas de calidad
        """
        elements = document.get('form', [])

        # Usar validator para métricas
        validator_metrics = self.bbox_validator.get_quality_metrics(elements)

        # Agregar métricas de procesamiento
        validator_metrics['processing_stats'] = self.get_stats()

        return validator_metrics

    def save_to_file(self, document: Dict, output_path: str):
        """
        Guarda documento en archivo JSON

        Args:
            document: Documento formateado
            output_path: Ruta de salida
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)

            logger.info(f"Documento guardado en: {output_path}")

        except Exception as e:
            logger.error(f"Error al guardar documento: {e}")
            raise
