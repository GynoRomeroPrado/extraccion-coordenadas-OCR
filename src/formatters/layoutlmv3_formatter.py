"""
Formateador para generar salida en formato LayoutLMv3 (FUNSD-style)

Convierte campos extraídos y matcheados a formato compatible con
LayoutLMv3 para entrenamiento.
"""
import logging
from typing import List, Dict, Any, Optional
import json

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class LayoutLMv3Formatter:
    """
    Genera formato LayoutLMv3 oficial (FUNSD-style)

    Formato de salida:
    {
        "form": [
            {
                "id": 0,
                "text": "FACTURA ELECTRONICA",
                "box": [120, 45, 380, 72],  # Normalizado 0-1000
                "label": "answer",
                "field_name": "tipo_documento",
                "words": [
                    {"text": "FACTURA", "box": [120, 45, 250, 72]},
                    {"text": "ELECTRONICA", "box": [255, 45, 380, 72]}
                ],
                "page": 0,
                "confidence": 0.95
            },
            ...
        ]
    }
    """

    def __init__(self):
        """Inicializa el formateador"""
        self.id_counter = 0

    def format_field(
        self,
        field_name: str,
        match_result: Optional[Dict],
        label: str = "answer"
    ) -> Optional[Dict]:
        """
        Formatea un campo individual

        Args:
            field_name: Nombre del campo (ej: "emisor_ruc")
            match_result: Resultado del matching (de FuzzyMatcher)
            label: Tipo de label ("answer", "question", "header", "other")

        Returns:
            Diccionario en formato LayoutLMv3, o None si match_result es None
        """
        if match_result is None:
            return None

        # Extraer información del match
        text = match_result.get('text', '')
        bbox = match_result.get('bbox', [0, 0, 0, 0])
        confidence = match_result.get('confidence', 0.0)
        page = match_result.get('page', 0)

        # Dividir texto en palabras si tiene múltiples
        words = self._split_into_words(text, bbox)

        # Crear elemento
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
        Formatea campos del header (nivel documento)

        Args:
            header_data: Diccionario con campos del header del JSON
            pdf_words: Palabras extraídas del PDF
            matcher: Instancia de FuzzyMatcher

        Returns:
            Lista de elementos formateados
        """
        formatted_fields = []

        for field_name, field_value in header_data.items():
            if field_value is None or field_value == "":
                continue

            # Buscar campo en el PDF
            if isinstance(field_value, (int, float)):
                # Valores numéricos
                match_result = matcher.match_numeric(field_value, pdf_words, field_name)
            elif isinstance(field_value, str) and len(field_value.split()) > 1:
                # Valores multi-palabra
                match_result = matcher.match_multiword(field_value, pdf_words, field_name)
            else:
                # Valores simples
                match_result = matcher.match(field_value, pdf_words, field_name)

            # Formatear campo
            formatted = self.format_field(field_name, match_result, label="answer")

            if formatted:
                formatted_fields.append(formatted)

        logger.info(
            f"Formateados {len(formatted_fields)}/{len(header_data)} campos del header"
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
        Formatea items (líneas de productos/servicios)

        Args:
            items: Lista de items del JSON
            pdf_words_by_page: Palabras del PDF por página
            matcher: Instancia de FuzzyMatcher
            table_detector: Opcional, detector de tablas para mejorar matching

        Returns:
            Lista de elementos formateados
        """
        if not items:
            return []

        formatted_items = []

        # Si hay detector de tabla, usarlo para matching más eficiente
        if table_detector:
            # TODO: Implementar detección de tabla
            # Por ahora, usar matching simple
            pass

        # Combinar todas las palabras del PDF
        all_pdf_words = []
        for page_words in pdf_words_by_page.values():
            all_pdf_words.extend(page_words)

        # Procesar cada item
        for item in items:
            # Formatear cada campo del item
            for field_name, field_value in item.items():
                if field_value is None or field_value == "":
                    continue

                # Buscar campo en el PDF
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
        sort_by_position: bool = True
    ) -> Dict:
        """
        Crea documento completo en formato LayoutLMv3

        Args:
            elements: Lista de elementos formateados
            sort_by_position: Si True, ordena por posición de lectura (top-left a bottom-right)

        Returns:
            Documento completo:
            {
                "form": [...]
            }
        """
        if sort_by_position:
            elements = self._sort_by_reading_order(elements)

        return {
            "form": elements
        }

    def _split_into_words(self, text: str, bbox: List[int]) -> List[Dict]:
        """
        Divide texto en palabras individuales con bboxes aproximados

        Args:
            text: Texto completo
            bbox: Bbox completo [x0, y0, x1, y1]

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
            word_width = int((len(word) / char_count) * total_width)

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

            current_x += word_width + int(total_width * 0.02)  # Espacio entre palabras

        return result

    def _sort_by_reading_order(self, elements: List[Dict]) -> List[Dict]:
        """
        Ordena elementos por orden de lectura (top-left a bottom-right)

        Args:
            elements: Lista de elementos

        Returns:
            Lista ordenada
        """
        def get_position_key(elem):
            bbox = elem.get('box', [0, 0, 0, 0])
            page = elem.get('page', 0)
            # Ordenar por: página, y0 (top), x0 (left)
            return (page, bbox[1], bbox[0])

        return sorted(elements, key=get_position_key)

    def _get_next_id(self) -> int:
        """Retorna siguiente ID y lo incrementa"""
        current_id = self.id_counter
        self.id_counter += 1
        return current_id

    def reset_id_counter(self):
        """Resetea el contador de IDs"""
        self.id_counter = 0

    def save_to_file(self, document: Dict, output_path: str):
        """
        Guarda documento en archivo JSON

        Args:
            document: Documento formateado
            output_path: Ruta del archivo de salida
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)

            logger.info(f"Documento guardado en: {output_path}")

        except Exception as e:
            logger.error(f"Error al guardar documento: {e}")
            raise

    def load_from_file(self, input_path: str) -> Dict:
        """
        Carga documento desde archivo JSON

        Args:
            input_path: Ruta del archivo de entrada

        Returns:
            Documento cargado
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                document = json.load(f)

            logger.info(f"Documento cargado desde: {input_path}")
            return document

        except Exception as e:
            logger.error(f"Error al cargar documento: {e}")
            raise

    def get_stats(self, document: Dict) -> Dict:
        """
        Calcula estadísticas del documento

        Args:
            document: Documento formateado

        Returns:
            Diccionario con estadísticas
        """
        elements = document.get('form', [])

        if not elements:
            return {
                "num_elements": 0,
                "num_words": 0,
                "avg_confidence": 0.0,
                "pages": []
            }

        pages = set(elem.get('page', 0) for elem in elements)
        confidences = [elem.get('confidence', 0.0) for elem in elements]
        total_words = sum(len(elem.get('words', [])) for elem in elements)

        return {
            "num_elements": len(elements),
            "num_words": total_words,
            "avg_confidence": round(sum(confidences) / len(confidences), 3),
            "pages": sorted(list(pages)),
            "num_pages": len(pages)
        }
