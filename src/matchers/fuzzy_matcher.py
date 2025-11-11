"""
Matcher difuso para buscar campos del JSON en el PDF

Utiliza RapidFuzz para matching flexible de campos, permitiendo
tolerancia a errores de OCR y variaciones en el texto.
"""
import logging
from typing import List, Dict, Optional, Tuple
import re
import unicodedata
from rapidfuzz import fuzz, process

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class FuzzyMatcher:
    """
    Matcher difuso para localizar campos en palabras del PDF

    Algoritmo:
    1. Normalizar texto (quitar acentos, puntuación, minúsculas)
    2. Usar RapidFuzz para matching
    3. Calcular confianza combinada (similitud × confianza OCR)
    4. Retornar mejor match con coordenadas
    """

    def __init__(self, threshold: int = 85):
        """
        Args:
            threshold: Umbral mínimo de similitud (0-100)
        """
        self.threshold = threshold

    def match(
        self,
        field_value: str,
        pdf_words: List[Dict],
        field_name: Optional[str] = None,
        top_n: int = 1
    ) -> Optional[Dict]:
        """
        Busca el valor del campo en las palabras del PDF

        Args:
            field_value: Valor a buscar (del JSON)
            pdf_words: Lista de palabras extraídas del PDF
            field_name: Nombre del campo (para logging)
            top_n: Número de mejores matches a retornar

        Returns:
            Diccionario con el mejor match:
            {
                "text": "20393920069",
                "bbox": [120, 45, 280, 65],
                "confidence": 0.92,
                "page": 0,
                "match_score": 0.95,  # Similitud del matching
                "field_name": "emisor_ruc"
            }

            O None si no se encuentra match válido
        """
        if not field_value or not pdf_words:
            return None

        # Convertir field_value a string y normalizar
        field_value_str = str(field_value).strip()
        if not field_value_str:
            return None

        # Normalizar valor buscado
        normalized_query = self._normalize_text(field_value_str)

        # Preparar textos del PDF
        pdf_texts = [w['text'] for w in pdf_words]
        normalized_texts = [self._normalize_text(t) for t in pdf_texts]

        # Buscar matches exactos primero (más rápido)
        exact_matches = []
        for i, norm_text in enumerate(normalized_texts):
            if norm_text == normalized_query:
                exact_matches.append(i)

        if exact_matches:
            # Si hay match exacto, retornar el primero
            idx = exact_matches[0]
            word = pdf_words[idx]
            result = {
                "text": word['text'],
                "bbox": word['bbox'],
                "confidence": word['confidence'],
                "page": word['page'],
                "match_score": 1.0,
                "field_name": field_name
            }
            logger.debug(f"Match exacto para '{field_name}': {field_value_str}")
            return result

        # Buscar con fuzzy matching
        matches = process.extract(
            normalized_query,
            normalized_texts,
            scorer=fuzz.ratio,
            limit=top_n
        )

        # Filtrar por threshold
        valid_matches = [(text, score, idx) for text, score, idx in matches if score >= self.threshold]

        if not valid_matches:
            logger.debug(
                f"No se encontró match para '{field_name}': {field_value_str} "
                f"(mejor score: {matches[0][1] if matches else 0})"
            )
            return None

        # Tomar el mejor match
        best_match_text, best_score, best_idx = valid_matches[0]
        word = pdf_words[best_idx]

        # Calcular confianza combinada (similitud × confianza OCR)
        combined_confidence = (best_score / 100.0) * word['confidence']

        result = {
            "text": word['text'],
            "bbox": word['bbox'],
            "confidence": combined_confidence,
            "page": word['page'],
            "match_score": best_score / 100.0,
            "field_name": field_name
        }

        logger.debug(
            f"Match difuso para '{field_name}': '{field_value_str}' → '{word['text']}' "
            f"(score: {best_score})"
        )

        return result

    def match_multiword(
        self,
        field_value: str,
        pdf_words: List[Dict],
        field_name: Optional[str] = None,
        window_size: int = 10
    ) -> Optional[Dict]:
        """
        Busca valores que pueden estar en múltiples palabras consecutivas

        Args:
            field_value: Valor a buscar (puede tener múltiples palabras)
            pdf_words: Lista de palabras extraídas del PDF
            field_name: Nombre del campo
            window_size: Tamaño de ventana para agrupar palabras

        Returns:
            Diccionario con el mejor match, con bbox que engloba todas las palabras
        """
        if not field_value or not pdf_words:
            return None

        field_value_str = str(field_value).strip()
        if not field_value_str:
            return None

        # Si es una sola palabra, usar match simple
        if len(field_value_str.split()) == 1:
            return self.match(field_value_str, pdf_words, field_name)

        # Crear ventanas deslizantes de palabras
        best_match = None
        best_score = 0

        for i in range(len(pdf_words)):
            for j in range(i + 1, min(i + window_size, len(pdf_words) + 1)):
                # Combinar palabras en la ventana
                window_words = pdf_words[i:j]
                window_text = ' '.join([w['text'] for w in window_words])

                # Calcular similitud
                normalized_window = self._normalize_text(window_text)
                normalized_query = self._normalize_text(field_value_str)

                score = fuzz.ratio(normalized_query, normalized_window)

                if score >= self.threshold and score > best_score:
                    # Calcular bbox que engloba todas las palabras
                    combined_bbox = self._combine_bboxes([w['bbox'] for w in window_words])

                    # Confianza promedio
                    avg_confidence = sum(w['confidence'] for w in window_words) / len(window_words)
                    combined_confidence = (score / 100.0) * avg_confidence

                    best_match = {
                        "text": window_text,
                        "bbox": combined_bbox,
                        "confidence": combined_confidence,
                        "page": window_words[0]['page'],
                        "match_score": score / 100.0,
                        "field_name": field_name,
                        "word_count": len(window_words)
                    }
                    best_score = score

        if best_match:
            logger.debug(
                f"Match multiword para '{field_name}': '{field_value_str}' → '{best_match['text']}' "
                f"(score: {best_score}, {best_match['word_count']} palabras)"
            )

        return best_match

    def match_numeric(
        self,
        field_value: float,
        pdf_words: List[Dict],
        field_name: Optional[str] = None,
        tolerance: float = 0.01
    ) -> Optional[Dict]:
        """
        Busca valores numéricos con tolerancia

        Args:
            field_value: Valor numérico a buscar
            pdf_words: Lista de palabras extraídas del PDF
            field_name: Nombre del campo
            tolerance: Tolerancia para diferencias numéricas (0.01 = 1%)

        Returns:
            Diccionario con el mejor match
        """
        if field_value is None or not pdf_words:
            return None

        # Convertir a float
        try:
            target_value = float(field_value)
        except (ValueError, TypeError):
            # Si no es numérico, usar match normal
            return self.match(str(field_value), pdf_words, field_name)

        # Buscar valores numéricos en el PDF
        best_match = None
        best_diff = float('inf')

        for word in pdf_words:
            # Intentar extraer número del texto
            text = word['text']
            number = self._extract_number(text)

            if number is not None:
                # Calcular diferencia relativa
                diff = abs(number - target_value) / max(abs(target_value), 0.01)

                if diff <= tolerance and diff < best_diff:
                    best_match = {
                        "text": text,
                        "bbox": word['bbox'],
                        "confidence": word['confidence'],
                        "page": word['page'],
                        "match_score": 1.0 - diff,  # Score basado en diferencia
                        "field_name": field_name,
                        "numeric_value": number
                    }
                    best_diff = diff

        if best_match:
            logger.debug(
                f"Match numérico para '{field_name}': {field_value} → {best_match['numeric_value']} "
                f"(diff: {best_diff * 100:.2f}%)"
            )

        return best_match

    def _normalize_text(self, text: str) -> str:
        """
        Normaliza texto para matching

        - Convierte a minúsculas
        - Quita acentos
        - Quita puntuación y espacios extras
        """
        if not text:
            return ""

        # Minúsculas
        text = text.lower()

        # Quitar acentos
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

        # Quitar puntuación y espacios extras
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _combine_bboxes(self, bboxes: List[List[int]]) -> List[int]:
        """
        Combina múltiples bboxes en uno que los engloba

        Args:
            bboxes: Lista de bboxes [[x0, y0, x1, y1], ...]

        Returns:
            Bbox combinado [x0, y0, x1, y1]
        """
        if not bboxes:
            return [0, 0, 0, 0]

        x0 = min(bbox[0] for bbox in bboxes)
        y0 = min(bbox[1] for bbox in bboxes)
        x1 = max(bbox[2] for bbox in bboxes)
        y1 = max(bbox[3] for bbox in bboxes)

        return [x0, y0, x1, y1]

    def _extract_number(self, text: str) -> Optional[float]:
        """
        Extrae número de un texto

        Args:
            text: Texto que puede contener un número

        Returns:
            Número como float, o None si no se puede extraer
        """
        # Quitar comas de miles y cambiar coma decimal por punto
        text = text.replace(',', '')
        text = text.replace('.', '')  # Quitar puntos de miles

        # Intentar extraer número con regex
        match = re.search(r'-?\d+\.?\d*', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return None
