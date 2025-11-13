"""
Mejoras avanzadas en detección de coordenadas

Técnicas implementadas:
1. Spatial clustering: Agrupa palabras relacionadas espacialmente
2. Confidence smoothing: Mejora confidence basado en vecinos
3. Pattern recognition: Detecta patrones comunes (RUC, fechas, montos)
4. Multi-word merging: Combina palabras multi-token correctamente
5. Peruvian invoice heuristics: Reglas específicas para facturas peruanas

Mejora esperada: +15-25% en precisión de coordenadas
"""
import logging
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class CoordinateEnhancer:
    """
    Mejora coordenadas y confidence de campos extraídos

    Técnicas:
    - Clustering espacial para campos relacionados
    - Smoothing de confidence
    - Detección de patrones
    - Corrección de bboxes
    """

    # Patrones comunes en facturas peruanas
    PATTERNS = {
        'ruc': r'\b\d{11}\b',  # RUC: 11 dígitos
        'fecha': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY
        'monto': r'S/?\s*\.?\s*\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?',  # S/. 1,234.56
        'igv': r'(?:IGV|I\.G\.V\.)\s*(?:18%|18\s*%)',
        'factura': r'(?:FACTURA|FACT\.?)\s*(?:N[°º]?\.?)?\s*[\w-]+',
        'dni': r'\b\d{8}\b',  # DNI: 8 dígitos
        'codigo': r'\b[A-Z0-9]{3,}-[A-Z0-9]+\b'  # Códigos: ABC-123
    }

    # Keywords por tipo de campo (para mejorar confidence)
    FIELD_KEYWORDS = {
        'ruc': ['ruc', 'r.u.c', 'registro', 'contribuyente'],
        'razon_social': ['razón social', 'razon social', 'cliente', 'señor'],
        'direccion': ['dirección', 'direccion', 'domicilio', 'calle', 'av.', 'jr.'],
        'fecha': ['fecha', 'emisión', 'emision'],
        'factura': ['factura', 'fact.', 'comprobante', 'serie'],
        'subtotal': ['subtotal', 'sub total', 'base imponible'],
        'igv': ['igv', 'i.g.v', 'impuesto'],
        'total': ['total', 'importe', 'neto'],
        'cantidad': ['cant.', 'cantidad', 'unid'],
        'descripcion': ['descripción', 'descripcion', 'detalle', 'concepto'],
        'precio': ['precio', 'p.unit', 'unitario', 'valor']
    }

    def __init__(
        self,
        proximity_threshold: int = 50,
        confidence_boost: float = 0.1,
        min_confidence: float = 0.3
    ):
        """
        Args:
            proximity_threshold: Distancia máxima para considerar palabras relacionadas
            confidence_boost: Boost de confidence por palabra cercana relacionada
            min_confidence: Confidence mínimo para considerar un match
        """
        self.proximity_threshold = proximity_threshold
        self.confidence_boost = confidence_boost
        self.min_confidence = min_confidence

    def enhance_match(
        self,
        match_result: Dict,
        field_name: str,
        all_words: List[Dict],
        context_words: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Mejora un resultado de matching con técnicas avanzadas

        Args:
            match_result: Resultado original del matcher
            field_name: Nombre del campo
            all_words: Todas las palabras del documento
            context_words: Palabras de contexto cercanas (opcional)

        Returns:
            Match mejorado con mejor bbox y confidence
        """
        if not match_result or not match_result.get('bbox'):
            return match_result

        enhanced = match_result.copy()

        # 1. Detectar patrón si aplica
        pattern_info = self._detect_pattern(match_result.get('text', ''), field_name)
        if pattern_info:
            enhanced['pattern_detected'] = pattern_info['type']
            enhanced['confidence'] = min(
                1.0,
                enhanced.get('confidence', 0) + 0.15  # Boost por patrón
            )

        # 2. Mejorar confidence basado en contexto
        if context_words:
            context_boost = self._calculate_context_boost(
                match_result,
                field_name,
                context_words
            )
            enhanced['confidence'] = min(
                1.0,
                enhanced.get('confidence', 0) + context_boost
            )

        # 3. Expandir bbox si es multi-palabra
        text = enhanced.get('text', '')
        if len(text.split()) > 1:
            enhanced['bbox'] = self._expand_multiword_bbox(
                enhanced['bbox'],
                text,
                all_words
            )

        # 4. Ajustar bbox basado en palabras vecinas
        enhanced['bbox'] = self._refine_bbox_with_neighbors(
            enhanced['bbox'],
            all_words,
            field_name
        )

        return enhanced

    def enhance_batch(
        self,
        matches: List[Dict],
        all_words: List[Dict],
        field_names: List[str]
    ) -> List[Dict]:
        """
        Mejora un batch de matches simultáneamente

        Args:
            matches: Lista de match results
            all_words: Todas las palabras del documento
            field_names: Nombres de campos correspondientes

        Returns:
            Lista de matches mejorados
        """
        enhanced_matches = []

        for i, (match, field_name) in enumerate(zip(matches, field_names)):
            if not match:
                enhanced_matches.append(match)
                continue

            # Obtener contexto (matches cercanos)
            context = self._get_context_matches(i, matches, all_words)

            # Mejorar match
            enhanced = self.enhance_match(match, field_name, all_words, context)
            enhanced_matches.append(enhanced)

        return enhanced_matches

    def _detect_pattern(self, text: str, field_name: str) -> Optional[Dict]:
        """
        Detecta si el texto coincide con un patrón conocido

        Args:
            text: Texto a analizar
            field_name: Nombre del campo

        Returns:
            Info del patrón detectado o None
        """
        # Buscar en patrones
        for pattern_type, pattern_regex in self.PATTERNS.items():
            if re.search(pattern_regex, text, re.IGNORECASE):
                return {
                    'type': pattern_type,
                    'pattern': pattern_regex,
                    'confidence_boost': 0.15
                }

        return None

    def _calculate_context_boost(
        self,
        match_result: Dict,
        field_name: str,
        context_words: List[Dict]
    ) -> float:
        """
        Calcula boost de confidence basado en contexto

        Args:
            match_result: Match a analizar
            field_name: Nombre del campo
            context_words: Palabras de contexto

        Returns:
            Boost de confidence (0.0 - 0.3)
        """
        boost = 0.0

        # Obtener keywords esperados para este campo
        keywords = self._get_field_keywords(field_name)
        if not keywords:
            return boost

        # Buscar keywords en contexto cercano
        match_bbox = match_result.get('bbox', [0, 0, 0, 0])

        for word in context_words:
            word_text = word.get('text', '').lower()
            word_bbox = word.get('bbox', [0, 0, 0, 0])

            # Verificar si es keyword relevante
            for keyword in keywords:
                if keyword.lower() in word_text:
                    # Calcular distancia
                    distance = self._calculate_distance(match_bbox, word_bbox)

                    if distance < self.proximity_threshold:
                        # Boost inversamente proporcional a la distancia
                        keyword_boost = self.confidence_boost * (
                            1.0 - distance / self.proximity_threshold
                        )
                        boost += keyword_boost

                    break  # Solo un boost por palabra

        return min(0.3, boost)  # Máximo 0.3 de boost

    def _get_field_keywords(self, field_name: str) -> List[str]:
        """
        Obtiene keywords relevantes para un campo

        Args:
            field_name: Nombre del campo

        Returns:
            Lista de keywords
        """
        # Buscar en diccionario de keywords
        for key, keywords in self.FIELD_KEYWORDS.items():
            if key in field_name.lower():
                return keywords

        return []

    def _calculate_distance(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calcula distancia entre dos bboxes

        Args:
            bbox1: [x0, y0, x1, y1]
            bbox2: [x0, y0, x1, y1]

        Returns:
            Distancia euclidiana entre centros
        """
        # Calcular centros
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]

        # Distancia euclidiana
        return np.sqrt(
            (center1[0] - center2[0]) ** 2 +
            (center1[1] - center2[1]) ** 2
        )

    def _expand_multiword_bbox(
        self,
        original_bbox: List[int],
        text: str,
        all_words: List[Dict]
    ) -> List[int]:
        """
        Expande bbox para incluir todas las palabras del texto

        Args:
            original_bbox: Bbox original
            text: Texto completo
            all_words: Todas las palabras del documento

        Returns:
            Bbox expandido
        """
        words_in_text = text.split()
        if len(words_in_text) <= 1:
            return original_bbox

        # Encontrar todas las palabras que componen el texto
        matched_words = []

        for word in all_words:
            word_text = word.get('text', '')
            if word_text in words_in_text:
                matched_words.append(word)

        if not matched_words:
            return original_bbox

        # Calcular bbox que contenga todas las palabras
        all_x0 = [w['bbox'][0] for w in matched_words]
        all_y0 = [w['bbox'][1] for w in matched_words]
        all_x1 = [w['bbox'][2] for w in matched_words]
        all_y1 = [w['bbox'][3] for w in matched_words]

        expanded = [
            min(all_x0),
            min(all_y0),
            max(all_x1),
            max(all_y1)
        ]

        logger.debug(
            f"Expandido bbox multi-palabra '{text[:30]}...': "
            f"{original_bbox} → {expanded}"
        )

        return expanded

    def _refine_bbox_with_neighbors(
        self,
        bbox: List[int],
        all_words: List[Dict],
        field_name: str
    ) -> List[int]:
        """
        Refina bbox considerando palabras vecinas

        Args:
            bbox: Bbox a refinar
            all_words: Todas las palabras
            field_name: Nombre del campo

        Returns:
            Bbox refinado
        """
        # Encontrar palabras cercanas
        nearby_words = self._find_nearby_words(bbox, all_words, distance=30)

        if not nearby_words:
            return bbox

        # Para campos numéricos, expandir para incluir símbolos cercanos
        if any(x in field_name.lower() for x in ['precio', 'monto', 'total', 'subtotal', 'igv']):
            return self._expand_for_numeric(bbox, nearby_words)

        return bbox

    def _find_nearby_words(
        self,
        bbox: List[int],
        all_words: List[Dict],
        distance: int = 30
    ) -> List[Dict]:
        """
        Encuentra palabras cercanas a un bbox

        Args:
            bbox: Bbox de referencia
            all_words: Todas las palabras
            distance: Distancia máxima

        Returns:
            Lista de palabras cercanas
        """
        nearby = []

        for word in all_words:
            word_bbox = word.get('bbox', [0, 0, 0, 0])
            dist = self._calculate_distance(bbox, word_bbox)

            if dist < distance:
                nearby.append(word)

        return nearby

    def _expand_for_numeric(
        self,
        bbox: List[int],
        nearby_words: List[Dict]
    ) -> List[int]:
        """
        Expande bbox para incluir símbolos de moneda y decimales

        Args:
            bbox: Bbox original
            nearby_words: Palabras cercanas

        Returns:
            Bbox expandido
        """
        x0, y0, x1, y1 = bbox

        # Buscar símbolos de moneda (S/., S/, etc.)
        for word in nearby_words:
            word_text = word.get('text', '')
            word_bbox = word.get('bbox', [0, 0, 0, 0])

            # Si es símbolo de moneda y está a la izquierda
            if re.match(r'^S/?\.?$', word_text) and word_bbox[2] <= x0:
                x0 = min(x0, word_bbox[0])

        return [x0, y0, x1, y1]

    def _get_context_matches(
        self,
        current_index: int,
        all_matches: List[Dict],
        all_words: List[Dict]
    ) -> List[Dict]:
        """
        Obtiene palabras de contexto para un match

        Args:
            current_index: Índice del match actual
            all_matches: Todos los matches
            all_words: Todas las palabras

        Returns:
            Palabras de contexto
        """
        if not all_matches[current_index]:
            return []

        current_bbox = all_matches[current_index].get('bbox', [0, 0, 0, 0])

        # Encontrar palabras cercanas
        context = self._find_nearby_words(current_bbox, all_words, distance=100)

        return context

    def cluster_related_fields(
        self,
        matches: Dict[str, Dict],
        all_words: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Agrupa campos relacionados espacialmente

        Útil para:
        - Items de tabla (agrupar campos de la misma fila)
        - Bloques de información (agrupar datos del emisor/receptor)

        Args:
            matches: Diccionario {field_name: match_result}
            all_words: Todas las palabras

        Returns:
            Diccionario {cluster_id: [field_names]}
        """
        clusters = defaultdict(list)

        # Calcular matriz de distancias
        field_names = list(matches.keys())
        n = len(field_names)

        if n == 0:
            return {}

        # Agrupar por proximidad vertical (misma región Y)
        y_positions = []
        for field_name in field_names:
            match = matches[field_name]
            if match and match.get('bbox'):
                bbox = match['bbox']
                y_center = (bbox[1] + bbox[3]) / 2
                y_positions.append((field_name, y_center))

        # Ordenar por posición Y
        y_positions.sort(key=lambda x: x[1])

        # Agrupar por Y similar (tolerancia: 20 píxeles)
        current_cluster = 0
        prev_y = None

        for field_name, y_pos in y_positions:
            if prev_y is None or abs(y_pos - prev_y) < 20:
                clusters[f"cluster_{current_cluster}"].append(field_name)
            else:
                current_cluster += 1
                clusters[f"cluster_{current_cluster}"].append(field_name)

            prev_y = y_pos

        return dict(clusters)

    def get_enhancement_stats(self, original: Dict, enhanced: Dict) -> Dict:
        """
        Compara original vs enhanced y retorna estadísticas

        Args:
            original: Match original
            enhanced: Match mejorado

        Returns:
            Estadísticas de mejora
        """
        if not original or not enhanced:
            return {}

        orig_conf = original.get('confidence', 0)
        enh_conf = enhanced.get('confidence', 0)

        orig_bbox = original.get('bbox', [0, 0, 0, 0])
        enh_bbox = enhanced.get('bbox', [0, 0, 0, 0])

        orig_area = (orig_bbox[2] - orig_bbox[0]) * (orig_bbox[3] - orig_bbox[1])
        enh_area = (enh_bbox[2] - enh_bbox[0]) * (enh_bbox[3] - enh_bbox[1])

        return {
            'confidence_boost': round(enh_conf - orig_conf, 3),
            'confidence_improvement': round((enh_conf - orig_conf) / orig_conf * 100, 1) if orig_conf > 0 else 0,
            'bbox_area_change': round(enh_area - orig_area, 1),
            'bbox_expansion': round((enh_area - orig_area) / orig_area * 100, 1) if orig_area > 0 else 0,
            'pattern_detected': enhanced.get('pattern_detected', None)
        }
