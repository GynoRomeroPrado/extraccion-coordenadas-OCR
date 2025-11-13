"""
Detector de tablas mejorado con análisis visual avanzado

Mejoras sobre V1:
- Usa RegionDetector para identificar zona de items
- Análisis visual de estructura de tabla (líneas, columnas)
- Detección de headers de columna
- Mejor alineación de campos con columnas
- Soporte para tablas multi-página
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from config import Config
from .region_detector import RegionDetector

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TableDetectorV2:
    """
    Detector de tablas mejorado con análisis visual

    Algoritmo avanzado:
    1. Usar RegionDetector para encontrar zona de items
    2. Detectar headers de columna
    3. Identificar columnas por alineación X
    4. Agrupar palabras en filas
    5. Mapear filas con items del JSON
    6. Validar estructura de tabla
    """

    def __init__(self, row_tolerance: int = 10, col_tolerance: int = 15):
        """
        Args:
            row_tolerance: Tolerancia vertical para filas
            col_tolerance: Tolerancia horizontal para columnas
        """
        self.row_tolerance = row_tolerance
        self.col_tolerance = col_tolerance
        self.region_detector = RegionDetector()

    def detect_table_structure(
        self,
        pdf_words: List[Dict]
    ) -> Dict:
        """
        Detecta estructura completa de la tabla

        Args:
            pdf_words: Palabras del PDF

        Returns:
            Diccionario con:
            {
                "region": {"y_start": 250, "y_end": 700},
                "columns": [(x_start, x_end, "descripcion"), ...],
                "header_row": [...],
                "data_rows": [[...], [...], ...],
                "num_columns": 5
            }
        """
        if not pdf_words:
            return self._empty_structure()

        # 1. Detectar regiones del documento
        regions = self.region_detector.detect_regions(pdf_words)
        items_region = regions['items']

        if items_region['word_count'] == 0:
            logger.warning("No se detectó región de items")
            return self._empty_structure()

        # Filtrar solo palabras en la región de items
        table_words = items_region['words']

        logger.info(
            f"Región de tabla detectada: Y[{items_region['y_start']}-{items_region['y_end']}], "
            f"{len(table_words)} palabras"
        )

        # 2. Detectar columnas
        columns = self._detect_columns(table_words)

        # 3. Detectar header de tabla
        header_row = self._detect_table_header(table_words, columns)

        # 4. Detectar filas de datos
        data_rows = self._detect_data_rows(table_words, columns, items_region['y_start'])

        return {
            "region": {
                "y_start": items_region['y_start'],
                "y_end": items_region['y_end']
            },
            "columns": columns,
            "header_row": header_row,
            "data_rows": data_rows,
            "num_columns": len(columns),
            "num_rows": len(data_rows)
        }

    def _detect_columns(self, table_words: List[Dict]) -> List[Tuple[int, int, str]]:
        """
        Detecta columnas de la tabla basándose en alineación X

        Args:
            table_words: Palabras en la región de tabla

        Returns:
            Lista de tuplas (x_start, x_end, column_name)
        """
        if not table_words:
            return []

        # Recolectar posiciones X de inicio de palabras
        x_positions = defaultdict(int)

        for word in table_words:
            bbox = word.get('bbox', [])
            if len(bbox) >= 4:
                x0 = bbox[0]
                # Agrupar en bandas de col_tolerance píxeles
                x_band = (x0 // self.col_tolerance) * self.col_tolerance
                x_positions[x_band] += 1

        # Filtrar posiciones con pocas palabras (ruido)
        min_words_per_column = len(table_words) * 0.05  # 5% de palabras mínimo

        significant_x = sorted([
            x for x, count in x_positions.items()
            if count >= min_words_per_column
        ])

        if not significant_x:
            return []

        # Crear columnas
        columns = []
        for i, x_start in enumerate(significant_x):
            if i < len(significant_x) - 1:
                x_end = significant_x[i + 1] - self.col_tolerance
            else:
                x_end = 1000  # Borde derecho

            # Intentar detectar nombre de columna
            column_name = self._infer_column_name(x_start, x_end, table_words)

            columns.append((x_start, x_end, column_name))

        logger.info(f"Detectadas {len(columns)} columnas")
        return columns

    def _infer_column_name(
        self,
        x_start: int,
        x_end: int,
        table_words: List[Dict]
    ) -> str:
        """
        Intenta inferir el nombre de la columna basándose en palabras clave

        Args:
            x_start: Inicio X de la columna
            x_end: Fin X de la columna
            table_words: Palabras de la tabla

        Returns:
            Nombre inferido de la columna
        """
        # Buscar palabras clave de columnas comunes
        column_keywords = {
            'item': 'item',
            'codigo': 'codigo',
            'descripcion': 'descripcion',
            'cantidad': 'cantidad',
            'unidad': 'unidad',
            'precio': 'precio',
            'unitario': 'precio_unitario',
            'valor': 'valor_venta',
            'importe': 'importe',
            'total': 'total'
        }

        # Buscar en palabras de la parte superior (posible header)
        min_y = min(w['bbox'][1] for w in table_words if len(w.get('bbox', [])) >= 2)

        for word in table_words:
            bbox = word.get('bbox', [])
            if len(bbox) >= 4:
                x0, y0 = bbox[0], bbox[1]

                # Si está en la columna y cerca del top
                if x_start <= x0 < x_end and y0 < min_y + 50:
                    text = word.get('text', '').lower()

                    for keyword, col_name in column_keywords.items():
                        if keyword in text:
                            return col_name

        return f"col_{x_start}"

    def _detect_table_header(
        self,
        table_words: List[Dict],
        columns: List[Tuple[int, int, str]]
    ) -> List[Dict]:
        """
        Detecta la fila de header de la tabla

        Args:
            table_words: Palabras de la tabla
            columns: Columnas detectadas

        Returns:
            Lista de palabras del header
        """
        if not table_words:
            return []

        # El header suele estar en la primera fila (Y más pequeño)
        min_y = min(w['bbox'][1] for w in table_words if len(w.get('bbox', [])) >= 2)

        header_words = []
        for word in table_words:
            bbox = word.get('bbox', [])
            if len(bbox) >= 2:
                y = bbox[1]
                # Considerar header si está en los primeros 30 píxeles
                if y < min_y + 30:
                    header_words.append(word)

        return header_words

    def _detect_data_rows(
        self,
        table_words: List[Dict],
        columns: List[Tuple[int, int, str]],
        region_y_start: int
    ) -> List[List[Dict]]:
        """
        Detecta filas de datos (excluyendo header)

        Args:
            table_words: Palabras de la tabla
            columns: Columnas detectadas
            region_y_start: Y de inicio de la región

        Returns:
            Lista de filas, cada fila es lista de palabras
        """
        # Agrupar palabras por Y (filas)
        rows_dict = defaultdict(list)

        for word in table_words:
            bbox = word.get('bbox', [])
            if len(bbox) >= 2:
                y = bbox[1]
                # Agrupar en bandas de row_tolerance
                y_band = (y // self.row_tolerance) * self.row_tolerance
                rows_dict[y_band].append(word)

        # Ordenar filas por Y
        sorted_y = sorted(rows_dict.keys())

        # Excluir primera fila (probablemente header)
        if len(sorted_y) > 1:
            data_y = sorted_y[1:]
        else:
            data_y = sorted_y

        # Crear lista de filas
        data_rows = []
        for y in data_y:
            row_words = rows_dict[y]
            # Ordenar palabras por X dentro de la fila
            row_words.sort(key=lambda w: w.get('bbox', [0])[0])
            data_rows.append(row_words)

        logger.info(f"Detectadas {len(data_rows)} filas de datos")
        return data_rows

    def map_items_to_rows(
        self,
        table_structure: Dict,
        items: List[Dict],
        matcher
    ) -> List[Dict]:
        """
        Mapea items del JSON con filas de la tabla

        Args:
            table_structure: Estructura de tabla detectada
            items: Items del JSON
            matcher: Instancia de FuzzyMatcher

        Returns:
            Lista de items con coordenadas
        """
        if not items or not table_structure['data_rows']:
            return []

        mapped_items = []
        data_rows = table_structure['data_rows']
        columns = table_structure['columns']

        # Para cada item, buscar la fila que mejor coincida
        for item_idx, item in enumerate(items):
            best_row_idx = None
            best_match_score = 0
            best_matched_fields = {}

            # Probar cada fila
            for row_idx, row_words in enumerate(data_rows):
                matched_fields = {}
                total_score = 0

                # Intentar matchear cada campo del item
                for field_name, field_value in item.items():
                    if field_value is None or field_value == "":
                        continue

                    # Buscar en palabras de la fila
                    match_result = matcher.match(str(field_value), row_words, f"item.{field_name}")

                    if match_result:
                        matched_fields[field_name] = match_result
                        total_score += match_result.get('match_score', 0)

                # Si esta fila tiene mejor score, es mejor candidata
                if len(matched_fields) > len(best_matched_fields) or \
                   (len(matched_fields) == len(best_matched_fields) and total_score > best_match_score):
                    best_row_idx = row_idx
                    best_match_score = total_score
                    best_matched_fields = matched_fields

            # Agregar item mapeado
            if best_row_idx is not None:
                mapped_items.append({
                    "item_index": item_idx,
                    "item_data": item,
                    "row_index": best_row_idx,
                    "row_words": data_rows[best_row_idx],
                    "matched_fields": best_matched_fields,
                    "match_score": best_match_score,
                    "fields_matched": len(best_matched_fields)
                })

        logger.info(
            f"Mapeados {len(mapped_items)}/{len(items)} items "
            f"(promedio {sum(m['fields_matched'] for m in mapped_items) / len(mapped_items):.1f} campos/item)"
        )

        return mapped_items

    def extract_from_columns(
        self,
        row_words: List[Dict],
        columns: List[Tuple[int, int, str]]
    ) -> Dict[str, str]:
        """
        Extrae valores de cada columna en una fila

        Args:
            row_words: Palabras de la fila
            columns: Definición de columnas

        Returns:
            Diccionario {column_name: value}
        """
        row_values = {}

        for x_start, x_end, col_name in columns:
            # Buscar palabras en este rango X
            col_words = []

            for word in row_words:
                bbox = word.get('bbox', [])
                if len(bbox) >= 4:
                    word_x = bbox[0]
                    if x_start <= word_x < x_end:
                        col_words.append(word['text'])

            # Combinar palabras de la columna
            if col_words:
                row_values[col_name] = ' '.join(col_words)

        return row_values

    def _empty_structure(self) -> Dict:
        """Retorna estructura vacía"""
        return {
            "region": {"y_start": 0, "y_end": 0},
            "columns": [],
            "header_row": [],
            "data_rows": [],
            "num_columns": 0,
            "num_rows": 0
        }

    def get_table_stats(self, table_structure: Dict) -> Dict:
        """
        Calcula estadísticas de la tabla

        Args:
            table_structure: Estructura detectada

        Returns:
            Diccionario con estadísticas
        """
        return {
            "num_columns": table_structure['num_columns'],
            "num_rows": table_structure['num_rows'],
            "region_height": table_structure['region']['y_end'] - table_structure['region']['y_start'],
            "has_header": len(table_structure['header_row']) > 0,
            "columns_detected": [col[2] for col in table_structure['columns']]
        }
