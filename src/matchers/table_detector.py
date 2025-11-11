"""
Detector de tablas para identificar filas de items

Detecta automáticamente filas de tabla y mapea items del JSON
con sus correspondientes filas en el PDF.
"""
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TableDetector:
    """
    Detector de filas de tabla

    Algoritmo:
    1. Agrupar palabras por alineación vertical (Y similar)
    2. Ordenar palabras por X en cada fila
    3. Mapear filas con items del JSON
    4. Extraer campos de cada fila

    Ventaja:
    - 100 items × 1 búsqueda por fila = 100 búsquedas
    - vs 100 items × 10 campos = 1000 búsquedas individuales
    """

    def __init__(self, row_tolerance: int = 10, col_tolerance: int = 15):
        """
        Args:
            row_tolerance: Tolerancia vertical para agrupar palabras en la misma fila (píxeles normalizados)
            col_tolerance: Tolerancia horizontal para agrupar palabras en la misma columna (píxeles normalizados)
        """
        self.row_tolerance = row_tolerance
        self.col_tolerance = col_tolerance

    def detect_rows(self, pdf_words: List[Dict]) -> List[List[Dict]]:
        """
        Agrupa palabras en filas

        Args:
            pdf_words: Lista de palabras extraídas del PDF

        Returns:
            Lista de filas, donde cada fila es una lista de palabras:
            [
                [word1, word2, word3],  # Fila 1
                [word4, word5, word6],  # Fila 2
                ...
            ]
        """
        if not pdf_words:
            return []

        # Ordenar palabras por Y (top) y luego por X (left)
        sorted_words = sorted(pdf_words, key=lambda w: (w['bbox'][1], w['bbox'][0]))

        # Agrupar palabras con Y similar
        rows = []
        current_row = [sorted_words[0]]
        current_y = sorted_words[0]['bbox'][1]  # y0

        for word in sorted_words[1:]:
            word_y = word['bbox'][1]  # y0

            # Si la diferencia en Y es menor que la tolerancia, pertenece a la misma fila
            if abs(word_y - current_y) <= self.row_tolerance:
                current_row.append(word)
            else:
                # Nueva fila
                # Ordenar palabras de la fila actual por X
                current_row.sort(key=lambda w: w['bbox'][0])
                rows.append(current_row)

                # Iniciar nueva fila
                current_row = [word]
                current_y = word_y

        # Agregar última fila
        if current_row:
            current_row.sort(key=lambda w: w['bbox'][0])
            rows.append(current_row)

        logger.debug(f"Detectadas {len(rows)} filas de tabla")
        return rows

    def filter_table_rows(
        self,
        rows: List[List[Dict]],
        min_words_per_row: int = 3,
        max_words_per_row: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Filtra filas que probablemente pertenecen a una tabla

        Args:
            rows: Lista de filas detectadas
            min_words_per_row: Mínimo de palabras por fila para considerarla parte de tabla
            max_words_per_row: Máximo de palabras por fila (opcional)

        Returns:
            Lista de filas filtradas
        """
        filtered_rows = []

        for row in rows:
            word_count = len(row)

            # Filtrar por número de palabras
            if word_count < min_words_per_row:
                continue

            if max_words_per_row and word_count > max_words_per_row:
                continue

            filtered_rows.append(row)

        logger.debug(
            f"Filtradas {len(filtered_rows)}/{len(rows)} filas "
            f"(min_words={min_words_per_row})"
        )

        return filtered_rows

    def detect_columns(self, rows: List[List[Dict]]) -> List[Tuple[int, int]]:
        """
        Detecta columnas de la tabla basándose en alineación horizontal

        Args:
            rows: Lista de filas detectadas

        Returns:
            Lista de tuplas (x_start, x_end) para cada columna
        """
        if not rows:
            return []

        # Recolectar todas las posiciones X de inicio de palabras
        all_x_positions = []
        for row in rows:
            for word in row:
                all_x_positions.append(word['bbox'][0])  # x0

        if not all_x_positions:
            return []

        # Ordenar posiciones X
        all_x_positions.sort()

        # Agrupar posiciones cercanas
        columns = []
        current_col_start = all_x_positions[0]
        current_col_positions = [all_x_positions[0]]

        for x in all_x_positions[1:]:
            if abs(x - current_col_positions[-1]) <= self.col_tolerance:
                # Misma columna
                current_col_positions.append(x)
            else:
                # Nueva columna
                col_center = int(np.mean(current_col_positions))
                columns.append(col_center)

                current_col_start = x
                current_col_positions = [x]

        # Agregar última columna
        if current_col_positions:
            col_center = int(np.mean(current_col_positions))
            columns.append(col_center)

        # Convertir a rangos (start, end)
        column_ranges = []
        for i, col_x in enumerate(columns):
            if i < len(columns) - 1:
                next_col_x = columns[i + 1]
                mid_point = (col_x + next_col_x) // 2
                column_ranges.append((col_x, mid_point))
            else:
                # Última columna
                column_ranges.append((col_x, 1000))  # Hasta el borde derecho

        logger.debug(f"Detectadas {len(column_ranges)} columnas")
        return column_ranges

    def map_items_to_rows(
        self,
        rows: List[List[Dict]],
        items: List[Dict],
        matcher
    ) -> List[Dict]:
        """
        Mapea cada item del JSON con su fila en el PDF

        Args:
            rows: Filas detectadas
            items: Items del JSON
            matcher: Instancia de FuzzyMatcher

        Returns:
            Lista de items con coordenadas agregadas:
            [
                {
                    "item_data": {...},  # Datos originales del item
                    "row_index": 5,
                    "row_words": [...],
                    "matched_fields": {
                        "descripcion": {"text": "...", "bbox": [...], ...},
                        "cantidad": {...},
                        ...
                    }
                },
                ...
            ]
        """
        if not items or not rows:
            return []

        mapped_items = []

        # Para cada item, buscar la fila que mejor coincida
        for item in items:
            best_row_idx = None
            best_score = 0
            best_matched_fields = {}

            # Probar cada fila
            for row_idx, row in enumerate(rows):
                # Contar cuántos campos del item se encuentran en esta fila
                matched_fields = {}
                total_score = 0

                for field_name, field_value in item.items():
                    if field_value is None or field_value == "":
                        continue

                    # Buscar campo en las palabras de la fila
                    match_result = matcher.match(str(field_value), row, f"item.{field_name}")

                    if match_result:
                        matched_fields[field_name] = match_result
                        total_score += match_result.get('match_score', 0)

                # Si esta fila tiene más matches, es mejor candidata
                if len(matched_fields) > len(best_matched_fields) or \
                   (len(matched_fields) == len(best_matched_fields) and total_score > best_score):
                    best_row_idx = row_idx
                    best_score = total_score
                    best_matched_fields = matched_fields

            # Agregar item mapeado
            if best_row_idx is not None:
                mapped_items.append({
                    "item_data": item,
                    "row_index": best_row_idx,
                    "row_words": rows[best_row_idx],
                    "matched_fields": best_matched_fields
                })

        logger.info(
            f"Mapeados {len(mapped_items)}/{len(items)} items a filas de tabla"
        )

        return mapped_items

    def get_row_bbox(self, row_words: List[Dict]) -> List[int]:
        """
        Calcula el bbox que engloba toda la fila

        Args:
            row_words: Palabras de la fila

        Returns:
            Bbox [x0, y0, x1, y1]
        """
        if not row_words:
            return [0, 0, 0, 0]

        x0 = min(word['bbox'][0] for word in row_words)
        y0 = min(word['bbox'][1] for word in row_words)
        x1 = max(word['bbox'][2] for word in row_words)
        y1 = max(word['bbox'][3] for word in row_words)

        return [x0, y0, x1, y1]

    def extract_column_values(
        self,
        rows: List[List[Dict]],
        column_ranges: List[Tuple[int, int]]
    ) -> List[List[str]]:
        """
        Extrae valores de cada columna para cada fila

        Args:
            rows: Filas detectadas
            column_ranges: Rangos de columnas detectadas

        Returns:
            Matriz de valores [fila][columna]
        """
        result = []

        for row in rows:
            row_values = []

            for col_start, col_end in column_ranges:
                # Encontrar palabras que caen en este rango de columna
                col_words = []

                for word in row:
                    word_x = word['bbox'][0]
                    if col_start <= word_x < col_end:
                        col_words.append(word['text'])

                # Combinar palabras de la columna
                col_value = ' '.join(col_words) if col_words else ""
                row_values.append(col_value)

            result.append(row_values)

        return result

    def detect_table_region(
        self,
        pdf_words: List[Dict],
        min_rows: int = 3
    ) -> Optional[Dict]:
        """
        Detecta la región rectangular que contiene la tabla

        Args:
            pdf_words: Palabras del PDF
            min_rows: Mínimo de filas para considerar que hay una tabla

        Returns:
            Diccionario con información de la tabla:
            {
                "bbox": [x0, y0, x1, y1],
                "rows": [[palabras], ...],
                "num_rows": 10,
                "num_columns": 6,
                "column_ranges": [(x0, x1), ...]
            }

            O None si no se detecta tabla
        """
        # Detectar filas
        rows = self.detect_rows(pdf_words)

        # Filtrar filas de tabla
        table_rows = self.filter_table_rows(rows, min_words_per_row=3)

        if len(table_rows) < min_rows:
            logger.debug(f"No se detectó tabla (solo {len(table_rows)} filas)")
            return None

        # Detectar columnas
        columns = self.detect_columns(table_rows)

        # Calcular bbox de la tabla
        all_table_words = [word for row in table_rows for word in row]
        table_bbox = self.get_row_bbox(all_table_words)

        return {
            "bbox": table_bbox,
            "rows": table_rows,
            "num_rows": len(table_rows),
            "num_columns": len(columns),
            "column_ranges": columns
        }
