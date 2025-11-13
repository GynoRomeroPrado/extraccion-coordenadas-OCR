"""
Detector de regiones de documento (header, items/tabla, totales)

Identifica automáticamente las zonas del documento usando:
- Análisis de densidad de texto
- Detección de patrones visuales (líneas, tablas)
- Heurísticas basadas en coordenadas Y
- Análisis de contenido (palabras clave)
"""
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class RegionDetector:
    """
    Detector de regiones del documento

    Identifica automáticamente:
    - Header (info general del documento)
    - Items/Tabla (productos/servicios)
    - Totales (subtotal, IGV, total)
    """

    # Palabras clave para detectar regiones
    TOTAL_KEYWORDS = {
        'subtotal', 'igv', 'total', 'importe', 'neto', 'valor',
        'descuento', 'suma', 'monto', 'pagar', 'cobrar'
    }

    ITEM_KEYWORDS = {
        'item', 'cantidad', 'descripcion', 'precio', 'unitario',
        'codigo', 'producto', 'servicio', 'unidad', 'medida'
    }

    HEADER_KEYWORDS = {
        'factura', 'ruc', 'razon', 'social', 'emisor', 'receptor',
        'fecha', 'emision', 'serie', 'numero', 'direccion'
    }

    def __init__(self, page_height: int = 1000):
        """
        Args:
            page_height: Altura de la página normalizada (default: 1000)
        """
        self.page_height = page_height

    def detect_regions(self, pdf_words: List[Dict]) -> Dict[str, Dict]:
        """
        Detecta regiones del documento

        Args:
            pdf_words: Lista de palabras extraídas del PDF

        Returns:
            Diccionario con regiones detectadas:
            {
                "header": {"y_start": 0, "y_end": 250, "words": [...]},
                "items": {"y_start": 250, "y_end": 700, "words": [...]},
                "totals": {"y_start": 700, "y_end": 1000, "words": [...]}
            }
        """
        if not pdf_words:
            return self._empty_regions()

        # Análisis 1: Densidad de palabras por franja vertical
        density_map = self._calculate_density_map(pdf_words)

        # Análisis 2: Detectar tabla (alta densidad de filas alineadas)
        table_region = self._detect_table_region(pdf_words)

        # Análisis 3: Detectar palabras clave de totales
        totals_region = self._detect_totals_region(pdf_words)

        # Análisis 4: Combinar información
        regions = self._combine_analyses(
            pdf_words,
            density_map,
            table_region,
            totals_region
        )

        logger.info(
            f"Regiones detectadas: "
            f"header[{regions['header']['y_start']}-{regions['header']['y_end']}], "
            f"items[{regions['items']['y_start']}-{regions['items']['y_end']}], "
            f"totals[{regions['totals']['y_start']}-{regions['totals']['y_end']}]"
        )

        return regions

    def _calculate_density_map(
        self,
        pdf_words: List[Dict],
        num_bands: int = 20
    ) -> List[Tuple[int, int, int]]:
        """
        Calcula densidad de palabras por franjas verticales

        Args:
            pdf_words: Palabras del PDF
            num_bands: Número de franjas verticales

        Returns:
            Lista de tuplas (y_start, y_end, word_count)
        """
        band_height = self.page_height // num_bands
        density = [0] * num_bands

        for word in pdf_words:
            bbox = word.get('bbox', [])
            if len(bbox) >= 2:
                y = bbox[1]  # y0
                band_idx = min(int(y / band_height), num_bands - 1)
                density[band_idx] += 1

        # Convertir a lista de tuplas
        density_map = []
        for i, count in enumerate(density):
            y_start = i * band_height
            y_end = (i + 1) * band_height
            density_map.append((y_start, y_end, count))

        return density_map

    def _detect_table_region(self, pdf_words: List[Dict]) -> Optional[Tuple[int, int]]:
        """
        Detecta región de tabla basándose en alineación de palabras

        Args:
            pdf_words: Palabras del PDF

        Returns:
            Tupla (y_start, y_end) o None
        """
        # Agrupar palabras por Y (filas)
        rows = defaultdict(list)
        for word in pdf_words:
            bbox = word.get('bbox', [])
            if len(bbox) >= 2:
                y = bbox[1]  # y0
                # Agrupar en bandas de 10 píxeles
                y_band = (y // 10) * 10
                rows[y_band].append(word)

        # Encontrar región con más filas con múltiples palabras
        # (indicativo de tabla)
        table_rows = []
        for y, words in rows.items():
            if len(words) >= 3:  # Fila con al menos 3 palabras
                table_rows.append((y, len(words)))

        if not table_rows:
            return None

        # Ordenar por Y
        table_rows.sort()

        # Encontrar la región más grande de filas consecutivas
        if len(table_rows) < 2:
            return None

        # Buscar la región más densa
        max_density = 0
        best_region = None

        for i in range(len(table_rows)):
            for j in range(i + 2, len(table_rows) + 1):
                y_start = table_rows[i][0]
                y_end = table_rows[j - 1][0] + 20  # +20 para incluir altura de fila
                density = sum(row[1] for row in table_rows[i:j])

                if density > max_density:
                    max_density = density
                    best_region = (y_start, y_end)

        return best_region

    def _detect_totals_region(self, pdf_words: List[Dict]) -> Optional[Tuple[int, int]]:
        """
        Detecta región de totales usando palabras clave

        Args:
            pdf_words: Palabras del PDF

        Returns:
            Tupla (y_start, y_end) o None
        """
        total_words = []

        for word in pdf_words:
            text = word.get('text', '').lower()
            bbox = word.get('bbox', [])

            if len(bbox) >= 2:
                # Buscar palabras clave de totales
                if any(keyword in text for keyword in self.TOTAL_KEYWORDS):
                    total_words.append((bbox[1], word))  # (y, word)

        if not total_words:
            return None

        # Ordenar por Y
        total_words.sort()

        # Tomar primer y último total para definir región
        y_start = total_words[0][0]
        y_end = total_words[-1][0] + 30  # +30 píxeles de margen

        # Expandir hacia arriba para capturar todo el bloque de totales
        y_start = max(0, y_start - 50)

        return (y_start, y_end)

    def _combine_analyses(
        self,
        pdf_words: List[Dict],
        density_map: List[Tuple[int, int, int]],
        table_region: Optional[Tuple[int, int]],
        totals_region: Optional[Tuple[int, int]]
    ) -> Dict[str, Dict]:
        """
        Combina todos los análisis para determinar regiones finales

        Args:
            pdf_words: Palabras del PDF
            density_map: Mapa de densidad
            table_region: Región de tabla detectada
            totals_region: Región de totales detectada

        Returns:
            Diccionario con regiones
        """
        # Estrategia de combinación:
        # 1. Si hay región de totales, esa es la región de totales
        # 2. Si hay región de tabla, esa es la región de items
        # 3. Lo demás es header

        if totals_region:
            totals_start, totals_end = totals_region
        else:
            # Asumir que el 20% inferior es totales
            totals_start = int(self.page_height * 0.80)
            totals_end = self.page_height

        if table_region:
            items_start, items_end = table_region
            # Ajustar si overlap con totales
            if items_end > totals_start:
                items_end = totals_start - 10
        else:
            # Asumir que del 25% al 80% es items
            items_start = int(self.page_height * 0.25)
            items_end = totals_start - 10

        # Header es todo lo anterior a items
        header_start = 0
        header_end = items_start - 10

        # Asignar palabras a cada región
        header_words = []
        items_words = []
        totals_words = []

        for word in pdf_words:
            bbox = word.get('bbox', [])
            if len(bbox) >= 2:
                y = bbox[1]  # y0

                if y < header_end:
                    header_words.append(word)
                elif y < items_end:
                    items_words.append(word)
                else:
                    totals_words.append(word)

        return {
            "header": {
                "y_start": header_start,
                "y_end": header_end,
                "words": header_words,
                "word_count": len(header_words)
            },
            "items": {
                "y_start": items_start,
                "y_end": items_end,
                "words": items_words,
                "word_count": len(items_words)
            },
            "totals": {
                "y_start": totals_start,
                "y_end": totals_end,
                "words": totals_words,
                "word_count": len(totals_words)
            }
        }

    def _empty_regions(self) -> Dict[str, Dict]:
        """Retorna regiones vacías"""
        return {
            "header": {"y_start": 0, "y_end": 0, "words": [], "word_count": 0},
            "items": {"y_start": 0, "y_end": 0, "words": [], "word_count": 0},
            "totals": {"y_start": 0, "y_end": 0, "words": [], "word_count": 0}
        }

    def extract_totals_fields(
        self,
        totals_words: List[Dict],
        annotations: Dict
    ) -> List[Dict]:
        """
        Extrae campos de totales de la región detectada

        Args:
            totals_words: Palabras en la región de totales
            annotations: Anotaciones del JSON

        Returns:
            Lista de campos de totales encontrados
        """
        totals_fields = {}

        # Campos típicos de totales
        total_field_names = [
            'subtotal', 'igv', 'importe_total', 'descuento_global',
            'total_operaciones_gravadas', 'total_operaciones_inafectas',
            'total_descuentos', 'redondeo', 'valor_venta', 'precio_venta'
        ]

        for field_name in total_field_names:
            if field_name in annotations:
                field_value = annotations[field_name]
                if field_value is not None and field_value != "":
                    totals_fields[field_name] = field_value

        logger.debug(f"Campos de totales identificados: {len(totals_fields)}")
        return totals_fields

    def get_region_stats(self, regions: Dict[str, Dict]) -> Dict:
        """
        Calcula estadísticas de las regiones

        Args:
            regions: Regiones detectadas

        Returns:
            Diccionario con estadísticas
        """
        return {
            "header": {
                "height": regions['header']['y_end'] - regions['header']['y_start'],
                "word_count": regions['header']['word_count'],
                "percentage": ((regions['header']['y_end'] - regions['header']['y_start']) / self.page_height) * 100
            },
            "items": {
                "height": regions['items']['y_end'] - regions['items']['y_start'],
                "word_count": regions['items']['word_count'],
                "percentage": ((regions['items']['y_end'] - regions['items']['y_start']) / self.page_height) * 100
            },
            "totals": {
                "height": regions['totals']['y_end'] - regions['totals']['y_start'],
                "word_count": regions['totals']['word_count'],
                "percentage": ((regions['totals']['y_end'] - regions['totals']['y_start']) / self.page_height) * 100
            }
        }
