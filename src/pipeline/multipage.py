"""
Pipeline para procesar documentos multipágina

Maneja facturas de 2+ páginas con estrategia de separación de header e items.
"""
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

from ..extractors.hybrid_extractor import HybridExtractor
from ..matchers.fuzzy_matcher import FuzzyMatcher
from ..matchers.table_detector import TableDetector
from ..formatters.layoutlmv3_formatter import LayoutLMv3Formatter
from ..formatters.chunker import TokenChunker
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class MultiPagePipeline:
    """
    Pipeline para procesar facturas multipágina

    Workflow:
    1. Extraer todas las páginas
    2. Separar header (primeras páginas) e items (páginas posteriores)
    3. Matchear campos del header
    4. Detectar tablas en páginas de items
    5. Mapear items con filas de tablas
    6. Formatear a LayoutLMv3
    7. Dividir en chunks si es necesario
    8. Guardar resultados
    """

    def __init__(
        self,
        dpi: int = 300,
        match_threshold: int = 85,
        auto_rotate: bool = True
    ):
        """
        Args:
            dpi: Resolución para OCR
            match_threshold: Umbral mínimo de similitud para matching
            auto_rotate: Detectar y corregir orientación automáticamente
        """
        self.extractor = HybridExtractor(dpi=dpi, auto_rotate=auto_rotate)
        self.matcher = FuzzyMatcher(threshold=match_threshold)
        self.table_detector = TableDetector()
        self.formatter = LayoutLMv3Formatter()
        self.chunker = TokenChunker()

    def process(
        self,
        pdf_path: str,
        json_path: str,
        output_dir: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Procesa un documento multipágina

        Args:
            pdf_path: Ruta al PDF
            json_path: Ruta al JSON de anotaciones
            output_dir: Directorio de salida
            document_id: ID del documento

        Returns:
            Metadata con estadísticas y resultados
        """
        start_time = time.time()

        # Validar archivos
        pdf_path = Path(pdf_path)
        json_path = Path(json_path)
        output_dir = Path(output_dir)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        if not json_path.exists():
            raise FileNotFoundError(f"JSON no encontrado: {json_path}")

        # Extraer document_id si no se proporciona
        if document_id is None:
            document_id = pdf_path.stem

        logger.info(f"Procesando documento multipágina: {document_id}")

        # Crear directorio de salida
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Cargar JSON de anotaciones
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            # 2. Extraer todas las páginas del PDF
            logger.info("Extrayendo todas las páginas del PDF...")
            pdf_words_by_page = self.extractor.extract_all_pages(str(pdf_path))

            num_pages = len(pdf_words_by_page)
            logger.info(f"Extraídas {num_pages} páginas")

            # 3. Separar header e items
            items = annotations.pop('items', [])
            cuotas = annotations.pop('cuotas', [])
            header_fields = annotations

            # 4. Determinar páginas de header vs items
            # Estrategia: Buscar algunos campos del header en cada página
            header_pages, items_pages = self._identify_header_and_items_pages(
                pdf_words_by_page,
                header_fields,
                items
            )

            logger.info(f"Header en páginas: {header_pages}, Items en páginas: {items_pages}")

            # 5. Matchear campos del header
            logger.info("Matcheando campos del header...")
            header_pdf_words = []
            for page in header_pages:
                header_pdf_words.extend(pdf_words_by_page[page])

            self.formatter.reset_id_counter()
            header_elements = self.formatter.format_header(
                header_fields,
                header_pdf_words,
                self.matcher
            )

            # 6. Matchear items usando detección de tabla
            logger.info(f"Matcheando {len(items)} items...")
            items_elements = []

            if items and items_pages:
                items_elements = self._match_items_with_table_detection(
                    items,
                    pdf_words_by_page,
                    items_pages
                )

            # 7. Crear documentos separados
            output_files = []

            # 7.1. Documento de header
            if header_elements:
                header_doc = self.formatter.create_document(header_elements, sort_by_position=True)

                # Verificar si necesita chunking
                if self.chunker.needs_chunking(header_elements):
                    header_chunks = self.chunker.chunk_document(header_doc)
                    for i, chunk in enumerate(header_chunks):
                        chunk_file = output_dir / f"{document_id}_header_chunk_{i}.json"
                        self.formatter.save_to_file(chunk, str(chunk_file))
                        output_files.append(chunk_file.name)
                else:
                    header_file = output_dir / f"{document_id}_header.json"
                    self.formatter.save_to_file(header_doc, str(header_file))
                    output_files.append(header_file.name)

            # 7.2. Documentos de items por página
            if items_elements:
                # Agrupar items por página
                items_by_page = {}
                for elem in items_elements:
                    page = elem.get('page', 0)
                    if page not in items_by_page:
                        items_by_page[page] = []
                    items_by_page[page].append(elem)

                # Guardar documento por cada página de items
                for page, page_items in sorted(items_by_page.items()):
                    items_doc = self.formatter.create_document(page_items, sort_by_position=True)

                    # Verificar si necesita chunking
                    if self.chunker.needs_chunking(page_items):
                        items_chunks = self.chunker.chunk_document(items_doc)
                        for i, chunk in enumerate(items_chunks):
                            chunk_file = output_dir / f"{document_id}_items_p{page}_chunk_{i}.json"
                            self.formatter.save_to_file(chunk, str(chunk_file))
                            output_files.append(chunk_file.name)
                    else:
                        items_file = output_dir / f"{document_id}_items_p{page}.json"
                        self.formatter.save_to_file(items_doc, str(items_file))
                        output_files.append(items_file.name)

            # 8. Calcular estadísticas
            extraction_stats = self.extractor.get_extraction_stats(
                str(pdf_path),
                pdf_words_by_page
            )

            header_match_rate = len(header_elements) / len(header_fields) if header_fields else 1.0
            items_total_fields = sum(
                len([v for v in item.values() if v is not None and v != ""])
                for item in items
            )
            items_match_rate = len(items_elements) / items_total_fields if items_total_fields > 0 else 1.0

            # 9. Generar metadata
            processing_time = time.time() - start_time

            metadata = {
                "document_id": document_id,
                "original_pdf": pdf_path.name,
                "original_json": json_path.name,
                "num_pages": num_pages,
                "orientation_detected": extraction_stats.get('rotation_angle', 0),
                "extraction_method": extraction_stats.get('method', 'unknown'),
                "components": {
                    "header": {
                        "pages": header_pages,
                        "fields_total": len(header_fields),
                        "fields_found": len(header_elements),
                        "match_rate": round(header_match_rate, 3),
                        "files": [f for f in output_files if 'header' in f]
                    },
                    "items": {
                        "pages": items_pages,
                        "items_total": len(items),
                        "items_found": len(items) if items_elements else 0,
                        "fields_total": items_total_fields,
                        "fields_found": len(items_elements),
                        "match_rate": round(items_match_rate, 3) if items else None,
                        "files": [f for f in output_files if 'items' in f]
                    }
                },
                "statistics": {
                    "total_elements": len(header_elements) + len(items_elements),
                    "total_words": extraction_stats.get('total_words', 0),
                    "words_per_page": extraction_stats.get('words_per_page', []),
                    "confidence_avg": extraction_stats.get('avg_confidence', 0),
                    "processing_time_seconds": round(processing_time, 2)
                },
                "warnings": [],
                "errors": []
            }

            # Agregar warnings
            if header_match_rate < Config.MIN_MATCH_RATE:
                metadata['warnings'].append(
                    f"Match rate del header ({header_match_rate:.1%}) está por debajo del umbral"
                )

            if items and items_match_rate < Config.MIN_MATCH_RATE:
                metadata['warnings'].append(
                    f"Match rate de items ({items_match_rate:.1%}) está por debajo del umbral"
                )

            # Guardar metadata
            metadata_file = output_dir / f"{document_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(
                f"✓ Documento procesado exitosamente en {processing_time:.2f}s\n"
                f"  - Páginas: {num_pages}\n"
                f"  - Header: {len(header_elements)}/{len(header_fields)} campos ({header_match_rate:.1%})\n"
                f"  - Items: {len(items_elements)}/{items_total_fields} campos ({items_match_rate:.1%})\n"
                f"  - Archivos: {len(output_files)}"
            )

            return metadata

        except Exception as e:
            logger.error(f"Error al procesar documento {document_id}: {e}")

            # Crear metadata de error
            error_metadata = {
                "document_id": document_id,
                "original_pdf": pdf_path.name,
                "original_json": json_path.name,
                "num_pages": None,
                "components": {},
                "statistics": {},
                "warnings": [],
                "errors": [str(e)]
            }

            metadata_file = output_dir / f"{document_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(error_metadata, f, ensure_ascii=False, indent=2)

            raise

    def _identify_header_and_items_pages(
        self,
        pdf_words_by_page: Dict[int, List[Dict]],
        header_fields: Dict,
        items: List[Dict]
    ) -> tuple:
        """
        Identifica qué páginas contienen header y cuáles contienen items

        Args:
            pdf_words_by_page: Palabras por página
            header_fields: Campos del header
            items: Items del JSON

        Returns:
            Tupla (header_pages, items_pages)
        """
        num_pages = len(pdf_words_by_page)

        # Buscar campos clave del header en cada página
        header_key_fields = ['serie_completa', 'emisor_ruc', 'receptor_numero_doc', 'importe_total']
        header_scores = {}

        for page_num, words in pdf_words_by_page.items():
            score = 0
            for field in header_key_fields:
                if field in header_fields:
                    match = self.matcher.match(str(header_fields[field]), words, field)
                    if match:
                        score += 1

            header_scores[page_num] = score

        # Las páginas con mayor score de header son páginas de header
        # Usualmente la primera página tiene el header
        max_score = max(header_scores.values()) if header_scores else 0

        if max_score == 0:
            # Si no encontramos ningún campo, asumir primera página es header
            header_pages = [0]
        else:
            # Páginas con score > 50% del máximo son header
            threshold = max_score * 0.5
            header_pages = [p for p, s in header_scores.items() if s >= threshold]

        # El resto son páginas de items
        items_pages = [p for p in range(num_pages) if p not in header_pages]

        # Si no hay items en el JSON, no hay páginas de items
        if not items:
            items_pages = []

        return sorted(header_pages), sorted(items_pages)

    def _match_items_with_table_detection(
        self,
        items: List[Dict],
        pdf_words_by_page: Dict[int, List[Dict]],
        items_pages: List[int]
    ) -> List[Dict]:
        """
        Matchea items usando detección de tabla

        Args:
            items: Items del JSON
            pdf_words_by_page: Palabras por página
            items_pages: Páginas que contienen items

        Returns:
            Lista de elementos formateados
        """
        # Combinar palabras de todas las páginas de items
        all_items_words = []
        for page in items_pages:
            all_items_words.extend(pdf_words_by_page[page])

        # Detectar filas de tabla
        rows = self.table_detector.detect_rows(all_items_words)
        table_rows = self.table_detector.filter_table_rows(rows, min_words_per_row=2)

        logger.info(f"Detectadas {len(table_rows)} filas de tabla")

        # Mapear items con filas
        mapped_items = self.table_detector.map_items_to_rows(
            table_rows,
            items,
            self.matcher
        )

        # Formatear elementos
        formatted_elements = []

        for mapped_item in mapped_items:
            matched_fields = mapped_item.get('matched_fields', {})

            for field_name, match_result in matched_fields.items():
                formatted = self.formatter.format_field(
                    f"item.{field_name}",
                    match_result,
                    label="answer"
                )

                if formatted:
                    formatted_elements.append(formatted)

        return formatted_elements
