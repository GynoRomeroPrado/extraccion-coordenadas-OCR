"""
Pipeline para procesar documentos de una sola página

Orquesta el flujo completo de procesamiento para facturas de 1 página:
1. Extraer texto con coordenadas
2. Matchear campos del JSON
3. Formatear a LayoutLMv3
4. Generar salida
"""
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
import time

from ..extractors.hybrid_extractor import HybridExtractor
from ..matchers.fuzzy_matcher import FuzzyMatcher
from ..formatters.layoutlmv3_formatter import LayoutLMv3Formatter
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class SinglePagePipeline:
    """
    Pipeline para procesar facturas de una sola página

    Workflow:
    1. Cargar JSON de anotaciones
    2. Extraer texto y coordenadas del PDF
    3. Matchear campos del header
    4. Matchear campos de items
    5. Formatear a LayoutLMv3
    6. Guardar resultado
    7. Generar metadata
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
        self.formatter = LayoutLMv3Formatter()

    def process(
        self,
        pdf_path: str,
        json_path: str,
        output_dir: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Procesa un documento completo

        Args:
            pdf_path: Ruta al PDF
            json_path: Ruta al JSON de anotaciones
            output_dir: Directorio de salida
            document_id: ID del documento (si None, se extrae del nombre del archivo)

        Returns:
            Metadata con estadísticas y resultados

        Raises:
            FileNotFoundError: Si los archivos no existen
            ValueError: Si el JSON tiene formato inválido
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

        logger.info(f"Procesando documento: {document_id}")

        # Crear directorio de salida
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Cargar JSON de anotaciones
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            logger.debug(f"Anotaciones cargadas: {len(annotations)} campos")

            # 2. Extraer texto y coordenadas del PDF
            logger.info("Extrayendo texto del PDF...")
            pdf_words = self.extractor.extract(str(pdf_path), page_num=0)

            if not pdf_words:
                raise ValueError(f"No se pudo extraer texto del PDF: {pdf_path}")

            logger.info(f"Extraídas {len(pdf_words)} palabras del PDF")

            # 3. Separar header e items
            items = annotations.pop('items', [])
            cuotas = annotations.pop('cuotas', [])  # Ignorar cuotas (siempre vacío)

            header_fields = annotations  # Todo lo demás es header

            # 4. Matchear campos del header
            logger.info("Matcheando campos del header...")
            self.formatter.reset_id_counter()
            header_elements = self.formatter.format_header(
                header_fields,
                pdf_words,
                self.matcher
            )

            # 5. Matchear campos de items
            logger.info(f"Matcheando {len(items)} items...")
            items_elements = []
            if items:
                items_elements = self.formatter.format_items(
                    items,
                    {0: pdf_words},  # Solo página 0
                    self.matcher
                )

            # 6. Combinar todos los elementos
            all_elements = header_elements + items_elements

            # 7. Crear documento
            document = self.formatter.create_document(all_elements, sort_by_position=True)

            # 8. Obtener estadísticas
            doc_stats = self.formatter.get_stats(document)
            extraction_stats = self.extractor.get_extraction_stats(str(pdf_path), {0: pdf_words})

            # 9. Calcular match rates
            header_match_rate = len(header_elements) / len(header_fields) if header_fields else 1.0
            items_total_fields = sum(len([v for v in item.values() if v is not None and v != ""]) for item in items)
            items_match_rate = len(items_elements) / items_total_fields if items_total_fields > 0 else 1.0

            # 10. Guardar documento
            output_file = output_dir / f"{document_id}_layoutlmv3.json"
            self.formatter.save_to_file(document, str(output_file))

            # 11. Generar metadata
            processing_time = time.time() - start_time

            metadata = {
                "document_id": document_id,
                "original_pdf": pdf_path.name,
                "original_json": json_path.name,
                "num_pages": 1,
                "orientation_detected": extraction_stats.get('rotation_angle', 0),
                "extraction_method": extraction_stats.get('method', 'unknown'),
                "components": {
                    "header": {
                        "pages": [0],
                        "fields_total": len(header_fields),
                        "fields_found": len(header_elements),
                        "match_rate": round(header_match_rate, 3),
                        "files": [output_file.name]
                    },
                    "items": {
                        "pages": [0],
                        "items_total": len(items),
                        "items_found": len(items) if items_elements else 0,
                        "fields_total": items_total_fields,
                        "fields_found": len(items_elements),
                        "match_rate": round(items_match_rate, 3) if items else None,
                        "files": [output_file.name]
                    }
                },
                "statistics": {
                    "total_elements": doc_stats['num_elements'],
                    "total_words": doc_stats['num_words'],
                    "confidence_avg": doc_stats['avg_confidence'],
                    "processing_time_seconds": round(processing_time, 2)
                },
                "warnings": [],
                "errors": []
            }

            # Agregar warnings si match rate es bajo
            if header_match_rate < Config.MIN_MATCH_RATE:
                metadata['warnings'].append(
                    f"Match rate del header ({header_match_rate:.1%}) está por debajo del umbral mínimo ({Config.MIN_MATCH_RATE:.1%})"
                )

            if items and items_match_rate < Config.MIN_MATCH_RATE:
                metadata['warnings'].append(
                    f"Match rate de items ({items_match_rate:.1%}) está por debajo del umbral mínimo ({Config.MIN_MATCH_RATE:.1%})"
                )

            # Guardar metadata
            metadata_file = output_dir / f"{document_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info(
                f"✓ Documento procesado exitosamente en {processing_time:.2f}s\n"
                f"  - Header: {len(header_elements)}/{len(header_fields)} campos ({header_match_rate:.1%})\n"
                f"  - Items: {len(items_elements)}/{items_total_fields} campos ({items_match_rate:.1%})\n"
                f"  - Archivo: {output_file.name}"
            )

            return metadata

        except Exception as e:
            logger.error(f"Error al procesar documento {document_id}: {e}")

            # Crear metadata de error
            error_metadata = {
                "document_id": document_id,
                "original_pdf": pdf_path.name,
                "original_json": json_path.name,
                "num_pages": 1,
                "components": {},
                "statistics": {},
                "warnings": [],
                "errors": [str(e)]
            }

            # Guardar metadata de error
            metadata_file = output_dir / f"{document_id}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(error_metadata, f, ensure_ascii=False, indent=2)

            raise

    def validate_output(self, output_dir: str, document_id: str) -> bool:
        """
        Valida que la salida sea correcta

        Args:
            output_dir: Directorio de salida
            document_id: ID del documento

        Returns:
            True si la validación pasa, False en caso contrario
        """
        output_dir = Path(output_dir)

        # Verificar que existe el archivo de salida
        output_file = output_dir / f"{document_id}_layoutlmv3.json"
        if not output_file.exists():
            logger.error(f"Archivo de salida no existe: {output_file}")
            return False

        # Verificar que existe metadata
        metadata_file = output_dir / f"{document_id}_metadata.json"
        if not metadata_file.exists():
            logger.error(f"Archivo de metadata no existe: {metadata_file}")
            return False

        try:
            # Cargar y validar estructura del documento
            with open(output_file, 'r', encoding='utf-8') as f:
                document = json.load(f)

            if 'form' not in document:
                logger.error("Documento no tiene campo 'form'")
                return False

            elements = document['form']
            if not isinstance(elements, list):
                logger.error("Campo 'form' no es una lista")
                return False

            # Validar que cada elemento tenga campos requeridos
            required_fields = ['id', 'text', 'box', 'label', 'field_name', 'words', 'page']
            for i, elem in enumerate(elements):
                for field in required_fields:
                    if field not in elem:
                        logger.error(f"Elemento {i} no tiene campo requerido: {field}")
                        return False

                # Validar bbox
                bbox = elem['box']
                if not isinstance(bbox, list) or len(bbox) != 4:
                    logger.error(f"Elemento {i} tiene bbox inválido: {bbox}")
                    return False

                # Validar que coordenadas estén en rango 0-1000
                if not all(0 <= coord <= 1000 for coord in bbox):
                    logger.error(f"Elemento {i} tiene coordenadas fuera de rango: {bbox}")
                    return False

            logger.info(f"Validación exitosa: {len(elements)} elementos válidos")
            return True

        except Exception as e:
            logger.error(f"Error al validar salida: {e}")
            return False
