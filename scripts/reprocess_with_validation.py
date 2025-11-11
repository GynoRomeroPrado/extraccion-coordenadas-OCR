#!/usr/bin/env python3
"""
Script para re-procesar documentos existentes con validaciones mejoradas

Usa el LayoutLMv3FormatterV2 con todas las mejoras:
- Filtrado de confidence < 0.5
- Validación y expansión de bboxes
- Resolución de duplicados
- Eliminación de metadata

Uso:
    python scripts/reprocess_with_validation.py \\
        --input-dir "output_old/" \\
        --output-dir "output_fixed/" \\
        --pdf-dir "facturas_pdf/" \\
        --json-dir "anotaciones_json/"
"""
import argparse
import sys
from pathlib import Path
import logging
import json
import time
from typing import Dict, Any

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.hybrid_extractor import HybridExtractor
from src.matchers.fuzzy_matcher import FuzzyMatcher
from src.formatters.layoutlmv3_formatter_v2 import LayoutLMv3FormatterV2
from src.formatters.bbox_validator import BBoxValidator
from config import Config

logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def reprocess_document(
    pdf_path: Path,
    json_path: Path,
    output_dir: Path,
    document_id: str,
    extractor: HybridExtractor,
    matcher: FuzzyMatcher,
    formatter: LayoutLMv3FormatterV2
) -> Dict[str, Any]:
    """
    Re-procesa un documento con validaciones mejoradas

    Args:
        pdf_path: Ruta al PDF
        json_path: Ruta al JSON de anotaciones
        output_dir: Directorio de salida
        document_id: ID del documento
        extractor: Instancia de HybridExtractor
        matcher: Instancia de FuzzyMatcher
        formatter: Instancia de LayoutLMv3FormatterV2

    Returns:
        Metadata con estadísticas y métricas de calidad
    """
    start_time = time.time()

    try:
        # Cargar JSON de anotaciones
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        # Extraer texto del PDF
        pdf_words = extractor.extract(str(pdf_path), page_num=0)

        if not pdf_words:
            raise ValueError(f"No se pudo extraer texto del PDF")

        # Separar header e items
        items = annotations.pop('items', [])
        annotations.pop('cuotas', [])  # Ignorar cuotas
        header_fields = annotations

        # Resetear formatter
        formatter.reset_id_counter()
        formatter.reset_stats()

        # Formatear campos del header
        header_elements = formatter.format_header(
            header_fields,
            pdf_words,
            matcher
        )

        # Formatear items si existen
        items_elements = []
        if items:
            items_elements = formatter.format_items(
                items,
                {0: pdf_words},
                matcher
            )

        # Combinar todos los elementos
        all_elements = header_elements + items_elements

        # Crear documento con validaciones
        document = formatter.create_document(
            all_elements,
            sort_by_position=True,
            apply_validations=True
        )

        # Obtener métricas de calidad
        quality_metrics = formatter.get_quality_metrics(document)
        processing_stats = formatter.get_stats()

        # Guardar documento mejorado
        output_file = output_dir / f"{document_id}_layoutlmv3.json"
        formatter.save_to_file(document, str(output_file))

        # Generar metadata
        processing_time = time.time() - start_time

        header_match_rate = len(header_elements) / len(header_fields) if header_fields else 1.0
        items_total_fields = sum(
            len([v for v in item.values() if v is not None and v != ""])
            for item in items
        )
        items_match_rate = len(items_elements) / items_total_fields if items_total_fields > 0 else 1.0

        metadata = {
            "document_id": document_id,
            "original_pdf": pdf_path.name,
            "original_json": json_path.name,
            "reprocessed": True,
            "reprocessing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "components": {
                "header": {
                    "fields_total": len(header_fields),
                    "fields_found": len(header_elements),
                    "match_rate": round(header_match_rate, 3)
                },
                "items": {
                    "items_total": len(items),
                    "fields_total": items_total_fields,
                    "fields_found": len(items_elements),
                    "match_rate": round(items_match_rate, 3) if items else None
                }
            },
            "quality_metrics": quality_metrics,
            "processing_stats": processing_stats,
            "statistics": {
                "total_elements": len(document['form']),
                "processing_time_seconds": round(processing_time, 2)
            },
            "improvements": {
                "filtered_low_confidence": processing_stats['filtered_low_confidence'],
                "filtered_invalid_bbox": processing_stats['filtered_invalid_bbox'],
                "expanded_bboxes": processing_stats['expanded_bboxes'],
                "resolved_duplicates": processing_stats['resolved_duplicates'],
                "removed_metadata_fields": processing_stats['removed_metadata_fields']
            }
        }

        # Guardar metadata
        metadata_file = output_dir / f"{document_id}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata

    except Exception as e:
        logger.error(f"Error al re-procesar {document_id}: {e}")
        raise


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Re-procesa documentos con validaciones mejoradas"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directorio con documentos procesados originalmente"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio para documentos mejorados"
    )

    parser.add_argument(
        "--pdf-dir",
        type=str,
        required=True,
        help="Directorio con PDFs originales"
    )

    parser.add_argument(
        "--json-dir",
        type=str,
        required=True,
        help="Directorio con JSONs de anotaciones"
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Confidence mínimo para incluir campos (default: 0.5)"
    )

    parser.add_argument(
        "--duplicate-strategy",
        type=str,
        choices=["keep_best", "expand", "remove_all"],
        default="keep_best",
        help="Estrategia para resolver duplicados (default: keep_best)"
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Máximo de documentos a procesar"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verbose"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validar directorios
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    pdf_dir = Path(args.pdf_dir)
    json_dir = Path(args.json_dir)

    if not input_dir.exists():
        logger.error(f"Directorio de entrada no existe: {input_dir}")
        sys.exit(1)

    if not pdf_dir.exists():
        logger.error(f"Directorio de PDFs no existe: {pdf_dir}")
        sys.exit(1)

    if not json_dir.exists():
        logger.error(f"Directorio de JSONs no existe: {json_dir}")
        sys.exit(1)

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encontrar documentos procesados
    metadata_files = list(input_dir.glob("*_metadata.json"))

    if args.max_documents:
        metadata_files = metadata_files[:args.max_documents]

    total_documents = len(metadata_files)

    logger.info("=" * 60)
    logger.info("🔧 RE-PROCESAMIENTO CON VALIDACIONES MEJORADAS")
    logger.info("=" * 60)
    logger.info(f"📁 Input: {input_dir}")
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"📄 Documentos a re-procesar: {total_documents}")
    logger.info(f"🎯 Min confidence: {args.min_confidence}")
    logger.info(f"🔀 Estrategia duplicados: {args.duplicate_strategy}")
    logger.info("=" * 60)

    # Crear procesadores
    extractor = HybridExtractor(dpi=300, auto_rotate=True)
    matcher = FuzzyMatcher(threshold=85)
    formatter = LayoutLMv3FormatterV2(
        min_confidence=args.min_confidence,
        resolve_duplicates=True,
        expand_small_bboxes=True,
        duplicate_strategy=args.duplicate_strategy
    )

    # Estadísticas globales
    processed = 0
    failed = 0
    total_improvements = {
        'filtered_low_confidence': 0,
        'filtered_invalid_bbox': 0,
        'expanded_bboxes': 0,
        'resolved_duplicates': 0,
        'removed_metadata_fields': 0
    }

    start_time = time.time()

    # Procesar cada documento
    for i, metadata_file in enumerate(metadata_files, 1):
        # Extraer document_id
        document_id = metadata_file.stem.replace('_metadata', '')

        # Buscar PDF y JSON
        pdf_path = pdf_dir / f"{document_id}.pdf"
        json_path = json_dir / f"{document_id}.json"

        if not pdf_path.exists():
            logger.warning(f"[{i}/{total_documents}] PDF no encontrado: {document_id}")
            failed += 1
            continue

        if not json_path.exists():
            logger.warning(f"[{i}/{total_documents}] JSON no encontrado: {document_id}")
            failed += 1
            continue

        try:
            logger.info(f"[{i}/{total_documents}] 🔧 Re-procesando {document_id}...")

            metadata = reprocess_document(
                pdf_path,
                json_path,
                output_dir,
                document_id,
                extractor,
                matcher,
                formatter
            )

            # Acumular mejoras
            improvements = metadata['improvements']
            for key in total_improvements:
                total_improvements[key] += improvements[key]

            processed += 1

            logger.info(
                f"[{i}/{total_documents}] ✅ {document_id} mejorado "
                f"(filtrados: {improvements['filtered_low_confidence']} confidence, "
                f"{improvements['filtered_invalid_bbox']} bbox, "
                f"{improvements['resolved_duplicates']} duplicados)"
            )

        except Exception as e:
            logger.error(f"[{i}/{total_documents}] ❌ Error en {document_id}: {e}")
            failed += 1

    # Tiempo total
    total_time = time.time() - start_time

    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DE RE-PROCESAMIENTO")
    logger.info("=" * 60)
    logger.info(f"📄 Total: {total_documents}")
    logger.info(f"✅ Procesados: {processed}")
    logger.info(f"❌ Fallidos: {failed}")
    logger.info(f"⏱️  Tiempo: {total_time:.2f}s ({total_time/processed:.2f}s/doc)")
    logger.info(f"\n🔧 MEJORAS TOTALES:")
    logger.info(f"   Filtrados por confidence: {total_improvements['filtered_low_confidence']}")
    logger.info(f"   Filtrados por bbox inválido: {total_improvements['filtered_invalid_bbox']}")
    logger.info(f"   Bboxes expandidos: {total_improvements['expanded_bboxes']}")
    logger.info(f"   Duplicados resueltos: {total_improvements['resolved_duplicates']}")
    logger.info(f"   Campos de metadata eliminados: {total_improvements['removed_metadata_fields']}")
    logger.info("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
