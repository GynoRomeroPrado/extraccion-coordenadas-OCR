#!/usr/bin/env python3
"""
Script para procesar documentos en lote

Uso:
    python scripts/process_batch.py \\
        --input-dir "/path/to/dataset_entrenamiento_101125" \\
        --output-dir "output/" \\
        --num-workers 4 \\
        --start-index 0 \\
        --end-index 1000
"""
import argparse
import sys
from pathlib import Path
import logging
import json
from typing import Dict, Any
import time

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.processor import InvoiceProcessor
from config import Config

# Configurar logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Procesa documentos en lote y genera formato LayoutLMv3"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directorio raíz con subdirectorios facturas_pdf/ y anotaciones_json/"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directorio de salida (default: output/)"
    )

    parser.add_argument(
        "--pdf-subdir",
        type=str,
        default="facturas_pdf",
        help="Subdirectorio con PDFs (default: facturas_pdf)"
    )

    parser.add_argument(
        "--json-subdir",
        type=str,
        default="anotaciones_json",
        help="Subdirectorio con JSONs (default: anotaciones_json)"
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Índice inicial (default: 0)"
    )

    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Índice final (default: todos)"
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Máximo de documentos a procesar (default: todos)"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Saltar documentos ya procesados"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI para OCR (default: 300)"
    )

    parser.add_argument(
        "--match-threshold",
        type=int,
        default=85,
        help="Umbral de matching (0-100, default: 85)"
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Guardar checkpoint cada N documentos (default: 1000)"
    )

    parser.add_argument(
        "--no-auto-rotate",
        action="store_true",
        help="Desactivar detección automática de orientación"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verbose"
    )

    args = parser.parse_args()

    # Configurar logging verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validar directorio de entrada
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"❌ Directorio de entrada no encontrado: {input_dir}")
        sys.exit(1)

    pdf_dir = input_dir / args.pdf_subdir
    json_dir = input_dir / args.json_subdir

    if not pdf_dir.exists():
        logger.error(f"❌ Directorio de PDFs no encontrado: {pdf_dir}")
        sys.exit(1)

    if not json_dir.exists():
        logger.error(f"❌ Directorio de JSONs no encontrado: {json_dir}")
        sys.exit(1)

    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encontrar todos los PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    # Aplicar filtros de índice
    if args.start_index > 0:
        pdf_files = pdf_files[args.start_index:]

    if args.end_index is not None:
        pdf_files = pdf_files[:args.end_index - args.start_index]

    if args.max_documents is not None:
        pdf_files = pdf_files[:args.max_documents]

    total_documents = len(pdf_files)

    logger.info("=" * 60)
    logger.info("🚀 PROCESAMIENTO EN LOTE")
    logger.info("=" * 60)
    logger.info(f"📁 Directorio de entrada: {input_dir}")
    logger.info(f"📁 Directorio de salida: {output_dir}")
    logger.info(f"📄 Total de documentos: {total_documents}")
    logger.info(f"⚙️  DPI: {args.dpi}")
    logger.info(f"🎯 Umbral de matching: {args.match_threshold}")
    logger.info(f"💾 Checkpoint cada: {args.checkpoint_interval} documentos")
    logger.info("=" * 60)

    if total_documents == 0:
        logger.warning("⚠️  No se encontraron documentos para procesar")
        sys.exit(0)

    # Crear procesador
    processor = InvoiceProcessor(
        dpi=args.dpi,
        match_threshold=args.match_threshold,
        auto_rotate=not args.no_auto_rotate
    )

    # Estadísticas
    processed = 0
    failed = 0
    skipped = 0
    failed_documents = []
    processing_times = []
    match_rates_header = []
    match_rates_items = []

    start_time = time.time()

    # Procesar cada documento
    for i, pdf_path in enumerate(pdf_files, 1):
        document_id = pdf_path.stem

        # Buscar JSON correspondiente
        json_path = json_dir / f"{document_id}.json"

        if not json_path.exists():
            logger.warning(f"[{i}/{total_documents}] ⚠️  JSON no encontrado para {document_id}, saltando...")
            skipped += 1
            continue

        # Verificar si ya existe
        metadata_file = output_dir / f"{document_id}_metadata.json"

        if args.skip_existing and metadata_file.exists():
            logger.info(f"[{i}/{total_documents}] ⏭️  {document_id} ya procesado, saltando...")
            skipped += 1
            continue

        # Procesar documento
        try:
            logger.info(f"[{i}/{total_documents}] 📄 Procesando {document_id}...")

            metadata = processor.process(
                pdf_path=str(pdf_path),
                json_path=str(json_path),
                output_dir=str(output_dir),
                document_id=document_id
            )

            # Recolectar estadísticas
            processing_time = metadata['statistics']['processing_time_seconds']
            processing_times.append(processing_time)

            header_match_rate = metadata['components']['header']['match_rate']
            match_rates_header.append(header_match_rate)

            if metadata['components']['items']['match_rate'] is not None:
                match_rates_items.append(metadata['components']['items']['match_rate'])

            processed += 1

            logger.info(
                f"[{i}/{total_documents}] ✅ {document_id} procesado en {processing_time:.2f}s "
                f"(header: {header_match_rate:.1%})"
            )

            # Checkpoint
            if processed % args.checkpoint_interval == 0:
                save_checkpoint(
                    output_dir,
                    processed,
                    failed,
                    skipped,
                    i,
                    total_documents
                )

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Interrupción detectada. Guardando checkpoint...")
            save_checkpoint(
                output_dir,
                processed,
                failed,
                skipped,
                i,
                total_documents
            )
            logger.info("💾 Checkpoint guardado. Saliendo...")
            sys.exit(0)

        except Exception as e:
            logger.error(f"[{i}/{total_documents}] ❌ Error en {document_id}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            failed += 1
            failed_documents.append(document_id)

    # Tiempo total
    total_time = time.time() - start_time

    # Calcular estadísticas finales
    success_rate = processed / total_documents if total_documents > 0 else 0
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_match_rate_header = sum(match_rates_header) / len(match_rates_header) if match_rates_header else 0
    avg_match_rate_items = sum(match_rates_items) / len(match_rates_items) if match_rates_items else 0

    # Resumen final
    batch_stats = {
        "total_documents": total_documents,
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "success_rate": round(success_rate, 3),
        "avg_processing_time": round(avg_processing_time, 2),
        "total_processing_time": round(total_time, 2),
        "avg_match_rate_header": round(avg_match_rate_header, 3),
        "avg_match_rate_items": round(avg_match_rate_items, 3) if match_rates_items else None,
        "failed_documents": failed_documents
    }

    # Guardar estadísticas
    stats_file = output_dir / "batch_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(batch_stats, f, ensure_ascii=False, indent=2)

    # Mostrar resumen
    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DEL BATCH")
    logger.info("=" * 60)
    logger.info(f"📄 Total de documentos: {total_documents}")
    logger.info(f"✅ Procesados: {processed} ({success_rate:.1%})")
    logger.info(f"❌ Fallidos: {failed}")
    logger.info(f"⏭️  Saltados: {skipped}")
    logger.info(f"⏱️  Tiempo total: {total_time:.2f}s ({total_time/60:.1f} min)")
    logger.info(f"⏱️  Tiempo promedio: {avg_processing_time:.2f}s/doc")
    logger.info(f"📊 Match rate promedio (header): {avg_match_rate_header:.1%}")
    if match_rates_items:
        logger.info(f"📊 Match rate promedio (items): {avg_match_rate_items:.1%}")
    logger.info(f"💾 Estadísticas guardadas en: {stats_file}")
    logger.info("=" * 60)

    if failed_documents:
        logger.warning("\n⚠️  Documentos fallidos:")
        for doc_id in failed_documents:
            logger.warning(f"   • {doc_id}")

    # Exit code
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


def save_checkpoint(
    output_dir: Path,
    processed: int,
    failed: int,
    skipped: int,
    current_index: int,
    total_documents: int
):
    """Guarda checkpoint del progreso"""
    checkpoint = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
        "current_index": current_index,
        "total_documents": total_documents,
        "progress": round((current_index / total_documents) * 100, 2)
    }

    checkpoint_file = output_dir / "checkpoint.json"
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    logger.info(
        f"💾 Checkpoint guardado: {processed} procesados, "
        f"{failed} fallidos, {skipped} saltados "
        f"({checkpoint['progress']:.1f}% completado)"
    )


if __name__ == "__main__":
    main()
