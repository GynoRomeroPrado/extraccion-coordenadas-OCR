#!/usr/bin/env python3
"""
Script para procesar un documento individual

Uso:
    python scripts/process_single.py \\
        --pdf "facturas_pdf/FACT-000001.pdf" \\
        --json "anotaciones_json/FACT-000001.json" \\
        --output "output/"
"""
import argparse
import sys
from pathlib import Path
import logging

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
        description="Procesa un documento individual y genera formato LayoutLMv3"
    )

    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Ruta al archivo PDF"
    )

    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Ruta al archivo JSON de anotaciones"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directorio de salida (default: output/)"
    )

    parser.add_argument(
        "--document-id",
        type=str,
        default=None,
        help="ID del documento (default: nombre del PDF sin extensión)"
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
        "--no-auto-rotate",
        action="store_true",
        help="Desactivar detección automática de orientación"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validar la salida después de procesar"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verbose (más información de logging)"
    )

    args = parser.parse_args()

    # Configurar logging verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validar archivos de entrada
    pdf_path = Path(args.pdf)
    json_path = Path(args.json)

    if not pdf_path.exists():
        logger.error(f"❌ PDF no encontrado: {pdf_path}")
        sys.exit(1)

    if not json_path.exists():
        logger.error(f"❌ JSON no encontrado: {json_path}")
        sys.exit(1)

    # Crear procesador
    processor = InvoiceProcessor(
        dpi=args.dpi,
        match_threshold=args.match_threshold,
        auto_rotate=not args.no_auto_rotate
    )

    try:
        # Procesar documento
        logger.info(f"📄 Procesando: {pdf_path.name}")
        logger.info("=" * 60)

        metadata = processor.process(
            pdf_path=str(pdf_path),
            json_path=str(json_path),
            output_dir=args.output,
            document_id=args.document_id
        )

        # Mostrar resumen
        logger.info("=" * 60)
        logger.info("✅ PROCESAMIENTO EXITOSO")
        logger.info("=" * 60)

        # Información del documento
        logger.info(f"📋 Documento: {metadata['document_id']}")
        logger.info(f"📄 Páginas: {metadata['num_pages']}")
        logger.info(f"🔄 Orientación: {metadata['orientation_detected']}°")
        logger.info(f"⚙️  Método: {metadata['extraction_method']}")

        # Estadísticas de header
        header_comp = metadata['components']['header']
        logger.info(
            f"📊 Header: {header_comp['fields_found']}/{header_comp['fields_total']} campos "
            f"({header_comp['match_rate']:.1%})"
        )

        # Estadísticas de items
        items_comp = metadata['components']['items']
        if items_comp['items_total'] > 0:
            logger.info(
                f"📦 Items: {items_comp['fields_found']}/{items_comp['fields_total']} campos "
                f"({items_comp['match_rate']:.1%})"
            )

        # Tiempos
        stats = metadata['statistics']
        logger.info(f"⏱️  Tiempo: {stats['processing_time_seconds']:.2f}s")

        # Archivos generados
        all_files = header_comp['files'] + items_comp['files']
        logger.info(f"📁 Archivos generados: {len(all_files)}")
        for file in all_files:
            logger.info(f"   • {file}")

        # Warnings
        if metadata['warnings']:
            logger.warning("⚠️  Advertencias:")
            for warning in metadata['warnings']:
                logger.warning(f"   • {warning}")

        # Validar si se solicitó
        if args.validate:
            logger.info("=" * 60)
            logger.info("🔍 Validando salida...")

            document_id = metadata['document_id']
            is_valid = processor.validate_output(args.output, document_id)

            if is_valid:
                logger.info("✅ Validación exitosa")
            else:
                logger.error("❌ Validación fallida")
                sys.exit(1)

        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Error al procesar documento: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
