#!/usr/bin/env python3
"""
Script para validar el dataset generado

Verifica que todos los documentos cumplan con el formato LayoutLMv3
y que las coordenadas sean válidas.

Uso:
    python scripts/validate_output.py --output-dir "output/"
"""
import argparse
import sys
from pathlib import Path
import logging
import json
from typing import Dict, List, Any

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

# Configurar logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


def validate_document(document_path: Path) -> Dict[str, Any]:
    """
    Valida un documento individual

    Returns:
        Dict con resultado de validación:
        {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "stats": {...}
        }
    """
    errors = []
    warnings = []

    try:
        # Cargar documento
        with open(document_path, 'r', encoding='utf-8') as f:
            document = json.load(f)

        # Validar estructura
        if 'form' not in document:
            errors.append("Documento no tiene campo 'form'")
            return {"valid": False, "errors": errors, "warnings": warnings, "stats": {}}

        elements = document['form']

        if not isinstance(elements, list):
            errors.append("Campo 'form' no es una lista")
            return {"valid": False, "errors": errors, "warnings": warnings, "stats": {}}

        # Validar elementos
        required_fields = ['id', 'text', 'box', 'label', 'field_name', 'words', 'page']

        for i, elem in enumerate(elements):
            # Verificar campos requeridos
            for field in required_fields:
                if field not in elem:
                    errors.append(f"Elemento {i} (id={elem.get('id', '?')}) no tiene campo '{field}'")

            # Validar bbox
            if 'box' in elem:
                bbox = elem['box']

                if not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"Elemento {i} tiene bbox inválido: {bbox}")
                    continue

                # Validar que coordenadas estén en rango 0-1000
                if not all(isinstance(coord, (int, float)) for coord in bbox):
                    errors.append(f"Elemento {i} tiene coordenadas no numéricas: {bbox}")

                if not all(0 <= coord <= 1000 for coord in bbox):
                    errors.append(f"Elemento {i} tiene coordenadas fuera de rango [0-1000]: {bbox}")

                # Validar que x1 > x0 y y1 > y0
                x0, y0, x1, y1 = bbox
                if x1 <= x0:
                    errors.append(f"Elemento {i} tiene x1 <= x0: {bbox}")
                if y1 <= y0:
                    errors.append(f"Elemento {i} tiene y1 <= y0: {bbox}")

            # Validar words
            if 'words' in elem:
                words = elem['words']

                if not isinstance(words, list):
                    errors.append(f"Elemento {i} tiene 'words' que no es una lista")
                    continue

                for j, word in enumerate(words):
                    if 'text' not in word:
                        warnings.append(f"Elemento {i}, palabra {j} no tiene 'text'")

                    if 'box' not in word:
                        warnings.append(f"Elemento {i}, palabra {j} no tiene 'box'")
                    else:
                        word_bbox = word['box']
                        if not isinstance(word_bbox, list) or len(word_bbox) != 4:
                            errors.append(f"Elemento {i}, palabra {j} tiene bbox inválido")

            # Validar confidence si existe
            if 'confidence' in elem:
                conf = elem['confidence']
                if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
                    warnings.append(f"Elemento {i} tiene confidence inválida: {conf}")

        # Calcular estadísticas
        stats = {
            "num_elements": len(elements),
            "num_words": sum(len(elem.get('words', [])) for elem in elements),
            "pages": sorted(list(set(elem.get('page', 0) for elem in elements))),
            "labels": list(set(elem.get('label', 'unknown') for elem in elements))
        }

        is_valid = len(errors) == 0

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "stats": stats
        }

    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "errors": [f"Error al parsear JSON: {e}"],
            "warnings": [],
            "stats": {}
        }

    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Error inesperado: {e}"],
            "warnings": [],
            "stats": {}
        }


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Valida el dataset generado en formato LayoutLMv3"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directorio con documentos generados"
    )

    parser.add_argument(
        "--check-metadata",
        action="store_true",
        help="También validar archivos de metadata"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Modo verbose (mostrar todos los detalles)"
    )

    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Guardar reporte de validación en archivo JSON"
    )

    args = parser.parse_args()

    # Configurar logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validar directorio
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        logger.error(f"❌ Directorio no encontrado: {output_dir}")
        sys.exit(1)

    # Encontrar todos los archivos JSON (excepto metadata y estadísticas)
    all_json_files = list(output_dir.glob("*.json"))

    # Filtrar archivos de metadata y batch_statistics
    document_files = [
        f for f in all_json_files
        if not f.name.endswith('_metadata.json')
        and f.name != 'batch_statistics.json'
        and f.name != 'checkpoint.json'
    ]

    metadata_files = [f for f in all_json_files if f.name.endswith('_metadata.json')]

    logger.info("=" * 60)
    logger.info("🔍 VALIDACIÓN DEL DATASET")
    logger.info("=" * 60)
    logger.info(f"📁 Directorio: {output_dir}")
    logger.info(f"📄 Documentos encontrados: {len(document_files)}")
    logger.info(f"📋 Metadatas encontradas: {len(metadata_files)}")
    logger.info("=" * 60)

    if len(document_files) == 0:
        logger.warning("⚠️  No se encontraron documentos para validar")
        sys.exit(0)

    # Validar cada documento
    valid_count = 0
    invalid_count = 0
    total_errors = []
    total_warnings = []

    validation_results = []

    for i, doc_path in enumerate(document_files, 1):
        logger.info(f"[{i}/{len(document_files)}] Validando {doc_path.name}...")

        result = validate_document(doc_path)
        result['file'] = doc_path.name

        validation_results.append(result)

        if result['valid']:
            valid_count += 1
            logger.info(
                f"   ✅ Válido ({result['stats']['num_elements']} elementos, "
                f"{result['stats']['num_words']} palabras)"
            )

            if result['warnings'] and args.verbose:
                for warning in result['warnings']:
                    logger.warning(f"      ⚠️  {warning}")

        else:
            invalid_count += 1
            logger.error(f"   ❌ Inválido - {len(result['errors'])} errores")

            for error in result['errors']:
                logger.error(f"      • {error}")
                total_errors.append(f"{doc_path.name}: {error}")

        total_warnings.extend([f"{doc_path.name}: {w}" for w in result['warnings']])

    # Validar metadatas si se solicitó
    if args.check_metadata:
        logger.info("\n" + "=" * 60)
        logger.info("🔍 Validando metadatas...")
        logger.info("=" * 60)

        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                required_fields = ['document_id', 'num_pages', 'components', 'statistics']
                missing_fields = [f for f in required_fields if f not in metadata]

                if missing_fields:
                    logger.warning(
                        f"⚠️  {metadata_file.name} le faltan campos: {missing_fields}"
                    )
                else:
                    logger.debug(f"✅ {metadata_file.name} válida")

            except Exception as e:
                logger.error(f"❌ Error al validar {metadata_file.name}: {e}")

    # Resumen final
    success_rate = valid_count / len(document_files) if document_files else 0

    logger.info("\n" + "=" * 60)
    logger.info("📊 RESUMEN DE VALIDACIÓN")
    logger.info("=" * 60)
    logger.info(f"📄 Total de documentos: {len(document_files)}")
    logger.info(f"✅ Válidos: {valid_count} ({success_rate:.1%})")
    logger.info(f"❌ Inválidos: {invalid_count}")
    logger.info(f"⚠️  Total de warnings: {len(total_warnings)}")
    logger.info(f"❌ Total de errores: {len(total_errors)}")
    logger.info("=" * 60)

    # Estadísticas agregadas
    if valid_count > 0:
        total_elements = sum(r['stats'].get('num_elements', 0) for r in validation_results if r['valid'])
        total_words = sum(r['stats'].get('num_words', 0) for r in validation_results if r['valid'])
        avg_elements = total_elements / valid_count
        avg_words = total_words / valid_count

        logger.info(f"📊 Elementos totales: {total_elements}")
        logger.info(f"📊 Palabras totales: {total_words}")
        logger.info(f"📊 Promedio de elementos/doc: {avg_elements:.1f}")
        logger.info(f"📊 Promedio de palabras/doc: {avg_words:.1f}")
        logger.info("=" * 60)

    # Guardar reporte si se solicitó
    if args.save_report:
        report = {
            "summary": {
                "total_documents": len(document_files),
                "valid": valid_count,
                "invalid": invalid_count,
                "success_rate": round(success_rate, 3),
                "total_warnings": len(total_warnings),
                "total_errors": len(total_errors)
            },
            "validation_results": validation_results,
            "all_errors": total_errors[:100],  # Primeros 100 errores
            "all_warnings": total_warnings[:100]  # Primeros 100 warnings
        }

        report_path = Path(args.save_report)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Reporte guardado en: {report_path}")

    # Exit code
    if invalid_count > 0:
        logger.error("\n❌ Validación fallida: hay documentos inválidos")
        sys.exit(1)
    else:
        logger.info("\n✅ Validación exitosa: todos los documentos son válidos")
        sys.exit(0)


if __name__ == "__main__":
    main()
