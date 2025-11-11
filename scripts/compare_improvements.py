#!/usr/bin/env python3
"""
Script para comparar mejoras entre versión original y mejorada

Genera reporte detallado de:
- Campos con [0,0] eliminados
- Duplicados resueltos
- Bboxes expandidos
- Calidad general mejorada

Uso:
    python scripts/compare_improvements.py \\
        --original-dir "output_old/" \\
        --improved-dir "output_fixed/" \\
        --report-file "improvements_report.json"
"""
import argparse
import sys
from pathlib import Path
import json
from typing import Dict, List
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.formatters.bbox_validator import BBoxValidator
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def analyze_document(doc_path: Path) -> Dict:
    """Analiza un documento y retorna métricas"""
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            document = json.load(f)

        elements = document.get('form', [])

        # Contar problemas
        zero_coords = 0
        invalid_bbox = 0
        small_bbox = 0
        low_confidence = 0

        # Detectar duplicados
        bbox_map = {}
        for elem in elements:
            bbox = tuple(elem.get('box', []))
            confidence = elem.get('confidence', 0)

            if bbox == (0, 0, 0, 0):
                zero_coords += 1

            if confidence < 0.5:
                low_confidence += 1

            # Agregar a mapa de duplicados
            if bbox not in bbox_map:
                bbox_map[bbox] = 0
            bbox_map[bbox] += 1

        # Contar duplicados
        duplicates = sum(count - 1 for count in bbox_map.values() if count > 1)

        return {
            'total_elements': len(elements),
            'zero_coords': zero_coords,
            'invalid_bbox': invalid_bbox,
            'small_bbox': small_bbox,
            'low_confidence': low_confidence,
            'duplicate_count': duplicates
        }

    except Exception as e:
        logger.error(f"Error al analizar {doc_path}: {e}")
        return None


def compare_documents(original_path: Path, improved_path: Path) -> Dict:
    """Compara un documento antes y después"""
    original_metrics = analyze_document(original_path)
    improved_metrics = analyze_document(improved_path)

    if not original_metrics or not improved_metrics:
        return None

    # Calcular mejoras
    improvements = {
        'document': original_path.stem.replace('_layoutlmv3', ''),
        'original': original_metrics,
        'improved': improved_metrics,
        'changes': {
            'total_elements': improved_metrics['total_elements'] - original_metrics['total_elements'],
            'zero_coords_removed': original_metrics['zero_coords'] - improved_metrics['zero_coords'],
            'duplicates_resolved': original_metrics['duplicate_count'] - improved_metrics['duplicate_count'],
            'low_confidence_filtered': original_metrics['low_confidence'] - improved_metrics['low_confidence']
        }
    }

    return improvements


def main():
    parser = argparse.ArgumentParser(
        description="Compara mejoras entre versión original y mejorada"
    )

    parser.add_argument(
        "--original-dir",
        type=str,
        required=True,
        help="Directorio con documentos originales"
    )

    parser.add_argument(
        "--improved-dir",
        type=str,
        required=True,
        help="Directorio con documentos mejorados"
    )

    parser.add_argument(
        "--report-file",
        type=str,
        default="improvements_report.json",
        help="Archivo para guardar reporte"
    )

    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Máximo de documentos a analizar"
    )

    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    improved_dir = Path(args.improved_dir)

    if not original_dir.exists():
        logger.error(f"Directorio original no existe: {original_dir}")
        sys.exit(1)

    if not improved_dir.exists():
        logger.error(f"Directorio mejorado no existe: {improved_dir}")
        sys.exit(1)

    # Encontrar documentos
    original_files = list(original_dir.glob("*_layoutlmv3.json"))

    if args.max_documents:
        original_files = original_files[:args.max_documents]

    logger.info("=" * 60)
    logger.info("📊 ANÁLISIS DE MEJORAS")
    logger.info("=" * 60)
    logger.info(f"📁 Original: {original_dir}")
    logger.info(f"📁 Mejorado: {improved_dir}")
    logger.info(f"📄 Documentos: {len(original_files)}")
    logger.info("=" * 60)

    # Comparar documentos
    all_improvements = []
    processed = 0

    for original_file in original_files:
        document_id = original_file.stem
        improved_file = improved_dir / original_file.name

        if not improved_file.exists():
            logger.warning(f"Documento mejorado no encontrado: {document_id}")
            continue

        improvements = compare_documents(original_file, improved_file)

        if improvements:
            all_improvements.append(improvements)
            processed += 1

            changes = improvements['changes']
            logger.info(
                f"✅ {document_id}: "
                f"elementos {changes['total_elements']:+d}, "
                f"[0,0] -{changes['zero_coords_removed']}, "
                f"duplicados -{changes['duplicates_resolved']}, "
                f"low conf -{changes['low_confidence_filtered']}"
            )

    # Calcular estadísticas agregadas
    if all_improvements:
        total_original_elements = sum(d['original']['total_elements'] for d in all_improvements)
        total_improved_elements = sum(d['improved']['total_elements'] for d in all_improvements)

        total_zero_removed = sum(d['changes']['zero_coords_removed'] for d in all_improvements)
        total_duplicates_resolved = sum(d['changes']['duplicates_resolved'] for d in all_improvements)
        total_low_conf_filtered = sum(d['changes']['low_confidence_filtered'] for d in all_improvements)

        # Promedios por documento
        avg_zero_original = sum(d['original']['zero_coords'] for d in all_improvements) / len(all_improvements)
        avg_zero_improved = sum(d['improved']['zero_coords'] for d in all_improvements) / len(all_improvements)

        avg_dup_original = sum(d['original']['duplicate_count'] for d in all_improvements) / len(all_improvements)
        avg_dup_improved = sum(d['improved']['duplicate_count'] for d in all_improvements) / len(all_improvements)

        summary = {
            'documents_analyzed': processed,
            'totals': {
                'original_elements': total_original_elements,
                'improved_elements': total_improved_elements,
                'elements_removed': total_original_elements - total_improved_elements,
                'zero_coords_removed': total_zero_removed,
                'duplicates_resolved': total_duplicates_resolved,
                'low_confidence_filtered': total_low_conf_filtered
            },
            'averages_per_document': {
                'zero_coords_original': round(avg_zero_original, 2),
                'zero_coords_improved': round(avg_zero_improved, 2),
                'duplicates_original': round(avg_dup_original, 2),
                'duplicates_improved': round(avg_dup_improved, 2)
            },
            'improvement_percentage': {
                'zero_coords_reduction': round((1 - avg_zero_improved / avg_zero_original) * 100 if avg_zero_original > 0 else 0, 1),
                'duplicates_reduction': round((1 - avg_dup_improved / avg_dup_original) * 100 if avg_dup_original > 0 else 0, 1)
            }
        }

        # Crear reporte
        report = {
            'summary': summary,
            'documents': all_improvements
        }

        # Guardar reporte
        report_path = Path(args.report_file)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Mostrar resumen
        logger.info("\n" + "=" * 60)
        logger.info("📊 RESUMEN DE MEJORAS")
        logger.info("=" * 60)
        logger.info(f"📄 Documentos analizados: {processed}")
        logger.info(f"\n📉 REDUCCIÓN DE PROBLEMAS:")
        logger.info(f"   [0,0] coords: {avg_zero_original:.2f} → {avg_zero_improved:.2f} por documento ({summary['improvement_percentage']['zero_coords_reduction']:.1f}% reducción)")
        logger.info(f"   Duplicados: {avg_dup_original:.2f} → {avg_dup_improved:.2f} por documento ({summary['improvement_percentage']['duplicates_reduction']:.1f}% reducción)")
        logger.info(f"\n✅ TOTAL ELIMINADO:")
        logger.info(f"   Campos [0,0]: {total_zero_removed}")
        logger.info(f"   Duplicados: {total_duplicates_resolved}")
        logger.info(f"   Low confidence: {total_low_conf_filtered}")
        logger.info(f"   Total elementos removidos: {total_original_elements - total_improved_elements}")
        logger.info(f"\n💾 Reporte guardado en: {report_path}")
        logger.info("=" * 60)

    else:
        logger.warning("No se pudieron comparar documentos")
        sys.exit(1)


if __name__ == "__main__":
    main()
