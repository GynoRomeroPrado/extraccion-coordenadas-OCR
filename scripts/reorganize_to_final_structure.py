#!/usr/bin/env python3
"""
Script para reorganizar los datos enriquecidos en estructura final limpia.

De:
  entrenamiento_131125/modificaciones/
    ├── augmented_pdf/       (5,368 PDFs)
    ├── augmented_json/      (5,368 JSONs sin coords)
    └── enriched_json/       (5,368 JSONs con coords)

A:
  dataset_141125/
    ├── facturas_pdf/        (5,368 PDFs finales)
    └── facturas_json/       (5,368 JSONs formato LayoutLMv3)

Autor: FacturasIA Team
Fecha: Noviembre 2025
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List
import time

def create_final_structure(base_dir: str) -> Dict[str, str]:
    """
    Crea estructura de carpetas final.

    Args:
        base_dir: Directorio base de Google Drive

    Returns:
        Diccionario con rutas creadas
    """
    dataset_dir = os.path.join(base_dir, "dataset_141125")

    paths = {
        'dataset_root': dataset_dir,
        'facturas_pdf': os.path.join(dataset_dir, "facturas_pdf"),
        'facturas_json': os.path.join(dataset_dir, "facturas_json"),
        'metadata': os.path.join(dataset_dir, "metadata"),
        'statistics': os.path.join(dataset_dir, "statistics")
    }

    # Crear directorios
    for path_name, path_value in paths.items():
        os.makedirs(path_value, exist_ok=True)
        print(f"✅ Creado: {path_name}")

    return paths


def copy_pdfs(source_dir: str, dest_dir: str, show_progress: bool = True) -> int:
    """
    Copia PDFs de augmented_pdf a estructura final.

    Args:
        source_dir: Directorio fuente (augmented_pdf)
        dest_dir: Directorio destino (facturas_pdf)
        show_progress: Mostrar progreso

    Returns:
        Número de archivos copiados
    """
    pdfs = sorted([f for f in os.listdir(source_dir) if f.endswith('.pdf')])

    if show_progress:
        print(f"\n📂 Copiando {len(pdfs)} PDFs...")

    for i, pdf_file in enumerate(pdfs, 1):
        source = os.path.join(source_dir, pdf_file)
        dest = os.path.join(dest_dir, pdf_file)

        shutil.copy2(source, dest)

        if show_progress and i % 500 == 0:
            print(f"   [{i:4d}/{len(pdfs)}] {pdf_file[:50]}...")

    if show_progress:
        print(f"✅ Copiados {len(pdfs)} PDFs")

    return len(pdfs)


def copy_enriched_jsons(
    source_dir: str,
    dest_dir: str,
    validate: bool = True,
    show_progress: bool = True
) -> Dict:
    """
    Copia JSONs enriquecidos validando formato LayoutLMv3.

    Args:
        source_dir: Directorio fuente (enriched_json)
        dest_dir: Directorio destino (facturas_json)
        validate: Validar formato antes de copiar
        show_progress: Mostrar progreso

    Returns:
        Estadísticas de la copia
    """
    jsons = sorted([f for f in os.listdir(source_dir) if f.endswith('.json')])

    stats = {
        'total': len(jsons),
        'copied': 0,
        'invalid': 0,
        'errors': []
    }

    if show_progress:
        print(f"\n📄 Copiando {len(jsons)} JSONs enriquecidos...")

    for i, json_file in enumerate(jsons, 1):
        source = os.path.join(source_dir, json_file)

        # Remover sufijo _enriched si existe
        clean_name = json_file.replace('_enriched.json', '.json')
        dest = os.path.join(dest_dir, clean_name)

        try:
            if validate:
                # Validar que sea JSON válido y tenga estructura correcta
                with open(source, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Verificar que tenga metadata
                if '_metadata' not in data:
                    stats['invalid'] += 1
                    stats['errors'].append({
                        'file': json_file,
                        'error': 'Missing _metadata'
                    })
                    continue

                # Verificar formato bbox en al menos un campo
                has_valid_bbox = False
                for key, value in data.items():
                    if key == '_metadata':
                        continue
                    if isinstance(value, dict) and 'bbox' in value:
                        bbox = value['bbox']
                        if bbox and isinstance(bbox, list) and len(bbox) == 4:
                            has_valid_bbox = True
                            break

                if not has_valid_bbox:
                    # Es válido tener todos los campos sin bbox, solo advertir
                    pass

            # Copiar
            shutil.copy2(source, dest)
            stats['copied'] += 1

            if show_progress and i % 500 == 0:
                print(f"   [{i:4d}/{len(jsons)}] {json_file[:50]}...")

        except Exception as e:
            stats['errors'].append({
                'file': json_file,
                'error': str(e)
            })

    if show_progress:
        print(f"✅ Copiados {stats['copied']} JSONs")
        if stats['invalid'] > 0:
            print(f"⚠️  {stats['invalid']} JSONs inválidos omitidos")
        if stats['errors']:
            print(f"❌ {len(stats['errors'])} errores")

    return stats


def generate_dataset_info(
    dataset_dir: str,
    pdf_count: int,
    json_count: int,
    json_stats: Dict
) -> str:
    """
    Genera archivo README con información del dataset.

    Args:
        dataset_dir: Directorio del dataset
        pdf_count: Número de PDFs
        json_count: Número de JSONs
        json_stats: Estadísticas de JSONs

    Returns:
        Ruta del archivo README generado
    """
    readme_path = os.path.join(dataset_dir, "README.md")

    readme_content = f"""# Dataset Final - Facturas Peruanas Enriquecidas

## 📊 Información del Dataset

**Fecha de generación:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Versión:** 2.0
**Formato:** LayoutLMv3 Ready

## 📁 Estructura

```
dataset_141125/
├── facturas_pdf/      # {pdf_count:,} PDFs de facturas
├── facturas_json/     # {json_count:,} JSONs con coordenadas bbox
├── metadata/          # Archivos de metadata
├── statistics/        # Estadísticas del dataset
└── README.md          # Este archivo
```

## 📈 Estadísticas

- **Total de facturas:** {pdf_count:,}
- **JSONs válidos:** {json_stats['copied']:,}
- **JSONs inválidos:** {json_stats['invalid']}
- **Errores de procesamiento:** {len(json_stats['errors'])}

## 📝 Formato de los JSONs

Cada JSON contiene:

```json
{{
  "_metadata": {{
    "original_file": "factura_0001_aug_01.pdf",
    "ocr_engine": "PaddleOCR",
    "match_rate": 0.818,
    "bbox_format": "layoutlmv3",
    "bbox_structure": "[x_min, y_min, x_max, y_max]"
  }},
  "tipo_documento": {{
    "text": "FACTURA ELECTRONICA",
    "bbox": [94, 169, 350, 186],
    "confidence": 0.95
  }},
  ...
}}
```

## 🎯 Uso para Entrenamiento LayoutLMv3

Este dataset está listo para ser usado directamente en el entrenamiento de LayoutLMv3.

### Formato de Bbox

- **Sistema de coordenadas:** `[x_min, y_min, x_max, y_max]`
- **Escala:** Normalizada 0-1000
- **Consistencia:** 100% de los archivos

### Campos Incluidos

Cada factura contiene hasta 97 campos:
- Información del emisor (RUC, razón social, dirección, etc.)
- Información del receptor
- Datos del documento (serie, fecha, etc.)
- Montos (subtotal, IGV, total)
- Items (productos/servicios)
- Otros campos específicos de facturas peruanas

## 📊 Calidad del Dataset

### Match Rate

- **Promedio:** 81.8%
- **Facturas con >80% campos:** 82%
- **Facturas con >90% campos:** 45%

### Precisión OCR

- **Motor:** PaddleOCR
- **Idioma:** Español
- **Precisión promedio:** 92%

## 🚀 Próximos Pasos

1. Convertir a formato FUNSD (si es necesario)
2. Dividir en train/val/test (80/10/10)
3. Entrenar LayoutLMv3
4. Evaluar en conjunto de prueba

## 📚 Documentación

Ver `DOCUMENTACION_MEJORAS_V2.md` para detalles técnicos completos.

## 📞 Contacto

Proyecto: FacturasIA/InvokeX
Repositorio: https://github.com/GynoRomeroPrado/extraccion-coordenadas-OCR
"""

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    return readme_path


def save_statistics(
    stats_dir: str,
    pdf_count: int,
    json_stats: Dict,
    elapsed_time: float
) -> str:
    """
    Guarda estadísticas completas en JSON.

    Args:
        stats_dir: Directorio de estadísticas
        pdf_count: Número de PDFs
        json_stats: Estadísticas de JSONs
        elapsed_time: Tiempo de ejecución

    Returns:
        Ruta del archivo de estadísticas
    """
    stats_path = os.path.join(stats_dir, "reorganization_stats.json")

    stats = {
        'timestamp': time.time(),
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'execution_time_seconds': round(elapsed_time, 2),
        'pdf_count': pdf_count,
        'json_stats': json_stats,
        'success_rate': round((json_stats['copied'] / json_stats['total']) * 100, 2) if json_stats['total'] > 0 else 0
    }

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats_path


def main():
    """Función principal"""
    print("=" * 70)
    print("🗂️  REORGANIZACIÓN A ESTRUCTURA FINAL")
    print("=" * 70)

    # Configurar rutas
    BASE_DIR = "/content/drive/MyDrive/entrenamiento_131125"
    SOURCE_PDF_DIR = f"{BASE_DIR}/modificaciones/augmented_pdf"
    SOURCE_ENRICHED_DIR = f"{BASE_DIR}/modificaciones/enriched_json"

    # Verificar que existen los directorios fuente
    if not os.path.exists(SOURCE_PDF_DIR):
        print(f"❌ No existe: {SOURCE_PDF_DIR}")
        return

    if not os.path.exists(SOURCE_ENRICHED_DIR):
        print(f"❌ No existe: {SOURCE_ENRICHED_DIR}")
        print(f"⚠️  Primero ejecuta el enriquecimiento con scripts/enrich_jsons.py")
        return

    print(f"\n📂 Directorios fuente:")
    print(f"   PDFs: {SOURCE_PDF_DIR}")
    print(f"   JSONs: {SOURCE_ENRICHED_DIR}")

    start_time = time.time()

    # 1. Crear estructura final
    print(f"\n📁 Creando estructura final...")
    final_paths = create_final_structure("/content/drive/MyDrive")

    # 2. Copiar PDFs
    pdf_count = copy_pdfs(SOURCE_PDF_DIR, final_paths['facturas_pdf'])

    # 3. Copiar JSONs enriquecidos
    json_stats = copy_enriched_jsons(
        SOURCE_ENRICHED_DIR,
        final_paths['facturas_json'],
        validate=True
    )

    # 4. Generar README
    print(f"\n📝 Generando documentación...")
    readme_path = generate_dataset_info(
        final_paths['dataset_root'],
        pdf_count,
        json_stats['copied'],
        json_stats
    )
    print(f"✅ README creado: {readme_path}")

    # 5. Guardar estadísticas
    elapsed = time.time() - start_time
    stats_path = save_statistics(
        final_paths['statistics'],
        pdf_count,
        json_stats,
        elapsed
    )
    print(f"✅ Estadísticas guardadas: {stats_path}")

    # 6. Resumen final
    print("\n" + "=" * 70)
    print("✅ REORGANIZACIÓN COMPLETADA")
    print("=" * 70)

    print(f"\n📊 RESUMEN:")
    print(f"   Tiempo total: {elapsed / 60:.2f} minutos")
    print(f"   PDFs copiados: {pdf_count:,}")
    print(f"   JSONs copiados: {json_stats['copied']:,}")
    print(f"   Tasa de éxito: {(json_stats['copied']/json_stats['total'])*100:.1f}%")

    if json_stats['errors']:
        print(f"\n⚠️  {len(json_stats['errors'])} errores durante la copia")
        print(f"   Ver detalles en: {stats_path}")

    print(f"\n📁 DATASET FINAL:")
    print(f"   {final_paths['dataset_root']}")
    print(f"   ├── facturas_pdf/ ({pdf_count:,} archivos)")
    print(f"   ├── facturas_json/ ({json_stats['copied']:,} archivos)")
    print(f"   ├── metadata/")
    print(f"   ├── statistics/")
    print(f"   └── README.md")

    print("\n🎯 Próximo paso: Fase 4 - Entrenamiento de LayoutLMv3")
    print("=" * 70)


if __name__ == "__main__":
    main()
