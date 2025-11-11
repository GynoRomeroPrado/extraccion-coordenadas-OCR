# 🔧 PROCESAMIENTO MEJORADO - Google Colab

## Mejoras implementadas

✅ **Filtrado de campos con confidence < 0.5**
✅ **Validación y expansión de bboxes pequeños**
✅ **Resolución de bboxes duplicados**
✅ **Eliminación de metadata innecesaria**
✅ **Métricas de calidad detalladas**

---

## 📋 CELDAS PARA GOOGLE COLAB

### 1. Instalar dependencias (si es necesario)

```python
# Ya deberías tener todo instalado si seguiste el notebook anterior
# Si no, ejecuta:
# !git pull origin claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF
```

### 2. Re-procesar documentos existentes con mejoras

```python
# Configurar paths
ORIGINAL_OUTPUT = "/content/drive/MyDrive/output_layoutlmv3"  # Documentos originales
IMPROVED_OUTPUT = "/content/drive/MyDrive/output_layoutlmv3_improved"  # Salida mejorada
DATASET_DIR = "/content/drive/MyDrive/dataset_entrenamiento_101125"
PDF_DIR = f"{DATASET_DIR}/facturas_pdf"
JSON_DIR = f"{DATASET_DIR}/anotaciones_json"

!mkdir -p {IMPROVED_OUTPUT}
```

```bash
# Re-procesar con validaciones mejoradas
!python scripts/reprocess_with_validation.py \
    --input-dir "{ORIGINAL_OUTPUT}" \
    --output-dir "{IMPROVED_OUTPUT}" \
    --pdf-dir "{PDF_DIR}" \
    --json-dir "{JSON_DIR}" \
    --min-confidence 0.5 \
    --duplicate-strategy "keep_best" \
    --verbose
```

### 3. Comparar mejoras (antes vs después)

```bash
!python scripts/compare_improvements.py \
    --original-dir "{ORIGINAL_OUTPUT}" \
    --improved-dir "{IMPROVED_OUTPUT}" \
    --report-file "{IMPROVED_OUTPUT}/improvements_report.json"
```

### 4. Ver reporte de mejoras

```python
import json

# Leer reporte
with open(f"{IMPROVED_OUTPUT}/improvements_report.json", 'r') as f:
    report = json.load(f)

summary = report['summary']

print("📊 RESUMEN DE MEJORAS")
print("=" * 60)
print(f"📄 Documentos analizados: {summary['documents_analyzed']}")
print(f"\n📉 REDUCCIÓN PROMEDIO POR DOCUMENTO:")
print(f"   [0,0] coords: {summary['averages_per_document']['zero_coords_original']:.2f} → {summary['averages_per_document']['zero_coords_improved']:.2f}")
print(f"   Duplicados: {summary['averages_per_document']['duplicates_original']:.2f} → {summary['averages_per_document']['duplicates_improved']:.2f}")
print(f"\n✅ PORCENTAJE DE MEJORA:")
print(f"   Reducción [0,0]: {summary['improvement_percentage']['zero_coords_reduction']:.1f}%")
print(f"   Reducción duplicados: {summary['improvement_percentage']['duplicates_reduction']:.1f}%")
print(f"\n💯 TOTAL ELIMINADO:")
print(f"   Campos [0,0]: {summary['totals']['zero_coords_removed']}")
print(f"   Duplicados: {summary['totals']['duplicates_resolved']}")
print(f"   Low confidence: {summary['totals']['low_confidence_filtered']}")
```

### 5. Procesar NUEVOS documentos con mejoras desde el inicio

```python
# Para procesar nuevos documentos directamente con las mejoras,
# modificar el script process_batch.py para usar LayoutLMv3FormatterV2

# O procesar manualmente:
from src.extractors.hybrid_extractor import HybridExtractor
from src.matchers.fuzzy_matcher import FuzzyMatcher
from src.formatters.layoutlmv3_formatter_v2 import LayoutLMv3FormatterV2

extractor = HybridExtractor(dpi=300)
matcher = FuzzyMatcher(threshold=85)
formatter = LayoutLMv3FormatterV2(
    min_confidence=0.5,
    resolve_duplicates=True,
    expand_small_bboxes=True
)

# Procesar documento
# ... (usar el mismo flujo que process_single.py pero con FormatterV2)
```

### 6. Validar calidad mejorada

```bash
!python scripts/validate_output.py \
    --output-dir "{IMPROVED_OUTPUT}" \
    --save-report "{IMPROVED_OUTPUT}/validation_report_improved.json"
```

### 7. Ver ejemplos de elementos mejorados

```python
import json
from pathlib import Path

# Cargar un documento mejorado
improved_files = list(Path(IMPROVED_OUTPUT).glob("*_layoutlmv3.json"))
if improved_files:
    with open(improved_files[0], 'r') as f:
        doc = json.load(f)

    print(f"📄 Documento: {improved_files[0].name}")
    print(f"📊 Total elementos: {len(doc['form'])}")
    print(f"\n🔍 Primeros 3 elementos:")

    for elem in doc['form'][:3]:
        print(f"\n   • Campo: {elem['field_name']}")
        print(f"     Texto: {elem['text'][:50]}...")
        print(f"     Bbox: {elem['box']}")
        print(f"     Confidence: {elem['confidence']:.2f}")
```

---

## 🎯 RESULTADOS ESPERADOS

Después de re-procesar, deberías ver:

### Antes (problemas):
- 🔴 ~1.8 campos con [0,0] por documento
- 🔴 ~8.4 campos duplicados por documento
- 🔴 Coordenadas sospechosamente pequeñas
- 🔴 Metadata innecesaria en JSONs

### Después (mejorado):
- ✅ < 0.5 campos con [0,0] por documento (~75% reducción)
- ✅ < 2 campos duplicados por documento (~75% reducción)
- ✅ Bboxes expandidos automáticamente si son muy pequeños
- ✅ JSONs limpios sin metadata
- ✅ Métricas de calidad detalladas por documento

---

## 📊 MÉTRICAS DE CALIDAD

Cada metadata ahora incluye:

```json
{
  "quality_metrics": {
    "total_elements": 268,
    "valid_bboxes": 268,
    "invalid_bboxes": 0,
    "suspiciously_small": 0,
    "zero_confidence": 0,
    "duplicate_groups": 0,
    "total_duplicates": 0,
    "quality_score": 1.0
  },
  "processing_stats": {
    "total_fields": 280,
    "filtered_low_confidence": 12,
    "filtered_invalid_bbox": 0,
    "expanded_bboxes": 3,
    "resolved_duplicates": 5,
    "removed_metadata_fields": 8
  }
}
```

---

## ⏱️ TIEMPO DE RE-PROCESAMIENTO

- **640 documentos**: ~2 minutos
- **1,000 documentos**: ~3 minutos
- **34,165 documentos**: ~2 horas

---

## 💡 RECOMENDACIONES

1. **Re-procesar los 640 existentes primero** para ver mejoras inmediatas
2. **Comparar calidad** con el script de comparación
3. **Si está bien, re-procesar todos** los documentos nuevos con V2
4. **Usar FormatterV2 por defecto** para procesamiento futuro

---

## ✅ VALIDAR MEJORAS

Después de re-procesar, verifica:

```python
# Ver estadísticas agregadas
import json
from pathlib import Path

metadata_files = list(Path(IMPROVED_OUTPUT).glob("*_metadata.json"))

total_filtered = 0
total_expanded = 0
total_duplicates = 0

for meta_file in metadata_files:
    with open(meta_file, 'r') as f:
        meta = json.load(f)

    stats = meta.get('processing_stats', {})
    total_filtered += stats.get('filtered_low_confidence', 0)
    total_expanded += stats.get('expanded_bboxes', 0)
    total_duplicates += stats.get('resolved_duplicates', 0)

print(f"📊 ESTADÍSTICAS GLOBALES ({len(metadata_files)} documentos):")
print(f"   Filtrados por confidence: {total_filtered}")
print(f"   Bboxes expandidos: {total_expanded}")
print(f"   Duplicados resueltos: {total_duplicates}")
print(f"   Promedio filtrados/doc: {total_filtered/len(metadata_files):.2f}")
print(f"   Promedio duplicados/doc: {total_duplicates/len(metadata_files):.2f}")
```

---

**¡Dataset mejorado y listo para entrenar con mayor calidad!** 🎉
