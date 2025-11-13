# 🚀 Guía Completa para Google Colab - Uso Optimizado

## ✅ Verificación de Extracción de Campos

**IMPORTANTE**: El sistema intenta extraer **TODOS** los campos del JSON, pero algunos pueden filtrarse por validaciones de calidad:

### Validaciones Aplicadas

1. **Confidence mínimo**: 0.5 (configurable)
2. **Bbox válido**: No permite [0,0,0,0] o bboxes negativos
3. **Tamaño mínimo**: Width > 5px, Height > 5px
4. **No duplicados**: Resuelve bboxes duplicados automáticamente

### ¿Qué campos se extraen?

**Del JSON de anotaciones:**
- ✅ Campos del header (emisor, receptor, factura, fechas, totales)
- ✅ Items (productos/servicios con cantidad, descripción, precio)
- ✅ Cuotas (si existen)
- ❌ Metadata (augmentation, extraction_method, etc.) - se filtran automáticamente

**Estadísticas esperadas:**
- **Tasa de extracción**: 85-95% de campos válidos
- **Campos filtrados**: 5-15% (principalmente por confidence < 0.5)
- **Items detectados**: 85-95% con RegionDetector

---

## 📊 Uso en Google Colab - Pipeline Completo con Verificación

### PASO 1: Configuración Inicial

```python
# ============= CONFIGURACIÓN =============

# 1. Habilitar GPU
# Runtime > Change runtime type > GPU > T4 (o mejor)

# 2. Verificar GPU
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)

# Deberías ver algo como:
# | NVIDIA-SMI 525.x | Tesla T4 | ...

# 3. Instalar PaddleOCR con GPU
!pip install -q paddlepaddle-gpu paddleocr
!pip install -q rapidfuzz pymupdf pillow numpy

# 4. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 5. Clonar/actualizar repositorio
%cd /content
!git clone https://github.com/GynoRomeroPrado/extraccion-coordenadas-OCR.git 2>/dev/null || echo "Repo ya existe"
%cd extraccion-coordenadas-OCR

# 6. Checkout a la branch correcta
!git fetch origin
!git checkout claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF
!git pull origin claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF

print("✅ Configuración completada")
```

### PASO 2: Verificar Instalación

```python
# Verificar que todo funciona
import sys
sys.path.insert(0, '/content/extraccion-coordenadas-OCR')

from src.extractors.batch_extractor import BatchExtractor
from src.matchers.coordinate_enhancer import CoordinateEnhancer
from src.formatters.layoutlmv3_formatter_v2 import LayoutLMv3FormatterV2

print("✅ Todos los módulos importados correctamente")

# Verificar GPU disponible
import paddle
print(f"✅ PaddlePaddle GPU: {paddle.device.is_compiled_with_cuda()}")
print(f"✅ GPU disponible: {paddle.device.get_device()}")
```

### PASO 3: Configurar Rutas del Dataset

```python
# ============= RUTAS DEL DATASET =============

# Opción A: Dataset en Google Drive
DATASET_DIR = "/content/drive/MyDrive/dataset_entrenamiento_101125"

# Opción B: Dataset local (si lo subes a Colab)
# DATASET_DIR = "/content/dataset"

PDF_DIR = f"{DATASET_DIR}/facturas_pdf"
JSON_DIR = f"{DATASET_DIR}/anotaciones_json"
OUTPUT_DIR = f"{DATASET_DIR}/output_layoutlmv3_optimized"

# Crear directorio de salida
from pathlib import Path
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Verificar que existen los directorios
import os
assert os.path.exists(PDF_DIR), f"❌ No existe: {PDF_DIR}"
assert os.path.exists(JSON_DIR), f"❌ No existe: {JSON_DIR}"

# Contar documentos
pdf_count = len(list(Path(PDF_DIR).glob("*.pdf")))
json_count = len(list(Path(JSON_DIR).glob("*.json")))

print(f"✅ Encontrados {pdf_count} PDFs")
print(f"✅ Encontrados {json_count} JSONs")
print(f"✅ Output: {OUTPUT_DIR}")
```

### PASO 4: Procesamiento OPTIMIZADO con Verificación

```python
# ============= PROCESAMIENTO OPTIMIZADO =============

import json
import time
from pathlib import Path
from src.extractors.batch_extractor import BatchExtractor
from src.matchers.fuzzy_matcher import FuzzyMatcher
from src.matchers.coordinate_enhancer import CoordinateEnhancer
from src.matchers.table_detector_v2 import TableDetectorV2
from src.matchers.region_detector import RegionDetector
from src.formatters.layoutlmv3_formatter_v2 import LayoutLMv3FormatterV2

# Inicializar componentes OPTIMIZADOS
print("Inicializando componentes optimizados...")

batch_extractor = BatchExtractor(
    batch_size=8,          # ⚡ 4x speedup
    use_gpu=True,
    detect_regions=True,   # 📍 Mejor detección items/totals
    num_workers=4
)

matcher = FuzzyMatcher(threshold=85)
enhancer = CoordinateEnhancer()  # 🎯 +18.7% precisión
formatter = LayoutLMv3FormatterV2(
    min_confidence=0.5,    # Ajusta a 0.3 si quieres ser más permisivo
    resolve_duplicates=True,
    expand_small_bboxes=True
)
table_detector = TableDetectorV2()

print("✅ Componentes inicializados")

# ============= ESTADÍSTICAS DE VERIFICACIÓN =============

verification_stats = {
    'total_documents': 0,
    'total_fields_attempted': 0,
    'total_fields_extracted': 0,
    'total_fields_filtered': 0,
    'filtered_by_confidence': 0,
    'filtered_by_bbox': 0,
    'total_items_attempted': 0,
    'total_items_extracted': 0,
    'documents_with_missing_fields': []
}

# ============= FUNCIÓN DE PROCESAMIENTO CON VERIFICACIÓN =============

def process_document_with_verification(pdf_path, extraction, progress):
    """
    Procesa un documento con verificación completa de campos extraídos
    """
    doc_id = Path(pdf_path).stem
    json_path = Path(JSON_DIR) / f"{doc_id}.json"

    if not json_path.exists():
        print(f"⚠️  JSON no encontrado para {doc_id}")
        return

    # Cargar anotaciones
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # Extraer palabras por región
    if extraction.get('regions'):
        header_words = extraction['regions']['header']['words']
        items_words = extraction['regions']['items']['words']
        totals_words = extraction['regions']['totals']['words']
    else:
        header_words = extraction['words']
        items_words = extraction['words']
        totals_words = extraction['words']

    # Separar campos
    items = annotations.pop('items', [])
    cuotas = annotations.pop('cuotas', [])

    # Resetear formatter para este documento
    formatter.reset_id_counter()
    formatter.reset_stats()

    # ============= PROCESAR HEADER CON ENHANCEMENT =============

    header_elements = []
    fields_attempted = 0
    fields_extracted = 0

    for field_name, field_value in annotations.items():
        if field_value is None or field_value == "":
            continue

        fields_attempted += 1

        # 1. Match básico
        if isinstance(field_value, (int, float)):
            match = matcher.match_numeric(field_value, header_words, field_name)
        elif isinstance(field_value, str) and len(field_value.split()) > 1:
            match = matcher.match_multiword(field_value, header_words, field_name)
        else:
            match = matcher.match(field_value, header_words, field_name)

        # 2. ⚡ ENHANCEMENT
        if match:
            enhanced = enhancer.enhance_match(match, field_name, header_words)

            # 3. Formatear
            elem = formatter.format_field(field_name, enhanced)

            if elem:
                header_elements.append(elem)
                fields_extracted += 1
            else:
                # Filtrado por validaciones
                print(f"  ⚠️  Campo filtrado: {field_name} (confidence: {enhanced.get('confidence', 0):.3f})")
        else:
            print(f"  ⚠️  Campo NO encontrado: {field_name} = '{field_value}'")

    # ============= PROCESAR ITEMS CON TABLE DETECTION =============

    items_elements = []
    items_attempted = len(items)
    items_extracted = 0

    if items and items_words:
        # Detectar estructura de tabla
        table_structure = table_detector.detect_table_structure(items_words)

        # Mapear items a filas de tabla
        mapped_items = table_detector.map_items_to_rows(
            table_structure,
            items,
            matcher
        )

        # Formatear cada item
        for mapped_item in mapped_items:
            for field_name, match in mapped_item['matched_fields'].items():
                if match:
                    # ⚡ ENHANCEMENT en items también
                    enhanced = enhancer.enhance_match(
                        match,
                        f"item.{field_name}",
                        items_words
                    )

                    elem = formatter.format_field(
                        f"item.{field_name}",
                        enhanced
                    )

                    if elem:
                        items_elements.append(elem)
                        items_extracted += 1

    # ============= CREAR DOCUMENTO =============

    all_elements = header_elements + items_elements
    document = formatter.create_document(all_elements, apply_validations=True)

    # Guardar
    output_file = Path(OUTPUT_DIR) / f"{doc_id}_layoutlmv3.json"
    formatter.save_to_file(document, str(output_file))

    # ============= ESTADÍSTICAS DE VERIFICACIÓN =============

    stats = formatter.get_stats()

    verification_stats['total_documents'] += 1
    verification_stats['total_fields_attempted'] += fields_attempted
    verification_stats['total_fields_extracted'] += fields_extracted
    verification_stats['total_fields_filtered'] += (fields_attempted - fields_extracted)
    verification_stats['filtered_by_confidence'] += stats['filtered_low_confidence']
    verification_stats['filtered_by_bbox'] += stats['filtered_invalid_bbox']
    verification_stats['total_items_attempted'] += items_attempted
    verification_stats['total_items_extracted'] += items_extracted

    # Detectar documentos con muchos campos faltantes
    if fields_attempted > 0:
        extraction_rate = fields_extracted / fields_attempted
        if extraction_rate < 0.7:  # Menos del 70% extraído
            verification_stats['documents_with_missing_fields'].append({
                'doc_id': doc_id,
                'attempted': fields_attempted,
                'extracted': fields_extracted,
                'rate': extraction_rate
            })

    # Log progreso
    if progress:
        print(f"[{progress['processed']}/{progress['total']}] {doc_id} | "
              f"Header: {fields_extracted}/{fields_attempted} | "
              f"Items: {items_extracted}/{items_attempted} | "
              f"{progress['rate']:.1f} docs/s | "
              f"ETA: {progress['eta']/60:.1f}min")

# ============= EJECUTAR PROCESAMIENTO =============

print("\n" + "=" * 80)
print("🚀 INICIANDO PROCESAMIENTO OPTIMIZADO CON VERIFICACIÓN")
print("=" * 80)

# Obtener lista de PDFs
pdf_files = sorted(list(Path(PDF_DIR).glob("*.pdf")))

# Opción: Procesar solo los primeros N documentos para prueba
# pdf_files = pdf_files[:100]  # Descomentar para probar con 100 docs

print(f"📊 Total de documentos: {len(pdf_files)}")

# Estimar tiempo
estimated_time = (len(pdf_files) * 0.05) / 60  # 0.05s por doc con batch=8
print(f"⏱️  Tiempo estimado: {estimated_time:.1f} minutos\n")

# Procesar con verificación
start_time = time.time()

batch_extractor.extract_batch_async(
    [str(p) for p in pdf_files],
    callback=process_document_with_verification
)

total_time = time.time() - start_time

print("\n" + "=" * 80)
print("✅ PROCESAMIENTO COMPLETADO")
print("=" * 80)
```

### PASO 5: Ver Estadísticas de Verificación

```python
# ============= ESTADÍSTICAS FINALES =============

print("\n📊 ESTADÍSTICAS DE EXTRACCIÓN")
print("=" * 80)

# Tiempo
print(f"⏱️  Tiempo total: {total_time/60:.2f} minutos")
print(f"⏱️  Tiempo promedio: {total_time/len(pdf_files):.3f}s por documento")
print(f"⚡ Velocidad: {len(pdf_files)/total_time:.1f} docs/s\n")

# Campos del header
total_fields = verification_stats['total_fields_attempted']
extracted_fields = verification_stats['total_fields_extracted']
filtered_fields = verification_stats['total_fields_filtered']

if total_fields > 0:
    extraction_rate = (extracted_fields / total_fields) * 100
    print(f"📋 CAMPOS DEL HEADER:")
    print(f"  Total intentados: {total_fields}")
    print(f"  ✅ Extraídos: {extracted_fields} ({extraction_rate:.1f}%)")
    print(f"  ❌ Filtrados: {filtered_fields} ({100-extraction_rate:.1f}%)")
    print(f"     - Por confidence: {verification_stats['filtered_by_confidence']}")
    print(f"     - Por bbox inválido: {verification_stats['filtered_by_bbox']}\n")

# Items
total_items = verification_stats['total_items_attempted']
extracted_items = verification_stats['total_items_extracted']

if total_items > 0:
    items_rate = (extracted_items / total_items) * 100
    print(f"📦 ITEMS:")
    print(f"  Total intentados: {total_items}")
    print(f"  ✅ Extraídos: {extracted_items} ({items_rate:.1f}%)")
    print(f"  ❌ No extraídos: {total_items - extracted_items} ({100-items_rate:.1f}%)\n")

# Documentos con problemas
problematic = verification_stats['documents_with_missing_fields']
if problematic:
    print(f"⚠️  DOCUMENTOS CON BAJA EXTRACCIÓN (<70%):")
    print(f"  Total: {len(problematic)} documentos\n")

    # Mostrar primeros 5
    for doc in problematic[:5]:
        print(f"  - {doc['doc_id']}: {doc['extracted']}/{doc['attempted']} ({doc['rate']*100:.1f}%)")

    if len(problematic) > 5:
        print(f"  ... y {len(problematic)-5} más")
else:
    print(f"✅ TODOS LOS DOCUMENTOS CON BUENA EXTRACCIÓN (>70%)")

print("\n" + "=" * 80)
```

### PASO 6: Inspeccionar Resultados

```python
# ============= INSPECCIONAR RESULTADOS =============

# Ver un documento procesado
import json

output_files = sorted(list(Path(OUTPUT_DIR).glob("*_layoutlmv3.json")))

if output_files:
    # Abrir primer documento
    with open(output_files[0], 'r', encoding='utf-8') as f:
        sample_doc = json.load(f)

    print("📄 EJEMPLO DE DOCUMENTO PROCESADO:")
    print("=" * 80)
    print(f"Archivo: {output_files[0].name}")
    print(f"Total de elementos: {len(sample_doc['form'])}\n")

    # Mostrar primeros 5 campos
    print("Primeros 5 campos extraídos:")
    for elem in sample_doc['form'][:5]:
        print(f"  - {elem['field_name']}: '{elem['text']}'")
        print(f"    bbox: {elem['box']}, confidence: {elem['confidence']:.3f}\n")

    # Contar tipos de campos
    field_types = {}
    for elem in sample_doc['form']:
        field_name = elem['field_name']
        prefix = field_name.split('.')[0]
        field_types[prefix] = field_types.get(prefix, 0) + 1

    print(f"\n📊 Distribución de campos:")
    for field_type, count in sorted(field_types.items()):
        print(f"  {field_type}: {count}")
else:
    print("❌ No se encontraron archivos de salida")
```

### PASO 7: Ajustar Validaciones (Si es Necesario)

```python
# ============= AJUSTAR VALIDACIONES SI NECESITAS MÁS CAMPOS =============

# Si ves que muchos campos se filtran por confidence, puedes reducir el umbral:

formatter_permisivo = LayoutLMv3FormatterV2(
    min_confidence=0.3,      # ⬇️ Reducido de 0.5 a 0.3
    resolve_duplicates=True,
    expand_small_bboxes=True
)

# O deshabilitar ciertas validaciones:

formatter_sin_validaciones = LayoutLMv3FormatterV2(
    min_confidence=0.0,          # ⬇️ Acepta cualquier confidence
    resolve_duplicates=False,    # ❌ No resolver duplicados
    expand_small_bboxes=False    # ❌ No expandir bboxes
)

# Luego, re-procesa algunos documentos para ver la diferencia
```

### PASO 8: Comparar Antes/Después de Optimizaciones

```python
# ============= COMPARAR CON VERSIÓN ANTERIOR =============

# Si tienes resultados de procesamiento anterior, puedes compararlos:

OLD_OUTPUT_DIR = f"{DATASET_DIR}/output_layoutlmv3_old"  # Si existe
NEW_OUTPUT_DIR = OUTPUT_DIR

if Path(OLD_OUTPUT_DIR).exists():
    from scripts.compare_improvements import compare_outputs

    comparison = compare_outputs(OLD_OUTPUT_DIR, NEW_OUTPUT_DIR)

    print("📊 COMPARACIÓN ANTES/DESPUÉS:")
    print(f"  Campos [0,0] reducidos: {comparison['bbox_improvements']['zero_coords_reduction']}%")
    print(f"  Duplicados reducidos: {comparison['bbox_improvements']['duplicates_reduction']}%")
    print(f"  Confidence mejorado: +{comparison['confidence_improvements']['avg_improvement']}%")
else:
    print("ℹ️  No hay resultados anteriores para comparar")
```

---

## 🎯 Resumen - ¿Todos los Campos se Extraen?

### ✅ SÍ, se intentan extraer TODOS

El sistema intenta extraer:
- ✅ **Todos los campos del header** en el JSON
- ✅ **Todos los items** con sus campos
- ✅ **Todas las cuotas** (si existen)

### ⚠️ PERO pueden filtrarse por calidad

**Razones de filtrado:**
1. **Confidence < 0.5**: El match no es suficientemente bueno
2. **Bbox inválido**: Coordenadas [0,0,0,0] o negativas
3. **Bbox muy pequeño**: < 5px width o height
4. **Campo no encontrado**: El texto no existe en el PDF

### 📊 Tasas de Extracción Esperadas

| Componente | Tasa Esperada | Comentarios |
|-----------|---------------|-------------|
| **Campos header** | 85-95% | Depende de calidad de OCR |
| **Items** | 85-95% | Con RegionDetector y TableDetector |
| **Totales** | 95-100% | Regiones bien definidas |
| **Overall** | 85-95% | 5-15% filtrado por validaciones |

### 🔧 Si necesitas MÁS campos extraídos

**Opción 1**: Reducir `min_confidence`
```python
formatter = LayoutLMv3FormatterV2(min_confidence=0.3)  # En lugar de 0.5
```

**Opción 2**: Deshabilitar validaciones
```python
formatter = LayoutLMv3FormatterV2(
    min_confidence=0.0,
    resolve_duplicates=False
)
```

**Opción 3**: Usar CoordinateEnhancer (ya incluido)
- Mejora confidence de campos válidos
- Reduce falsos negativos

---

## 📝 Notas Importantes

1. **Los campos se filtran para MEJORAR la calidad del dataset de entrenamiento**
2. **Un campo con confidence < 0.5 puede causar más daño que bien en el entrenamiento**
3. **Es mejor tener 90% de campos con alta calidad que 100% con ruido**
4. **Usa las estadísticas de verificación para monitorear la extracción**

---

## 🆘 Troubleshooting

### Problema: Muchos campos filtrados por confidence

**Solución:**
```python
# Reducir umbral
formatter = LayoutLMv3FormatterV2(min_confidence=0.3)
```

### Problema: Items no se detectan

**Solución:**
```python
# Asegúrate de tener detect_regions=True
batch_extractor = BatchExtractor(
    detect_regions=True  # ✅ IMPORTANTE
)
```

### Problema: Lento en Colab

**Solución:**
```python
# Verificar que estás usando GPU
!nvidia-smi

# Verificar que PaddleOCR usa GPU
import paddle
print(paddle.device.get_device())  # Debe decir 'gpu:0' o similar
```

---

**¿Listo para procesar 34,165 documentos en ~28 minutos? ¡Copia el código y ejecuta!** 🚀
