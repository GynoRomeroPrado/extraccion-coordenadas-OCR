# 📘 Guía Completa: Enriquecer JSONs con Coordenadas en Google Colab

## 🎯 Objetivo

Tomar tus JSONs de anotaciones (que tienen valores pero NO tienen coordenadas bbox) y enriquecerlos automáticamente con coordenadas usando **PaddleOCR**.

**Entrada:**
```json
{
  "tipo_documento": "FACTURA ELECTRONICA",
  "serie_completa": "F003-00015692",
  "emisor_ruc": "20137291313",
  "subtotal": 13350.64
}
```

**Salida:**
```json
{
  "_metadata": {
    "original_file": "factura_001.pdf",
    "ocr_engine": "PaddleOCR",
    "match_rate": 0.92
  },
  "tipo_documento": {
    "text": "FACTURA ELECTRONICA",
    "bbox": [94, 169, 350, 186],
    "confidence": 0.95
  },
  "serie_completa": {
    "text": "F003-00015692",
    "bbox": [240, 200, 380, 220],
    "confidence": 0.98
  },
  "emisor_ruc": {
    "text": "20137291313",
    "bbox": [150, 250, 280, 270],
    "confidence": 0.99
  },
  "subtotal": {
    "text": "13350.64",
    "bbox": [700, 800, 800, 820],
    "confidence": 0.97
  }
}
```

---

## 🚀 Uso en Google Colab - GUÍA PASO A PASO

### PASO 1: Configuración Inicial

```python
# ============= HABILITAR GPU =============
# IMPORTANTE: Runtime > Change runtime type > GPU > T4

# Verificar GPU
!nvidia-smi

# ============= INSTALAR DEPENDENCIAS =============
# Instalar PaddleOCR con GPU
!pip install -q paddlepaddle-gpu paddleocr
!pip install -q rapidfuzz pymupdf pillow numpy tqdm

print("✅ Dependencias instaladas")

# ============= MONTAR GOOGLE DRIVE =============
from google.colab import drive
drive.mount('/content/drive')

# ============= CLONAR REPOSITORIO =============
%cd /content

# Si es primera vez:
!git clone https://github.com/GynoRomeroPrado/extraccion-coordenadas-OCR.git

# Si ya existe, actualizar:
%cd extraccion-coordenadas-OCR
!git fetch origin
!git checkout claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF
!git pull origin claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF

print("✅ Repositorio configurado")
```

### PASO 2: Configurar Rutas de tu Dataset

```python
# ============= CONFIGURAR RUTAS =============

# ⚠️ MODIFICA ESTAS RUTAS SEGÚN TU ESTRUCTURA
BASE_DIR = "/content/drive/MyDrive/entrenamiento_131125/modificaciones"

PDF_DIR = f"{BASE_DIR}/augmented_pdf"      # Tus PDFs
JSON_DIR = f"{BASE_DIR}/augmented_json"    # Tus JSONs sin coordenadas
OUTPUT_DIR = f"{BASE_DIR}/enriched_json"   # Salida con coordenadas

# Verificar que existen
import os
assert os.path.exists(PDF_DIR), f"❌ No existe: {PDF_DIR}"
assert os.path.exists(JSON_DIR), f"❌ No existe: {JSON_DIR}"

# Contar archivos
from pathlib import Path
pdf_count = len(list(Path(PDF_DIR).glob("*.pdf")))
json_count = len(list(Path(JSON_DIR).glob("*.json")))

print(f"✅ Encontrados {pdf_count} PDFs")
print(f"✅ Encontrados {json_count} JSONs")
print(f"📁 Output: {OUTPUT_DIR}")
```

### PASO 3: Ejecutar Enriquecimiento (PRUEBA con 10 documentos)

```python
# ============= PRUEBA CON 10 DOCUMENTOS =============

%cd /content/extraccion-coordenadas-OCR

!python scripts/enrich_jsons.py \
    --pdf-dir "{PDF_DIR}" \
    --json-dir "{JSON_DIR}" \
    --output-dir "{OUTPUT_DIR}/test" \
    --batch-size 10 \
    --use-gpu \
    --dpi 300 \
    --match-threshold 85 \
    --min-confidence 0.5

print("\n✅ Prueba completada. Revisa los resultados en {OUTPUT_DIR}/test/")
```

### PASO 4: Revisar Resultados de la Prueba

```python
# ============= INSPECCIONAR RESULTADOS =============

import json

# Ver primer archivo enriquecido
test_output = Path(f"{OUTPUT_DIR}/test")
enriched_files = sorted(list(test_output.glob("*_enriched.json")))

if enriched_files:
    print(f"📊 Archivos enriquecidos: {len(enriched_files)}\n")

    # Abrir primer archivo
    with open(enriched_files[0], 'r', encoding='utf-8') as f:
        sample = json.load(f)

    # Mostrar metadata
    print("📋 METADATA:")
    metadata = sample.get('_metadata', {})
    print(f"  OCR Engine: {metadata.get('ocr_engine')}")
    print(f"  Palabras detectadas: {metadata.get('total_words_detected')}")
    print(f"  Campos totales: {metadata.get('fields_total')}")
    print(f"  Campos con coordenadas: {metadata.get('fields_matched')}")
    print(f"  Match rate: {metadata.get('match_rate') * 100:.1f}%")

    # Mostrar primeros 5 campos
    print("\n📝 PRIMEROS 5 CAMPOS:")
    count = 0
    for key, value in sample.items():
        if key == '_metadata' or not isinstance(value, dict):
            continue

        if count >= 5:
            break

        print(f"\n  {key}:")
        print(f"    text: {value.get('text')}")
        print(f"    bbox: {value.get('bbox')}")
        print(f"    confidence: {value.get('confidence')}")

        if value.get('not_found'):
            print(f"    ⚠️  NO ENCONTRADO")

        count += 1

    # Ver items si existen
    if 'items' in sample and sample['items']:
        print(f"\n📦 ITEMS: {len(sample['items'])} encontrados")
        first_item = sample['items'][0]
        print("\n  Primer item:")
        for key, value in first_item.items():
            if isinstance(value, dict):
                print(f"    {key}: {value.get('text')} (bbox: {value.get('bbox')}, conf: {value.get('confidence')})")
else:
    print("❌ No se generaron archivos enriquecidos")
```

### PASO 5: Ajustar Parámetros (Si es Necesario)

```python
# ============= SI EL MATCH RATE ES BAJO (<80%) =============

# OPCIÓN 1: Reducir umbral de matching (más permisivo)
# Default: 85, prueba con 75 o 70

!python scripts/enrich_jsons.py \
    --pdf-dir "{PDF_DIR}" \
    --json-dir "{JSON_DIR}" \
    --output-dir "{OUTPUT_DIR}/test_low_threshold" \
    --batch-size 10 \
    --use-gpu \
    --match-threshold 70 \
    --min-confidence 0.4

# OPCIÓN 2: Reducir confidence mínimo
# Default: 0.5, prueba con 0.3

!python scripts/enrich_jsons.py \
    --pdf-dir "{PDF_DIR}" \
    --json-dir "{JSON_DIR}" \
    --output-dir "{OUTPUT_DIR}/test_low_conf" \
    --batch-size 10 \
    --use-gpu \
    --match-threshold 85 \
    --min-confidence 0.3

# Luego compara los resultados y elige los mejores parámetros
```

### PASO 6: Procesar TODOS los Documentos (5,368)

```python
# ============= PROCESAMIENTO COMPLETO =============
# ⚠️ ESTO TOMARÁ ~2-4 HORAS con GPU para 5,368 documentos

import time

%cd /content/extraccion-coordenadas-OCR

print("🚀 INICIANDO PROCESAMIENTO COMPLETO")
print(f"📊 Total: {json_count} documentos")
print(f"⏱️  Tiempo estimado: {(json_count * 3) / 3600:.1f} horas")
print("\n" + "=" * 80 + "\n")

start_time = time.time()

# Ejecutar sin --batch-size para procesar todos
!python scripts/enrich_jsons.py \
    --pdf-dir "{PDF_DIR}" \
    --json-dir "{JSON_DIR}" \
    --output-dir "{OUTPUT_DIR}" \
    --use-gpu \
    --dpi 300 \
    --match-threshold 85 \
    --min-confidence 0.5

elapsed = time.time() - start_time

print("\n" + "=" * 80)
print(f"✅ COMPLETADO en {elapsed / 60:.2f} minutos")
print("=" * 80)
```

### PASO 7: Verificar Calidad Final

```python
# ============= ESTADÍSTICAS FINALES =============

import json
from pathlib import Path

output_path = Path(OUTPUT_DIR)
enriched_files = list(output_path.glob("*_enriched.json"))

print("📊 ESTADÍSTICAS FINALES")
print("=" * 80)
print(f"Archivos enriquecidos: {len(enriched_files)}")

# Analizar todos los archivos
total_fields = 0
matched_fields = 0
total_items = 0
matched_items = 0
total_words = 0
low_match_docs = []

for file_path in enriched_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('_metadata', {})

    fields_total = metadata.get('fields_total', 0)
    fields_matched = metadata.get('fields_matched', 0)
    match_rate = metadata.get('match_rate', 0)

    total_fields += fields_total
    matched_fields += fields_matched
    total_items += metadata.get('items_total', 0)
    matched_items += metadata.get('items_matched', 0)
    total_words += metadata.get('total_words_detected', 0)

    # Detectar documentos con match rate bajo
    if match_rate < 0.7:
        low_match_docs.append({
            'file': file_path.name,
            'match_rate': match_rate,
            'fields_matched': fields_matched,
            'fields_total': fields_total
        })

overall_match_rate = matched_fields / total_fields if total_fields > 0 else 0

print(f"\n📋 CAMPOS:")
print(f"  Total: {total_fields:,}")
print(f"  Con coordenadas: {matched_fields:,} ({overall_match_rate * 100:.1f}%)")
print(f"  Sin coordenadas: {total_fields - matched_fields:,}")

print(f"\n📦 ITEMS:")
print(f"  Total: {total_items:,}")
print(f"  Con coordenadas: {matched_items:,}")

print(f"\n🔤 PALABRAS:")
print(f"  Total extraídas: {total_words:,}")
print(f"  Promedio por doc: {total_words // len(enriched_files) if enriched_files else 0}")

if low_match_docs:
    print(f"\n⚠️  DOCUMENTOS CON MATCH RATE <70%: {len(low_match_docs)}")
    print("\nPrimeros 5:")
    for doc in low_match_docs[:5]:
        print(f"  - {doc['file']}: {doc['match_rate'] * 100:.1f}% ({doc['fields_matched']}/{doc['fields_total']})")
else:
    print(f"\n✅ TODOS LOS DOCUMENTOS CON MATCH RATE >70%")

print("\n" + "=" * 80)
```

### PASO 8: Descargar o Usar los Resultados

```python
# ============= OPCIÓN 1: Mantener en Drive =============
# Los archivos ya están en tu Drive en OUTPUT_DIR
print(f"✅ Archivos disponibles en: {OUTPUT_DIR}")

# ============= OPCIÓN 2: Descargar ZIP =============
# Crear ZIP para descarga
import shutil

zip_path = f"{OUTPUT_DIR}_enriched_jsons"
shutil.make_archive(zip_path, 'zip', OUTPUT_DIR)

print(f"✅ ZIP creado: {zip_path}.zip")
print(f"📦 Tamaño: {Path(zip_path + '.zip').stat().st_size / (1024*1024):.2f} MB")

# Para descargar (descomentar si quieres):
# from google.colab import files
# files.download(f"{zip_path}.zip")

# ============= OPCIÓN 3: Usar directamente para entrenamiento =============
# Los JSONs enriquecidos ya tienen el formato necesario para LayoutLMv3
# Puedes usarlos directamente en tu pipeline de entrenamiento
```

---

## 📊 Parámetros del Script

### Parámetros Principales

| Parámetro | Default | Descripción | Cuándo Ajustar |
|-----------|---------|-------------|----------------|
| `--pdf-dir` | - | Directorio con PDFs | Requerido |
| `--json-dir` | - | Directorio con JSONs originales | Requerido |
| `--output-dir` | - | Directorio de salida | Requerido |
| `--batch-size` | None | Procesar solo N documentos | Para testing |
| `--use-gpu` | False | Usar GPU | Siempre en Colab |
| `--dpi` | 300 | Resolución de imagen | Si PDFs muy grandes: 200 |
| `--match-threshold` | 85 | Umbral fuzzy matching (0-100) | Si match rate <80%: probar 70-75 |
| `--min-confidence` | 0.5 | Confidence mínimo para match | Si match rate <80%: probar 0.3-0.4 |

### Ejemplos de Uso

**Prueba rápida (10 docs):**
```bash
python scripts/enrich_jsons.py \
    --pdf-dir "pdfs/" \
    --json-dir "jsons/" \
    --output-dir "output/" \
    --batch-size 10 \
    --use-gpu
```

**Producción (todos los docs, más permisivo):**
```bash
python scripts/enrich_jsons.py \
    --pdf-dir "pdfs/" \
    --json-dir "jsons/" \
    --output-dir "output/" \
    --use-gpu \
    --match-threshold 75 \
    --min-confidence 0.4
```

**Alta calidad (más estricto):**
```bash
python scripts/enrich_jsons.py \
    --pdf-dir "pdfs/" \
    --json-dir "jsons/" \
    --output-dir "output/" \
    --use-gpu \
    --match-threshold 90 \
    --min-confidence 0.7
```

---

## ⏱️ Tiempos Estimados

| Documentos | Con GPU T4 | Con CPU | Notas |
|------------|------------|---------|-------|
| 10 docs | ~30 segundos | ~5 minutos | Prueba |
| 100 docs | ~5 minutos | ~50 minutos | Testing |
| 1,000 docs | ~50 minutos | ~8 horas | - |
| 5,368 docs | **~4.5 horas** | **~45 horas** | Dataset completo |

**Velocidad promedio con GPU:** ~3 segundos por documento
**Velocidad promedio con CPU:** ~30 segundos por documento

---

## ❓ Troubleshooting

### Problema: Match rate muy bajo (<70%)

**Soluciones:**

1. **Reducir umbral de matching:**
   ```python
   --match-threshold 70  # En lugar de 85
   ```

2. **Reducir confidence mínimo:**
   ```python
   --min-confidence 0.3  # En lugar de 0.5
   ```

3. **Verificar calidad de PDFs:**
   - ¿Los PDFs son imágenes escaneadas o nativos?
   - ¿La resolución es suficiente?
   - ¿El texto está en español?

4. **Aumentar DPI (para PDFs de baja calidad):**
   ```python
   --dpi 400  # En lugar de 300
   ```

### Problema: "Out of memory" en GPU

**Soluciones:**

1. **Reducir DPI:**
   ```python
   --dpi 200  # En lugar de 300
   ```

2. **Procesar en lotes más pequeños:**
   ```python
   # Procesar 1000 a la vez
   --batch-size 1000
   ```

3. **Usar CPU en lugar de GPU:**
   ```python
   # Omitir --use-gpu
   # (Será más lento pero funcionará)
   ```

### Problema: Algunos campos nunca se encuentran

**Causas comunes:**

1. **El texto no está en el PDF** - Verificar PDF manualmente
2. **Formato diferente** - Ej: JSON dice "20137291313" pero PDF muestra "RUC: 20137291313"
3. **OCR no lo detecta** - Texto muy pequeño o baja calidad

**Solución:**
```python
# Ver qué palabras detectó el OCR para un documento específico
# (Agregar debug al script si es necesario)
```

### Problema: Script se detiene a mitad de proceso

**Soluciones:**

1. **Guardar progreso:** El script procesa de uno en uno, puedes reiniciar
2. **Revisar logs:** Ver qué documento causó el error
3. **Saltar documentos problemáticos:** Procesar por lotes

---

## 📝 Formato del JSON Enriquecido

### Estructura completa:

```json
{
  "_metadata": {
    "original_file": "factura_0001_aug_01.pdf",
    "processing_date": "2025-01-14T10:30:00",
    "ocr_engine": "PaddleOCR",
    "total_words_detected": 245,
    "fields_matched": 92,
    "fields_total": 98,
    "items_matched": 12,
    "items_total": 12,
    "match_rate": 0.939
  },

  "tipo_documento": {
    "text": "FACTURA ELECTRONICA",
    "bbox": [94, 169, 350, 186],
    "confidence": 0.95
  },

  "serie_completa": {
    "text": "F003-00015692",
    "bbox": [240, 200, 380, 220],
    "confidence": 0.98
  },

  "emisor_ruc": {
    "text": "20137291313",
    "bbox": [150, 250, 280, 270],
    "confidence": 0.99
  },

  "subtotal": {
    "text": "13350.64",
    "bbox": [700, 800, 800, 820],
    "confidence": 0.97
  },

  "campo_no_encontrado": {
    "text": "valor original",
    "bbox": null,
    "confidence": 0.0,
    "not_found": true
  },

  "items": [
    {
      "descripcion": {
        "text": "Servicio de Alimentación",
        "bbox": [100, 400, 350, 420],
        "confidence": 0.92
      },
      "cantidad": {
        "text": "1.0",
        "bbox": [360, 400, 390, 420],
        "confidence": 0.98
      },
      "precio_unitario": {
        "text": "15753.7552",
        "bbox": [400, 400, 480, 420],
        "confidence": 0.96
      }
    }
  ]
}
```

### Campos de metadata:

- `original_file`: Nombre del PDF procesado
- `processing_date`: Fecha/hora de procesamiento
- `ocr_engine`: Siempre "PaddleOCR"
- `total_words_detected`: Palabras extraídas del PDF
- `fields_matched`: Campos con coordenadas encontradas
- `fields_total`: Total de campos en el JSON
- `items_matched`: Items con coordenadas
- `items_total`: Total de items
- `match_rate`: Tasa de éxito (0.0-1.0)

### Campos enriquecidos:

- `text`: Valor original del campo
- `bbox`: `[x0, y0, x1, y1]` normalizado a escala 0-1000
- `confidence`: Confidence del OCR (0.0-1.0)
- `not_found`: (opcional) True si no se encontró el campo

---

## ✅ Checklist de Verificación

Antes de procesar todo:

- [ ] GPU habilitada en Colab
- [ ] PaddleOCR instalado con GPU (`paddlepaddle-gpu`)
- [ ] Rutas de PDF y JSON correctas
- [ ] Prueba con 10 documentos completada
- [ ] Match rate de prueba >80%
- [ ] Resultados revisados manualmente

Durante procesamiento:

- [ ] Monitor de GPU activo (`nvidia-smi`)
- [ ] No hay errores en logs
- [ ] Match rate general >80%

Después de procesar:

- [ ] Total de archivos enriquecidos correcto
- [ ] Metadata en todos los JSONs
- [ ] Coordenadas presentes en >80% de campos
- [ ] Items enriquecidos correctamente

---

## 🎉 ¡Listo para Entrenar!

Una vez enriquecidos los JSONs, puedes usarlos directamente para entrenar LayoutLMv3. Las coordenadas bbox están en el formato correcto (0-1000 normalizado).

**Siguiente paso:** Convertir estos JSONs enriquecidos al formato FUNSD para LayoutLMv3 (si es necesario).

---

¿Preguntas? Revisa los logs del script o ajusta los parámetros según tu caso de uso.
