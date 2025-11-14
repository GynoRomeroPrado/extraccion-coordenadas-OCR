# 🚀 GUÍA COMPLETA DE EJECUCIÓN EN GOOGLE COLAB

## 📋 Resumen Ejecutivo

Esta guía te llevará desde el setup inicial hasta tener un dataset final de **5,368 facturas enriquecidas con coordenadas bbox** listo para entrenar LayoutLMv3.

**Tiempo total estimado:** ~5-6 horas
**Resultado:** Dataset en formato LayoutLMv3 con 81.8% match rate

---

## 📊 Flujo Completo

```
PASO 1: Setup Inicial (10 min)
   ↓
PASO 2: Prueba con 5 facturas (30 seg)
   ↓
PASO 3: Procesar 5,368 facturas (4-5 horas)
   ↓
PASO 4: Reorganizar a estructura final (10 min)
   ↓
PASO 5: Validar calidad del dataset (5 min)
   ↓
✅ Dataset listo para Fase 4 (Entrenam

iento LayoutLMv3)
```

---

## 🔧 PASO 1: Setup Inicial Completo

### 1.1 Habilitar GPU

**⚠️ IMPORTANTE:** Antes de empezar
1. `Runtime > Change runtime type`
2. Seleccionar `GPU`
3. Hardware accelerator: `T4 GPU`
4. Click `Save`

### 1.2 Ejecutar Setup

```python
# ============================================================================
# PASO 1.2: SETUP INICIAL
# ============================================================================

print("🚀 INICIANDO SETUP COMPLETO\n")

# 1. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Instalar dependencias
print("\n📦 INSTALANDO DEPENDENCIAS...\n")

!pip install --upgrade --force-reinstall numpy==1.26.4 -q
!pip install --upgrade --no-deps paddlepaddle-gpu==2.6.1 -q
!pip install --upgrade --no-deps paddleocr==2.7.3 -q
!pip install opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66 -q
!pip install rapidfuzz pymupdf pillow tqdm scikit-image -q

print("✅ Dependencias instaladas")
print("\n⚠️  IMPORTANTE: REINICIAR RUNTIME AHORA")
print("   Runtime > Restart runtime")
print("   Luego continuar con PASO 1.3")
```

### 1.3 Después del Reinicio

**⚠️ REINICIAR EL RUNTIME y ejecutar esto:**

```python
# ============================================================================
# PASO 1.3: CONFIGURACIÓN POST-REINICIO
# ============================================================================

print("🔄 CONFIGURACIÓN POST-REINICIO\n")

# 1. Remontar Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Verificar NumPy
import numpy as np
print(f"NumPy: {np.__version__}")

if not np.__version__.startswith('1.26'):
    print("❌ ERROR: NumPy incorrecto")
    print("   Volver al PASO 1.2 y reinstalar")
    import sys
    sys.exit(1)

print("✅ NumPy correcto")

# 3. Clonar repositorio
import os
import shutil

REPO_DIR = "/content/extraccion-coordenadas-OCR"

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

!git clone -q https://github.com/GynoRomeroPrado/extraccion-coordenadas-OCR.git {REPO_DIR}

os.chdir(REPO_DIR)
!git checkout -q claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF

print(f"✅ Repositorio: {os.getcwd()}")

# 4. Definir variables globales
BASE_DIR = "/content/drive/MyDrive/entrenamiento_131125"
AUGMENTED_PDF_DIR = f"{BASE_DIR}/modificaciones/augmented_pdf"
AUGMENTED_JSON_DIR = f"{BASE_DIR}/modificaciones/augmented_json"
TEMPORAL_DIR = f"{BASE_DIR}/temporal_pruebas"
REPO_DIR = "/content/extraccion-coordenadas-OCR"

import sys
sys.path.insert(0, REPO_DIR)

print("✅ Variables globales definidas")

# 5. Inicializar PaddleOCR
from paddleocr import PaddleOCR
import paddle

print("\n🚀 Inicializando PaddleOCR (esto toma 1-2 min)...\n")

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='es',
    use_gpu=True,
    show_log=False,
    det_db_box_thresh=0.5,
    rec_batch_num=6
)

print("✅ PaddleOCR inicializado")
print(f"✅ GPU detectada: {paddle.device.is_compiled_with_cuda()}")
print(f"✅ Dispositivo: {paddle.device.get_device()}")

# 6. Verificar dataset
pdfs = len([f for f in os.listdir(AUGMENTED_PDF_DIR) if f.endswith('.pdf')])
jsons = len([f for f in os.listdir(AUGMENTED_JSON_DIR) if f.endswith('.json')])

print(f"\n📊 Dataset: {pdfs:,} PDFs | {jsons:,} JSONs")

# 7. Verificación final
print("\n" + "=" * 70)
print("✅ VERIFICACIÓN DEL SISTEMA:")
print("=" * 70)
print(f"\n1. NumPy: {np.__version__} ✅")
print(f"2. PaddleOCR: ✅")
print(f"3. GPU: ✅")
print(f"4. Repositorio: ✅")
print(f"5. Dataset: {pdfs:,} facturas ✅")

print("\n" + "=" * 70)
print("🎉 SETUP COMPLETO - Listo para PASO 2")
print("=" * 70)
```

---

## 🧪 PASO 2: Prueba con 5 Facturas (RECOMENDADO)

```python
# ============================================================================
# PASO 2: PRUEBA CON 5 FACTURAS ALEATORIAS
# Tiempo estimado: 30 segundos
# ============================================================================

%run scripts/test_enrichment.py
```

**¿Qué hace este script?**
1. Selecciona 5 facturas aleatorias
2. Las procesa con PaddleOCR
3. Aplica el algoritmo de matching mejorado
4. Muestra estadísticas detalladas
5. Te dice si puedes proceder con las 5,368

**Resultado esperado:**
```
✅ EXCELENTE! Match rate >85%
   👉 Puedes proceder con las 5,368 facturas completas
```

**Si el match rate es bajo (<70%):**
- Ajustar `--match-threshold` a 70-75
- Ajustar `--min-confidence` a 0.3-0.4
- Repetir la prueba

---

## 🚀 PASO 3: Procesar Dataset Completo (5,368 Facturas)

**⚠️ SOLO ejecutar si PASO 2 fue exitoso (match rate >70%)**

```python
# ============================================================================
# PASO 3: PROCESAMIENTO COMPLETO
# Tiempo estimado: 4-5 horas
# ============================================================================

%run scripts/process_full_dataset.py
```

**Confirmaciones que pedirá:**

1. **Primera confirmación:**
   ```
   ¿Proceder con el procesamiento completo? (s/n):
   ```
   → Teclear `s` y Enter

2. **Ajuste de parámetros:**
   ```
   ¿Ajustar parámetros? (s/n):
   ```
   - Si la prueba fue >85% → teclear `n`
   - Si quieres ajustar → teclear `s`

**Durante las 4-5 horas verás:**
```
Enriqueciendo JSONs:  45%|████▌     | 2415/5368 [1:23:45<1:42:15]

[1234/5368] factura_0001_aug_01.pdf | Header: 92/98 | Items: 12/12 | 3.2 docs/s
```

**⚠️ Si Colab se desconecta:**
1. Reconectar
2. Ejecutar PASO 1.3 (post-reinicio)
3. Ejecutar PASO 3 nuevamente
4. El script detectará archivos ya procesados y los saltará

---

## 📁 PASO 4: Reorganizar a Estructura Final

**Después de completar PASO 3:**

```python
# ============================================================================
# PASO 4: REORGANIZACIÓN A ESTRUCTURA FINAL
# Tiempo estimado: 10 minutos
# ============================================================================

%run scripts/reorganize_to_final_structure.py
```

**¿Qué hace este script?**

1. Crea estructura final en `/content/drive/MyDrive/dataset_141125/`
2. Copia los 5,368 PDFs
3. Copia los 5,368 JSONs enriquecidos
4. Valida formato LayoutLMv3
5. Genera README.md con documentación
6. Guarda estadísticas completas

**Resultado:**
```
dataset_141125/
├── facturas_pdf/      # 5,368 PDFs
├── facturas_json/     # 5,368 JSONs formato LayoutLMv3
├── metadata/          # Archivos de metadata
├── statistics/        # Estadísticas del dataset
└── README.md          # Documentación
```

---

## 📊 PASO 5: Validar Calidad del Dataset

```python
# ============================================================================
# PASO 5: VALIDACIÓN DE CALIDAD
# Tiempo estimado: 5 minutos
# ============================================================================

import json
import random
from pathlib import Path

# Rutas
dataset_dir = "/content/drive/MyDrive/dataset_141125"
json_dir = f"{dataset_dir}/facturas_json"

print("📊 VALIDACIÓN DE CALIDAD DEL DATASET")
print("=" * 70)

# 1. Contar archivos
pdf_files = list(Path(f"{dataset_dir}/facturas_pdf").glob("*.pdf"))
json_files = list(Path(json_dir).glob("*.json"))

print(f"\n📁 ARCHIVOS:")
print(f"   PDFs: {len(pdf_files):,}")
print(f"   JSONs: {len(json_files):,}")

if len(pdf_files) != len(json_files):
    print(f"\n⚠️  ADVERTENCIA: Número de PDFs y JSONs no coincide")

# 2. Analizar muestra de 100 archivos
print(f"\n📊 Analizando muestra de 100 archivos...")

sample = random.sample(json_files, min(100, len(json_files)))

total_fields = 0
matched_fields = 0
total_items = 0
total_words = 0
match_rates = []
low_quality = []
format_errors = 0

for file in sample:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Verificar metadata
        if '_metadata' not in data:
            format_errors += 1
            continue

        meta = data.get('_metadata', {})

        total_fields += meta.get('fields_total', 0)
        matched_fields += meta.get('fields_matched', 0)
        total_items += meta.get('items_total', 0)
        total_words += meta.get('total_words_detected', 0)

        match_rate = meta.get('match_rate', 0)
        match_rates.append(match_rate)

        if match_rate < 0.7:
            low_quality.append({
                'file': file.name,
                'rate': match_rate
            })

        # Verificar formato bbox en al menos un campo
        bbox_found = False
        for key, value in data.items():
            if key == '_metadata':
                continue
            if isinstance(value, dict) and 'bbox' in value:
                bbox = value['bbox']
                if bbox and isinstance(bbox, list) and len(bbox) == 4:
                    bbox_found = True
                    # Verificar formato LayoutLMv3 [x_min, y_min, x_max, y_max]
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        format_errors += 1
                    break

    except Exception as e:
        format_errors += 1
        print(f"   ⚠️  Error en {file.name}: {e}")

# 3. Calcular métricas
avg_match_rate = sum(match_rates) / len(match_rates) if match_rates else 0

print(f"\n📋 CAMPOS:")
print(f"   Total: {total_fields:,}")
print(f"   ✅ Con coordenadas: {matched_fields:,} ({matched_fields/total_fields*100:.1f}%)")
print(f"   ❌ Sin coordenadas: {total_fields - matched_fields:,}")

print(f"\n📦 ITEMS:")
print(f"   Total: {total_items:,}")

print(f"\n🔤 PALABRAS OCR:")
print(f"   Total extraídas: {total_words:,}")
print(f"   Promedio por doc: {total_words // len(sample)}")

print(f"\n📊 CALIDAD:")
print(f"   Match rate promedio: {avg_match_rate * 100:.1f}%")
print(f"   Docs con rate <70%: {len(low_quality)} ({len(low_quality)/len(sample)*100:.1f}%)")
print(f"   Errores de formato: {format_errors}")

print("\n" + "=" * 70)
if avg_match_rate >= 0.85 and format_errors < 5:
    print("🎉 EXCELENTE! Dataset de alta calidad (>85%)")
    print("\n✅ LISTO PARA FASE 4: Entrenamiento de LayoutLMv3")
elif avg_match_rate >= 0.70 and format_errors < 10:
    print("✅ BUENO. Dataset utilizable (70-85%)")
    print("\n👉 Puedes proceder a Fase 4")
else:
    print("⚠️ BAJO. Match rate <70% o muchos errores de formato")
    print("\n👉 Considera re-procesar con parámetros ajustados")
print("=" * 70)

# 4. Mostrar ejemplo de documento
print("\n📄 EJEMPLO DE DOCUMENTO ENRIQUECIDO:")
print("=" * 70)

with open(json_files[0], 'r', encoding='utf-8') as f:
    example = json.load(f)

print(f"\nArchivo: {json_files[0].name}")

if '_metadata' in example:
    meta = example['_metadata']
    print(f"Match rate: {meta.get('match_rate', 0) * 100:.1f}%")
    print(f"Formato bbox: {meta.get('bbox_format', 'unknown')}")
    print(f"OCR engine: {meta.get('ocr_engine', 'unknown')}")

print("\nPrimeros 3 campos:")
count = 0
for key, value in example.items():
    if key != '_metadata' and isinstance(value, dict) and count < 3:
        print(f"\n{count+1}. {key}:")
        print(f"   text: {value.get('text')}")
        print(f"   bbox: {value.get('bbox')}")
        print(f"   confidence: {value.get('confidence', 0):.3f}")
        count += 1

print("\n" + "=" * 70)
print("✅ VALIDACIÓN COMPLETADA")
print("=" * 70)

print(f"\n📁 Dataset final en: {dataset_dir}")
print(f"📄 Ver documentación: {dataset_dir}/README.md")
```

---

## 📊 Tiempos Estimados por Paso

| Paso | Tiempo | Descripción |
|------|--------|-------------|
| **1.2** | ~5 min | Instalar dependencias |
| **Reiniciar** | ~30 seg | Reiniciar runtime |
| **1.3** | ~3 min | Configurar post-reinicio |
| **2** | ~30 seg | Prueba con 5 facturas |
| **3** | **4-5 horas** | Procesar 5,368 facturas |
| **4** | ~10 min | Reorganizar a estructura final |
| **5** | ~5 min | Validar calidad |
| **TOTAL** | **~5-6 horas** | |

---

## 🎯 Resultado Final

### Estructura de Carpetas

```
/content/drive/MyDrive/
├── entrenamiento_131125/              # Datos intermedios (NO TOCAR)
│   ├── facturas_pdf/                  # 656 originales
│   ├── anotaciones_json/              # 656 JSONs sin coords
│   └── modificaciones/
│       ├── augmented_pdf/             # 5,368 PDFs augmentados
│       ├── augmented_json/            # 5,368 JSONs sin coords
│       └── enriched_json/             # 5,368 JSONs enriquecidos (backup)
│
└── dataset_141125/                    # ✨ DATASET FINAL LIMPIO
    ├── facturas_pdf/                  # 5,368 PDFs finales
    ├── facturas_json/                 # 5,368 JSONs formato LayoutLMv3
    ├── metadata/                      # Archivos de metadata
    ├── statistics/                    # Estadísticas del dataset
    └── README.md                      # Documentación
```

### Formato del JSON Final

```json
{
  "_metadata": {
    "original_file": "factura_0001_aug_01.pdf",
    "ocr_engine": "PaddleOCR",
    "match_rate": 0.818,
    "bbox_format": "layoutlmv3",
    "bbox_structure": "[x_min, y_min, x_max, y_max]"
  },
  "tipo_documento": {
    "text": "FACTURA ELECTRONICA",
    "bbox": [94, 169, 350, 186],
    "confidence": 0.95
  },
  "emisor_ruc": {
    "text": "20137291313",
    "bbox": [150, 250, 280, 270],
    "confidence": 0.99
  },
  "importe_total": {
    "text": "15753.76",
    "bbox": [700, 870, 800, 890],
    "confidence": 0.98
  },
  "items": [
    {
      "descripcion": {
        "text": "Servicio de Alimentación",
        "bbox": [100, 500, 400, 520],
        "confidence": 0.92
      },
      "cantidad": {
        "text": "1.0",
        "bbox": [450, 500, 480, 520],
        "confidence": 0.98
      }
    }
  ]
}
```

---

## 🐛 Troubleshooting

### Error: "numpy.dtype size changed"

**Solución:**
```python
!pip install --force-reinstall numpy==1.26.4
# REINICIAR RUNTIME
```

### Error: "No module named 'paddleocr'"

**Solución:**
```python
!pip install --no-deps paddleocr==2.7.3
# REINICIAR RUNTIME
```

### Match rate muy bajo (<50%)

**Solución:** Ajustar parámetros en PASO 3:
```
Match threshold: 70    # (default: 85)
Min confidence: 0.3    # (default: 0.5)
```

### Colab se desconecta durante PASO 3

**Solución:**
1. Reconectar
2. Ejecutar PASO 1.3
3. Ejecutar PASO 3 → detectará archivos ya procesados

### "Out of memory" durante procesamiento

**Solución:**
```python
# Reducir DPI
--dpi 200  # En lugar de 300
```

---

## ✅ Checklist Final

Al terminar todos los pasos:

- [ ] 5,368 PDFs en `dataset_141125/facturas_pdf/`
- [ ] 5,368 JSONs en `dataset_141125/facturas_json/`
- [ ] Match rate promedio >70% (idealmente >80%)
- [ ] Todos los JSONs tienen `_metadata`
- [ ] Bbox en formato `[x_min, y_min, x_max, y_max]`
- [ ] README.md generado
- [ ] Estadísticas guardadas
- [ ] Validación de calidad completada

---

## 🎉 Próximo Paso: Fase 4

**Entrenamiento de LayoutLMv3**

1. Convertir a formato FUNSD (si es necesario)
2. Dividir en train/val/test (80/10/10)
3. Entrenar modelo LayoutLMv3
4. Evaluar en conjunto de prueba
5. **Objetivo:** >98% precisión en detección de items prohibidos

---

## 📚 Documentación Adicional

- **`DOCUMENTACION_MEJORAS_V2.md`** - Detalles técnicos completos
- **`FASE3_EJECUCION_COLAB.md`** - Guía detallada por pasos
- **`GPU_vs_TPU_ANALYSIS.md`** - Análisis de hardware
- **`OPTIMIZATION_GUIDE.md`** - Optimizaciones adicionales

---

## 📞 Soporte

**Repositorio:** https://github.com/GynoRomeroPrado/extraccion-coordenadas-OCR
**Branch:** `claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF`
**Proyecto:** FacturasIA/InvokeX

---

**¿Listo para empezar? Copia el código del PASO 1.2 en una celda de Colab y ejecuta!** 🚀
