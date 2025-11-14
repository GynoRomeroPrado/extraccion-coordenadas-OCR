# 🚀 FASE 3: Ejecución en Google Colab - Extracción de Coordenadas

## 📋 Contexto del Proyecto

**Proyecto:** FacturasIA/InvokeX - Detección de items prohibidos en facturas peruanas
**Fase actual:** Fase 3 - Extracción de coordenadas bbox
**Dataset:** 5,368 facturas augmentadas (8x multiplicación de 656 originales)
**Objetivo:** Enriquecer JSONs con coordenadas bbox para entrenamiento de LayoutLMv3

---

## ⚠️ IMPORTANTE: SETUP INICIAL

**ANTES de ejecutar cualquier código, asegúrate de:**

1. ✅ **GPU habilitada** en Colab: `Runtime > Change runtime type > GPU > T4`
2. ✅ **NumPy 1.26.4 instalado** (NO 2.x)
3. ✅ **Runtime REINICIADO** después de instalar NumPy
4. ✅ **PaddleOCR inicializado** correctamente

Si no has hecho el setup, **EJECUTA PRIMERO el SETUP COMPLETO** (ver sección siguiente).

---

## 🔧 PASO 1: SETUP COMPLETO (Copiar y Pegar)

```python
# ============================================================================
# SETUP COMPLETO PARA FASE 3 - EXTRACCIÓN DE COORDENADAS
# Ejecutar esto SOLO UNA VEZ por sesión de Colab
# ============================================================================

print("🚀 INICIANDO SETUP COMPLETO\n")
print("=" * 70)

# 1. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Instalar dependencias con versiones específicas
print("\n📦 INSTALANDO DEPENDENCIAS...\n")

# CRÍTICO: NumPy 1.26.4 (NO 2.x)
!pip install --upgrade --force-reinstall numpy==1.26.4 -q

# PaddlePaddle y PaddleOCR (sin dependencias para evitar conflictos)
!pip install --upgrade --no-deps paddlepaddle-gpu==2.6.1 -q
!pip install --upgrade --no-deps paddleocr==2.7.3 -q

# OpenCV versiones específicas
!pip install opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66 -q

# Otras dependencias
!pip install rapidfuzz pymupdf pillow tqdm scikit-image -q

print("✅ Dependencias instaladas")
print("\n⚠️  IMPORTANTE: REINICIAR RUNTIME AHORA")
print("   Runtime > Restart runtime")
print("   Luego volver a ejecutar desde 'PASO 1B' abajo")
```

### PASO 1B: Después del Reinicio del Runtime

**⚠️ EJECUTAR ESTO DESPUÉS DE REINICIAR EL RUNTIME:**

```python
# ============================================================================
# PASO 1B: CONFIGURACIÓN POST-REINICIO
# ============================================================================

print("🚀 CONFIGURACIÓN POST-REINICIO\n")
print("=" * 70)

# 1. Volver a montar Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Verificar NumPy
import numpy as np
print(f"\n📊 NumPy: {np.__version__}")

if not np.__version__.startswith('1.26'):
    print("❌ ERROR: NumPy incorrecto")
    print("   Necesitas NumPy 1.26.x, tienes:", np.__version__)
    print("   Vuelve al PASO 1 y reinstala")
    import sys
    sys.exit(1)

print("✅ NumPy correcto")

# 3. Clonar repositorio
print("\n📥 CLONANDO REPOSITORIO...\n")

import os
import shutil

REPO_DIR = "/content/extraccion-coordenadas-OCR"

if os.path.exists(REPO_DIR):
    shutil.rmtree(REPO_DIR)

!git clone -q https://github.com/GynoRomeroPrado/extraccion-coordenadas-OCR.git {REPO_DIR}

os.chdir(REPO_DIR)
!git checkout -q claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF

print(f"✅ Repositorio: {os.getcwd()}")

# 4. Definir rutas globales
BASE_DIR = "/content/drive/MyDrive/entrenamiento_131125"
AUGMENTED_PDF_DIR = f"{BASE_DIR}/modificaciones/augmented_pdf"
AUGMENTED_JSON_DIR = f"{BASE_DIR}/modificaciones/augmented_json"
TEMPORAL_DIR = f"{BASE_DIR}/temporal_pruebas"

# Agregar al path
import sys
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("✅ Rutas definidas")

# 5. Inicializar PaddleOCR
print("\n🚀 INICIALIZANDO PADDLEOCR...\n")
print("   (Esto puede tardar 1-2 minutos la primera vez)\n")

try:
    from paddleocr import PaddleOCR
    import paddle

    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='es',
        use_gpu=True,
        show_log=False,
        det_db_box_thresh=0.5,
        rec_batch_num=6
    )

    print("✅ PaddleOCR LISTO")
    print(f"✅ GPU detectada: {paddle.device.is_compiled_with_cuda()}")
    print(f"✅ Dispositivo: {paddle.device.get_device()}")

except Exception as e:
    print(f"❌ ERROR inicializando PaddleOCR: {e}")
    print("\n⚠️  SOLUCIÓN: Asegúrate de haber reiniciado el runtime")
    import sys
    sys.exit(1)

# 6. Verificar dataset
pdfs = len([f for f in os.listdir(AUGMENTED_PDF_DIR) if f.endswith('.pdf')])
jsons = len([f for f in os.listdir(AUGMENTED_JSON_DIR) if f.endswith('.json')])

print(f"\n📊 Dataset encontrado: {pdfs:,} PDFs | {jsons:,} JSONs")

if pdfs != 5368 or jsons != 5368:
    print(f"\n⚠️  ADVERTENCIA: Esperábamos 5,368 archivos")
    print(f"   PDFs: {pdfs:,}")
    print(f"   JSONs: {jsons:,}")

# 7. Verificación final
print("\n" + "=" * 70)
print("✅ VERIFICACIÓN DEL SISTEMA:")
print("=" * 70)
print(f"\n1. NumPy: {np.__version__} {'✅' if np.__version__.startswith('1.26') else '❌'}")
print(f"2. PaddleOCR: ✅")
print(f"3. GPU: {'✅' if paddle.device.is_compiled_with_cuda() else '❌'}")
print(f"4. Repositorio: ✅")
print(f"5. Dataset: {pdfs:,} facturas {'✅' if pdfs == 5368 else '⚠️'}")

print("\n" + "=" * 70)
print("🎉 SETUP COMPLETO - LISTO PARA PROCESAR")
print("=" * 70)
print("\n👉 Ahora puedes ejecutar PASO 2 (Prueba) o PASO 3 (Completo)")
```

---

## 🧪 PASO 2: PRUEBA CON 5 FACTURAS (RECOMENDADO)

**⚠️ EJECUTAR PRIMERO una prueba con 5 facturas antes de procesar las 5,368**

```python
# ============================================================================
# PRUEBA CON 5 FACTURAS ALEATORIAS
# ============================================================================

# Ejecutar el script de prueba
%run scripts/test_enrichment.py
```

**¿Qué hace este script?**

1. Selecciona 5 facturas aleatorias
2. Las copia a `temporal_pruebas/`
3. Ejecuta el enriquecimiento
4. Muestra estadísticas y ejemplos
5. Te dice si puedes proceder con el dataset completo

**Tiempo estimado:** ~30 segundos

### Interpretar Resultados de la Prueba

**✅ Si Match rate >85%:**
```
✅ EXCELENTE! Match rate >85%
   👉 Puedes proceder con las 5,368 facturas completas
```
→ **Continuar al PASO 3**

**⚠️ Si Match rate 70-85%:**
```
⚠️ ACEPTABLE. Match rate entre 70-85%
   👉 Considera ajustar parámetros:
      --match-threshold 75 (en lugar de 85)
      --min-confidence 0.4 (en lugar de 0.5)
```
→ **Ajustar parámetros y repetir prueba**

**❌ Si Match rate <70%:**
```
❌ BAJO. Match rate <70%
   👉 Ajusta parámetros:
      --match-threshold 70
      --min-confidence 0.3
```
→ **Revisar PDFs y ajustar parámetros**

---

## 🚀 PASO 3: PROCESAR DATASET COMPLETO (5,368 Facturas)

**⚠️ SOLO ejecutar si la prueba del PASO 2 fue exitosa (match rate >70%)**

```python
# ============================================================================
# PROCESAMIENTO COMPLETO - 5,368 FACTURAS
# ============================================================================

# Ejecutar el script de procesamiento completo
%run scripts/process_full_dataset.py
```

**Confirmaciones que pedirá:**

1. **Primera confirmación:**
   ```
   ¿Proceder con el procesamiento completo? (s/n):
   ```
   → Teclear `s` y Enter

2. **Ajuste de parámetros (opcional):**
   ```
   ¿Ajustar parámetros? (s/n):
   ```
   - Si la prueba fue buena → teclear `n`
   - Si quieres ser más/menos estricto → teclear `s` y ajustar

**Tiempo estimado:** ~4-5 horas con GPU T4

### Durante el Procesamiento

El script mostrará:
- Progress bar en tiempo real
- Documentos procesados
- Match rate por documento
- Errores (si ocurren)

**Ejemplo del output:**
```
Enriqueciendo JSONs:  45%|████▌     | 2415/5368 [1:23:45<1:42:15, 0.48docs/s]

[1234/5368] factura_0001_aug_01.pdf | Header: 92/98 | Items: 12/12 | 3.2 docs/s | ETA: 1.2h
```

### Si Colab se Desconecta

El script procesa un documento a la vez, por lo que:
1. Puedes reiniciar
2. Volver a hacer PASO 1B (setup)
3. El script detectará archivos ya procesados y los saltará

---

## 📊 PASO 4: VERIFICAR RESULTADOS FINALES

```python
# ============================================================================
# VERIFICACIÓN DE RESULTADOS FINALES
# ============================================================================

import json
import random
from pathlib import Path

output_dir = "/content/drive/MyDrive/entrenamiento_131125/modificaciones/enriched_json"

# Contar archivos
enriched_files = list(Path(output_dir).glob("*_enriched.json"))

print("📊 ESTADÍSTICAS FINALES DEL DATASET")
print("=" * 70)
print(f"\nArchivos enriquecidos: {len(enriched_files):,}/5,368")

if len(enriched_files) < 5368:
    print(f"⚠️  Faltantes: {5368 - len(enriched_files):,}")

# Analizar muestra de 100 archivos
print("\nAnalizando muestra de 100 archivos...")

sample = random.sample(enriched_files, min(100, len(enriched_files)))

total_fields = 0
matched_fields = 0
total_items = 0
matched_items = 0
total_words = 0
match_rates = []
low_quality = []

for file in sample:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data.get('_metadata', {})

    total_fields += meta.get('fields_total', 0)
    matched_fields += meta.get('fields_matched', 0)
    total_items += meta.get('items_total', 0)
    matched_items += meta.get('items_matched', 0)
    total_words += meta.get('total_words_detected', 0)

    match_rate = meta.get('match_rate', 0)
    match_rates.append(match_rate)

    if match_rate < 0.7:
        low_quality.append({
            'file': file.name,
            'rate': match_rate
        })

avg_match_rate = sum(match_rates) / len(match_rates) if match_rates else 0

print(f"\n📋 CAMPOS:")
print(f"  Total: {total_fields:,}")
print(f"  ✅ Con coordenadas: {matched_fields:,} ({matched_fields/total_fields*100:.1f}%)")
print(f"  ❌ Sin coordenadas: {total_fields - matched_fields:,}")

print(f"\n📦 ITEMS:")
print(f"  Total: {total_items:,}")
print(f"  ✅ Con coordenadas: {matched_items:,}")

print(f"\n🔤 PALABRAS:")
print(f"  Total extraídas: {total_words:,}")
print(f"  Promedio por doc: {total_words // len(sample)}")

print(f"\n📊 CALIDAD:")
print(f"  Match rate promedio: {avg_match_rate * 100:.1f}%")
print(f"  Docs con rate <70%: {len(low_quality)} ({len(low_quality)/len(sample)*100:.1f}%)")

print("\n" + "=" * 70)
if avg_match_rate >= 0.85:
    print("🎉 EXCELENTE! Dataset de alta calidad (>85%)")
    print("\n✅ LISTO PARA FASE 4: Entrenamiento de LayoutLMv3")
elif avg_match_rate >= 0.70:
    print("✅ BUENO. Dataset utilizable (70-85%)")
    print("\n👉 Puedes proceder a Fase 4")
else:
    print("⚠️ BAJO. Match rate <70%")
    print("\n👉 Considera re-procesar con parámetros ajustados")

print("=" * 70)

# Mostrar ejemplo de un documento
print("\n📄 EJEMPLO DE DOCUMENTO ENRIQUECIDO:")
print("=" * 70)

with open(enriched_files[0], 'r', encoding='utf-8') as f:
    example = json.load(f)

print(f"\nArchivo: {enriched_files[0].name}")
print(f"Match rate: {example['_metadata']['match_rate'] * 100:.1f}%")

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
```

---

## 🎯 RESUMEN DE PASOS

### Flujo Completo

1. **PASO 1:** Setup inicial (10 minutos)
   - Instalar dependencias
   - **REINICIAR RUNTIME**
   - PASO 1B: Configuración post-reinicio

2. **PASO 2:** Prueba con 5 facturas (30 segundos)
   - Verificar que todo funciona
   - Analizar match rate

3. **PASO 3:** Procesar dataset completo (4-5 horas)
   - Procesar 5,368 facturas
   - Monitorear progreso

4. **PASO 4:** Verificar resultados (2 minutos)
   - Analizar estadísticas finales
   - Confirmar calidad del dataset

### Tiempo Total Estimado

- Setup: ~10 minutos
- Prueba: ~30 segundos
- Procesamiento: **~4-5 horas**
- Verificación: ~2 minutos

**Total: ~4.5-5.5 horas**

---

## ⚙️ AJUSTE DE PARÁMETROS

Si necesitas ajustar el match rate, modifica estos parámetros:

### Match Threshold (Fuzzy Matching)

```python
--match-threshold 85  # Default - Estricto
--match-threshold 75  # Permisivo - Acepta más matches
--match-threshold 70  # Muy permisivo
--match-threshold 90  # Muy estricto - Solo matches perfectos
```

**Cuándo reducir:** Match rate <80%
**Cuándo aumentar:** Demasiados falsos positivos

### Min Confidence (OCR)

```python
--min-confidence 0.5  # Default - Estricto
--min-confidence 0.4  # Permisivo
--min-confidence 0.3  # Muy permisivo
--min-confidence 0.7  # Muy estricto
```

**Cuándo reducir:** Match rate <80%
**Cuándo aumentar:** Demasiados errores en coordenadas

### DPI (Resolución)

```python
--dpi 300  # Default - Buena calidad
--dpi 400  # Alta calidad - Para PDFs muy pequeños
--dpi 200  # Baja calidad - Para ahorrar memoria
```

**Cuándo aumentar:** Texto muy pequeño o difícil de leer
**Cuándo reducir:** "Out of memory" errors

---

## 🐛 Troubleshooting

### Error: "numpy.dtype size changed"

**Causa:** NumPy 2.x instalado
**Solución:**
```python
!pip install --force-reinstall numpy==1.26.4
# REINICIAR RUNTIME
```

### Error: "CUDA out of memory"

**Causa:** GPU sin memoria
**Solución:**
```python
# Reducir DPI
--dpi 200  # En lugar de 300
```

### Error: "ModuleNotFoundError: No module named 'paddleocr'"

**Causa:** PaddleOCR no instalado o runtime reiniciado sin re-setup
**Solución:** Ejecutar PASO 1B completo

### Match rate muy bajo (<50%)

**Causas posibles:**
- PDFs de muy baja calidad
- Texto no en español
- Parámetros muy estrictos

**Solución:**
```python
--match-threshold 70
--min-confidence 0.3
```

### Script se detiene sin error

**Causa:** Colab se desconectó
**Solución:**
- Reconectar
- Ejecutar PASO 1B
- Ejecutar PASO 3 (detectará archivos ya procesados)

---

## 📁 Estructura de Archivos de Salida

```
/content/drive/MyDrive/entrenamiento_131125/
└── modificaciones/
    ├── augmented_pdf/              # 5,368 PDFs (input)
    ├── augmented_json/             # 5,368 JSONs sin coords (input)
    └── enriched_json/              # 5,368 JSONs con coords (OUTPUT) ✨
        ├── factura_0001_aug_01_enriched.json
        ├── factura_0001_aug_02_enriched.json
        ├── ...
        └── factura_0656_aug_08_enriched.json
```

---

## ✅ Checklist Final

Antes de considerar la Fase 3 completa:

- [ ] 5,368 archivos `*_enriched.json` generados
- [ ] Match rate promedio >70% (idealmente >85%)
- [ ] Metadata presente en todos los JSONs
- [ ] Coordenadas bbox en formato correcto `[x0, y0, x1, y1]`
- [ ] Items enriquecidos con coordenadas
- [ ] Sin errores masivos en los logs

---

## 🎉 Siguiente Paso: Fase 4

Una vez completada la Fase 3 con éxito:

**Fase 4: Entrenamiento de LayoutLMv3**
- Convertir JSONs enriquecidos a formato FUNSD
- Entrenar modelo LayoutLMv3
- Objetivo: >98% precisión en detección de items prohibidos

---

**¿Listo para empezar? Ejecuta el PASO 1 y sigue la guía paso a paso!** 🚀
