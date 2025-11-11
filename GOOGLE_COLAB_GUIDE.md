# 🚀 Guía Rápida para Google Colab

## Instrucciones paso a paso para procesar las 34,165 facturas

---

## 📋 PREPARACIÓN (5 minutos)

### 1. Abrir Google Colab

Ve a: https://colab.research.google.com/

### 2. Crear un nuevo notebook o usar el notebook incluido

**Opción A**: Subir el notebook incluido
- Ir a "Archivo" → "Subir notebook"
- Seleccionar `notebooks/LayoutLMv3_Invoice_Processor_Colab.ipynb`

**Opción B**: Crear uno nuevo y copiar los comandos de abajo

### 3. Cambiar a GPU (Opcional pero recomendado)

- Ir a "Entorno de ejecución" → "Cambiar tipo de entorno"
- Seleccionar "GPU" como acelerador por hardware
- Click "Guardar"

---

## 🔧 INSTALACIÓN (10 minutos)

### Paso 1: Montar Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Paso 2: Verificar dataset

```python
import os

DATASET_DIR = "/content/drive/MyDrive/dataset_entrenamiento_101125"
PDF_DIR = f"{DATASET_DIR}/facturas_pdf"
JSON_DIR = f"{DATASET_DIR}/anotaciones_json"

print(f"Dataset existe: {os.path.exists(DATASET_DIR)}")
print(f"PDFs: {len([f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])}")
print(f"JSONs: {len([f for f in os.listdir(JSON_DIR) if f.endswith('.json')])}")
```

### Paso 3: Instalar Tesseract OCR

```bash
!apt-get update -qq
!apt-get install -y tesseract-ocr tesseract-ocr-spa
!tesseract --version
```

### Paso 4: Clonar repositorio e instalar dependencias

```bash
%cd /content
!git clone https://github.com/GynoRomeroPrado/extraccion-coordenadas-OCR.git
%cd extraccion-coordenadas-OCR
!git checkout claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF
!pip install -q -r requirements.txt
```

---

## 🧪 PRUEBA CON 1 DOCUMENTO (2 minutos)

```bash
OUTPUT_DIR="/content/drive/MyDrive/output_layoutlmv3"
!mkdir -p {OUTPUT_DIR}

!python scripts/process_single.py \
    --pdf "{PDF_DIR}/FACT-000001.pdf" \
    --json "{JSON_DIR}/FACT-000001.json" \
    --output "{OUTPUT_DIR}" \
    --verbose \
    --validate
```

Si todo funciona correctamente, verás:
- ✅ Documento procesado
- 📊 Estadísticas del header
- 📦 Estadísticas de items
- ⏱️ Tiempo de procesamiento

---

## 🚀 PROCESAMIENTO MASIVO

### Opción 1: Primeros 100 documentos (PRUEBA - 15 minutos)

```bash
!python scripts/process_batch.py \
    --input-dir "{DATASET_DIR}" \
    --output-dir "{OUTPUT_DIR}" \
    --start-index 0 \
    --end-index 100 \
    --skip-existing \
    --checkpoint-interval 10
```

### Opción 2: Primeros 1,000 documentos (2-3 horas)

```bash
!python scripts/process_batch.py \
    --input-dir "{DATASET_DIR}" \
    --output-dir "{OUTPUT_DIR}" \
    --start-index 0 \
    --end-index 1000 \
    --skip-existing \
    --checkpoint-interval 100
```

### Opción 3: TODOS los documentos en lotes (~75 horas total)

**IMPORTANTE**: Google Colab tiene límite de 12 horas por sesión.
Necesitas ejecutar este código **35 veces** (lotes de 1,000).

```python
# CAMBIAR ESTE NÚMERO: 0, 1, 2, 3, ... 34
BATCH_NUMBER = 0  # <-- MODIFICAR AQUÍ

BATCH_SIZE = 1000
start = BATCH_NUMBER * BATCH_SIZE
end = (BATCH_NUMBER + 1) * BATCH_SIZE

print(f"🚀 Lote {BATCH_NUMBER}: documentos {start} a {end}")
```

```bash
!python scripts/process_batch.py \
    --input-dir "{DATASET_DIR}" \
    --output-dir "{OUTPUT_DIR}" \
    --start-index {start} \
    --end-index {end} \
    --skip-existing \
    --checkpoint-interval 100
```

**Estrategia recomendada**:
1. Ejecutar lote 0 (docs 0-1000)
2. Esperar ~2.2 horas
3. Cambiar `BATCH_NUMBER = 1`
4. Ejecutar lote 1 (docs 1000-2000)
5. Repetir hasta lote 34

---

## ✅ VALIDACIÓN

```bash
!python scripts/validate_output.py \
    --output-dir "{OUTPUT_DIR}" \
    --check-metadata \
    --save-report "{OUTPUT_DIR}/validation_report.json"
```

---

## 📊 VER ESTADÍSTICAS

```python
import json

# Ver estadísticas del batch
stats_file = f"{OUTPUT_DIR}/batch_statistics.json"
with open(stats_file, 'r') as f:
    stats = json.load(f)

print(f"Total procesados: {stats['processed']}")
print(f"Tasa de éxito: {stats['success_rate']:.1%}")
print(f"Tiempo promedio: {stats['avg_processing_time']:.2f}s")
print(f"Match rate (header): {stats['avg_match_rate_header']:.1%}")
```

---

## 💾 DESCARGAR RESULTADOS

### Opción A: Ya están en tu Google Drive

Los resultados están en:
```
/content/drive/MyDrive/output_layoutlmv3/
```

### Opción B: Crear ZIP para descargar

```python
import shutil

zip_file = "/content/output_layoutlmv3_backup"
shutil.make_archive(zip_file, 'zip', OUTPUT_DIR)
print(f"ZIP creado: {zip_file}.zip")

# Descargar
from google.colab import files
files.download(f"{zip_file}.zip")
```

---

## 🔧 COMANDOS ÚTILES

### Ver progreso actual

```python
import json
checkpoint_file = f"{OUTPUT_DIR}/checkpoint.json"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    print(f"Progreso: {checkpoint['progress']}%")
    print(f"Procesados: {checkpoint['processed']}")
```

### Contar archivos generados

```bash
!ls {OUTPUT_DIR}/*.json | grep -v metadata | grep -v batch_statistics | wc -l
```

### Ver uso de disco

```bash
!du -sh {OUTPUT_DIR}
!df -h /content/drive
```

### Limpiar archivos temporales

```python
import glob
temp_files = glob.glob("/tmp/*_rotated_*.pdf")
for f in temp_files:
    os.remove(f)
print(f"Eliminados {len(temp_files)} archivos temporales")
```

---

## ⚠️ SOLUCIÓN DE PROBLEMAS

### Error: "No module named 'fitz'"

```bash
!pip install PyMuPDF
```

### Error: "Tesseract not found"

```bash
!apt-get install -y tesseract-ocr tesseract-ocr-spa
```

### Error: "Permission denied"

Verificar que Google Drive esté montado:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Sesión se interrumpió

No hay problema, los datos están guardados:
1. Volver a ejecutar celdas de instalación
2. El sistema continuará desde el último checkpoint
3. Usar `--skip-existing` para no reprocesar

### Match rate muy bajo (<80%)

1. Verificar que PDF y JSON corresponden
2. Intentar ajustar threshold:
   ```bash
   --match-threshold 80
   ```
3. Ver detalles con `--verbose`

---

## 📈 TIEMPO ESTIMADO POR LOTE

| Documentos | Tiempo Estimado |
|------------|-----------------|
| 1 | ~8 segundos |
| 10 | ~1.3 minutos |
| 100 | ~13 minutos |
| 1,000 | ~2.2 horas |
| 34,165 | ~75 horas (dividir en 35 lotes) |

---

## 📊 FORMATO DE SALIDA

Para cada documento `FACT-NNNNNN.pdf` se genera:

### Documento de 1 página:
```
FACT-NNNNNN_layoutlmv3.json    # Formato LayoutLMv3
FACT-NNNNNN_metadata.json      # Estadísticas
```

### Documento multipágina:
```
FACT-NNNNNN_header.json        # Campos generales
FACT-NNNNNN_items_p1.json      # Items página 1
FACT-NNNNNN_items_p2.json      # Items página 2
FACT-NNNNNN_metadata.json      # Estadísticas
```

---

## ✅ CHECKLIST DE PROCESAMIENTO

- [ ] Google Drive montado
- [ ] Dataset verificado (34,165 PDFs y JSONs)
- [ ] Tesseract instalado
- [ ] Repositorio clonado
- [ ] Dependencias instaladas
- [ ] Prueba con 1 documento exitosa
- [ ] Lote 0 (0-1000) procesado
- [ ] Lote 1 (1000-2000) procesado
- [ ] ... (continuar hasta lote 34)
- [ ] Validación completa realizada
- [ ] Estadísticas revisadas
- [ ] Resultados respaldados

---

## 🎯 OBJETIVO FINAL

Al completar todos los lotes tendrás:

- ✅ **34,165 documentos** en formato LayoutLMv3
- ✅ **Match rate promedio** > 90%
- ✅ **Coordenadas normalizadas** (0-1000)
- ✅ **Formato FUNSD** compatible con LayoutLMv3
- ✅ **Dataset listo** para entrenar el modelo

---

## 📞 SOPORTE

Si tienes problemas:
1. Revisar sección "Solución de Problemas"
2. Verificar logs con `--verbose`
3. Abrir issue en GitHub

---

**¡Éxito procesando las facturas!** 🚀
