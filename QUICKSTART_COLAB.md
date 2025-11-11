# ⚡ QUICK START - Google Colab (5 minutos)

## 🎯 Copiar y pegar estos comandos en orden

### 1️⃣ Montar Drive (30 seg)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2️⃣ Instalar Tesseract (2 min)
```bash
!apt-get update -qq && apt-get install -y tesseract-ocr tesseract-ocr-spa
```

### 3️⃣ Clonar e instalar (2 min)
```bash
%cd /content
!git clone https://github.com/GynoRomeroPrado/extraccion-coordenadas-OCR.git
%cd extraccion-coordenadas-OCR
!git checkout claude/layoutlmv3-invoice-processor-011CV2g24oh8fxvVVWutKLKF
!pip install -q -r requirements.txt
```

### 4️⃣ Configurar paths
```python
DATASET_DIR = "/content/drive/MyDrive/dataset_entrenamiento_101125"
OUTPUT_DIR = "/content/drive/MyDrive/output_layoutlmv3"
PDF_DIR = f"{DATASET_DIR}/facturas_pdf"
JSON_DIR = f"{DATASET_DIR}/anotaciones_json"
!mkdir -p {OUTPUT_DIR}
```

### 5️⃣ Probar con 1 documento (8 seg)
```bash
!python scripts/process_single.py \
    --pdf "{PDF_DIR}/FACT-000001.pdf" \
    --json "{JSON_DIR}/FACT-000001.json" \
    --output "{OUTPUT_DIR}" \
    --validate
```

---

## 🚀 PROCESAMIENTO MASIVO

### Opción rápida: 100 documentos (13 min)
```bash
!python scripts/process_batch.py \
    --input-dir "{DATASET_DIR}" \
    --output-dir "{OUTPUT_DIR}" \
    --end-index 100 \
    --skip-existing
```

### Lote completo: 1,000 documentos (2.2 horas)
```bash
!python scripts/process_batch.py \
    --input-dir "{DATASET_DIR}" \
    --output-dir "{OUTPUT_DIR}" \
    --end-index 1000 \
    --skip-existing
```

### TODOS (34,165) en lotes de 1,000
```python
# CAMBIAR ESTE NÚMERO cada vez: 0, 1, 2, ... 34
LOTE = 0

!python scripts/process_batch.py \
    --input-dir "{DATASET_DIR}" \
    --output-dir "{OUTPUT_DIR}" \
    --start-index {LOTE * 1000} \
    --end-index {(LOTE + 1) * 1000} \
    --skip-existing
```

---

## ✅ VALIDAR

```bash
!python scripts/validate_output.py --output-dir "{OUTPUT_DIR}"
```

---

## 📊 VER ESTADÍSTICAS

```python
import json
with open(f"{OUTPUT_DIR}/batch_statistics.json") as f:
    stats = json.load(f)
print(f"✅ Procesados: {stats['processed']}/{stats['total_documents']}")
print(f"📊 Match rate: {stats['avg_match_rate_header']:.1%}")
print(f"⏱️  Tiempo/doc: {stats['avg_processing_time']:.1f}s")
```

---

## 💡 TIPS

- **Si se interrumpe**: Volver a ejecutar, usa `--skip-existing` automáticamente
- **Ver progreso**: Archivos se guardan en Google Drive en tiempo real
- **Límite Colab**: 12 horas/sesión → máximo ~1,500 docs por sesión
- **Lotes recomendados**: 1,000 documentos por sesión

---

## 🎯 RESULTADO

Cada factura genera:
```
FACT-NNNNNN_layoutlmv3.json  → Formato LayoutLMv3 listo para entrenar
FACT-NNNNNN_metadata.json    → Estadísticas y calidad
```

---

**¡Listo! En 5 minutos puedes empezar a procesar.** 🚀

Ver guía completa: `GOOGLE_COLAB_GUIDE.md`
