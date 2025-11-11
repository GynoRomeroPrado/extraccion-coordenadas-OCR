# LayoutLMv3 Invoice Processor para Facturas Peruanas

Sistema completo para procesar facturas peruanas en PDF y convertirlas al formato LayoutLMv3 (FUNSD-style) para entrenamiento de modelos de extracción de campos.

## 📋 Características

- ✅ **Detección automática de orientación** (0°, 90°, 180°, 270°)
- ✅ **Extracción híbrida**: texto nativo + OCR (Tesseract)
- ✅ **Matching difuso** con RapidFuzz para tolerancia a errores OCR
- ✅ **Detección de tablas** para items
- ✅ **Formato LayoutLMv3 oficial** (FUNSD-style)
- ✅ **Chunking automático** para documentos >512 tokens
- ✅ **Soporte multipágina** con sliding window
- ✅ **Procesamiento en lote** con checkpoints
- ✅ **Validación completa** del dataset generado

## 🏗️ Arquitectura del Sistema

```
layoutlmv3-invoice-processor/
├── config.py                      # Configuración central
├── requirements.txt               # Dependencias
│
├── src/
│   ├── preprocessor/              # Preprocesamiento de PDFs
│   │   ├── orientation_detector.py    # Detectar y corregir rotación
│   │   ├── deskewer.py                 # Corregir inclinación
│   │   └── image_enhancer.py           # Mejorar calidad OCR
│   │
│   ├── extractors/                # Extracción de texto
│   │   ├── pdf_extractor.py           # Texto nativo (PyMuPDF)
│   │   ├── ocr_extractor.py           # OCR (Tesseract)
│   │   └── hybrid_extractor.py        # Combinación inteligente
│   │
│   ├── matchers/                  # Matching de campos
│   │   ├── fuzzy_matcher.py           # Matching difuso
│   │   └── table_detector.py          # Detectar filas de tabla
│   │
│   ├── formatters/                # Formato de salida
│   │   ├── layoutlmv3_formatter.py    # Formato FUNSD
│   │   └── chunker.py                 # Dividir en chunks
│   │
│   └── pipeline/                  # Orquestación
│       ├── single_page.py             # Pipeline 1 página
│       ├── multipage.py               # Pipeline multipágina
│       └── processor.py               # Orquestador principal
│
├── scripts/                       # Scripts de uso
│   ├── process_single.py              # Procesar 1 documento
│   ├── process_batch.py               # Procesar lote
│   └── validate_output.py             # Validar dataset
│
├── tests/                         # Tests unitarios
└── output/                        # Salida generada
```

## 🚀 Instalación

### Requisitos

- Python 3.8+
- Tesseract OCR 4.0+
- 4GB+ RAM recomendado

### 1. Clonar repositorio

```bash
git clone https://github.com/GynoRomeroPrado/extraccion-coordenadas-OCR.git
cd extraccion-coordenadas-OCR
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar Tesseract OCR

#### Ubuntu/Debian:
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-spa
```

#### macOS:
```bash
brew install tesseract tesseract-lang
```

#### Windows:
Descargar desde: https://github.com/UB-Mannheim/tesseract/wiki

### 5. Verificar instalación

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

## 📖 Uso

### Procesar un documento individual

```bash
python scripts/process_single.py \
    --pdf "facturas_pdf/FACT-000001.pdf" \
    --json "anotaciones_json/FACT-000001.json" \
    --output "output/"
```

**Opciones disponibles:**
- `--dpi`: DPI para OCR (default: 300)
- `--match-threshold`: Umbral de matching 0-100 (default: 85)
- `--no-auto-rotate`: Desactivar detección de orientación
- `--validate`: Validar salida después de procesar
- `--verbose`: Modo verbose

### Procesar lote de documentos

```bash
python scripts/process_batch.py \
    --input-dir "/path/to/dataset_entrenamiento_101125" \
    --output-dir "output/" \
    --start-index 0 \
    --end-index 1000 \
    --skip-existing
```

**Opciones disponibles:**
- `--pdf-subdir`: Subdirectorio con PDFs (default: facturas_pdf)
- `--json-subdir`: Subdirectorio con JSONs (default: anotaciones_json)
- `--max-documents`: Máximo de documentos a procesar
- `--checkpoint-interval`: Guardar checkpoint cada N docs (default: 1000)
- `--num-workers`: (futuro) Workers para procesamiento paralelo

### Validar dataset generado

```bash
python scripts/validate_output.py \
    --output-dir "output/" \
    --check-metadata \
    --save-report "validation_report.json"
```

## 📊 Formato de Entrada

### Estructura del JSON de anotaciones:

```json
{
  "tipo_documento": "FACTURA ELECTRONICA",
  "serie_completa": "F985-390210",
  "fecha_emision": "2025-12-31",
  "emisor_ruc": "20393920069",
  "emisor_razon_social": "La Positiva S.A.",
  "receptor_numero_doc": "20421583138",
  "subtotal": 15234.50,
  "igv": 2742.21,
  "importe_total": 17976.71,
  "items": [
    {
      "item": 1,
      "codigo": "90111500",
      "descripcion": "SERVICIO DE CONSULTORÍA",
      "cantidad": 5.0,
      "precio_unitario": 850.00,
      "valor_venta": 4250.00
    }
  ]
}
```

## 📤 Formato de Salida

### Documento de 1 página:

```
output/
├── FACT-000001_layoutlmv3.json    # Documento completo
└── FACT-000001_metadata.json      # Metadata y estadísticas
```

### Documento multipágina:

```
output/
├── FACT-000002_header.json        # Campos generales (página 0)
├── FACT-000002_items_p1.json      # Items (página 1)
├── FACT-000002_items_p2.json      # Items (página 2)
└── FACT-000002_metadata.json      # Metadata
```

### Formato LayoutLMv3 (FUNSD-style):

```json
{
  "form": [
    {
      "id": 0,
      "text": "FACTURA ELECTRONICA",
      "box": [120, 45, 380, 72],
      "label": "answer",
      "field_name": "tipo_documento",
      "words": [
        {"text": "FACTURA", "box": [120, 45, 250, 72]},
        {"text": "ELECTRONICA", "box": [255, 45, 380, 72]}
      ],
      "page": 0,
      "confidence": 0.95
    }
  ]
}
```

### Metadata:

```json
{
  "document_id": "FACT-000001",
  "num_pages": 1,
  "orientation_detected": 0,
  "extraction_method": "native",
  "components": {
    "header": {
      "fields_found": 87,
      "fields_total": 96,
      "match_rate": 0.906
    },
    "items": {
      "items_total": 14,
      "fields_found": 140,
      "match_rate": 0.952
    }
  },
  "statistics": {
    "total_elements": 227,
    "confidence_avg": 0.91,
    "processing_time_seconds": 8.3
  }
}
```

## ⚙️ Configuración

Editar `config.py` para ajustar parámetros:

```python
class Config:
    # OCR
    DPI = 300
    TESSERACT_LANG = "spa"

    # Matching
    MATCH_THRESHOLD = 85
    CONFIDENCE_THRESHOLD = 0.75

    # Tokenización
    MAX_TOKENS = 384  # Margen de seguridad (LayoutLMv3 acepta 512)
    TOKEN_OVERLAP = 128

    # Coordenadas
    BBOX_SCALE = 1000  # Escala estándar LayoutLMv3

    # Validación
    MIN_MATCH_RATE = 0.85
    MIN_SUCCESS_RATE = 0.95
```

## 🔧 Flujo de Procesamiento

### Pipeline de 1 página:

```
PDF → Detectar orientación → Rotar si necesario → Extraer texto
   → Matchear campos del header → Matchear items
   → Formatear LayoutLMv3 → Chunking si >512 tokens → Guardar
```

### Pipeline multipágina:

```
PDF → Detectar orientación → Extraer todas las páginas
   → Separar páginas de header vs items
   → Matchear header → Detectar tablas en páginas de items
   → Mapear items con filas → Formatear por página → Guardar
```

## 📈 Procesamiento Masivo (34,165 documentos)

### Estrategia recomendada:

```bash
# 1. Dividir en lotes de 1000
for i in {0..34}; do
    start=$((i * 1000))
    end=$(((i + 1) * 1000))

    python scripts/process_batch.py \
        --input-dir "/content/drive/MyDrive/dataset_entrenamiento_101125" \
        --output-dir "output/" \
        --start-index $start \
        --end-index $end \
        --skip-existing \
        --checkpoint-interval 100
done

# 2. Validar dataset completo
python scripts/validate_output.py \
    --output-dir "output/" \
    --save-report "validation_report.json"
```

### Estimaciones de tiempo:

- **1 documento**: ~8 segundos
- **1,000 documentos**: ~2.2 horas
- **34,165 documentos**: ~75 horas (~3 días)

### Optimizaciones:

- Usar SSD para I/O más rápido
- Aumentar DPI solo si OCR falla (300 DPI es suficiente)
- Procesar en lotes pequeños para poder reiniciar si hay errores

## 🧪 Tests

### Ejecutar todos los tests:

```bash
# Test individual
python tests/test_fuzzy_matcher.py
python tests/test_formatter.py

# Con pytest (si instalado)
pytest tests/ -v
```

## 🐛 Troubleshooting

### Error: Tesseract not found

```bash
# Verificar instalación
which tesseract

# Si no está en PATH, agregar a config.py:
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
```

### Error: PDF corrupto

- Verificar que el PDF se puede abrir manualmente
- Intentar reparar con herramientas como `pdftk` o `ghostscript`

### Match rate bajo (<85%)

- Verificar que el PDF y JSON corresponden
- Revisar calidad del PDF (resolución, rotación)
- Ajustar `MATCH_THRESHOLD` en config.py
- Usar `--verbose` para debug

### Out of memory

- Reducir DPI (de 300 a 200)
- Procesar menos documentos en paralelo
- Aumentar RAM del sistema

## 📊 Métricas de Éxito

El sistema será exitoso si alcanza:

- ✅ Match rate promedio > 90%
- ✅ Tiempo promedio < 10 seg/doc
- ✅ Tasa de éxito > 95%
- ✅ Todos los outputs < 512 tokens (o divididos correctamente)
- ✅ Orientación correcta en > 98% de casos

## 🤝 Contribuir

1. Fork el repositorio
2. Crear branch de feature (`git checkout -b feature/amazing-feature`)
3. Commit cambios (`git commit -m 'Add amazing feature'`)
4. Push al branch (`git push origin feature/amazing-feature`)
5. Abrir Pull Request

## 📝 Licencia

Este proyecto está bajo licencia MIT.

## 👥 Autores

- **Gynno Romero Prado** - *Desarrollo inicial*

## 🙏 Agradecimientos

- Tesseract OCR
- PyMuPDF (fitz)
- RapidFuzz
- LayoutLMv3 team

## 📚 Referencias

- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [FUNSD Dataset Format](https://guillaumejaume.github.io/FUNSD/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## 🔮 Roadmap

- [ ] Soporte para más tipos de documentos (boletas, guías de remisión)
- [ ] Procesamiento paralelo con multiprocessing
- [ ] API REST para procesamiento en tiempo real
- [ ] Dashboard web para monitoreo
- [ ] Integración con cloud storage (S3, GCS)
- [ ] Fine-tuning automático de LayoutLMv3 con dataset generado

---

**¿Preguntas o problemas?** Abrir un issue en GitHub.
