# ============================================================================
# SCRIPT PARA EJECUTAR ENRIQUECIMIENTO DE JSONs - PRUEBA CON 5 FACTURAS
# ============================================================================

import random
import shutil
import os
import sys
from pathlib import Path

print("🧪 PREPARANDO PRUEBA CON 5 FACTURAS\n")
print("=" * 70)

# 1. Limpiar y crear directorio temporal
print("\n📁 Configurando temporal_pruebas/...\n")

temporal_pdf = f"{TEMPORAL_DIR}/facturas_pdf"
temporal_json = f"{TEMPORAL_DIR}/anotaciones_json"
temporal_output = f"{TEMPORAL_DIR}/enriched_json"

for folder in [temporal_pdf, temporal_json, temporal_output]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# 2. Seleccionar 5 facturas aleatorias
print("🎲 Seleccionando 5 facturas aleatorias...\n")

all_pdfs = [f for f in os.listdir(AUGMENTED_PDF_DIR) if f.endswith('.pdf')]
selected = random.sample(all_pdfs, 5)

for pdf_name in selected:
    json_name = pdf_name.replace('.pdf', '.json')

    # Copiar PDF
    shutil.copy2(
        f"{AUGMENTED_PDF_DIR}/{pdf_name}",
        f"{temporal_pdf}/{pdf_name}"
    )

    # Copiar JSON
    shutil.copy2(
        f"{AUGMENTED_JSON_DIR}/{json_name}",
        f"{temporal_json}/{json_name}"
    )

    print(f"  ✅ {pdf_name}")

print(f"\n✅ 5 facturas copiadas a temporal_pruebas/")

# 3. Ejecutar script de enriquecimiento
print("\n" + "=" * 70)
print("🚀 EJECUTANDO ENRIQUECIMIENTO")
print("=" * 70 + "\n")

# Cambiar al directorio del repositorio
os.chdir(REPO_DIR)

# Ejecutar el script usando subprocess para capturar output
import subprocess

cmd = [
    "python", "scripts/enrich_jsons.py",
    "--pdf-dir", temporal_pdf,
    "--json-dir", temporal_json,
    "--output-dir", temporal_output,
    "--use-gpu",
    "--dpi", "300",
    "--match-threshold", "85",
    "--min-confidence", "0.5"
]

print(f"Comando: {' '.join(cmd)}\n")

# Ejecutar
result = subprocess.run(cmd, capture_output=True, text=True)

# Mostrar output
print(result.stdout)

if result.returncode != 0:
    print("\n❌ ERROR en el procesamiento:")
    print(result.stderr)
else:
    print("\n" + "=" * 70)
    print("✅ PRUEBA COMPLETADA")
    print("=" * 70)

    # 4. Verificar resultados
    print("\n📊 VERIFICANDO RESULTADOS:\n")

    enriched_files = list(Path(temporal_output).glob("*_enriched.json"))

    if enriched_files:
        import json

        print(f"Archivos enriquecidos: {len(enriched_files)}/5\n")

        # Analizar primer archivo
        with open(enriched_files[0], 'r', encoding='utf-8') as f:
            sample = json.load(f)

        metadata = sample.get('_metadata', {})

        print("📋 METADATA del primer archivo:")
        print(f"  Archivo original: {metadata.get('original_file')}")
        print(f"  OCR Engine: {metadata.get('ocr_engine')}")
        print(f"  Palabras detectadas: {metadata.get('total_words_detected')}")
        print(f"  Campos totales: {metadata.get('fields_total')}")
        print(f"  Campos con coordenadas: {metadata.get('fields_matched')}")
        print(f"  Match rate: {metadata.get('match_rate', 0) * 100:.1f}%")

        # Mostrar ejemplos de campos
        print("\n📝 PRIMEROS 3 CAMPOS ENRIQUECIDOS:")
        count = 0
        for key, value in sample.items():
            if key == '_metadata' or not isinstance(value, dict):
                continue

            if count >= 3:
                break

            print(f"\n  {count+1}. {key}:")
            print(f"     text: {value.get('text')}")
            print(f"     bbox: {value.get('bbox')}")
            print(f"     confidence: {value.get('confidence', 0):.3f}")

            if value.get('not_found'):
                print(f"     ⚠️  NO ENCONTRADO")

            count += 1

        # Verificar items si existen
        if 'items' in sample and sample['items']:
            print(f"\n📦 ITEMS: {len(sample['items'])} encontrados")

            if sample['items']:
                first_item = sample['items'][0]
                print("\n  Ejemplo del primer item:")

                item_count = 0
                for key, value in first_item.items():
                    if isinstance(value, dict) and item_count < 3:
                        bbox_str = str(value.get('bbox'))
                        conf_str = f"{value.get('confidence', 0):.3f}"
                        print(f"    {key}: {value.get('text')} (bbox: {bbox_str}, conf: {conf_str})")
                        item_count += 1

        # Estadísticas agregadas
        print("\n" + "=" * 70)
        print("📊 ESTADÍSTICAS AGREGADAS DE LOS 5 ARCHIVOS:")
        print("=" * 70 + "\n")

        total_fields = 0
        matched_fields = 0
        total_items = 0
        matched_items = 0
        match_rates = []

        for file in enriched_files:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            meta = data.get('_metadata', {})
            total_fields += meta.get('fields_total', 0)
            matched_fields += meta.get('fields_matched', 0)
            total_items += meta.get('items_total', 0)
            matched_items += meta.get('items_matched', 0)

            match_rate = meta.get('match_rate', 0)
            if match_rate > 0:
                match_rates.append(match_rate)

        avg_match_rate = sum(match_rates) / len(match_rates) if match_rates else 0

        print(f"Archivos procesados: {len(enriched_files)}/5")
        print(f"\nCampos totales: {total_fields}")
        print(f"  ✅ Con coordenadas: {matched_fields} ({matched_fields/total_fields*100:.1f}%)")
        print(f"  ❌ Sin coordenadas: {total_fields - matched_fields}")

        print(f"\nItems totales: {total_items}")
        print(f"  ✅ Con coordenadas: {matched_items}")

        print(f"\nMatch rate promedio: {avg_match_rate * 100:.1f}%")

        # Recomendación
        print("\n" + "=" * 70)
        if avg_match_rate >= 0.85:
            print("✅ EXCELENTE! Match rate >85%")
            print("   👉 Puedes proceder con las 5,368 facturas completas")
        elif avg_match_rate >= 0.70:
            print("⚠️  ACEPTABLE. Match rate entre 70-85%")
            print("   👉 Considera ajustar parámetros:")
            print("      --match-threshold 75 (en lugar de 85)")
            print("      --min-confidence 0.4 (en lugar de 0.5)")
        else:
            print("❌ BAJO. Match rate <70%")
            print("   👉 Ajusta parámetros:")
            print("      --match-threshold 70")
            print("      --min-confidence 0.3")
        print("=" * 70)

    else:
        print("❌ No se generaron archivos enriquecidos")
        print("\nRevisa los errores arriba para diagnosticar el problema")

print("\n📁 Archivos de salida en:")
print(f"   {temporal_output}")
