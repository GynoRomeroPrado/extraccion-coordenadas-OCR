# ============================================================================
# SCRIPT PARA PROCESAR DATASET COMPLETO - 5,368 FACTURAS
# ============================================================================

import os
import sys
import time
import subprocess
from pathlib import Path

print("🚀 PROCESAMIENTO COMPLETO DEL DATASET")
print("=" * 70)
print(f"Total de facturas: 5,368")
print(f"Tiempo estimado: ~4-5 horas con GPU T4")
print("=" * 70 + "\n")

# Verificar que las variables globales existan
required_vars = ['BASE_DIR', 'AUGMENTED_PDF_DIR', 'AUGMENTED_JSON_DIR', 'REPO_DIR']
missing = [var for var in required_vars if var not in globals()]

if missing:
    print(f"❌ ERROR: Variables faltantes: {missing}")
    print("\n⚠️  Ejecuta primero el SETUP COMPLETO antes de este script")
    sys.exit(1)

# Configurar rutas
output_dir = f"{BASE_DIR}/modificaciones/enriched_json"

print(f"📁 RUTAS:")
print(f"  PDFs:   {AUGMENTED_PDF_DIR}")
print(f"  JSONs:  {AUGMENTED_JSON_DIR}")
print(f"  Output: {output_dir}\n")

# Verificar que los directorios existan
pdf_count = len([f for f in os.listdir(AUGMENTED_PDF_DIR) if f.endswith('.pdf')])
json_count = len([f for f in os.listdir(AUGMENTED_JSON_DIR) if f.endswith('.json')])

print(f"📊 DATASET:")
print(f"  PDFs:  {pdf_count:,}")
print(f"  JSONs: {json_count:,}\n")

if pdf_count != 5368 or json_count != 5368:
    print(f"⚠️  ADVERTENCIA: Esperábamos 5,368 archivos de cada tipo")
    response = input("¿Continuar de todos modos? (s/n): ")
    if response.lower() != 's':
        print("❌ Cancelado por el usuario")
        sys.exit(0)

# Crear directorio de salida
os.makedirs(output_dir, exist_ok=True)

# Preguntar confirmación
print("\n" + "=" * 70)
print("⚠️  CONFIRMACIÓN REQUERIDA")
print("=" * 70)
print(f"\nEstás a punto de procesar {pdf_count:,} facturas")
print(f"Tiempo estimado: {(pdf_count * 3) / 3600:.1f} horas")
print(f"Costo aproximado GPU Colab: ~$0 (Colab gratuito con límites)")
print(f"\nArchivos de salida: ~{pdf_count * 50 / 1024:.1f} MB estimados")

response = input("\n¿Proceder con el procesamiento completo? (s/n): ")

if response.lower() != 's':
    print("\n❌ Procesamiento cancelado")
    sys.exit(0)

# Cambiar al directorio del repositorio
os.chdir(REPO_DIR)

# Construir comando
cmd = [
    "python", "scripts/enrich_jsons.py",
    "--pdf-dir", AUGMENTED_PDF_DIR,
    "--json-dir", AUGMENTED_JSON_DIR,
    "--output-dir", output_dir,
    "--use-gpu",
    "--dpi", "300",
    "--match-threshold", "85",
    "--min-confidence", "0.5"
]

# Permitir ajuste de parámetros
print("\n" + "=" * 70)
print("⚙️  PARÁMETROS DE PROCESAMIENTO")
print("=" * 70)
print("\nParámetros actuales:")
print(f"  --match-threshold: 85 (umbral fuzzy matching)")
print(f"  --min-confidence: 0.5 (confidence mínimo)")
print(f"  --dpi: 300 (resolución)")

adjust = input("\n¿Ajustar parámetros? (s/n): ")

if adjust.lower() == 's':
    try:
        threshold = input("  Match threshold (70-95, default 85): ").strip()
        if threshold:
            cmd[cmd.index("85")] = threshold

        confidence = input("  Min confidence (0.3-0.7, default 0.5): ").strip()
        if confidence:
            cmd[cmd.index("0.5")] = confidence

        dpi = input("  DPI (200-400, default 300): ").strip()
        if dpi:
            cmd[cmd.index("300")] = dpi

        print("\n✅ Parámetros actualizados")
    except Exception as e:
        print(f"\n⚠️  Error ajustando parámetros: {e}")
        print("   Usando valores por defecto")

# Ejecutar procesamiento
print("\n" + "=" * 70)
print("🚀 INICIANDO PROCESAMIENTO")
print("=" * 70 + "\n")

print(f"Comando: {' '.join(cmd)}\n")

start_time = time.time()

# Ejecutar con output en tiempo real
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

# Mostrar output en tiempo real
for line in iter(process.stdout.readline, ''):
    if line:
        print(line, end='')

process.wait()

elapsed = time.time() - start_time

# Verificar resultado
if process.returncode == 0:
    print("\n" + "=" * 70)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 70)

    # Verificar archivos generados
    enriched_files = list(Path(output_dir).glob("*_enriched.json"))

    print(f"\n📊 RESULTADOS:")
    print(f"  Tiempo total: {elapsed / 3600:.2f} horas")
    print(f"  Tiempo promedio: {elapsed / pdf_count:.2f}s por documento")
    print(f"  Archivos generados: {len(enriched_files):,}/{pdf_count:,}")

    if len(enriched_files) < pdf_count:
        print(f"\n⚠️  ADVERTENCIA: Se generaron menos archivos de los esperados")
        print(f"   Esperados: {pdf_count:,}")
        print(f"   Generados: {len(enriched_files):,}")
        print(f"   Faltantes: {pdf_count - len(enriched_files):,}")

    # Calcular estadísticas agregadas
    if enriched_files:
        print("\n📊 ESTADÍSTICAS AGREGADAS:")
        print("   (Analizando muestra de 100 archivos...)\n")

        import json
        import random

        sample = random.sample(enriched_files, min(100, len(enriched_files)))

        total_fields = 0
        matched_fields = 0
        total_items = 0
        matched_items = 0
        match_rates = []

        for file in sample:
            try:
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
            except Exception as e:
                print(f"   ⚠️  Error leyendo {file.name}: {e}")

        if match_rates:
            avg_match_rate = sum(match_rates) / len(match_rates)

            print(f"  Muestra analizada: {len(sample)} archivos")
            print(f"\n  Campos totales: {total_fields:,}")
            print(f"    ✅ Con coordenadas: {matched_fields:,} ({matched_fields/total_fields*100:.1f}%)")
            print(f"    ❌ Sin coordenadas: {total_fields - matched_fields:,}")

            print(f"\n  Items totales: {total_items:,}")
            print(f"    ✅ Con coordenadas: {matched_items:,}")

            print(f"\n  Match rate promedio: {avg_match_rate * 100:.1f}%")

            print("\n" + "=" * 70)
            if avg_match_rate >= 0.85:
                print("🎉 EXCELENTE! Match rate >85%")
                print("   Dataset listo para Fase 4 (Entrenamiento LayoutLMv3)")
            elif avg_match_rate >= 0.70:
                print("✅ BUENO. Match rate entre 70-85%")
                print("   Dataset utilizable, pero podrías mejorar ajustando parámetros")
            else:
                print("⚠️  BAJO. Match rate <70%")
                print("   Considera re-procesar con parámetros más permisivos")
            print("=" * 70)

    print(f"\n📁 Archivos de salida en:")
    print(f"   {output_dir}")

    print("\n✅ Siguiente paso: Fase 4 - Entrenamiento de LayoutLMv3")

else:
    print("\n" + "=" * 70)
    print("❌ ERROR EN EL PROCESAMIENTO")
    print("=" * 70)
    print(f"\nCódigo de error: {process.returncode}")
    print("\nRevisa los mensajes de error arriba para diagnosticar")

print("\n" + "=" * 70)
print("FIN DEL PROCESAMIENTO")
print("=" * 70)
