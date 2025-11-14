#!/usr/bin/env python3
"""
Script para enriquecer JSONs con coordenadas bbox usando PaddleOCR

Este script toma:
- Directorio con PDFs de facturas
- Directorio con JSONs de anotaciones (SIN coordenadas)

Y genera:
- JSONs enriquecidos con coordenadas bbox para cada campo

Uso:
    python scripts/enrich_jsons.py \
        --pdf-dir "augmented_pdf/" \
        --json-dir "augmented_json/" \
        --output-dir "enriched_json/" \
        --batch-size 8 \
        --use-gpu

Ejemplo:
    python scripts/enrich_jsons.py \
        --pdf-dir "/content/drive/MyDrive/dataset/augmented_pdf" \
        --json-dir "/content/drive/MyDrive/dataset/augmented_json" \
        --output-dir "/content/drive/MyDrive/dataset/enriched_json" \
        --batch-size 16 \
        --use-gpu
"""
import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paddleocr import PaddleOCR
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm

from src.matchers.fuzzy_matcher import FuzzyMatcher
from src.matchers.coordinate_enhancer import CoordinateEnhancer
from config import Config

# Configurar logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT
)
logger = logging.getLogger(__name__)


class JSONEnricher:
    """
    Enriquece JSONs de anotaciones con coordenadas bbox usando PaddleOCR
    """

    def __init__(
        self,
        use_gpu: bool = False,
        dpi: int = 300,
        match_threshold: int = 85,
        min_confidence: float = 0.5
    ):
        """
        Args:
            use_gpu: Usar GPU para PaddleOCR
            dpi: DPI para conversión de PDF a imagen
            match_threshold: Umbral para fuzzy matching (0-100)
            min_confidence: Confidence mínimo para incluir match
        """
        self.dpi = dpi
        self.match_threshold = match_threshold
        self.min_confidence = min_confidence
        self.bbox_scale = Config.BBOX_SCALE

        # Inicializar PaddleOCR
        logger.info(f"Inicializando PaddleOCR (GPU: {use_gpu})...")
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='es',
                use_gpu=use_gpu,
                show_log=False
            )
            logger.info("✅ PaddleOCR inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error al inicializar PaddleOCR: {e}")
            raise

        # Inicializar matcher y enhancer
        self.matcher = FuzzyMatcher(threshold=match_threshold)
        self.enhancer = CoordinateEnhancer()

        # Estadísticas globales
        self.stats = {
            'total_documents': 0,
            'total_fields_attempted': 0,
            'total_fields_matched': 0,
            'total_items_attempted': 0,
            'total_items_matched': 0,
            'total_words_extracted': 0,
            'processing_time': 0.0
        }

    def enrich_json(
        self,
        pdf_path: Path,
        json_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Enriquece un JSON individual con coordenadas

        Args:
            pdf_path: Ruta al PDF
            json_path: Ruta al JSON original
            output_path: Ruta del JSON enriquecido

        Returns:
            Diccionario con metadata del procesamiento
        """
        start_time = time.time()

        try:
            # 1. Extraer palabras del PDF con PaddleOCR
            logger.debug(f"Extrayendo texto de {pdf_path.name}...")
            words = self._extract_words_from_pdf(str(pdf_path))

            if not words:
                logger.warning(f"⚠️  No se extrajo texto de {pdf_path.name}")
                return {
                    'success': False,
                    'error': 'No se extrajo texto del PDF',
                    'words_extracted': 0
                }

            # 2. Cargar JSON original
            with open(json_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)

            # 3. Enriquecer campos
            enriched_data, metadata = self._enrich_fields(
                original_data,
                words,
                pdf_path.name
            )

            # 4. Guardar JSON enriquecido
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_data, f, indent=2, ensure_ascii=False)

            # Actualizar estadísticas
            processing_time = time.time() - start_time
            metadata['processing_time'] = round(processing_time, 3)
            metadata['success'] = True

            return metadata

        except Exception as e:
            logger.error(f"❌ Error procesando {pdf_path.name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def _extract_words_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extrae palabras con coordenadas usando PaddleOCR

        Args:
            pdf_path: Ruta al PDF

        Returns:
            Lista de diccionarios con {text, bbox, confidence}
        """
        try:
            # Convertir PDF a imagen
            image = self._pdf_to_image(pdf_path)
            img_width, img_height = image.size

            # Convertir a numpy array
            img_array = np.array(image)

            # Aplicar OCR
            result = self.ocr.ocr(img_array, cls=True)

            if result is None or len(result) == 0 or result[0] is None:
                return []

            # Procesar resultado
            words = []
            for line in result[0]:
                if line is None:
                    continue

                bbox_poly = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (texto, confidence)

                text = text_info[0].strip()
                confidence = float(text_info[1])

                if not text:
                    continue

                # Convertir bbox poligonal a rectangular
                x_coords = [p[0] for p in bbox_poly]
                y_coords = [p[1] for p in bbox_poly]

                x0 = int(min(x_coords))
                y0 = int(min(y_coords))
                x1 = int(max(x_coords))
                y1 = int(max(y_coords))

                # Normalizar a escala 0-1000
                bbox_normalized = self._normalize_bbox(
                    [x0, y0, x1, y1],
                    img_width,
                    img_height
                )

                words.append({
                    'text': text,
                    'bbox': bbox_normalized,
                    'confidence': confidence
                })

            logger.debug(f"  Extraídas {len(words)} palabras con PaddleOCR")
            return words

        except Exception as e:
            logger.error(f"Error en _extract_words_from_pdf: {e}")
            return []

    def _pdf_to_image(self, pdf_path: str, page_num: int = 0) -> Image.Image:
        """
        Convierte una página de PDF a imagen PIL

        Args:
            pdf_path: Ruta al PDF
            page_num: Número de página

        Returns:
            Imagen PIL
        """
        doc = fitz.open(pdf_path)

        if page_num >= len(doc):
            raise ValueError(f"Página {page_num} no existe")

        page = doc[page_num]

        # Calcular zoom para DPI deseado
        zoom = self.dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Convertir a PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        doc.close()

        return img

    def _normalize_bbox(
        self,
        bbox: List[float],
        img_width: float,
        img_height: float
    ) -> List[int]:
        """
        Normaliza bbox a escala 0-1000

        Args:
            bbox: [x0, y0, x1, y1]
            img_width: Ancho de imagen
            img_height: Alto de imagen

        Returns:
            Bbox normalizado [x0, y0, x1, y1]
        """
        x0, y0, x1, y1 = bbox

        x0_norm = int((x0 / img_width) * self.bbox_scale)
        y0_norm = int((y0 / img_height) * self.bbox_scale)
        x1_norm = int((x1 / img_width) * self.bbox_scale)
        y1_norm = int((y1 / img_height) * self.bbox_scale)

        # Asegurar rango válido
        x0_norm = max(0, min(self.bbox_scale, x0_norm))
        y0_norm = max(0, min(self.bbox_scale, y0_norm))
        x1_norm = max(0, min(self.bbox_scale, x1_norm))
        y1_norm = max(0, min(self.bbox_scale, y1_norm))

        return [x0_norm, y0_norm, x1_norm, y1_norm]

    def _enrich_fields(
        self,
        original_data: Dict,
        words: List[Dict],
        pdf_name: str
    ) -> tuple[Dict, Dict]:
        """
        Enriquece todos los campos del JSON con coordenadas

        Args:
            original_data: Datos originales del JSON
            words: Palabras extraídas con OCR
            pdf_name: Nombre del PDF

        Returns:
            (datos_enriquecidos, metadata)
        """
        enriched = {
            "_metadata": {
                "original_file": pdf_name,
                "processing_date": datetime.now().isoformat(),
                "ocr_engine": "PaddleOCR",
                "total_words_detected": len(words),
                "fields_matched": 0,
                "fields_total": 0,
                "items_matched": 0,
                "items_total": 0,
                "match_rate": 0.0
            }
        }

        fields_matched = 0
        fields_total = 0

        # Separar items y cuotas del resto
        items = original_data.pop('items', [])
        cuotas = original_data.pop('cuotas', [])

        # Enriquecer campos del header
        for field_name, field_value in original_data.items():
            if field_value is None or field_value == "":
                # Campo sin valor, no enriquecer
                enriched[field_name] = field_value
                continue

            fields_total += 1

            # Buscar match en palabras OCR
            match = self._find_match(field_name, field_value, words)

            if match and match['confidence'] >= self.min_confidence:
                # Campo encontrado con buena confidence
                enriched[field_name] = {
                    "text": str(field_value),
                    "bbox": match['bbox'],
                    "confidence": round(match['confidence'], 3)
                }
                fields_matched += 1
            else:
                # No encontrado o baja confidence
                enriched[field_name] = {
                    "text": str(field_value),
                    "bbox": None,
                    "confidence": 0.0,
                    "not_found": True
                }

        # Enriquecer items
        enriched_items, items_stats = self._enrich_items(items, words)
        if enriched_items:
            enriched['items'] = enriched_items

        # Enriquecer cuotas (si existen)
        if cuotas:
            enriched_cuotas, _ = self._enrich_items(cuotas, words)
            if enriched_cuotas:
                enriched['cuotas'] = enriched_cuotas

        # Actualizar metadata
        enriched['_metadata']['fields_matched'] = fields_matched
        enriched['_metadata']['fields_total'] = fields_total
        enriched['_metadata']['items_matched'] = items_stats['matched']
        enriched['_metadata']['items_total'] = items_stats['total']

        match_rate = fields_matched / fields_total if fields_total > 0 else 0
        enriched['_metadata']['match_rate'] = round(match_rate, 3)

        metadata = {
            'words_extracted': len(words),
            'fields_matched': fields_matched,
            'fields_total': fields_total,
            'items_matched': items_stats['matched'],
            'items_total': items_stats['total'],
            'match_rate': match_rate
        }

        return enriched, metadata

    def _find_match(
        self,
        field_name: str,
        field_value: Any,
        words: List[Dict]
    ) -> Optional[Dict]:
        """
        Encuentra match para un campo en las palabras OCR

        Args:
            field_name: Nombre del campo
            field_value: Valor del campo
            words: Palabras extraídas

        Returns:
            Match con bbox y confidence, o None
        """
        # Usar matcher apropiado según tipo
        if isinstance(field_value, (int, float)):
            match = self.matcher.match_numeric(field_value, words, field_name)
        elif isinstance(field_value, str) and len(field_value.split()) > 1:
            match = self.matcher.match_multiword(field_value, words, field_name)
        else:
            match = self.matcher.match(str(field_value), words, field_name)

        # Aplicar enhancement si hay match
        if match:
            enhanced = self.enhancer.enhance_match(
                match,
                field_name,
                words
            )
            return enhanced

        return None

    def _enrich_items(
        self,
        items: List[Dict],
        words: List[Dict]
    ) -> tuple[List[Dict], Dict]:
        """
        Enriquece lista de items con coordenadas

        Args:
            items: Lista de items originales
            words: Palabras extraídas

        Returns:
            (items_enriquecidos, stats)
        """
        enriched_items = []
        items_matched = 0
        items_total = len(items)

        for item in items:
            enriched_item = {}

            for field_name, field_value in item.items():
                if field_value is None or field_value == "":
                    enriched_item[field_name] = field_value
                    continue

                # Buscar match
                match = self._find_match(f"item.{field_name}", field_value, words)

                if match and match['confidence'] >= self.min_confidence:
                    enriched_item[field_name] = {
                        "text": str(field_value),
                        "bbox": match['bbox'],
                        "confidence": round(match['confidence'], 3)
                    }
                    items_matched += 1
                else:
                    enriched_item[field_name] = {
                        "text": str(field_value),
                        "bbox": None,
                        "confidence": 0.0,
                        "not_found": True
                    }

            enriched_items.append(enriched_item)

        stats = {
            'matched': items_matched,
            'total': items_total
        }

        return enriched_items, stats

    def process_batch(
        self,
        pdf_dir: Path,
        json_dir: Path,
        output_dir: Path,
        max_documents: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Procesa un lote de documentos

        Args:
            pdf_dir: Directorio con PDFs
            json_dir: Directorio con JSONs originales
            output_dir: Directorio de salida
            max_documents: Máximo de documentos (None = todos)

        Returns:
            Estadísticas del procesamiento
        """
        # Encontrar archivos
        pdf_files = sorted(list(pdf_dir.glob("*.pdf")))

        if max_documents:
            pdf_files = pdf_files[:max_documents]

        logger.info(f"📊 Encontrados {len(pdf_files)} PDFs para procesar")

        # Crear directorio de salida
        output_dir.mkdir(parents=True, exist_ok=True)

        # Procesar cada archivo
        success_count = 0
        failed_count = 0
        total_time = 0.0

        for pdf_path in tqdm(pdf_files, desc="Enriqueciendo JSONs"):
            # Buscar JSON correspondiente
            json_path = json_dir / f"{pdf_path.stem}.json"

            if not json_path.exists():
                logger.warning(f"⚠️  JSON no encontrado para {pdf_path.name}")
                failed_count += 1
                continue

            # Procesar
            output_path = output_dir / f"{pdf_path.stem}_enriched.json"

            metadata = self.enrich_json(pdf_path, json_path, output_path)

            if metadata.get('success', False):
                success_count += 1
                total_time += metadata.get('processing_time', 0)

                # Actualizar estadísticas globales
                self.stats['total_fields_attempted'] += metadata.get('fields_total', 0)
                self.stats['total_fields_matched'] += metadata.get('fields_matched', 0)
                self.stats['total_items_attempted'] += metadata.get('items_total', 0)
                self.stats['total_items_matched'] += metadata.get('items_matched', 0)
                self.stats['total_words_extracted'] += metadata.get('words_extracted', 0)
            else:
                failed_count += 1
                logger.error(f"❌ Falló: {pdf_path.name} - {metadata.get('error', 'Unknown')}")

        self.stats['total_documents'] = len(pdf_files)
        self.stats['processing_time'] = total_time

        # Calcular tasas finales
        success_rate = success_count / len(pdf_files) if pdf_files else 0
        match_rate = (
            self.stats['total_fields_matched'] / self.stats['total_fields_attempted']
            if self.stats['total_fields_attempted'] > 0 else 0
        )

        return {
            'total_documents': len(pdf_files),
            'success': success_count,
            'failed': failed_count,
            'success_rate': round(success_rate, 3),
            'match_rate': round(match_rate, 3),
            'avg_time_per_doc': round(total_time / success_count, 3) if success_count > 0 else 0,
            'total_time': round(total_time, 2),
            'fields_matched': self.stats['total_fields_matched'],
            'fields_total': self.stats['total_fields_attempted'],
            'items_matched': self.stats['total_items_matched'],
            'items_total': self.stats['total_items_attempted'],
            'words_extracted': self.stats['total_words_extracted']
        }


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Enriquece JSONs de anotaciones con coordenadas bbox usando PaddleOCR"
    )

    parser.add_argument(
        '--pdf-dir',
        type=str,
        required=True,
        help="Directorio con PDFs de facturas"
    )

    parser.add_argument(
        '--json-dir',
        type=str,
        required=True,
        help="Directorio con JSONs de anotaciones (sin coordenadas)"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help="Directorio de salida para JSONs enriquecidos"
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help="Procesar solo los primeros N documentos (para testing)"
    )

    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help="Usar GPU para PaddleOCR (requiere paddlepaddle-gpu)"
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help="DPI para conversión de PDF (default: 300)"
    )

    parser.add_argument(
        '--match-threshold',
        type=int,
        default=85,
        help="Umbral de fuzzy matching 0-100 (default: 85)"
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help="Confidence mínimo para incluir match (default: 0.5)"
    )

    args = parser.parse_args()

    # Validar directorios
    pdf_dir = Path(args.pdf_dir)
    json_dir = Path(args.json_dir)
    output_dir = Path(args.output_dir)

    if not pdf_dir.exists():
        logger.error(f"❌ Directorio de PDFs no existe: {pdf_dir}")
        sys.exit(1)

    if not json_dir.exists():
        logger.error(f"❌ Directorio de JSONs no existe: {json_dir}")
        sys.exit(1)

    # Inicializar enricher
    logger.info("\n" + "=" * 80)
    logger.info("🚀 INICIANDO ENRIQUECIMIENTO DE JSONs")
    logger.info("=" * 80)

    enricher = JSONEnricher(
        use_gpu=args.use_gpu,
        dpi=args.dpi,
        match_threshold=args.match_threshold,
        min_confidence=args.min_confidence
    )

    # Procesar batch
    start_time = time.time()

    stats = enricher.process_batch(
        pdf_dir,
        json_dir,
        output_dir,
        max_documents=args.batch_size
    )

    total_time = time.time() - start_time

    # Mostrar resultados
    logger.info("\n" + "=" * 80)
    logger.info("✅ PROCESAMIENTO COMPLETADO")
    logger.info("=" * 80)

    logger.info(f"\n⏱️  Tiempo total: {total_time / 60:.2f} minutos")
    logger.info(f"⏱️  Tiempo promedio: {stats['avg_time_per_doc']:.3f}s por documento")

    logger.info(f"\n📊 ESTADÍSTICAS:")
    logger.info(f"  Total documentos: {stats['total_documents']}")
    logger.info(f"  ✅ Exitosos: {stats['success']} ({stats['success_rate'] * 100:.1f}%)")
    logger.info(f"  ❌ Fallidos: {stats['failed']}")

    logger.info(f"\n📋 CAMPOS:")
    logger.info(f"  Total campos: {stats['fields_total']}")
    logger.info(f"  ✅ Con coordenadas: {stats['fields_matched']} ({stats['match_rate'] * 100:.1f}%)")
    logger.info(f"  ❌ Sin coordenadas: {stats['fields_total'] - stats['fields_matched']}")

    logger.info(f"\n📦 ITEMS:")
    logger.info(f"  Total items: {stats['items_total']}")
    logger.info(f"  ✅ Con coordenadas: {stats['items_matched']}")

    logger.info(f"\n🔤 PALABRAS EXTRAÍDAS:")
    logger.info(f"  Total: {stats['words_extracted']:,}")
    logger.info(f"  Promedio por documento: {stats['words_extracted'] // stats['success'] if stats['success'] > 0 else 0}")

    logger.info(f"\n📁 OUTPUT:")
    logger.info(f"  Directorio: {output_dir}")
    logger.info(f"  Archivos generados: {stats['success']}")

    logger.info("\n" + "=" * 80)

    # Verificar si el match rate es bueno
    if stats['match_rate'] < 0.8:
        logger.warning(
            f"\n⚠️  ADVERTENCIA: Match rate bajo ({stats['match_rate'] * 100:.1f}%)\n"
            f"   Considera:\n"
            f"   - Reducir --match-threshold (actual: {args.match_threshold})\n"
            f"   - Reducir --min-confidence (actual: {args.min_confidence})\n"
            f"   - Verificar calidad de los PDFs\n"
        )


if __name__ == "__main__":
    main()
