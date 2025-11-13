"""
Extractor con procesamiento batch para maximizar throughput GPU

Procesa múltiples documentos simultáneamente para:
- Amortizar overhead de GPU
- Maximizar utilización de VRAM
- Reducir tiempo total de procesamiento

Performance esperado:
- Sin batch: ~0.2s por documento
- Con batch=8: ~0.05s por documento (4x speedup)
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

from .hybrid_extractor_v2 import HybridExtractorV2
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class BatchExtractor:
    """
    Extractor optimizado para procesamiento batch

    Features:
    - Procesa múltiples documentos en paralelo
    - Pipeline asíncrono (carga + OCR + guardado)
    - Utilización óptima de GPU
    - Progress tracking
    """

    def __init__(
        self,
        batch_size: int = 8,
        use_gpu: bool = True,
        detect_regions: bool = True,
        num_workers: int = 4,
        dpi: int = 300
    ):
        """
        Args:
            batch_size: Número de documentos a procesar simultáneamente
            use_gpu: Usar GPU para OCR
            detect_regions: Detectar regiones automáticamente
            num_workers: Threads para pre/post-procesamiento
            dpi: Resolución para OCR
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Crear extractor base
        self.extractor = HybridExtractorV2(
            dpi=dpi,
            use_gpu=use_gpu,
            detect_regions=detect_regions
        )

        logger.info(
            f"BatchExtractor inicializado: batch_size={batch_size}, "
            f"GPU={use_gpu}, workers={num_workers}"
        )

    def extract_batch(
        self,
        pdf_paths: List[str],
        page_num: int = 0,
        show_progress: bool = True
    ) -> Dict[str, Dict]:
        """
        Procesa batch de documentos

        Args:
            pdf_paths: Lista de rutas a PDFs
            page_num: Número de página
            show_progress: Mostrar progreso

        Returns:
            Diccionario {pdf_path: extraction_result}
        """
        start_time = time.time()
        results = {}

        total_docs = len(pdf_paths)
        logger.info(f"Procesando {total_docs} documentos en batch_size={self.batch_size}")

        # Procesar en batches
        for i in range(0, total_docs, self.batch_size):
            batch = pdf_paths[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_docs + self.batch_size - 1) // self.batch_size

            logger.info(f"Batch {batch_num}/{total_batches}: procesando {len(batch)} documentos")

            # Procesar batch en paralelo
            batch_results = self._process_batch_parallel(batch, page_num)
            results.update(batch_results)

            if show_progress:
                elapsed = time.time() - start_time
                processed = len(results)
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_docs - processed) / rate if rate > 0 else 0

                logger.info(
                    f"Progreso: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) | "
                    f"Velocidad: {rate:.1f} docs/s | ETA: {eta/60:.1f}min"
                )

        # Estadísticas finales
        total_time = time.time() - start_time
        avg_time = total_time / total_docs if total_docs > 0 else 0

        logger.info(
            f"✅ Batch completado: {total_docs} documentos en {total_time:.2f}s "
            f"({avg_time:.3f}s/doc, {total_docs/total_time:.1f} docs/s)"
        )

        return results

    def _process_batch_parallel(
        self,
        batch_paths: List[str],
        page_num: int
    ) -> Dict[str, Dict]:
        """
        Procesa un batch en paralelo usando ThreadPoolExecutor

        Args:
            batch_paths: Rutas del batch
            page_num: Número de página

        Returns:
            Resultados del batch
        """
        results = {}

        # Usar ThreadPoolExecutor para paralelizar carga
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Enviar todas las tareas
            future_to_path = {
                executor.submit(self._extract_single, pdf_path, page_num): pdf_path
                for pdf_path in batch_paths
            }

            # Recolectar resultados
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    result = future.result()
                    results[pdf_path] = result
                except Exception as e:
                    logger.error(f"Error procesando {pdf_path}: {e}")
                    results[pdf_path] = {"error": str(e)}

        return results

    def _extract_single(self, pdf_path: str, page_num: int) -> Dict:
        """
        Extrae un documento individual

        Args:
            pdf_path: Ruta al PDF
            page_num: Número de página

        Returns:
            Resultado de extracción
        """
        try:
            return self.extractor.extract(pdf_path, page_num)
        except Exception as e:
            logger.error(f"Error en {pdf_path}: {e}")
            return {"error": str(e), "words": []}

    def extract_batch_async(
        self,
        pdf_paths: List[str],
        page_num: int = 0,
        callback=None
    ):
        """
        Procesa batch de forma asíncrona con callback

        Args:
            pdf_paths: Lista de rutas a PDFs
            page_num: Número de página
            callback: Función a llamar por cada documento completado
                      callback(pdf_path, result, progress_info)
        """
        total_docs = len(pdf_paths)
        processed = 0
        start_time = time.time()

        logger.info(f"Iniciando procesamiento asíncrono de {total_docs} documentos")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {
                executor.submit(self._extract_single, pdf_path, page_num): pdf_path
                for pdf_path in pdf_paths
            }

            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                processed += 1

                try:
                    result = future.result()

                    # Calcular progreso
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total_docs - processed) / rate if rate > 0 else 0

                    progress_info = {
                        "processed": processed,
                        "total": total_docs,
                        "percentage": (processed / total_docs) * 100,
                        "elapsed": elapsed,
                        "rate": rate,
                        "eta": eta
                    }

                    # Llamar callback si existe
                    if callback:
                        callback(pdf_path, result, progress_info)

                except Exception as e:
                    logger.error(f"Error procesando {pdf_path}: {e}")
                    if callback:
                        callback(pdf_path, {"error": str(e)}, None)

        total_time = time.time() - start_time
        logger.info(
            f"✅ Procesamiento asíncrono completado: {total_docs} documentos en {total_time:.2f}s"
        )

    def extract_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        page_num: int = 0,
        max_documents: Optional[int] = None
    ) -> Dict[str, Dict]:
        """
        Procesa todos los PDFs en un directorio

        Args:
            directory: Directorio con PDFs
            pattern: Patrón de archivos (ej: "*.pdf", "FACT-*.pdf")
            page_num: Número de página
            max_documents: Máximo de documentos a procesar (None = todos)

        Returns:
            Diccionario con resultados
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {directory}")

        # Encontrar PDFs
        pdf_files = list(dir_path.glob(pattern))

        if max_documents:
            pdf_files = pdf_files[:max_documents]

        pdf_paths = [str(p) for p in pdf_files]

        logger.info(f"Encontrados {len(pdf_paths)} PDFs en {directory}")

        return self.extract_batch(pdf_paths, page_num)

    def get_stats(self) -> Dict:
        """
        Retorna estadísticas del extractor

        Returns:
            Diccionario con capacidades y configuración
        """
        return {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "gpu_available": self.extractor.get_capabilities()["gpu_available"],
            "ocr_method": self.extractor.ocr_method,
            "detect_regions": self.extractor.detect_regions
        }

    def benchmark_batch_sizes(
        self,
        pdf_path: str,
        batch_sizes: List[int] = [1, 2, 4, 8, 16],
        num_iterations: int = 10
    ) -> Dict:
        """
        Benchmark para encontrar batch_size óptimo

        Args:
            pdf_path: PDF de prueba
            batch_sizes: Lista de batch sizes a probar
            num_iterations: Iteraciones por batch size

        Returns:
            Resultados del benchmark
        """
        results = {}

        logger.info(f"Iniciando benchmark de batch sizes: {batch_sizes}")

        # Crear lista de PDFs duplicados
        pdf_paths = [pdf_path] * num_iterations

        for batch_size in batch_sizes:
            logger.info(f"Testing batch_size={batch_size}...")

            # Cambiar batch size
            original_batch_size = self.batch_size
            self.batch_size = batch_size

            # Medir tiempo
            start = time.time()
            self.extract_batch(pdf_paths, show_progress=False)
            elapsed = time.time() - start

            # Restaurar batch size
            self.batch_size = original_batch_size

            # Calcular métricas
            avg_time = elapsed / num_iterations
            throughput = num_iterations / elapsed

            results[batch_size] = {
                "total_time": round(elapsed, 3),
                "avg_per_doc": round(avg_time, 3),
                "throughput": round(throughput, 2),
                "speedup": round(results[1]["avg_per_doc"] / avg_time, 2) if 1 in results else 1.0
            }

            logger.info(
                f"  batch_size={batch_size}: {avg_time:.3f}s/doc, "
                f"{throughput:.1f} docs/s, speedup={results[batch_size]['speedup']}x"
            )

        # Encontrar mejor batch size
        best_batch_size = max(results.items(), key=lambda x: x[1]["throughput"])[0]

        logger.info(f"✅ Mejor batch_size: {best_batch_size}")

        return {
            "results": results,
            "recommended_batch_size": best_batch_size
        }
