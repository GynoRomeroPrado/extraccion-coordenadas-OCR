"""
Orquestador principal del procesamiento

Decide automáticamente si usar SinglePagePipeline o MultiPagePipeline
según el número de páginas del documento.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF

from .single_page import SinglePagePipeline
from .multipage import MultiPagePipeline
from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class InvoiceProcessor:
    """
    Orquestador principal del procesamiento

    Workflow:
    1. Detectar número de páginas
    2. Si 1 página: usar SinglePagePipeline
    3. Si 2+ páginas: usar MultiPagePipeline
    4. Generar archivos de salida
    5. Crear metadata
    6. Validar resultados
    """

    def __init__(
        self,
        dpi: int = 300,
        match_threshold: int = 85,
        auto_rotate: bool = True
    ):
        """
        Args:
            dpi: Resolución para OCR
            match_threshold: Umbral mínimo de similitud para matching
            auto_rotate: Detectar y corregir orientación automáticamente
        """
        self.dpi = dpi
        self.match_threshold = match_threshold
        self.auto_rotate = auto_rotate

        # Inicializar pipelines
        self.single_page_pipeline = SinglePagePipeline(
            dpi=dpi,
            match_threshold=match_threshold,
            auto_rotate=auto_rotate
        )

        self.multipage_pipeline = MultiPagePipeline(
            dpi=dpi,
            match_threshold=match_threshold,
            auto_rotate=auto_rotate
        )

    def process(
        self,
        pdf_path: str,
        json_path: str,
        output_dir: str,
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Procesa un documento completo

        Detecta automáticamente el número de páginas y delega al pipeline apropiado.

        Args:
            pdf_path: Ruta al archivo PDF
            json_path: Ruta al archivo JSON de anotaciones
            output_dir: Directorio de salida
            document_id: ID del documento (opcional, se extrae del nombre si no se provee)

        Returns:
            Metadata con estadísticas y resultados del procesamiento

        Raises:
            FileNotFoundError: Si el PDF o JSON no existe
            ValueError: Si el PDF está corrupto o vacío
        """
        # Validar archivos
        pdf_path = Path(pdf_path)
        json_path = Path(json_path)
        output_dir = Path(output_dir)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

        if not json_path.exists():
            raise FileNotFoundError(f"JSON no encontrado: {json_path}")

        # Extraer document_id si no se proporciona
        if document_id is None:
            document_id = pdf_path.stem

        # Detectar número de páginas
        try:
            num_pages = self._get_page_count(str(pdf_path))

            if num_pages == 0:
                raise ValueError(f"El PDF no tiene páginas: {pdf_path}")

            logger.info(f"Documento: {document_id} - {num_pages} página(s)")

        except Exception as e:
            logger.error(f"Error al abrir PDF {pdf_path}: {e}")
            raise

        # Seleccionar pipeline según número de páginas
        if num_pages == 1:
            logger.info("Usando SinglePagePipeline")
            metadata = self.single_page_pipeline.process(
                str(pdf_path),
                str(json_path),
                str(output_dir),
                document_id
            )
        else:
            logger.info("Usando MultiPagePipeline")
            metadata = self.multipage_pipeline.process(
                str(pdf_path),
                str(json_path),
                str(output_dir),
                document_id
            )

        return metadata

    def process_batch(
        self,
        pdf_dir: str,
        json_dir: str,
        output_dir: str,
        max_documents: Optional[int] = None,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Procesa un lote de documentos

        Args:
            pdf_dir: Directorio con archivos PDF
            json_dir: Directorio con archivos JSON
            output_dir: Directorio de salida
            max_documents: Máximo de documentos a procesar (None = todos)
            skip_existing: Si True, salta documentos ya procesados

        Returns:
            Diccionario con estadísticas del batch:
            {
                "total_documents": 100,
                "processed": 98,
                "failed": 2,
                "skipped": 0,
                "success_rate": 0.98,
                "avg_processing_time": 5.3,
                "failed_documents": ["FACT-000050", "FACT-001234"]
            }
        """
        pdf_dir = Path(pdf_dir)
        json_dir = Path(json_dir)
        output_dir = Path(output_dir)

        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directorio de PDFs no encontrado: {pdf_dir}")

        if not json_dir.exists():
            raise FileNotFoundError(f"Directorio de JSONs no encontrado: {json_dir}")

        # Encontrar todos los PDFs
        pdf_files = sorted(pdf_dir.glob("*.pdf"))

        if max_documents:
            pdf_files = pdf_files[:max_documents]

        logger.info(f"Encontrados {len(pdf_files)} documentos para procesar")

        # Estadísticas
        total_documents = len(pdf_files)
        processed = 0
        failed = 0
        skipped = 0
        failed_documents = []
        processing_times = []

        # Procesar cada documento
        for i, pdf_path in enumerate(pdf_files, 1):
            document_id = pdf_path.stem

            # Buscar JSON correspondiente
            json_path = json_dir / f"{document_id}.json"

            if not json_path.exists():
                logger.warning(f"JSON no encontrado para {document_id}, saltando...")
                skipped += 1
                continue

            # Verificar si ya existe la metadata (documento ya procesado)
            metadata_file = output_dir / f"{document_id}_metadata.json"

            if skip_existing and metadata_file.exists():
                logger.info(f"[{i}/{total_documents}] {document_id} ya procesado, saltando...")
                skipped += 1
                continue

            # Procesar documento
            try:
                logger.info(f"[{i}/{total_documents}] Procesando {document_id}...")

                metadata = self.process(
                    str(pdf_path),
                    str(json_path),
                    str(output_dir),
                    document_id
                )

                processing_time = metadata.get('statistics', {}).get('processing_time_seconds', 0)
                processing_times.append(processing_time)

                processed += 1

                logger.info(
                    f"[{i}/{total_documents}] ✓ {document_id} procesado en {processing_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"[{i}/{total_documents}] ✗ Error en {document_id}: {e}")
                failed += 1
                failed_documents.append(document_id)

        # Calcular estadísticas finales
        success_rate = processed / total_documents if total_documents > 0 else 0
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

        batch_stats = {
            "total_documents": total_documents,
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "success_rate": round(success_rate, 3),
            "avg_processing_time": round(avg_processing_time, 2),
            "total_processing_time": round(sum(processing_times), 2),
            "failed_documents": failed_documents
        }

        logger.info(
            f"\n{'='*60}\n"
            f"RESUMEN DEL BATCH\n"
            f"{'='*60}\n"
            f"Total de documentos: {total_documents}\n"
            f"Procesados: {processed} ({success_rate:.1%})\n"
            f"Fallidos: {failed}\n"
            f"Saltados: {skipped}\n"
            f"Tiempo promedio: {avg_processing_time:.2f}s\n"
            f"Tiempo total: {sum(processing_times):.2f}s\n"
            f"{'='*60}"
        )

        return batch_stats

    def validate_output(
        self,
        output_dir: str,
        document_id: str
    ) -> bool:
        """
        Valida que la salida sea correcta

        Args:
            output_dir: Directorio de salida
            document_id: ID del documento

        Returns:
            True si la validación pasa
        """
        # Delegar a pipeline de página simple
        # (la validación es la misma para ambos)
        return self.single_page_pipeline.validate_output(output_dir, document_id)

    def _get_page_count(self, pdf_path: str) -> int:
        """
        Obtiene el número de páginas del PDF

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Número de páginas

        Raises:
            Exception: Si no se puede abrir el PDF
        """
        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
            doc.close()
            return num_pages
        except Exception as e:
            logger.error(f"Error al obtener número de páginas: {e}")
            raise

    @staticmethod
    def get_default_config() -> Config:
        """
        Retorna la configuración por defecto

        Returns:
            Instancia de Config
        """
        return Config
