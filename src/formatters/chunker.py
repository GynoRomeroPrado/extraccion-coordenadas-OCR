"""
Chunker para dividir documentos largos en chunks de máximo 512 tokens

Divide documentos que exceden el límite de tokens de LayoutLMv3
aplicando overlap entre chunks para mantener contexto.
"""
import logging
from typing import List, Dict, Optional

from config import Config

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class TokenChunker:
    """
    Divide documentos en chunks de máximo N tokens

    Estrategia:
    1. Estimar tokens por elemento (1 word ≈ 2 tokens)
    2. Dividir en chunks de max_tokens (default: 384 para margen de seguridad)
    3. Aplicar overlap de tokens entre chunks para contexto
    4. Generar múltiples archivos si es necesario
    """

    def __init__(
        self,
        max_tokens: int = 384,
        overlap_tokens: int = 128,
        tokens_per_word: float = 2.0
    ):
        """
        Args:
            max_tokens: Máximo de tokens por chunk (LayoutLMv3 acepta 512)
            overlap_tokens: Tokens de overlap entre chunks
            tokens_per_word: Estimación de tokens por palabra
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokens_per_word = tokens_per_word

        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens debe ser menor que max_tokens")

    def estimate_tokens(self, elements: List[Dict]) -> int:
        """
        Estima el número total de tokens de un documento

        Args:
            elements: Lista de elementos del documento

        Returns:
            Número estimado de tokens
        """
        total_tokens = 0

        for elem in elements:
            # Contar palabras en el elemento
            words = elem.get('words', [])
            word_count = len(words) if words else len(elem.get('text', '').split())

            # Estimar tokens (incluyendo tokens especiales)
            element_tokens = int(word_count * self.tokens_per_word)
            element_tokens += 4  # Tokens especiales por elemento (ej: [CLS], [SEP])

            total_tokens += element_tokens

        return total_tokens

    def needs_chunking(self, elements: List[Dict]) -> bool:
        """
        Determina si el documento necesita ser dividido en chunks

        Args:
            elements: Lista de elementos del documento

        Returns:
            True si excede max_tokens, False en caso contrario
        """
        estimated_tokens = self.estimate_tokens(elements)
        needs_split = estimated_tokens > self.max_tokens

        if needs_split:
            logger.info(
                f"Documento excede límite de tokens: {estimated_tokens} > {self.max_tokens}"
            )

        return needs_split

    def chunk_elements(self, elements: List[Dict]) -> List[List[Dict]]:
        """
        Divide elementos en chunks

        Args:
            elements: Lista de elementos del documento

        Returns:
            Lista de chunks, donde cada chunk es una lista de elementos:
            [
                [elem1, elem2, elem3],  # Chunk 0
                [elem3, elem4, elem5],  # Chunk 1 (con overlap)
                ...
            ]
        """
        if not elements:
            return []

        # Verificar si necesita chunking
        if not self.needs_chunking(elements):
            logger.debug("Documento no necesita chunking")
            return [elements]

        chunks = []
        current_chunk = []
        current_tokens = 0

        # Variables para overlap
        overlap_elements = []
        overlap_tokens = 0

        for elem in elements:
            # Estimar tokens del elemento
            words = elem.get('words', [])
            word_count = len(words) if words else len(elem.get('text', '').split())
            elem_tokens = int(word_count * self.tokens_per_word) + 4

            # Si agregar este elemento excede el límite, crear nuevo chunk
            if current_tokens + elem_tokens > self.max_tokens and current_chunk:
                # Guardar chunk actual
                chunks.append(current_chunk)

                # Iniciar nuevo chunk con overlap
                current_chunk = overlap_elements.copy()
                current_tokens = overlap_tokens

                # Limpiar overlap para siguiente chunk
                overlap_elements = []
                overlap_tokens = 0

            # Agregar elemento al chunk actual
            current_chunk.append(elem)
            current_tokens += elem_tokens

            # Mantener elementos para overlap del siguiente chunk
            overlap_elements.append(elem)
            overlap_tokens += elem_tokens

            # Si overlap excede límite, remover elementos más antiguos
            while overlap_tokens > self.overlap_tokens and len(overlap_elements) > 1:
                removed_elem = overlap_elements.pop(0)
                removed_words = removed_elem.get('words', [])
                removed_word_count = len(removed_words) if removed_words else len(removed_elem.get('text', '').split())
                removed_tokens = int(removed_word_count * self.tokens_per_word) + 4
                overlap_tokens -= removed_tokens

        # Agregar último chunk si no está vacío
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"Documento dividido en {len(chunks)} chunks "
            f"(overlap: {self.overlap_tokens} tokens)"
        )

        return chunks

    def chunk_document(self, document: Dict) -> List[Dict]:
        """
        Divide un documento completo en chunks

        Args:
            document: Documento en formato LayoutLMv3:
                {
                    "form": [elementos]
                }

        Returns:
            Lista de documentos (chunks):
            [
                {"form": [elementos chunk 0]},
                {"form": [elementos chunk 1]},
                ...
            ]
        """
        elements = document.get('form', [])

        if not elements:
            logger.warning("Documento vacío")
            return [document]

        # Dividir elementos en chunks
        chunks = self.chunk_elements(elements)

        # Crear documentos para cada chunk
        chunked_documents = []
        for i, chunk_elements in enumerate(chunks):
            chunked_doc = {
                "form": chunk_elements,
                "chunk_info": {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "num_elements": len(chunk_elements),
                    "estimated_tokens": self.estimate_tokens(chunk_elements)
                }
            }
            chunked_documents.append(chunked_doc)

        return chunked_documents

    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """
        Calcula estadísticas de los chunks

        Args:
            chunks: Lista de chunks (documentos)

        Returns:
            Diccionario con estadísticas
        """
        if not chunks:
            return {
                "num_chunks": 0,
                "total_elements": 0,
                "total_tokens_estimated": 0,
                "elements_per_chunk": [],
                "tokens_per_chunk": []
            }

        elements_per_chunk = []
        tokens_per_chunk = []

        for chunk in chunks:
            elements = chunk.get('form', [])
            elements_per_chunk.append(len(elements))
            tokens_per_chunk.append(self.estimate_tokens(elements))

        return {
            "num_chunks": len(chunks),
            "total_elements": sum(elements_per_chunk),
            "total_tokens_estimated": sum(tokens_per_chunk),
            "elements_per_chunk": elements_per_chunk,
            "tokens_per_chunk": tokens_per_chunk,
            "avg_elements_per_chunk": sum(elements_per_chunk) / len(chunks),
            "avg_tokens_per_chunk": sum(tokens_per_chunk) / len(chunks),
            "max_tokens_in_chunk": max(tokens_per_chunk)
        }

    def validate_chunks(self, chunks: List[Dict]) -> bool:
        """
        Valida que los chunks sean correctos

        Args:
            chunks: Lista de chunks

        Returns:
            True si todos los chunks son válidos
        """
        if not chunks:
            logger.error("No hay chunks para validar")
            return False

        for i, chunk in enumerate(chunks):
            elements = chunk.get('form', [])

            if not elements:
                logger.error(f"Chunk {i} está vacío")
                return False

            # Verificar que no exceda límite de tokens
            tokens = self.estimate_tokens(elements)
            if tokens > self.max_tokens:
                logger.error(
                    f"Chunk {i} excede límite de tokens: {tokens} > {self.max_tokens}"
                )
                return False

        logger.debug(f"Validación exitosa: {len(chunks)} chunks válidos")
        return True

    def reassemble_chunks(self, chunks: List[Dict]) -> Dict:
        """
        Reensambla chunks en un documento único (inverso de chunk_document)

        Args:
            chunks: Lista de chunks

        Returns:
            Documento completo
        """
        if not chunks:
            return {"form": []}

        if len(chunks) == 1:
            # Si hay un solo chunk, retornar directamente (sin chunk_info)
            doc = chunks[0].copy()
            doc.pop('chunk_info', None)
            return doc

        # Combinar elementos de todos los chunks, eliminando duplicados por overlap
        all_elements = []
        seen_ids = set()

        for chunk in chunks:
            elements = chunk.get('form', [])

            for elem in elements:
                elem_id = elem.get('id')

                # Si no tiene ID o no lo hemos visto, agregarlo
                if elem_id is None or elem_id not in seen_ids:
                    all_elements.append(elem)
                    if elem_id is not None:
                        seen_ids.add(elem_id)

        logger.info(
            f"Reensamblados {len(chunks)} chunks en {len(all_elements)} elementos "
            f"(eliminados {sum(len(c.get('form', [])) for c in chunks) - len(all_elements)} duplicados)"
        )

        return {"form": all_elements}
