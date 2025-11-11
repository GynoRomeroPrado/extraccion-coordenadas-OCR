"""
Tests para LayoutLMv3Formatter
"""
import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.formatters.layoutlmv3_formatter import LayoutLMv3Formatter


def test_format_field():
    """Test de formateo de campo individual"""
    formatter = LayoutLMv3Formatter()

    match_result = {
        "text": "FACTURA ELECTRONICA",
        "bbox": [120, 45, 380, 72],
        "confidence": 0.95,
        "page": 0
    }

    formatted = formatter.format_field("tipo_documento", match_result, "answer")

    assert formatted is not None
    assert formatted["text"] == "FACTURA ELECTRONICA"
    assert formatted["box"] == [120, 45, 380, 72]
    assert formatted["label"] == "answer"
    assert formatted["field_name"] == "tipo_documento"
    assert formatted["confidence"] == 0.95
    assert "words" in formatted
    assert len(formatted["words"]) >= 1


def test_create_document():
    """Test de creación de documento completo"""
    formatter = LayoutLMv3Formatter()

    elements = [
        {
            "id": 0,
            "text": "FACTURA",
            "box": [100, 50, 200, 70],
            "label": "answer",
            "field_name": "tipo_documento",
            "words": [{"text": "FACTURA", "box": [100, 50, 200, 70]}],
            "page": 0,
            "confidence": 1.0
        },
        {
            "id": 1,
            "text": "20393920069",
            "box": [100, 100, 250, 120],
            "label": "answer",
            "field_name": "emisor_ruc",
            "words": [{"text": "20393920069", "box": [100, 100, 250, 120]}],
            "page": 0,
            "confidence": 1.0
        }
    ]

    document = formatter.create_document(elements, sort_by_position=True)

    assert "form" in document
    assert len(document["form"]) == 2
    assert document["form"][0]["id"] == 0
    assert document["form"][1]["id"] == 1


def test_get_stats():
    """Test de cálculo de estadísticas"""
    formatter = LayoutLMv3Formatter()

    document = {
        "form": [
            {
                "id": 0,
                "text": "FACTURA",
                "box": [100, 50, 200, 70],
                "label": "answer",
                "field_name": "tipo_documento",
                "words": [{"text": "FACTURA", "box": [100, 50, 200, 70]}],
                "page": 0,
                "confidence": 0.95
            },
            {
                "id": 1,
                "text": "ELECTRONICA",
                "box": [210, 50, 350, 70],
                "label": "answer",
                "field_name": "tipo_documento",
                "words": [{"text": "ELECTRONICA", "box": [210, 50, 350, 70]}],
                "page": 0,
                "confidence": 0.90
            }
        ]
    }

    stats = formatter.get_stats(document)

    assert stats["num_elements"] == 2
    assert stats["num_words"] == 2
    assert stats["avg_confidence"] == 0.925
    assert stats["pages"] == [0]
    assert stats["num_pages"] == 1


def test_validate_bbox():
    """Test de validación de coordenadas"""
    formatter = LayoutLMv3Formatter()

    match_result = {
        "text": "TEST",
        "bbox": [0, 0, 1000, 1000],  # Límites válidos
        "confidence": 1.0,
        "page": 0
    }

    formatted = formatter.format_field("test_field", match_result)

    assert formatted is not None
    assert all(0 <= coord <= 1000 for coord in formatted["box"])


if __name__ == "__main__":
    print("Running LayoutLMv3Formatter tests...")

    test_format_field()
    print("✓ test_format_field passed")

    test_create_document()
    print("✓ test_create_document passed")

    test_get_stats()
    print("✓ test_get_stats passed")

    test_validate_bbox()
    print("✓ test_validate_bbox passed")

    print("\nAll tests passed! ✓")
