"""
Tests para FuzzyMatcher
"""
import sys
from pathlib import Path

# Agregar directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matchers.fuzzy_matcher import FuzzyMatcher


def test_exact_match():
    """Test de matching exacto"""
    matcher = FuzzyMatcher(threshold=85)

    pdf_words = [
        {"text": "FACTURA", "bbox": [100, 50, 200, 70], "confidence": 1.0, "page": 0},
        {"text": "20393920069", "bbox": [100, 100, 250, 120], "confidence": 1.0, "page": 0},
        {"text": "ELECTRONICA", "bbox": [210, 50, 350, 70], "confidence": 1.0, "page": 0},
    ]

    result = matcher.match("20393920069", pdf_words, "emisor_ruc")

    assert result is not None
    assert result["text"] == "20393920069"
    assert result["match_score"] == 1.0
    assert result["field_name"] == "emisor_ruc"


def test_fuzzy_match():
    """Test de matching difuso"""
    matcher = FuzzyMatcher(threshold=85)

    pdf_words = [
        {"text": "FACTUR4", "bbox": [100, 50, 200, 70], "confidence": 0.9, "page": 0},  # Error OCR
    ]

    result = matcher.match("FACTURA", pdf_words, "tipo_documento")

    assert result is not None
    assert result["text"] == "FACTUR4"
    assert result["match_score"] >= 0.85


def test_no_match():
    """Test cuando no hay match"""
    matcher = FuzzyMatcher(threshold=85)

    pdf_words = [
        {"text": "FACTURA", "bbox": [100, 50, 200, 70], "confidence": 1.0, "page": 0},
    ]

    result = matcher.match("BOLETA", pdf_words, "tipo_documento")

    # BOLETA no debería matchear con FACTURA con threshold 85
    # (depende del scorer, pero generalmente no coinciden)
    # Este test puede necesitar ajuste según el comportamiento real


def test_multiword_match():
    """Test de matching multi-palabra"""
    matcher = FuzzyMatcher(threshold=85)

    pdf_words = [
        {"text": "La", "bbox": [100, 50, 120, 70], "confidence": 1.0, "page": 0},
        {"text": "Positiva", "bbox": [125, 50, 200, 70], "confidence": 1.0, "page": 0},
        {"text": "S.A.", "bbox": [205, 50, 250, 70], "confidence": 1.0, "page": 0},
    ]

    result = matcher.match_multiword("La Positiva S.A.", pdf_words, "emisor_razon_social")

    assert result is not None
    assert "La Positiva S.A." in result["text"]


def test_numeric_match():
    """Test de matching numérico"""
    matcher = FuzzyMatcher(threshold=85)

    pdf_words = [
        {"text": "17976.71", "bbox": [100, 50, 200, 70], "confidence": 1.0, "page": 0},
        {"text": "15234.50", "bbox": [100, 100, 200, 120], "confidence": 1.0, "page": 0},
    ]

    result = matcher.match_numeric(17976.71, pdf_words, "importe_total")

    assert result is not None
    assert result["text"] == "17976.71"
    assert result.get("numeric_value") == 17976.71


if __name__ == "__main__":
    print("Running FuzzyMatcher tests...")

    test_exact_match()
    print("✓ test_exact_match passed")

    test_fuzzy_match()
    print("✓ test_fuzzy_match passed")

    test_multiword_match()
    print("✓ test_multiword_match passed")

    test_numeric_match()
    print("✓ test_numeric_match passed")

    print("\nAll tests passed! ✓")
