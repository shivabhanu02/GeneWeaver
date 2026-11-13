
import pytest

from src.alignment import smith_waterman


def test_identical_sequences():
    result = smith_waterman("ACGT", "ACGT")

    assert result["score"] == 8
    assert result["aligned_sequence1"] == "ACGT"
    assert result["aligned_sequence2"] == "ACGT"


def test_local_alignment():
    result = smith_waterman("TTACGTAA", "GGACGTCC")

    assert result["score"] == 8
    assert result["aligned_sequence1"] == "ACGT"
    assert result["aligned_sequence2"] == "ACGT"


def test_mismatch():
    result = smith_waterman("ACGT", "AGGT")

    assert result["score"] > 0


def test_empty_sequence():
    with pytest.raises(ValueError):
        smith_waterman("", "ACGT")


def test_invalid_dna_sequence():
    with pytest.raises(ValueError):
        smith_waterman("ACGTX", "ACGT")


def test_lowercase_sequences():
    result = smith_waterman("acgt", "acgt")

    assert result["score"] == 8
    assert result["aligned_sequence1"] == "ACGT"
    assert result["aligned_sequence2"] == "ACGT"
