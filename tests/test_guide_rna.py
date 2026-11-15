
import pytest

from src.guide_rna import (
    validate_guide_rna,
    validate_genome_sequence,
    reverse_complement,
    find_exact_matches,
)


def test_valid_guide_rna():
    assert validate_guide_rna("ACGU") is True


def test_invalid_guide_rna():
    assert validate_guide_rna("ACGTX") is False


def test_valid_genome_sequence():
    assert validate_genome_sequence("ACGTACGT") is True


def test_invalid_genome_sequence():
    assert validate_genome_sequence("ACGTX") is False


def test_reverse_complement():
    assert reverse_complement("ACGT") == "ACGT"


def test_exact_forward_match():
    result = find_exact_matches("ACGU", "TTACGTAA")

    assert len(result) == 1
    assert result[0]["start"] == 2
    assert result[0]["end"] == 6
    assert result[0]["strand"] == "+"


def test_exact_reverse_match():
    result = find_exact_matches("AAGC", "GCTT")

    assert len(result) == 1
    assert result[0]["start"] == 0
    assert result[0]["end"] == 4
    assert result[0]["strand"] == "-"


def test_multiple_forward_matches():
    result = find_exact_matches("ACGU", "ACGTACGT")

    forward_matches = [
        match for match in result
        if match["strand"] == "+"
    ]

    assert len(forward_matches) == 2


def test_empty_guide_rna():
    with pytest.raises(ValueError):
        find_exact_matches("", "ACGT")


def test_empty_genome():
    with pytest.raises(ValueError):
        find_exact_matches("ACGU", "")


def test_lowercase_sequences():
    result = find_exact_matches("acgu", "ttacgtaa")

    assert len(result) == 1
    assert result[0]["strand"] == "+"
