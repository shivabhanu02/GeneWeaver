
import pytest

from src.parser import (
    parse_fasta,
    validate_dna_sequence
)


def test_valid_dna_sequence():
    assert validate_dna_sequence("ATCG")
    assert validate_dna_sequence("ATCGN")


def test_lowercase_dna_sequence():
    assert validate_dna_sequence("atcg")


def test_invalid_dna_sequence():
    assert not validate_dna_sequence("ATCX")
    assert not validate_dna_sequence("ATG123")


def test_empty_dna_sequence():
    assert not validate_dna_sequence("")


def test_parse_single_fasta_record(tmp_path):
    fasta_file = tmp_path / "sample.fasta"

    fasta_file.write_text(
        ">gene1 Sample DNA sequence\nATCGATCG\n"
    )

    records = list(parse_fasta(fasta_file))

    assert len(records) == 1
    assert records[0]["id"] == "gene1"
    assert records[0]["sequence"] == "ATCGATCG"


def test_parse_multiple_fasta_records(tmp_path):
    fasta_file = tmp_path / "multiple.fasta"

    fasta_file.write_text(
        ">gene1\nATCG\n"
        ">gene2\nGGCA\n"
    )

    records = list(parse_fasta(fasta_file))

    assert len(records) == 2
    assert records[0]["id"] == "gene1"
    assert records[1]["id"] == "gene2"


def test_missing_fasta_file():
    with pytest.raises(FileNotFoundError):
        list(parse_fasta("missing.fasta"))


def test_empty_fasta_file(tmp_path):
    fasta_file = tmp_path / "empty.fasta"
    fasta_file.write_text("")

    with pytest.raises(ValueError):
        list(parse_fasta(fasta_file))


def test_invalid_fasta_sequence(tmp_path):
    fasta_file = tmp_path / "invalid.fasta"

    fasta_file.write_text(
        ">gene1\nATCX123\n"
    )

    with pytest.raises(ValueError):
        list(parse_fasta(fasta_file))
