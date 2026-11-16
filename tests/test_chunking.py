
import pytest

from src.chunking import chunk_sequence, chunk_fasta_record


def test_basic_chunking():
    chunks = list(chunk_sequence("ACGTACGT", chunk_size=4))

    assert len(chunks) == 2
    assert chunks[0]["sequence"] == "ACGT"
    assert chunks[1]["sequence"] == "ACGT"


def test_chunk_coordinates():
    chunks = list(chunk_sequence("ACGTACGT", chunk_size=4))

    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == 4
    assert chunks[1]["start"] == 4
    assert chunks[1]["end"] == 8


def test_overlapping_chunks():
    chunks = list(
        chunk_sequence("ACGTACGTAA", chunk_size=6, overlap=2)
    )

    assert len(chunks) == 2
    assert chunks[0]["sequence"] == "ACGTAC"
    assert chunks[1]["sequence"] == "ACGTAA"
    assert chunks[1]["start"] == 4
    assert chunks[1]["end"] == 10


def test_overlapping_chunk_coordinates():
    sequence = "ACGTACGTAA"
    chunks = list(chunk_sequence(sequence, chunk_size=6, overlap=2))

    for chunk in chunks:
        assert chunk["sequence"] == sequence[
            chunk["start"]:chunk["end"]
        ]


def test_sequence_shorter_than_chunk_size():
    chunks = list(chunk_sequence("ACGT", chunk_size=10))

    assert len(chunks) == 1
    assert chunks[0]["sequence"] == "ACGT"
    assert chunks[0]["start"] == 0
    assert chunks[0]["end"] == 4


def test_chunk_fasta_record():
    record = {
        "id": "chromosome_1",
        "sequence": "ACGTACGT",
    }

    chunks = list(chunk_fasta_record(record, chunk_size=4))

    assert len(chunks) == 2
    assert chunks[0]["record_id"] == "chromosome_1"
    assert chunks[1]["record_id"] == "chromosome_1"


def test_invalid_dna_sequence():
    with pytest.raises(ValueError):
        list(chunk_sequence("ACGTXYZ", chunk_size=4))


def test_empty_sequence():
    with pytest.raises(ValueError):
        list(chunk_sequence("", chunk_size=4))


def test_zero_chunk_size():
    with pytest.raises(ValueError):
        list(chunk_sequence("ACGT", chunk_size=0))


def test_negative_chunk_size():
    with pytest.raises(ValueError):
        list(chunk_sequence("ACGT", chunk_size=-1))


def test_negative_overlap():
    with pytest.raises(ValueError):
        list(chunk_sequence("ACGT", chunk_size=4, overlap=-1))


def test_overlap_equal_to_chunk_size():
    with pytest.raises(ValueError):
        list(chunk_sequence("ACGT", chunk_size=4, overlap=4))


def test_non_integer_chunk_size():
    with pytest.raises(TypeError):
        list(chunk_sequence("ACGT", chunk_size=2.5))


def test_invalid_fasta_record():
    with pytest.raises(ValueError):
        list(chunk_fasta_record({"id": "chromosome_1"}, 4))
