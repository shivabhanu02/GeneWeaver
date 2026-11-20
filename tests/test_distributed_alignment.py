
import pytest

from src.distributed_alignment import distributed_chunk_alignment
from src.alignment import smith_waterman


@pytest.fixture
def sample_record():
    return {
        "id": "chromosome_1",
        "sequence": "ACGTACGTACGT",
    }


def test_distributed_alignment_returns_all_chunks(sample_record):
    results = distributed_chunk_alignment(
        sample_record,
        target_sequence="ACGT",
        chunk_size=4,
    )

    assert len(results) == 3


def test_distributed_alignment_preserves_record_id(sample_record):
    results = distributed_chunk_alignment(
        sample_record,
        target_sequence="ACGT",
        chunk_size=4,
    )

    assert all(
        result["record_id"] == "chromosome_1"
        for result in results
    )


def test_distributed_alignment_preserves_coordinates(sample_record):
    results = distributed_chunk_alignment(
        sample_record,
        target_sequence="ACGT",
        chunk_size=4,
    )

    assert results[0]["start"] == 0
    assert results[0]["end"] == 4

    assert results[1]["start"] == 4
    assert results[1]["end"] == 8

    assert results[2]["start"] == 8
    assert results[2]["end"] == 12


def test_distributed_alignment_results_are_ordered(sample_record):
    results = distributed_chunk_alignment(
        sample_record,
        target_sequence="ACGT",
        chunk_size=4,
    )

    assert [r["chunk_id"] for r in results] == [0, 1, 2]


def test_distributed_alignment_matches_cpu(sample_record):
    results = distributed_chunk_alignment(
        sample_record,
        target_sequence="ACGT",
        chunk_size=4,
    )

    for result in results:
        expected = smith_waterman(
            sample_record["sequence"][
                result["start"]:result["end"]
            ],
            "ACGT",
        )

        assert result["alignment"]["score"] == expected["score"]


def test_distributed_alignment_with_overlap(sample_record):
    results = distributed_chunk_alignment(
        sample_record,
        target_sequence="ACGT",
        chunk_size=6,
        overlap=2,
    )

    # Sequence length = 12
    # Chunk size = 6
    # Overlap = 2
    # Step = 6 - 2 = 4
    # Expected chunks: [0:6], [4:10], [8:12]

    assert len(results) == 3

    assert [result["chunk_id"] for result in results] == [0, 1, 2]

    assert [
        (result["start"], result["end"])
        for result in results
    ] == [
        (0, 6),
        (4, 10),
        (8, 12),
    ]

    assert [
        result["end"] - result["start"]
        for result in results
    ] == [6, 6, 4]


def test_invalid_target_sequence():
    record = {
        "id": "chromosome_1",
        "sequence": "ACGTACGT",
    }

    with pytest.raises(ValueError):
        distributed_chunk_alignment(
            record,
            target_sequence="",
            chunk_size=4,
        )


def test_invalid_record():
    with pytest.raises(ValueError):
        distributed_chunk_alignment(
            {"id": "chromosome_1"},
            target_sequence="ACGT",
            chunk_size=4,
        )
