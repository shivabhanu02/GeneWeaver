
"""
GeneWeaver - Genomic Sequence Chunking

Split genomic DNA sequences into manageable chunks
while preserving their original genomic coordinates.
"""

from typing import Iterator

from src.parser import validate_dna_sequence


def chunk_sequence(
    sequence: str,
    chunk_size: int,
    overlap: int = 0,
) -> Iterator[dict]:
    """
    Split a DNA sequence into overlapping chunks.

    Coordinates use 0-based, half-open indexing.
    Avoids generating redundant trailing chunks.
    """

    if not isinstance(sequence, str):
        raise TypeError("Sequence must be a string.")

    sequence = sequence.upper()

    if not validate_dna_sequence(sequence):
        raise ValueError("Sequence must contain valid DNA bases.")

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("Chunk size must be an integer.")

    if not isinstance(overlap, int) or isinstance(overlap, bool):
        raise TypeError("Overlap must be an integer.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0:
        raise ValueError("Overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    step = chunk_size - overlap
    chunk_id = 0

    for start in range(0, len(sequence), step):
        end = min(start + chunk_size, len(sequence))

        yield {
            "chunk_id": chunk_id,
            "start": start,
            "end": end,
            "sequence": sequence[start:end],
        }

        chunk_id += 1

        # Stop when the complete sequence has been covered.
        if end == len(sequence):
            break


def chunk_fasta_record(
    record: dict,
    chunk_size: int,
    overlap: int = 0,
) -> Iterator[dict]:
    """Chunk a FASTA record while preserving its ID."""

    if not isinstance(record, dict):
        raise TypeError("Record must be a dictionary.")

    if "id" not in record or "sequence" not in record:
        raise ValueError("Record must contain id and sequence.")

    for chunk in chunk_sequence(
        record["sequence"],
        chunk_size,
        overlap,
    ):
        yield {
            "record_id": record["id"],
            **chunk,
        }
