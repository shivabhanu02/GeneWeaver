
"""
GeneWeaver - Distributed DNA Sequence Alignment.

Uses Dask to schedule independent Smith-Waterman
alignments across genomic sequence chunks.
"""

from dask import compute, delayed

from src.alignment import smith_waterman
from src.chunking import chunk_fasta_record


def _align_chunk(
    chunk: dict,
    target_sequence: str,
    match_score: int,
    mismatch_score: int,
    gap_penalty: int,
) -> dict:
    """Align one genomic chunk against the target sequence."""

    alignment = smith_waterman(
        chunk["sequence"],
        target_sequence,
        match_score,
        mismatch_score,
        gap_penalty,
    )

    return {
        "record_id": chunk["record_id"],
        "chunk_id": chunk["chunk_id"],
        "start": chunk["start"],
        "end": chunk["end"],
        "alignment": alignment,
    }


def distributed_chunk_alignment(
    record: dict,
    target_sequence: str,
    chunk_size: int,
    overlap: int = 0,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
    client=None,
) -> list:
    """
    Align genomic chunks in parallel using Dask.

    If a Dask Distributed Client is supplied, tasks are
    submitted to that client. Otherwise, Dask's threaded
    scheduler is used locally.

    Results are returned in original chunk order.
    """

    if not isinstance(target_sequence, str) or not target_sequence:
        raise ValueError("Target sequence must be a non-empty DNA string.")

    target_sequence = target_sequence.upper()

    # Generate chunks using the existing project implementation.
    chunks = list(
        chunk_fasta_record(
            record,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    )

    if not chunks:
        return []

    # Create one independent Dask task per chunk.
    tasks = [
        delayed(_align_chunk)(
            chunk,
            target_sequence,
            match_score,
            mismatch_score,
            gap_penalty,
        )
        for chunk in chunks
    ]

    # Execute tasks using the selected scheduler.
    if client is not None:
        futures = client.compute(tasks)
        results = client.gather(futures)
    else:
        results = list(compute(*tasks, scheduler="threads"))

    # Ensure deterministic genomic ordering.
    results.sort(key=lambda result: result["chunk_id"])

    return results
