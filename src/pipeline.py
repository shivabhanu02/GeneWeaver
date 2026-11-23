
"""
GeneWeaver - End-to-End DNA Analysis Pipeline.

Connects FASTA parsing, Dask-based chunk alignment,
genomic coordinate mapping, and guide RNA matching.
"""

from pathlib import Path

from src.parser import parse_fasta
from src.distributed_alignment import distributed_chunk_alignment
from src.guide_rna import find_exact_matches


VALID_ALIGNMENT_BASES = set("ACGT")


def _validate_alignment_sequence(sequence: str, name: str) -> str:
    """Validate a sequence for the current alignment algorithms."""

    if not isinstance(sequence, str) or not sequence:
        raise ValueError(f"{name} must be a non-empty DNA sequence.")

    sequence = sequence.upper()

    if not set(sequence).issubset(VALID_ALIGNMENT_BASES):
        raise ValueError(
            f"{name} must contain only A, C, G, and T. "
            "Ambiguous IUPAC bases are not supported by the "
            "current alignment algorithms."
        )

    return sequence


def run_pipeline(
    fasta_path: str | Path,
    target_sequence: str,
    chunk_size: int,
    overlap: int = 0,
    guide_rna: str | None = None,
    client=None,
) -> dict:
    """
    Run the GeneWeaver analysis pipeline.

    Processes each FASTA record, aligns its chunks against
    the target sequence, converts alignment coordinates to
    genomic coordinates, and optionally searches for guide RNA.

    Returns a dictionary containing record-level results
    and an overall summary.
    """

    target_sequence = _validate_alignment_sequence(
        target_sequence,
        "Target sequence",
    )

    if guide_rna is not None and (
        not isinstance(guide_rna, str) or not guide_rna
    ):
        raise ValueError("Guide RNA must be a non-empty string.")

    processed_records = []
    total_chunks = 0
    total_guide_matches = 0

    for record in parse_fasta(fasta_path):

        sequence = _validate_alignment_sequence(
            record["sequence"],
            f"FASTA record '{record['id']}'",
        )

        # Align genomic chunks using Dask.
        alignments = distributed_chunk_alignment(
            record=record,
            target_sequence=target_sequence,
            chunk_size=chunk_size,
            overlap=overlap,
            client=client,
        )

        # Convert chunk-relative alignment coordinates
        # into original genomic coordinates.
        for result in alignments:
            alignment = result["alignment"]

            alignment["start1"] += result["start"]
            alignment["end1"] += result["start"]

        # Search the complete genomic sequence for guide RNA.
        if guide_rna is not None:
            guide_matches = find_exact_matches(
                guide_rna,
                sequence,
            )
        else:
            guide_matches = []

        total_chunks += len(alignments)
        total_guide_matches += len(guide_matches)

        processed_records.append({
            "record_id": record["id"],
            "description": record["description"],
            "sequence_length": len(sequence),
            "alignments": alignments,
            "guide_matches": guide_matches,
        })

    return {
        "records": processed_records,
        "summary": {
            "records_processed": len(processed_records),
            "chunks_aligned": total_chunks,
            "guide_matches_found": total_guide_matches,
        },
    }
