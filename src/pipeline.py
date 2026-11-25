
"""
GeneWeaver - End-to-End DNA Analysis Pipeline.

Connects FASTA parsing, CPU/Dask or optional CUDA alignment,
genomic coordinate mapping, and guide RNA matching.

Numba is optional for CPU-only environments.
"""

from pathlib import Path

from src.parser import parse_fasta
from src.chunking import chunk_fasta_record
from src.distributed_alignment import distributed_chunk_alignment
from src.guide_rna import find_exact_matches


VALID_ALIGNMENT_BASES = set("ACGT")

# Keep synchronized with src.cuda_alignment.MAX_SEQUENCE_LENGTH.
MAX_SEQUENCE_LENGTH = 1023


def _validate_alignment_sequence(sequence: str, name: str) -> str:
    """Validate a DNA sequence for the current alignment algorithms."""

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


def _cuda_available() -> bool:
    """Check CUDA availability without requiring Numba for CPU execution."""

    try:
        from numba import cuda
        return cuda.is_available()
    except ImportError:
        return False


def _gpu_alignment_supported(chunks: list, target_sequence: str) -> bool:
    """Check whether the inputs fit the current CUDA implementation."""

    if len(target_sequence) > MAX_SEQUENCE_LENGTH:
        return False

    return all(
        len(chunk["sequence"]) <= MAX_SEQUENCE_LENGTH
        for chunk in chunks
    )


def _run_gpu_alignment(chunks: list, target_sequence: str) -> list:
    """Run the existing CUDA batch alignment implementation."""

    # Import only when GPU execution is actually requested.
    from src.cuda_alignment import batch_smith_waterman_gpu

    candidate_sequences = [
        chunk["sequence"] for chunk in chunks
    ]

    gpu_results = batch_smith_waterman_gpu(
        reference_sequence=target_sequence,
        candidate_sequences=candidate_sequences,
    )

    if len(gpu_results) != len(chunks):
        raise RuntimeError(
            "CUDA returned an unexpected number of alignment results."
        )

    results = []

    for chunk, gpu_result in zip(chunks, gpu_results):
        # CUDA sequence 1 is the target; sequence 2 is the genomic chunk.
        # Swap them to preserve the CPU result convention.
        alignment = {
            "score": gpu_result["gpu_score"],
            "aligned_sequence1": gpu_result["aligned_sequence2"],
            "aligned_sequence2": gpu_result["aligned_sequence1"],
            "start1": gpu_result["start2"],
            "start2": gpu_result["start1"],
            "end1": gpu_result["end2"],
            "end2": gpu_result["end1"],
        }

        results.append({
            "record_id": chunk["record_id"],
            "chunk_id": chunk["chunk_id"],
            "start": chunk["start"],
            "end": chunk["end"],
            "alignment": alignment,
        })

    return results


def run_pipeline(
    fasta_path: str | Path,
    target_sequence: str,
    chunk_size: int,
    overlap: int = 0,
    guide_rna: str | None = None,
    client=None,
    execution_backend: str = "auto",
) -> dict:
    """
    Run the GeneWeaver DNA analysis pipeline.

    execution_backend:
        auto - use CUDA when available and supported;
               otherwise use CPU/Dask.
        cpu  - always use CPU/Dask.
        gpu  - require CUDA GPU execution.

    Returns record-level results and an overall summary.
    """

    if execution_backend not in {"auto", "cpu", "gpu"}:
        raise ValueError(
            "execution_backend must be 'auto', 'cpu', or 'gpu'."
        )

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
    used_backends = set()

    cuda_available = (
        execution_backend != "cpu" and _cuda_available()
    )

    if execution_backend == "gpu" and not cuda_available:
        raise RuntimeError(
            "GPU execution was requested, but Numba or CUDA "
            "is not available in this environment."
        )

    for record in parse_fasta(fasta_path):

        sequence = _validate_alignment_sequence(
            record["sequence"],
            f"FASTA record '{record['id']}'",
        )

        # Generate chunks once so CPU and GPU use identical boundaries.
        chunks = list(
            chunk_fasta_record(
                record,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

        gpu_supported = _gpu_alignment_supported(
            chunks,
            target_sequence,
        )

        use_gpu = cuda_available and gpu_supported

        if execution_backend == "gpu" and not gpu_supported:
            raise ValueError(
                "GPU execution supports sequences up to "
                f"{MAX_SEQUENCE_LENGTH} bases. Reduce the target "
                "sequence or chunk size."
            )

        if use_gpu:
            alignments = _run_gpu_alignment(
                chunks,
                target_sequence,
            )
            backend_used = "gpu"
        else:
            alignments = distributed_chunk_alignment(
                record=record,
                target_sequence=target_sequence,
                chunk_size=chunk_size,
                overlap=overlap,
                client=client,
            )
            backend_used = "cpu"

        # Convert chunk-relative genomic positions into original
        # FASTA record coordinates.
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
        used_backends.add(backend_used)

        processed_records.append({
            "record_id": record["id"],
            "description": record["description"],
            "sequence_length": len(sequence),
            "alignment_backend": backend_used,
            "alignments": alignments,
            "guide_matches": guide_matches,
        })

    return {
        "records": processed_records,
        "summary": {
            "records_processed": len(processed_records),
            "chunks_aligned": total_chunks,
            "guide_matches_found": total_guide_matches,
            "alignment_backends": sorted(used_backends),
        },
    }
