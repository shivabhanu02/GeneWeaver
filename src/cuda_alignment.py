
"""
GeneWeaver - CUDA Batched Smith-Waterman Alignment.

GPU: Calculates Smith-Waterman maximum scores.
CPU: Reconstructs alignments and verifies GPU scores.
"""

import numpy as np
from numba import cuda

from src.alignment import smith_waterman


MAX_SEQUENCE_LENGTH = 1023


@cuda.jit
def _smith_waterman_scores(
    reference,
    candidates,
    candidate_lengths,
    workspace,
    scores,
    match_score,
    mismatch_score,
    gap_penalty,
):
    candidate_id = cuda.grid(1)

    if candidate_id >= candidates.shape[0]:
        return

    candidate_length = candidate_lengths[candidate_id]

    # Two DP rows in GPU global memory.
    previous_row = 0
    current_row = 1

    for j in range(candidate_length + 1):
        workspace[candidate_id, previous_row, j] = 0

    max_score = 0

    for i in range(1, reference.size + 1):

        workspace[candidate_id, current_row, 0] = 0

        for j in range(1, candidate_length + 1):

            if reference[i - 1] == candidates[candidate_id, j - 1]:
                diagonal = (
                    workspace[candidate_id, previous_row, j - 1]
                    + match_score
                )
            else:
                diagonal = (
                    workspace[candidate_id, previous_row, j - 1]
                    + mismatch_score
                )

            up = (
                workspace[candidate_id, previous_row, j]
                + gap_penalty
            )

            left = (
                workspace[candidate_id, current_row, j - 1]
                + gap_penalty
            )

            value = 0

            if diagonal > value:
                value = diagonal

            if up > value:
                value = up

            if left > value:
                value = left

            workspace[candidate_id, current_row, j] = value

            if value > max_score:
                max_score = value

        # Swap the two DP rows.
        temp = previous_row
        previous_row = current_row
        current_row = temp

    scores[candidate_id] = max_score


def _validate_sequence(sequence, name):

    if not isinstance(sequence, str) or not sequence:
        raise ValueError(f"{name} must be a non-empty DNA string.")

    sequence = sequence.upper()

    if not set(sequence).issubset({"A", "C", "G", "T"}):
        raise ValueError(f"{name} contains invalid DNA bases.")

    return sequence


def batch_smith_waterman_gpu(
    reference_sequence,
    candidate_sequences,
    match_score=2,
    mismatch_score=-1,
    gap_penalty=-2,
):
    """Run batched Smith-Waterman scoring on an NVIDIA CUDA GPU."""

    if not cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    reference = _validate_sequence(
        reference_sequence, "Reference sequence"
    )

    if not isinstance(candidate_sequences, (list, tuple)):
        raise TypeError("Candidate sequences must be a list or tuple.")

    if not candidate_sequences:
        raise ValueError("At least one candidate sequence is required.")

    for value in (match_score, mismatch_score, gap_penalty):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise TypeError("Scoring parameters must be integers.")

    candidates = [
        _validate_sequence(sequence, f"Candidate {index}")
        for index, sequence in enumerate(candidate_sequences)
    ]

    max_length = max(len(sequence) for sequence in candidates)

    if len(reference) > MAX_SEQUENCE_LENGTH or max_length > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequences cannot exceed {MAX_SEQUENCE_LENGTH} bases."
        )

    # Convert DNA strings into ASCII arrays.
    reference_array = np.frombuffer(
        reference.encode("ascii"), dtype=np.uint8
    ).copy()

    candidate_array = np.zeros(
        (len(candidates), max_length), dtype=np.uint8
    )

    candidate_lengths = np.array(
        [len(sequence) for sequence in candidates],
        dtype=np.int32,
    )

    for index, sequence in enumerate(candidates):
        candidate_array[index, :len(sequence)] = np.frombuffer(
            sequence.encode("ascii"), dtype=np.uint8
        )

    # Allocate two DP rows per candidate in host memory.
    workspace = np.zeros(
        (len(candidates), 2, max_length + 1),
        dtype=np.int32,
    )

    scores = np.zeros(len(candidates), dtype=np.int32)

    # Transfer arrays to GPU.
    d_reference = cuda.to_device(reference_array)
    d_candidates = cuda.to_device(candidate_array)
    d_lengths = cuda.to_device(candidate_lengths)
    d_workspace = cuda.to_device(workspace)
    d_scores = cuda.to_device(scores)

    # Launch CUDA kernel.
    threads_per_block = 32
    blocks = (len(candidates) + threads_per_block - 1) // threads_per_block

    _smith_waterman_scores[blocks, threads_per_block](
        d_reference,
        d_candidates,
        d_lengths,
        d_workspace,
        d_scores,
        match_score,
        mismatch_score,
        gap_penalty,
    )

    gpu_scores = d_scores.copy_to_host()

    # Verify GPU scores against the existing CPU implementation.
    results = []

    for index, candidate in enumerate(candidates):

        cpu_result = smith_waterman(
            reference,
            candidate,
            match_score,
            mismatch_score,
            gap_penalty,
        )

        gpu_score = int(gpu_scores[index])

        if gpu_score != cpu_result["score"]:
            raise RuntimeError(
                f"GPU/CPU score mismatch for candidate {index}: "
                f"GPU={gpu_score}, CPU={cpu_result['score']}"
            )

        results.append({
            "candidate_index": index,
            "gpu_score": gpu_score,
            "aligned_sequence1": cpu_result["aligned_sequence1"],
            "aligned_sequence2": cpu_result["aligned_sequence2"],
            "start1": cpu_result["start1"],
            "start2": cpu_result["start2"],
            "end1": cpu_result["end1"],
            "end2": cpu_result["end2"],
        })

    return results
