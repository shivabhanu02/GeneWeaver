
def smith_waterman(
    sequence1: str,
    sequence2: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
) -> dict:
    """
    Perform local DNA sequence alignment using the
    Smith-Waterman algorithm.
    """

    sequence1 = sequence1.upper()
    sequence2 = sequence2.upper()

    if not sequence1 or not sequence2:
        raise ValueError("Both DNA sequences must be non-empty.")

    valid_bases = set("ACGT")

    if not set(sequence1).issubset(valid_bases):
        raise ValueError("Sequence 1 contains invalid DNA bases.")

    if not set(sequence2).issubset(valid_bases):
        raise ValueError("Sequence 2 contains invalid DNA bases.")

    rows = len(sequence1) + 1
    cols = len(sequence2) + 1

    # Initialize scoring matrix.
    matrix = [[0] * cols for _ in range(rows)]

    max_score = 0
    max_position = (0, 0)

    # Fill the scoring matrix.
    for i in range(1, rows):
        for j in range(1, cols):

            if sequence1[i - 1] == sequence2[j - 1]:
                diagonal = matrix[i - 1][j - 1] + match_score
            else:
                diagonal = matrix[i - 1][j - 1] + mismatch_score

            up = matrix[i - 1][j] + gap_penalty
            left = matrix[i][j - 1] + gap_penalty

            matrix[i][j] = max(0, diagonal, up, left)

            if matrix[i][j] > max_score:
                max_score = matrix[i][j]
                max_position = (i, j)

    # Traceback to find the best local alignment.
    i, j = max_position
    aligned1 = []
    aligned2 = []

    while i > 0 and j > 0 and matrix[i][j] > 0:

        current = matrix[i][j]

        if sequence1[i - 1] == sequence2[j - 1]:
            diagonal = matrix[i - 1][j - 1] + match_score
        else:
            diagonal = matrix[i - 1][j - 1] + mismatch_score

        if current == diagonal:
            aligned1.append(sequence1[i - 1])
            aligned2.append(sequence2[j - 1])
            i -= 1
            j -= 1

        elif current == matrix[i - 1][j] + gap_penalty:
            aligned1.append(sequence1[i - 1])
            aligned2.append("-")
            i -= 1

        else:
            aligned1.append("-")
            aligned2.append(sequence2[j - 1])
            j -= 1

    return {
        "score": max_score,
        "aligned_sequence1": "".join(reversed(aligned1)),
        "aligned_sequence2": "".join(reversed(aligned2)),
        "start1": i,
        "start2": j,
        "end1": max_position[0],
        "end2": max_position[1],
    }
