
"""
GeneWeaver - Guide RNA Matching

Find exact guide RNA matches in a genomic DNA sequence
on forward and reverse-complement strands.
"""

VALID_DNA_BASES = set("ACGT")
VALID_RNA_BASES = set("ACGU")


def validate_guide_rna(guide_rna: str) -> bool:
    """Validate a guide RNA sequence."""
    if not isinstance(guide_rna, str) or not guide_rna:
        return False

    return set(guide_rna.upper()).issubset(VALID_RNA_BASES)


def validate_genome_sequence(genome_sequence: str) -> bool:
    """Validate a genomic DNA sequence."""
    if not isinstance(genome_sequence, str) or not genome_sequence:
        return False

    return set(genome_sequence.upper()).issubset(VALID_DNA_BASES)


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGT", "TGCA")
    return sequence.translate(complement)[::-1]


def find_exact_matches(
    guide_rna: str,
    genome_sequence: str,
) -> list[dict]:
    """
    Find exact guide RNA matches in both DNA orientations.

    Guide RNA U bases are converted to T for DNA comparison.
    Coordinates are 0-based and end-exclusive.

    A palindromic guide is searched only once per location
    to prevent duplicate records.
    """

    if not validate_guide_rna(guide_rna):
        raise ValueError(
            "Guide RNA must contain only A, C, G, and U."
        )

    if not validate_genome_sequence(genome_sequence):
        raise ValueError(
            "Genome sequence must contain only A, C, G, and T."
        )

    guide_dna = guide_rna.upper().replace("U", "T")
    genome = genome_sequence.upper()

    reverse_guide = reverse_complement(guide_dna)

    matches = []

    # Search the forward orientation.
    start = 0

    while True:
        position = genome.find(guide_dna, start)

        if position == -1:
            break

        matches.append({
            "start": position,
            "end": position + len(guide_dna),
            "strand": "+",
            "matched_sequence": guide_dna,
        })

        # Advance by one to preserve overlapping matches.
        start = position + 1

    # Search reverse orientation only if it is different.
    if reverse_guide != guide_dna:
        start = 0

        while True:
            position = genome.find(reverse_guide, start)

            if position == -1:
                break

            matches.append({
                "start": position,
                "end": position + len(reverse_guide),
                "strand": "-",
                "matched_sequence": reverse_guide,
            })

            start = position + 1

    return sorted(
        matches,
        key=lambda match: (match["start"], match["strand"])
    )
