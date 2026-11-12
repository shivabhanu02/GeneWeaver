
from pathlib import Path
from typing import Iterator

from Bio import SeqIO


# Valid DNA bases, including standard IUPAC ambiguity codes
VALID_DNA_BASES = set("ACGTRYSWKMBDHVN")


def validate_dna_sequence(sequence: str) -> bool:
    """
    Validate a DNA sequence.

    Returns True if all characters are valid DNA bases.
    """
    if not sequence:
        return False

    sequence = sequence.upper()

    return all(base in VALID_DNA_BASES for base in sequence)


def parse_fasta(file_path: str | Path) -> Iterator[dict]:
    """
    Parse a FASTA file incrementally.

    Extracts:
        - Sequence ID
        - Description
        - DNA sequence

    Yields one sequence record at a time.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    record_count = 0

    with file_path.open("r", encoding="utf-8") as fasta_file:

        for record in SeqIO.parse(fasta_file, "fasta"):

            sequence = str(record.seq).upper()

            if not validate_dna_sequence(sequence):
                raise ValueError(
                    f"Invalid or empty DNA sequence: {record.id}"
                )

            record_count += 1

            yield {
                "id": record.id,
                "description": record.description,
                "sequence": sequence
            }

    if record_count == 0:
        raise ValueError("FASTA file contains no valid records.")
