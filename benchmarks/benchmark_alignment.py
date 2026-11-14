
import random
import time
import statistics

from src.alignment import smith_waterman


def generate_dna_sequence(length: int, seed: int) -> str:
    """Generate a reproducible random DNA sequence."""
    random_generator = random.Random(seed)
    return "".join(
        random_generator.choices("ACGT", k=length)
    )


def benchmark_alignment(
    sequence_length: int,
    repetitions: int = 3,
) -> dict:
    """Measure CPU Smith-Waterman alignment performance."""

    sequence1 = generate_dna_sequence(sequence_length, seed=42)
    sequence2 = generate_dna_sequence(sequence_length, seed=43)

    execution_times = []
    alignment_score = None

    for _ in range(repetitions):
        start_time = time.perf_counter()

        result = smith_waterman(sequence1, sequence2)

        elapsed_time = time.perf_counter() - start_time
        execution_times.append(elapsed_time)
        alignment_score = result["score"]

    return {
        "sequence_length": sequence_length,
        "repetitions": repetitions,
        "average_time_seconds": statistics.mean(execution_times),
        "minimum_time_seconds": min(execution_times),
        "maximum_time_seconds": max(execution_times),
        "alignment_score": alignment_score,
    }


def main():
    print("\nGENEWEAVER CPU ALIGNMENT BENCHMARK")
    print("=" * 65)

    sequence_lengths = [100, 250, 500]

    print(
        f"{'Length':<12}"
        f"{'Average (s)':<18}"
        f"{'Minimum (s)':<18}"
        f"{'Score':<10}"
    )

    print("-" * 65)

    for length in sequence_lengths:
        result = benchmark_alignment(length)

        print(
            f"{result['sequence_length']:<12}"
            f"{result['average_time_seconds']:<18.6f}"
            f"{result['minimum_time_seconds']:<18.6f}"
            f"{result['alignment_score']:<10}"
        )

    print("=" * 65)
    print("CPU benchmark completed.")


if __name__ == "__main__":
    main()
