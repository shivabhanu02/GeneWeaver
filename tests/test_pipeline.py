
from pathlib import Path

import pytest

from src.pipeline import run_pipeline


@pytest.fixture
def sample_fasta(tmp_path):
    fasta_file = tmp_path / "sample.fasta"
    fasta_file.write_text(
        ">chromosome_1 Sample chromosome\n"
        "ACGTACGT\n",
        encoding="utf-8",
    )
    return fasta_file


def test_pipeline_processes_fasta_record(sample_fasta):
    result = run_pipeline(
        fasta_path=sample_fasta,
        target_sequence="ACGT",
        chunk_size=4,
    )

    assert result["summary"]["records_processed"] == 1
    assert result["records"][0]["record_id"] == "chromosome_1"


def test_pipeline_aligns_all_chunks(sample_fasta):
    result = run_pipeline(
        fasta_path=sample_fasta,
        target_sequence="ACGT",
        chunk_size=4,
    )

    assert result["summary"]["chunks_aligned"] == 2
    assert len(result["records"][0]["alignments"]) == 2


def test_pipeline_preserves_genomic_coordinates(sample_fasta):
    result = run_pipeline(
        fasta_path=sample_fasta,
        target_sequence="ACGT",
        chunk_size=4,
    )

    alignments = result["records"][0]["alignments"]

    assert alignments[0]["alignment"]["start1"] == 0
    assert alignments[0]["alignment"]["end1"] == 4

    assert alignments[1]["alignment"]["start1"] == 4
    assert alignments[1]["alignment"]["end1"] == 8


def test_pipeline_finds_guide_rna_matches(sample_fasta):
    result = run_pipeline(
        fasta_path=sample_fasta,
        target_sequence="ACGT",
        chunk_size=4,
        guide_rna="ACGU",
    )

    matches = result["records"][0]["guide_matches"]

    assert len(matches) == 2
    assert result["summary"]["guide_matches_found"] == 2
    assert matches[0]["start"] == 0
    assert matches[1]["start"] == 4


def test_pipeline_without_guide_rna(sample_fasta):
    result = run_pipeline(
        fasta_path=sample_fasta,
        target_sequence="ACGT",
        chunk_size=4,
    )

    assert result["records"][0]["guide_matches"] == []
    assert result["summary"]["guide_matches_found"] == 0


def test_pipeline_rejects_ambiguous_fasta_bases(tmp_path):
    fasta_file = tmp_path / "ambiguous.fasta"
    fasta_file.write_text(
        ">chromosome_1\nACGTN\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only A, C, G, and T"):
        run_pipeline(
            fasta_path=fasta_file,
            target_sequence="ACGT",
            chunk_size=4,
        )


def test_pipeline_rejects_invalid_target(sample_fasta):
    with pytest.raises(ValueError):
        run_pipeline(
            fasta_path=sample_fasta,
            target_sequence="ACGTN",
            chunk_size=4,
        )


def test_pipeline_rejects_invalid_guide_rna(sample_fasta):
    with pytest.raises(ValueError):
        run_pipeline(
            fasta_path=sample_fasta,
            target_sequence="ACGT",
            chunk_size=4,
            guide_rna="ACGT",
        )


def test_pipeline_rejects_missing_fasta():
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            fasta_path=Path("missing_file.fasta"),
            target_sequence="ACGT",
            chunk_size=4,
        )
