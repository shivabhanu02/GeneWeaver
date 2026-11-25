
import pytest

import src.pipeline as pipeline_module
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


# --------------------------------------------------
# Existing pipeline tests
# --------------------------------------------------

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
            fasta_path="missing_file.fasta",
            target_sequence="ACGT",
            chunk_size=4,
        )


# --------------------------------------------------
# Day 14: CPU/GPU integration tests
# --------------------------------------------------

def test_pipeline_cpu_backend(sample_fasta, monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "_cuda_available",
        lambda: False,
    )

    result = run_pipeline(
        fasta_path=sample_fasta,
        target_sequence="ACGT",
        chunk_size=4,
        execution_backend="cpu",
    )

    assert result["summary"]["alignment_backends"] == ["cpu"]
    assert result["records"][0]["alignment_backend"] == "cpu"


def test_pipeline_auto_falls_back_to_cpu(sample_fasta, monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "_cuda_available",
        lambda: False,
    )

    result = run_pipeline(
        fasta_path=sample_fasta,
        target_sequence="ACGT",
        chunk_size=4,
        execution_backend="auto",
    )

    assert result["summary"]["alignment_backends"] == ["cpu"]


def test_pipeline_gpu_backend_maps_coordinates(
    sample_fasta,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "_cuda_available",
        lambda: True,
    )

    def fake_gpu_alignment(chunks, target_sequence):
        assert target_sequence == "ACGT"
        assert [chunk["sequence"] for chunk in chunks] == [
            "ACGT",
            "ACGT",
        ]

        results = []

        for chunk in chunks:
            results.append({
                "record_id": chunk["record_id"],
                "chunk_id": chunk["chunk_id"],
                "start": chunk["start"],
                "end": chunk["end"],
                "alignment": {
                    "score": 8,
                    "aligned_sequence1": "ACGT",
                    "aligned_sequence2": "ACGT",
                    "start1": 0,
                    "start2": 0,
                    "end1": 4,
                    "end2": 4,
                },
            })

        return results

    monkeypatch.setattr(
        pipeline_module,
        "_run_gpu_alignment",
        fake_gpu_alignment,
    )

    result = run_pipeline(
        fasta_path=sample_fasta,
        target_sequence="ACGT",
        chunk_size=4,
        execution_backend="gpu",
    )

    alignments = result["records"][0]["alignments"]

    assert result["summary"]["alignment_backends"] == ["gpu"]
    assert result["records"][0]["alignment_backend"] == "gpu"

    assert alignments[0]["alignment"]["start1"] == 0
    assert alignments[0]["alignment"]["end1"] == 4

    assert alignments[1]["alignment"]["start1"] == 4
    assert alignments[1]["alignment"]["end1"] == 8


def test_pipeline_gpu_requested_without_cuda(
    sample_fasta,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "_cuda_available",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="Numba or CUDA"):
        run_pipeline(
            fasta_path=sample_fasta,
            target_sequence="ACGT",
            chunk_size=4,
            execution_backend="gpu",
        )


def test_pipeline_gpu_rejects_unsupported_sequence_length(
    sample_fasta,
    monkeypatch,
):
    monkeypatch.setattr(
        pipeline_module,
        "_cuda_available",
        lambda: True,
    )

    with pytest.raises(ValueError, match="supports sequences up to"):
        run_pipeline(
            fasta_path=sample_fasta,
            target_sequence="A" * (
                pipeline_module.MAX_SEQUENCE_LENGTH + 1
            ),
            chunk_size=4,
            execution_backend="gpu",
        )


def test_pipeline_rejects_invalid_backend(sample_fasta):
    with pytest.raises(ValueError, match="execution_backend"):
        run_pipeline(
            fasta_path=sample_fasta,
            target_sequence="ACGT",
            chunk_size=4,
            execution_backend="invalid",
        )
