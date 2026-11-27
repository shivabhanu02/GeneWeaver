
from pathlib import Path
import tempfile

import streamlit as st

from src.pipeline import run_pipeline


st.set_page_config(
    page_title="GeneWeaver",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 GeneWeaver")
st.subheader("GPU-Accelerated CRISPR Alignment Engine")
st.write(
    "Analyze DNA sequences using FASTA parsing, "
    "Smith-Waterman alignment, and guide RNA matching."
)

st.sidebar.header("Analysis Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload FASTA file",
    type=["fasta", "fa", "fna"],
)

target_sequence = st.sidebar.text_input(
    "Target DNA sequence",
    value="ACGTACGT",
)

chunk_size = st.sidebar.number_input(
    "Chunk size",
    min_value=1,
    value=8,
    step=1,
)

guide_rna = st.sidebar.text_input(
    "Guide RNA (optional)",
    value="ACGU",
)

backend = st.sidebar.selectbox(
    "Execution backend",
    ["auto", "cpu", "gpu"],
    index=1,
)

if st.sidebar.button("Run DNA Analysis", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a FASTA file.")
    elif not target_sequence.strip():
        st.error("Please enter a target DNA sequence.")
    else:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                fasta_path = Path(temp_dir) / uploaded_file.name
                fasta_path.write_bytes(uploaded_file.getvalue())

                with st.spinner("Running GeneWeaver analysis..."):
                    result = run_pipeline(
                        fasta_path=fasta_path,
                        target_sequence=target_sequence,
                        chunk_size=int(chunk_size),
                        guide_rna=guide_rna.strip() or None,
                        execution_backend=backend,
                    )

            st.success("DNA analysis completed successfully!")

            summary = result["summary"]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Records Processed",
                summary["records_processed"],
            )
            col2.metric(
                "Chunks Aligned",
                summary["chunks_aligned"],
            )
            col3.metric(
                "Guide RNA Matches",
                summary["guide_matches_found"],
            )

            st.subheader("Execution Summary")
            st.write(
                "Execution backends:",
                ", ".join(summary["alignment_backends"]) or "None",
            )

            for record in result["records"]:
                st.divider()
                st.subheader(f"Record: {record['record_id']}")

                st.write("Description:", record["description"])
                st.write("Sequence length:", record["sequence_length"])
                st.write("Backend:", record["alignment_backend"])

                st.markdown("#### Alignment Results")

                alignment_rows = []

                for item in record["alignments"]:
                    alignment_rows.append({
                        "Chunk": item["chunk_id"],
                        "Start": item["start"],
                        "End": item["end"],
                        "Score": item["alignment"]["score"],
                        "Aligned DNA": item["alignment"]["aligned_sequence1"],
                        "Aligned Target": item["alignment"]["aligned_sequence2"],
                    })

                st.dataframe(
                    alignment_rows,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### Guide RNA Matches")

                if record["guide_matches"]:
                    st.dataframe(
                        record["guide_matches"],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No guide RNA matches found.")

        except Exception as error:
            st.error(f"Analysis failed: {type(error).__name__}: {error}")
