
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Static, Input, Button

from src.pipeline import run_pipeline


class GeneWeaverTerminalApp(App):
    TITLE = "GeneWeaver - DNA Analysis Engine"
    SUB_TITLE = "GPU-Accelerated CRISPR Alignment"

    BINDINGS = [
        ("ctrl+r", "run_analysis", "Run Analysis"),
    ]

    CSS = """
    Screen {
        overflow-y: auto;
    }

    #main {
        width: 100%;
        padding: 1 2;
    }

    .field-title {
        color: cyan;
        margin-top: 1;
        margin-bottom: 1;
    }

    Input {
        width: 100%;
        margin-bottom: 1;
    }

    #run-analysis {
        width: 100%;
        margin-top: 1;
        margin-bottom: 1;
    }

    #output {
        width: 100%;
        min-height: 12;
        height: auto;
        border: round blue;
        padding: 1;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="main"):
            yield Static(
                "GENEWEAVER DNA ANALYSIS",
                classes="field-title",
            )

            yield Static("FASTA file path:", classes="field-title")
            yield Input(
                value="data/test_genome.fasta",
                placeholder="Enter FASTA file path",
                id="fasta",
            )

            yield Static("Target DNA sequence:", classes="field-title")
            yield Input(
                value="ACGTACGT",
                placeholder="Example: ACGTACGT",
                id="target",
            )

            yield Static("Chunk size:", classes="field-title")
            yield Input(value="8", id="chunk-size")

            yield Static("Guide RNA (optional):", classes="field-title")
            yield Input(value="ACGU", id="guide")

            yield Static(
                "Execution backend (auto/cpu/gpu):",
                classes="field-title",
            )
            yield Input(value="cpu", id="backend")

            yield Button(
                "RUN DNA ANALYSIS",
                id="run-analysis",
                variant="success",
            )

            yield Static(
                "Press Ctrl+R or select RUN DNA ANALYSIS.",
                id="output",
            )

        yield Footer()

    def action_run_analysis(self) -> None:
        self.run_analysis()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-analysis":
            self.run_analysis()

    def run_analysis(self) -> None:
        output = self.query_one("#output", Static)

        try:
            fasta_path = self.query_one("#fasta", Input).value.strip()
            target = self.query_one("#target", Input).value.strip()

            chunk_size = int(
                self.query_one("#chunk-size", Input).value
            )

            guide = self.query_one("#guide", Input).value.strip()
            backend = self.query_one("#backend", Input).value.strip().lower()

            if not fasta_path or not target:
                output.update(
                    "ERROR: FASTA path and target sequence are required."
                )
                return

            if chunk_size <= 0:
                output.update("ERROR: Chunk size must be positive.")
                return

            if backend not in {"auto", "cpu", "gpu"}:
                output.update(
                    "ERROR: Backend must be auto, cpu, or gpu."
                )
                return

            output.update("Running GeneWeaver pipeline...")

            result = run_pipeline(
                fasta_path=fasta_path,
                target_sequence=target,
                chunk_size=chunk_size,
                guide_rna=guide or None,
                execution_backend=backend,
            )

            summary = result["summary"]

            lines = [
                "GENEWEAVER ANALYSIS RESULTS",
                "=" * 35,
                f"Records processed: {summary['records_processed']}",
                f"Chunks aligned: {summary['chunks_aligned']}",
                f"Guide RNA matches: {summary['guide_matches_found']}",
                f"Backends used: {summary['alignment_backends']}",
                "",
            ]

            for record in result["records"]:
                lines.extend([
                    f"Record: {record['record_id']}",
                    f"Sequence length: {record['sequence_length']}",
                    f"Backend: {record['alignment_backend']}",
                    "Alignments:",
                ])

                for item in record["alignments"]:
                    lines.append(
                        f"  Chunk {item['chunk_id']} | "
                        f"Score: {item['alignment']['score']} | "
                        f"Coordinates: {item['start']}-{item['end']}"
                    )

                lines.append(
                    f"Guide matches: {record['guide_matches']}"
                )
                lines.append("")

            output.update("\n".join(lines))

        except Exception as error:
            output.update(
                f"ANALYSIS ERROR:\n{type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    GeneWeaverTerminalApp().run()
