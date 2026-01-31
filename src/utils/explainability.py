"""
Explainability Reports

Generate human-readable reports from conversation sessions.

Critical for competition judging (25% explainability score).

Outputs:
- Reasoning chain (what the AI thought at each step)
- Evidence trail (which guidelines were used)
- Decision confidence (why certain diagnoses were chosen)
- Comparison reports (MedGemma vs Gemini)
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime

from src.agents.conversation_manager import ConversationSession


class ExplainabilityReport:
    """
    Generate explainability reports from conversation sessions.

    Formats:
    - Text (human-readable)
    - Markdown (for documentation)
    - JSON (for programmatic access)
    - HTML (for web display)
    """

    def __init__(self, session: ConversationSession):
        """
        Initialize report generator.

        Args:
            session: Conversation session to analyze
        """
        self.session = session

    def generate_reasoning_chain(self) -> str:
        """
        Generate step-by-step reasoning chain.

        Returns:
            Formatted reasoning chain
        """
        lines = [
            "=" * 70,
            "REASONING CHAIN",
            "=" * 70,
            "",
            f"Session: {self.session.session_id}",
            f"Case: {self.session.case_id}",
            f"Model: {self.session.model_name}",
            f"Timestamp: {self.session.timestamp_start}",
            "",
            "-" * 70,
            ""
        ]

        for step in self.session.workflow_steps:
            lines.append(f"STEP {step['step']}: {step['agent'].upper()} AGENT")
            lines.append("-" * 70)

            if "reasoning" in step and step["reasoning"]:
                lines.append(f"Reasoning: {step['reasoning']}")

            if "output" in step and step["output"]:
                lines.append(f"Output: {json.dumps(step['output'], indent=2)}")

            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_evidence_trail(self) -> str:
        """
        Generate evidence trail showing guideline citations.

        Returns:
            Formatted evidence trail
        """
        lines = [
            "=" * 70,
            "EVIDENCE TRAIL - GUIDELINE CITATIONS",
            "=" * 70,
            ""
        ]

        citations = self.session.get_cited_guidelines()

        if not citations:
            lines.append("No guidelines cited.")
        else:
            lines.append(f"Total guidelines cited: {len(citations)}")
            lines.append("")

            for i, citation in enumerate(citations, 1):
                lines.append(f"{i}. {citation.get('title', 'Unknown')}")

                if "source" in citation:
                    lines.append(f"   Source: {citation['source']}")

                if "similarity" in citation:
                    lines.append(f"   Relevance: {citation['similarity']:.2f}")

                if "text_preview" in citation:
                    lines.append(f"   Preview: {citation['text_preview']}")

                lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def generate_performance_metrics(self) -> str:
        """
        Generate performance metrics report.

        Returns:
            Formatted metrics
        """
        metadata = self.session.to_dict()["metadata"]

        lines = [
            "=" * 70,
            "PERFORMANCE METRICS",
            "=" * 70,
            "",
            f"Total Steps: {metadata['total_steps']}",
            f"Total Tokens: {metadata['total_tokens']:,}",
            f"Total Latency: {metadata['total_latency_ms']:,} ms "
            f"({metadata['total_latency_ms'] / 1000:.2f} seconds)",
            "",
            f"Average Tokens/Step: {metadata['total_tokens'] / max(metadata['total_steps'], 1):.0f}",
            f"Average Latency/Step: {metadata['total_latency_ms'] / max(metadata['total_steps'], 1):.0f} ms",
            "",
            "=" * 70
        ]

        return "\n".join(lines)

    def generate_text_report(self) -> str:
        """
        Generate complete text report.

        Returns:
            Full report as text
        """
        sections = [
            self.generate_reasoning_chain(),
            "",
            self.generate_evidence_trail(),
            "",
            self.generate_performance_metrics()
        ]

        return "\n".join(sections)

    def generate_markdown_report(self) -> str:
        """
        Generate Markdown report.

        Returns:
            Markdown-formatted report
        """
        lines = [
            f"# Explainability Report",
            "",
            f"**Session ID:** `{self.session.session_id}`  ",
            f"**Case ID:** `{self.session.case_id}`  ",
            f"**Model:** `{self.session.model_name}`  ",
            f"**Timestamp:** {self.session.timestamp_start}  ",
            "",
            "---",
            "",
            "## Reasoning Chain",
            ""
        ]

        # Add each step
        for step in self.session.workflow_steps:
            lines.append(f"### Step {step['step']}: {step['agent'].title()} Agent")
            lines.append("")

            if "reasoning" in step and step["reasoning"]:
                lines.append(f"**Reasoning:** {step['reasoning']}")
                lines.append("")

            if "output" in step and step["output"]:
                lines.append("**Output:**")
                lines.append("```json")
                lines.append(json.dumps(step['output'], indent=2))
                lines.append("```")
                lines.append("")

        # Evidence trail
        lines.append("---")
        lines.append("")
        lines.append("## Evidence Trail")
        lines.append("")

        citations = self.session.get_cited_guidelines()

        if citations:
            for i, citation in enumerate(citations, 1):
                lines.append(f"{i}. **{citation.get('title', 'Unknown')}**")

                if "source" in citation:
                    lines.append(f"   - Source: {citation['source']}")

                if "similarity" in citation:
                    lines.append(f"   - Relevance: {citation['similarity']:.2f}")

                lines.append("")

        # Metrics
        metadata = self.session.to_dict()["metadata"]

        lines.append("---")
        lines.append("")
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append(f"- **Total Steps:** {metadata['total_steps']}")
        lines.append(f"- **Total Tokens:** {metadata['total_tokens']:,}")
        lines.append(f"- **Total Latency:** {metadata['total_latency_ms']:,} ms")
        lines.append("")

        return "\n".join(lines)

    def generate_html_report(self) -> str:
        """
        Generate HTML report.

        Returns:
            HTML-formatted report
        """
        # Simple HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Explainability Report - {self.session.session_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; margin-top: 30px; }}
        .meta {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .step {{ background: #f8f9fa; padding: 15px; margin: 15px 0; border-left: 4px solid #3498db; }}
        .reasoning {{ color: #16a085; font-weight: bold; }}
        .citation {{ background: #e8f5e9; padding: 10px; margin: 10px 0; border-radius: 3px; }}
        .metrics {{ background: #fff3cd; padding: 15px; border-radius: 5px; }}
        code {{ background: #272822; color: #f8f8f2; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background: #272822; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Explainability Report</h1>

    <div class="meta">
        <p><strong>Session ID:</strong> <code>{self.session.session_id}</code></p>
        <p><strong>Case ID:</strong> <code>{self.session.case_id}</code></p>
        <p><strong>Model:</strong> <code>{self.session.model_name}</code></p>
        <p><strong>Timestamp:</strong> {self.session.timestamp_start}</p>
    </div>

    <h2>Reasoning Chain</h2>
"""

        # Add steps
        for step in self.session.workflow_steps:
            html += f"""
    <div class="step">
        <h3>Step {step['step']}: {step['agent'].title()} Agent</h3>
"""
            if "reasoning" in step and step["reasoning"]:
                html += f"""
        <p class="reasoning">Reasoning: {step['reasoning']}</p>
"""

            if "output" in step and step["output"]:
                output_json = json.dumps(step['output'], indent=2)
                html += f"""
        <p><strong>Output:</strong></p>
        <pre><code>{output_json}</code></pre>
"""

            html += "    </div>\n"

        # Citations
        html += "<h2>Evidence Trail</h2>\n"

        citations = self.session.get_cited_guidelines()

        if citations:
            for citation in citations:
                title = citation.get('title', 'Unknown')
                source = citation.get('source', '')
                similarity = citation.get('similarity', 0)

                html += f"""
    <div class="citation">
        <strong>{title}</strong><br>
        Source: {source} | Relevance: {similarity:.2f}
    </div>
"""

        # Metrics
        metadata = self.session.to_dict()["metadata"]

        html += f"""
    <h2>Performance Metrics</h2>
    <div class="metrics">
        <p><strong>Total Steps:</strong> {metadata['total_steps']}</p>
        <p><strong>Total Tokens:</strong> {metadata['total_tokens']:,}</p>
        <p><strong>Total Latency:</strong> {metadata['total_latency_ms']:,} ms ({metadata['total_latency_ms'] / 1000:.2f} seconds)</p>
    </div>

</body>
</html>
"""

        return html

    def save_report(
        self,
        format: str = "markdown",
        directory: Optional[Path] = None
    ) -> Path:
        """
        Save report to file.

        Args:
            format: Output format (text, markdown, html, json)
            directory: Output directory

        Returns:
            Path to saved file
        """
        if directory is None:
            from config.config import settings
            directory = Path(settings.log_dir) / "reports"

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Generate report
        if format == "text":
            content = self.generate_text_report()
            extension = "txt"
        elif format == "markdown":
            content = self.generate_markdown_report()
            extension = "md"
        elif format == "html":
            content = self.generate_html_report()
            extension = "html"
        elif format == "json":
            content = self.session.to_json()
            extension = "json"
        else:
            raise ValueError(f"Unknown format: {format}")

        # Save file
        filename = f"report_{self.session.session_id}.{extension}"
        filepath = directory / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return filepath


def compare_sessions(
    sessions: List[ConversationSession],
    output_file: Optional[Path] = None
) -> str:
    """
    Generate comparison report for multiple sessions.

    Useful for comparing MedGemma vs Gemini performance.

    Args:
        sessions: List of sessions to compare
        output_file: Optional file to save report

    Returns:
        Comparison report as text
    """
    lines = [
        "=" * 70,
        "MODEL COMPARISON REPORT",
        "=" * 70,
        "",
        f"Comparing {len(sessions)} sessions",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
        "-" * 70,
        ""
    ]

    # Summary table
    lines.append("SUMMARY")
    lines.append("")
    lines.append(f"{'Model':<15} {'Steps':<10} {'Tokens':<10} {'Latency (ms)':<15} {'Completed':<10}")
    lines.append("-" * 70)

    for session in sessions:
        metadata = session.to_dict()["metadata"]
        lines.append(
            f"{session.model_name:<15} "
            f"{metadata['total_steps']:<10} "
            f"{metadata['total_tokens']:<10} "
            f"{metadata['total_latency_ms']:<15} "
            f"{str(session.completed):<10}"
        )

    lines.append("")
    lines.append("-" * 70)
    lines.append("")

    # Detailed comparison
    for i, session in enumerate(sessions, 1):
        lines.append(f"SESSION {i}: {session.model_name}")
        lines.append("-" * 70)

        # Reasoning
        reasoning_chain = session.get_reasoning_chain()
        lines.append(f"Reasoning steps: {len(reasoning_chain)}")

        for reasoning in reasoning_chain:
            lines.append(f"  - {reasoning}")

        lines.append("")

        # Citations
        citations = session.get_cited_guidelines()
        lines.append(f"Guidelines cited: {len(citations)}")
        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)

    # Save if requested
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

    return report
