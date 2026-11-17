"""Unit tests for comparative visualization generator (src/compare.py)."""

from __future__ import annotations

from pathlib import Path

from src.compare import generate_comparison_chart


def test_generate_comparison_chart_creates_png(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    out_file = tmp_path / "comparison_bert_roberta.png"

    saved_path = generate_comparison_chart(results_dir=results_dir, output_path=out_file)
    assert saved_path == out_file
    assert out_file.exists()
    assert out_file.stat().st_size > 10_000
