"""Geracao de grafico comparativo entre BERT e RoBERTa para o portfolio/README.

Le os artefatos consolidados em `results/bert/` e `results/roberta/`
(`metrics.json` e `per_class_f1.csv`) e gera uma figura de dois paineis:
  1. Comparacao das 4 metricas agregadas no conjunto de teste (F1 Macro,
     F1 Micro, F1 Ponderado e Subset Accuracy).
  2. Comparacao do F1 por classe nas 17 emocoes com suporte ativo no RoBERTa,
     evidenciando as 11 classes de frequencia intermediaria que o RoBERTa
     destrava enquanto o BERT permanece em zero.

Uso:
    python -m src.compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def generate_comparison_chart(
    results_dir: Path | str = "results",
    output_path: Path | str | None = None,
) -> Path:
    """Gera e salva o grafico comparativo BERT vs. RoBERTa em PNG."""
    res_dir = Path(results_dir)
    if output_path is None:
        out_path = res_dir / "comparison_bert_roberta.png"
    else:
        out_path = Path(output_path)

    with open(res_dir / "bert" / "metrics.json", encoding="utf-8") as f:
        bert_metrics = json.load(f)["test"]
    with open(res_dir / "roberta" / "metrics.json", encoding="utf-8") as f:
        roberta_metrics = json.load(f)["test"]

    bert_cls = pd.read_csv(res_dir / "bert" / "per_class_f1.csv", index_col=0)
    roberta_cls = pd.read_csv(res_dir / "roberta" / "per_class_f1.csv", index_col=0)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )

    color_bert = "#64748B"     # Slate gray
    color_roberta = "#6D28D9"  # Deep purple

    # --- Painel 1: Metricas Globais no Teste ---
    metric_keys = [
        ("test_macro_f1", "Macro F1\n(Principal)"),
        ("test_micro_f1", "Micro F1"),
        ("test_weighted_f1", "Weighted F1"),
        ("test_subset_accuracy", "Subset\nAccuracy"),
    ]
    labels_m = [label for _, label in metric_keys]
    vals_bert = [float(bert_metrics[k]) for k, _ in metric_keys]
    vals_roberta = [float(roberta_metrics[k]) for k, _ in metric_keys]

    x = np.arange(len(labels_m))
    width = 0.36

    bars1 = ax1.bar(
        x - width / 2,
        vals_bert,
        width,
        label="BERT (bert-base-uncased)",
        color=color_bert,
        alpha=0.9,
    )
    bars2 = ax1.bar(
        x + width / 2,
        vals_roberta,
        width,
        label="RoBERTa (roberta-base)",
        color=color_roberta,
        alpha=0.95,
    )

    ax1.set_title(
        "Métricas Globais no Conjunto de Teste (GoEmotions)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_m, fontsize=10.5)
    ax1.set_ylim(0, 0.62)
    ax1.set_ylabel("Score (0.0 – 1.0)", fontsize=10.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#334155",
            fontweight="bold",
        )
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=color_roberta,
            fontweight="bold",
        )

    # --- Painel 2: Top 15 Classes por F1 (RoBERTa vs BERT) ---
    top_emotions = roberta_cls.sort_values("f1-score", ascending=False).head(15).index.tolist()
    top_emotions_rev = list(reversed(top_emotions))

    y = np.arange(len(top_emotions_rev))
    h_width = 0.38
    f1_bert_cls = [float(bert_cls.loc[e, "f1-score"]) for e in top_emotions_rev]
    f1_roberta_cls = [float(roberta_cls.loc[e, "f1-score"]) for e in top_emotions_rev]

    ax2.barh(y - h_width / 2, f1_bert_cls, h_width, label="BERT", color=color_bert, alpha=0.9)
    bars_r = ax2.barh(
        y + h_width / 2,
        f1_roberta_cls,
        h_width,
        label="RoBERTa",
        color=color_roberta,
        alpha=0.95,
    )

    ax2.set_title(
        "F1 por Classe — Top 15 Emoções (Destaque para Classes Intermediárias)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels(top_emotions_rev, fontsize=10)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("F1-Score no Conjunto de Teste", fontsize=10.5)

    for bar, b_val in zip(bars_r, f1_bert_cls):
        w = bar.get_width()
        suffix = "  (BERT = 0.00)" if b_val == 0.0 and w > 0.15 else ""
        ax2.annotate(
            f"{w:.2f}{suffix}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.8,
            color=color_roberta if suffix else "#1E293B",
            fontweight="bold" if suffix else "normal",
        )

    fig.suptitle(
        "Comparação Controlada no GoEmotions (28 Classes): "
        "RoBERTa Quase Dobra o F1 Macro do BERT (0.264 vs. 0.137)",
        fontsize=13.5,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera grafico comparativo BERT vs RoBERTa")
    parser.add_argument("--results_dir", default="results", help="Diretorio raiz dos resultados")
    parser.add_argument("--output", default=None, help="Caminho do PNG de saida")
    args = parser.parse_args()

    saved = generate_comparison_chart(results_dir=args.results_dir, output_path=args.output)
    print(f"[ok] Grafico comparativo salvo em: {saved}")


if __name__ == "__main__":
    main()
