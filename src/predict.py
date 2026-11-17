"""Inferencia multi-label de emocoes (API e CLI) para o modelo treinado no GoEmotions.

Permite classificar textos avulsos diretamente pela linha de comando, exibindo:
  - As emocoes ativadas acima do limiar de decisao (sigmoid >= threshold).
  - O grupo de sentimento correspondente (positive, negative, ambiguous, neutral).
  - O ranking Top-K de probabilidades para inspecao de confianca (util quando
    nenhum rotulo ultrapassa o limiar conservador de 0.5).

Uso:
    python -m src.predict --model_dir results/roberta/best_model --text "Thank you!"
    python -m src.predict --text "Wait, what?" --threshold 0.35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.labels import GOEMOTIONS_LABELS, LABEL2SENTIMENT
from src.metrics import sigmoid


def decode_logits_to_predictions(
    texts: list[str],
    logits: np.ndarray,
    threshold: float = 0.5,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Converte matriz de logits (N, 28) em predicoes estruturadas por texto.

    Args:
        texts: lista de frases de entrada (tamanho N).
        logits: array numpy de shape (N, 28) com os logits brutos do modelo.
        threshold: limiar de decisao sobre a probabilidade sigmoide (padrao 0.5).
        top_k: quantidade de classes mais provaveis a retornar no ranking.

    Returns:
        Lista de dicionarios com texto, emocoes ativas, resumo de sentimento e top_k.
    """
    probs_matrix = sigmoid(np.asarray(logits, dtype=np.float64))
    results: list[dict[str, Any]] = []

    for text, probs in zip(texts, probs_matrix):
        ranked_indices = np.argsort(probs)[::-1]

        active_emotions: list[dict[str, Any]] = []
        for idx in ranked_indices:
            prob = float(probs[idx])
            if prob >= threshold:
                emotion = GOEMOTIONS_LABELS[int(idx)]
                active_emotions.append(
                    {
                        "emotion": emotion,
                        "probability": round(prob, 4),
                        "sentiment": LABEL2SENTIMENT[emotion],
                    }
                )

        top_k_list: list[dict[str, Any]] = []
        for idx in ranked_indices[: max(1, top_k)]:
            emotion = GOEMOTIONS_LABELS[int(idx)]
            top_k_list.append(
                {
                    "emotion": emotion,
                    "probability": round(float(probs[idx]), 4),
                    "sentiment": LABEL2SENTIMENT[emotion],
                }
            )

        if active_emotions:
            seen_groups: list[str] = []
            for item in active_emotions:
                grp = item["sentiment"]
                if grp not in seen_groups:
                    seen_groups.append(grp)
            sentiment_summary = seen_groups
        else:
            sentiment_summary = ["none (below threshold)"]

        results.append(
            {
                "text": text,
                "active_emotions": active_emotions,
                "sentiment_summary": sentiment_summary,
                "top_k": top_k_list,
            }
        )

    return results


def predict_emotions(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    threshold: float = 0.5,
    top_k: int = 5,
    max_length: int = 64,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Executa tokenizacao e forward pass no modelo Transformer e decodifica as emocoes."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval().to(device)
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits.float().cpu().numpy()

    return decode_logits_to_predictions(texts, logits, threshold=threshold, top_k=top_k)


def format_prediction_report(results: list[dict[str, Any]], threshold: float = 0.5) -> str:
    """Formata a lista de predicoes em um relatorio legivel para o terminal."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"  GoEmotions Multi-Label Inference  (threshold = {threshold:.2f})")
    lines.append("=" * 72)

    for idx, item in enumerate(results, start=1):
        lines.append(f"\n[{idx}] Text: \"{item['text']}\"")
        if item["active_emotions"]:
            active_str = ", ".join(
                f"{e['emotion']} ({e['probability']:.1%} | {e['sentiment']})"
                for e in item["active_emotions"]
            )
            lines.append(f"    Predicted Emotions : {active_str}")
        else:
            lines.append("    Predicted Emotions : (none above threshold)")

        lines.append(f"    Sentiment Group(s) : {', '.join(item['sentiment_summary'])}")
        top_str = ", ".join(
            f"{e['emotion']}: {e['probability']:.1%}" for e in item["top_k"]
        )
        lines.append(f"    Top Probabilities  : {top_str}")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferencia multi-label de emocoes (GoEmotions) com Transformers"
    )
    parser.add_argument(
        "--model_dir",
        default="results/roberta/best_model",
        help="Checkpoint local ou ID no HuggingFace Hub (padrao: results/roberta/best_model)",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="Uma ou mais frases para classificar",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Limiar de decisao da sigmoide por classe (padrao: 0.5)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Numero de classes mais provaveis a exibir no ranking (padrao: 5)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=64,
        help="Comprimento maximo em tokens (padrao: 64)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime a saida em formato JSON estruturado",
    )
    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if args.model_dir.startswith("results/") and not model_path.exists():
        raise SystemExit(
            f"[erro] Diretorio do modelo nao encontrado em '{args.model_dir}'.\n"
            "Os pesos binarios de treino nao sao versionados no Git. "
            "Treine o modelo antes com:\n"
            "    python -m src.train --config configs/roberta.yaml"
        )

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    results = predict_emotions(
        texts=args.text,
        model=model,
        tokenizer=tokenizer,
        threshold=args.threshold,
        top_k=args.top_k,
        max_length=args.max_length,
    )

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(format_prediction_report(results, threshold=args.threshold))


if __name__ == "__main__":
    main()
