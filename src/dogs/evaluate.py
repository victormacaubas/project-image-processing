"""Avaliação de modelos, cálculo de métricas e análise de erros."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dogs.config import PREDICTIONS_DIR, RESULTS_CSV, ensure_dirs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Metrics:
    top1: float
    top5: float
    f1_macro: float
    loss: float

    def __str__(self) -> str:
        return (
            f"top1={self.top1:.4f} top5={self.top5:.4f} "
            f"f1={self.f1_macro:.4f} loss={self.loss:.4f}"
        )


@torch.no_grad()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna logits e rótulos de todos os batches de um data loader."""
    model.eval()
    model.to(device)

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for inputs, targets in loader:
        logits = model(inputs.to(device, non_blocking=True))
        all_logits.append(logits.float().cpu().numpy())
        all_labels.append(targets.numpy())

    return np.concatenate(all_logits), np.concatenate(all_labels)


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Metrics:
    """Calcula métricas de classificação a partir de logits e rótulos."""
    from sklearn.metrics import f1_score

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits e labels desalinhados: {logits.shape[0]} vs {labels.shape[0]}"
        )

    predictions = logits.argmax(axis=1)

    k = min(5, logits.shape[1])
    top_k = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    top5_hit = (top_k == labels[:, None]).any(axis=1)

    loss = nn.functional.cross_entropy(
        torch.from_numpy(logits).float(), torch.from_numpy(labels).long()
    ).item()

    return Metrics(
        top1=float((predictions == labels).mean()),
        top5=float(top5_hit.mean()),
        f1_macro=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        loss=loss,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Metrics:
    """Avalia um modelo em um data loader."""
    logits, labels = predict(model, loader, device)
    return metrics_from_logits(logits, labels)


def log_result(experiment_name: str, split: str, metrics: Metrics, notes: str = "") -> None:
    """Acrescenta as métricas de um experimento ao CSV de resultados."""
    ensure_dirs()
    is_new = not RESULTS_CSV.exists()

    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(
                ["experiment", "split", "top1", "top5", "f1_macro", "loss", "notes"]
            )
        writer.writerow(
            [
                experiment_name,
                split,
                f"{metrics.top1:.4f}",
                f"{metrics.top5:.4f}",
                f"{metrics.f1_macro:.4f}",
                f"{metrics.loss:.4f}",
                notes,
            ]
        )

    logger.info("registrado %s/%s: %s", experiment_name, split, metrics)


def most_confused_pairs(
    logits: np.ndarray, labels: np.ndarray, class_names: list[str], top_n: int = 20
) -> list[tuple[str, str, int]]:
    """Retorna as confusões mais frequentes entre classe real e prevista."""
    from sklearn.metrics import confusion_matrix

    predictions = logits.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    np.fill_diagonal(matrix, 0)

    flat_order = np.argsort(matrix, axis=None)[::-1][:top_n]

    pairs: list[tuple[str, str, int]] = []
    for flat_index in flat_order:
        true_index, predicted_index = np.unravel_index(flat_index, matrix.shape)
        count = int(matrix[true_index, predicted_index])
        if count == 0:
            break
        pairs.append((class_names[true_index], class_names[predicted_index], count))

    return pairs


def save_predictions(
    experiment_name: str, split: str, logits: np.ndarray, labels: np.ndarray
) -> "Path":
    """Salva logits e rótulos comprimidos de um split do dataset."""
    ensure_dirs()
    path = PREDICTIONS_DIR / f"{experiment_name}_{split}.npz"
    np.savez_compressed(path, logits=logits.astype(np.float32), labels=labels)

    logger.info("predições salvas: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path


def load_predictions(experiment_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Carrega logits e rótulos salvos de um split do dataset."""
    path = PREDICTIONS_DIR / f"{experiment_name}_{split}.npz"
    if not path.exists():
        disponiveis = sorted(p.stem for p in PREDICTIONS_DIR.glob("*.npz"))
        raise FileNotFoundError(
            f"Predições não encontradas: {path.name}\n"
            f"Disponíveis: {disponiveis or 'nenhuma'}\n"
            "Rode o experimento com RETRAIN = True para gerá-las."
        )

    with np.load(path) as payload:
        return payload["logits"], payload["labels"]


def hardest_examples(
    logits: np.ndarray, labels: np.ndarray, top_n: int = 10
) -> list[tuple[int, float, int]]:
    """Retorna índices, perdas e previsões ordenados por perda."""
    per_sample = nn.functional.cross_entropy(
        torch.from_numpy(logits).float(),
        torch.from_numpy(labels).long(),
        reduction="none",
    ).numpy()

    worst = np.argsort(-per_sample)[:top_n]
    predictions = logits.argmax(axis=1)

    return [(int(i), float(per_sample[i]), int(predictions[i])) for i in worst]
