"""Loop compartilhado de treinamento e carregamento de checkpoints."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dogs.config import TrainConfig, ensure_dirs
from dogs.evaluate import Metrics, evaluate

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    experiment_name: str
    best_val_metrics: Metrics
    best_epoch: int
    epochs_ran: int
    seconds_elapsed: float

    def __str__(self) -> str:
        return (
            f"{self.experiment_name}: melhor epoch {self.best_epoch}/{self.epochs_ran} "
            f"({self.seconds_elapsed / 60:.1f} min) | {self.best_val_metrics}"
        )


def get_device(explicit: torch.device | None = None) -> torch.device:
    if explicit is not None:
        return explicit
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device | None = None,
) -> TrainResult:
    """Treina um modelo e salva o checkpoint com melhor top-1 de validação."""
    ensure_dirs()
    device = get_device(device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("Nenhum parâmetro com requires_grad=True — nada a treinar.")

    total = sum(p.numel() for p in model.parameters())
    treinaveis = sum(p.numel() for p in trainable)
    logger.info(
        "%s | device=%s | %d parâmetros (%d treináveis, %.1f%%)",
        config.experiment_name,
        device,
        total,
        treinaveis,
        100 * treinaveis / total,
    )

    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    best_accuracy = -1.0
    best_metrics: Metrics | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    epoch = 0
    started_at = time.time()

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        metrics = evaluate(model, val_loader, device)

        marker = ""
        if metrics.top1 > best_accuracy:
            best_accuracy = metrics.top1
            best_metrics = metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_top1": metrics.top1,
                },
                config.checkpoint_path(),
            )
            marker = "  <- melhor"
        else:
            epochs_without_improvement += 1

        logger.info(
            "epoch %2d/%d | train_loss %.4f | val %s%s",
            epoch,
            config.num_epochs,
            train_loss,
            metrics,
            marker,
        )

        if epochs_without_improvement >= config.early_stopping_patience:
            logger.info(
                "Early stopping no epoch %d (%d epochs sem melhora)",
                epoch,
                epochs_without_improvement,
            )
            break

    if best_metrics is None:
        raise RuntimeError("Treino terminou sem completar nenhum epoch.")

    result = TrainResult(
        experiment_name=config.experiment_name,
        best_val_metrics=best_metrics,
        best_epoch=best_epoch,
        epochs_ran=epoch,
        seconds_elapsed=time.time() - started_at,
    )
    logger.info("%s", result)
    return result


def load_checkpoint(model: nn.Module, config: TrainConfig, device=None) -> nn.Module:
    """Carrega no modelo o checkpoint salvo com melhor resultado."""
    path = config.checkpoint_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {path}\n"
            "Rode o treino primeiro, ou baixe os artefatos."
        )

    device = get_device(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to(device)

    logger.info(
        "checkpoint carregado: %s (epoch %s, val_top1 %.4f)",
        path.name,
        payload.get("epoch", "?"),
        payload.get("val_top1", float("nan")),
    )
    return model
