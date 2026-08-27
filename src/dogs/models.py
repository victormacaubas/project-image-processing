"""Arquiteturas de modelo para os experimentos de classificação de raças."""

from __future__ import annotations

import torch.nn as nn

from dogs.config import NUM_CLASSES


class SmallCNN(nn.Module):
    """Classificador convolucional de baseline."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class LinearProbe(nn.Module):
    """Classificador linear para embeddings de imagens pré-treinados."""

    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def build_finetune_model(backbone_name: str, unfreeze_last_n_blocks: int) -> nn.Module:
    """Cria um backbone pré-treinado com cabeça de classificação de raças."""
    raise NotImplementedError
