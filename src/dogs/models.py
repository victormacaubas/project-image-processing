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
    from torchvision import models

    if backbone_name != "resnet50":
        raise ValueError(f"Backbone desconhecido: {backbone_name}. Opções: ('resnet50',)")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False

    stages = [model.layer1, model.layer2, model.layer3, model.layer4]
    for stage in stages[len(stages) - unfreeze_last_n_blocks :]:
        for param in stage.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model
