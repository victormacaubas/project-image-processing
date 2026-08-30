"""Arquiteturas de modelo para os experimentos de classificação de raças."""

from __future__ import annotations

from torch import nn

from dogs.config import NUM_CLASSES


class SmallCNN(nn.Module):
    """CNN de baseline com quatro blocos convolucionais para imagens de cães.

    Cada bloco preserva a resolução na convolução e a reduz pela metade com
    max pooling. Ao final, ``AdaptiveAvgPool2d(1)`` produz sempre 256 features,
    independentemente da resolução espacial de entrada.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LinearProbe(nn.Module):
    """Classificador linear para embeddings de imagens pré-treinados.

    Os embeddings da ResNet50 são produzidos após uma ReLU: são não negativos e
    podem ter escalas bem diferentes entre dimensões. A normalização por
    dimensão feita por ``BatchNorm1d`` torna a otimização da camada linear melhor
    condicionada. Ela pode ser desligada para a ablação do experimento E2.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = NUM_CLASSES,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(input_dim) if use_batchnorm else nn.Identity()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.normalization(x)
        return self.classifier(x)


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
