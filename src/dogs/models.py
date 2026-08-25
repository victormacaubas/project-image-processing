"""Definições de modelo para cada experimento.

STATUS: esqueleto. SmallCNN e LinearProbe são da Mari (Passo 6 e 7 do
roteiro); build_finetune_model é do Victor (Passo 8).
"""

from __future__ import annotations

import torch.nn as nn

from dogs.config import NUM_CLASSES


class SmallCNN(nn.Module):
    """CNN do zero — baseline do experimento E1.

    TAREFA DO PARCEIRO. É a parte mais didática do trabalho: construir uma
    arquitetura do zero e ver, na prática, por que ela não dá conta.

    Sugestão de arquitetura (não é a única possível):
      - 4 blocos de [Conv2d 3x3 -> BatchNorm2d -> ReLU -> MaxPool2d(2)]
      - canais dobrando: 3 -> 32 -> 64 -> 128 -> 256
      - AdaptiveAvgPool2d(1) no fim das features
      - classifier: Flatten -> Dropout -> Linear(256, num_classes)

    Detalhes que valem discussão no relatório:
      - bias=False na Conv quando vem BatchNorm logo depois (o beta do BN já
        cumpre o papel do bias)
      - AdaptiveAvgPool em vez de Flatten direto: deixa o modelo independente
        do tamanho exato da entrada e reduz muito o número de parâmetros

    Expectativa: ~15-25% top-1. É pra ir mal mesmo — esse é o ponto do baseline.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class LinearProbe(nn.Module):
    """Classificador linear sobre embeddings pré-computados — experimento E2.

    TAREFA DO PARCEIRO. Treina em segundos, na CPU.

    Mede o quanto a representação congelada já separa as classes, sem nenhuma
    adaptação ao domínio.

    Sugestão: BatchNorm1d(input_dim) seguido de Linear(input_dim, num_classes).
    O BatchNorm ajuda porque as escalas dos embeddings variam bastante entre
    dimensões.

    Vale testar (e reportar) a variante sem BatchNorm — é uma ablação barata.
    """

    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def build_finetune_model(backbone_name: str, unfreeze_last_n_blocks: int) -> nn.Module:
    """Backbone pré-treinado com cabeça nova — experimento E3.

    TAREFA DO VICTOR.

    Passos:
      1. Carregar resnet50 com pesos de ImageNet
      2. Congelar TODOS os parâmetros (requires_grad = False)
      3. Descongelar os últimos N estágios entre [layer1..layer4]
         — os estágios finais codificam features mais específicas do domínio,
           por isso são os que mais ganham com adaptação
      4. Substituir model.fc por Linear(in_features, NUM_CLASSES)
         — a cabeça nova já nasce com requires_grad=True

    Com unfreeze_last_n_blocks=0 o backbone fica todo congelado e só a cabeça
    treina. Isso é diferente do LinearProbe (E2): aqui as imagens passam pelo
    backbone a cada batch, o que é muito mais lento mas permite augmentation.
    Vale explicar essa diferença no relatório.
    """
    raise NotImplementedError
