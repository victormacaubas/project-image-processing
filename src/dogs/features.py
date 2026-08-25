"""Extração e cache de embeddings de backbones congelados.

Este é o módulo que destrava o cronograma. Roda uma vez (~5-8 min na GPU do
Colab); depois disso qualquer classificador linear treina em segundos na CPU,
e a Mari fica autônoma sem precisar de GPU.

PRIORIDADE MÁXIMA DA TERÇA. Enquanto os embeddings não existirem, E2 e E5
estão bloqueados.

STATUS: esqueleto. Ver docs/roteiro.md, Passo 4.

Uso pretendido:
    python -m dogs.features --backbone resnet50
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """cuda se disponível, senão cpu."""
    raise NotImplementedError


def build_backbone(name: str) -> tuple[nn.Module, int]:
    """Devolve (backbone sem cabeça de classificação, dimensão da saída).

    Para cada arquitetura, o truque é substituir a camada final por
    nn.Identity() e guardar in_features antes de trocar:

      resnet50         -> model.fc         (2048)
      efficientnet_b0  -> model.classifier (1280)
      vit_b_16         -> model.heads      (768)

    Usar os pesos pré-treinados de ImageNet (models.<Arch>_Weights).
    Levantar ValueError para nome desconhecido — falhar cedo é melhor que
    devolver o backbone errado silenciosamente.
    """
    raise NotImplementedError


@torch.no_grad()
def extract(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Passa o loader inteiro pelo backbone e devolve (embeddings, labels).

    Não esquecer:
      - model.eval() antes do laço (senão BatchNorm atualiza estatísticas)
      - o decorator @torch.no_grad() já evita construir o grafo
      - trazer para CPU e numpy antes de acumular, senão a VRAM estoura
      - logar progresso a cada N lotes: são ~320 lotes, e sem log parece travado
    """
    raise NotImplementedError


def main() -> None:
    """CLI.

    Passos:
      1. argparse: --backbone (default resnet50), --batch-size
      2. ensure_dirs()
      3. Montar TrainConfig com use_augmentation=False
         — IMPORTANTE: embeddings precisam ser determinísticos para servir de
           cache. Com augmentation, o embedding de uma imagem muda a cada
           execução e o cache perde o sentido.
      4. load_data(config) e build_backbone(args.backbone)
      5. Para cada split (train/val/test): extract() e salvar dois .npy
         em FEATURES_DIR com o padrão {backbone}_{split}_{X,y}.npy
      6. Avisar para subir FEATURES_DIR no Drive compartilhado
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
