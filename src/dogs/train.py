"""Loop de treino compartilhado por todos os experimentos.

Um único loop, usado por E1, E2 e E3. Isso garante que a comparação entre
experimentos seja honesta — mesma lógica de early stopping, mesmo critério
de seleção de checkpoint, mesmas métricas.

STATUS: esqueleto. Tarefa do Victor, Passo 5 do roteiro. Precisa estar pronto
antes de E1 — o parceiro depende dele.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dogs.evaluate import Metrics


@dataclass
class TrainResult:
    experiment_name: str
    best_val_metrics: Metrics
    epochs_ran: int
    seconds_elapsed: float


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    device: torch.device | None = None,
) -> TrainResult:
    """Treina com early stopping na acurácia de validação.

    Estrutura:

      setup
        - device, model.to(device)
        - criterion: CrossEntropyLoss (considerar label_smoothing=0.1)
        - optimizer: AdamW SÓ sobre os parâmetros com requires_grad=True
          — passar parâmetros congelados para o otimizador é um bug comum e
            silencioso; ele não treina, mas mantém estado de momento à toa
        - scheduler: CosineAnnealingLR

      por epoch
        - model.train(), laço sobre train_loader
        - zero_grad(set_to_none=True) / forward / loss / backward / step
        - acumular loss ponderada pelo batch size
        - scheduler.step() no fim do epoch, não a cada batch
        - avaliar na validação com evaluate()
        - logar: epoch, train loss, val top1, top5, f1

      seleção de checkpoint
        - salvar quando val top-1 melhora, não no último epoch
          — o último costuma estar mais overfitado que o melhor
        - salvar model_state + config + epoch, para conseguir recarregar depois
        - contar epochs sem melhora; parar em early_stopping_patience

    Devolver TrainResult com as métricas do MELHOR epoch, não do último.
    """
    raise NotImplementedError
