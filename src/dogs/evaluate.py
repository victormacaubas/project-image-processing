"""Métricas e análise de erros.

Todo experimento reporta as mesmas métricas pelo mesmo caminho. Sem isso, a
tabela final de comparação não tem valor.

STATUS: esqueleto. Métricas são do Victor (Passo 5, junto com train.py);
most_confused_pairs é do Passo 9 (análise de erros).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch


@dataclass(frozen=True)
class Metrics:
    top1: float
    top5: float
    f1_macro: float
    loss: float


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (logits, labels) para o loader inteiro.

    Separado de evaluate() de propósito: a análise de erros precisa dos logits
    crus, não só das métricas agregadas.

    Lembrar de model.eval().
    """
    raise NotImplementedError


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> Metrics:
    """Calcula as quatro métricas a partir dos logits.

      top1     -> (argmax == labels).mean()
      top5     -> pegar os 5 maiores por linha (np.argsort(-logits)[:, :5])
                  e checar se o label verdadeiro está entre eles.
                  Relevante em fine-grained: mostra se o modelo está "quase certo".
      f1_macro -> sklearn f1_score com average="macro", zero_division=0.
                  Macro para que classes pequenas não sumam na média.
      loss     -> cross_entropy sobre os logits
    """
    raise NotImplementedError


def evaluate(model: nn.Module, loader: DataLoader, device) -> Metrics:
    """Conveniência: predict() + metrics_from_logits()."""
    raise NotImplementedError


def log_result(experiment_name: str, split: str, metrics: Metrics, notes: str = "") -> None:
    """Acrescenta uma linha em reports/results.csv.

    Resultado que não foi registrado não existe. Chamar SEMPRE que terminar um
    experimento, inclusive quando o número for ruim — resultado ruim também
    entra no relatório.

    Cabeçalho na primeira escrita:
        experiment, split, top1, top5, f1_macro, loss, notes

    Abrir em modo append. Formatar os floats com 4 casas para a tabela ficar
    legível.
    """
    raise NotImplementedError


def most_confused_pairs(
    logits: np.ndarray, labels: np.ndarray, class_names: list[str], top_n: int = 20
) -> list[tuple[str, str, int]]:
    """Pares (classe verdadeira, classe prevista) mais confundidos.

    Base do experimento E4. Uma matriz 120x120 é ilegível; o que interessa é
    quais raças o modelo troca entre si, e se essas trocas fazem sentido visual.

    Passos:
      1. confusion_matrix(labels, predictions, labels=range(len(class_names)))
      2. zerar a diagonal — acertos não interessam aqui
      3. ordenar todas as células decrescente, pegar as top_n
      4. np.unravel_index para voltar de índice achatado para (linha, coluna)
      5. traduzir índices para nomes de classe

    A pergunta a responder no relatório: um humano erraria os mesmos pares?
    """
    raise NotImplementedError
