"""Visualizações de amostras, métricas de experimentos e erros do modelo.

Todas as funções que exibem imagens recebem o dataset bruto do Hugging Face (ex.:
dogs.data._load_raw()["test"]), nunca um DataLoader: as imagens que já passaram por
transforms.Normalize saem com cores lavadas. Se algum dia precisar plotar um tensor
normalizado diretamente, lembre de desfazer a normalização e converter de (C, H, W) para
(H, W, C) antes de chamar imshow — PIL/matplotlib esperam canais por último.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dogs.evaluate import hardest_examples, most_confused_pairs

MAX_COLUNAS_GRID_ERROS = 5


def nome_legivel(nome: str) -> str:
    """Converte o nome bruto da classe do WordNet (ex.: n02085620-Chihuahua) em rótulo legível."""
    _, _, resto = nome.partition("-")
    rotulo = resto if resto else nome
    return rotulo.replace("_", " ")


def grid_de_amostras(
    dataset: Any,
    indices: Sequence[int],
    class_names: list[str],
    n_colunas: int = 4,
    titulo: str | None = None,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Exibe um grid de imagens do dataset bruto com o nome da classe embaixo de cada uma.

    Recebe o dataset bruto (não um DataLoader) para evitar imagens lavadas pela Normalize.
    Um número de índices que não completa a última linha deixa eixos vazios, desligados com
    ax.axis("off"). Sugestão de destino para salvar_em: dogs.config.FIGURES_DIR.
    """
    n_linhas = math.ceil(len(indices) / n_colunas)
    fig, axes = plt.subplots(n_linhas, n_colunas, figsize=(n_colunas * 3, n_linhas * 3))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, indice in zip(axes_flat, indices):
        registro = dataset[indice]
        imagem = registro["image"].convert("RGB")
        ax.imshow(imagem)
        ax.set_title(nome_legivel(class_names[int(registro["label"])]), fontsize=9)
        ax.axis("off")

    for ax in axes_flat[len(indices) :]:
        ax.axis("off")

    if titulo:
        fig.suptitle(titulo)
    fig.tight_layout()

    if salvar_em is not None:
        fig.savefig(salvar_em, dpi=150, bbox_inches="tight")

    return fig


def comparar_experimentos(
    results_df: pd.DataFrame,
    metrica: str = "top1",
    ax: plt.Axes | None = None,
    salvar_em: Path | None = None,
) -> plt.Axes:
    """Compara experimentos com barras horizontais ordenadas pela métrica escolhida.

    results_df é o DataFrame lido de reports/results.csv (colunas: experiment, split, top1,
    top5, f1_macro, loss, notes). Sugestão de destino para salvar_em: dogs.config.FIGURES_DIR.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.5 * len(results_df) + 1))

    ordenado = results_df.sort_values(metrica)
    barras = ax.barh(ordenado["experiment"], ordenado[metrica])
    ax.set_xlabel(metrica)
    ax.set_ylabel("experimento")

    for barra, valor in zip(barras, ordenado[metrica]):
        ax.text(
            barra.get_width(),
            barra.get_y() + barra.get_height() / 2,
            f"{valor:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    if salvar_em is not None:
        ax.figure.savefig(salvar_em, dpi=150, bbox_inches="tight")

    return ax


def matriz_confusao_recorte(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    indices_classes: Sequence[int],
    ax: plt.Axes | None = None,
    salvar_em: Path | None = None,
) -> plt.Axes:
    """Heatmap normalizado por linha de um recorte da matriz de confusão.

    A matriz completa 120x120 é ilegível; passe em indices_classes apenas as classes de
    interesse. Cada linha é normalizada pelo total de amostras da classe real, mostrando
    proporção em vez de contagem; uma classe sem amostras no recorte fica com linha zerada
    em vez de dividir por zero. Sugestão de destino para salvar_em: dogs.config.FIGURES_DIR.
    """
    from sklearn.metrics import confusion_matrix

    predictions = logits.argmax(axis=1)
    matriz_completa = confusion_matrix(labels, predictions, labels=range(len(class_names)))
    recorte = matriz_completa[np.ix_(indices_classes, indices_classes)].astype(float)

    somas = recorte.sum(axis=1, keepdims=True)
    recorte_normalizado = np.divide(recorte, somas, out=np.zeros_like(recorte), where=somas != 0)

    rotulos = [nome_legivel(class_names[i]) for i in indices_classes]

    if ax is None:
        lado = len(rotulos) * 0.6 + 2
        _, ax = plt.subplots(figsize=(lado, lado))

    sns.heatmap(
        recorte_normalizado,
        xticklabels=rotulos,
        yticklabels=rotulos,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")

    if salvar_em is not None:
        ax.figure.savefig(salvar_em, dpi=150, bbox_inches="tight")

    return ax


def _primeiro_indice_com_label(dataset: Any, indice_classe: int) -> int | None:
    """Retorna a posição da primeira imagem da classe, ou None se não houver nenhuma.

    Lê a coluna inteira de rótulos em vez de indexar registro por registro: no dataset do
    Hugging Face, tocar um registro decodifica a imagem, e varrer os 8.580 do teste assim
    leva minutos.
    """
    for posicao, rotulo in enumerate(dataset["label"]):
        if int(rotulo) == indice_classe:
            return posicao
    return None


def grid_pares_confundidos(
    dataset: Any,
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    top_n: int = 5,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Mostra, para cada par de classes mais confundido, uma imagem real ao lado de uma prevista.

    Recebe o dataset bruto para evitar cores lavadas. most_confused_pairs devolve nomes de
    classe, não índices; aqui eles são convertidos com class_names.index() e depois usados
    para achar a primeira imagem do split com aquele rótulo. Sugestão de destino para
    salvar_em: dogs.config.FIGURES_DIR.
    """
    pares = most_confused_pairs(logits, labels, class_names, top_n=top_n)

    fig, axes = plt.subplots(len(pares), 2, figsize=(6, 3 * len(pares)))
    axes_grid = np.atleast_2d(axes)

    for linha, (nome_real, nome_previsto, contagem) in enumerate(pares):
        indice_real = class_names.index(nome_real)
        indice_previsto = class_names.index(nome_previsto)

        for coluna, (indice_classe, rotulo) in enumerate(
            [(indice_real, nome_real), (indice_previsto, nome_previsto)]
        ):
            ax = axes_grid[linha, coluna]
            posicao = _primeiro_indice_com_label(dataset, indice_classe)
            if posicao is None:
                ax.axis("off")
                continue

            imagem = dataset[posicao]["image"].convert("RGB")
            ax.imshow(imagem)
            sufixo = f" ({contagem}x)" if coluna == 1 else ""
            ax.set_title(f"{nome_legivel(rotulo)}{sufixo}", fontsize=9)
            ax.axis("off")

    fig.suptitle("Pares de classes mais confundidos (real | previsto)")
    fig.tight_layout()

    if salvar_em is not None:
        fig.savefig(salvar_em, dpi=150, bbox_inches="tight")

    return fig


def grid_piores_erros(
    dataset: Any,
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    top_n: int = 10,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Mostra as imagens com maior perda individual, com classe real, prevista e valor da perda.

    Recebe o dataset bruto para evitar cores lavadas. Um top_n que não completa a última
    linha do grid deixa eixos vazios, desligados com ax.axis("off"). Sugestão de destino
    para salvar_em: dogs.config.FIGURES_DIR.
    """
    piores = hardest_examples(logits, labels, top_n=top_n)

    n_colunas = min(MAX_COLUNAS_GRID_ERROS, len(piores))
    n_linhas = math.ceil(len(piores) / n_colunas)
    fig, axes = plt.subplots(n_linhas, n_colunas, figsize=(n_colunas * 3, n_linhas * 3.5))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (indice, perda, indice_previsto) in zip(axes_flat, piores):
        registro = dataset[indice]
        imagem = registro["image"].convert("RGB")
        real = nome_legivel(class_names[int(registro["label"])])
        previsto = nome_legivel(class_names[indice_previsto])
        ax.imshow(imagem)
        ax.set_title(f"real: {real}\nprevisto: {previsto}\nloss={perda:.2f}", fontsize=8)
        ax.axis("off")

    for ax in axes_flat[len(piores) :]:
        ax.axis("off")

    fig.tight_layout()

    if salvar_em is not None:
        fig.savefig(salvar_em, dpi=150, bbox_inches="tight")

    return fig
