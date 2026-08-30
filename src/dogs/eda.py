"""Funções para a análise exploratória do dataset Stanford Dogs.

As visualizações recebem o dataset bruto do Hugging Face para preservar as cores
originais das imagens. Não use os tensores de um DataLoader aqui: eles já passaram
por ``Normalize`` e ficariam visualmente incorretos no matplotlib.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dogs.viz import nome_legivel


def _save(fig: plt.Figure, salvar_em: Path | None) -> None:
    """Salva uma figura no padrão de entrega quando um destino é informado."""
    if salvar_em is not None:
        fig.savefig(salvar_em, dpi=150, bbox_inches="tight")


def indices_por_classe(
    dataset: Any, class_indices: Sequence[int], imagens_por_classe: int = 4
) -> list[int]:
    """Seleciona as primeiras imagens de cada classe sem decodificar imagens extras."""
    pendentes = set(class_indices)
    encontrados = {indice: [] for indice in class_indices}

    for posicao, rotulo in enumerate(dataset["label"]):
        indice = int(rotulo)
        if indice not in pendentes:
            continue
        encontrados[indice].append(posicao)
        if len(encontrados[indice]) == imagens_por_classe:
            pendentes.remove(indice)
            if not pendentes:
                break

    return [posicao for indice in class_indices for posicao in encontrados[indice]]


def distribuicao_classes(
    labels: Sequence[int], class_names: list[str], salvar_em: Path | None = None
) -> tuple[plt.Figure, dict[str, float]]:
    """Plota a distribuição ordenada por classe e retorna estatísticas descritivas."""
    contagens = Counter(map(int, labels))
    valores = np.array([contagens[indice] for indice in range(len(class_names))])
    ordem = np.argsort(valores)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(ordem)), valores[ordem], color="#4C78A8")
    ax.set_title("Distribuição de imagens por raça no conjunto de treino")
    ax.set_xlabel("Classes ordenadas por número de imagens")
    ax.set_ylabel("Número de imagens")
    ax.set_xticks([])
    fig.tight_layout()
    _save(fig, salvar_em)

    estatisticas = {
        "minimo": float(valores.min()),
        "maximo": float(valores.max()),
        "mediana": float(np.median(valores)),
    }
    return fig, estatisticas


def dispersao_resolucoes(
    dataset: Any,
    sample_size: int = 500,
    seed: int = 42,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Plota largura e altura de uma amostra determinística de imagens brutas."""
    tamanho = min(sample_size, len(dataset))
    gerador = np.random.default_rng(seed)
    indices = gerador.choice(len(dataset), size=tamanho, replace=False)
    tamanhos = [dataset[int(indice)]["image"].size for indice in indices]
    larguras, alturas = zip(*tamanhos, strict=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(larguras, alturas, alpha=0.45, s=15, color="#F58518")
    ax.axvline(224, color="black", linestyle="--", linewidth=1, label="224 px")
    ax.axhline(224, color="black", linestyle="--", linewidth=1)
    ax.set_title("Resoluções originais das imagens")
    ax.set_xlabel("Largura (px)")
    ax.set_ylabel("Altura (px)")
    ax.legend()
    fig.tight_layout()
    _save(fig, salvar_em)
    return fig


def grid_pares_parecidos(
    dataset: Any,
    class_names: list[str],
    pairs: Sequence[tuple[str, str]],
    imagens_por_raca: int = 2,
    salvar_em: Path | None = None,
) -> plt.Figure:
    """Compara visualmente pares de raças escolhidos antes do treinamento."""
    nomes = [nome for pair in pairs for nome in pair]
    class_indices = [class_names.index(nome) for nome in nomes]
    indices = indices_por_classe(dataset, class_indices, imagens_por_raca)

    n_colunas = imagens_por_raca * 2
    fig, axes = plt.subplots(
        len(pairs), n_colunas, figsize=(n_colunas * 2.8, len(pairs) * 3), squeeze=False
    )

    for linha, (nome_esquerda, nome_direita) in enumerate(pairs):
        for bloco, nome in enumerate((nome_esquerda, nome_direita)):
            indice_classe = class_names.index(nome)
            posicoes = [
                indice
                for indice in indices
                if int(dataset["label"][indice]) == indice_classe
            ]
            for offset in range(imagens_por_raca):
                ax = axes[linha, bloco * imagens_por_raca + offset]
                if offset < len(posicoes):
                    ax.imshow(dataset[posicoes[offset]]["image"].convert("RGB"))
                ax.axis("off")
                if offset == 0:
                    ax.set_title(nome_legivel(nome), fontsize=9)

    fig.suptitle("Pares de raças visualmente próximas", y=1.01)
    fig.tight_layout()
    _save(fig, salvar_em)
    return fig
