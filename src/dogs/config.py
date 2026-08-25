"""Configuração central do projeto.

Todos os caminhos e hiperparâmetros ficam aqui. Nenhum outro módulo deve conter
caminho hardcoded — é isso que permite o notebook rodar igual na máquina de
vocês, no Colab de vocês e no Colab da professora.

REGRA DE OURO
-------------
`PROJECT_ROOT` é derivado da localização deste arquivo, nunca de um caminho
absoluto. Se este arquivo está em <X>/src/dogs/config.py, então a raiz é <X>.
Funciona em qualquer máquina, clonado em qualquer lugar.

O Google Drive é usado APENAS como cache opcional de artefatos pesados, e só
quando estiver de fato montado. Ninguém além de vocês tem esse Drive — se o
código depender dele para rodar, quebra na correção.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ─── Raiz do projeto ────────────────────────────────────────────────────────
# parents[0] = dogs/ | parents[1] = src/ | parents[2] = raiz do repositório
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_CSV = REPORTS_DIR / "results.csv"

# Logits + labels de cada experimento (~3 MB comprimido). Vão para o Git: é o
# que permite ao notebook final reproduzir toda a análise sem baixar
# checkpoints. Ver save_predictions() em evaluate.py.
PREDICTIONS_DIR = REPORTS_DIR / "predictions"


# ─── Ambiente ───────────────────────────────────────────────────────────────
def in_colab() -> bool:
    """True se estamos rodando dentro do Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def drive_mounted() -> bool:
    """True se o Google Drive está montado E acessível.

    Note a diferença em relação a "estamos no Colab": a professora estará no
    Colab, mas sem o Drive de vocês. Toda escrita em Drive precisa passar por
    esta checagem.
    """
    return Path("/content/drive/MyDrive").is_dir()


def dataset_cache_dir() -> Path:
    """Onde o dataset baixado fica.

    No Colab, `/content` é disco efêmero: some quando a sessão desconecta, e o
    download de 776 MB teria que ser refeito toda vez. Se o Drive estiver
    montado, o cache vai para lá e o download acontece uma única vez.

    Fora do Colab, usa `data/raw` do próprio repositório.
    """
    if (cache := drive_cache_dir()) is not None:
        destino = cache / "hf_cache"
        destino.mkdir(parents=True, exist_ok=True)
        return destino

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR


def drive_cache_dir() -> Path | None:
    """Pasta de cache no Drive, ou None se indisponível.

    Uso pretendido: durante a semana, vocês apontam artefatos pesados para cá
    e não perdem nada quando o Colab desconecta. Na correção, devolve None e o
    código simplesmente usa o disco local — sem quebrar.
    """
    if not drive_mounted():
        return None

    cache = Path("/content/drive/MyDrive/project-image-processing")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# ─── Dataset ────────────────────────────────────────────────────────────────
# ATENÇÃO: não trocar por "Voxel51/StanfordDogs" (o sugerido no enunciado).
# Aquele é um dataset FiftyOne: load_dataset() não dá erro, mas devolve as
# 20.580 imagens SEM rótulo e num split único. Falha silenciosa — o pior tipo.
#
# Este aqui é parquet padrão, com o split oficial 12.000/8.580, coluna `label`
# do tipo ClassLabel e os nomes originais das raças preservados
# ("n02085620-Chihuahua"). Download de ~776 MB.
HF_DATASET = "maurice-fp/stanford-dogs"
NUM_CLASSES = 120
SEED = 42

# Normalização do ImageNet — obrigatória ao usar backbones pré-treinados.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparâmetros de um experimento.

    Congelado de propósito: um experimento não deve mutar a própria config no
    meio do treino. Para variar, crie outra instância.
    """

    experiment_name: str
    image_size: int = 224
    batch_size: int = 64
    num_epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    num_workers: int = 2
    early_stopping_patience: int = 4
    use_augmentation: bool = True
    # Quantos blocos finais do backbone descongelar (0 = backbone congelado).
    unfreeze_last_n_blocks: int = 0
    notes: str = ""

    def checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / f"{self.experiment_name}.pt"


def ensure_dirs() -> None:
    """Cria a árvore de diretórios. Idempotente — pode chamar sempre."""
    for directory in (
        RAW_DIR,
        PROCESSED_DIR,
        FEATURES_DIR,
        CHECKPOINT_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        PREDICTIONS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def describe_environment() -> str:
    """Resumo legível do ambiente. Útil no topo do notebook e em bug report."""
    import sys

    lines = [
        f"Python       {sys.version.split()[0]}",
        f"PROJECT_ROOT {PROJECT_ROOT}",
        f"Colab        {in_colab()}",
        f"Drive        {'montado' if drive_mounted() else 'não montado (ok)'}",
    ]

    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        lines.append(f"torch        {torch.__version__} ({device})")
    except ImportError:
        lines.append("torch        NÃO INSTALADO")

    return "\n".join(lines)
