"""Caminhos do projeto, verificações de ambiente e configuração de treino."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_CSV = REPORTS_DIR / "results.csv"

PREDICTIONS_DIR = REPORTS_DIR / "predictions"


def in_colab() -> bool:
    """Retorna se o interpretador está em execução no Google Colab."""
    try:
        from google import colab

    except ImportError:
        return False
    return colab is not None


def drive_mounted() -> bool:
    """Retorna se a montagem esperada do Google Drive está acessível."""
    return Path("/content/drive/MyDrive").is_dir()


def dataset_cache_dir() -> Path:
    """Retorna o diretório de cache do dataset, priorizando o Google Drive."""
    if (cache := drive_cache_dir()) is not None:
        destino = cache / "hf_cache"
        destino.mkdir(parents=True, exist_ok=True)
        return destino

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DIR


def drive_cache_dir() -> Path | None:
    """Retorna o diretório de cache do Google Drive quando disponível."""
    if not drive_mounted():
        return None

    cache = Path("/content/drive/MyDrive/project-image-processing")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


HF_DATASET = "maurice-fp/stanford-dogs"
NUM_CLASSES = 120
SEED = 42

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    """Hiperparâmetros imutáveis de um experimento de treinamento."""

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
    unfreeze_last_n_blocks: int = 0
    notes: str = ""

    def checkpoint_path(self) -> Path:
        return CHECKPOINT_DIR / f"{self.experiment_name}.pt"


def ensure_dirs() -> None:
    """Cria os diretórios de artefatos do projeto, se necessários."""
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
    """Retorna um resumo legível do ambiente de execução atual."""
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
