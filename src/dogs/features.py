"""Extração e cache de embeddings de backbones pré-treinados."""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dogs.config import FEATURES_DIR, TrainConfig, drive_cache_dir, ensure_dirs
from dogs.data import load_data

logger = logging.getLogger(__name__)

BACKBONES = ("resnet50", "efficientnet_b0", "vit_b_16")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_backbone(name: str) -> tuple[nn.Module, int]:
    """Cria um backbone pré-treinado que retorna embeddings e sua dimensão."""
    from torchvision import models

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        dim = model.heads.head.in_features
        model.heads = nn.Identity()
    else:
        raise ValueError(f"Backbone desconhecido: {name}. Opções: {BACKBONES}")

    return model, dim


@torch.no_grad()
def extract(
    model: nn.Module, loader: DataLoader, device: torch.device, *, log_every: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Extrai embeddings e rótulos de todos os batches de um data loader."""
    model.eval()
    model.to(device)

    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    started_at = time.time()
    total = len(loader)

    for index, (images, targets) in enumerate(loader, start=1):
        output = model(images.to(device, non_blocking=True))
        features.append(output.float().cpu().numpy())
        labels.append(targets.numpy())

        if index % log_every == 0 or index == total:
            elapsed = time.time() - started_at
            restante = elapsed / index * (total - index)
            logger.info("  lote %d/%d (~%.0fs restantes)", index, total, restante)

    return np.concatenate(features), np.concatenate(labels)


def feature_paths(backbone: str, split: str) -> tuple:
    """Retorna caminhos dos arquivos de embeddings e rótulos de um split."""
    return (
        FEATURES_DIR / f"{backbone}_{split}_X.npy",
        FEATURES_DIR / f"{backbone}_{split}_y.npy",
    )


def load_features(backbone: str = "resnet50", split: str = "train"):
    """Carrega os arrays salvos de embeddings e rótulos de um split."""
    path_x, path_y = feature_paths(backbone, split)
    if not path_x.exists():
        raise FileNotFoundError(
            f"Embeddings não encontrados: {path_x}\n"
            f"Rode `python -m dogs.features --backbone {backbone}`, "
            "ou baixe a pasta features/ do Drive compartilhado."
        )

    X, y = np.load(path_x), np.load(path_y)
    if len(X) != len(y):
        raise ValueError(f"X e y desalinhados em {split}: {len(X)} vs {len(y)}")

    return X, y


def make_feature_loader(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int = 256,
    shuffle: bool,
) -> DataLoader:
    """Cria um loader de embeddings com batch final descartado apenas no treino.

    O ``drop_last`` no treino evita um último lote unitário, incompatível com o
    ``BatchNorm1d`` usado pelo linear probe. Na validação ele permanece desligado
    para que cada logit salvo corresponda exatamente a um rótulo.
    """
    if len(features) != len(labels):
        raise ValueError(f"features e labels desalinhados: {len(features)} vs {len(labels)}")

    dataset = TensorDataset(
        torch.from_numpy(features).float(), torch.from_numpy(labels).long()
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="resnet50", choices=BACKBONES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recomputa mesmo se os .npy já existirem.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S"
    )
    ensure_dirs()

    config = TrainConfig(
        experiment_name=f"features_{args.backbone}",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=False,
    )

    data = load_data(config)
    model, dim = build_backbone(args.backbone)
    device = get_device()
    logger.info("Backbone %s (dim=%d) em %s", args.backbone, dim, device)

    for split_name, loader in (
        ("train", data.train_loader),
        ("val", data.val_loader),
        ("test", data.test_loader),
    ):
        path_x, path_y = feature_paths(args.backbone, split_name)
        if path_x.exists() and not args.force:
            logger.info("%s já existe, pulando (use --force para refazer)", path_x.name)
            continue

        logger.info("Extraindo %s ...", split_name)
        features, labels = extract(model, loader, device)
        np.save(path_x, features)
        np.save(path_y, labels)
        logger.info("  %s salvo: %s", split_name, features.shape)

    tamanho_mb = sum(p.stat().st_size for p in FEATURES_DIR.glob("*.npy")) / 1e6
    logger.info("Pronto. %s (%.0f MB)", FEATURES_DIR, tamanho_mb)

    if (cache := drive_cache_dir()) is not None:
        logger.info("Drive montado: copie para %s/features/ e avise a Mari.", cache)
    else:
        logger.info("Drive não montado — suba a pasta features/ manualmente.")


if __name__ == "__main__":
    main()
