"""Carregamento, divisão e transformações de imagem do Stanford Dogs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from dogs.config import (
    HF_DATASET,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
    TrainConfig,
    dataset_cache_dir,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


def build_transforms(
    image_size: int, *, train: bool, augment: bool
) -> transforms.Compose:
    """Cria transformações de treinamento ou de avaliação determinística."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train and augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class HFImageDataset(Dataset):
    """Adapta um split do Hugging Face à interface de datasets do PyTorch."""

    def __init__(self, hf_split, transform: transforms.Compose) -> None:
        self._split = hf_split
        self._transform = transform

    def __len__(self) -> int:
        return len(self._split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self._split[int(index)]
        image = record["image"].convert("RGB")
        return self._transform(image), int(record["label"])


@dataclass
class DataBundle:
    """Data loaders e rótulos de classe de um experimento de treinamento."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@lru_cache(maxsize=1)
def _load_raw():
    """Carrega e valida o dataset do Hugging Face armazenado em cache."""
    from datasets import load_dataset

    ensure_dirs()
    cache = dataset_cache_dir()
    logger.info("Carregando %s (cache: %s)", HF_DATASET, cache)
    dataset = load_dataset(HF_DATASET, cache_dir=str(cache))

    faltando = {"train", "test"} - set(dataset.keys())
    if faltando:
        raise RuntimeError(
            f"Splits ausentes em {HF_DATASET}: {faltando}. "
            f"Disponíveis: {list(dataset)}.\n"
            "Datasets no formato FiftyOne trazem tudo num split só e sem "
            "rótulos — confira se HF_DATASET aponta para um dataset parquet."
        )

    features = dataset["train"].features
    if "label" not in features or "image" not in features:
        raise RuntimeError(
            f"{HF_DATASET} não tem as colunas esperadas. "
            f"Encontradas: {list(features)}; esperadas: 'image' e 'label'."
        )
    if not hasattr(features["label"], "names"):
        raise RuntimeError(
            f"A coluna 'label' de {HF_DATASET} não é ClassLabel "
            f"(é {type(features['label']).__name__}), então não há nomes de classe."
        )

    return dataset


def load_data(config: TrainConfig) -> DataBundle:
    """Cria data loaders com uma divisão treino-validação determinística."""
    dataset = _load_raw()
    class_names = list(dataset["train"].features["label"].names)

    train_transform = build_transforms(
        config.image_size, train=True, augment=config.use_augmentation
    )
    eval_transform = build_transforms(config.image_size, train=False, augment=False)

    full_train = dataset["train"]

    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(len(full_train), generator=generator).tolist()
    split_at = int(len(full_train) * (1 - config.val_fraction))

    train_ds = Subset(HFImageDataset(full_train, train_transform), permutation[:split_at])
    val_ds = Subset(HFImageDataset(full_train, eval_transform), permutation[split_at:])
    test_ds = HFImageDataset(dataset["test"], eval_transform)

    logger.info(
        "Split: treino=%d val=%d teste=%d | %d classes",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        len(class_names),
    )

    def make_loader(source: Dataset, *, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            source,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
            persistent_workers=config.num_workers > 0,
        )

    return DataBundle(
        train_loader=make_loader(train_ds, shuffle=True, drop_last=True),
        val_loader=make_loader(val_ds, shuffle=False),
        test_loader=make_loader(test_ds, shuffle=False),
        class_names=class_names,
    )
