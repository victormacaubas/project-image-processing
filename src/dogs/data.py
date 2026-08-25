"""Carregamento do Stanford Dogs, splits e transforms.

Responsabilidade única: entregar DataLoaders corretos. Nenhum modelo aqui.

STATUS: esqueleto. Implementar na segunda/terça — ver docs/roteiro.md, Passo 3.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def build_transforms(image_size: int, *, train: bool, augment: bool) -> transforms.Compose:
    """Pipeline de transformação de imagem.

    Decisões a tomar aqui:
      - Augmentation só quando train=True. Validação e teste precisam ser
        determinísticos, senão as métricas variam entre execuções e a
        comparação entre experimentos deixa de valer.
      - Normalizar com IMAGENET_MEAN/IMAGENET_STD (já em config.py) sempre que
        usar backbone pré-treinado.
      - Para eval: Resize maior que image_size, depois CenterCrop.

    Candidatos de augmentation: RandomResizedCrop, RandomHorizontalFlip,
    ColorJitter. Cuidado com flip vertical — cão de cabeça pra baixo não
    existe no mundo real e só atrapalha.
    """
    raise NotImplementedError


class HFImageDataset(Dataset):
    """Adapta um split do HuggingFace datasets para a interface do PyTorch.

    Precisa implementar __len__ e __getitem__. No __getitem__: pegar o
    registro, converter a imagem para RGB (algumas são grayscale e quebram
    o batch), aplicar o transform, devolver (tensor, label).
    """

    def __init__(self, hf_split, transform: transforms.Compose) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError


@dataclass
class DataBundle:
    """Tudo que um experimento precisa em matéria de dados."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def load_data(config) -> DataBundle:
    """Baixa o dataset, faz o split e devolve os DataLoaders.

    Regras que não podem ser violadas:
      - O split de validação sai do TREINO, nunca do teste.
      - O teste é tocado uma única vez, no fim do projeto.
      - Usar SEED de config.py no gerador do shuffle, senão o split muda
        entre execuções e os experimentos param de ser comparáveis.

    Passos:
      1. load_dataset(HF_DATASET, cache_dir=RAW_DIR)
      2. Extrair class_names de dataset["train"].features["label"].names
      3. Construir os transforms (treino com augment, eval sem)
      4. Permutar índices com generator semeado, cortar em val_fraction
      5. Envolver em Subset + HFImageDataset
      6. Montar os três DataLoaders (shuffle só no treino)
    """
    raise NotImplementedError
