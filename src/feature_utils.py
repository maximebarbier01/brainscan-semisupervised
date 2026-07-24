"""Utilitaires reutilisables pour l'extraction des features image."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

# **************#
# * Parametres *#
# **************#

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE = (224, 224)


# ***************#
# * Definitions *#
# ***************#


def build_preprocess(
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Construit le pipeline de pretraitement attendu par ResNet18."""
    return transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def denormalize_image(
    tensor: torch.Tensor,
    *,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    """Ramene un tenseur normalise vers une image affichable dans matplotlib."""
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image * np.array(std)) + np.array(mean)
    return np.clip(image, 0.0, 1.0)


class BrainScanImageDataset(Dataset):
    """Dataset PyTorch base sur l'index construit pendant l'EDA."""

    def __init__(self, frame: pd.DataFrame, project_root: Path, transform: transforms.Compose):
        self.frame = frame.reset_index(drop=True).copy()
        self.project_root = project_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.frame.iloc[idx]
        image_path = self.project_root / row["relative_path"]

        with Image.open(image_path) as image:
            # On harmonise tout en RGB pour rester compatible avec le modele pre-entraine.
            image_tensor = self.transform(image.convert("RGB"))

        # La valeur -1 permet de marquer explicitement les observations sans label fort.
        label_strong_value = -1 if pd.isna(row["label_strong"]) else int(row["label_strong"])
        label_strong_name = row["label_strong_name"] if pd.notna(row["label_strong_name"]) else "unlabeled"

        return {
            "image": image_tensor,
            "relative_path": row["relative_path"],
            "source_split": row["source_split"],
            "label_group": row["label_group"],
            "label_strong": label_strong_value,
            "label_strong_name": label_strong_name,
            "y_ssl": int(row["y_ssl"]),
        }


def build_feature_extractor(
    *,
    weights: ResNet18_Weights | None = ResNet18_Weights.DEFAULT,
    device: torch.device,
) -> nn.Module:
    """Charge ResNet18 et remplace sa tete de classification par une sortie d'embedding."""
    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False

    # On conserve uniquement le backbone convolutionnel pour recuperer un vecteur de features.
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def infer_embedding_dim(model: nn.Module, image_size: tuple[int, int], device: torch.device) -> int:
    """Estime la dimension de sortie des embeddings a partir d'un batch factice."""
    dummy_batch = torch.zeros(1, 3, image_size[0], image_size[1], device=device)
    return int(model(dummy_batch).shape[1])


def infer_feature_dim(model: nn.Module, image_size: tuple[int, int], device: torch.device) -> int:
    """Alias conserve pour eviter de casser le notebook existant."""
    return infer_embedding_dim(model, image_size=image_size, device=device)


@torch.inference_mode()
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[pd.DataFrame, np.ndarray]:
    """Extrait un embedding par image et conserve les metadonnees alignees."""
    metadata_batches: list[pd.DataFrame] = []
    feature_batches: list[np.ndarray] = []

    for batch in tqdm(loader, desc="Extraction des embeddings"):
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        embeddings = model(images).detach().cpu().numpy().astype(np.float32)

        feature_batches.append(embeddings)
        metadata_batches.append(
            pd.DataFrame(
                {
                    "relative_path": batch["relative_path"],
                    "source_split": batch["source_split"],
                    "label_group": batch["label_group"],
                    "label_strong_name": batch["label_strong_name"],
                    "label_strong": batch["label_strong"].cpu().numpy().astype(int),
                    "y_ssl": batch["y_ssl"].cpu().numpy().astype(int),
                }
            )
        )

    metadata_df = pd.concat(metadata_batches, ignore_index=True)
    feature_array = np.concatenate(feature_batches, axis=0)
    return metadata_df, feature_array


def build_feature_table(metadata_df: pd.DataFrame, feature_array: np.ndarray) -> pd.DataFrame:
    """Assemble les metadonnees et les dimensions d'embedding dans une meme table."""
    feature_columns = [f"feat_{idx:04d}" for idx in range(feature_array.shape[1])]
    feature_df = pd.DataFrame(feature_array, columns=feature_columns)
    feature_table_df = pd.concat([metadata_df.reset_index(drop=True), feature_df], axis=1)
    feature_table_df["embedding_l2_norm"] = np.linalg.norm(feature_array, axis=1)
    return feature_table_df


# *****************#
# * Visualisation *#
# *****************#


def show_preprocessing_examples(
    df: pd.DataFrame,
    *,
    project_root: Path,
    preprocess: transforms.Compose,
    random_seed: int = 42,
) -> plt.Figure | None:
    """Affiche un exemple brut et pretraite pour chaque groupe disponible."""
    sample_groups = ["cancer", "normal", "unlabeled"]
    sample_rows: list[pd.DataFrame] = []

    for group in sample_groups:
        subset = df.loc[df["label_group"] == group]
        if not subset.empty:
            sample_rows.append(subset.sample(n=1, random_state=random_seed))

    if not sample_rows:
        return None

    sample_df = pd.concat(sample_rows, ignore_index=True)
    fig, axes = plt.subplots(len(sample_df), 2, figsize=(10, 4 * len(sample_df)))

    if len(sample_df) == 1:
        axes = np.array([axes])

    for row_idx, row in sample_df.iterrows():
        image_path = project_root / row["relative_path"]
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            processed_image = preprocess(rgb_image)

        axes[row_idx, 0].imshow(rgb_image)
        axes[row_idx, 0].set_title(f"Original - {row['label_group']}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(denormalize_image(processed_image))
        axes[row_idx, 1].set_title(f"Pretraitee - {row['label_group']}")
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    return fig
