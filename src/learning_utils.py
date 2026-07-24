"""Utilitaires reutilisables pour l'etape d'apprentissage semi-supervise."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


# **************#
# * Parametres *#
# **************#

DEFAULT_LABEL_NAME_MAP = {
    0: "normal",
    1: "cancer",
}

DEFAULT_IMAGE_SIZE = (224, 224)


# ***************#
# * Definitions *#
# ***************#


class BrainScanClassificationDataset(Dataset):
    """Dataset PyTorch pour la classification binaire des radios cerebrales."""

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
            # On harmonise toutes les entrees en RGB pour rester compatibles avec ResNet18.
            image_tensor = self.transform(image.convert("RGB"))

        return {
            "image": image_tensor,
            "label": int(row["label"]),
            "relative_path": row["relative_path"],
            "label_name": row["label_name"],
            "dataset_role": row["dataset_role"],
        }


def prepare_learning_frames(
    image_index_df: pd.DataFrame,
    weak_label_df: pd.DataFrame,
    *,
    label_name_map: dict[int, str] = DEFAULT_LABEL_NAME_MAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit les tables fortes et faibles a partir des artefacts precedents."""
    strong_df = image_index_df.loc[
        image_index_df["source_split"] == "strong_labeled_pool",
        ["relative_path", "source_split", "label_strong", "label_strong_name"],
    ].copy()
    strong_df["label"] = strong_df["label_strong"].astype(int)
    strong_df["label_name"] = strong_df["label"].map(label_name_map)

    weak_df = weak_label_df.merge(
        image_index_df[["relative_path", "source_split"]],
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    weak_df["label"] = weak_df["weak_label_kmeans"].astype(int)
    weak_df["label_name"] = weak_df["label"].map(label_name_map)

    return strong_df, weak_df


def split_strong_labeled_data(
    strong_df: pd.DataFrame,
    *,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separe le jeu fortement labellise en train, validation et test stratifies."""
    strong_train_df, strong_temp_df = train_test_split(
        strong_df,
        test_size=validation_fraction + test_fraction,
        stratify=strong_df["label"],
        random_state=random_seed,
    )

    validation_share_within_temp = validation_fraction / (validation_fraction + test_fraction)
    strong_validation_df, strong_test_df = train_test_split(
        strong_temp_df,
        test_size=1.0 - validation_share_within_temp,
        stratify=strong_temp_df["label"],
        random_state=random_seed,
    )

    strong_train_df = strong_train_df.copy()
    strong_validation_df = strong_validation_df.copy()
    strong_test_df = strong_test_df.copy()

    strong_train_df["dataset_role"] = "strong_train"
    strong_validation_df["dataset_role"] = "strong_validation"
    strong_test_df["dataset_role"] = "strong_test"

    return strong_train_df, strong_validation_df, strong_test_df


def annotate_weak_training_frame(weak_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le role de dataset pour la phase de pre-entrainement faible."""
    weak_df = weak_df.copy()
    weak_df["dataset_role"] = "weak_train"
    return weak_df


def build_split_manifest(
    strong_train_df: pd.DataFrame,
    strong_validation_df: pd.DataFrame,
    strong_test_df: pd.DataFrame,
    weak_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit le manifeste global des splits et son resume par role."""
    split_manifest_df = pd.concat(
        [
            strong_train_df[["relative_path", "dataset_role", "label_name"]],
            strong_validation_df[["relative_path", "dataset_role", "label_name"]],
            strong_test_df[["relative_path", "dataset_role", "label_name"]],
            weak_df[["relative_path", "dataset_role", "label_name"]],
        ],
        ignore_index=True,
    )

    split_summary_df = (
        split_manifest_df.groupby(["dataset_role", "label_name"])
        .size()
        .rename("n_images")
        .reset_index()
        .sort_values(["dataset_role", "label_name"])
    )

    return split_manifest_df, split_summary_df


def build_transforms(
    *,
    weights: ResNet18_Weights = ResNet18_Weights.DEFAULT,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    train_flip_p: float = 0.5,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Construit les pipelines train et eval alignes sur les poids ImageNet."""
    weights_transform = weights.transforms()

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.RandomHorizontalFlip(p=train_flip_p),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights_transform.mean, std=weights_transform.std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=weights_transform.mean, std=weights_transform.std),
        ]
    )
    return train_transform, eval_transform


def make_loader(
    frame: pd.DataFrame,
    project_root: Path,
    transform: transforms.Compose,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Construit un DataLoader a partir d'un DataFrame de chemins et labels."""
    dataset = BrainScanClassificationDataset(frame, project_root, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_loader_summary(
    loader_specs: list[dict[str, object]],
) -> pd.DataFrame:
    """Construit un tableau recapitulatif simple des DataLoaders."""
    return pd.DataFrame(loader_specs)


def build_resnet18_binary_classifier(
    *,
    device: torch.device,
    weights: ResNet18_Weights = ResNet18_Weights.DEFAULT,
    seed: int = 42,
) -> nn.Module:
    """Charge ResNet18, gele le backbone et remplace la tete par une classification binaire."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcule les metriques principales pour la classification binaire."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


@torch.inference_mode()
def predict_with_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    split_name: str,
    label_name_map: dict[int, str] = DEFAULT_LABEL_NAME_MAP,
    pin_memory: bool = False,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Genere les predictions detaillees et les metriques associees pour un loader donne."""
    model.eval()
    rows: list[dict[str, object]] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=pin_memory)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predicted_labels = logits.argmax(dim=1).cpu().numpy().astype(int)
        true_labels = batch["label"].cpu().numpy().astype(int)

        for idx in range(len(predicted_labels)):
            rows.append(
                {
                    "split_name": split_name,
                    "relative_path": batch["relative_path"][idx],
                    "true_label": int(true_labels[idx]),
                    "true_label_name": label_name_map[int(true_labels[idx])],
                    "pred_label": int(predicted_labels[idx]),
                    "pred_label_name": label_name_map[int(predicted_labels[idx])],
                    "prob_normal": float(probabilities[idx, 0]),
                    "prob_cancer": float(probabilities[idx, 1]),
                }
            )

    predictions_df = pd.DataFrame(rows)
    metrics = compute_binary_metrics(
        predictions_df["true_label"].to_numpy(dtype=int),
        predictions_df["pred_label"].to_numpy(dtype=int),
    )
    return predictions_df, metrics


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    device: torch.device,
    stage_name: str,
    epochs: int,
    learning_rate: float,
    label_name_map: dict[int, str] = DEFAULT_LABEL_NAME_MAP,
    pin_memory: bool = False,
) -> tuple[nn.Module, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Entraine la tete de classification et conserve le meilleur etat selon le F1 validation."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_f1 = -1.0
    best_validation_predictions_df = pd.DataFrame()
    best_validation_metrics = {
        "accuracy": np.nan,
        "f1": np.nan,
        "precision": np.nan,
        "recall": np.nan,
    }
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=pin_memory)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            seen_samples += images.size(0)

        validation_predictions_df, validation_metrics = predict_with_model(
            model,
            validation_loader,
            device=device,
            split_name="validation",
            label_name_map=label_name_map,
            pin_memory=pin_memory,
        )
        epoch_train_loss = running_loss / max(seen_samples, 1)

        history_rows.append(
            {
                "stage_name": stage_name,
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                **validation_metrics,
            }
        )

        if validation_metrics["f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_validation_predictions_df = validation_predictions_df.copy()
            best_validation_metrics = validation_metrics.copy()

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history_rows)
    return model, history_df, best_validation_predictions_df, best_validation_metrics


def build_experiment_comparison_df(experiment_rows: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble les resultats finaux des experiences et les trie par F1."""
    return (
        pd.DataFrame(experiment_rows)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )


def build_confusion_table(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Construit une matrice de confusion lisible a partir des predictions."""
    matrix = confusion_matrix(
        predictions_df["true_label"],
        predictions_df["pred_label"],
        labels=[0, 1],
    )
    return pd.DataFrame(
        matrix,
        index=["true_normal", "true_cancer"],
        columns=["pred_normal", "pred_cancer"],
    )


def plot_training_history(history_comparison_df: pd.DataFrame) -> plt.Figure:
    """Trace les courbes de perte et de F1 validation pour chaque phase d'apprentissage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(
        data=history_comparison_df,
        x="epoch",
        y="train_loss",
        hue="stage_name",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Perte d'entrainement par phase")
    axes[0].set_xlabel("Epoque")
    axes[0].set_ylabel("Train loss")

    sns.lineplot(
        data=history_comparison_df,
        x="epoch",
        y="f1",
        hue="stage_name",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("F1 de validation par phase")
    axes[1].set_xlabel("Epoque")
    axes[1].set_ylabel("Validation F1")

    plt.tight_layout()
    return fig


def plot_confusion_matrices(
    confusion_specs: list[tuple[pd.DataFrame, str, str]],
) -> plt.Figure:
    """Trace plusieurs matrices de confusion sur une meme figure."""
    fig, axes = plt.subplots(1, len(confusion_specs), figsize=(6 * len(confusion_specs), 5))
    axes = np.atleast_1d(axes)

    for ax, (confusion_df, title, cmap) in zip(axes, confusion_specs):
        sns.heatmap(confusion_df, annot=True, fmt="d", cmap=cmap, ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    return fig


def combine_test_predictions(
    prediction_specs: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Concatene les predictions de test en conservant le nom de l'experience."""
    return pd.concat(
        [
            predictions_df.assign(experiment=experiment_name)
            for experiment_name, predictions_df in prediction_specs
        ],
        ignore_index=True,
    )
