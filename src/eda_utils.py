from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from tqdm.auto import tqdm

from src.notebook_utils import find_project_root, build_figure_saver, ensure_directory

# **************#
# * Paramètres *#
# **************#

# Recherche automatiquement la racine du projet
PROJECT_ROOT = find_project_root()

# Dossiers utilisés dans le notebook
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
EDA_OUTPUT_DIR = ensure_directory(PROJECT_ROOT / "data" / "interim")
FIGURES_DIR = ensure_directory(PROJECT_ROOT / "reports" / "figures" / "eda")

# Extensions reconnues selon le type de fichier
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}
TEXT_EXTENSIONS = {".txt", ".md"}

# Correspondance entre le nom du dossier et le label numérique
LABEL_MAP = {"normal": 0, "cancer": 1}

# Correspondance entre l'arborescence et le pool de données
SPLIT_DIR_MAP = {
    "avec_labels": "strong_labeled_pool",
    "sans_label": "unlabeled_pool",
}


SAVE_FIGURES = True
save_figure = build_figure_saver(FIGURES_DIR, enabled=SAVE_FIGURES)

# ***************#
# * Définitions *#
# ***************#


def list_files_by_suffix(root: Path, suffixes: set[str]) -> list[Path]:
    """Liste récursivement les fichiers ayant une extension autorisée."""
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    )


def infer_source_split(relative_path: Path) -> str:
    """Déduit le pool (avec ou sans label) d'une image à partir de son chemin."""
    lower_parts = {part.lower() for part in relative_path.parts}
    for folder_name, split_name in SPLIT_DIR_MAP.items():
        if folder_name in lower_parts:
            return split_name
    return "unknown_pool"


def infer_strong_label(relative_path: Path, source_split: str) -> tuple[object, object]:
    """Déduit le label fort depuis les dossiers normal ou cancer."""

    # Les images non labellisées n'ont pas de label fort
    if source_split != "strong_labeled_pool":
        return pd.NA, pd.NA

    lower_parts = [part.lower() for part in relative_path.parts]
    for label_name, label_value in LABEL_MAP.items():
        if label_name in lower_parts:
            return label_name, label_value
    return pd.NA, pd.NA


def safe_inspect_image(path: Path, project_root: Path) -> dict:
    """Extrait les métadonnées d'une image sans interrompre le scan."""

    # Chemin relatif utilisé pour rendre l'index portable
    relative_path = path.relative_to(project_root)

    # Identification du pool et du label depuis l'arborescence
    source_split = infer_source_split(relative_path)

    label_name, label_value = infer_strong_label(relative_path, source_split)

    # Métadonnées disponibles même si l'image est corrompue
    base_row = {
        "relative_path": relative_path.as_posix(),
        "file_name": path.name,
        "file_stem": path.stem,
        "parent_dir": path.parent.name,
        "source_split": source_split,
        "label_strong_name": label_name,
        "label_strong": label_value,
        "is_strongly_labeled": pd.notna(label_value),
        # Convention scikit-learn : -1 représente une observation non labellisée
        "y_ssl": int(label_value) if pd.notna(label_value) else -1,
        "suffix": path.suffix.lower(),
        "file_size_kb": round(path.stat().st_size / 1024, 2),
    }

    try:
        with Image.open(path) as image:
            # Version RGB utilisée pour comparer les canaux
            rgb_image = image.convert("RGB")
            # Version en niveaux de gris utilisée pour les statistiques
            gray_image = image.convert("L")
            width, height = image.size
            bands = image.getbands()

            # int16 évite les débordements lors des soustractions
            rgb_array = np.asarray(rgb_image, dtype=np.int16)
            gray_array = np.asarray(gray_image, dtype=np.float32)

            # Vérifie si les trois canaux RGB sont strictement identiques
            rgb_channels_identical = bool(
                np.array_equal(rgb_array[:, :, 0], rgb_array[:, :, 1])
                and np.array_equal(rgb_array[:, :, 1], rgb_array[:, :, 2])
            )

            # Mesure le plus grand écart observé entre deux canaux
            rgb_channel_max_diff = int(
                max(
                    np.abs(rgb_array[:, :, 0] - rgb_array[:, :, 1]).max(),
                    np.abs(rgb_array[:, :, 1] - rgb_array[:, :, 2]).max(),
                    np.abs(rgb_array[:, :, 0] - rgb_array[:, :, 2]).max(),
                )
            )

            return {
                **base_row,
                "width": width,
                "height": height,
                "mode": image.mode,
                "n_channels": len(bands),
                "bands": ",".join(bands),
                # Ratio largeur / hauteur
                "aspect_ratio": round(width / height, 4) if height else np.nan,
                # Statistiques globales en niveaux de gris
                "gray_mean": round(float(gray_array.mean()), 3),
                "gray_std": round(float(gray_array.std()), 3),
                "gray_min": int(gray_array.min()),
                "gray_max": int(gray_array.max()),
                "rgb_channels_identical": rgb_channels_identical,
                "rgb_channel_max_diff": rgb_channel_max_diff,
                "is_corrupt": False,
                "error": None,
            }
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        # L'image reste dans l'index, mais elle est marquée comme corrompue
        return {
            **base_row,
            "width": np.nan,
            "height": np.nan,
            "mode": None,
            "n_channels": np.nan,
            "bands": None,
            "aspect_ratio": np.nan,
            "gray_mean": np.nan,
            "gray_std": np.nan,
            "gray_min": np.nan,
            "gray_max": np.nan,
            "rgb_channels_identical": pd.NA,
            "rgb_channel_max_diff": np.nan,
            "is_corrupt": True,
            "error": str(exc),
        }


def build_image_index(
    image_paths: list[Path], project_root: Path, limit: int | None = None
) -> pd.DataFrame:
    """Construit un DataFrame contenant les métadonnées des images."""

    # Possibilité de limiter le scan pour un premier test
    scan_paths = image_paths if limit is None else image_paths[:limit]
    rows = [
        safe_inspect_image(path, project_root)
        for path in tqdm(
            scan_paths,
            desc="Scan des images",
            unit="image",
            colour="#CA50E6",
            dynamic_ncols=True,
        )
    ]
    return pd.DataFrame(rows)


# *****************#
# * Visualisation *#
# *****************#


def show_image_grid(
    df: pd.DataFrame,
    title: str,
    caption_columns: list[str],
    figure_name: str | None = None,
    ncols: int = 3,
) -> None:
    """Affiche les images du DataFrame sous forme de grille."""
    if df.empty:
        print(f"Aucune image a afficher pour: {title}")
        return

    # Nombre de lignes nécessaire selon le nombre de colonnes
    nrows = math.ceil(len(df) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))

    # Garantit un tableau d'axes, même avec une seule image
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, (_, row) in zip(axes, df.iterrows()):
        image_path = PROJECT_ROOT / row["relative_path"]
        with Image.open(image_path) as image:
            if image.mode == "L":
                ax.imshow(image, cmap="gray")
            else:
                ax.imshow(image.convert("RGB"))
        caption = "\n".join(str(row[col]) for col in caption_columns)
        ax.set_title(caption, fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if figure_name is not None:
        save_figure(fig, figure_name)
    plt.show()


def show_sample_grid(
    df: pd.DataFrame,
    title: str,
    n: int = 9,
    seed: int = 42,
    figure_name: str | None = None,
) -> None:
    if df.empty:
        print(f"Aucun exemple a afficher pour: {title}")
        return
    sample_df = df.sample(n=min(n, len(df)), random_state=seed)
    show_image_grid(
        sample_df,
        title=title,
        caption_columns=["file_name", "source_split", "label_group"],
        figure_name=figure_name,
        ncols=3,
    )


# **********#
# * Autres *#
# **********#


def preview_table(path: Path, n_rows: int = 5) -> pd.DataFrame:
    """Affiche les premières lignes d'un fichier tabulaire."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).head(n_rows)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t").head(n_rows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path).head(n_rows)
    if suffix == ".json":
        return pd.read_json(path).head(n_rows)
    if suffix == ".parquet":
        return pd.read_parquet(path).head(n_rows)
    raise ValueError(f"Extension tabulaire non supportee: {suffix}")


def preview_text(path: Path, n_lines: int = 20) -> str:
    """Lit les premières lignes d'un fichier texte."""

    # latin-1 est testé si la lecture en UTF-8 échoue
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                lines = [next(handle) for _ in range(n_lines)]
            return "".join(lines)
        except StopIteration:
            return "".join(lines)
        except UnicodeDecodeError:
            continue
    return "[Impossible de lire le fichier texte avec utf-8 ou latin-1]"
