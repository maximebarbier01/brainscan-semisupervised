"""Shared helpers for project notebooks."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def configure_notebook(
    *,
    display_max_columns: int = 50,
    display_max_rows: int = 100,
    figure_size: tuple[int, int] = (10, 6),
    style: str = "whitegrid",
) -> None:
    """Apply a consistent plotting and display configuration in notebooks."""
    sns.set_theme(style=style)
    plt.rcParams["figure.figsize"] = figure_size
    pd.set_option("display.max_columns", display_max_columns)
    pd.set_option("display.max_rows", display_max_rows)


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root from the current working directory."""
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "environment.yml").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Impossible de localiser la racine du projet.")


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    """Create a filesystem-friendly slug from a display name."""
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def build_figure_saver(
    figure_dir: Path,
    *,
    enabled: bool = True,
    default_dpi: int = 200,
) -> Callable[[plt.Figure, str, int], Path | None]:
    """Return a notebook-friendly figure saver bound to a directory."""
    ensure_directory(figure_dir)

    def save_figure(fig: plt.Figure, figure_name: str, dpi: int = default_dpi) -> Path | None:
        if not enabled:
            return None
        figure_path = figure_dir / f"{slugify(figure_name)}.png"
        fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure sauvegardée : {figure_path}")
        return figure_path

    return save_figure


def set_global_seed(seed: int, *, torch_module=None) -> None:
    """Set seeds for Python, NumPy and optionally PyTorch."""
    random.seed(seed)
    np.random.seed(seed)

    if torch_module is not None:
        torch_module.manual_seed(seed)
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)
