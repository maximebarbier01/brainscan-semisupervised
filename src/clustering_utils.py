"""Utilitaires reutilisables pour l'analyse non supervisee des embeddings."""

from __future__ import annotations

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
import umap


# **************#
# * Parametres *#
# **************#

LABEL_NAME_MAP = {
    0: "normal",
    1: "cancer",
    -1: "noise_or_unknown",
}


# ***************#
# * Definitions *#
# ***************#


def compute_noise_ratio(labels: np.ndarray) -> float:
    """Calcule la part d'observations marquées comme bruit par DBSCAN."""
    return float(np.mean(labels == -1))


def count_clusters(labels: np.ndarray) -> int:
    """Compte le nombre de clusters réels en ignorant le label de bruit -1."""
    unique_labels = set(labels)
    return len(unique_labels) - (1 if -1 in unique_labels else 0)


def safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float:
    """Calcule le silhouette score seulement si la configuration est exploitable."""
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]

    # Le score de silhouette n'a de sens que s'il reste au moins deux clusters.
    if valid_mask.sum() < 2:
        return np.nan
    if len(np.unique(valid_labels)) < 2:
        return np.nan

    return float(silhouette_score(features[valid_mask], valid_labels))


def compute_pca_variance_tables(
    X_scaled: np.ndarray,
    *,
    pca_component_grid: list[int],
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construit les tables de variance expliquee utiles a l'analyse PCA."""
    pca_full = PCA(random_state=random_seed)
    pca_full.fit(X_scaled)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    pca_variance_df = pd.DataFrame(
        {
            "n_components": np.arange(1, len(cumulative_variance) + 1),
            "explained_variance_ratio": pca_full.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative_variance,
        }
    )

    pca_component_summary_df = pd.DataFrame(
        {
            "n_components": pca_component_grid,
            "cumulative_explained_variance": [
                float(cumulative_variance[n_components - 1])
                for n_components in pca_component_grid
            ],
        }
    )

    return pca_variance_df, pca_component_summary_df


def build_projection_spaces(
    X_scaled: np.ndarray,
    *,
    pca_component_grid: list[int],
    random_seed: int,
    tsne_perplexity: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    tsne_source_components: int = 50,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Projette les embeddings dans plusieurs espaces pour clustering et visualisation."""
    pca_embeddings: dict[int, np.ndarray] = {}
    for n_components in pca_component_grid:
        pca_embeddings[n_components] = PCA(
            n_components=n_components,
            random_state=random_seed,
        ).fit_transform(X_scaled)

    if tsne_source_components not in pca_embeddings:
        raise ValueError(
            "La dimension PCA utilisee pour t-SNE/UMAP doit etre presente dans pca_component_grid."
        )

    # t-SNE et UMAP partent du même espace PCA intermédiaire pour comparer des projections 2D coherentes
    X_tsne2 = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_seed,
    ).fit_transform(pca_embeddings[tsne_source_components])

    X_umap2 = umap.UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=random_seed,
    ).fit_transform(pca_embeddings[tsne_source_components])

    return pca_embeddings, X_tsne2, X_umap2


def build_embedding_visualization_frame(
    feature_table_df: pd.DataFrame,
    X_pca2: np.ndarray,
    X_tsne2: np.ndarray,
    X_umap2: np.ndarray,
) -> pd.DataFrame:
    """Assemble les metadonnees utiles et les coordonnees 2D des projections."""
    embedding_viz_df = feature_table_df[
        [
            "relative_path",
            "source_split",
            "label_group",
            "label_strong",
            "label_strong_name",
            "y_ssl",
        ]
    ].copy()
    embedding_viz_df[["pca_1", "pca_2"]] = X_pca2
    embedding_viz_df[["tsne_1", "tsne_2"]] = X_tsne2
    embedding_viz_df[["umap_1", "umap_2"]] = X_umap2
    return embedding_viz_df


def evaluate_kmeans_grid(
    pca_embeddings: dict[int, np.ndarray],
    X_scaled: np.ndarray,
    *,
    pca_component_grid: list[int],
    strong_mask: np.ndarray,
    y_strong: np.ndarray,
    random_seed: int,
) -> pd.DataFrame:
    """Compare KMeans sur plusieurs espaces projetes avec le meme protocole."""
    comparison_rows: list[dict[str, object]] = []

    for n_components in pca_component_grid:
        features_for_clustering = pca_embeddings[n_components]
        model = KMeans(n_clusters=2, random_state=random_seed, n_init=20)
        labels = model.fit_predict(features_for_clustering)

        comparison_rows.append(
            {
                "method": "KMeans",
                "space": f"pca{n_components}",
                "ari_strong": float(adjusted_rand_score(y_strong, labels[strong_mask])),
                "silhouette": safe_silhouette(features_for_clustering, labels),
                "n_clusters": count_clusters(labels),
                "noise_ratio": 0.0,
            }
        )

    # Point de comparaison supplementaire : clustering directement dans l'espace standardise.
    kmeans_scaled = KMeans(n_clusters=2, random_state=random_seed, n_init=20)
    kmeans_scaled_labels = kmeans_scaled.fit_predict(X_scaled)
    comparison_rows.append(
        {
            "method": "KMeans",
            "space": "scaled_features",
            "ari_strong": float(
                adjusted_rand_score(y_strong, kmeans_scaled_labels[strong_mask])
            ),
            "silhouette": safe_silhouette(X_scaled, kmeans_scaled_labels),
            "n_clusters": count_clusters(kmeans_scaled_labels),
            "noise_ratio": 0.0,
        }
    )

    return pd.DataFrame(comparison_rows)


def evaluate_dbscan_configs(
    space_lookup: dict[str, np.ndarray],
    dbscan_configs: list[dict[str, object]],
    *,
    strong_mask: np.ndarray,
    y_strong: np.ndarray,
) -> pd.DataFrame:
    """Compare plusieurs reglages DBSCAN sur les espaces fournis."""
    comparison_rows: list[dict[str, object]] = []

    for config in dbscan_configs:
        features_for_clustering = space_lookup[str(config["space"])]
        model = DBSCAN(
            eps=float(config["eps"]),
            min_samples=int(config["min_samples"]),
        )
        labels = model.fit_predict(features_for_clustering)

        comparison_rows.append(
            {
                "method": "DBSCAN",
                "space": config["space"],
                "ari_strong": float(adjusted_rand_score(y_strong, labels[strong_mask])),
                "silhouette": safe_silhouette(features_for_clustering, labels),
                "n_clusters": count_clusters(labels),
                "noise_ratio": compute_noise_ratio(labels),
                "eps": float(config["eps"]),
                "min_samples": int(config["min_samples"]),
            }
        )

    return pd.DataFrame(comparison_rows)


def build_clustering_comparison_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatene et trie les resultats pour faciliter la lecture du benchmark."""
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ari_strong", "silhouette"], ascending=[False, False])
        .reset_index(drop=True)
    )


def majority_vote_mapping(
    cluster_labels: np.ndarray, true_labels: np.ndarray
) -> dict[int, int]:
    """Associe chaque cluster a la classe forte majoritaire observee."""
    mapping: dict[int, int] = {}
    temp_df = pd.DataFrame(
        {
            "cluster": cluster_labels,
            "true_label": true_labels,
        }
    )
    temp_df = temp_df.loc[temp_df["cluster"] != -1].copy()

    for cluster_id, cluster_df in temp_df.groupby("cluster"):
        mapping[int(cluster_id)] = int(cluster_df["true_label"].mode().iloc[0])

    return mapping


def map_clusters_to_weak_labels(
    cluster_labels: np.ndarray,
    mapping: dict[int, int],
    *,
    noise_value: int = -1,
) -> np.ndarray:
    """Convertit les ids de clusters en pseudo-labels metiers."""
    weak_labels = []
    for cluster_id in cluster_labels:
        weak_labels.append(mapping.get(int(cluster_id), noise_value))
    return np.asarray(weak_labels, dtype=int)


def label_names_from_int(values: np.ndarray) -> np.ndarray:
    """Traduit les labels entiers en libelles lisibles pour les tableaux et figures."""
    return np.asarray([LABEL_NAME_MAP.get(int(value), "unknown") for value in values])


def build_mapping_summary_df(mapping: dict[int, int]) -> pd.DataFrame:
    """Construit un petit tableau recapitulatif du sens donne a chaque cluster."""
    mapped_values = np.asarray(list(mapping.values()), dtype=int)
    return pd.DataFrame(
        {
            "cluster_id": list(mapping.keys()),
            "mapped_strong_label": list(mapping.values()),
            "mapped_strong_label_name": label_names_from_int(mapped_values),
        }
    )


def scatter_embedding(
    ax: Axes,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    *,
    alpha: float = 0.75,
    s: int = 18,
) -> None:
    """Trace un nuage 2D avec une mise en forme cohérente dans le notebook."""
    sns.scatterplot(
        data=frame,
        x=x_col,
        y=y_col,
        hue=hue_col,
        alpha=alpha,
        s=s,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
