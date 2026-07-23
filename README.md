# BrainScan Semi-Supervised

Projet OpenClassrooms autour de l'analyse d'images medicales avec des methodes non supervisees et semi-supervisees pour la detection de tumeurs cerebrales.

## Objectifs

- Explorer le jeu de radiographies et verifier sa coherence.
- Extraire des embeddings visuels avec un modele pre-entraine.
- Generer des labels faibles via clustering sans melanger labels forts et faibles.
- Comparer un entrainement supervise seul a un entrainement semi-supervise.

## Structure

```text
brainscan-semisupervised/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
├── .gitignore
├── environment.yml
└── README.md
```

## Mise en place

Creer ou mettre a jour l'environnement:

```bash
mamba env update -n brainscan -f environment.yml --prune
conda activate brainscan
python -m ipykernel install --user --name brainscan --display-name "Python (brainscan)"
```

Installer PyTorch avec support GPU dans WSL:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verification rapide:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
```

## Workflow

1. `notebooks/01_eda.ipynb` : import, inspection visuelle, resolution, canaux, structure du dataset.
2. `notebooks/02_feature_extraction.ipynb` : preprocessing et extraction d'embeddings.
3. `notebooks/03_unsupervised_analysis.ipynb` : PCA/t-SNE, clustering et labels faibles.
4. `notebooks/04_semi_supervised_training.ipynb` : comparaison supervise vs semi-supervise.

## Regles de projet

- Les donnees brutes restent dans `data/raw/`.
- Les labels faibles et forts doivent rester separes dans les artefacts et dans le code.
- Les poids de modeles et sorties lourdes ne sont pas versionnes.
- Les notebooks servent a l'exploration; la logique reutilisable doit migrer dans `src/`.
