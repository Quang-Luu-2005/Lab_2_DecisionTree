# Dataset sources

The project plan and assignment distinguish two datasets:

| Dataset | Role | Expected shape | Target | Source |
|---|---|---:|---|---|
| UCI Letter Recognition | Primary Decision Tree study | 20,000 x 16 | 26 uppercase letters A-Z | [UCI dataset 59](https://archive.ics.uci.edu/dataset/59/letter%2Brecognition) |
| Handwritten Digits | Cross-dataset comparison | 1,797 x 64 | 10 digits 0-9 | [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) |

Run `python scripts/download_datasets.py` from the repository root. The script:

- downloads and extracts the official UCI archive to `data/raw/letter_recognition/`;
- materializes the sklearn Digits bundle to `data/raw/handwritten_digits/digits.csv`;
- writes checksums and source metadata to `data/raw/dataset_manifest.json`.

Raw data is ignored by Git. Keep `dataset_manifest.json` locally with the downloaded files and regenerate it when refreshing data.

The PDF assignment requires at least one suitable dataset. The project plan expands the scope to two: Letter Recognition remains the primary dataset for the complete baseline/improvement study, while Digits is used only for the cross-dataset representation experiment.

## Candidate checked but not used

The Kaggle candidate `ghnshymsaini/mnist-handwritten-digits-dataset` was inspected locally. It contains 28x28 PNGs and its downloaded train split has only labels `0`-`3` (22,544 images), while the test split has 10,000 images for labels `0`-`9`. It is therefore not the planned `load_digits()` dataset and is not copied into the project.
