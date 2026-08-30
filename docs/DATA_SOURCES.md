# Dataset sources

The assignment requires one dataset. The project uses three datasets with distinct roles:

| Dataset | Role | Expected shape | Target | Source |
|---|---|---:|---|---|
| UCI Letter Recognition | Robustness: multiclass geometric features | 20,000 x 16 | 26 uppercase letters A-Z | [UCI dataset 59](https://archive.ics.uci.edu/dataset/59/letter%2Brecognition) |
| Handwritten Digits | Robustness: small image dataset | 1,797 x 64 | 10 digits 0-9 | [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) |
| UCI Covertype | Primary Decision Tree case study | 581,012 x 54 | 7 forest cover types | [UCI dataset 31](https://archive.ics.uci.edu/dataset/31/covertype) |

Run `python scripts/download_datasets.py` from the repository root. The script:

- downloads and extracts the official UCI archive to `data/raw/letter_recognition/`;
- materializes the sklearn Digits bundle to `data/raw/handwritten_digits/digits.csv`;
- downloads and materializes Covertype to `data/raw/covertype/covertype.csv`;
- writes checksums and source metadata to `data/raw/dataset_manifest.json`.

Raw data is ignored by Git. Keep `dataset_manifest.json` locally with the downloaded files and regenerate it when refreshing data.

Covertype is the primary case study used for the complete baseline, resulting-tree analysis,
error analysis and E0-E4 improvement comparison. Letter Recognition and Digits are robustness
datasets used to test whether conclusions generalize to different data regimes.

## Candidate checked but not used

The Kaggle candidate `ghnshymsaini/mnist-handwritten-digits-dataset` was inspected locally. It contains 28x28 PNGs and its downloaded train split has only labels `0`-`3` (22,544 images), while the test split has 10,000 images for labels `0`-`9`. It is therefore not the planned `load_digits()` dataset and is not copied into the project.
