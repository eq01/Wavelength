# Wavelength 🎵

A multi-model music genre classifier trained on 114,000+ Spotify tracks across 10 genres. Built as a final project for DS 4400

## Overview

Wavelength explores whether audio-based features like tempo, energy, and acousticness can reliably distinguish music genres. We compare multiple classical ML approaches and analyze where genre overlap creates classification noise.

**Team:** Elyssa Querubin & Suhana Singh Madia

---

## Dataset

- **Source:** [Kaggle Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Size:** ~114,000 tracks
- **Genres (10):** acoustic, afrobeat, alt-rock, ambient, cantopop, tango, chicago-house, forro, bluegrass, study
- **Features:** Tempo, energy, danceability, acousticness, instrumentalness, valence, speechiness, and more

---

## Methodology

### Preprocessing (Elyssa)
- Removed nulls and duplicates
- Normalized continuous features
- Feature engineering
- Applied PCA for dimensionality reduction (6 components, ~90% variance retained)

### Models

| Model | Accuracy | Owner |
|---|---|---|
| k-NN | 55.06% | Elyssa |
| Decision Tree | 59.21% | Elyssa |
| Logistic Regression | ~65% | Suhana |
| Random Forest | ~70% | Suhana |

**Best model:** Random Forest

#### k-NN config
- k = 24, Manhattan distance, distance-weighted voting

#### Decision Tree config
- max_depth = 10, Gini impurity, tuned via GridSearchCV

---

## Results

Spotify audio features can classify music genres with moderate success, though genre overlap makes the task inherently challenging. Random Forest performed best (70.77%), followed by Logistic Regression (60.91%), Decision Tree (59.21%), and k-NN (55.06%). Tree-based models outperformed linear and distance-based approaches, likely due to their ability to capture non-linear patterns in audio features. Future work could incorporate lyrics, artist metadata, or deep learning audio embeddings to improve separation on the hardest genres.

Evaluation includes:
- Confusion matrices per model
- ROC curves 
- Cross-validation

---

## Stack

- Python, scikit-learn, pandas, NumPy
- Matplotlib / Seaborn for visualizations
- Jupyter Notebook
