# Wavelength 🎵

A multi-model music genre classifier trained on 114,000+ Spotify tracks across 10 genres. Built as a final project for DS 4400 (Machine Learning and Data Intelligence) at Northeastern University.

## Overview

Wavelength explores whether audio-based features like tempo, energy, and acousticness can reliably distinguish music genres. We compare multiple classical ML approaches and analyze where genre overlap creates classification noise.

**Team:** Elyssa Querubin & Suhana Singh Madia

---

## Dataset

- **Source:** [Kaggle Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Size:** ~114,000 tracks
- **Genres (10):** Electronic, Anime, Jazz, Alternative, Country, Rap, R&B, Soul, Classical, Pop
- **Features:** Tempo, energy, danceability, acousticness, instrumentalness, valence, loudness, speechiness, and more

---

## Methodology

### Preprocessing (Elyssa)
- Removed nulls and duplicates
- Normalized continuous features
- Applied PCA for dimensionality reduction (6 components, ~90% variance retained)

### Models

| Model | Accuracy | Owner |
|---|---|---|
| k-NN | 55.06% | Elyssa |
| Decision Tree | 59.21% | Elyssa |
| Logistic Regression | ~65% | Suhana |
| Random Forest | ~70% | Suhana |

**Best model:** Random Forest (~70% accuracy on a 10-class task)

#### k-NN config
- k = 24, Manhattan distance, distance-weighted voting

#### Decision Tree config
- max_depth = 10, Gini impurity, tuned via GridSearchCV

---

## Results

~70% accuracy on a 10-genre classification task is competitive given the degree of label overlap in genres like R&B, Soul, and Pop. Confusion matrices show the classifier performs strongest on genres with distinct audio signatures (Classical, Electronic) and weakest on overlapping ones (R&B vs. Soul).

Evaluation includes:
- Confusion matrices per model
- ROC curves (one-vs-rest)
- Cross-validation (5-fold) to validate generalization

---

## Stack

- Python, scikit-learn, pandas, NumPy
- Matplotlib / Seaborn for visualizations
- Jupyter Notebook
