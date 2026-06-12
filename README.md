# WrldCup - WC Match Predictor

A machine learning project to predict outcomes of international football (soccer) matches, with a focus on World Cup fixtures.

## Goal

Predict match results (Home Win / Draw / Away Win) using historical match data, team form, and rating-based features — and eventually generate live predictions for upcoming World Cup matches.

## Project Structure
worldcup-predictor/

├── data/

│   ├── raw/              # raw downloaded datasets

│   └── processed/        # cleaned, feature-engineered data

├── src/

│   ├── data_prep.py      # data loading and cleaning

│   ├── features.py        # feature engineering (Elo ratings, form, etc.)

│   ├── train.py           # model training

│   └── predict.py         # generate predictions

├── models/                # saved trained models

├── requirements.txt

└── main.py

## Approach

**Data**: Historical international football results (1990–present), sourced from publicly available match datasets.

**Features**:
- Elo ratings for each team, updated chronologically after every match
- Recent form — rolling averages of goals scored/conceded over the last 5–10 matches
- Head-to-head win rate between the two teams
- Home advantage / neutral venue flag

**Models**:
- *Baseline*: Logistic Regression — a simple multiclass classifier to sanity-check the pipeline and set a benchmark
- *Primary*: XGBoost — gradient-boosted decision trees, generally stronger on tabular data with mixed feature types
- *Possible extension*: Poisson regression to model goals scored by each team independently, then derive win/draw/loss probabilities (the classic Dixon-Coles approach used in football analytics)

**Evaluation**:
- Chronological train/test split (no shuffling — train on past matches, test on future ones to avoid lookahead bias)
- Metrics: accuracy, log loss, and confusion matrix

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Status

🚧 Work in progress — data pipeline and feature engineering in development.