import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix


def main():
    # 1. Load data
    df = pd.read_csv("../data/processed/final_features.csv")

    # 2. Feature columns
    feature_cols = [
        "home_elo", "away_elo", "elo_diff",
        "neutral_venue",
        "home_form_goals_scored", "home_form_goals_conceded",
        "away_form_goals_scored", "away_form_goals_conceded",
    ]
    X = df[feature_cols]
    y = df["result"]

    # 3. Chronological train/test split (first 80% train, last 20% test)
    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")
    print()

    # 4. Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # 6. Evaluation
    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Log loss:  {ll:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 7. Save model
    joblib.dump(model, "../models/logistic_regression.pkl")
    print("\nModel saved to models/logistic_regression.pkl")


if __name__ == "__main__":
    main()
