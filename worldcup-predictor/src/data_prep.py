import pandas as pd


def main():
    # 1. Load raw data
    df = pd.read_csv("../data/raw/results.csv")

    # 2. Parse date and sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # 3. Filter to matches from 1990-01-01 onwards
    df = df[df["date"] >= "1990-01-01"]

    # 4. Drop rows with missing home_score or away_score
    df = df.dropna(subset=["home_score", "away_score"])

    # 5. Create target column 'result'
    def classify_result(row):
        if row["home_score"] > row["away_score"]:
            return "H"
        elif row["home_score"] == row["away_score"]:
            return "D"
        else:
            return "A"

    df["result"] = df.apply(classify_result, axis=1)

    # 6. Save cleaned data
    df.to_csv("../data/processed/cleaned_matches.csv", index=False)

    # 7. Print summary
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nResult value counts:")
    print(df["result"].value_counts().to_string())


if __name__ == "__main__":
    main()
