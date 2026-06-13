from collections import defaultdict, deque
import pandas as pd


def main():
    # 1. Load cleaned data
    df = pd.read_csv("data/processed/cleaned_matches.csv")

    # --- Pre-compute dataset-wide average goals per team per match ---
    total_goals = df["home_score"].sum() + df["away_score"].sum()
    num_matches = len(df)
    avg_goals_per_team_per_match = total_goals / (2 * num_matches)

    # --- Elo rating system ---
    K = 30
    initial_elo = 1500
    elo_ratings = {}

    # --- Rolling form tracking ---
    team_history = defaultdict(lambda: deque(maxlen=5))

    # Lists to collect new columns
    home_elo_list = []
    away_elo_list = []
    elo_diff_list = []
    neutral_venue_list = []
    home_form_gs_list = []
    home_form_gc_list = []
    away_form_gs_list = []
    away_form_gc_list = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]

        # --- Neutral venue ---
        neutral_venue_list.append(1 if row["neutral"] else 0)

        # --- Current Elo ratings ---
        elo_home = elo_ratings.get(home, initial_elo)
        elo_away = elo_ratings.get(away, initial_elo)
        home_elo_list.append(elo_home)
        away_elo_list.append(elo_away)
        elo_diff_list.append(elo_home - elo_away)

        # --- Rolling form features (from PAST matches only) ---
        home_hist = team_history[home]
        away_hist = team_history[away]

        if len(home_hist) > 0:
            gs, gc = zip(*home_hist)
            home_form_gs_list.append(sum(gs) / len(gs))
            home_form_gc_list.append(sum(gc) / len(gc))
        else:
            home_form_gs_list.append(avg_goals_per_team_per_match)
            home_form_gc_list.append(avg_goals_per_team_per_match)

        if len(away_hist) > 0:
            gs, gc = zip(*away_hist)
            away_form_gs_list.append(sum(gs) / len(gs))
            away_form_gc_list.append(sum(gc) / len(gc))
        else:
            away_form_gs_list.append(avg_goals_per_team_per_match)
            away_form_gc_list.append(avg_goals_per_team_per_match)

        # --- Update team histories with THIS match (for future matches) ---
        team_history[home].append((home_score, away_score))
        team_history[away].append((away_score, home_score))

        # --- Elo update ---
        E_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
        E_away = 1 - E_home

        result = row["result"]
        if result == "H":
            S_home = 1.0
            S_away = 0.0
        elif result == "D":
            S_home = 0.5
            S_away = 0.5
        else:
            S_home = 0.0
            S_away = 1.0

        elo_ratings[home] = elo_home + K * (S_home - E_home)
        elo_ratings[away] = elo_away + K * (S_away - E_away)

    # --- Add all new columns ---
    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["elo_diff"] = elo_diff_list
    df["neutral_venue"] = neutral_venue_list
    df["home_form_goals_scored"] = home_form_gs_list
    df["home_form_goals_conceded"] = home_form_gc_list
    df["away_form_goals_scored"] = away_form_gs_list
    df["away_form_goals_conceded"] = away_form_gc_list

    # --- Save to final_features.csv ---
    df.to_csv("data/processed/final_features.csv", index=False)

    # --- Print summary ---
    print("Column names:", list(df.columns), "\n")

    cols = [
        "date", "home_team", "away_team",
        "home_elo", "away_elo", "elo_diff", "neutral_venue",
        "home_form_goals_scored", "home_form_goals_conceded",
        "away_form_goals_scored", "away_form_goals_conceded",
        "result",
    ]
    print("First 5 rows:")
    print(df[cols].head().to_string(index=False))

    print(f"\nShape: {df.shape}  (rows preserved)")

    print("\nTop 10 teams by final Elo rating:")
    top10 = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    for rank, (team, elo) in enumerate(top10, 1):
        print(f"  {rank:2d}. {team:25s} {elo:.1f}")


if __name__ == "__main__":
    main()
