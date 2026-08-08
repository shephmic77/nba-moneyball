"""
NBA Moneyball — Clean Final Reconstruction
===========================================

Goal
----
Estimate 2023-24 NBA player salary from box-score and advanced statistics,
then identify players whose statistical profile suggests more value than
their listed salary.

This final reconstruction combines the strongest ideas from the original
development notebooks:
  * v1: multivariable salary regression and predicted-vs-actual salary gap
  * v2: VORP-per-$1M as a simple secondary "value score"

Important interpretation
------------------------
This is an exploratory valuation model, not a contract-pricing system.
Salary is affected by factors the model does not include: contract timing,
rookie-scale rules, max-contract rules, injuries, team situation, free-agent
market conditions, reputation, age curves, and negotiation leverage.

The script uses the two local Basketball-Reference CSVs already preserved in
the repository and fetches 2023-24 salary data from HoopsHype at runtime.
You may instead provide a local salary CSV with --salary-csv.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


HOOPSHYPE_URL = "https://hoopshype.com/salaries/players/2023-2024/"

FEATURES = [
    "PTS", "AST", "TRB", "STL", "BLK", "MP",
    "TS%", "BPM", "VORP", "WS", "WS/48", "Age"
]


def normalize_name(value: object) -> str:
    """Normalize accents, suffixes, punctuation, and whitespace for joining."""
    if pd.isna(value):
        return ""
    name = unicodedata.normalize("NFKD", str(value))
    name = name.encode("ascii", errors="ignore").decode("ascii")
    name = re.sub(r"\b(Jr\.?|Sr\.?|II|III|IV)\b", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s'-]", "", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def clean_bref_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Rk" in df.columns:
        df = df[df["Rk"].astype(str) != "Rk"]
    if "Team" in df.columns and "Tm" not in df.columns:
        df = df.rename(columns={"Team": "Tm"})
    return df.reset_index(drop=True)


def choose_one_row_per_player(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basketball-Reference includes a TOT row plus team rows for traded players.
    Keep TOT when present; otherwise keep the row with the most games.
    """
    df = df.copy()
    if "Player-additional" in df.columns:
        key = "Player-additional"
    else:
        df["_player_key"] = df["Player"].map(normalize_name)
        key = "_player_key"

    if "G" in df.columns:
        df["G"] = pd.to_numeric(df["G"], errors="coerce")

    def pick(group: pd.DataFrame) -> pd.DataFrame:
        team_col = "Tm" if "Tm" in group.columns else None
        if team_col and (group[team_col] == "TOT").any():
            return group[group[team_col] == "TOT"].head(1)
        if "G" in group.columns:
            return group.sort_values("G", ascending=False).head(1)
        return group.head(1)

    return (
        df.groupby(key, group_keys=False, dropna=False)
          .apply(pick, include_groups=False)
          .reset_index(drop=True)
    )


def load_stats(per_game_path: Path, advanced_path: Path) -> pd.DataFrame:
    per_game = choose_one_row_per_player(clean_bref_table(pd.read_csv(per_game_path)))
    advanced = choose_one_row_per_player(clean_bref_table(pd.read_csv(advanced_path)))

    # Prefer the stable Basketball-Reference player id when available.
    if "Player-additional" in per_game.columns and "Player-additional" in advanced.columns:
        merged = per_game.merge(
            advanced,
            on="Player-additional",
            how="inner",
            suffixes=("", "_adv"),
        )
    else:
        per_game["_player_key"] = per_game["Player"].map(normalize_name)
        advanced["_player_key"] = advanced["Player"].map(normalize_name)
        merged = per_game.merge(
            advanced,
            on="_player_key",
            how="inner",
            suffixes=("", "_adv"),
        )

    # Pull advanced fields from their advanced-table columns.
    for col in ["TS%", "BPM", "VORP", "WS", "WS/48"]:
        adv_col = f"{col}_adv"
        if adv_col in merged.columns:
            merged[col] = merged[adv_col]

    # Ensure the basic fields come from the per-game table.
    for col in ["Age", "G", "MP", "PTS", "AST", "TRB", "STL", "BLK"]:
        if col not in merged.columns and f"{col}_x" in merged.columns:
            merged[col] = merged[f"{col}_x"]

    keep = ["Player", "Tm", "Pos", "G"] + FEATURES
    keep = [c for c in keep if c in merged.columns]
    merged = merged[keep].copy()

    for col in [c for c in keep if c not in {"Player", "Tm", "Pos"}]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    merged["_player_key"] = merged["Player"].map(normalize_name)
    return merged


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join(str(x) for x in tup if str(x) != "nan").strip()
            for tup in df.columns
        ]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_money(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(r"[$,]", "", regex=True).str.strip(),
        errors="coerce",
    )


def load_salary_data(salary_csv: Path | None = None) -> pd.DataFrame:
    """
    Load a local salary CSV or scrape HoopsHype.

    The original v1 notebook accidentally treated the last monetary column
    from HoopsHype as salary. This reconstruction intentionally chooses the
    first salary-like monetary column after Player, which corresponds to the
    listed 2023-24 salary rather than an adjusted figure.
    """
    if salary_csv is not None:
        salary = flatten_columns(pd.read_csv(salary_csv))
    else:
        salary = flatten_columns(pd.read_html(HOOPSHYPE_URL)[0])

    player_candidates = [c for c in salary.columns if "player" in c.lower()]
    if not player_candidates:
        raise ValueError(f"Could not identify a Player column. Columns: {salary.columns.tolist()}")
    player_col = player_candidates[0]

    # Find columns that contain dollar-formatted values.
    money_cols = []
    for col in salary.columns:
        if col == player_col:
            continue
        sample = salary[col].astype(str).head(25)
        if sample.str.contains(r"\$", regex=True).any():
            money_cols.append(col)

    if not money_cols:
        # Fallback for already-numeric local files.
        salary_candidates = [c for c in salary.columns if "salary" in c.lower()]
        if not salary_candidates:
            raise ValueError(f"Could not identify a salary column. Columns: {salary.columns.tolist()}")
        salary_col = salary_candidates[0]
    else:
        salary_col = money_cols[0]

    out = salary[[player_col, salary_col]].copy()
    out.columns = ["Player_salary", "Salary"]
    out["Salary"] = parse_money(out["Salary"])
    out["_player_key"] = out["Player_salary"].map(normalize_name)
    out = out.dropna(subset=["Salary"])
    out = out.sort_values("Salary", ascending=False).drop_duplicates("_player_key")
    return out


def build_model_table(
    stats: pd.DataFrame,
    salary: pd.DataFrame,
    min_games: int = 1,
    min_minutes: float = 0.0,
) -> pd.DataFrame:
    df = stats.merge(salary[["_player_key", "Salary"]], on="_player_key", how="left")
    df = df[(df["G"] >= min_games) & (df["MP"] >= min_minutes)].copy()

    required = FEATURES + ["Salary"]
    return df.dropna(subset=required).reset_index(drop=True)


def run_model(model_df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, dict]:
    X = model_df[FEATURES]
    y = model_df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    test_model = LinearRegression()
    test_model.fit(X_train, y_train)
    test_pred = test_model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
    r2 = float(r2_score(y_test, test_pred))

    final_model = LinearRegression()
    final_model.fit(X, y)

    result = model_df.copy()
    result["PredictedSalary"] = final_model.predict(X)
    result["SalaryGap"] = result["PredictedSalary"] - result["Salary"]
    result["AbsoluteError"] = (result["PredictedSalary"] - result["Salary"]).abs()

    # Secondary idea preserved from the v2 notebook.
    result["VORP_per_$1M"] = np.where(
        result["Salary"] > 0,
        result["VORP"] / (result["Salary"] / 1_000_000),
        np.nan,
    )

    result["ActualSalary_M"] = result["Salary"] / 1_000_000
    result["PredictedSalary_M"] = result["PredictedSalary"] / 1_000_000
    result["SalaryGap_M"] = result["SalaryGap"] / 1_000_000

    summary = {
        "players_used": int(len(result)),
        "features": len(FEATURES),
        "test_size": int(len(y_test)),
        "random_state": random_state,
        "rmse_dollars": rmse,
        "rmse_millions": rmse / 1_000_000,
        "r2": r2,
    }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="NBA Moneyball salary-value model")
    parser.add_argument("--per-game", type=Path, default=Path("../data/nba_2023-24.csv"))
    parser.add_argument("--advanced", type=Path, default=Path("../data/2023-24_advanced.csv"))
    parser.add_argument("--salary-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--min-games", type=int, default=1)
    parser.add_argument("--min-minutes", type=float, default=0.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.per_game, args.advanced)
    salary = load_salary_data(args.salary_csv)
    model_df = build_model_table(stats, salary, args.min_games, args.min_minutes)

    if len(model_df) < 30:
        raise RuntimeError(
            f"Only {len(model_df)} players matched stats and salary data. "
            "Check salary source/name matching before modeling."
        )

    valuations, summary = run_model(model_df)

    ordered_cols = [
        "Player", "Tm", "Pos", "G", "Age",
        "PTS", "AST", "TRB", "STL", "BLK", "MP",
        "TS%", "BPM", "VORP", "WS", "WS/48",
        "ActualSalary_M", "PredictedSalary_M", "SalaryGap_M", "VORP_per_$1M",
    ]
    ordered_cols = [c for c in ordered_cols if c in valuations.columns]

    valuations[ordered_cols].sort_values(
        "SalaryGap_M", ascending=False
    ).to_csv(args.output_dir / "player_valuations_2023_24.csv", index=False)

    valuations[ordered_cols].sort_values(
        "SalaryGap_M", ascending=False
    ).head(25).to_csv(args.output_dir / "undervalued_players_2023_24.csv", index=False)

    valuations[ordered_cols].sort_values(
        "SalaryGap_M", ascending=True
    ).head(25).to_csv(args.output_dir / "overvalued_players_2023_24.csv", index=False)

    pd.DataFrame([summary]).to_csv(
        args.output_dir / "model_summary_2023_24.csv", index=False
    )

    print("\nNBA Moneyball — Final Reconstruction")
    print("------------------------------------")
    print(f"Players used: {summary['players_used']}")
    print(f"Test RMSE: ${summary['rmse_dollars']:,.0f}")
    print(f"Test R²: {summary['r2']:.3f}")

    print("\nTop 10 model-identified undervalued players:")
    print(
        valuations.sort_values("SalaryGap_M", ascending=False)[
            ["Player", "ActualSalary_M", "PredictedSalary_M", "SalaryGap_M"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
