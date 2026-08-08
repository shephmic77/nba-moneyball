# NBA Moneyball

A personal NBA analytics project exploring whether player performance statistics can be used to estimate salary and identify potentially undervalued players.

## Project Question

Can NBA player production be translated into an estimated market salary, and can the gap between predicted and actual salary reveal potential bargains?

The project was inspired by the general Moneyball idea of finding value that the market may be underpricing.

## Project Evolution

The repository preserves the original development history rather than replacing it with only a polished final result.

### V1 — Salary Regression

The first major version:

- collected 2023-24 NBA player statistics
- combined basic and advanced statistics
- collected salary data
- merged performance and salary information
- trained a multiple linear regression model
- predicted player salaries
- compared predicted salary with actual salary
- ranked players by potential underpayment and overpayment

The saved notebook output reported:

- **RMSE:** approximately $6.63 million
- **R²:** approximately 0.588

Some of the strongest model-identified bargains included:

- Tyrese Maxey
- Desmond Bane
- Tyrese Haliburton
- Cam Thomas
- Alperen Sengun
- Eric Gordon

These results should be interpreted as model outputs, not claims that these contracts were objectively mispriced.

### V2 — Value Score Experiment

A later experiment explored a simpler value concept based partly on:

**VORP per $1 million of salary**

This version also experimented with player contract information and physical attributes.

The idea was useful as a secondary way of thinking about efficiency, but the salary-regression approach became the stronger foundation for the final reconstruction.

## Final Reconstruction

The cleaned final script combines the strongest ideas from both development paths.

It uses:

- PTS
- AST
- TRB
- STL
- BLK
- MP
- TS%
- BPM
- VORP
- WS
- WS/48
- Age

to estimate player salary using multiple linear regression.

The final workflow also:

- removes duplicate traded-player rows
- keeps one statistical profile per player
- normalizes player names before salary matching
- evaluates performance on a held-out test set
- retrains on the complete matched dataset
- calculates predicted salary
- calculates predicted-minus-actual salary gap
- calculates VORP per $1 million as a secondary value metric
- exports valuation tables for further analysis

## Repository Structure

```text
nba-moneyball/
├── README.md
├── data/
│   ├── nba_2023-24.csv
│   └── 2023-24_advanced.csv
├── development/
│   ├── v1_salary_regression/
│   │   ├── NBA_MoneyBall_v1.ipynb
│   │   └── v1.ipynb
│   └── v2_value_score_experiment/
│       └── v2.ipynb
├── final/
│   ├── nba_moneyball_final.py
│   ├── original_model_summary_2023_24.csv
│   ├── original_notebook_saved_results_2023_24.csv
│   └── requirements.txt
└── portfolio/
```

## Running the Final Model

From the `final/` folder:

```bash
pip install -r requirements.txt
python nba_moneyball_final.py
```

The script uses the local statistical datasets and retrieves 2023-24 salary information.

It exports:

```text
player_valuations_2023_24.csv
undervalued_players_2023_24.csv
overvalued_players_2023_24.csv
model_summary_2023_24.csv
```

## Limitations

Salary is not determined only by basketball production.

Important factors not fully represented by the model include:

- rookie-scale contracts
- maximum-contract rules
- contract timing
- age and projected development
- injury history
- positional scarcity
- free-agent market conditions
- team-specific needs
- reputation
- negotiation leverage
- playoff performance
- defensive and off-ball impact not captured by the chosen statistics

Because of this, the salary gap should be interpreted as **how unusual a player's salary looks relative to players with similar statistical production**, rather than a definitive estimate of a player's true economic value.

## What I Learned

This project helped me practice:

- acquiring and combining data from multiple sources
- cleaning inconsistent player names
- handling traded-player duplicates
- selecting model features
- building and evaluating regression models
- interpreting RMSE and R²
- distinguishing statistical prediction from real-world valuation
- turning exploratory notebooks into a cleaner reproducible workflow

## Future Improvements

Possible extensions include:

- multi-season salary modeling
- contract-year and contract-length features
- age curves
- rookie and max-contract indicators
- position and role information
- defensive impact metrics
- lineup data
- shot-location data
- tracking data
- nonlinear models
- regularized regression
- tree-based models
- comparison of model performance across methods

A larger version could also ask whether the NBA market systematically pays more for certain types of production than others.

## Author

**Michael Shepherd**

Personal Data Science Project
