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
