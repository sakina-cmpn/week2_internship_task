# Task 4 – Complex Data Munging & Statistical Modeling

## 🧹 Part 1: Data Cleaning
- Loaded raw cricket T20 player data
- Performed:
  - Duplicate removal
  - Type conversion
  - Missing value handling
  - Column renaming and formatting
- Final cleaned dataset saved as: `cleaned_t20_dataset.csv`

## 🧠 Part 2: Feature Engineering & Modeling
- Selected numeric predictors: Matches, Innings, Strike Rate, Average
- Added constant for intercept
- Fit a linear regression using `statsmodels`:
  ```python
  Runs ~ Matches + Innings + SR + Average
