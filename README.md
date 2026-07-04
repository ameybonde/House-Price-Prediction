# House Price Prediction

A linear regression model built on the Kaggle "House Prices - Advanced Regression Techniques" dataset (`train.csv` / `test.csv`, 1460 rows, ~80 raw columns). The notebook selects a subset of features believed to drive `SalePrice`, engineers a few derived features, encodes categoricals, and fits `sklearn.linear_model.LinearRegression`.

## What it does

1. Loads `train.csv` and `test.csv`.
2. Drops columns with heavy missing data: `Alley`, `MiscFeature`, `Fence`, `PoolQC`.
3. Selects a manually chosen feature subset based on domain reasoning (property size, bathrooms, location, structural quality, garage, basement, porch area, sale type/condition).
4. Fills missing values — mode for categorical columns, median for numeric columns.
5. Builds a final feature set:
   - `LotArea`, `GrLivArea`, `GarageArea`, `OverallQual`, `OverallCond`, `Neighborhood`, `TotRmsAbvGrd`, `SaleType`, `SaleCondition`
   - Engineered: `HouseAge = YrSold - YearBuilt`, `TotalConstructionArea = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`, `TotalPorch = OpenPorchSF + EnclosedPorch + ScreenPorch + 3SsnPorch`
6. Splits 80/20 train/test.
7. Preprocesses with `ColumnTransformer`: `OneHotEncoder` on categoricals, `StandardScaler` on numerics.
8. Fits `LinearRegression` and evaluates with MAE, MSE, RMSE, R².

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

No `requirements.txt` exists yet — you should add one instead of relying on whatever's installed on your machine.

## How to run

The notebook currently hardcodes Windows paths (`C:\Users\USER\Downloads\train.csv`). This will break on any other machine, including yours in six months when you reorganize your folders. Fix this before sharing the notebook with anyone:

```python
df = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
```

## Known issues — read this before trusting any result here

**The reported metrics are wrong, not good.**
```
MAE: 0.0
MSE: 0.0
RMSE: 0.0
R2: 1.0
```
A linear regression with 9 hand-picked features does not perfectly predict house prices. This is a textbook sign of a bug — most likely:
- `x['HouseAge']`, `x['TotalConstructionArea']`, `x['TotalPorch']` were assigned with `df[...]` after `x` had already been column-filtered and index-modified, triggering `SettingWithCopyWarning` — which you saw and ignored. That warning exists because the assignment might not be doing what you think it's doing.
- Possible train/test row misalignment introduced by reassigning `x_train = preprocessor.fit_transform(x_train)` in place, so downstream cells may not be operating on what you assume.

Until this is root-caused, **do not use this model or its numbers for anything**, including a resume bullet point. A perfect R² is a red flag, not a result worth reporting.

**Other things not done, in order of importance:**
- No cross-validation — a single 80/20 split tells you almost nothing about generalization.
- Feature selection was manual/eyeballed from a PDF description, not validated with correlation analysis, VIF, or feature importance.
- No baseline model (e.g. mean prediction) to sanity-check whether the "good" metrics are actually good.
- `test.csv` is loaded and cleaned but never actually used to generate predictions or a submission file — it's dead weight in the notebook right now.
- No model persistence (`joblib`/`pickle`) — the model is retrained from scratch every run and can't be reused.
- Column selection lists were typed out by hand twice (once for `df`, once for `test`) instead of defined once and reused — a copy-paste error waiting to happen.

## Suggested next steps

1. Fix the leakage/indexing bug and confirm R² lands somewhere realistic (Kaggle leaderboard baselines for this dataset are typically R² 0.85–0.92 with proper feature engineering, not 1.0).
2. Add k-fold cross-validation.
3. Actually generate predictions on `test.csv` and write a submission CSV.
4. Try a regularized model (Ridge/Lasso) or a tree-based model (RandomForest/XGBoost) as a comparison baseline.
5. Move hardcoded paths and feature lists into constants or a config file.
