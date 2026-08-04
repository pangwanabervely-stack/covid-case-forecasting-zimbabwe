# COVID-19 Weekly Case Forecasting Zimbabwe

**Author:** Bervely Pangwana
**Tools used:** Python (pandas, numpy, matplotlib), statsmodels (ARIMA), scikit-learn (Random Forest)

## Project Overview
This project forecasts weekly COVID-19 case counts for Zimbabwe using two
different modeling approaches a classic statistical time-series model
(ARIMA) and a machine learning approach (Random Forest with engineered lag
features) — and rigorously compares their performance against each other
and a naive baseline.

**Note:** This is a technical forecasting exercise using historical public
health data for skill demonstration. It is not intended for clinical or
policy decision-making.

## Objectives
- Build a time-series forecasting pipeline from raw case data to evaluated predictions
- Apply and compare a classic statistical method (ARIMA) against a machine
  learning method (Random Forest with lag features)
- Critically evaluate model performance, including understanding *why* a
  model underperforms, not just reporting whether it did

## Data Source
Our World in Data COVID-19 dataset, filtered to Zimbabwe, resampled from
daily to weekly case counts to reduce reporting noise.

## Methodology
1. **Data Preparation:** Filtered global dataset to Zimbabwe, cleaned missing/negative values, resampled daily counts to weekly totals.
2. **Train/Test Split:** Held out the final 12 weeks as a test set (never seen during training).
3. **Baseline:** Naive forecast (next week = last known week) as the benchmark to beat.
4. **Model 1 - ARIMA:** Tested for stationarity (Augmented Dickey-Fuller test), selected the best (p,d,q) order via AIC grid search, fit and forecasted.
5. **Model 2 - Random Forest:** Engineered lag features (previous 4 weeks + rolling mean), trained a Random Forest Regressor, and generated forecasts recursively (each prediction feeds into the next week's input features).
6. **Evaluation:** Compared all three approaches using MAE and RMSE.

## Key Findings

**Model Performance (12-week test period, spanning the Omicron wave peak and decline):**

| Model | MAE | RMSE |
|---|---|---|
| Random Forest | 6,598 | 7,359 |
| Naive Baseline | 22,104 | 23,089 |
| ARIMA(2,1,0) | 258,297 | 280,190 |

**Random Forest substantially outperformed both the naive baseline and ARIMA**, reducing error by ~70% versus the naive approach.

**Why ARIMA failed dramatically:** The test period began during a steep upward climb in cases (the onset of the Omicron wave). ARIMA's linear autoregressive structure extrapolated that upward momentum and continued predicting exponential growth (forecasts exceeded 350,000 weekly cases), while actual cases peaked and sharply reversed within weeks. This highlights a key limitation of classic ARIMA models: they assume recent trends continue and cannot anticipate nonlinear turning points, which are common in epidemic waves.

**Why Random Forest performed better:** Having already seen two earlier wave-shaped cycles in the training data, the lag-feature-based model could better recognize "wave" patterns, even though it still lagged behind the speed of the actual decline.

**Practical implication:** This result argues for either (a) preferring ensemble/ML approaches over pure ARIMA for volatile epidemic data, or (b) using ARIMA only with careful human monitoring during periods of rapid trend change, rather than trusting it blindly during a wave's onset.

## Limitations & Future Improvements
- **ARIMA order selection:** The (p,d,q) order was chosen purely by minimizing AIC on training data, which does not account for forecast stability a known weakness of naive grid search. A more robust approach (e.g. damped-trend methods, or penalizing unstable forecasts during model selection) would likely reduce ARIMA's blow-up behavior.
- **No exogenous variables:** Neither model used external predictors (e.g. vaccination rates, policy stringency index - both available in the source dataset) that could help anticipate wave turning points. Incorporating these is a natural next step.
- **Single train/test split:** Results reflect one specific 12-week window. A more rigorous evaluation would use rolling-window cross-validation across multiple time periods to confirm these findings hold generally, not just for this particular wave.
- **Recursive forecasting compounding error:** The Random Forest's recursive forecasting approach (feeding predictions back in as inputs) can compound errors over longer horizons worth testing at different forecast horizons (4 weeks vs. 12 weeks vs. 24 weeks) to see how performance degrades.

## Files in this Repository
- `Covid_Case_Forecasting.ipynb` - full notebook: data preparation, both models, evaluation, and findings
- `weekly_cases.png` - exploratory chart of weekly case trends
- `forecast_comparison.png` - chart comparing actual vs. ARIMA vs. Random Forest forecasts

## How to Run
1. Download the Our World in Data COVID-19 dataset (available via Kaggle or OWID directly) and place it in the same folder as this notebook
2. Install requirements: `pip install pandas numpy matplotlib scikit-learn statsmodels jupyter`
3. Open and run `Covid_Case_Forecasting.ipynb` cell by cell
