# Wheat Yield Prediction with Climate Data

## Project Overview
This project explores whether simple climate variables can explain and predict
wheat yield variability across countries and years.

The focus is not on achieving maximum predictive accuracy, but on building a
**clean, reproducible modeling pipeline** and applying **correct evaluation
strategies** for panel-like agricultural data.

## Data
The dataset contains country–year observations with:
- Wheat yield (kg/ha)
- Mean growing-season temperature
- Total growing-season precipitation

The dataset is intentionally small and is used to demonstrate methodology rather
than to produce deployable forecasts.

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Inspected yield distributions and climate–yield relationships
- Identified large differences in yield levels across countries

### 2. Baseline Modeling
- Dummy regressor (mean yield)
- Ridge regression using climate variables
- Metrics: RMSE and R²

### 3. Evaluation Strategy
Initial random K-fold cross-validation produced overly optimistic results.
Because observations are grouped by country, this leads to **data leakage**.

To address this, evaluation was switched to:

**GroupKFold with country as the grouping variable**

This tests whether a model trained on some countries can generalize to an unseen
country, which is a much harder and more realistic problem.

### 4. Feature Engineering
- Climate-only features:
  - Mean temperature
  - Precipitation
  - Squared terms
  - Interaction term
- Country fixed effects (one-hot encoding) to model country-level yield baselines

## Results

### Climate-only models
- Ridge regression outperforms a dummy baseline
- Performance collapses under GroupKFold evaluation
- Strongly negative R² indicates climate alone cannot explain between-country
  yield differences

### Climate + country fixed effects
- Adding country indicators reduces RMSE under GroupKFold
- Performance improves but remains unstable due to:
  - very small sample size
  - leave-one-country-out evaluation
- Results highlight that yield differences are largely structural
  (management, genetics, soils), not climate-only

## Key Takeaways
- Correct evaluation strategy matters more than model choice
- Random CV can dramatically overestimate performance in grouped data
- Climate variables explain within-country variability better than
  between-country yield levels
- Country-level structure must be modeled explicitly

## Limitations
- Extremely small dataset (few countries, few years)
- No management, soil, or genetic covariates
- Results are illustrative, not predictive

## Future Work
- Expand dataset to more countries and years
- Add management and soil variables
- Use hierarchical or mixed-effects models
- Explore time-aware validation strategies

## Project Structure
