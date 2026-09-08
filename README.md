# Burnout-Prediction
# Employee Burnout Prediction

A Machine Learning project that predicts an employee's burnout risk score from workplace-related factors, using Linear Regression.

## Why

Employee burnout is expensive and hard to catch early. This project explores whether a simple, interpretable regression model can flag at-risk employees using structured workplace data — as a baseline before reaching for heavier models.

## Dataset

`employee_burnout_dataset_1000_records.csv` — 1,000 employee records with workplace-related features (e.g. workload, resource allocation, mental fatigue indicators) and a target burnout score.

## Approach

1. **Data preprocessing** — load and clean the dataset with Pandas, handle missing values, split into features/target
2. **Exploratory analysis** — visualize feature distributions and correlations with Matplotlib
3. **Modeling** — fit a **Linear Regression** model (scikit-learn) to predict the continuous burnout score
4. **Evaluation** — train/test split, evaluated with **R² (coefficient of determination)**

## Results

- **Model:** Linear Regression
- **R²: 0.61** — the model explains about 61% of the variance in burnout scores

### Why not higher?

Burnout is genuinely multi-causal and not fully linear — a straight-line model can't capture interaction effects between features (e.g. workload combined with lack of WFH flexibility compounding differently than either alone). The dataset is also relatively small (1,000 records). Ridge/Lasso regularization or tree-based models (Random Forest, Gradient Boosting) would likely improve this — noted as a next step below.

## Tech stack

- Python
- NumPy, Pandas — data handling
- Matplotlib — visualization
- scikit-learn — model training and evaluation

## Files

| File | Purpose |
|---|---|
| `burnout_prediction.ipynb` | Full notebook: EDA, model training, evaluation, plots |
| `burnout_prediction.py` | Script version of the pipeline |
| `employee_burnout_dataset_1000_records.csv` | Dataset |

## Running it

```bash
pip install numpy pandas matplotlib scikit-learn
python3 burnout_prediction.py
```

Or open `burnout_prediction.ipynb` in Jupyter to see the full analysis with plots.

## Possible next steps

- Try Ridge/Lasso regression or a Random Forest to capture non-linear effects
- Cross-validation instead of a single train/test split
- Feature engineering (interaction terms between workload and WFH availability)
- Larger, more diverse dataset
