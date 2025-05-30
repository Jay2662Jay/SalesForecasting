# Tyre Sales Forecasting System using Machine Learning

This system leverages machine learning to forecast monthly tyre sales for Ceat Kelani International Tyres (Pvt) Ltd. By integrating historical sales, external macroeconomic data, and promotional inputs, it delivers precise group-level forecasts and visualizations to support strategic planning and inventory optimization.

## Project Objectives

- Build a machine learning-based time series forecasting system.
- Integrate external factors such as fuel prices, GDP, NCPI, and promotion.
- Use recursive forecasting and feature engineering techniques.
- Visualize forecasts through a Streamlit dashboard.
- Track model accuracy using MAE, RMSE, MAPE, and R².

## Repository Structure

Each script in the repository plays a role in the end-to-end pipeline:

- `a.py` – Feature prediction (fuel, crude oil, traffic, etc.)
- `b.py` – Promotion prediction (advertising & marketing spend)
- `c.py` – Merging actual data: sales + external + promotion for 2016–2024
- `d.py` – Merging predicted feature + promotion data for 2025
- `e.py` – Feature engineering (lag features, rolling means, time features)
- `f.py` – Model training with CatBoost, validation, and recursive forecasting
- `g.py` – Streamlit dashboard for interactive forecast visualization
- `h.py` – Master script to execute the entire pipeline end-to-end

## Getting Started

Install the dependencies:

```bash
pip install pandas numpy catboost optuna shap streamlit matplotlib scikit-learn
```

Run full pipeline with dashboard:

```bash
python3 h.py --mode full
```

Run only the dashboard:

```bash
python3 h.py --mode dashboard
```

Run only the model training and prediction:

```bash
python3 h.py --mode process
```

## Model Performance Example

Example forecast accuracy for selected product groups:

| Group | MAPE (%) | R² Score |
|-------|----------|----------|
| 0     | 7.8      | 0.67     |
| 5     | 9.3      | 0.71     |
| 13    | 6.2      | 0.79     |
| 21    | 4.5      | 0.84     |

## License and Contact

This project is developed for academic purposes by Jaindu Gamage as part of the final year project at NSBM Green University.

- **Supervisor:** Mr. Gayan Perera  
- **Email:** jay2002jay@icloud.com  
- **University:** NSBM Green University, BSc (Hons) Data Science(PLY)  
- **Module:** PUSL3190 – Final Year Individual Project
