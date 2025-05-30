# Tyre Sales Forecasting System using Machine Learning

This machine learning pipeline forecasts monthly tyre sales for JAY Tyres (Pvt) Ltd. The system combines historical sales, macroeconomic indicators, and promotional spending to provide highly accurate predictions.

## 🎯 Project Objectives

- Build accurate machine learning models to forecast tyre sales.
- Include external features like fuel prices, GDP, NCPI, traffic index, and advertising spend.
- Apply recursive forecasting to generate future predictions.
- Provide visual insights via a Streamlit dashboard.
- Track performance using MAE, RMSE, MAPE, and R² metrics.

## 📁 Folder Structure

```
project/
├── data/                   # Contains all raw and processed CSVs
├── feature_outputs/        # Stores generated features and processed datasets
├── main/                   # All Python scripts and main pipeline files
│   ├── a.py                # Feature prediction (e.g. crude oil, traffic, etc.)
│   ├── b.py                # Promotion prediction
│   ├── c.py                # Merge actual sales + features + promo (2016–2024)
│   ├── d.py                # Merge predicted features + promo (2025)
│   ├── e.py                # Feature engineering (lags, rolling, etc.)
│   ├── f.py                # Model training and validation
│   ├── g.py                # Streamlit dashboard
│   ├── h.py                # Full pipeline controller
│   └── readme              # Readme file (this)
└── MISC/                   # Miscellaneous resources (e.g. previous trails, etc.)
```

## ⚙️ Execution Modes

You can execute different parts of the pipeline using `h.py`:

```bash
# Full pipeline: feature + promo prediction, merging, feature engineering, model training, dashboard
python3 h.py --mode full

# Only Streamlit dashboard
python3 h.py --mode dashboard

# Only model training & prediction (assumes features already engineered)
python3 h.py --mode process
```

## 🚀 Getting Started

### Install dependencies

```bash
pip install pandas numpy catboost optuna shap streamlit matplotlib scikit-learn
```

### Launch dashboard

```bash
streamlit run g.py
```

## 📈 Model Evaluation Example

| Group | MAPE (%) | R² Score |
|-------|----------|----------|
| 0     | 7.8      | 0.67     |
| 5     | 9.3      | 0.71     |
| 13    | 6.2      | 0.79     |
| 21    | 4.5      | 0.84     |

## 📧 Contact

**Developer:** Jaindu Gamage  
**Supervisor:** Mr. Gayan Perera  
**Email:** jay2002jay@icloud.com  
**University:** NSBM Green University, BSc (Hons) Data Science(PLY)  
**Module:** PUSL3190 – Final Year Individual Project
