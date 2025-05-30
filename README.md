
# 🛞 Tyre Sales Forecasting System using Machine Learning

This project builds a robust, intelligent, and business-ready sales forecasting system for **Ceat Kelani International Tyres (Pvt) Ltd**. The system uses historical tyre sales data combined with external economic indicators and promotional data to deliver highly accurate monthly forecasts across product groups.

---

## 📌 Project Objectives

- Build a **data-driven forecasting model** using machine learning and time series techniques (e.g., CatBoost, XGBoost, Prophet).
- Integrate **external features** like GDP, fuel prices, inflation (NCPI), traffic index, and advertising spend.
- Apply **recursive forecasting** to predict future sales using prior predicted values.
- Implement **model interpretability** using SHAP.
- Design a **user-friendly Streamlit dashboard** for visualization and decision support.

---

## 📂 Repository Structure

```
.
├── a.py                      # Feature prediction (e.g., fuel, crude oil, traffic)
├── b.py                      # Advertising & promotion prediction
├── c.py                      # Merging actual data: sales + features + promotion (2016–2024)
├── d.py                      # Merging predicted features + promo for 2025
├── e.py                      # Feature engineering (lags, rolling, etc.)
├── f.py                      # CatBoost forecasting with recursive predictions
├── g.py                      # Forecasting evaluation + metrics tracking
├── h.py                      # Streamlit dashboard for visualization
├── data/                     # Folder containing CSVs used or generated (optional)
│   ├── dataf_c.csv
│   ├── rdata.csv
│   ├── merged_data.csv
│   └── ...
├── PUSL3190 Project Proposal 10899309.pdf
└── README.md
```

---

## 🔍 Methodology Overview

### 1. Data Collection & Integration
- Historical sales from 2016–2024 (monthly)
- External features: GDP, NCPI, crude oil, fuel prices, traffic index
- Promotional data (ad spend per group)
- Combined and cleaned for training and 2025 forecasting.

### 2. Feature Engineering
- Added: `lag_1...lag_12`, `rolling_mean_3/6/9/12`, `sales_change_pct`, `month_sin/cos`, and a promo flag.

### 3. Modeling
- Group-level models using **CatBoost** with:
  - Walk-forward validation
  - Recursive predictions for 2025
  - Optuna hyperparameter tuning
  - Quantile regression for forecast confidence intervals

### 4. Evaluation
- Tracked **RMSE**, **MAE**, **MAPE**, and **R²** at group level.
- Overfitting mitigated via:
  - Expanding window validation
  - Noise (bagging_temperature, random_strength)
  - Feature simplification + SHAP-based pruning

### 5. Deployment
- **Streamlit dashboard** to display forecasted vs. actuals and total predicted sales for 2025.

---

## 📊 Dashboard Features

- Forecasts per product group
- Comparison of actual vs predicted
- Total 2025 sales estimate
- Historical trends and feature view
- Filters by tyre type and group_code

---

## 🚀 Getting Started

### Requirements

Install dependencies:
```bash
pip install pandas numpy catboost optuna shap streamlit matplotlib scikit-learn
```

### Run Forecasting Pipeline

```bash
python f.py
```

### Launch Dashboard

```bash
streamlit run h.py
```

---

## 📈 Model Performance (Example)

| Group | MAPE (%) | R² Score |
|-------|----------|----------|
| 0     | 7.8      | 0.67     |
| 5     | 9.3      | 0.71     |
| 13    | 6.2      | 0.79     |
| 21    | 4.5      | 0.84     |

---

## 📑 License

This repository is for academic and demonstration purposes only. All rights reserved © 2025 – Jay Gamage.

---

## 📧 Contact

**Developer:** Jay Gamage  
**Supervisor:** Dr. Pabudi Abeyrathne  
**Email:** jaygamage@cinecraft.lk  
**University:** CINEC Campus, BSc (Hons) Data Science  
**Module:** PUSL3190 – Final Year Individual Project

---

## 📚 References

Key sources include:
- S. Makridakis et al., *The M3-Competition*, IJF (2000)
- S. Taylor & B. Letham, *Forecasting at Scale* (Prophet), American Statistician (2018)
- G. Chandrashekar, F. Sahin, *Feature Selection Survey*, CEE (2014)
- K. Pauwels et al., *Sales Promotions & Firm Value*, Journal of Marketing (2004)

See full reference list in `PUSL3190 Project Proposal.pdf`.
