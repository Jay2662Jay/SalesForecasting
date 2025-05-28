import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV  # Add this import

# Load dataset
sales_data = pd.read_csv('/Users/jaygamage/Downloads/data1.csv')  # Update with the correct path
sales_data["date"] = pd.to_datetime(sales_data["date"])
sales_data.set_index("date", inplace=True)

# Convert 'gdp' to float
sales_data['gdp'] = sales_data['gdp'].astype(str).str.replace(',', '').astype(float)

# Resample data to monthly frequency, keeping only numeric columns
numeric_cols = sales_data.select_dtypes(include=[np.number]).columns
sales_data = sales_data[numeric_cols].resample('M').mean()

# Define features and target for XGBoost
features = ["gdp", "ccpi", "ncpi", "crudeoil", "number of holidays", "traffic index"]
target = "sales quantity"
sales_data = sales_data.dropna()
X, y = sales_data[features], sales_data[target]
train_size = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X.iloc[:train_size], X.iloc[train_size:], y.iloc[:train_size], y.iloc[train_size:]

# Calculate mean of actual values for accuracy percentage
mean_actual = y_test.mean()

# Hyperparameter tuning for XGBoost model
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}
xgb_model = XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_xgb_model = grid_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
accuracy_xgb = (1 - (mae_xgb / mean_actual)) * 100

# Train SARIMA model using auto_arima to find optimal parameters
auto_arima_model = auto_arima(y_train, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
order = auto_arima_model.order
seasonal_order = auto_arima_model.seasonal_order
sarima_model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False).fit()
sarima_forecast = sarima_model.get_forecast(steps=len(y_test)).predicted_mean
rmse_sarima = np.sqrt(mean_squared_error(y_test, sarima_forecast))
mae_sarima = mean_absolute_error(y_test, sarima_forecast)
r2_sarima = r2_score(y_test, sarima_forecast)
accuracy_sarima = (1 - (mae_sarima / mean_actual)) * 100

# Train Prophet model
prophet_data = sales_data.reset_index().rename(columns={"date": "ds", "sales quantity": "y"})
prophet_train = prophet_data.iloc[:train_size]
prophet_model = Prophet()
prophet_model.fit(prophet_train)
prophet_test = prophet_data.iloc[train_size:]
future = prophet_model.make_future_dataframe(periods=len(prophet_test), freq='M')
prophet_forecast = prophet_model.predict(future)[['ds', 'yhat']].set_index('ds').loc[prophet_test['ds']]
rmse_prophet = np.sqrt(mean_squared_error(y_test, prophet_forecast['yhat']))
mae_prophet = mean_absolute_error(y_test, prophet_forecast['yhat'])
r2_prophet = r2_score(y_test, prophet_forecast['yhat'])
accuracy_prophet = (1 - (mae_prophet / mean_actual)) * 100

# Model comparison
model_comparison = pd.DataFrame({
    "Model": ["SARIMA", "XGBoost", "Prophet"],
    "RMSE": [rmse_sarima, rmse_xgb, rmse_prophet],
    "MAE": [mae_sarima, mae_xgb, mae_prophet],
    "R²": [r2_sarima, r2_xgb, r2_prophet],
    "Accuracy (%)": [accuracy_sarima, accuracy_xgb, accuracy_prophet]
})

# Streamlit interface
st.title("Sales Quantity Forecasting")

def show_model_comparison():
    st.write("## Model Comparison")
    st.dataframe(model_comparison)
    
    st.subheader("RMSE Comparison")
    fig, ax = plt.subplots()
    ax.bar(model_comparison["Model"], model_comparison["RMSE"], color=['blue', 'red', 'green'])
    ax.set_ylabel("RMSE")
    st.pyplot(fig)

    st.subheader("MAE Comparison")
    fig, ax = plt.subplots()
    ax.bar(model_comparison["Model"], model_comparison["MAE"], color=['blue', 'red', 'green'])
    ax.set_ylabel("MAE")
    st.pyplot(fig)

    st.subheader("R² Comparison")
    fig, ax = plt.subplots()
    ax.bar(model_comparison["Model"], model_comparison["R²"], color=['blue', 'red', 'green'])
    ax.set_ylabel("R²")
    st.pyplot(fig)

    st.subheader("Accuracy (%) Comparison")
    fig, ax = plt.subplots()
    ax.bar(model_comparison["Model"], model_comparison["Accuracy (%)"], color=['blue', 'red', 'green'])
    ax.set_ylabel("Accuracy (%)")
    st.pyplot(fig)

def show_sales_trends():
    st.header("Sales Trends Over Time")
    fig, ax = plt.subplots()
    ax.plot(sales_data.index, sales_data["sales quantity"], marker='o', linestyle='-')
    ax.set_title("Sales Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Quantity")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Streamlit app layout
st.title("Tyre Sales Forecasting Dashboard")
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Select a Page", ["Model Comparison", "Sales Trends"])

if page == "Model Comparison":
    show_model_comparison()
elif page == "Sales Trends":
    show_sales_trends()

st.sidebar.write("Ensure dataset is updated and models are run locally before using the dashboard.")