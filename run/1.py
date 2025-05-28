import streamlit as st
import plotly.express as px
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load the preprocessed dataset
file_path = '/Users/jaygamage/Downloads/data.csv'  # Ensure this file is present in the working directory
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

# Convert relevant columns to numeric
numeric_columns = ['sales_quantity', 'petrol95', 'petrol92', 'auto_diesel', 'super_diesel', 'gdp', 'ncpi', 'crudeoil', 'num_of_holidays', 'traffic_index']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

def train_xgboost_model(df):
    # Aggregate and preprocess data
    df_agg = df.groupby('date').agg({
        'sales_quantity': 'sum',
        'petrol95': 'mean',
        'petrol92': 'mean',
        'auto_diesel': 'mean',
        'super_diesel': 'mean',
        'gdp': 'mean',
        'ncpi': 'mean',
        'crudeoil': 'mean',
        'num_of_holidays': 'sum',
        'traffic_index': 'mean'
    }).reset_index()
    
    for lag in range(1, 4):
        df_agg[f'sales_lag_{lag}'] = df_agg['sales_quantity'].shift(lag)
    df_agg = df_agg.dropna()
    
    # Define features
    features = ['petrol95', 'petrol92', 'auto_diesel', 'super_diesel', 'gdp', 'ncpi', 'crudeoil', 'num_of_holidays', 'traffic_index',
                'sales_lag_1', 'sales_lag_2', 'sales_lag_3']
    target = 'sales_quantity'
    
    # Train-test split
    train_size = int(len(df_agg) * 0.8)
    train, test = df_agg[:train_size], df_agg[train_size:]
    
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]
    
    # Train XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=50, max_depth=3, learning_rate=0.1, subsample=0.8)
    model.fit(X_train, y_train)
    
    test['forecast_xgb'] = model.predict(X_test)
    mae = mean_absolute_error(y_test, test['forecast_xgb'])
    
    return df_agg, test, mae

# Train Model
df_agg, test_xgb, mae_xgb = train_xgboost_model(df)

# Streamlit App
st.title("Tyre Sales Forecasting Dashboard")
st.write(f"Optimized XGBoost Model MAE: {mae_xgb:.2f}")

# Plot Sales Trend
graph1 = px.line(df_agg, x='date', y='sales_quantity', title='Sales Trend Over Time')
st.plotly_chart(graph1)

# Forecasting Results
graph2 = px.line(test_xgb, x='date', y=['sales_quantity', 'forecast_xgb'],
                 labels={'value': 'Sales Quantity', 'date': 'Date'},
                 title='Actual vs. Forecasted Sales',
                 line_dash_sequence=['solid', 'dash'])
st.plotly_chart(graph2)

st.write("### Insights:")
st.write("- The forecasting model captures sales trends and fluctuations.")
st.write("- Seasonal and economic indicators influence tyre sales trends.")