import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm

# Load the dataset
file_path = "/Users/jaygamage/Downloads/Refined_Sales_Quantity_Dataset.csv"  # Update with the correct file path
df = pd.read_csv(file_path)

# Convert Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extract Year and Month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Aggregate sales quantity per category per month
seasonality_df = df.groupby(['Material_Group', 'Year', 'Month'])['Sales_Quantity'].sum().reset_index()

# Pivot for seasonality trends
seasonality_pivot = seasonality_df.pivot_table(values='Sales_Quantity', index=['Material_Group', 'Month'], columns='Year')

### 1. Seasonality Analysis Per Category ###
categories = seasonality_pivot.index.get_level_values(0).unique()

for category in categories:
    plt.figure(figsize=(12, 6))
    category_data = seasonality_pivot.loc[category]
    
    for year in category_data.columns:
        plt.plot(category_data.index, category_data[year], label=f"{year}")
    
    plt.xlabel("Month")
    plt.ylabel("Sales Quantity")
    plt.title(f"Seasonality Trend for {category}")
    plt.legend(title="Year")
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.show()

### 2. Heatmap of Sales Trends ###
plt.figure(figsize=(12, 6))
heatmap_data = seasonality_pivot.fillna(0)  # Fill NaN values with 0
sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".0f", linewidths=0.5)
plt.title("Heatmap of Monthly Sales Trends per Category")
plt.xlabel("Year")
plt.ylabel("Category and Month")
plt.show()

### 3. Boxplot of Monthly Sales Distribution ###
plt.figure(figsize=(12, 6))
sns.boxplot(x=seasonality_df['Month'], y=seasonality_df['Sales_Quantity'])
plt.title("Boxplot of Monthly Sales Distribution Across Years")
plt.xlabel("Month")
plt.ylabel("Sales Quantity")
plt.grid(True)
plt.show()

### 4. Rolling Average Trend ###
df['Sales_Rolling_Avg'] = df.groupby('Material_Group')['Sales_Quantity'].transform(lambda x: x.rolling(3, min_periods=1).mean())
plt.figure(figsize=(12, 6))
for category in df['Material_Group'].unique():
    plt.plot(df[df['Material_Group'] == category].groupby('Date')['Sales_Rolling_Avg'].mean(), label=category)
plt.title("Rolling Average Trend of Sales Quantity")
plt.xlabel("Year")
plt.ylabel("3-Month Rolling Average Sales")
plt.legend(title="Category")
plt.grid(True)
plt.show()

### 5. Year-over-Year Comparison ###
plt.figure(figsize=(12, 6))
yearly_sales = seasonality_df.groupby(['Year', 'Month'])['Sales_Quantity'].sum().unstack(level=0)
yearly_sales.plot(kind='line', figsize=(12, 6))
plt.title("Year-over-Year Comparison of Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales Quantity")
plt.legend(title="Year")
plt.grid(True)
plt.show()

### 6. Correlation Matrix ###
plt.figure(figsize=(10, 6))
corr_matrix = df[['Sales_Quantity', 'petrol95', 'petrol92', 'auto_diesel', 'super_diesel', 'gdp', 'ncpi', 'crudeoil', 'num_of_holidays', 'traffic_index']].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Sales and External Factors")
plt.show()

### 7. Autocorrelation Plot ###
plt.figure(figsize=(12, 6))
sm.graphics.tsa.plot_acf(df.groupby('Date')['Sales_Quantity'].sum(), lags=24)
plt.title("Autocorrelation of Sales Quantity")
plt.grid(True)
plt.show()

### 8. Time Series Decomposition for Forecasting ###
df_grouped = df.groupby('Date')['Sales_Quantity'].sum()
decomposition = sm.tsa.seasonal_decompose(df_grouped, model='additive', period=12)

# Plot Decomposition
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.suptitle("Time Series Decomposition of Sales Data")
plt.show()

### 9. Distribution of Sales Quantity Over Time ###
plt.figure(figsize=(12, 6))
sns.histplot(df['Sales_Quantity'], bins=50, kde=True)
plt.title("Distribution of Sales Quantity Over Time")
plt.xlabel("Sales Quantity")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
