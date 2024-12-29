import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
file_path = 'C:/Users/Lenovo/Downloads/index.csv'  # Update with your actual file path
sales_data = pd.read_csv(file_path)

# Print the column names to verify the correct column name for the date
print(sales_data.columns)

# Use the correct name for the date column
sales_data['date'] = pd.to_datetime(sales_data['date'])  # Update with the correct column name
sales_data = sales_data.sort_values('date')

# -------------------------------------
# Image 1: Time Series Exploratory Data Analysis (Coffee Sales Over Time)
# -------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(sales_data['date'], sales_data['money'], color='blue')  # Update 'money' if needed
plt.title('Coffee Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.savefig('coffee_sales_over_time.png')
plt.show()

# -------------------------------------
# Image 2: Sales Forecasting for Next 30 Days (ARIMA)
# -------------------------------------
# Fit the ARIMA model
arima_model = ARIMA(sales_data['money'], order=(5, 1, 0))  # Adjust order based on your data
arima_result = arima_model.fit()

# Predict the next 30 days
future_dates = pd.date_range(start=sales_data['date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
arima_forecast = arima_result.forecast(steps=30)

# Create a DataFrame for forecast results
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Sales': arima_forecast
})

# Display the forecast table
print("Forecasted Sales for the Next 30 Days:")
print(forecast_df)

# Plot forecast
plt.figure(figsize=(10, 6))
plt.plot(sales_data['date'], sales_data['money'], label='Historical Sales', color='blue')
plt.plot(future_dates, arima_forecast, label='Forecast', color='red')
plt.title('Next 30 Days Coffee Sales Forecast (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.savefig('coffee_sales_forecast_arima.png')
plt.show()

# -------------------------------------
# Image 3: Customer-Specific Purchases (Top 10 Customers)
# -------------------------------------
customer_purchases = sales_data.groupby('coffee_name')['money'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
customer_purchases.head(10).plot(kind='bar', color='green')
plt.title('Top 10 Coffee Types by Total Sales')
plt.xlabel('Coffee Type')
plt.ylabel('Total Sales')
plt.savefig('top_10_customers.png')
plt.show()

# Display top 10 customers in a table
top_customers_df = customer_purchases.head(10).reset_index()
print("Top 10 Coffee Types by Total Sales:")
print(top_customers_df)

# -------------------------------------
# Image 4: Seasonal Analysis (Sales by Month)
# -------------------------------------
sales_data['Month'] = sales_data['date'].dt.month
monthly_sales = sales_data.groupby('Month')['money'].sum()

plt.figure(figsize=(10, 6))
monthly_sales.plot(kind='bar', color='orange')
plt.title('Total Coffee Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.savefig('monthly_sales.png')
plt.show()

# Display monthly sales in a table
monthly_sales_df = monthly_sales.reset_index()
monthly_sales_df.columns = ['Month', 'Total Sales']
print("Total Coffee Sales by Month:")
print(monthly_sales_df)

# -------------------------------------
# Image 5: Sales by Coffee Type
# -------------------------------------
coffee_type_sales = sales_data.groupby('coffee_name')['money'].sum()

plt.figure(figsize=(10, 6))
coffee_type_sales.plot(kind='bar', color='brown')
plt.title('Sales by Coffee Type')
plt.xlabel('Coffee Type')
plt.ylabel('Total Sales')
plt.savefig('sales_by_coffee_type.png')
plt.show()

# Display sales by coffee type in a table
coffee_type_sales_df = coffee_type_sales.reset_index()
coffee_type_sales_df.columns = ['Coffee Type', 'Total Sales']
print("Sales by Coffee Type:")
print(coffee_type_sales_df)