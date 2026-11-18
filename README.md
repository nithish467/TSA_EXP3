# Ex.No: 03   COMPUTE THE AUTO FUNCTION(ACF)
Date: 01/11/2026

### AIM:
To compute the Auto Correlation Function (ACF) of the Tesla Vehicle Sales Performance data for the first 35 lags to determine the model type to fit the data.
### ALGORITHM:
1. Import the necessary packages
2. Find the mean, variance and then implement normalization for the data.
3. Implement the correlation using necessary logic and obtain the results.
4. Store the results in an array.
5. Represent the result in graphical representation as given below.
### PROGRAM:
```python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read Tesla Vehicle Sales dataset
df = pd.read_csv('/content/tesla_vehicle_sales_performance.csv')

# Convert date into datetime format
df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

# Convert revenue into numeric format
df['net_revenue_usd'] = pd.to_numeric(
    df['net_revenue_usd'], errors='coerce'
)

df = df.dropna(subset=['order_date', 'net_revenue_usd'])

# Group monthly revenue
monthly_revenue = df.groupby(
    df['order_date'].dt.to_period('M')
)['net_revenue_usd'].sum()

data = monthly_revenue.values

lags = range(35)

# Pre-allocate autocorrelation table
autocorr_values = []

# Mean
mean_data = np.mean(data)

# Variance
variance_data = np.var(data)

# Normalized data
normalized_data = (data - mean_data) / np.sqrt(variance_data)

# Go through lag components one-by-one
N = len(data)

for lag in lags:
    if lag == 0:
        autocorr_values.append(1)
    else:
        auto_cov = np.sum(
            normalized_data[:-lag] * normalized_data[lag:]
        ) / N

        autocorr_values.append(auto_cov)

# Display the graph
plt.figure(figsize=(10, 6))

plt.stem(lags, autocorr_values)

plt.title('Autocorrelation of Tesla Vehicle Sales Revenue')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')

plt.grid(True)
plt.show()


```

### OUTPUT:

<img width="670" height="430" alt="image" src="https://github.com/user-attachments/assets/7302e9a5-71b4-485d-9355-30fee71ee010" />


### RESULT:
Thus, we have successfully implemented the Auto Correlation Function (ACF) in Python using the Tesla Vehicle Sales Performance dataset. The autocorrelation values for the first 35 lags were calculated and represented graphically.
