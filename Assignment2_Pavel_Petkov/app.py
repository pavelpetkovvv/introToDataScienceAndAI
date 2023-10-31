import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

# Data Cleaning and Preprocessing
# Handling missing values (in this case, we remove rows with missing values in the columns Living_area and Selling_price)
columns_to_check = ['Living_area', 'Selling_price']
data_cleaned = data.dropna(subset=columns_to_check)

# Split the data into features (living area) and target (selling price)
x = data_cleaned[['Living_area']]  # Feature (independent variable)
y = data_cleaned['Selling_price']  # Target (dependent variable)

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(x, y)
# model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Extract the slope (coefficient) and intercept
slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope (Coefficient): {slope}")
print(f"Intercept: {intercept}")

# Assuming you already have the trained model (model) and the input feature values
living_areas_to_predict = np.array([100, 150, 200]).reshape(-1, 1)  # Reshape the input to a 2D array

# Use the model to make predictions
predicted_prices = model.predict(living_areas_to_predict)

# Display the predicted selling prices
for living_area, price in zip([100, 150, 200], predicted_prices):
    print(f"Living Area: {living_area} m², Predicted Selling Price: {price:.2f}")

# Visualize the model fit
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred, color='red', linewidth=2, label='Linear Regression Fit')
plt.xlabel('Living Area')
plt.ylabel('Selling Price')
plt.legend()
plt.title('Linear Regression Model')
plt.show()