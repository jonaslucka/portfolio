"""
This module is experimenting with the LinearRegression class implemented in 
linear_regression.py and the tetouan power consumption dataset.

Before running the script replace USERNAME, PASSWORD, HOST and PORT 
with your POSTGRESQL username, password, host and port.

Author: Jonas Lucka
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from machine_learning import LinearRegression
from mathematical_optimization import optimize_minibatch_sgd


# Connect to PostgreSQL
engine = create_engine(
    "postgresql+psycopg2://USERNAME:PASSWORD@HOST:PORT/tetouan_power"
)

# Load the data
QUERY = """
SELECT
    datetime,
    hour,
    day_of_week,
    month,
    temperature,
    humidity,
    wind_speed,
    general_diffuse_flows,
    diffuse_flows,
    target_power_consumption
FROM power_consumption_ml
ORDER BY datetime;
"""

tetouan_data = pd.read_sql(QUERY, engine)

tetouan_data["datetime"] = pd.to_datetime(tetouan_data["datetime"])


# Define features and target
features = [
    "hour",
    "day_of_week",
    "month",
    "temperature",
    "humidity",
    "wind_speed",
    "general_diffuse_flows",
    "diffuse_flows",
]

target = "target_power_consumption"

X = tetouan_data[features].to_numpy(dtype=float)
y = tetouan_data[target].to_numpy(dtype=float)


# Create a 80/20 train/test split
split_index = int(len(tetouan_data) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print("\nTraining period:")
print(
    tetouan_data["datetime"].iloc[0],
    "to",
    tetouan_data["datetime"].iloc[split_index - 1],
)

print("\nTesting period:")
print(
    tetouan_data["datetime"].iloc[split_index],
    "to",
    tetouan_data["datetime"].iloc[-1],
)


# Standardize the input features
# Minibatch SGD works better when the input features are on similar numerical scales.
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

# Protect against division by zero if a feature is constant.
X_std[X_std == 0] = 1.0

X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std


# Create and train the model
model = LinearRegression()

model.fit(
    X_train_scaled,
    y_train,
    optimizer=optimize_minibatch_sgd,
    learning_rate=0.001,
    epochs=100,
    batch_size=32,
)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evauate accuracy of the model
mae = np.mean(np.abs(y_test - y_pred))

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print("\nModel performance:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")


# Inspect model parameters
# Keep in mind: The model was trained using standardized features.
coefficients = pd.DataFrame({
    "feature": features,
    "coefficient": model.params[:-1]
})

coefficients = coefficients.sort_values(
    by="coefficient",
    ascending=False
)

print("\nModel coefficients:")
print(coefficients.to_string(index=False))

print("\nBias:")
print(f"{model.params[-1]:.4f}")


# Compare actual and predicted value
comparison = pd.DataFrame({
    "datetime": tetouan_data["datetime"].iloc[split_index:].values,
    "actual": y_test,
    "predicted": y_pred
})

print("\nFirst 10 test predictions:")
print(
    comparison.head(10).to_string(index=False)
)


# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(
    model.loss_history,
    label="Training MSE"
)
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss - Mini-batch SGD")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()


# Plot actual vs predicted energy consumption
plot_data = comparison.head(500)
plt.figure(figsize=(12, 5))
plt.plot(
    plot_data["datetime"],
    plot_data["actual"],
    label="Actual"
)
plt.plot(
    plot_data["datetime"],
    plot_data["predicted"],
    label="Predicted"
)
plt.xlabel("Date")
plt.ylabel("Zone 1 Power COnsumption")
plt.title("Linear Regression: Actual vs Predicted Power Consumption")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Close the database conection
engine.dispose()
