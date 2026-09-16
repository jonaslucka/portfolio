"""
This module explores the dataset Tetouan power consumption
to understand its structure. 

Before running the script replace USERNAME, PASSWORD, HOST and PORT 
with your POSTGRESQL username, password, host and port.

Author: Jonas Lucka
Date: 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


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


# Check for missing values
print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(tetouan_data.isnull().sum())


# Descriptive statistics
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(tetouan_data.describe())


# Inspect the target variable Zone 1 Power Consumption
print("\n" + "=" * 60)
print("TARGET VARIABLE")
print("=" * 60)

print("\nAverage consumption:")
print(tetouan_data["target_power_consumption"].mean())

print("\nMinimum consumption:")
print(tetouan_data["target_power_consumption"].min())

print("\nMaximum consumption:")
print(tetouan_data["target_power_consumption"].max())


# Average power consumption by hour
hourly_consumption = (
    tetouan_data.groupby("hour")["target_power_consumption"]
    .mean()
)

print("\n" + "=" * 60)
print("AVERAGE POWER CONSUMPTION BY HOUR")
print("=" * 60)

print(hourly_consumption)

# Correlation between numerical variabkes
correlation = tetouan_data.select_dtypes(include="number").corr()

print("\n" + "=" * 60)
print("CORRELATION")
print("=" * 60)

print(correlation["target_power_consumption"].sort_values(
    ascending=False
))


#Plot average consumption by hour
plt.figure(figsize=(10, 5))
plt.plot(
    hourly_consumption.index,
    hourly_consumption.values
)
plt.xlabel("Hour of Day")
plt.ylabel("Average Zone 1 Power Consumption")
plt.title("Average Power Consumption by Hour")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()


# Scatter plot: Temperature & Power Consumption
plt.figure(figsize=(8, 5))
plt.scatter(
    tetouan_data["temperature"],
    tetouan_data["target_power_consumption"],
    alpha=0.2
)
plt.xlabel("Temperature")
plt.ylabel("Zone 1 Power Consumption")
plt.title("Temperature vs Power Consumption")
plt.tight_layout()
plt.show()


# Scatter plot: Humidity & Power Consumption
plt.figure(figsize=(8, 5))
plt.scatter(
    tetouan_data["humidity"],
    tetouan_data["target_power_consumption"],
    alpha=0.2
)
plt.xlabel("Humidity")
plt.ylabel("Zone 1 Power Consumption")
plt.title("Humidity vs Power Consumption")
plt.tight_layout()
plt.show()


# Close the database conection
engine.dispose()
