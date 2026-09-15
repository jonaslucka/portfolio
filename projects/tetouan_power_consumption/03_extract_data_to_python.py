"""
This module extracts the Tetouan power consumption data from power_consumption_ml to Python.

Before running the script replace USERNAME, PASSWORD, HOST and PORT
with your POSTGRESQL username, password, host and port.

Author: Jonas Lucka
Date: 2026
"""

import pandas as pd
from sqlalchemy import create_engine

# Create a connection to PostgreSQL
engine = create_engine(
    "postgresql+psycopg2://USERNAME:PASSWORD@HOST:PORT/tetouan_power"
)


# Select the ML ready columns that is created in 02_prepare_ml_data.sql.
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


# Execute the query and load the result into pandas
tetouan_data = pd.read_sql(QUERY, engine)


# Datetime is stored as a datetime object
tetouan_data["datetime"] = pd.to_datetime(tetouan_data["datetime"])


# Inspect the extracted data

print("First five observations:")
print(tetouan_data.head())

print("\nDataset shape:")
print(tetouan_data.shape)

print("\nColumns:")
print(tetouan_data.columns.tolist())

print("\nData types:")
print(tetouan_data.dtypes)

print("\nMissing values:")
print(tetouan_data.isnull().sum())


#Close the database conection
engine.dispose()
