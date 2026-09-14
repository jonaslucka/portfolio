/* 
Preparing a Machine Learning Dataset that can be loaded into Python.

The Target variable is zone_1_power_consumption.

Author: Jonas Lucka
Date: 2026
*/


-- Check the earliest and latest observations in the dataset.
SELECT
    MIN(datetime) AS first_timestamp,
    MAX(datetime) AS last_timestamp
FROM power_consumption;


-- Check for missing values
SELECT
    COUNT(*) - COUNT(datetime) AS missing_datetime,
    COUNT(*) - COUNT(temperature) AS missing_temperature,
    COUNT(*) - COUNT(humidity) AS missing_humidity,
    COUNT(*) - COUNT(wind_speed) AS missing_wind_speed,
    COUNT(*) - COUNT(general_diffuse_flows) AS missing_general_diffuse_flows,
    COUNT(*) - COUNT(diffuse_flows) AS missing_diffuse_flows,
    COUNT(*) - COUNT(zone_1_power_consumption) AS missing_zone_1,
    COUNT(*) - COUNT(zone_2_power_consumption) AS missing_zone_2,
    COUNT(*) - COUNT(zone_3_power_consumption) AS missing_zone_3
FROM power_consumption;


/*
Create an ML-ready dataset
We extract useful information from datetime:

hour: Hour of the day (0-23)
day_of_week: Day of the week (0 = Sunday, 6 = Saturday)
month: Month of the year (1-12)


We keep the weather and light variables from the original dataset.

Zone 1 Power Consumption will be our target variable
*/

CREATE TABLE power_consumption_ml AS
SELECT
    datetime,
    EXTRACT(HOUR FROM datetime) AS hour,
    EXTRACT(DOW FROM datetime) AS day_of_week,
    EXTRACT(MONTH FROM datetime) AS month,
    temperature,
    humidity,
    wind_speed,
    general_diffuse_flows,
    diffuse_flows,
    zone_1_power_consumption AS target_power_consumption
FROM power_consumption;


-- Inspect the ML-ready dataset
SELECT *
FROM power_consumption_ml
LIMIT 10;


-- Check the number of observations for the ML ready dataset
SELECT COUNT(*)
FROM power_consumption_ml;


-- Inspect the target variabl
SELECT
    MIN(target_power_consumption) AS minimum_consumption,
    MAX(target_power_consumption) AS maximum_consumption,
    AVG(target_power_consumption) AS average_consumption
FROM power_consumption_ml;
