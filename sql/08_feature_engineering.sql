/* 
Transforming raw data into new features (also called feature engineering)

Author: Jonas Lucka
Date: 2026
*/

-- Create a rush hour indicator.
SELECT
date,
hour,
rented_bike_count,
CASE
WHEN hour IN (8, 18) THEN 1
ELSE 0
END AS is_rush_hour
FROM bike_rentals
LIMIT 100;


-- Covert the numerical temperature into categories
SELECT
date,
hour,
temperature,
CASE
WHEN temperature < 0 THEN 'Cold'
WHEN temperature < 15 THEN 'Mild'
WHEN temperature < 25 THEN 'Warm'
ELSE 'Hot'
END AS temperature_category
FROM bike_rentals;


-- Convert Functioning Day from text into a numerical feature.
SELECT
date,
hour,
functioning_day,
CASE
WHEN functioning_day = 'Yes' THEN 1
ELSE 0
END AS functioning_day_numeric
FROM bike_rentals
LIMIT 100;


-- Convert the Holiday column into a numerical feature.
SELECT
date,
hour,
holiday,
CASE
WHEN holiday = 'Holiday' THEN 1
ELSE 0
END AS holiday_numeric
FROM bike_rentals
LIMIT 100;


-- Create a weekend indicator
SELECT
date,
hour,
EXTRACT(DOW FROM date) AS day_of_week,
CASE
WHEN EXTRACT(DOW FROM date) IN (0, 6) THEN 1
ELSE 0
END AS is_weekend
FROM bike_rentals
LIMIT 100;


-- Create a time of day category
SELECT
date,
hour,
CASE
WHEN hour < 6 THEN 'Night'
WHEN hour < 12 THEN 'Morning'
WHEN hour < 18 THEN 'Afternoon'
ELSE 'Evening'
END AS time_of_day
FROM bike_rentals
LIMIT 100;


-- Create a weather condition feature --
SELECT
date,
hour,
rainfall,
snowfall,
CASE
WHEN rainfall > 0 OR snowfall > 0 THEN 1
ELSE 0
END AS bad_weather
FROM bike_rentals
LIMIT 100;


--Create a demand category
SELECT
date,
hour,
rented_bike_count,
CASE
WHEN rented_bike_count < 200 THEN 'Low'
WHEN rented_bike_count < 800 THEN 'Medium'
ELSE 'High'
END AS demand_category
FROM bike_rentals
LIMIT 100;


-- Create a final dataset
SELECT
date,
hour,
temperature,
humidity,
wind_speed,
visibility,
rainfall,
snowfall,
seasons,
rented_bike_count,

-- Holiday
CASE
    WHEN holiday = 'Holiday' THEN 1
    ELSE 0
END AS holiday_numeric,

-- Functioning day
CASE
    WHEN functioning_day = 'Yes' THEN 1
    ELSE 0
END AS functioning_day_numeric,

-- Rush hour
CASE
    WHEN hour IN (8, 18) THEN 1
    ELSE 0
END AS is_rush_hour,

-- Temperature Categories
CASE
    WHEN temperature < 0 THEN 'Cold'
    WHEN temperature < 15 THEN 'Mild'
    WHEN temperature < 25 THEN 'Warm'
    ELSE 'Hot'
END AS temperature_category,

-- Weekend
CASE
    WHEN EXTRACT(DOW FROM date) IN (0, 6) THEN 1
    ELSE 0
END AS is_weekend,

-- Bad weather
CASE
    WHEN rainfall > 0 OR snowfall > 0 THEN 1
    ELSE 0
END AS bad_weather,

-- Time of day
CASE
    WHEN hour < 6 THEN 'Night'
    WHEN hour < 12 THEN 'Morning'
    WHEN hour < 18 THEN 'Afternoon'
    ELSE 'Evening'
END AS time_of_day,

-- Demand
CASE
    WHEN rented_bike_count < 200 THEN 'Low'
    WHEN rented_bike_count < 800 THEN 'Medium'
    ELSE 'High'
END AS demand_category

FROM bike_rentals
ORDER BY date, hour
LIMIT 100;
