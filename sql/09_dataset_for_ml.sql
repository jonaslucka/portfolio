/* 
Preparing a Machine Learning Dataset that can be loaded into Python

The Target variable is rented_bike_count

Author: Jonas Lucka
Date: 2026
*/

-- Examine rented_bike_count
SELECT
COUNT(*) AS observations,
AVG(rented_bike_count) AS average_rentals,
MIN(rented_bike_count) AS minimum_rentals,
MAX(rented_bike_count) AS maximum_rentals
FROM bike_rentals
-- Only observations where the bike sharing system was functioning should be included.
WHERE functioning_day = 'Yes';


-- Check for missing values
SELECT
COUNT(*) - COUNT(temperature) AS missing_temperature,
COUNT(*) - COUNT(humidity) AS missing_humidity,
COUNT(*) - COUNT(wind_speed) AS missing_wind_speed,
COUNT(*) - COUNT(visibility) AS missing_visibility,
COUNT(*) - COUNT(rainfall) AS missing_rainfall,
COUNT(*) - COUNT(snowfall) AS missing_snowfall,
COUNT(*) - COUNT(rented_bike_count) AS missing_target
FROM bike_rentals
-- Only observations where the bike sharing system was functioning should be included.
WHERE functioning_day = 'Yes';


-- This is the final query that can later be exported and loaded into Python.

-- Features:
-- date
-- hour
-- temperature
-- humidity
-- wind_speed
-- visibility
-- rainfall
-- snowfall
-- hour
-- holiday_numeric
-- functioning_day_numeric
-- is_weekend
-- is_rush_hour

---------------

-- Target (y):
-- rented_bike_count

WITH ml_dataset AS (
SELECT
date,
hour,
temperature,
humidity,
wind_speed,
visibility,
rainfall,
snowfall,

    CASE
        WHEN holiday = 'Holiday' THEN 1
        ELSE 0
    END AS holiday_numeric,

    CASE
        WHEN EXTRACT(DOW FROM date) IN (0, 6) THEN 1
        ELSE 0
    END AS is_weekend,

    CASE
        WHEN hour IN (8, 18) THEN 1
        ELSE 0
    END AS is_rush_hour,

    rented_bike_count

FROM bike_rentals

-- Only observations where the bike-sharing system was functioning should be included.
WHERE functioning_day = 'Yes'
)

SELECT *
FROM ml_dataset
ORDER BY date, hour;
