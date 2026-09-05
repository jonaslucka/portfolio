/* 
Showing Common Table Expressions (CTEs)

Author: Jonas Lucka
Date: 2026
*/

-- Basic CTE --

-- Find observations where more than 1,000 bikes were rented.
WITH high_rentals AS (
SELECT
date,
hour,
rented_bike_count
FROM bike_rentals
WHERE rented_bike_count > 1000
)
SELECT *
FROM high_rentals
ORDER BY rented_bike_count DESC
LIMIT 10;


-- CTEs with aggregations --

-- Calculate the average number of rentals for each hour.
WITH hourly_average AS (
SELECT
hour,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY hour
)
SELECT
hour,
average_rentals
FROM hourly_average
ORDER BY hour;

-- First calculate the average rentals for each hour then keep only hours with an average above 500.
WITH hourly_average AS (
SELECT
hour,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY hour
)
SELECT
hour,
average_rentals
FROM hourly_average
WHERE average_rentals > 500
ORDER BY average_rentals DESC;


-- CTE compared with the overall average --

-- First calculate the overall average rental count then find observations that are above that average
WITH overall_average AS (
SELECT
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
)
SELECT
b.date,
b.hour,
b.rented_bike_count,
o.average_rentals
FROM bike_rentals AS b
CROSS JOIN overall_average AS o
WHERE b.rented_bike_count > o.average_rentals
ORDER BY b.rented_bike_count DESC
LIMIT 10;


-- CTE with multiple colums --

-- Calculate several weather and rental statistics for each season.
WITH seasonal_summary AS (
SELECT
seasons,
AVG(temperature) AS average_temperature,
AVG(humidity) AS average_humidity,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY seasons
)
SELECT
seasons,
average_temperature,
average_humidity,
average_rentals
FROM seasonal_summary
ORDER BY average_rentals DESC;


-- Multiple CTEs --

-- The first CTE calculates the average rental count for every hour.
-- The second CTE finds the highest hourly average.
-- The final query compares every hour with that maximum.
WITH hourly_average AS (
SELECT
hour,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY hour
),
maximum_average AS (
SELECT
MAX(average_rentals) AS highest_average_rentals
FROM hourly_average
)
SELECT
h.hour,
h.average_rentals,
m.highest_average_rentals
FROM hourly_average AS h
CROSS JOIN maximum_average AS m
ORDER BY h.average_rentals DESC;
