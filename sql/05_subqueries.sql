/* 
Showing subqueries

Author: Jonas Lucka
Date: 2026
*/

-- Basic subquery --

-- Find observations where the number of rented bikes is higher than the overall average.
SELECT
date,
hour,
rented_bike_count
FROM bike_rentals
WHERE rented_bike_count > (
SELECT AVG(rented_bike_count)
FROM bike_rentals
)
ORDER BY rented_bike_count DESC;


-- Subqueries with MAX/MIN --

-- Find all observations that have the maximum rental count in the dataset.
SELECT
date,
hour,
rented_bike_count
FROM bike_rentals
WHERE rented_bike_count = (
SELECT MAX(rented_bike_count)
FROM bike_rentals
);

-- Find all observations wiht the minimum rental count.
SELECT
date,
hour,
rented_bike_count,
functioning_day
FROM bike_rentals
WHERE rented_bike_count = (
SELECT MIN(rented_bike_count)
FROM bike_rentals
);


-- Subquery with IN --

-- Find observations belonging to seasons whose average rental count is above 500.
SELECT
date,
hour,
seasons,
rented_bike_count
FROM bike_rentals
WHERE seasons IN (
SELECT seasons
FROM bike_rentals
GROUP BY seasons
HAVING AVG(rented_bike_count) > 500
)
ORDER BY rented_bike_count DESC;


-- Subqueries in FROM --

-- First calculate the average rentlas for every hour then order the results.
SELECT
hourly_data.hour,
hourly_data.average_rentals
FROM (
SELECT
hour,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY hour
) AS hourly_data
ORDER BY average_rentals DESC;

-- First calculate the average rentals for every hour then keep only hours where the average is above 500.
SELECT
hourly_data.hour,
hourly_data.average_rentals
FROM (
SELECT
hour,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY hour
) AS hourly_data
WHERE hourly_data.average_rentals > 500
ORDER BY average_rentals DESC;

-- Find the season with the highest average number of rented bikes.
SELECT
seasons,
average_rentals
FROM (
SELECT
seasons,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY seasons
) AS seasonal_data
ORDER BY average_rentals DESC
LIMIT 1;


-- Correlated subquery --

-- Find observations where the rental count is higher than the average rental count for that same hour.
SELECT
b.date,
b.hour,
b.rented_bike_count
FROM bike_rentals AS b
WHERE b.rented_bike_count > (
SELECT AVG(b2.rented_bike_count)
FROM bike_rentals AS b2
WHERE b2.hour = b.hour
)
ORDER BY b.rented_bike_count DESC;
