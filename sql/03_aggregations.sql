/* 
Showing Aggregate Functions like COUNT(), SUM(), AVG(), MIN(), MAX() and GROUP BY and HAVING statements

Author: Jonas Lucka
Date: 2026
*/

-- COUNT() --

-- Count the total number of rows in the table.
SELECT COUNT(*) AS total_rows
FROM bike_rentals;

-- Count how many rental observations have a recorded rented_bike_count value.
SELECT COUNT(rented_bike_count) AS rental_observations
FROM bike_rentals;


-- SUM() --

-- Calculate the total number of bikes rented across all observations.
SELECT SUM(rented_bike_count) AS total_rentals
FROM bike_rentals;


-- AVG() --

-- Calculate the average number of bikes rented per observation.
SELECT AVG(rented_bike_count) AS average_rentals
FROM bike_rentals;

-- Calculate the average temperature.
SELECT AVG(temperature) AS average_temperature
FROM bike_rentals;


-- MIN() and MAX() --

-- Find the lowest number of rented bikes in an observation.
SELECT MIN(rented_bike_count) AS minimum_rentals
FROM bike_rentals;

-- Find the highest number of rented bikes in an observation.
SELECT MAX(rented_bike_count) AS maximum_rentals
FROM bike_rentals;

-- Find the lowest and highest temperature.
SELECT
MIN(temperature) AS minimum_temperature,
MAX(temperature) AS maximum_temperature
FROM bike_rentals;


-- Combining several aggregate functions --

-- Create a basic summary of bike rentals.

SELECT
COUNT(*) AS observations,
SUM(rented_bike_count) AS total_rentals,
AVG(rented_bike_count) AS average_rentals,
MIN(rented_bike_count) AS minimum_rentals,
MAX(rented_bike_count) AS maximum_rentals
FROM bike_rentals;


-- GROUP BY --

-- Count how many observations belong to each season.
SELECT
seasons,
COUNT(*) AS number_of_observations
FROM bike_rentals
GROUP BY seasons;

-- Calculate the average number of rented bikes for each season.
SELECT
seasons,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY seasons;

-- Calculate the average number of rentals for each hour.
SELECT
hour,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY hour
ORDER BY hour;

-- Calculate the average number of rentlas for each hour and each season.
SELECT
seasons,
hour,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY seasons, hour
ORDER BY seasons, hour;


-- Find which hour has the highest average number of rented bikes.
SELECT
hour,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY hour
ORDER BY average_rentals DESC
LIMIT 1;


-- HAVING --

-- Show only hours where the average number of rentals is greater than 1000.
SELECT
hour,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY hour
HAVING AVG(rented_bike_count) > 1000
ORDER BY average_rentals DESC;

-- Show only sesons where the average number of rentals is greater than 500.
SELECT
seasons,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
GROUP BY seasons
HAVING AVG(rented_bike_count) > 500
ORDER BY average_rentals DESC;

-- Calculate the average rentals for each season using only observations where the temperature is above 10°C.
-- Then keep only seasons with an average above 500 rentals.
SELECT
seasons,
AVG(rented_bike_count) AS average_rentals
FROM bike_rentals
WHERE temperature > 10
GROUP BY seasons
HAVING AVG(rented_bike_count) > 500
ORDER BY average_rentals DESC;
