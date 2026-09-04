/* 
Showing how to join Tables using INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN, SELF JOIN

Author: Jonas Lucka
Date: 2026
*/

-- Create a small reference table --


-- Create a table containing additional information about each season.
-- Is only used to demonstrate JOINs

CREATE TABLE season_info (
seasons VARCHAR(20),
average_temperature_description VARCHAR(50)
);

-- Add information about the four seasons.

INSERT INTO season_info (
seasons,
average_temperature_description
)
VALUES
('Winter', 'Cold'),
('Spring', 'Mild'),
('Summer', 'Warm'),
('Autumn', 'Cool');

-- Check the new table.

SELECT *
FROM season_info;


-- INNER JOIN --

-- INNER JOIN returns only rows where there is a match in both tables.

SELECT
b.date,
b.seasons,
b.temperature,
s.average_temperature_description
FROM bike_rentals AS b
INNER JOIN season_info AS s
ON b.seasons = s.seasons
LIMIT 365;

-- LEFT JOIN --

-- LEFT JOIN keeps every row from the left table (bike_rentals).
-- If there is no matching row in season_info the colums from season_info will contain NULL.

SELECT
b.date,
b.seasons,
b.temperature,
s.average_temperature_description
FROM bike_rentals AS b
LEFT JOIN season_info AS s
ON b.seasons = s.seasons
LIMIT 365;


-- RIGHT JOIN --

-- RIGHT JOIN keeps every row from the right table (season_info).
-- If there is no matching row in bike_rentals, the columns from bike_rentals will contain NULL.

SELECT
b.date,
b.seasons,
b.temperature,
s.average_temperature_description
FROM bike_rentals AS b
RIGHT JOIN season_info AS s
ON b.seasons = s.seasons
LIMIT 365;


-- FULL OUTER JOIN --

-- FULL OUTER JOIN keeps all rows from both tables.
-- Rows without a match contain NULL values for the other table.

SELECT
b.date,
b.seasons,
b.temperature,
s.average_temperature_description
FROM bike_rentals AS b
FULL OUTER JOIN season_info AS s
ON b.seasons = s.seasons
LIMIT 365;

-- All the JOINs above should give the same table


-- SELF JOIN --

-- Compare rental observatoins from two different hours on the same date.

SELECT
b1.date,
b1.hour AS hour_1,
b1.rented_bike_count AS rentals_hour_1,
b2.hour AS hour_2,
b2.rented_bike_count AS rentals_hour_2
FROM bike_rentals AS b1
INNER JOIN bike_rentals AS b2
ON b1.date = b2.date
WHERE b1.hour = 8
AND b2.hour = 18
LIMIT 10;


-- Combine Everything --

-- Use the JOIN to get the season description then filter for warm seasons.

SELECT
b.date,
b.hour,
b.seasons,
b.temperature,
b.rented_bike_count,
s.average_temperature_description
FROM bike_rentals AS b
INNER JOIN season_info AS s
ON b.seasons = s.seasons
WHERE s.average_temperature_description = 'Warm'
LIMIT 10;

-- Calculate the average rentals for each temperature
-- description and keep only groups with an average above 500 rentals.

SELECT
s.average_temperature_description,
AVG(b.rented_bike_count) AS average_rentals
FROM bike_rentals AS b
INNER JOIN season_info AS s
ON b.seasons = s.seasons
GROUP BY s.average_temperature_description
HAVING AVG(b.rented_bike_count) > 500
ORDER BY average_rentals DESC;
