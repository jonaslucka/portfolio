/* 
Showing basic queries like LIMIT, ORDER BY, DISTINCT, WHERE, AND/OR.

Author: Jonas Lucka
Date: 2026
*/

--  SELECT, AS and LIMIT --

-- Show the first 10 rows
SELECT *
FROM bike_rentals
LIMIT 10;

-- Select specific colums and give aliases
SELECT
date,
hour,
rented_bike_count AS bike_count,
temperature AS temp,
humidity AS humid
FROM bike_rentals
LIMIT 10;


-- ORDER BY --

-- Show the date and hours with the highest number of rentals
SELECT
date,
hour,
rented_bike_count
FROM bike_rentals
ORDER BY rented_bike_count DESC
LIMIT 10;

-- Show the coldest observed hours
SELECT
date,
hour,
temperature
FROM bike_rentals
ORDER BY temperature ASC
LIMIT 10;


-- DISTINCT --

-- Show all different seasons
SELECT DISTINCT seasons
FROM bike_rentals;

-- Show all different holiday categories
SELECT DISTINCT holiday
FROM bike_rentals;


-- WHERE --

-- Show observations where more than 3,000 bikes were rented
SELECT
date,
hour,
rented_bike_count
FROM bike_rentals
WHERE rented_bike_count > 3000;

-- Show observations where the temperature is below 10°C
SELECT
date,
hour,
temperature
FROM bike_rentals
WHERE temperature < -10;

-- Show observations from hour 12
SELECT
date,
hour,
rented_bike_count
FROM bike_rentals
WHERE hour = 12;


-- AND / OR --

-- Show observations where it is cold AND more than 500 bikes were rented
SELECT
date,
hour,
temperature,
rented_bike_count
FROM bike_rentals
WHERE temperature < 0 AND rented_bike_count > 500;

-- Show observations from either hour 8 or hour 18
SELECT
date,
hour,
rented_bike_count
FROM bike_rentals
WHERE hour = 8 OR hour = 18;


-- Combining everything

-- Find the highest rental counts during snow
SELECT
date,
hour,
temperature,
rainfall,
rented_bike_count
FROM bike_rentals
WHERE temperature < 0 AND rainfall != 0 
ORDER BY rented_bike_count DESC
LIMIT 10;

-- Find the lowest rental counts on a warm day during summer OR spring AND on a holiday
-- with the use of IN
SELECT
date,
hour,
temperature,
rented_bike_count
FROM bike_rentals
WHERE temperature >= 20
AND holiday = Holiday
AND seasons IN ('Spring', 'Summer')
ORDER BY rented_bike_count ASC
LIMIT 10;
