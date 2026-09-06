/* 
Showing Window Functions with OVER() like PARTITION BY, ROW_NUMBER(), RANK(), LAG(), LEAD()

Author: Jonas Lucka
Date: 2026
*/

-- Basic window function with OVER() --

-- Calculate the overall average rental count while keeping every individual observation.
SELECT
date,
hour,
rented_bike_count,
AVG(rented_bike_count) OVER () AS overall_average
FROM bike_rentals
LIMIT 10;

-- Compare each observation with the overall average
SELECT
date,
hour,
rented_bike_count,
AVG(rented_bike_count) OVER () AS overall_average,
rented_bike_count
- AVG(rented_bike_count) OVER () AS difference_from_average
FROM bike_rentals
LIMIT 10;


-- PARTITION BY --

-- Calculate the average rental count for each hour.
SELECT
date,
hour,
rented_bike_count,
AVG(rented_bike_count) 
OVER (PARTITION BY hour)
AS hourly_average
FROM bike_rentals;


-- ORDER BY and RANK--

-- Rank every observation from highest to lowest rental count.
SELECT
date,
hour,
rented_bike_count,
RANK()
OVER(ORDER BY rented_bike_count DESC)
AS rental_rank
FROM bike_rentals;

-- Rank rental observations separately for each season.
SELECT
date,
hour,
seasons,
rented_bike_count,
RANK()
OVER (PARTITION BY seasons
ORDER BY rented_bike_count DESC)
AS seasonal_rank
FROM bike_rentals;

-- Calculate a cumulative total of rented bikes over time
SELECT
date,
hour,
rented_bike_count,
SUM(rented_bike_count)
OVER (ORDER BY date, hour)
AS running_total_rentals
FROM bike_rentals
LIMIT 10;

-- ROW_NUMBER() --

-- Assign a uniwue row number to observations ordered by rental count.
SELECT
date,
hour,
rented_bike_count,
ROW_NUMBER()
OVER (ORDER BY rented_bike_count DESC)
AS row_number
FROM bike_rentals
LIMIT 10;


-- LAG() --

-- Compare each rental count with the previous observation.
SELECT
date,
hour,
rented_bike_count,
LAG(rented_bike_count)
OVER (ORDER BY date, hour)
AS previous_rentals
FROM bike_rentals
LIMIT 10;

-- Calculate change from the previous observation
SELECT
date,
hour,
rented_bike_count,
LAG(rented_bike_count)
OVER (ORDER BY date, hour)
AS previous_rentals,
rented_bike_count
- LAG(rented_bike_count)
OVER (ORDER BY date, hour)
AS change_from_previous
FROM bike_rentals
LIMIT 10;


-- LEAD() --

-- Compare the current observation with the next observation.
SELECT
date,
hour,
rented_bike_count,
LEAD(rented_bike_count)
OVER (ORDER BY date, hour)
AS next_rentals
FROM bike_rentals
LIMIT 10;
