/* 
Creating and testing a Database about Seoul bike sharing.
The Dataset can be found here: https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand

Author: Jonas Lucka
Date: 2026
*/

-- Create Database --
CREATE DATABASE seoul_bike;

-- Create the bike_rentals table -- 
CREATE TABLE bike_rentals (
    date DATE,
    rented_bike_count INTEGER,
    hour INTEGER,
    temperature DECIMAL(5,2),
    humidity INTEGER,
    wind_speed DECIMAL(5,2),
    visibility INTEGER,
    dew_point_temperature DECIMAL(5,2),
    solar_radiation DECIMAL(6,3),
    rainfall DECIMAL(5,2),
    snowfall DECIMAL(5,2),
    seasons VARCHAR(20),
    holiday VARCHAR(20),
    functioning_day VARCHAR(20)
    );

-- Configure the date format. --
-- The source CSV uses DD/MM/YYYY dates. --

SET datestyle = 'DMY';

--  Import the CSV data --
\copy bike_rentals FROM 'portfolio\sql\data\SeoulBikeData.csv/SeoulBikeData.csv' WITH (FORMAT csv, HEADER true);

--  Verify the number of rows --
-- Expected result: 8760 --
SELECT COUNT(*) FROM bike_rentals;
