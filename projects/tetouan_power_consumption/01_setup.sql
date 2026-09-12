/*
Setup file for the project Tetouan City Power Consumption
The Dataset can be found here: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city

Before running the import command replace PATH_TO_TETOUAN_CSV with the local path to the dataset on your computer.

Author: Jonas Lucka
Date: 2026
*/

-- Create the project database.
CREATE DATABASE tetouan_power;

-- Create the main table.
CREATE TABLE power_consumption (
datetime TIMESTAMP,
temperature DECIMAL(6,3),
humidity DECIMAL(6,2),
wind_speed DECIMAL(6,3),
general_diffuse_flows DECIMAL(10,3),
diffuse_flows DECIMAL(10,3),
zone_1_power_consumption DECIMAL(12,4),
zone_2_power_consumption DECIMAL(12,4),
zone_3_power_consumption DECIMAL(12,4)
);

-- Configure the date format.
-- The source CSV uses MM/DD/YYYY dates.
SET datestyle = 'MDY';

-- Import the CSV file.
\copy power_consumption
FROM 'PATH_TO_TETOUAN_CSV\Tetuan City power consumption.csv'
WITH (FORMAT csv,HEADER true);

-- Check the number of imported observations.
SELECT COUNT(*)
FROM power_consumption;

-- Display the table structure.
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'power_consumption'
ORDER BY ordinal_position;

-- Preview the imported data.
SELECT *
FROM power_consumption
LIMIT 100;
