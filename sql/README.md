# SQL

This directory contains SQL exercises and data analysis using PostgreSQL.

The exercises use the Seoul Bike Sharing Demand dataset from the UC Irvine Machine Learning Repository to practice SQL concepts ranging from basic queries to more advanced analytical techniques.

The SQL used as part of larger end to end projects is kept inside the corresponding project directory under `projects/`.

## Database Setup

The dataset is imported into PstgreSQL and stored un a table named bike_rentals.

Before running the import command replace PATH_TO_SEOUL_CSV with the local path to the dataset on your computer.

The setup file contains the table definiton, data import and basic verification queries.

See [`01_setup.sql`](01_setup.sql)

## Basic Queries

The basic query exercises cover selecting, filtering and sorting data.

Topics included:

* SELECT
* WHERE
* ORDER BY
* DISTINCT
* AND and OR
* LIMIT

See [`02_basic_queries.sql`](02_basic_queries.sql)

## Aggregations

Aggregation exercises cover summarizing and analyzing observations.

Topicis included:

* COUNT
* SUM
* AVG
* MIN and MAX
* GROUP BY
* HAVING

See [`03_aggregations.sql`](03_aggregations.sql)

## Joins

Join exercises cover combining data from multiple tabes and comparing related observations.

Topics included:

* INNER JOIN
* LEFT JOIN
* RIGHT JOIN
* FULL JOIN
* SELF JOIN

See [`04_joins.sql`](04_joins.sql)

## Subqueries

Subquery exercises cover using queries inside other queries to perform more complex analysis.

Topics included:

* Basic subqueries
* IN subqueries
* Subqueries in FROM
* Correlated subqueries

See [`05_subqueries.sql`](05_subqueries.sql)

## Common Table Expression

Common Table Expressions (CTEs) are used to create temporary named result sets that can be referenced within a query.

Topics included:

* Basic CTEs
* Aggregation with CTEs
* Multiple CTEs

See [`06_ctes.sql`](06_ctes.sql)

## Window Functions

Window functions perform calculations across related rwos without collapsing the result into groups.

Topics included:

* OVER
* PARTITION BY
* ROW_NUMBER
* RANK
* LAG and LEAD
* Running totals

See [`07_window_functions.sql`](07_window_functions.sql)

## Feature Engineering

SQL can also be used to derive and create new features.

Examples:

* Rush hour indicator
* Temperature categories
* Numeric representations of categorical variables
* Weekend indicators
* Time-of-day categories
* Weather and demand categories

See [`08_feature_engineering.sql`](08_feature_engineering.sql)

## Preperation for Machine Learning

SQL can also be used to prepare a cleaned Machine Learning Dataset.

See [`09_dataset_for_ml.sql`](09_dataset_for_ml.sql)