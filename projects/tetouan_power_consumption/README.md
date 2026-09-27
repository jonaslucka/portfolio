# Tetouan City Power Consumption

This project uses the Tetouan City Power Consumption dataset to investigate whether electricity consumption can be predicted from time, weather and light measurements.

The project combines:

* PostgreSQL for data storage and preparation
* SQL for data and feature preparation
* Python and NumPy for model training
* Linear Regression implemented from scratch
* Minibatch Stochastic Gradient Descent from scratch for optimization
* Matplotlib for visualization

## Dataset

The dataset contains the electricity consumption and various additional measurements from Tetouan in Morocco.

The Measurements of this dataset were recorded at 10 minutes intervals during 2017.

The dataset contains:

* 52,416 observations (rows)
* 9 columns:
* 1 datetime column
* 5 environmental and light measurements
* 3 electricity consumption measurements

| Variable                   | Description                       |
| -------------------------- | --------------------------------- |
| `DateTime`                 | Date and time of the measurement  |
| `Temperature`              | Temperature                       |
| `Humidity`                 | Relative humidity                 |
| `Wind Speed`               | Wind speed                        |
| `general diffuse flows`    | General diffuse light flow        |
| `diffuse flows`            | Diffuse light flow                |
| `Zone 1 Power Consumption` | Electricity consumption in Zone 1 |
| `Zone 2 Power Consumption` | Electricity consumption in Zone 2 |
| `Zone 3 Power Consumption` | Electricity consumption in Zone 3 |

This project focuses on **Zone 1 Power consumption**

## Data Preperation

### PostgreSQL

The raw CSV datset is imported into PostgreSQL and stored in the power_consumption table.

SQL is used to:

* create the database table
* import the dataset
* verify the number of observations
* check the avaible time period
* check for missing values
* prepare the machine learning dataset

The prepared data is stored in power_consumption_ml.

Additional time based features hour, day of week and month are extracted from DateTime:

## Data Exploration

Before training the model the prepared dataset is explored to understand understand its structure and identify relationships between the variables.

### Data Quality

No missing values were found in the dataset and therefore all 52,416 observations are used in the analysis.

### Zone 1 Consumption
The target variable has the following values:

| Statistic | Consumption |
| --------- | ----------: |
| Mean      |   32,344.97 |
| Minimum   |   13,895.70 |
| Maximum   |   52,204.40 |


### Consumption Throughout the Day

Average electricity consumption changes considerably depending on the hour of the day.

The average consumption reaches its lowest level around the early morning and its highest level around the evening.

![Average power consumption by hour](images/average_power_by_hour.png)

This strong daily pattern is also reflected in the model where hour receives the largest standardized coefficient.

### Correlation

The correlation between the features an Zone 1 power consumption was also examined.

| Feature               | Correlation |
| --------------------- | ----------: |
| Hour                  |       0.728 |
| Temperature           |       0.440 |
| General diffuse flows |       0.188 |
| Wind speed            |       0.167 |
| Diffuse flows         |       0.080 |
| Day of week           |       0.040 |
| Month                 |      -0.005 |
| Humidity              |      -0.287 |

The strongest linear relationship is between hour and power consumption.

## Machine Learning

### Model

The project uses the portfolios own Linear Regression class implemented from scratch using NumPy.

The model is:

$$\hat{y} = Xw + b$$

where:

* X represents the input features
* w represents the learned coefficients
* b represents the bias
* $\hat{y}$ represents the predicted power consumption

The optimization is performed using the portfolios own Minibatch SGD implementation.

### Features

The model uses eight input features:

```text
hour
day_of_week
month
temperature
humidity
wind_speed
general_diffuse_flows
diffuse_flows
```

The target is:

```text
target_power_consumption
```

### Train/Test Split

Because the observations are time ordered a chronological split is used for the train and testing instead of randomly shuffling the complete dataset.

Training:

**41,932 observations: 80%**

```text
2017-01-01 00:00:00
        ↓
2017-10-19 04:30:00
```

Testing:

**10,484 observations: 20%**

```text
2017-10-19 04:40:00
        ↓
2017-12-30 23:50:00
```

### Feature Standardization

The input variables have different numerical scales so the features are standardized before the training.

For each feature:

$$x_{scaled} =\frac{x-\mu}{\sigma}$$

Here $\mu$ is the mean adn $\sigma$ is the standard deviation.

### Optimization

The Linear Regression model is optimized using a custom implementation of Minibatch Stochastic Gradient Descent.

For each epoch:

1. The training data is shuffled
2. The observations are divided into minibatches.
3. The gradient is calculated for each minibatch.
4. The parameters are updated.
5. The training loss is recorded.

The current experiment uses:

| Parameter     | Value |
| ------------- | ----: |
| Learning rate | 0.001 |
| Epochs        |   100 |
| Batch size    |    32 |

## Results

### Test performance

The MAE measures the average absolute difference between the predicted and actual consumption:

$$MAE =\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$$

RMSE penalize larger errors more strongly than MAE:

$$RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}$$

$R^2$ measures the proportion of variation explained by the model relative to a mean based baseline:

$$R^2 =1 -\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$$

The model achieved the following performance:

| Metric |       Result |
| ------ | -----------: |
| MAE    | **3,259.46** |
| RMSE   | **4,093.23** |
| R²     |   **0.5594** |

### Model Parameters

Because the input feature were standardized each coefficient represents the change in predicted power consumption associated with a one standard deviation increase in that feature while the other features are held constant.

| Feature                 | Coefficient |
| ----------------------- | ----------: |
| `hour`                  |     4921.89 |
| `temperature`           |     2412.46 |
| `day_of_week`           |      306.36 |
| `humidity`              |       28.67 |
| `wind_speed`            |      -56.88 |
| `month`                 |     -195.20 |
| `general_diffuse_flows` |     -201.99 |
| `diffuse_flows`         |     -539.13 |

The learned bias is:

```text
33042.6112
```

These coefficients describe the fitted statistical relationship in the model.
They should not be interpreted ass causal effects.

### Predictions

The following plot compares the actual Zone 1 consumption wiht the model predictions during the test period.

![Actual vs predicted power consumption](images/actual_vs_predicted.png)

### Model Interpretation

The linear regression model achieved an $R^2$ of 0.559 on the test set meaning that the model explains approximately $56\%$ of the variation in power consumption.

The model provides a useful baseline for this dataset but the remaining unexplained variation suggests that power consumption is influenced by factors that are not fully captured by the current features or by a linear relationship.

The results also provide a starting point for future experiments such as adding more informative time based features, transforming existing variables or comparing the linear mode with more flexible machine learning algorithms.

## Reproducibility

The project should be run from the root of the portfolio repository.

First configure the PostgreSQL database and import the dataset using the SQl scripts.

The PostgreSQL connection settings in the Python scripts must be configured for the local environment before running them.

### Extract the data

```powershell
python -m projects.tetouan_power_consumption.03_extract_data_to_python
```

### Explore the data

```powershell
python -m projects.tetouan_power_consumption.04_data_exploration
```

### Train and evaluate the model

```powershell
python -m projects.tetouan_power_consumption.05_linear_regression
```

## Dataset Reference

* Salam, A., & El Hibaoui, A. (2018). *Comparison of Machine Learning Algorithms for the Power Consumption Prediction: Case Study of Tetouan city*. UCI Machine Learning Repository.
  https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city