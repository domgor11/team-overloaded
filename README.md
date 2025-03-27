# TEAM OVERLOADED

## Wearable Data Processing for Athlete Performance Analysis - Project Overview

This repository contains code for processing wearable device data from athletes to enable clustering and analysis of athlete states for health and performance optimization. The script handles data cleaning, preprocessing, and imputation of missing values to prepare the data for advanced analysis.

## Project Context

This work supports the development of tools to help athletes better understand and manage their health and performance. The dataset includes:
- Psychological scores (tiredness, motivation, mood)
- Physiological data (HRV, heart rate, steps)
- Perceived performance indicators

The processed data will be used to identify distinct athlete states (such as Overloaded, Underloaded, or Effective) through clustering techniques, and to investigate the relationship between physiological/psychological factors and performance.

## Hackathon Participation

This project was developed as part of the HealthTech AI Hub Hackathon.

The challenge focused on using AI techniques to analyze wearable data from athletes and identify patterns that could optimize health and performance.


## Features

The data processing script provides the following key features:

- **Data Cleaning and Preprocessing**: Handles missing data and sparse features in the dataset
- **Emotion Processing**: Categorizes emotions into high/low positive/negative groups
- **Time-Series Data Processing**: Splits data chronologically per user (80/20 train/test)
- **Advanced Imputation**:
  - Missing Sleep Heart Rate values filled using HRV
  - Missing HRV values filled using Sleep Heart Rate
  - Missing Steps values imputed using regression and distribution-based methods
  - Missing emotional data imputed using exponential weighted moving average

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Installation

Clone this repository and install the required packages:

```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Place the raw data file (`wearable_users_raw_data.csv`) in the project directory.
2. Run the main script:

```bash
python wearable_data_processing.py
```

3. The script will generate:
   - `Train_80_Data.csv`: 80% training data split by user and time
   - `Test_20_Data.csv`: 20% testing data split by user and time
   - `training_imputed.csv`: Training data with imputed values
   - `test_imputed.csv`: Test data with imputed values
   - Visualization figures saved in the `./output` directory

## Data Processing Steps

1. **Initial Preprocessing**:
   - Removal of unnecessary columns
   - Creation of aggregated emotion categories
   - Removal of empty rows

2. **Train-Test Split**:
   - Chronological split (80/20) for each user
   - Ensures that models trained on historical data can be evaluated on future data

3. **HRV and Sleep Heart Rate Imputation**:
   - Linear regression models to predict missing values based on user-specific patterns
   - Values converted to integers for consistency

4. **Steps Data Imputation**:
   - Linear interpolation for short gaps
   - Regression modeling with feature scaling for moderate gaps
   - Distribution-based sampling for difficult cases

5. **Sleep Data Imputation**:
   - User-specific probability-based filling
   - Global statistics for fallback

6. **Emotion Data Processing**:
   - Exponential weighted moving average to capture trends
   - Forward-fill method for remaining gaps

## Completed Subtasks

This script addresses the core sub-challenges of the Athlete State Clustering project:

1. **Data Exploration and Preprocessing**: The script provides comprehensive cleaning and handling of missing values across all critical features.

2. **Preparation for Clustering**: The output datasets maintain temporal information and user identifiers while ensuring complete data for effective clustering.

3. **Preparation for Label Assignment**: The clean datasets can be used to develop supervised models for assigning labels to new users.

4. **Support for Performance Analysis**: The processed data preserves the relationship between self-reported performance and other variables, making it suitable for developing predictive models.

## Output Files

- `Train_80_Data.csv`: Raw training split (80% of data)
- `Test_20_Data.csv`: Raw testing split (20% of data)
- `training_imputed.csv`: Fully processed training data with all imputed values
- `test_imputed.csv`: Fully processed test data with all imputed values

## Visualizations

The script generates several visualizations in the `./output` directory:

- Missing values heatmaps
- Steps distribution before and after imputation
- Example of nutrition data imputation for a sample user
- Correlation matrices

These visualizations help validate the quality of the imputation methods.

## Contributors

- Amy Duguid
- Dominika Gorgosz
- Eve Day
- Jingying Liang
- Scott Brooks
