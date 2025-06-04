# TEAM OVERLOADED

## Wearable Data Processing for Athlete Performance Analysis 

## Project Overview

This repository contains code for processing wearable device data from athletes to enable clustering and analysis of athlete states for health and performance optimization. The script handles data cleaning, preprocessing, and imputation of missing values to prepare the data for advanced analysis.

## Project Context

This work supports the development of tools to help athletes better understand and manage their health and performance. The dataset includes:
- Psychological scores (tiredness, motivation, mood)
- Physiological data (HRV, heart rate, steps)
- Perceived performance indicators

The processed data will be used to identify distinct athlete states (such as Overloaded, Underloaded, or Effective) through clustering techniques, and to investigate the relationship between physiological/psychological factors and performance.

## Hackathon Participation

This project was developed as part of the HealthTech AI Hub Hackathon:
[HealthTech AI Hub Hackathon 2025](https://www.linkedin.com/posts/healthtech-ai-hub_ai-uobhealthtechaihub-aihackathon-activity-7309941056320507904-U8QH?utm_source=share&utm_medium=member_desktop&rcm=ACoAACxJjQYBGdIzOwy05JLYrmDssykd68M9gYQ)

The challenge focused on using AI techniques to analyze wearable data from athletes and identify patterns that could optimize health and performance.

## Repository Structure
The repository contains:

1. **wearable_data_processing.py**: Handles data cleaning, preprocessing, and imputation of missing values
2. **clustering.py**: Performs clustering analysis on the processed data to identify athlete states

## Features

### Data Processing

- **Data Cleaning and Preprocessing**: Handles missing data and sparse features in the dataset
- **Emotion Processing**: Categorizes emotions into high/low positive/negative groups
- **Time-Series Data Processing**: Splits data chronologically per user (80/20 train/test)
- **Advanced Imputation**:
  - Missing Sleep Heart Rate values filled using HRV
  - Missing HRV values filled using Sleep Heart Rate
  - Missing Steps values imputed using regression and distribution-based methods
  - Missing emotional data imputed using exponential weighted moving average

### Clustering Analysis

- **K-Means Clustering**: Identifies distinct athlete states from processed data
- **Optimal Cluster Selection**: Uses the elbow method to determine the optimal number of clusters
- **PCA Visualization**: Projects high-dimensional data into 2D space for visualization
- **Feature Importance Analysis**: Identifies which features contribute most to cluster differentiation
- **Cluster Interpretation**: Labels clusters with meaningful names (Overloaded, Effective, Underloaded, etc.)

## Requirements

- Python 3.8+
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

### Data Processing

1. Place the raw data file (`wearable_users_raw_data.csv`) in the project directory
2. Run the data processing script:

```bash
python wearable_data_processing.py
```

3. The script will generate:
   - `Train_80_Data.csv`: 80% training data split by user and time
   - `Test_20_Data.csv`: 20% testing data split by user and time
   - `training_imputed.csv`: Training data with imputed values
   - `test_imputed.csv`: Test data with imputed values
   - Visualization figures saved in the `./output` directory

### Clustering Analysis

After processing the data, run the clustering script:

```bash
python clustering.py
```

This will:
1. Load the processed training data
2. Apply K-means clustering to identify athlete states
3. Generate visualizations of the clusters and feature importance
4. Label the clusters based on the characteristics of each group

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

## Clustering Methodology

1. **Data Scaling**:
   - Features scaled per individual to capture relative changes
   - Demographic features (age, gender) scaled globally

2. **K-Means Clustering**:
   - Applied to scaled data to identify distinct athlete states
   - Optimal number of clusters determined using the elbow method

3. **Cluster Interpretation**:
   - Clusters labeled based on emotional and performance characteristics:
     - Overloaded: High negative emotions, lower performance
     - Effective: Higher positive emotions, better performance
     - Underloaded: Lower emotional activation, moderate performance
     - Need More User Input: Sparse data patterns

4. **Dimensionality Reduction**:
   - PCA applied to visualize clusters in 2D space

## Completed Subtasks

This project addresses the core sub-challenges of the Athlete State Clustering project:

1. **Data Exploration and Preprocessing**: The script provides comprehensive cleaning and handling of missing values across all critical features.

2. **Clustering and Cluster Evaluation**: Applied K-means clustering to identify optimal number of clusters, evaluated using the elbow method, and interpreted clusters with feature importance analysis.

3. **Preparation for Label Assignment**: The clean datasets can be used to develop supervised models for assigning labels to new users.

4. **Support for Performance Analysis**: The processed data preserves the relationship between self-reported performance and other variables, making it suitable for developing predictive models.

## Output Files

- `Train_80_Data.csv`: Raw training split (80% of data)
- `Test_20_Data.csv`: Raw testing split (20% of data)
- `training_imputed.csv`: Fully processed training data with all imputed values
- `test_imputed.csv`: Fully processed test data with all imputed values

## Visualizations

The project generates several visualizations in the `./output` directory:

### Data Processing Visualizations
- Missing values heatmaps
- Steps distribution before and after imputation
- Example of nutrition data imputation for a sample user
- Correlation matrices

### Clustering Visualizations
- Elbow curve for optimal cluster selection
- PCA projection of clusters
- Emotion features by cluster
- Performance scores by cluster
- Feature importance in clustering

## Contributors

- Amy Duguid - acd446@student.bham.ac.uk
- Dominika Gorgosz - dag449@student.bham.ac.uk
- Eve Day - exd120@student.bham.ac.uk
- Jingying Liang - jxl1880@student.bham.ac.uk
- Scott Brooks - S.Brooks.2@warwick.ac.uk
