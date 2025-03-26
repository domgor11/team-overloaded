"""
Wearable Data Preprocessing Pipeline

This script preprocesses wearable device data by:
1. Cleaning and transforming raw data
2. Creating emotion indicators
3. Train-test splitting based on chronological order
4. Imputing missing values for physiological metrics
5. Processing emotion-related data
6. Visualizing data quality and preprocessing results
"""

# Standard library imports
import os
from datetime import timedelta

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Set plot style
plt.style.use("seaborn-v0_8")
sns.set_palette('viridis')

# Configure larger font size for plots
plt.rcParams.update({'font.size': 12})

# Set random seed for reproducibility
np.random.seed(11)

# Global constants
EMOTION_CATEGORIES = {
    'high_pos': ['Excited', 'Motivated', 'Powerful', 'Lively', 'Proud', 'Hopeful', 'Confident', 'Determined', 'Optimistic'],
    'low_pos': ['Peaceful', 'Content', 'Relieved', 'Valued', 'Loving', 'Helpful', 'Happy'],
    'high_neg': ['Annoyed', 'Irritated', 'Overwhelmed', 'Nervous', 'Anxious', 'Afraid', 'Worried', 'Angry'],
    'low_neg': ['Weary', 'Bored', 'Lonely', 'Disappointed', 'Confused', 'Embarrassed', 'Sad']
}

# ======================================================
# VISUALIZATION FUNCTIONS
# ======================================================

def visualize_missing_values(df, title="Missing Values Heatmap", figsize=(14, 10), save_path=None):
    """
    Create a heatmap showing missing values in the dataset.
    
    Args:
        df: DataFrame to visualize
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
        
    Returns:
        Missing values percentage information
    """
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    }).sort_values(by='Percentage', ascending=False)
    
    # Filter columns with missing values
    missing_data = missing_data[missing_data['Missing Values'] > 0]
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Create the heatmap - we use a subset of 100 rows to make the visualization clearer
    sample_size = min(100, len(df))
    plt.subplot(2, 1, 1)
    sns.heatmap(
        df.sample(sample_size).isnull(), 
        cbar=False, 
        yticklabels=False,
        cmap='viridis'
    )
    plt.title(f"{title} - Sample of {sample_size} Rows")
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    
    # Create the missing values percentage bar plot
    plt.subplot(2, 1, 2)
    if not missing_data.empty:
        sns.barplot(x=missing_data.index, y='Percentage', data=missing_data)
        plt.title('Percentage of Missing Values by Column')
        plt.xticks(rotation=90)
        plt.ylabel('Percentage Missing')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'No missing values!', ha='center', va='center', fontsize=14)
        plt.axis('off')
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return missing_data[missing_data['Percentage'] > 0]

def plot_correlation_matrix(df, columns=None, title="Correlation Matrix", figsize=(10, 8), save_path=None):
    """
    Create a correlation matrix heatmap for selected columns.
    
    Args:
        df: DataFrame to visualize
        columns: List of columns to include (None for all numeric columns)
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
    """
    # Select columns if specified, otherwise use all numeric columns
    if columns:
        data = df[columns]
    else:
        data = df.select_dtypes(include=[np.number])
    
    # Calculate the correlation matrix
    corr_matrix = data.corr(method='pearson')
    
    # Create the figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        square=True, 
        cbar=True,
        vmin=-1, 
        vmax=1
    )
    
    plt.title(title)
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print the correlation values for reference
    print(f"\n{title}:")
    print(corr_matrix)
    
    return corr_matrix

def visualize_imputation_comparison(original_column, imputed_column, user_id_column, column_name, 
                                   max_users=7, figsize=(14, 6), save_path=None):
    """
    Visualize the distribution of original vs. imputed values for a specific column, grouped by user.
    
    Args:
        original_column: Series with original values (with NaNs)
        imputed_column: Series with imputed values (NaNs filled)
        user_id_column: Series with user identifiers
        column_name: Name of the column being visualized
        max_users: Maximum number of users to display
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)
    """
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'User ID': user_id_column,
        'Original': original_column,
        'Imputed': imputed_column
    })
    
    # Add a source indicator (Original or Filled)
    plot_df['Source'] = plot_df.apply(
        lambda row: 'Original' if pd.notnull(row['Original']) else 'Filled', 
        axis=1
    )
    
    # Use imputed values for the actual values to display
    plot_df['Value'] = plot_df['Imputed']
    
    # Sample a subset of users
    sample_users = plot_df['User ID'].unique()[:max_users]
    
    # Filter to selected users
    sample_df = plot_df[plot_df['User ID'].isin(sample_users)]
    
    # Create the visualization
    plt.figure(figsize=figsize)
    
    # Create swarmplot
    sns.swarmplot(
        data=sample_df, 
        x='User ID', 
        y='Value', 
        hue='Source', 
        palette={'Original': 'skyblue', 'Filled': 'lightcoral'}
    )
    
    plt.title(f'{column_name} Distribution by User (Original vs Filled)')
    plt.ylabel(f'{column_name} Value')
    plt.xticks(rotation=30)
    plt.legend(title='Data Source')
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    fill_count = (plot_df['Source'] == 'Filled').sum()
    total_count = len(plot_df)
    fill_percentage = (fill_count / total_count) * 100
    
    print(f"{column_name} Imputation Statistics:")
    print(f"Total values: {total_count}")
    print(f"Imputed values: {fill_count} ({fill_percentage:.2f}%)")
    print(f"Original values: {total_count - fill_count} ({100 - fill_percentage:.2f}%)")

# ======================================================
# DATA LOADING AND INITIAL PROCESSING
# ======================================================

def load_and_clean_data(file_path):
    """
    Load raw wearable data and perform initial cleaning.
    
    Args:
        file_path: Path to the CSV file containing raw data
        
    Returns:
        Cleaned pandas DataFrame
    """
    # Load data
    wearable_users_raw_data = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    cols_to_drop = ["Height", "Weight", "Actions", "With", "Reflections", 
                    "Phone", "Glucose", "Callories Burn", "Calories Intake"]
    wearable_users_raw_data = wearable_users_raw_data.drop(columns=cols_to_drop)
    
    return wearable_users_raw_data

def create_emotion_indicators(df):
    """
    Create binary indicators for emotion groups.
    
    Args:
        df: DataFrame with emotion columns
        
    Returns:
        DataFrame with binary emotion indicators
    """
    wearables_merged = df.copy()
    
    # Create binary indicators for each emotion group based on whether any value is > 0
    for emotion_category, emotion_list in EMOTION_CATEGORIES.items():
        wearables_merged[emotion_category] = df[emotion_list].gt(0).any(axis=1).astype(int)
    
    # Drop the original emotion columns
    all_emotions = []
    for emotions in EMOTION_CATEGORIES.values():
        all_emotions.extend(emotions)
    
    wearables_merged = wearables_merged.drop(columns=all_emotions)
    
    # Create a Neutral column: if none of the four categories is 1, then Neutral is 1; otherwise 0
    emotion_categories = list(EMOTION_CATEGORIES.keys())
    wearables_merged['Neutral'] = (wearables_merged[emotion_categories].sum(axis=1) == 0).astype(int)
    
    return wearables_merged

def drop_empty_rows(df):
    """
    Drop rows with empty data based on specific conditions.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame with empty rows removed
    """
    # Define columns to check
    tiredness_to_performance_cols = ['Tiredness', 'Calm', 'Nutrition', 'Hydration', 'Performance', 'Concentrate']
    
    # Check for rows where these values are all 50.0 or NaN
    group_6_emotions_50 = df[tiredness_to_performance_cols].apply(lambda x: np.isclose(x, 50.0)).all(axis=1)
    group_6_emotions_na = df[tiredness_to_performance_cols].isna().all(axis=1)
    
    # Check for rows where emotion indicators are all 0
    other_emotions_0 = df[['high_pos', 'low_pos', 'high_neg', 'low_neg']].apply(lambda x: np.isclose(x, 0.0)).all(axis=1)
    
    # Combine conditions and drop rows
    mask_combined = (group_6_emotions_50 | group_6_emotions_na) & other_emotions_0
    print(f"Rows to drop: {mask_combined.sum()}")
    
    return df[~mask_combined]

# ======================================================
# TRAIN-TEST SPLITTING
# ======================================================

def create_user_train_test_split(df, train_ratio=0.8):
    """
    Creates training and test datasets from wearable data.
    For each user, takes the first 80% of chronologically sorted data for training
    and the remaining 20% for testing.
    
    Args:
        df: DataFrame with 'User ID' and 'Date/Time' columns
        train_ratio: Proportion of data to use for training (default: 0.8)
        
    Returns:
        train_df, test_df: Training and test DataFrames
    """
    # Ensure data is properly formatted for splitting
    processed_data = df.copy()
    
    # Convert Date/Time to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(processed_data['Date/Time']):
        processed_data['Date/Time'] = pd.to_datetime(processed_data['Date/Time'])
    
    # Sort data by User ID and Date/Time (ascending)
    processed_data = processed_data.sort_values(by=['User ID', 'Date/Time'])
    
    # Initialize empty lists to collect each user's data splits
    train_dfs = []
    test_dfs = []
    
    # Get unique user IDs
    unique_users = processed_data['User ID'].unique()
    
    # Track statistics for reporting
    total_rows = len(processed_data)
    train_count = 0
    test_count = 0
    
    # Process each user individually
    for user_id in unique_users:
        # Get this user's data (already sorted by Date/Time)
        user_data = processed_data[processed_data['User ID'] == user_id].copy()
        
        # Calculate the split point at 80% of this user's data
        split_index = int(len(user_data) * train_ratio)
        
        # Split the user's data
        user_train = user_data.iloc[:split_index]  # First 80%
        user_test = user_data.iloc[split_index:]   # Remaining 20%
        
        # Add to our collections
        train_dfs.append(user_train)
        test_dfs.append(user_test)
        
        # Update counts for reporting
        train_count += len(user_train)
        test_count += len(user_test)
    
    # Combine all users' data into final datasets
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Print summary statistics
    print(f"Data successfully split into training and test sets:")
    print(f"Training set: {train_count} rows ({train_count/total_rows:.1%} of total)")
    print(f"Test set: {test_count} rows ({test_count/total_rows:.1%} of total)")
    
    return train_df, test_df

# ======================================================
# IMPUTATION FUNCTIONS
# ======================================================

def process_wearable_data(df, output_csv=None, min_samples=5):
    """
    Impute missing Sleep Heart Rate and HRV values using linear regression.
    
    Args:
        df: DataFrame with Sleep Heart Rate and HRV columns
        output_csv: Optional path to save processed data
        min_samples: Minimum samples needed for regression
        
    Returns:
        DataFrame with imputed values
    """
    print("=" * 50)
    print("IMPUTING SLEEP HEART RATE AND HRV VALUES")
    print("=" * 50)
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    print(f"Total rows: {len(result_df)}")
    print(f"Missing HRV values: {result_df['HRV'].isna().sum()}")
    print(f"Missing Sleep Heart Rate values: {result_df['Sleep Heart Rate'].isna().sum()}")
    
    # Impute Sleep Heart Rate using HRV
    filled_shr_values = {}
    users_with_shr_models = 0
    
    for user_id, user_data in result_df.groupby('User ID'):
        # Find rows with both values available
        complete_data = user_data.dropna(subset=['HRV', 'Sleep Heart Rate'])
        
        # Find rows where HRV is available but Sleep Heart Rate is missing
        incomplete_data = user_data[user_data['HRV'].notna() & user_data['Sleep Heart Rate'].isna()]
        
        if len(complete_data) >= min_samples and len(incomplete_data) > 0:
            users_with_shr_models += 1
            
            # Train a linear regression model
            model = LinearRegression()
            X = complete_data['HRV'].values.reshape(-1, 1)
            y = complete_data['Sleep Heart Rate'].values
            model.fit(X, y)
            
            # Predict missing Sleep Heart Rate values
            incomplete_X = incomplete_data['HRV'].values.reshape(-1, 1)
            predicted_y = model.predict(incomplete_X)
            
            # Store the predictions
            for i, idx in enumerate(incomplete_data.index):
                filled_shr_values[idx] = predicted_y[i]
    
    # Fill the values
    for idx, value in filled_shr_values.items():
        result_df.loc[idx, 'Sleep Heart Rate'] = value
    
    # Impute HRV using Sleep Heart Rate
    filled_hrv_values = {}
    users_with_hrv_models = 0
    
    for user_id, user_data in result_df.groupby('User ID'):
        # Find rows with both values available
        complete_data = user_data.dropna(subset=['HRV', 'Sleep Heart Rate'])
        
        # Find rows where Sleep Heart Rate is available but HRV is missing
        incomplete_data = user_data[user_data['Sleep Heart Rate'].notna() & user_data['HRV'].isna()]
        
        if len(complete_data) >= min_samples and len(incomplete_data) > 0:
            users_with_hrv_models += 1
            
            # Train a linear regression model
            model = LinearRegression()
            X = complete_data['Sleep Heart Rate'].values.reshape(-1, 1)
            y = complete_data['HRV'].values
            model.fit(X, y)
            
            # Predict missing HRV values
            incomplete_X = incomplete_data['Sleep Heart Rate'].values.reshape(-1, 1)
            predicted_y = model.predict(incomplete_X)
            
            # Store the predictions
            for i, idx in enumerate(incomplete_data.index):
                filled_hrv_values[idx] = predicted_y[i]
    
    # Fill the values
    for idx, value in filled_hrv_values.items():
        result_df.loc[idx, 'HRV'] = value
    
    # Convert to integers
    result_df['Sleep Heart Rate'] = result_df['Sleep Heart Rate'].apply(
        lambda x: int(round(x)) if pd.notna(x) else x
    )
    
    result_df['HRV'] = result_df['HRV'].apply(
        lambda x: int(round(x)) if pd.notna(x) else x
    )
    
    # Save if requested
    if output_csv:
        result_df.to_csv(output_csv, index=False)
    
    # Print summary
    print(f"Filled {len(filled_shr_values)} Sleep Heart Rate values")
    print(f"Filled {len(filled_hrv_values)} HRV values")
    print(f"Remaining missing HRV values: {result_df['HRV'].isna().sum()}")
    print(f"Remaining missing Sleep Heart Rate values: {result_df['Sleep Heart Rate'].isna().sum()}")
    
    return result_df

def impute_steps(df):
    """
    Impute missing step values using regression and interpolation.
    
    Args:
        df: DataFrame with Steps column
        
    Returns:
        DataFrame with imputed steps
    """
    print("=" * 50)
    print("IMPUTING MISSING STEPS VALUES")
    print("=" * 50)
    
    # Prepare data
    df_imputed = df.copy()
    df_imputed['Date/Time'] = pd.to_datetime(df_imputed['Date/Time'])
    df_imputed['Date'] = df_imputed['Date/Time'].dt.date
    df_imputed['Weekday'] = df_imputed['Date/Time'].dt.day_name()
    df_imputed['Weekend'] = df_imputed['Weekday'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Get distribution statistics from original data
    steps_mean = df_imputed['Steps'].mean()
    steps_std = df_imputed['Steps'].std()
    steps_min = max(0, df_imputed['Steps'].quantile(0.01))  # Avoid negative steps
    steps_max = df_imputed['Steps'].quantile(0.99)  # Avoid extreme outliers
    
    all_imputations = pd.Series(index=df_imputed.index)
    
    def _apply_distribution_sampling(user_data, missing_mask):
        """Apply distribution-based sampling for users with insufficient data"""
        for i in range(len(user_data)):
            if missing_mask.iloc[i]:
                # First try to sample from the user's own distribution
                user_values = user_data['Steps'].dropna()
                if len(user_values) >= 5:  # If we have enough values from this user
                    # Sample existing value and add small noise for variety
                    base_value = np.random.choice(user_values)
                    noise = np.random.normal(0, steps_std * 0.15)  # Less noise for stability
                    sampled_value = max(steps_min, min(steps_max, base_value + noise))
                    user_data.loc[i, 'Steps_Imputed'] = sampled_value
                else:
                    # Not enough user values, use weekday-based sampling from global data
                    weekend = user_data.loc[i, 'Weekend']
                    if weekend:
                        wknd_mean = df_imputed[df_imputed['Weekend'] == 1]['Steps'].mean()
                        wknd_std = df_imputed[df_imputed['Weekend'] == 1]['Steps'].std() * 0.8
                        sampled_value = np.random.normal(wknd_mean, wknd_std)
                    else:
                        wkday_mean = df_imputed[df_imputed['Weekend'] == 0]['Steps'].mean()
                        wkday_std = df_imputed[df_imputed['Weekend'] == 0]['Steps'].std() * 0.8
                        sampled_value = np.random.normal(wkday_mean, wkday_std)
                    
                    # Keep within reasonable bounds
                    user_data.loc[i, 'Steps_Imputed'] = max(steps_min, min(steps_max, sampled_value))
    
    def impute_steps_with_distribution_preservation(user_data):
        """Impute missing step values for a single user"""
        # Replace 0 values in Steps with NaN
        user_data = user_data.copy()
        user_data.loc[user_data['Steps'] == 0, 'Steps'] = np.nan
        
        # Reset index for safer operations
        user_data = user_data.sort_values('Date/Time').reset_index(drop=False)
        original_index = user_data['index'].copy()
        
        # Add imputation column
        user_data['Steps_Imputed'] = user_data['Steps'].copy()
        
        # Linear interpolation for short gaps
        user_data['Steps_Imputed'] = user_data['Steps_Imputed'].interpolate(method='linear', limit=3)
        
        # Create lag features
        user_data['Steps_Lag1'] = user_data['Steps_Imputed'].shift(1)
        user_data['Steps_Lag7'] = user_data['Steps_Imputed'].shift(7)  # Same day last week
        
        # Identify remaining missing values
        missing_mask = user_data['Steps_Imputed'].isna()
        
        if missing_mask.sum() > 0:
            # If user has enough data for modeling approach
            if user_data['Steps'].notna().sum() >= 10:
                potential_predictors = ['Performance', 'Sleep', 'HRV', 'Tiredness']
                available_predictors = [p for p in potential_predictors if p in user_data.columns]
                
                # Build feature matrix
                predictor_cols = ['Weekend', 'Steps_Lag1', 'Steps_Lag7'] + available_predictors
                predictor_cols = [p for p in predictor_cols if p in user_data.columns]
                
                # Create train data from non-missing values
                train_mask = ~user_data['Steps'].isna()
                
                if train_mask.sum() >= 5:  # Ensure enough samples for training
                    try:
                        # Prepare training data
                        X_train = user_data.loc[train_mask, predictor_cols].copy()
                        y_train = user_data.loc[train_mask, 'Steps'].copy()
                        
                        # Fill missing values in predictors for training
                        for col in predictor_cols:
                            if col in ['Steps_Lag1', 'Steps_Lag7']:
                                X_train.loc[:, col] = X_train[col].fillna(user_data.loc[train_mask, 'Steps'].median())
                            else:
                                X_train.loc[:, col] = X_train[col].fillna(X_train[col].median())
                        
                        # Set up model pipeline
                        scaler = StandardScaler()
                        try:
                            huber = HuberRegressor(epsilon=1.35, max_iter=500, tol=1e-3)
                            pipeline = make_pipeline(scaler, huber)
                            pipeline.fit(X_train, y_train)
                            model_type = "huber"
                        except:
                            # Fallback to Ridge
                            ridge = Ridge(alpha=1.0)
                            pipeline = make_pipeline(scaler, ridge)
                            pipeline.fit(X_train, y_train)
                            model_type = "ridge"
                        
                        # Predict missing values sequentially
                        for i in range(len(user_data)):
                            if missing_mask.iloc[i]:
                                # Create features for this row
                                X_pred = user_data.iloc[[i]][predictor_cols].copy()
                                
                                # Fill missing predictors
                                for col in predictor_cols:
                                    if pd.isna(X_pred[col].iloc[0]):
                                        if col == 'Steps_Lag1' and i > 0:
                                            X_pred.loc[X_pred.index[0], col] = user_data.loc[i-1, 'Steps_Imputed']
                                        elif col == 'Steps_Lag7' and i >= 7:
                                            X_pred.loc[X_pred.index[0], col] = user_data.loc[i-7, 'Steps_Imputed']
                                        else:
                                            X_pred.loc[X_pred.index[0], col] = user_data[col].median()
                                
                                # Predict and add noise
                                try:
                                    prediction = pipeline.predict(X_pred)[0]
                                    noise_factor = 0.2 if model_type == "ridge" else 0.25
                                    noise = np.random.normal(0, steps_std * noise_factor)
                                    prediction_with_noise = max(steps_min, min(steps_max, prediction + noise))
                                    user_data.loc[i, 'Steps_Imputed'] = prediction_with_noise
                                    
                                    # Update lag features for subsequent predictions
                                    if i+1 < len(user_data):
                                        user_data.loc[i+1, 'Steps_Lag1'] = prediction_with_noise
                                    if i+7 < len(user_data):
                                        user_data.loc[i+7, 'Steps_Lag7'] = prediction_with_noise
                                except:
                                    # Fallback if prediction fails
                                    user_values = user_data['Steps'].dropna()
                                    if len(user_values) > 0:
                                        user_data.loc[i, 'Steps_Imputed'] = np.random.choice(user_values)
                                    else:
                                        user_data.loc[i, 'Steps_Imputed'] = np.random.normal(steps_mean, steps_std)
                    except Exception as e:
                        # If all modeling approaches fail, use distribution-based imputation
                        _apply_distribution_sampling(user_data, missing_mask)
                else:
                    # Not enough training data
                    _apply_distribution_sampling(user_data, missing_mask)
            else:
                # Too little data overall
                _apply_distribution_sampling(user_data, missing_mask)
        
        # Ensure values are reasonable and convert to integers
        user_data['Steps_Imputed'] = user_data['Steps_Imputed'].apply(lambda x: max(steps_min, min(steps_max, x)))
        user_data['Steps_Imputed'] = user_data['Steps_Imputed'].round().astype('Int64')
        
        # Return with original index
        return pd.DataFrame({'Steps_Imputed': user_data['Steps_Imputed'].values}, index=original_index)
    
    # Process each user
    print("Starting steps imputation process...")
    unique_users = df_imputed['User ID'].unique()
    total_users = len(unique_users)
    
    for idx, user_id in enumerate(unique_users):
        if idx % 50 == 0:  # Show progress updates
            print(f"Processing user {idx+1}/{total_users}...")
        
        user_data = df_imputed[df_imputed['User ID'] == user_id].copy()
        imputed_result = impute_steps_with_distribution_preservation(user_data)
        
        # Update the results Series using the original indices
        all_imputations.loc[imputed_result.index] = imputed_result['Steps_Imputed']
    
    print("Steps imputation completed.")
    
    # Update the main dataframe
    df_imputed['Steps_Imputed'] = all_imputations
    df_imputed['Steps_Final'] = df_imputed['Steps'].fillna(df_imputed['Steps_Imputed'])
    df_imputed['Steps_Final'] = df_imputed['Steps_Final'].round().astype('Int64')
    
    # Clean up and return
    result_df = df_imputed.drop(columns=['Steps', 'Steps_Imputed'], errors='ignore')
    result_df = result_df.rename(columns={'Steps_Final': 'Steps'})
    
    print(f"Filled {df_imputed['Steps'].isna().sum()} missing steps values")
    
    return result_df

def standardize_datetime_format(df):
    """
    Standardize the Date/Time column format to ISO 8601 with Z suffix.
    
    Args:
        df: DataFrame with Date/Time column
        
    Returns:
        DataFrame with standardized Date/Time
    """
    result_df = df.copy()
    
    # Ensure the Date/Time column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result_df['Date/Time']):
        result_df['Date/Time'] = pd.to_datetime(result_df['Date/Time'])
    
    # Convert to ISO format with Z suffix
    result_df['Date/Time'] = result_df['Date/Time'].apply(
        lambda dt: dt.isoformat().replace('+00:00', 'Z') if pd.notnull(dt) else dt
    )
    
    # Standardize the format for consistency
    def standardize_format(dt_str):
        if pd.isna(dt_str):
            return dt_str
        
        # If it already has a Z at the end
        if isinstance(dt_str, str) and dt_str.endswith('Z'):
            # Split by the decimal point
            parts = dt_str.split('.')
            if len(parts) > 1:
                # Take base part and add 3 decimal places + Z
                base = parts[0]
                decimal_part = parts[1].rstrip('Z')
                # Ensure exactly 3 decimal places
                if len(decimal_part) > 3:
                    decimal_part = decimal_part[:3]
                elif len(decimal_part) < 3:
                    decimal_part = decimal_part.ljust(3, '0')
                return f"{base}.{decimal_part}Z"
            else:
                # No decimal part present
                return f"{dt_str[:-1]}.000Z"
        return dt_str
    
    result_df['Date/Time'] = result_df['Date/Time'].apply(standardize_format)
    
    return result_df

def impute_physiological_metrics(df, visualize=True):
    """
    Impute missing values for Sleep, HRV, and Sleep Heart Rate.
    
    Args:
        df: DataFrame with physiological metrics
        visualize: Whether to visualize the imputation results
        
    Returns:
        DataFrame with imputed physiological metrics
    """
    print("=" * 50)
    print("IMPUTING PHYSIOLOGICAL METRICS")
    print("=" * 50)
    
    # Calculate global statistics for fallback
    global_stats = {}
    for col in ['Sleep Heart Rate', 'HRV', 'Sleep']:
        if df[col].notnull().any():
            global_stats[col] = int(round(df[col].median()))
        else:
            # Reasonable defaults
            if col == 'Sleep Heart Rate':
                global_stats[col] = 65
            elif col == 'HRV':
                global_stats[col] = 50
            elif col == 'Sleep':
                global_stats[col] = 7
    
    print(f"Global fallback values: {global_stats}")
    
    def fill_column_with_user_probs(df_input, column):
        """Fill missing values based on user's value distribution"""
        df_col = df_input.copy()
        df_col['Date'] = pd.to_datetime(df_col['Date/Time']).dt.date
        
        # Calculate each user's value distribution
        user_probs = {}
        for user_id, user_data in df_col.groupby('User ID'):
            non_null_values = user_data[column].dropna()
            if len(non_null_values) > 0:
                value_counts = non_null_values.value_counts(normalize=True)
                user_probs[user_id] = value_counts
        
        # Cache fill values
        user_date_fill_values = {}
        
        def fill_group(group):
            user_id = group['User ID'].iloc[0]
            date = group['Date'].iloc[0]
            
            # If there are already non-null values, reuse them
            existing_values = group[column].dropna().unique()
            if len(existing_values) > 0:
                fill_value = existing_values[0]
            else:
                # If a fill value has already been sampled, reuse it
                if (user_id, date) in user_date_fill_values:
                    fill_value = user_date_fill_values[(user_id, date)]
                else:
                    # Sample according to the user's probability distribution
                    probs = user_probs.get(user_id, None)
                    if probs is None:
                        fill_value = global_stats[column]
                    else:
                        fill_value = np.random.choice(probs.index, p=probs.values)
                    user_date_fill_values[(user_id, date)] = fill_value
            
            # Fill missing values
            group[column] = group[column].fillna(fill_value)
            return group
        
        # Apply the fill logic
        df_col = df_col.groupby(['User ID', 'Date'], group_keys=False).apply(fill_group)
        
        # Drop the temporary Date column
        df_col = df_col.drop(columns=['Date'])
        
        return df_col
    
    # Process each column
    print("Imputing Sleep values...")
    df_sleep = df[['User ID', 'Date/Time', 'Sleep']]
    df_sleep_filled = fill_column_with_user_probs(df_sleep, "Sleep")
    
    print("Imputing HRV values...")
    df_hrv = df[['User ID', 'Date/Time', 'HRV']]
    df_hrv_filled = fill_column_with_user_probs(df_hrv, "HRV")
    
    print("Imputing Sleep Heart Rate values...")
    df_shr = df[['User ID', 'Date/Time', 'Sleep Heart Rate']]
    df_shr_filled = fill_column_with_user_probs(df_shr, "Sleep Heart Rate")
    
    # Visualize Sleep Heart Rate imputation if requested
    if visualize:
        visualize_imputation_comparison(
            df_shr['Sleep Heart Rate'], 
            df_shr_filled['Sleep Heart Rate'],
            df_shr['User ID'],
            "Sleep Heart Rate",
            save_path="sleep_heart_rate_imputation.png"
        )
    
    # Check for remaining missing values
    remaining_missing = df_shr_filled['Sleep Heart Rate'].isna().sum()
    if remaining_missing > 0:
        print(f"WARNING: {remaining_missing} Sleep Heart Rate values still missing after imputation")
        # Apply final global fallback
        df_shr_filled['Sleep Heart Rate'] = df_shr_filled['Sleep Heart Rate'].fillna(global_stats['Sleep Heart Rate'])
    
    # Create result DataFrame
    result_df = df.copy()
    result_df['Sleep'] = df_sleep_filled['Sleep'].values
    result_df['HRV'] = df_hrv_filled['HRV'].values
    result_df['Sleep Heart Rate'] = df_shr_filled['Sleep Heart Rate'].values
    
    print("Physiological metrics imputation complete.")
    return result_df

def impute_emotions(df):
    """
    Impute missing emotion-related values using EWMA.
    
    Args:
        df: DataFrame with emotion columns
        
    Returns:
        DataFrame with imputed emotion values
    """
    print("=" * 50)
    print("IMPUTING EMOTION VALUES")
    print("=" * 50)
    
    # Prepare data
    processed_data = df.copy()
    processed_data['Time'] = processed_data['Date/Time'].str[11:19]  # Extract time portion
    processed_data = processed_data.sort_values(by=['User ID', 'Time'])
    
    # Define emotion columns
    emotion_columns = ['Tiredness', 'Calm', 'Nutrition', 'Hydration', 'Performance']
    emotion_columns_imputed = [f"{col}_imputed" for col in emotion_columns]
    
    # Initialize imputed columns
    processed_data[emotion_columns_imputed] = np.nan
    
    # Use EWMA for imputation
    processed_data[emotion_columns_imputed] = processed_data.groupby('User ID')[emotion_columns].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )
    
    # Fill remaining NaNs with forward fill
    processed_data[emotion_columns_imputed] = processed_data[emotion_columns].fillna(method='ffill')
    
    # Replace original columns with imputed versions
    result_df = processed_data.drop(columns=emotion_columns)
    
    # Rename imputed columns to original names
    rename_mapping = {f"{col}_imputed": col for col in emotion_columns}
    result_df = result_df.rename(columns=rename_mapping)
    
    # Drop the time column
    result_df = result_df.drop(columns=['Time'])
    
    print("Emotion imputation complete.")
    return result_df

def encode_gender(df):
    """
    Encode gender column (man=0, woman=1).
    
    Args:
        df: DataFrame with Gender column
        
    Returns:
        DataFrame with encoded gender
    """
    result_df = df.copy()
    result_df['Gender_encode'] = result_df['Gender'].map({'man': 0, 'woman': 1})
    result_df = result_df.drop(columns=['Gender'])
    
    return result_df

# ======================================================
# MAIN EXECUTION PIPELINE
# ======================================================

def preprocess_wearable_data(input_file, train_csv, test_csv, output_dir="./"):
    """
    Complete preprocessing pipeline for wearable data.
    
    Args:
        input_file: Path to input CSV file
        train_csv: Output path for training data
        test_csv: Output path for test data
        output_dir: Directory to save output files and visualizations
    
    Returns:
        train_df, test_df: Processed training and test DataFrames
    """
    print("Starting wearable data preprocessing pipeline...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load and clean data
    raw_data = load_and_clean_data(input_file)
    print(f"Loaded raw data with {len(raw_data)} rows and {len(raw_data.columns)} columns")
    
    # Step 1.5: Visualize missing values in raw data
    print("\nVisualizing missing values in raw data:")
    missing_data_raw = visualize_missing_values(
        raw_data, 
        title="Missing Values in Raw Data",
        save_path=os.path.join(output_dir, "missing_values_raw.png")
    )
    
    # Step 2: Create emotion indicators
    data_with_emotions = create_emotion_indicators(raw_data)
    print("Created emotion indicators")
    
    # Step 3: Drop empty rows
    cleaned_data = drop_empty_rows(data_with_emotions)
    print(f"Cleaned data has {len(cleaned_data)} rows")
    
    # Step 4: Convert datetime
    cleaned_data['Date/Time'] = pd.to_datetime(cleaned_data['Date/Time'])
    
    # Step 5: Train-test splitting
    train_data, test_data = create_user_train_test_split(cleaned_data)
    
    # Save raw train/test splits
    train_data.to_csv(os.path.join(output_dir, train_csv), index=False)
    test_data.to_csv(os.path.join(output_dir, test_csv), index=False)
    print(f"Saved raw train/test splits to {train_csv} and {test_csv}")
    
    # Process training data
    print("\nProcessing training data...")
    
    # Step 6: Visualize training data missing values
    print("\nVisualizing missing values in training data:")
    missing_data_train = visualize_missing_values(
        train_data, 
        title="Missing Values in Training Data",
        save_path=os.path.join(output_dir, "missing_values_train.png")
    )
    
    # Step 7: Impute Sleep Heart Rate and HRV
    train_processed = process_wearable_data(
        train_data, 
        output_csv=os.path.join(output_dir, "training_hrv_shr_imputed.csv")
    )
    
    # Step 8: Plot correlation matrix for Concentrate and Performance
    print("\nVisualizing correlation between Concentrate and Performance:")
    plot_correlation_matrix(
        train_processed, 
        columns=['Concentrate', 'Performance'],
        title="Correlation: Concentrate vs Performance",
        save_path=os.path.join(output_dir, "concentrate_performance_correlation.png")
    )
    
    # Step 9: Impute Steps
    train_processed = impute_steps(train_processed)
    
    # Step 10: Standardize DateTime format
    train_processed = standardize_datetime_format(train_processed)
    
    # Step 11: Impute physiological metrics
    train_processed = impute_physiological_metrics(train_processed, visualize=True)
    
    # Step 12: Impute emotions
    train_processed = impute_emotions(train_processed)
    
    # Step 13: Visualize final missing values in processed training data
    print("\nVisualizing missing values in processed training data:")
    missing_data_train_processed = visualize_missing_values(
        train_processed, 
        title="Missing Values in Processed Training Data",
        save_path=os.path.join(output_dir, "missing_values_train_processed.png")
    )
    
    # Step 14: Encode gender
    train_processed = encode_gender(train_processed)
    
    # Save final training data
    train_processed.to_csv(os.path.join(output_dir, "training_imputed.csv"), index=False)
    print("Saved processed training data to training_imputed.csv")
    
    # Process test data - similar steps
    print("\nProcessing test data...")
    
    # Repeat steps 6-14 for test data
    test_processed = process_wearable_data(test_data)
    test_processed = impute_steps(test_processed)
    test_processed = standardize_datetime_format(test_processed)
    test_processed = impute_physiological_metrics(test_processed, visualize=True)
    test_processed = impute_emotions(test_processed)
    test_processed = encode_gender(test_processed)
    
    # Save final test data
    test_processed.to_csv(os.path.join(output_dir, "test_imputed.csv"), index=False)
    print("Saved processed test data to test_imputed.csv")
    
    print("\nPreprocessing pipeline complete!")
    return train_processed, test_processed

# Execute the pipeline if run directly
if __name__ == "__main__":
    input_file = 'wearable_users_raw_data.csv'
    train_csv = "Train_80_Data.csv"
    test_csv = "Test_20_Data.csv"
    output_dir = "./outputs"
    
    train_df, test_df = preprocess_wearable_data(input_file, train_csv, test_csv, output_dir)
