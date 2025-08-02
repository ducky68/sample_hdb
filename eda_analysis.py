import pandas as pd
"""
Performs exploratory data analysis (EDA) on an HDB resale dataset.

This script executes the following steps:
1. Loads the dataset from a specified CSV file.
2. Displays basic information about the dataset, including shape, column data types, and missing values.
3. Provides descriptive statistics for numerical and categorical features.
4. Analyzes the target variable ('resale_price') distribution.
5. Computes and visualizes the correlation matrix for numerical features.
6. Plots distributions and boxplots for numerical features to detect outliers.
7. Visualizes relationships between numerical features and the target variable using scatter plots.
8. Examines the relationship between categorical features (with fewer than 20 unique values) and the target variable using boxplots.

Note:
- Plots are displayed using matplotlib and seaborn for visual inspection.
- Adjust the file path as needed to point to the correct dataset location.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# Load the dataset
file_path = 'datasets/train.csv'  # Change to your file if needed
df = pd.read_csv(file_path)

# 1. Basic Info
print("\n--- Shape of the dataset ---")
print(df.shape)

print("\n--- Columns and Data Types ---")
print(df.dtypes)

print("\n--- Missing Values ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 2. Descriptive Statistics
print("\n--- Numerical Features Summary ---")
print(df.describe())

print("\n--- Categorical Features Summary ---")
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts().head(10))

# 3. Target Variable Analysis
if 'resale_price' in df.columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df['resale_price'], kde=True)
    plt.title('Distribution of Resale Price')
    plt.xlabel('Resale Price')
    plt.show()

# 4. Correlation Analysis
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# 5. Feature Distributions
for col in num_cols:
    if col != 'resale_price':
        plt.figure(figsize=(6, 3))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()

# 6. Boxplots for Outlier Detection
for col in num_cols:
    if col != 'resale_price':
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()

# 7. Relationship with Target
if 'resale_price' in df.columns:
    for col in num_cols:
        if col != 'resale_price':
            plt.figure(figsize=(6, 4))
            sns.scatterplot(x=df[col], y=df['resale_price'])
            plt.title(f'{col} vs Resale Price')
            plt.xlabel(col)
            plt.ylabel('Resale Price')
            plt.show()

# 8. Categorical Features vs Target
if 'resale_price' in df.columns:
    for col in cat_cols:
        if df[col].nunique() < 20:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[col], y=df['resale_price'])
            plt.title(f'{col} vs Resale Price')
            plt.xticks(rotation=45)
            plt.show()

print("\nEDA complete. Please review the plots and outputs above for insights.")
