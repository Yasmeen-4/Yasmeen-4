import pandas as pd
import numpy as np

# Load the Titanic dataset
titanic_df = pd.read_csv('titanic.csv')

# Display the first few rows of the dataframe
print("First 5 rows of the dataset:")
print(titanic_df.head())

# Get basic information about the dataset
print("\nDataset information:")
titanic_df.info()

# Generate descriptive statistics
print("\nDescriptive statistics:")
print(titanic_df.describe())

# Check for missing values
print("\nMissing values:")
print(titanic_df.isnull().sum())

# Calculate survival rate
print("\nSurvival rate:")
print(titanic_df['Survived'].value_counts(normalize=True))

# Survival rate by gender
print("\nSurvival rate by gender:")
print(titanic_df.groupby('Sex')['Survived'].value_counts(normalize=True))

# Survival rate by passenger class
print("\nSurvival rate by passenger class:")
print(titanic_df.groupby('Pclass')['Survived'].value_counts(normalize=True))

# Age distribution
print("\nAge distribution:")
print(titanic_df['Age'].hist())
