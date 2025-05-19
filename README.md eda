# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map species to their names
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_map)

# Summary statistics
print(df.describe())

# Data visualization
sns.pairplot(df, hue='species')
plt.show()

# Correlation analysis
corr_matrix = df.drop('species', axis=1).corr()
print(corr_matrix)

# Heatmap for correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Box plots for each feature
for feature in df.columns[:-1]:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'Box plot of {feature} by species')
    plt.show()
