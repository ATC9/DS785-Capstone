#This script performs EDA against the panel data set thath includes traditional golf statistics

#Load and Prepare DAta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data
df = pd.read_csv('adjusted_traditional_model_data.csv')

# Rename columns for convenience
df = df.rename(columns={
    'Money_Per_Event_Adjusted': 'WINNINGS',
    'Driving_Distance': 'DD',
    'Driving_Accuracy': 'DA',
    'Greens_In_Regulation': 'GIR',
    'Sand_Saves': 'SS',
    'Putts_Per_GIR': 'PPGIR',
    'Events': 'EVENTS',
    'ShortGam': 'SHORTGAM'
})

# Create additional features
df['EVENTS^2'] = df['EVENTS'] ** 2
df['TIME'] = df['Season'] - 2004 + 1

###

#Summary Stats, ex. time

eda_vars = ['WINNINGS', 'DD', 'DA', 'GIR', 'SS', 'PPGIR', 'EVENTS', 'EVENTS^2', 'SHORTGAM']
summary = df[eda_vars].describe().T
summary['skew'] = df[eda_vars].skew()
summary['kurtosis'] = df[eda_vars].kurtosis()
print(summary[['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurtosis']])


###

#Correlation Matrix

correlation_matrix = df[eda_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

###

#VIF Calcs - remove events, not needed here

# VIF for all predictors (excluding WINNINGS)
X = df[['DD', 'DA', 'GIR', 'SS', 'PPGIR', 'SHORTGAM']]
X = sm.add_constant(X)

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif[vif['Variable'] != 'const'])


###

#Histogram Matrix

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(eda_vars):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f'Histogram of {col}')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Frequency')

for j in range(len(eda_vars), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

###

#Scatter plots

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(eda_vars[1:]):  # Exclude WINNINGS vs WINNINGS
    sns.scatterplot(x=df[col], y=df['WINNINGS'], ax=axes[i])
    sns.regplot(x=df[col], y=df['WINNINGS'], scatter=False, ax=axes[i], color='red')
    axes[i].set_title(f'WINNINGS vs. {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('WINNINGS')

for j in range(len(eda_vars[1:]), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

###

