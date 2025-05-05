# EDA across the panel data set that focuses on Strokes Gained stats as feature variables
# Summary Stats
# Histogram Matrix
# Correlation Matrix Heatmap
# VIF Calcs
# Scatter Plot Relationships

# Load libraries and data

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

#Load the dataset
df = pd.read_csv("adjusted_panel_model_data_with_time.csv")


###

# Sumary Statistics

# Summary Stats
df = df.rename(columns={'Money_Per_Event_Adjusted': 'Winnings'})

# === Summary Statistics
summary = df[['Winnings', 'SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'Events']].describe().T
summary['Skewness'] = df[['Winnings', 'SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'Events']].skew()
summary['Kurtosis'] = df[['Winnings', 'SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT','Events']].kurtosis()
print(summary.round(2))

###

# histograms 

df[['Winnings', 'SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT']].hist(bins=20, figsize=(10, 8))
plt.suptitle("Histograms of SG Metrics and Log Winnings")
plt.tight_layout()
plt.show()

###

# correlation matrix heatmap

corr_matrix = df[['Winnings', 'SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

###

# Vif scores less events vars

X_vif = df[['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'TIME']]
X_vif = sm.add_constant(X_vif)
vif_data = pd.DataFrame()
vif_data['Variable'] = X_vif.columns
vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(vif_data)

###

# Scatter plots visualized - pair plots -- do not use in paper


sns.pairplot(df, vars=['Winnings', 'SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT'],
             kind='reg', plot_kws={'line_kws': {'color': 'red'}, 'scatter_kws': {'alpha': 0.5}})
plt.suptitle("Scatter Plots of SG Metrics vs. Log Earnings", y=1.02)
plt.tight_layout()
plt.show()

###

# simple scatter, not pair plot


# Define predictors to plot against Winnings
features = ['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'LN_EVENTS', 'TIME']

# Create scatter plots
for feature in features:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[feature], df['Winnings'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Winnings')
    plt.title(f'Winnings vs. {feature}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

###

# I dont like the above verson, so went back to the drawing board to keep it simple. 

import pandas as pd
import matplotlib.pyplot as plt

# Define predictors
features = ['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT']

# Set up subplot grid
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
axes = axes.flatten()

# Plot each scatterplot
for i, feature in enumerate(features):
    axes[i].scatter(df[feature], df['Winnings'], alpha=0.5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Winnings')
    axes[i].set_title(f'WINNINGS vs. {feature}')
    axes[i].grid(True)

# Clean layout
plt.tight_layout()
plt.suptitle('Scatter Plots of WINNINGS vs. Predictors', y=1.03, fontsize=16)
plt.show()
