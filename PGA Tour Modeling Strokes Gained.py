# This notebook walks through a modeling process to determine PGA Tour winnings based on strokes gained feature variables
# Progression - OLS, Tuned LAsso/Ridge/Elastic NEt
# Evaluation for final model
# Interaction of TIME with Final model

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
df = pd.read_csv("adjusted_panel_model_data_with_time.csv")

# Create engineered features
df['LN_WINNINGS'] = np.log(df['Money_Per_Event_Adjusted'])
df['LN_EVENTS'] = np.log(df['Events'])
df['LN_EVENTS_SQ'] = df['LN_EVENTS'] ** 2

# Define features and target (unstandardized)
X = df[['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']]
y = df['LN_WINNINGS']

# Add intercept
X = sm.add_constant(X)

# Fit OLS model
ols_model = sm.OLS(y, X).fit()

# Print full results
print(ols_model.summary())

# Optional: Clean coefficient table
ols_results = ols_model.summary2().tables[1]
print(ols_results)

###

# Set up OLS Model


# Unstandardized features
X = df[['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']]
y = df['LN_WINNINGS']

# Add constant and fit with statsmodels
X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()

###

# run and print base model

import statsmodels.api as sm

# Reuse the same unstandardized feature matrix
X = df[['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']]
y = df['LN_WINNINGS']

# Add constant and fit with statsmodels
X = sm.add_constant(X)
ols_model = sm.OLS(y, X).fit()
print(ols_model.summary())

###

# CV OLS for model evaluation

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score

# DO NOT add constant manually here
X_cv = df[['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']]
y_cv = df['LN_WINNINGS']

# cross validated model
cv = KFold(n_splits=5, shuffle=True, random_state=42)
ols_cv_model = LinearRegression()
y_pred_cv = cross_val_predict(ols_cv_model, X_cv, y_cv, cv=cv)

# Evaluate
rmse = mean_squared_error(y_cv, y_pred_cv, squared=False)
r2 = r2_score(y_cv, y_pred_cv)

print(f"Cross-Validated RMSE: {rmse:.4f}")
print(f"Cross-Validated R²: {r2:.4f}")

###

# regularlization and compare coefficients

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load and prepare data
df = pd.read_csv("adjusted_panel_model_data_with_time.csv")
df['LN_WINNINGS'] = np.log(df['Money_Per_Event_Adjusted'])
df['LN_EVENTS'] = np.log(df['Events'])
df['LN_EVENTS_SQ'] = df['LN_EVENTS'] ** 2

# Define features and target
X = df[['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']]
y = df['LN_WINNINGS']
features = X.columns.tolist()

# === LassoCV
lasso_model = make_pipeline(
    StandardScaler(),
    LassoCV(cv=10, random_state=42)
)
lasso_model.fit(X, y)
lasso_coefs = lasso_model.named_steps['lassocv'].coef_

# === RidgeCV
ridge_model = make_pipeline(
    StandardScaler(),
    RidgeCV(cv=10)
)
ridge_model.fit(X, y)
ridge_coefs = ridge_model.named_steps['ridgecv'].coef_

# === ElasticNetCV
enet_model = make_pipeline(
    StandardScaler(),
    ElasticNetCV(cv=10, random_state=42)
)
enet_model.fit(X, y)
enet_coefs = enet_model.named_steps['elasticnetcv'].coef_

# === OLS with statsmodels for coefficients and p-values
import statsmodels.api as sm
X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()
ols_coefs = ols_model.params
ols_pvals = ols_model.pvalues

# === Combine all results into one DataFrame
coef_table = pd.DataFrame({
    'ModelVar': ['Intercept'] + features,
    'Description': [
        'Constant Term for OLS Model',
        'SG Off the Tee',
        'SG Approach the Green',
        'SG Around the Green',
        'SG Putting',
        '# Events Participated Annually',
        '# Events Participated In Annually Squared Term',
        'Time Trend Index'
    ],
    'OLS Coef': np.round(ols_coefs.values, 4),
    'OLS p-value': np.round(ols_pvals.values, 4),
    'Lasso Coef': ['N/A'] + list(np.round(lasso_coefs, 4)),
    'Ridge Coef': ['N/A'] + list(np.round(ridge_coefs, 4)),
    'Elastic Net Coef': ['N/A'] + list(np.round(enet_coefs, 4))
})

print(coef_table)

###

# Model Evaluation

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Define features and target again (unstandardized for performance evaluation)
X = df[['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']]
y = df['LN_WINNINGS']
n = len(y)
k = X.shape[1] + 1  # +1 for intercept

# === OLS Cross-Validated RMSE/R2 (from earlier)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold

ols_model = LinearRegression()
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_ols = cross_val_predict(ols_model, X, y, cv=cv)
rmse_ols = mean_squared_error(y, y_pred_ols, squared=False)
r2_ols = r2_score(y, y_pred_ols)

# AIC & BIC for OLS (fit on full data)
import statsmodels.api as sm
X_sm = sm.add_constant(X)
ols_sm_model = sm.OLS(y, X_sm).fit()
aic_ols = ols_sm_model.aic
bic_ols = ols_sm_model.bic

# === Lasso Performance (already fit earlier)
lasso_rmse = mean_squared_error(y, lasso_model.predict(X), squared=False)
lasso_r2 = r2_score(y, lasso_model.predict(X))
lasso_alpha = lasso_model.named_steps['lassocv'].alpha_
lasso_rss = np.sum((y - lasso_model.predict(X))**2)
aic_lasso = n * np.log(lasso_rss / n) + 2 * k
bic_lasso = n * np.log(lasso_rss / n) + k * np.log(n)

# === Ridge Performance
ridge_rmse = mean_squared_error(y, ridge_model.predict(X), squared=False)
ridge_r2 = r2_score(y, ridge_model.predict(X))
ridge_alpha = ridge_model.named_steps['ridgecv'].alpha_
ridge_rss = np.sum((y - ridge_model.predict(X))**2)
aic_ridge = n * np.log(ridge_rss / n) + 2 * k
bic_ridge = n * np.log(ridge_rss / n) + k * np.log(n)

# === Elastic Net Performance
enet_rmse = mean_squared_error(y, enet_model.predict(X), squared=False)
enet_r2 = r2_score(y, enet_model.predict(X))
enet_alpha = enet_model.named_steps['elasticnetcv'].alpha_
enet_rss = np.sum((y - enet_model.predict(X))**2)
aic_enet = n * np.log(enet_rss / n) + 2 * k
bic_enet = n * np.log(enet_rss / n) + k * np.log(n)

# === Summary Table To Paste In XLS
performance_df = pd.DataFrame([
    {
        'Model': 'OLS (5-Fold Cross Validated)',
        'RMSE': round(rmse_ols, 4),
        'R2': round(r2_ols, 3),
        'AIC': round(aic_ols, 2),
        'BIC': round(bic_ols, 2),
        'Alpha': 'N/A'
    },
    {
        'Model': 'Lasso',
        'RMSE': round(lasso_rmse, 4),
        'R2': round(lasso_r2, 3),
        'AIC': round(aic_lasso, 2),
        'BIC': round(bic_lasso, 2),
        'Alpha': round(lasso_alpha, 3)
    },
    {
        'Model': 'Ridge',
        'RMSE': round(ridge_rmse, 4),
        'R2': round(ridge_r2, 3),
        'AIC': round(aic_ridge, 2),
        'BIC': round(bic_ridge, 2),
        'Alpha': round(ridge_alpha, 3)
    },
    {
        'Model': 'Elastic Net',
        'RMSE': round(enet_rmse, 4),
        'R2': round(enet_r2, 3),
        'AIC': round(aic_enet, 2),
        'BIC': round(bic_enet, 2),
        'Alpha': round(enet_alpha, 3)
    }
])

print(performance_df)

###

# extended ridge model with interaction terms 

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load and prepare data
df = pd.read_csv("adjusted_panel_model_data_with_time.csv")
df['LN_WINNINGS'] = np.log(df['Money_Per_Event_Adjusted'])
df['LN_EVENTS'] = np.log(df['Events'])
df['LN_EVENTS_SQ'] = df['LN_EVENTS'] ** 2

# Create interaction terms
df['SG_OTT_TIME'] = df['SG_OTT'] * df['TIME']
df['SG_APP_TIME'] = df['SG_APP'] * df['TIME']
df['SG_ATG_TIME'] = df['SG_ATG'] * df['TIME']
df['SG_PUT_TIME'] = df['SG_PUT'] * df['TIME']

# Define full feature set with interactions
features = [
    'SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT',
    'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME',
    'SG_OTT_TIME', 'SG_APP_TIME', 'SG_ATG_TIME', 'SG_PUT_TIME'
]
X = df[features]
y = df['LN_WINNINGS']

# Define and fit RidgeCV model
ridge_model = make_pipeline(
    StandardScaler(),
    RidgeCV(cv=10)
)
ridge_model.fit(X, y)

# Extract tuned alpha and coefficients
alpha_selected = ridge_model.named_steps['ridgecv'].alpha_
ridge_coefs = ridge_model.named_steps['ridgecv'].coef_

# Package results
coef_table = pd.DataFrame({
    'ModelVar': features,
    'Description': [
        'SG Off the Tee',
        'SG Approach the Green',
        'SG Around the Green',
        'SG Putting',
        '# Events Participated Annually',
        '# Events Participated In Annually Squared Term',
        'Time Trend Index',
        'SG Off the Tee*TIME',
        'SG Approach the Green*TIME',
        'SG Around the Green*TIME',
        'SG Putting*TIME'
    ],
    'Ridge Coefficient': np.round(ridge_coefs, 4)
})

print(f"Selected alpha: {alpha_selected}")
print(coef_table)


###

# rerun ridge, compare model performance to original ridge without interaction terms

from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Feature engineering just in case
df['LN_WINNINGS'] = np.log(df['Money_Per_Event_Adjusted'])
df['LN_EVENTS'] = np.log(df['Events'])
df['LN_EVENTS_SQ'] = df['LN_EVENTS'] ** 2

# Create interaction terms
df['SG_OTT_TIME'] = df['SG_OTT'] * df['TIME']
df['SG_APP_TIME'] = df['SG_APP'] * df['TIME']
df['SG_ATG_TIME'] = df['SG_ATG'] * df['TIME']
df['SG_PUT_TIME'] = df['SG_PUT'] * df['TIME']

# === Set up datasets for side by side comparison
X_orig = df[['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT', 'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']]
X_ext = df[['SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT',
            'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME',
            'SG_OTT_TIME', 'SG_APP_TIME', 'SG_ATG_TIME', 'SG_PUT_TIME']]
y = df['LN_WINNINGS']
n = len(y)
k_orig = X_orig.shape[1] + 1
k_ext = X_ext.shape[1] + 1

# === Fit Original Ridge
ridge_orig = make_pipeline(StandardScaler(), RidgeCV(cv=10))
ridge_orig.fit(X_orig, y)
y_pred_orig = ridge_orig.predict(X_orig)
rss_orig = np.sum((y - y_pred_orig) ** 2)
rmse_orig = mean_squared_error(y, y_pred_orig, squared=False)
r2_orig = r2_score(y, y_pred_orig)
aic_orig = n * np.log(rss_orig / n) + 2 * k_orig
bic_orig = n * np.log(rss_orig / n) + k_orig * np.log(n)
alpha_orig = ridge_orig.named_steps['ridgecv'].alpha_

# === Fit Ridge with Interaction Terms
ridge_ext = make_pipeline(StandardScaler(), RidgeCV(cv=10))
ridge_ext.fit(X_ext, y)
y_pred_ext = ridge_ext.predict(X_ext)
rss_ext = np.sum((y - y_pred_ext) ** 2)
rmse_ext = mean_squared_error(y, y_pred_ext, squared=False)
r2_ext = r2_score(y, y_pred_ext)
aic_ext = n * np.log(rss_ext / n) + 2 * k_ext
bic_ext = n * np.log(rss_ext / n) + k_ext * np.log(n)
alpha_ext = ridge_ext.named_steps['ridgecv'].alpha_

# === Final Comparison Table
ridge_compare = pd.DataFrame([
    {
        'Model': 'Original Ridge',
        'RMSE': round(rmse_orig, 4),
        'R-squared': round(r2_orig, 3),
        'AIC': round(aic_orig, 2),
        'BIC': round(bic_orig, 2),
        'Optimal Alpha': alpha_orig
    },
    {
        'Model': 'Ridge with Interaction Terms',
        'RMSE': round(rmse_ext, 4),
        'R-squared': round(r2_ext, 3),
        'AIC': round(aic_ext, 2),
        'BIC': round(bic_ext, 2),
        'Optimal Alpha': alpha_ext
    }
])

print(ridge_compare)

###

# late addition - marginal effect of SG statistics over time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ridge model with interactions has been fit already as `ridge_ext`
# Extract coefficients
coefs = dict(zip(X_ext.columns, ridge_ext.named_steps['ridgecv'].coef_))

# Create a 'time' range (assuming TIME ranges from 1 to 18 seasons, corresponding to 2004-2021)
years = np.arange(df['TIME'].min(), df['TIME'].max() + 1)

# Compute marginal effect over time
marginal_effects = {
    'SG_OTT': coefs['SG_OTT'] + coefs['SG_OTT_TIME'] * years,
    'SG_APP': coefs['SG_APP'] + coefs['SG_APP_TIME'] * years,
    'SG_ATG': coefs['SG_ATG'] + coefs['SG_ATG_TIME'] * years,
    'SG_PUT': coefs['SG_PUT'] + coefs['SG_PUT_TIME'] * years
}

# Build a DataFrame for plotting
marginal_df = pd.DataFrame({
    'Year': years + 2003,  # If TIME=1 is 2004
    'SG_OTT': marginal_effects['SG_OTT'],
    'SG_APP': marginal_effects['SG_APP'],
    'SG_ATG': marginal_effects['SG_ATG'],
    'SG_PUT': marginal_effects['SG_PUT']
})

# Melt for seaborn
marginal_df_melted = marginal_df.melt(id_vars='Year', var_name='Skill', value_name='Marginal Effect')

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=marginal_df_melted, x='Year', y='Marginal Effect', hue='Skill', linewidth=2)
plt.title('Estimated Marginal Effect of SG Metrics on Log Earnings Over Time')
plt.ylabel('Marginal Effect on LN(Winnings)')
plt.xlabel('Year')
plt.grid(True)
plt.tight_layout()
plt.show()
