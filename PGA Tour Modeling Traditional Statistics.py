# This notebook includes code for various regression models using traditional statistics
# Starts with OLS cross validated
# Moves on to regularlization models and hypertuning
# then based on final model, we use that to extend it to include interaction terms with TIME

# Base OLS Model with Cross Validation

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# === Load and Prepare Data === #
df = pd.read_csv("adjusted_traditional_model_data_with_time.csv")

# Log transforms
df['LN_WINNINGS'] = np.log(df['Money_Per_Event_Adjusted'])
df['LN_DD'] = np.log(df['Driving_Distance'])
df['LN_PPGIR'] = np.log(df['Putts_Per_GIR'])
df['LN_SHORTGAM'] = np.log(df['ShortGam'])
df['LN_EVENTS'] = np.log(df['Events'])
df['LN_EVENTS_SQ'] = df['LN_EVENTS'] ** 2

# Define features and target
features = ['LN_DD', 'Driving_Accuracy', 'Greens_In_Regulation',
            'LN_PPGIR', 'Sand_Saves', 'LN_SHORTGAM',
            'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']
X = df[features]
y = df['LN_WINNINGS']

# === Train/test split === #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Standardize predictors === #
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Add constant and fit OLS === #
X_train_sm = sm.add_constant(X_train_scaled)
X_test_sm = sm.add_constant(X_test_scaled)

ols_model = sm.OLS(y_train, X_train_sm).fit()
y_pred = ols_model.predict(X_test_sm)

# === Evaluate Performance === #
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
aic = ols_model.aic
bic = ols_model.bic

# === Output Results === #
print("=== Cross-Validated OLS Model Performance ===")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"AIC: {aic:.2f}")
print(f"BIC: {bic:.2f}")

###

# OLS Coefficients

# Create a DataFrame with OLS model results
ols_results_df = pd.DataFrame({
    "Variable": ['Intercept'] + features,
    "Coefficient": ols_model.params,
    "Std_Error": ols_model.bse,
    "t_Value": ols_model.tvalues,
    "p_Value": ols_model.pvalues
})

# Display the coefficients
print(ols_results_df.round(4))

###

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# === Load and Prepare Data === #
df = pd.read_csv("adjusted_traditional_model_data_with_time.csv")

# Log transform variables
df['LN_WINNINGS'] = np.log(df['Money_Per_Event_Adjusted'])
df['LN_DD'] = np.log(df['Driving_Distance'])
df['LN_PPGIR'] = np.log(df['Putts_Per_GIR'])
df['LN_SHORTGAM'] = np.log(df['ShortGam'])
df['LN_EVENTS'] = np.log(df['Events'])
df['LN_EVENTS_SQ'] = df['LN_EVENTS'] ** 2

# Define features and target
features = ['LN_DD', 'Driving_Accuracy', 'Greens_In_Regulation',
            'LN_PPGIR', 'Sand_Saves', 'LN_SHORTGAM',
            'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']
X = df[features]
y = df['LN_WINNINGS']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize predictors
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit regularized models
alphas = np.logspace(-3, 1, 50)
lasso = LassoCV(alphas=alphas, cv=5, max_iter=10000).fit(X_train_scaled, y_train)
ridge = RidgeCV(alphas=alphas, cv=5).fit(X_train_scaled, y_train)
enet = ElasticNetCV(alphas=alphas, l1_ratio=np.linspace(0.1, 1, 10), cv=5, max_iter=10000).fit(X_train_scaled, y_train)

# Predictions
y_pred_lasso = lasso.predict(X_test_scaled)
y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_enet = enet.predict(X_test_scaled)

# === Statsmodels-based AIC/BIC and performance printing === #
def report_model_metrics(y_true, y_pred, model_name):
    y_true_aligned = pd.Series(y_true).reset_index(drop=True)
    y_pred_aligned = pd.Series(y_pred).reset_index(drop=True)

    X_fake = sm.add_constant(y_pred_aligned)
    ols = sm.OLS(y_true_aligned, X_fake).fit()

    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_aligned))
    r2 = r2_score(y_true_aligned, y_pred_aligned)
    
    print(f"\n=== {model_name} ===")
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"AIC: {ols.aic:.2f}")
    print(f"BIC: {ols.bic:.2f}")
    return {"Model": model_name, "R2": r2, "RMSE": rmse, "AIC": ols.aic, "BIC": ols.bic}

# Output all models
report_model_metrics(y_test, y_pred_lasso, "Lasso")
report_model_metrics(y_test, y_pred_ridge, "Ridge")
report_model_metrics(y_test, y_pred_enet, "Elastic Net")

###

# show coefficients

# Create a DataFrame for each model’s coefficients
coef_df_lasso = pd.DataFrame({
    "Variable": ['Intercept'] + features,
    "Lasso Coefficients": [lasso.intercept_] + list(lasso.coef_)
})

coef_df_ridge = pd.DataFrame({
    "Variable": ['Intercept'] + features,
    "Ridge Coefficients": [ridge.intercept_] + list(ridge.coef_)
})

coef_df_enet = pd.DataFrame({
    "Variable": ['Intercept'] + features,
    "Elastic Net Coefficients": [enet.intercept_] + list(enet.coef_)
})

# Merge all into one table
coefficients_df = coef_df_lasso.merge(coef_df_ridge, on="Variable").merge(coef_df_enet, on="Variable")

# Display the coefficients table
print(coefficients_df)

###

# OLS with cross validation with time index

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# === Load and Prepare Data === #
df = pd.read_csv("adjusted_traditional_model_data_with_time.csv")

# Log transform variables
df['LN_WINNINGS'] = np.log(df['Money_Per_Event_Adjusted'])
df['LN_DD'] = np.log(df['Driving_Distance'])
df['LN_PPGIR'] = np.log(df['Putts_Per_GIR'])
df['LN_SHORTGAM'] = np.log(df['ShortGam'])
df['LN_EVENTS'] = np.log(df['Events'])
df['LN_EVENTS_SQ'] = df['LN_EVENTS'] ** 2

# Create interaction terms with TIME
df['LN_DD_TIME'] = df['LN_DD'] * df['TIME']
df['DA_TIME'] = df['Driving_Accuracy'] * df['TIME']
df['GIR_TIME'] = df['Greens_In_Regulation'] * df['TIME']
df['LN_PPGIR_TIME'] = df['LN_PPGIR'] * df['TIME']
df['SS_TIME'] = df['Sand_Saves'] * df['TIME']
df['LN_SHORTGAM_TIME'] = df['LN_SHORTGAM'] * df['TIME']

# === Define Variables for Original and Interaction Models === #
features_original = ['LN_DD', 'Driving_Accuracy', 'Greens_In_Regulation',
                     'LN_PPGIR', 'Sand_Saves', 'LN_SHORTGAM',
                     'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME']

features_interaction = features_original + ['LN_DD_TIME', 'DA_TIME', 'GIR_TIME',
                                            'LN_PPGIR_TIME', 'SS_TIME', 'LN_SHORTGAM_TIME']

X_orig = df[features_original]
X_inter = df[features_interaction]
y = df['LN_WINNINGS']

# === Train/Test Split === #
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X_orig, y, test_size=0.2, random_state=42)
X_train_inter, X_test_inter, y_train_inter, y_test_inter = train_test_split(X_inter, y, test_size=0.2, random_state=42)

# === Standardize Predictors === #
scaler = StandardScaler()
X_train_orig_scaled = scaler.fit_transform(X_train_orig)
X_test_orig_scaled = scaler.transform(X_test_orig)
X_train_inter_scaled = scaler.fit_transform(X_train_inter)
X_test_inter_scaled = scaler.transform(X_test_inter)

# === Add Constant for Intercept === #
X_train_orig_const = sm.add_constant(X_train_orig_scaled)
X_test_orig_const = sm.add_constant(X_test_orig_scaled)
X_train_inter_const = sm.add_constant(X_train_inter_scaled)
X_test_inter_const = sm.add_constant(X_test_inter_scaled)

# === Fit OLS Models === #
ols_model_orig = sm.OLS(y_train_orig, X_train_orig_const).fit()
ols_model_inter = sm.OLS(y_train_inter, X_train_inter_const).fit()

# === Predict on Test Sets === #
y_pred_orig = ols_model_orig.predict(X_test_orig_const)
y_pred_inter = ols_model_inter.predict(X_test_inter_const)

# === Evaluate Models === #
def model_metrics(y_true, y_pred, model, X):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R-squared": r2_score(y_true, y_pred),
        "AIC": model.aic,
        "BIC": model.bic
    }

metrics_orig = model_metrics(y_test_orig, y_pred_orig, ols_model_orig, X_test_orig_const)
metrics_inter = model_metrics(y_test_inter, y_pred_inter, ols_model_inter, X_test_inter_const)

# === Print Results === #
print("Original OLS Metrics:\n", metrics_orig)
print("OLS with Interaction Terms Metrics:\n", metrics_inter)
print("\nModel Summary (with interaction terms):")
print(ols_model_inter.summary())

###




