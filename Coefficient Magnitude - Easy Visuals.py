# traditional stats model coefficient magnitudes - just plopping the coefficient values from my final model manually

import pandas as pd
import matplotlib.pyplot as plt

# === Input Data: Traditional Stats Model Coefficients, ex constant
data = {
    'Variable': [
        'LN_PPGIR', 'LN_DD', 'GIR', 'Driving_Accuracy',
        'Sand_Saves', 'LN_SHORTGAM', 'LN_EVENTS',
        'LN_EVENTS_SQUARED', 'TIME'
    ],
    'Coefficient': [
        -0.3955, 0.3076, 0.2643, 0.1467,
        0.1378, -0.1387, 0.8541,
        -0.9645, -0.2114
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Sort by absolute coefficient magnitude
df['Abs_Coefficient'] = df['Coefficient'].abs()
df = df.sort_values('Abs_Coefficient', ascending=False)

# Color events-related variables differently
colors = df['Variable'].apply(
    lambda x: 'orange' if 'EVENTS' in x else 'skyblue'
)

# Create Horizontal Bar Chart
plt.figure(figsize=(10, 6))
plt.barh(df['Variable'], df['Coefficient'], color=colors) 
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Coefficient Estimate')
plt.title('Magnitude of Coefficients from Traditional Statistics Model\nPredicting PGA Tour Earnings per Event')
plt.tight_layout()
plt.gca().invert_yaxis()  
plt.show()

###

# Strokes Gained Coefficients - Im just mannually inputting the coefficient values here from ridge model, using above code

import pandas as pd
import matplotlib.pyplot as plt

# === Input Data: Strokes Gained Ridge Coefficients
data_sg = {
    'Variable': [
        'SG_OTT', 'SG_APP', 'SG_ATG', 'SG_PUT',
        'LN_EVENTS', 'LN_EVENTS_SQ', 'TIME'
    ],
    'Coefficient': [
        0.4664, 0.4643, 0.2562, 0.3825,
        0.4170, -0.4442, 0.0573
    ]
}

# === Create DataFrame
df_sg = pd.DataFrame(data_sg)

# === Sort by absolute magnitude
df_sg['Abs_Coefficient'] = df_sg['Coefficient'].abs()
df_sg = df_sg.sort_values('Abs_Coefficient', ascending=False)

# === Create Horizontal Bar Chart
plt.figure(figsize=(10, 6))
plt.barh(df_sg['Variable'], df_sg['Coefficient'], color='lightgreen')
plt.axvline(0, color='black', linewidth=0.8)
plt.xlabel('Ridge Coefficient Estimate')
plt.title('Magnitude of Coefficients from Strokes Gained Ridge Model\nPredicting PGA Tour Earnings per Event')
plt.tight_layout()
plt.gca().invert_yaxis()  
plt.show()
