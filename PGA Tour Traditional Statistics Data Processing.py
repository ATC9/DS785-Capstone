#ReadME:

# This script proccess data stored in multiple folders on a local machine to build a modeling panel data set.
# This particular script was used to build the model data set leveraging traditional golf statistics.

# loop through folders to merge data together via primary key player_id and season

import os
import pandas as pd
from functools import reduce

root_dir = #REDACTED - INSERT FILE PATH HERE

stat_map = {
    'DD': 'Driving_Distance',
    'DA': 'Driving_Accuracy',
    'GIR': 'Greens_In_Regulation',
    'SS': 'Sand_Saves',
    'PPGIR': 'Putts_Per_GIR',
    'PUTTS': 'Total_Putts'
}

stat_dfs = []

for folder, col_name in stat_map.items():
    folder_path = os.path.join(root_dir, folder)
    season_dfs = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            season = '20' + ''.join(filter(str.isdigit, file))[-2:]
            path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(path)
                df.columns = [c.upper() for c in df.columns]

                if folder in ['DA', 'GIR', 'SS']:
                    metric_col = '%'
                elif folder == 'PUTTS':
                    metric_col = 'TOTAL PUTTS'
                else:
                    metric_col = 'AVG'

                if 'PLAYER_ID' in df.columns and metric_col in df.columns:
                    subset = df[['PLAYER_ID', metric_col]].copy()
                    subset.columns = ['Player_ID', col_name]
                    subset['Season'] = season
                    subset['Rounds'] = df['TOTAL ROUNDS'] if folder == 'PUTTS' and 'TOTAL ROUNDS' in df.columns else None
                    season_dfs.append(subset)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    if season_dfs:
        stat_dfs.append(pd.concat(season_dfs, ignore_index=True))

# Remove 'Rounds' from all stat_dfs except the one that contains it
for i, df in enumerate(stat_dfs):
    round_cols = [col for col in df.columns if 'Rounds' in col]
    if round_cols and 'Total_Putts' not in df.columns:
        stat_dfs[i] = df.drop(columns=round_cols)

# Merge all stat DataFrames
merged_stats_df = reduce(
    lambda left, right: pd.merge(left, right, on=['Player_ID', 'Season'], how='outer', suffixes=('', '_dup')),
    stat_dfs
)

# Fill Rounds column and drop dupes
round_cols = [col for col in merged_stats_df.columns if 'Rounds' in col]
if len(round_cols) > 1:
    merged_stats_df['Rounds'] = merged_stats_df[round_cols].bfill(axis=1).iloc[:, 0]
    merged_stats_df.drop(columns=[col for col in round_cols if col != 'Rounds'], inplace=True)

# === Load Money Data
money_folder = os.path.join(root_dir, 'Money per Event')
money_dfs = []

for file in os.listdir(money_folder):
    if file.endswith('.csv'):
        season = '20' + ''.join(filter(str.isdigit, file))[-2:]
        path = os.path.join(money_folder, file)

        try:
            df = pd.read_csv(path)
            df.columns = [c.upper() for c in df.columns]
            if {'PLAYER_ID', 'MONEY PER EVENT', 'TOTAL MONEY'}.issubset(df.columns):
                subset = df[['PLAYER_ID', 'MONEY PER EVENT', 'TOTAL MONEY']].copy()
                subset.columns = ['Player_ID', 'Money_Per_Event', 'Total_Money']
                subset['Season'] = season
                money_dfs.append(subset)
        except Exception as e:
            print(f"Error processing money file {path}: {e}")

if not money_dfs:
    raise ValueError("No valid money data found.")
money_df = pd.concat(money_dfs, ignore_index=True)

# === Extract Player Names from DD folder
name_dfs = []
dd_folder = os.path.join(root_dir, 'DD')

for file in os.listdir(dd_folder):
    if file.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join(dd_folder, file))
            df.columns = [c.upper() for c in df.columns]
            if 'PLAYER_ID' in df.columns and 'PLAYER' in df.columns:
                name_dfs.append(df[['PLAYER_ID', 'PLAYER']].rename(columns={'PLAYER_ID': 'Player_ID', 'PLAYER': 'Player_Name'}))
        except:
            continue

if not name_dfs:
    raise ValueError("No player name data found.")
name_df = pd.concat(name_dfs).drop_duplicates(subset='Player_ID')

# === Final Merge
final_df = pd.merge(merged_stats_df, money_df, on=['Player_ID', 'Season'], how='inner')
final_df = pd.merge(final_df, name_df, on='Player_ID', how='left')

# Final Rounds cleanup
round_cols_final = [col for col in final_df.columns if 'Rounds' in col]
if len(round_cols_final) > 1:
    final_df['Rounds'] = final_df[round_cols_final].bfill(axis=1).iloc[:, 0]
    final_df.drop(columns=[col for col in round_cols_final if col != 'Rounds'], inplace=True)

# Reorder and save
final_cols = ['Player_ID', 'Player_Name', 'Season', 'Total_Money', 'Money_Per_Event',
              'Driving_Distance', 'Driving_Accuracy', 'Greens_In_Regulation',
              'Sand_Saves', 'Putts_Per_GIR', 'Total_Putts', 'Rounds']
final_df = final_df[[col for col in final_cols if col in final_df.columns]]

print(final_df.head())
print(f"Final dataset shape: {final_df.shape}")
final_df.to_csv('traditional_model_data.csv', index=False)


####

# Create shortgam feature, and clean existing features

import pandas as pd

# Load dataset
final_df = pd.read_csv('traditional_model_data.csv')

# Clean money columns
money_cols = ['Total_Money', 'Money_Per_Event']
for col in money_cols:
    final_df[col] = final_df[col].astype(str).str.replace('[\$,]', '', regex=True).astype(float)

# Clean percentage columns
percent_cols = ['Greens_In_Regulation', 'Driving_Accuracy', 'Sand_Saves']
for col in percent_cols:
    final_df[col] = pd.to_numeric(final_df[col].astype(str).str.strip(), errors='coerce')

# Create derived features
final_df['Events'] = final_df['Total_Money'] / final_df['Money_Per_Event']
final_df['GIR_Decimal'] = final_df['Greens_In_Regulation'] / 100
final_df['ShortGam'] = (
    ((1 / 18) * (final_df['Total_Putts'] / final_df['Rounds'])) - 
    (final_df['GIR_Decimal'] * final_df['Putts_Per_GIR'])
) / ((100 - final_df['Greens_In_Regulation']) / 100)

# Drop helper columns
final_df.drop(columns=['GIR_Decimal'], inplace=True)

# Reorder and save
final_cols = ['Player_ID', 'Player_Name', 'Season', 'Total_Money', 'Money_Per_Event',
              'Driving_Distance', 'Driving_Accuracy', 'Greens_In_Regulation', 'Sand_Saves',
              'Putts_Per_GIR', 'Total_Putts', 'Rounds', 'Events', 'ShortGam']
final_df = final_df[final_cols]

final_df.to_csv('traditional_model_data_with_features.csv', index=False)
print(final_df.head())
print(f"Final dataset shape with new features: {final_df.shape}")

#####

# adjust money features for inflation - need to download CPI data from bls.gove website and save it locally (i.e., 'cpi.xlsx')


import pandas as pd

# Load CPI and PGA Tour data
cpi_data = pd.read_excel('cpi.xlsx', engine='openpyxl')
df = pd.read_csv('traditional_model_data_with_features.csv')

# Merge CPI data based on year
df = df.merge(cpi_data, left_on='Season', right_on='Year', how='left')

# Adjust earnings for inflation using 2021 as the base year
base_cpi = cpi_data.loc[cpi_data['Year'] == 2021, 'CPI'].values[0]
money_cols = ['Total_Money', 'Money_Per_Event']
for col in money_cols:
    df[f'{col}_Adjusted'] = df[col] * (base_cpi / df['CPI'])

# Drop merge helper columns and save
df.drop(columns=['CPI', 'Year'], inplace=True)
df.to_csv('adjusted_traditional_model_data.csv', index=False)

print(df.head())
print("Earnings adjusted for inflation and saved to 'adjusted_traditional_model_data.csv'")

###

# add time trend to data set for modeling purposes - this really isnt necessary here, could be done at any point


# Import required libraries
import pandas as pd

# Define file path
file_path = "adjusted_traditional_model_data.csv"

# Load the CSV into a DataFrame
data = pd.read_csv(file_path)

# Create the TIME variable (starting at 1 for 2004)
data['TIME'] = data['Season'] - 2004 + 1

# Save the updated DataFrame to a new CSV
output_path = "adjusted_traditional_model_data_with_time.csv"
data.to_csv(output_path, index=False)

print(f"File successfully updated! Saved as: {output_path}")
