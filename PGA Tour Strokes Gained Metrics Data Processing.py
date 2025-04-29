#ReadMe

#This notebook processes data downlnoaded from the PGA Tour stats portal saved in various folders locally

###

import os
import pandas as pd

folder = #Redacted - insert file path where local files are saved
stat_type = 'SG_OTT'
all_dfs = []

for file in os.listdir(folder):
    if file.endswith('.csv'):
        season = file.replace('.csv', '')
        path = os.path.join(folder, file)
        try:
            df = pd.read_csv(path)
            df['Season'] = season
            df['Stat_Type'] = stat_type
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {path}: {e}")

ott_df = pd.concat(all_dfs, ignore_index=True)

print(ott_df.head())
print(f"Processed {len(ott_df)} rows from {stat_type}")

###

import os
import pandas as pd

root_dir = #Redacted - insert file path
stat_folders = ['SG OTT', 'SG APP', 'SG ATG', 'SG PUT', 'Money']

all_dfs = []

for folder in stat_folders:
    folder_path = os.path.join(root_dir, folder)
    stat_type = folder.replace(' ', '_')

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            season = file.replace('.csv', '')
            path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(path)
                df['Season'] = season
                df['Stat_Type'] = stat_type
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {path}: {e}")

combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.to_csv('combined_golf_stats_long.csv', index=False)

print(combined_df.head())
print(f"Total rows combined: {len(combined_df)}")

###

#combine stats with money files initially, will require some clean up

import os
import pandas as pd
from functools import reduce

root_dir = #redacted - file path

stat_map = {
    'SG OTT': 'SG_OTT',
    'SG APP': 'SG_APP',
    'SG ATG': 'SG_ATG',
    'SG PUT': 'SG_PUT'
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
                if 'PLAYER_ID' in df.columns and 'AVG' in df.columns:
                    subset = df[['PLAYER_ID', 'AVG']].copy()
                    subset.columns = ['Player_ID', col_name]
                    subset['Season'] = season
                    season_dfs.append(subset)
            except Exception as e:
                print(f"Error reading {path}: {e}")

    if season_dfs:
        stat_dfs.append(pd.concat(season_dfs, ignore_index=True))

# Merge SG stats
merged_sg_df = reduce(lambda left, right: pd.merge(left, right, on=['Player_ID', 'Season'], how='outer'), stat_dfs)

# Process Money data
money_folder = os.path.join(root_dir, 'Money')
money_dfs = []

for file in os.listdir(money_folder):
    if file.endswith('.csv'):
        season = '20' + ''.join(filter(str.isdigit, file))[-2:]
        path = os.path.join(money_folder, file)

        try:
            df = pd.read_csv(path)
            df.columns = [c.upper() for c in df.columns]

            money_col = next((c for c in df.columns if 'MONEY' in c or 'EARNINGS' in c), None)
            if money_col and 'PLAYER_ID' in df.columns:
                subset = df[['PLAYER_ID', money_col]].copy()
                subset.columns = ['Player_ID', 'Money']
                subset['Season'] = season
                money_dfs.append(subset)
        except Exception as e:
            print(f"Error reading {path}: {e}")

if not money_dfs:
    raise ValueError("No valid money data found.")

money_df = pd.concat(money_dfs, ignore_index=True)

# Final merge
final_df = pd.merge(merged_sg_df, money_df, on=['Player_ID', 'Season'], how='inner')
final_df = final_df[['Player_ID', 'Season', 'Money', 'SG_APP', 'SG_OTT', 'SG_ATG', 'SG_PUT']]

print(final_df.head())
print(f"Final dataset shape: {final_df.shape}")

final_df.to_csv('panel_model_data.csv', index=False)

###

#begin cleaning merged SG files together over time, with player_ID and season being primary keys

import os
import pandas as pd
from functools import reduce

root_dir = #Redacted - file path

stat_map = {
    'SG OTT': 'SG_OTT',
    'SG APP': 'SG_APP',
    'SG ATG': 'SG_ATG',
    'SG PUT': 'SG_PUT'
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
                if 'PLAYER_ID' in df.columns and 'AVG' in df.columns:
                    subset = df[['PLAYER_ID', 'AVG']].copy()
                    subset.columns = ['Player_ID', col_name]
                    subset['Season'] = season
                    season_dfs.append(subset)
            except Exception as e:
                print(f"Error reading {path}: {e}")

    if season_dfs:
        stat_dfs.append(pd.concat(season_dfs, ignore_index=True))

merged_sg_df = reduce(lambda left, right: pd.merge(left, right, on=['Player_ID', 'Season'], how='outer'), stat_dfs)

# Process Money data
money_folder = os.path.join(root_dir, 'Money')
money_dfs = []

for file in os.listdir(money_folder):
    if file.endswith('.csv'):
        season = '20' + ''.join(filter(str.isdigit, file))[-2:]
        path = os.path.join(money_folder, file)

        try:
            df = pd.read_csv(path)
            df.columns = [c.upper() for c in df.columns]
            money_col = next((c for c in df.columns if 'MONEY' in c or 'EARNINGS' in c), None)
            if money_col and 'PLAYER_ID' in df.columns:
                subset = df[['PLAYER_ID', money_col]].copy()
                subset.columns = ['Player_ID', 'Money']
                subset['Season'] = season
                money_dfs.append(subset)
        except Exception as e:
            print(f"Error reading {path}: {e}")

if not money_dfs:
    raise ValueError("No valid money data found.")

money_df = pd.concat(money_dfs, ignore_index=True)
money_df['Money'] = money_df['Money'].astype(str).str.replace('[\$,]', '', regex=True).astype(float)

# Extract player names from SG OTT
name_dfs = []
ott_folder = os.path.join(root_dir, 'SG OTT')

for file in os.listdir(ott_folder):
    if file.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join(ott_folder, file))
            df.columns = [c.upper() for c in df.columns]
            if 'PLAYER_ID' in df.columns and 'PLAYER' in df.columns:
                subset = df[['PLAYER_ID', 'PLAYER']].rename(columns={'PLAYER_ID': 'Player_ID', 'PLAYER': 'Player_Name'})
                name_dfs.append(subset)
        except:
            continue

if not name_dfs:
    raise ValueError("Could not extract player names.")

name_df = pd.concat(name_dfs).drop_duplicates(subset='Player_ID')

# Final merge
final_df = pd.merge(merged_sg_df, money_df, on=['Player_ID', 'Season'], how='inner')
final_df = pd.merge(final_df, name_df, on='Player_ID', how='left')

final_df = final_df[['Player_ID', 'Player_Name', 'Season', 'Money', 'SG_APP', 'SG_OTT', 'SG_ATG', 'SG_PUT']]

print(final_df.head())
print(f"Final dataset shape: {final_df.shape}")
final_df.to_csv('panel_model_data.csv', index=False)


###

#create money per event variable

import os
import pandas as pd
from functools import reduce

root_dir = #redacted

stat_map = {
    'SG OTT': 'SG_OTT',
    'SG APP': 'SG_APP',
    'SG ATG': 'SG_ATG',
    'SG PUT': 'SG_PUT'
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
                if 'PLAYER_ID' in df.columns and 'AVG' in df.columns:
                    subset = df[['PLAYER_ID', 'AVG']].copy()
                    subset.columns = ['Player_ID', col_name]
                    subset['Season'] = season
                    season_dfs.append(subset)
            except Exception as e:
                print(f"Error reading {path}: {e}")

    if season_dfs:
        stat_dfs.append(pd.concat(season_dfs, ignore_index=True))

merged_sg_df = reduce(lambda left, right: pd.merge(left, right, on=['Player_ID', 'Season'], how='outer'), stat_dfs)

# Load money per event data
money_folder = os.path.join(root_dir, 'Money per Event')
money_dfs = []

for file in os.listdir(money_folder):
    if file.endswith('.csv'):
        season = '20' + ''.join(filter(str.isdigit, file))[-2:]
        path = os.path.join(money_folder, file)

        try:
            df = pd.read_csv(path)
            df.columns = [c.upper() for c in df.columns]
            required = {'PLAYER_ID', 'MONEY PER EVENT', 'TOTAL MONEY'}

            if required.issubset(df.columns):
                subset = df[['PLAYER_ID', 'MONEY PER EVENT', 'TOTAL MONEY']].copy()
                subset.columns = ['Player_ID', 'Money_Per_Event', 'Total_Money']
                subset['Season'] = season
                money_dfs.append(subset)
        except Exception as e:
            print(f"Error reading {path}: {e}")

if not money_dfs:
    raise ValueError("No valid 'Money per Event' data found.")

money_df = pd.concat(money_dfs, ignore_index=True)

# Clean money columns
for col in ['Total_Money', 'Money_Per_Event']:
    money_df[col] = money_df[col].astype(str).str.replace('[\$,]', '', regex=True).astype(float)

# Extract player names from SG OTT
name_dfs = []
ott_folder = os.path.join(root_dir, 'SG OTT')

for file in os.listdir(ott_folder):
    if file.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join(ott_folder, file))
            df.columns = [c.upper() for c in df.columns]
            if 'PLAYER_ID' in df.columns and 'PLAYER' in df.columns:
                subset = df[['PLAYER_ID', 'PLAYER']].rename(columns={'PLAYER_ID': 'Player_ID', 'PLAYER': 'Player_Name'})
                name_dfs.append(subset)
        except:
            continue

if not name_dfs:
    raise ValueError("Could not extract player names.")

name_df = pd.concat(name_dfs).drop_duplicates(subset='Player_ID')

# Final merge
final_df = pd.merge(merged_sg_df, money_df, on=['Player_ID', 'Season'], how='inner')
final_df = pd.merge(final_df, name_df, on='Player_ID', how='left')

final_df = final_df[['Player_ID', 'Player_Name', 'Season', 'Total_Money', 'Money_Per_Event',
                     'SG_APP', 'SG_OTT', 'SG_ATG', 'SG_PUT']]

print(final_df.head())
print(f"Final dataset shape: {final_df.shape}")
final_df.to_csv('panel_model_data.csv', index=False)

###

#adjust money features for inflation. need to have CPI data downloaded to machien locally ('cpi.xlsx')

import pandas as pd

# Load CPI data and panel data
cpi_data = pd.read_excel('cpi.xlsx', engine='openpyxl')
df = pd.read_csv('panel_model_data.csv')

# Clean money columns
for col in ['Total_Money', 'Money_Per_Event']:
    df[col] = df[col].astype(str).str.replace('[\$,]', '', regex=True).astype(float)

# Merge CPI and adjust for inflation (base year: 2021)
df = df.merge(cpi_data, left_on='Season', right_on='Year', how='left')
base_cpi = cpi_data.loc[cpi_data['Year'] == 2021, 'CPI'].values[0]

for col in ['Total_Money', 'Money_Per_Event']:
    df[f'{col}_Adjusted'] = df[col] * (base_cpi / df['CPI'])

# Clean up and save
df.drop(columns=['CPI', 'Year'], inplace=True)
df.to_csv('adjusted_panel_model_data.csv', index=False)

print(df.head())
print(f"Adjusted dataset saved: {df.shape} rows")

###
