import feather
import pandas as pd
import numpy as np
from utils import normalize_by_year, season_to_playoff_year, tune_classifier

# Read datasets
series_data = feather.read_dataframe('series.data')
year_data = feather.read_dataframe('year.data')

# Normalize year by year data
drop_names = ['CONF_COUNT', 'DIV_COUNT', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME', 'NBA_FINALS_APPEARANCE']
year_data = normalize_by_year(year_data, not_considering=drop_names)

# Resolve team name discrepancies
year_data['TEAM_FULLNAME'] = year_data['TEAM_CITY'] + ' ' + year_data['TEAM_NAME']
year_data['PLAYOFF_YEAR'] = year_data['YEAR'].map(season_to_playoff_year)

# Get cartesian product of year_data with itself by year
year_data = pd.merge(year_data, year_data, on='PLAYOFF_YEAR', suffixes=('_winner', '_loser'))

# Join with our playoff results data
joined_data = pd.merge(year_data, series_data, left_on=['TEAM_FULLNAME_winner', 'TEAM_FULLNAME_loser', 'PLAYOFF_YEAR'], right_on=['Winner', 'Loser', 'Year'], how='inner')

# Remove tiebreaker rounds from mid 1900s
joined_data = joined_data[joined_data['Winner Wins'] != 1]

# Calculate the home team for each series
def calc_home_team(row):
    if row['Series'] == 'Finals' or not row['Winner Seed'] or not row['Loser Seed']:
        return row['Winner'] if row['WINS_winner'] > row['WINS_loser'] else row['Loser']
    return row['Winner'] if row['Winner Seed'] < row['Loser Seed'] else row['Loser']
joined_data['home_team'] = joined_data.apply(calc_home_team, axis=1)

# Normalize the margin by the number of games played in the series
def calc_norm_margin(row):
    if row['home_team'] == row['Winner']:
        return (row['Margin'] * 2 - 1) / (row['Winner Wins'] * 2 - 1)
    return -1 * (row['Margin'] * 2 - 1) / (row['Winner Wins'] * 2 - 1)
joined_data['Normalized_margin'] = joined_data.apply(calc_norm_margin, axis=1)

# Pairwise subtract home team stats from away team stats
def pairwise_subtract(row):
    # Parse all numerical winner team stats
    winner_stats = row.filter(regex='_winner$')
    winner_stats = winner_stats[winner_stats.apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
    winner_stats.reset_index(drop=True, inplace=True)

    # Parse all numberical loser team stats
    loser_stats = row.filter(regex='_loser$')
    loser_stats = loser_stats[loser_stats.apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
    loser_stats.reset_index(drop=True, inplace=True)

    # Save our column names as they will be dropped to make this work
    cols = list(winner_stats.index)

    # Subtract winner from loser
    pairwise = winner_stats.subtract(loser_stats)

    # Negate if necessary
    if row['home_team'] == row['Loser']:
        pairwise = -1*pairwise
    
    pairwise['Normalized_margin'] = row['Normalized_margin']
    cols.append('Normalized_margin')
    
    additional_cols = ['Series', 'Year', 'home_team', 'Winner', 'Loser']
    for col in additional_cols:
        pairwise[col] = row[col]

    cols.extend(additional_cols)
    
    return pd.Series(pairwise.values, index=cols)
joined_data = joined_data.apply(pairwise_subtract, axis=1, result_type='expand')

# Save data as feather file
joined_data.reset_index(drop=True).to_feather('joined.data')
