import feather
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from data_preprocessing import calc_home_team
from utils import normalize_by_year, season_to_playoff_year



def concat_home_away(row):
    # Parse all numerical winner team stats
    winner_stats = row.filter(regex='_winner$')
    winner_stats = winner_stats[winner_stats.apply(lambda x: type(x) in [int, np.int64, float, np.float64])]

    # Save our column names as they will be dropped to make this work
    cols = list(winner_stats.index)
    winner_stats.reset_index(drop=True, inplace=True)

    # Parse all numberical loser team stats
    loser_stats = row.filter(regex='_loser$')
    loser_stats = loser_stats[loser_stats.apply(lambda x: type(x) in [int, np.int64, float, np.float64])]
    loser_stats.reset_index(drop=True, inplace=True)

    # Generate final column names
    cols = [col.replace('_winner', '') for col in cols]
    home_cols = [col + '_home' for col in cols]
    away_cols = [col + '_away' for col in cols]
    cols = home_cols + away_cols

    margin = row['Normalized_margin']
    if 'Loser' in row and row['home_team'] == row['Loser']:
        home_stats = loser_stats
        away_stats = winner_stats
        margin = -1* margin
    else:
        home_stats = winner_stats
        away_stats = loser_stats
    
    concatenated = home_stats.append(away_stats)

    concatenated['Normalized_margin'] = margin
    cols.append('Normalized_margin')
    
    additional_cols = ['Series', 'Year', 'home_team', 'Winner', 'Loser']
    added_cols = []
    for col in additional_cols:
        if col in row:
            concatenated[col] = row[col]
            added_cols.append(col)

    cols.extend(added_cols)
    
    return pd.Series(concatenated.values, index=cols)

def generate_concat_data():
    # read datasets
    series_data = feather.read_dataframe('series.data')
    year_data = feather.read_dataframe('year.data')

    # normalize year-by-year data
    # drop_names = ['CONF_COUNT', 'DIV_COUNT', 'TEAM_ID', 'TEAM_CITY', 'TEAM_NAME', 'NBA_FINALS_APPEARANCE']
    # year_data = normalize_by_year(year_data, not_considering=drop_names)

    # Normalize nae and year across our two data sources
    year_data['TEAM_FULLNAME'] = year_data['TEAM_CITY'] + ' ' + year_data['TEAM_NAME']
    year_data['PLAYOFF_YEAR'] = year_data['YEAR'].map(season_to_playoff_year)

    # Get cartesian product of year_data with itself by year
    year_data = pd.merge(year_data, year_data, on='PLAYOFF_YEAR', suffixes=('_winner', '_loser'))

    # Join with our playoff results data
    joined_data = pd.merge(year_data, series_data, left_on=['TEAM_FULLNAME_winner', 'TEAM_FULLNAME_loser', 'PLAYOFF_YEAR'], right_on=['Winner', 'Loser', 'Year'], how='inner')

    # Remove tiebreaker rounds from the ancient days
    joined_data = joined_data[joined_data['Winner Wins'] != 1]

    joined_data['home_team'] = joined_data.apply(calc_home_team, axis=1)

    joined_data['Normalized_margin'] = joined_data['Margin'] / joined_data['Winner Wins']

    joined_data = joined_data.apply(concat_home_away, axis=1, result_type='expand')
    
    return joined_data




def visualize_with_pca(X, y, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    Xax=X_pca[:,0]
    Yax=X_pca[:,1]
    labels = (y > 0).astype(int)
    cdict =  {0 : 'red', 1 : 'green'}
    labl = { 0 : 'home_loss', 1 : 'home_win'}
    marker={0:'*',1:'o'}
    alpha={0:.3, 1:.5}

    fig,ax=plt.subplots(figsize=(7,5))
    fig.patch.set_facecolor('white')

    for l in np.unique(labels):
        ix=np.where(labels==l)
        ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40, label=labl[l],marker=marker[l],alpha=alpha[l])

    plt.xlabel("First Principal Component",fontsize=14)
    plt.ylabel("Second Principal Component",fontsize=14)
    plt.legend()
    plt.title(title)
    plt.show()

def visualize_with_cca(X, y, title):
    cca = CCA(n_components=2)
    cca.fit(X, y)
    X_cca = cca.transform(X)
    Xax=X_cca[:,0]
    Yax=X_cca[:,1]
    labels = (y > 0).astype(int)
    cdict =  {0 : 'red', 1 : 'green'}
    labl = { 0 : 'home_loss', 1 : 'home_win'}
    marker={0:'*',1:'o'}
    alpha={0:.3, 1:.5}

    fig,ax=plt.subplots(figsize=(7,5))
    fig.patch.set_facecolor('white')

    for l in np.unique(labels):
        ix=np.where(labels==l)
        ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40, label=labl[l],marker=marker[l],alpha=alpha[l])

    plt.xlabel("First Principal Component",fontsize=14)
    plt.ylabel("Second Principal Component",fontsize=14)
    plt.legend()
    plt.title(title)
    plt.show()

concat_data = generate_concat_data()
pairwise_data = feather.read_dataframe('joined.data')

pairwise_drop_names = ['TEAM_ID_winner', 'GP_winner', 'WINS_winner', 'LOSSES_winner',  'CONF_RANK_winner', 'DIV_RANK_winner',  'PO_WINS_winner', 'PO_LOSSES_winner', 'PTS_RANK_winner', 'CONF_COUNT_winner', 'Series', 'Winner', 'Loser', 'home_team', 'DIV_COUNT_winner', 'Year']
base_drop_names = ['TEAM_ID', 'GP', 'WINS', 'LOSSES', 'CONF_RANK', 'PO_WINS', 'PO_LOSSES', 'PTS_RANK', 'CONF_COUNT', 'DIV_RANK', 'DIV_COUNT']
non_team_names = [ 'Series', 'Winner', 'Loser', 'home_team','Year']
home_drop_names = [name + '_home' for name in base_drop_names]
away_drop_names = [name + '_away' for name in base_drop_names]
concat_drop_names = home_drop_names + away_drop_names + non_team_names

concat_data = concat_data.drop(concat_drop_names, axis=1)
pairwise_data = pairwise_data.drop(pairwise_drop_names, axis=1)
concat_data.reset_index(drop=True, inplace=True)
pairwise_data.reset_index(drop=True, inplace=True)

concat_X = concat_data.drop(['Normalized_margin'], axis=1)
concat_y = concat_data['Normalized_margin']

pairwise_X = pairwise_data.drop(['Normalized_margin'], axis=1)
pairwise_y = pairwise_data['Normalized_margin']


visualize_with_pca(concat_X, concat_y, 'Concatenated Data PCA')
visualize_with_pca(pairwise_X, pairwise_y, 'Pairwise Subtracted PCA')
visualize_with_cca(concat_X, concat_y, 'Concatenated Data CCA')
visualize_with_cca(pairwise_X, pairwise_y, 'Pairwise Subtracted CCA')