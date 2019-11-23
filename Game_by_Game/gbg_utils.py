from nba_api.stats.endpoints import commonplayerinfo, playercareerstats, leaguegamefinder, commonteamroster
from nba_api.stats.static import players, teams
import numpy as np
from os import path
import pandas as pd
import pickle

def save_obj(obj, filename, dirname='pickle_files', ):
    with open(path.join(dirname, filename + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename, dirname='pickle_files'):
    with open(path.join(dirname, filename + '.pkl'), 'rb') as f:
        return pickle.load(f)

""" gets the season games given the seasonID and abbreviation of the team """
""" Season is based on the year that the season STARTED in. I.E 2016-2017 has seasonID 2016"""
""" Reg season starts with 2, playoffs 4, pre 1 """
""" 
For some reason summer league and preseason games are sometimes included in reg season IDs.
The function accounts for this by only looking at Game IDs that start with 0021 for reg reason,
0041 for playoffs, 0011 for preseason
"""
def get_season_team(seasonID, team_abr, team_names):
    seasonID = str(seasonID)
    gameID_prefix = seasonID[0]
    return team_names[team_abr].loc[(team_names[team_abr]["SEASON_ID"].str.contains(seasonID)) & \
                                    (team_names[team_abr]['GAME_ID'].str.startswith('00{0}1'.format(gameID_prefix)))]

""" gets the number of rows in the passed df"""
def num_rows(df):
    return len(df.index)

""" returns the number of features of the df"""
def num_features(df):
    return len(df.columns)

""" returns a list of the features from the start index moving forward. Using 9 for now """
def get_features(df, start_idx=9):
    return [feature for feature in df.columns.tolist()[start_idx:]]

""" 
util func: gets last n games given the curr game
in the df with the correct season, year
"""
def last_n_games(df, n, curr_game):
    n_rows = num_rows(df)
    # the first row is the latest game. I.E for reg season it's 82nd game
    
"""
returns avg team stats vector over the last n games given
the current game. Assumes games 0 indexed
"""
def get_team_avgs(season_games, curr_game, n):
    total_games = num_rows(season_games)
    assert curr_game > n, "curr game ({0}) > n ({1})".format(curr_game, n)
    assert total_games > n, "n ({0}) is more than the total number of games ({1})".format(n, total_games)
    # the real index of the current game
    game_idx = total_games - curr_game - 1
    # start at 9th index to get meaningful stats
    start_idx = 9
    team_avgs = np.zeros(num_features(season_games) - start_idx)
    for i in range(game_idx - n, game_idx):
        game_stats = np.array(season_games.iloc[i].tolist()[start_idx:])
        team_avgs += game_stats
    return team_avgs / n

def print_team_avgs(season_games, team_avgs):
    features = get_features(season_games)
    for stat, feature in zip(team_avgs, features):
        print("{0}:{1}".format(feature, stat))