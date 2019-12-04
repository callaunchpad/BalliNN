from data import BoxScoreAdvancedV2
from multiprocessing import Pool
from nba_api.stats.static import players, teams
from utils import save_obj, load_obj, parse_matchup

import pandas as pd
import os

game_data = {}
team_history = load_obj('team_history')

# find set of game_ids
print('find set of game ids...')
game_ids = set()
for team, history in team_history.items():
    for _, game in history.iterrows():
        game_ids.add(game['GAME_ID'])

# join by game_id
print('joining by game_id...')
for team, history in team_history.items():
	print('team: %s' % team)
	for _, game in history.iterrows():
		game_id = game['GAME_ID']
		print('game_id: %s' % game_id)
		if game_id not in game_data:
			game_data[game_id] = [game]
		else:
			game_data[game_id].append(game)

game_data = {}
for team, history in team_history.items():
	print('team: %s' % team)
	for index, game in history.iterrows():
		game_id = game['GAME_ID']
		print('joining game_id: %s [%d/%d]' % (game_id, len(game_data), len(game_ids)))

		if not game_id in game_data:
			data = {}
			matchup = game['MATCHUP']
			home_team, away_team = parse_matchup(matchup)

			data['home_team'] = home_team
			data['away_team'] = away_team
			data['game_date'] = game['GAME_DATE']

			lookup = {'W':'L', 'L':'W', 'T':'T', None:None}
			win_loss = game['WL']
			if home_team == game['TEAM_ABBREVIATION']:
				home_win = win_loss
			home_win = lookup[win_loss]
			data['home_win'] = home_win
			if home_win is None: continue

			team_stats = game.drop(['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL'])
			dfs = load_obj(filename=game_id, dirname='advanced')
			advanced_player_stats, advanced_team_stats = dfs[0], dfs[1]
			if home_team == game['TEAM_ABBREVIATION']:
				data['home_team_stats'] = team_stats
				data['away_team_stats'] = None
			else:
				data['away_team_stats'] = team_stats
				data['home_team_stats'] = None

			data['advanced_player_stats'] = advanced_player_stats
			data['advanced_team_stats'] = advanced_team_stats

			game_data[game_id] = data
		else:
			if game_data[game_id]['home_team_stats'] is not None and game_data[game_id]['away_team_stats'] is not None:
				continue
			if game_data[game_id]['home_team_stats'] is None:
				key = 'home_team_stats'
			else:
				key = 'away_team_stats'
			game_data[game_id][key] = game.drop(['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL'])

save_obj(game_data, 'game_data')