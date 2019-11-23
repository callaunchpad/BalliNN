from data import BoxScoreAdvancedV2
from nba_api.stats.static import players, teams
from utils import save_obj, load_obj, parse_matchup

import pandas as pd

# game_data = {}
team_history = load_obj('team_history')

# find set of game_ids
print('find set of game ids...')
game_ids = set()
for team, history in team_history.items():
	for _, game in history.iterrows():
		game_ids.add(game['GAME_ID'])

# send request for advanced stats
print('requesting advanced_stats...')
advanced_stats = {}
for i, game_id in enumerate(game_ids):
	bs = BoxScoreAdvancedV2(game_id=game_id)
	dfs = bs.get_data_frames()
	advanced_stats[game_id] = dfs
	print('%d/%d' % (i+1, len(game_ids)))
	if i % 50 == 0:
		save_obj(advanced_stats, 'advanced_stats')

'''
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
print('number of games: %d' % len(game_data))

print('collating data...')
for i, game_id in enumerate(game_data):
	print('game_id: %s [%d/%d]' % (game_id, i, len(game_data)))
	data = game_data[game_id]
	assert len(data) == 2
	team1_data, team2_data = data[0], data[1]

	stats = {}
	matchup = team1_data['MATCHUP']
	home_team, away_team = parse_matchup(matchup)
	stats['home_team'] = home_team
	stats['away_team'] = away_team
	stats['game_date'] = team1_data['GAME_DATE']

	lookup = {'W':'L', 'L':'W', None:None}
	win_loss = team1_data['WL']
	if home_team == team1_data['TEAM_ABBREVIATION']:
		home_win = win_loss
	else:
		home_win = lookup[win_loss]
	stats['home_win'] = home_win

	team1_stats = team1_data.drop(['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL'])
	team2_stats = team2_data.drop(['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL'])

	bs = BoxScoreAdvancedV2(game_id=game_id)
	dfs = bs.get_data_frames()
	player_stats, team_stats = dfs[0], dfs[1]
	stats['advanced_player_stats'] = player_stats
	stats['advanced_team_stats'] = team_stats

	if home_team == team1_data['TEAM_ABBREVIATION']:
		stats['home_team_stats'] = team1_stats
		stats['away_team_stats'] = team2_stats
	else:
		stats['home_team_stats'] = team2_stats
		stats['away_team_stats'] = team1_stats

	game_data[game_id] = stats

save_obj(game_data, 'game_data')
'''

'''
for team, history in team_history.items():
	print('scrolling through history of team: %s' % team)
	for index, game in history.iterrows():
		game_id = game['GAME_ID']
		print('fetching advanced stats for game id: %s' % game_id)

		if game_id in game_data:
			gd = game_data[game_id]
			assert gd['home_team_stats'] is None or gd['away_team_stats'] is None
			if gd['home_team_stats'] is not None:
				key = 'away_team_stats'
			else:
				key = 'home_team_stats'
			game_data[game_id][key] = game.drop(['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL'])
		else:
			stats = {}
			matchup = game['MATCHUP']
			home_team, away_team = parse_matchup(matchup)

			stats['home_team'] = home_team
			stats['away_team'] = away_team
			stats['game_date'] = game['GAME_DATE']

			lookup = {'W':'L', 'L':'W', None:None}
			win_loss = game['WL']
			if home_team == game['TEAM_ABBREVIATION']:
				home_win = win_loss
			home_win = lookup[win_loss]
			stats['home_win'] = home_win

			team_stats = game.drop(['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL'])
			bs = BoxScoreAdvancedV2(game_id=game_id)
			dfs = bs.get_data_frames()
			player_stats, team_stats = dfs[0], dfs[1]
			if home_team == game['TEAM_ABBREVIATION']:
				stats['home_team_stats'] = team_stats
				stats['away_team_stats'] = None
			else:
				stats['away_team_stats'] = team_stats
				stats['home_team_stats'] = None

			stats['advanced_player_stats'] = player_stats
			stats['advanced_team_stats'] = team_stats

			game_data[game_id] = stats

save_obj(game_data, 'game_data')
'''
