from os import path
import pickle

def save_obj(obj, filename, dirname='pickle_files'):
    with open(path.join(dirname, filename + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filename, dirname='pickle_files'):
    with open(path.join(dirname, filename + '.pkl'), 'rb') as f:
        return pickle.load(f)

def parse_matchup(matchup):
	if '@' in matchup: 
		away_team, home_team = matchup.split(' @ ')
	elif 'vs.' in matchup:
		home_team, away_team = matchup.split(' vs. ')
	else:
		return None, None
	return home_team, away_team