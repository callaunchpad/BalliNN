from nba_api.stats.endpoints import playbyplayv2 as pbp
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import Season
from nba_api.stats.library.parameters import SeasonType
from nba_api.stats.static import teams

import pandas as pd

import sys, pickle

GAMES_OUTFILE = 'games.feather'
FAILURES_OUTFILE = 'failures.p'

all_teams = teams.get_teams()
all_team_ids = [t['id'] for t in all_teams]

headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://stats.nba.com/',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
    }

def get_game_ids_for_season(season):
    gamefinder = leaguegamefinder.LeagueGameFinder(headers=headers, season_nullable=season,
                            season_type_nullable=SeasonType.regular)
    games_df = gamefinder.get_data_frames()
    games = games_df[0]
    games.to_feather(GAMES_OUTFILE)
    return list(set(games['GAME_ID'].tolist()))

def get_pbp_df(game_id):
    return pbp.PlayByPlayV2(headers=headers, game_id=game_id).get_data_frames()[0]


def get_all_pbp_data_for_season(season):
    game_ids = get_game_ids_for_season(season)
    game_ids = list(game_ids)[:2]
    failures = set()
    all_dfs = []
    MAX_TRIES = 5

    for game in game_ids:
        curr_try = 0
        print('Requesting game {0}'.format(game))
        while True:
            try:
                data = get_pbp_df(game)
            except Exception:
                print('Request for game {0} failed, retrying...'.format(game))
                curr_try += 1
                if curr_try == MAX_TRIES:
                    print('Game {0} failed {1} times. Aborting'.format(game, MAX_TRIES))
                    failures.add(game)
                    break
            else:
                all_dfs.append(data)
                break

    with open(FAILURES_OUTFILE, 'wb') as f:
        pickle.dump(failures, f)

    merged_df = pd.concat(all_dfs)
    return merged_df.reset_index(drop=True)

if __name__ == '__main__':
    args = sys.argv
    assert len(args) == 3
    season = args[1]
    save_loc = args[2]
    data = get_all_pbp_data_for_season(season)
    data.to_feather(save_loc)