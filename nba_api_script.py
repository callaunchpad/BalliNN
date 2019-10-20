from nba_api.stats.endpoints import teamdashboardbyyearoveryear as dash
from nba_api.stats.endpoints import teamyearbyyearstats as team_stats_endpoint
from nba_api.stats.static import teams

import pandas as pd
import time, sys

def get_year_by_year_data(outfile):
    all_teams = teams.get_teams()

    # Contains list of (id, name) tuples
    team_data = [(t['id'], t['full_name']) for t in all_teams]


    # Normalize stats by dividing by number of games
    params = {
        "per_mode_simple"   : "PerGame"
    }

    all_data = []

    for team in team_data:
        team_id, team_name = team

        print('fetching data for team ', team_name)
        team_stats_dataset = team_stats_endpoint.TeamYearByYearStats(team_id=team_id, **params)
        team_stats = team_stats_dataset.team_stats.get_data_frame()
        all_data.append(team_stats)

        # So we don't get throttled
        time.sleep(1)

    # Combine everyting into one bigass DF and compress it out 'outfile'
    full_data_frame = pd.concat(all_data, ignore_index=True)
    full_data_frame.to_feather(outfile)

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 1:
        print('Please specify file to dump data')
        sys.exit(1)
    
    fname = sys.argv[1]
    get_year_by_year_data(fname)



