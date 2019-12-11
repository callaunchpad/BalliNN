import os, feather
from pbp_processing import DVOAEventType, OUTFILE

# Construct map from game cols (period, time, margin, momentum) to game-state (int)
# Construct map from game-state X DvoaEvent -> reward (int)
# Construct dataframe with (game_id, team_id, opp_team_id, game_state, reward)
# Group by game_state and normalize to calculate VOA
# While not converged:
#   group by opp_team_id
#   calculate defensive weights
#   reweight all rewards


def game_state(period, time, margin, momentum):
    pass

def reward(game_state, dvoa_event):
    pass
