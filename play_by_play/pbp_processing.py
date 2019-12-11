import os, feather, sys, re
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_FILE_LOC = os.path.join(ROOT_DIR, 'data_acquisition/games.feather')

# Default location to store all pre-processed data
OUTFLIE = os.path.join(ROOT_DIR, 'play_by_play/dvoa_processed.feather')

# Uncomment this line to use all of the pbp data
# PBP_FILE_LOC = os.path.join(ROOT_DIR, 'data_acquisition/out.feather')

# Uncomment this line to use a single game test sample of the pbp data
PBP_FILE_LOC = os.path.join(ROOT_DIR, 'data_acquisition/pbp_sample.feather')


def process_games(df):
    df = df[['GAME_ID', 'TEAM_ID', 'MATCHUP']].copy()
    df['IS_HOME'] = df['MATCHUP'].str.match(re.compile('\w{3} vs. \w{3}'))
    df = df.merge(df, on='GAME_ID', how='inner', suffixes=('_left', '_right'))
    df = df.loc[df['TEAM_ID_left'] != df['TEAM_ID_right']]

    def team_map(row):
        ret_row = pd.Series()
        home_team = row['TEAM_ID_left'] if row['IS_HOME_left'] else row['TEAM_ID_right']
        away_team = row['TEAM_ID_left'] if not row['IS_HOME_left'] else row['TEAM_ID_right']
        ret_row['HOME_TEAM_ID'] = home_team
        ret_row['AWAY_TEAM_ID'] = away_team
        ret_row['GAME_ID'] = row['GAME_ID']
        return ret_row
    
    return df.apply(team_map, axis=1, result_type='expand').drop_duplicates(subset='GAME_ID', keep='first')




class EventMsgType:
    FIELD_GOAL_MADE = 1
    FIELD_GOAL_MISSED = 2
    FREE_THROW_ATTEMP = 3
    REBOUND = 4
    TURNOVER = 5
    FOUL = 6
    VIOLATION = 7
    SUBSTITUTION = 8
    TIMEOUT = 9
    JUMP_BALL = 10
    EJECTION = 11
    PERIOD_BEGIN = 12
    PERIOD_END = 13

class DVOAEventType:
    MISS_DER_REB = 1
    TWO_PTS = 2
    THREE_PTS = 3
    LB_TURNOVER = 4
    DB_TURNOVER = 5
    MISS_OFF_REB = 6
    TWO_PTS_AND_ONE = 7
    THREE_PTS_AND_ONE = 8
    SHOOTING_FOUL = 9


def parse_score(df):
    p = re.compile('(\d+) - (\d+)')
    def parse_score_row(row):
        score_str = row['SCORE']
        matches = p.search(score_str) if score_str else None
        matches = matches.groups() if matches else None
        visitor_score, home_score = (-1, -1) if not matches else (int(matches[0]), int(matches[1]))
        row['VISITOR_SCORE'] = visitor_score
        row['HOME_SCORE'] = home_score
        return row
    return df.apply(parse_score_row, axis=1, result_type='expand')

def calculate_momentum(df):
    MOMENTUM_WINDOW = 10
    momentum = []
    curr_scores = []
    prev_margin = 0
    prev_score = ''
    for _, row in df.iterrows():
        if prev_score != row['SCORE']:
            curr_score = row['SCOREMARGIN'] - prev_margin
            prev_margin = row['SCOREMARGIN']
            curr_scores.append(curr_score)
            while sum(map(abs, curr_scores)) > MOMENTUM_WINDOW:
                curr_scores.pop(0)
        curr_momentum = sum(curr_scores)
        momentum.append(curr_momentum)
    df['MOMENTUM'] = pd.Series(momentum)
    return df

def calc_dvoa_events(df):
    size = len(df)
    list_of_rows = []
    for i, row in df.iterrows():
        # Row that we will be building in our loop
        curr_row = {}
        # represents encoding of type of offensive event 
        dvoa_event = None
        # Flag: 0 => no event, 1 => home team event, 2 => away team event
        team = 0

        # Copy over all columns
        for col in row.index:
            curr_row[col] = row[col]

        ### Home team dvoa events


        # Made shot
        if row['EVENTMSGTYPE'] == EventMsgType.FIELD_GOAL_MADE and row['HOMEDESCRIPTION']:
            # Made 3pt shot
            if row['EVENTMSGACTIONTYPE'] == 1 or row['EVENTMSGACTIONTYPE'] == 79 or row['EVENTMSGACTIONTYPE'] == 80:
                # And one
                if i + 1 < size and df['EVENTMSGTYPE'][i + 1] == EventMsgType.FOUL and df['EVENTMSGACTIONTYPE'][i + 1] == 2 and df['VISITORDESCRIPTION'][i + 1]:
                    dvoa_event = DVOAEventType.THREE_PTS_AND_ONE
                # Not and one
                else:
                    dvoa_event = DVOAEventType.THREE_PTS
            # Made 2 pt shot
            else:
                # And one
                if i + 1 < size and df['EVENTMSGTYPE'][i + 1] == EventMsgType.FOUL and df['EVENTMSGACTIONTYPE'][i + 1] == 2 and df['VISITORDESCRIPTION'][i + 1]:
                    dvoa_event = DVOAEventType.TWO_PTS_AND_ONE
                # Not and one
                else:
                    dvoa_event = DVOAEventType.TWO_PTS
            team = 1
        # Missed shot
        if row['EVENTMSGTYPE'] == EventMsgType.FIELD_GOAL_MISSED and row['HOMEDESCRIPTION'] and not 'block' in row['HOMEDESCRIPTION'].lower():
            # Defensive rebound
            if i + 1 < size and df['EVENTMSGTYPE'][i+1] == EventMsgType.REBOUND and df['VISITORDESCRIPTION'][i+1] and 'rebound' in df['VISITORDESCRIPTION'][i+1].lower():
                dvoa_event = DVOAEventType.MISS_DER_REB
            # Offensive rebound
            else:
                dvoa_event = DVOAEventType.MISS_OFF_REB
            team = 1
        # Turnover
        if row['EVENTMSGTYPE'] == EventMsgType.TURNOVER and row['HOMEDESCRIPTION'] and 'turnover' in row['HOMEDESCRIPTION'].lower():
            # Live ball turnover
            if row['VISITORDESCRIPTION'] and 'steal' in row['VISITORDESCRIPTION'].lower():
                dvoa_event = DVOAEventType.LB_TURNOVER
            # Dead ball turnover
            else:
                dvoa_event = DVOAEventType.DB_TURNOVER
            team = 1
        # Foul 
        if row['EVENTMSGTYPE'] == EventMsgType.FOUL and row['EVENTMSGACTIONTYPE'] == 2 and row['VISITORDESCRIPTION']:
            # Verify this wasn't an and-one to avoid double counting
            if not (i - 1 >= 0 and df['EVENTMSGTYPE'][i - 1] == 1 and df['HOMEDESCRIPTION'][i - 1]):
                dvoa_event = DVOAEventType.SHOOTING_FOUL
                team = 1
        # Goaltending
        if row['EVENTMSGTYPE'] == EventMsgType.VIOLATION and row['EVENTMSGACTIONTYPE'] == 2 and row['VISITORDESCRIPTION']:
            # The goaltending results in 3 points
            if i - 1 < size and row['HOME_SCORE'] - df['HOME_SCORE'][i - 1] == 3:
                dvoa_event = DVOAEventType.THREE_PTS
            else:
                dvoa_event = DVOAEventType.TWO_PTS
            team = 1


        ### Away team dvoa events

        
        # Made shot
        if row['EVENTMSGTYPE'] == EventMsgType.FIELD_GOAL_MADE and row['VISITORDESCRIPTION']:
            # Made 3pt shot
            if row['EVENTMSGACTIONTYPE'] == 1 or row['EVENTMSGACTIONTYPE'] == 79 or row['EVENTMSGACTIONTYPE'] == 80:
                # And one
                if i + 1 < size and df['EVENTMSGTYPE'][i + 1] == EventMsgType.FOUL and df['EVENTMSGACTIONTYPE'][i + 1] == 2 and df['HOMEDESCRIPTION'][i + 1]:
                    dvoa_event = DVOAEventType.THREE_PTS_AND_ONE
                # Not and one
                else:
                    dvoa_event = DVOAEventType.THREE_PTS
            # Made 2 pt shot
            else:
                # And one
                if i + 1 < size and df['EVENTMSGTYPE'][i + 1] == EventMsgType.FOUL and df['EVENTMSGACTIONTYPE'][i + 1] == 2 and df['HOMEDESCRIPTION'][i + 1]:
                    dvoa_event = DVOAEventType.TWO_PTS_AND_ONE
                # Not and one
                else:
                    dvoa_event = DVOAEventType.TWO_PTS
            team = 2
        # Missed shot
        if row['EVENTMSGTYPE'] == EventMsgType.FIELD_GOAL_MISSED and row['VISITORDESCRIPTION'] and not 'block' in row['VISITORDESCRIPTION'].lower():
            # Defensive rebound
            if i + 1 < size and df['EVENTMSGTYPE'][i+1] == EventMsgType.REBOUND and df['HOMEDESCRIPTION'][i+1] and 'rebound' in df['HOMEDESCRIPTION'][i+1].lower():
                dvoa_event = DVOAEventType.MISS_DER_REB
            # Offensive rebound
            else:
                dvoa_event = DVOAEventType.MISS_OFF_REB
            team = 2
        # Turnover
        if row['EVENTMSGTYPE'] == EventMsgType.TURNOVER and row['VISITORDESCRIPTION'] and 'turnover' in row['VISITORDESCRIPTION'].lower():
            # Live ball turnover
            if row['HOMEDESCRIPTION'] and 'steal' in row['HOMEDESCRIPTION'].lower():
                dvoa_event = DVOAEventType.LB_TURNOVER
            # Dead ball turnover
            else:
                dvoa_event = DVOAEventType.DB_TURNOVER
            team = 2
        # Foul 
        if row['EVENTMSGTYPE'] == EventMsgType.FOUL and row['EVENTMSGACTIONTYPE'] == 2 and row['HOMEDESCRIPTION']:
            # Verify this wasn't an and-one to avoid double counting
            if not (i - 1 >= 0 and df['EVENTMSGTYPE'][i - 1] == 1 and df['VISITORDESCRIPTION'][i - 1]):
                dvoa_event = DVOAEventType.SHOOTING_FOUL
                team = 2
        # Goaltending
        if row['EVENTMSGTYPE'] == EventMsgType.VIOLATION and row['EVENTMSGACTIONTYPE'] == 2 and row['HOMEDESCRIPTION']:
            # The goaltending results in 3 points
            if i - 1 < size and row['VISITOR_SCORE'] - df['VISITOR_SCORE'][i - 1] == 3:
                dvoa_event = DVOAEventType.THREE_PTS
            else:
                dvoa_event = DVOAEventType.TWO_PTS
            team = 2

        curr_row['dvoa_event'] = dvoa_event

        # Get the team id corresponding to the team that performed the event, if any
        if team:
            curr_row['dvoa_team_id'] = row['HOME_TEAM_ID'] if team == 1 else row['AWAY_TEAM_ID']
            curr_row['dvoa_opp_team_id'] = row['AWAY_TEAM_ID'] if team == 1 else row['HOME_TEAM_ID']

        # Negate score differentials if away team event
        if team == 2:
            curr_row['MOMENTUM'] *= -1
            curr_row['SCOREMARGIN'] *= -1

        # Add our new row to list
        list_of_rows.append(curr_row)
    
    return pd.DataFrame(list_of_rows)


def process_game(pbp_data):
    # Propogate current scores forward for non-scoring plays
    pbp_data[['SCORE', 'SCOREMARGIN']] = pbp_data[['SCORE', 'SCOREMARGIN']].fillna(method='ffill')
    pbp_data['SCORE'] = pbp_data['SCORE'].fillna('0 - 0')
    pbp_data['SCOREMARGIN'] = pbp_data['SCOREMARGIN'].fillna(0)
    pbp_data['SCOREMARGIN'] = pbp_data['SCOREMARGIN'].replace('TIE', 0)
    pbp_data = pbp_data.astype({'SCOREMARGIN': 'int32'})

    # Convert score strings into ints
    pbp_data = parse_score(pbp_data)

    # Calculate running 'momentum' of the game
    pbp_data = calculate_momentum(pbp_data)

    # Process all dvoa events
    dvoa_df = calc_dvoa_events(pbp_data)
    return dvoa_df

def main(outfile):
    # Read in our data
    pbp_data = feather.read_dataframe(PBP_FILE_LOC)
    games = feather.read_dataframe(GAMES_FILE_LOC)

    # Determine the home and away team for each game
    games = process_games(games)

    # Drop unnecessary data
    pbp_data = pbp_data.drop(['NEUTRALDESCRIPTION', 'PERSON1TYPE',
       'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ID', 'PLAYER1_TEAM_CITY',
       'PLAYER1_TEAM_NICKNAME', 'PLAYER1_TEAM_ABBREVIATION', 'PERSON2TYPE',
       'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ID', 'PLAYER2_TEAM_CITY',
       'PLAYER2_TEAM_NICKNAME', 'PLAYER2_TEAM_ABBREVIATION', 'PERSON3TYPE',
       'PLAYER3_ID', 'PLAYER3_NAME', 'PLAYER3_TEAM_ID', 'PLAYER3_TEAM_CITY',
       'PLAYER3_TEAM_NICKNAME', 'PLAYER3_TEAM_ABBREVIATION',
       'VIDEO_AVAILABLE_FLAG', 'WCTIMESTRING'], axis=1)

    # Merge game data with pbp data
    pbp_data = pbp_data.merge(games, on='GAME_ID', how='inner')

    # Calculate all dvoa events one game at a time
    all_dfs = []
    for grp in pbp_data.groupby('GAME_ID'):
        game_id, df = grp
        print('Processing dvoa events for game {0}'.format(game_id))
        dvoa_df = process_game(pbp_data)
        all_dfs.append(dvoa_df)
    # Vertically stack all our single game dataframes into one big-un
    final_dvoa_df = pd.concat(all_dfs)

    # Write out our dataframe to the specified save location
    final_dvoa_df.to_feather(outfile)




if __name__ == '__main__':
    outfile = OUTFLIE
    
    if len(sys.argv) > 1:
        outfile = sys.argv[1]
    
    main(outfile)



    



