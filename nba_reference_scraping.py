import requests, sys, re
import pandas as pd
from pyquery import PyQuery as pq
from requests.exceptions import HTTPError

# URL we're scraping from
url = 'https://www.basketball-reference.com/playoffs/series.html'

# Columns of our output dataframe
columns = ['Winner', 'Loser', 'Margin', 'Year']

def get_playoff_data():
    data = None
    try:
        # Ping the basketball-reference url
        print('Fetching playoff data')
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the returned html into formatted DataFrame
        data = parse_document(response.text)
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}') 
    except Exception as err:
        print(f'Other error occurred: {err}')
    else:
        print('Success!')
    finally:
        return data

def parse_document(html):
    d = pq(html)
    data = []

    # Iterate over every row and individually append each point
    for row in d('table#playoffs_series').find('tbody').find('tr').not_('.toggleable, .thead'):
        data.append(parse_point(row))

    # Our final dataframe
    return pd.DataFrame(data, columns=columns)

def parse_point(row):
    # Pattern for removign all non-alpha characters
    regex = re.compile('[^a-zA-Z\s]')
    row = pq(row)

    # Parse the data we want from the row
    winner = row('td[data-stat="winner"]').text()
    loser = row('td[data-stat="loser"]').text()
    winner_wins = row('td[data-stat="wins_winner"]').text()
    lower_wins = row('td[data-stat="wins_loser"]').text()
    year = row('th[data-stat="season"]').text()

    # Super advanced calulation here for win margin
    margin = int(winner_wins) - int(lower_wins)

    # Clean up our strings
    winner = regex.sub('', winner).strip()
    loser = regex.sub('', loser).strip()

    # Return our datapoint as ordered array of fields
    return [winner, loser, margin, year]




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please specify an output file location')
        sys.exit(1)

    file_loc = sys.argv[1]

    data = get_playoff_data()
    data.to_feather(file_loc)
