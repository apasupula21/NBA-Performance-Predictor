from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import requests
import datetime
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo

def get_player_id(name):
    player_dict = players.find_players_by_full_name(name)
    return player_dict[0]['id'] if player_dict else None

def get_player_team(player_name):
    player_dict = players.find_players_by_full_name(player_name)
    if player_dict:
        player_id = player_dict[0]['id']
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        team_id = info.get_data_frames()[0]['TEAM_ID'][0]
        return team_id
    else:
        raise ValueError(f"No player found with name {player_name}")

def get_all_game_logs(player_name, start_season='2010-11', end_season='2024-25'):
    player_id = get_player_id(player_name)
    if not player_id:
        raise ValueError(f"No player found with name {player_name}")

    all_logs = []
    start_year = int(start_season.split('-')[0])
    end_year = int(end_season.split('-')[0])

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        print(f"Fetching season {season}...")
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            df = gamelog.get_data_frames()[0]
            df['SEASON'] = season
            all_logs.append(df)
        except Exception as e:
            print(f"Failed to fetch data for season {season}: {e}")

    if not all_logs:
        raise ValueError("No data found for player across selected seasons.")

    return pd.concat(all_logs, ignore_index=True)

def load_team_defense_data(stats_folder='team_stats'):
    all_seasons = []

    for file in os.listdir(stats_folder):
        if file.endswith('.csv'):
            season = file.split('_')[-1].split('.')[0]
            df = pd.read_csv(os.path.join(stats_folder, file))
            df = df[pd.to_numeric(df['Rk'], errors='coerce').notna()] 

            df['Team'] = df['Team'].replace({
                'Charlotte Bobcats': 'Charlotte Hornets',
                'New Jersey Nets': 'Brooklyn Nets',
                'Seattle SuperSonics': 'Oklahoma City Thunder'
            })

            drtg_col = next((col for col in df.columns if 'Rtg' in col and 'D' in col), None)
            if not drtg_col:
                raise ValueError(f"No defensive rating column found in file: {file}")

            df['SEASON'] = f"{int(season)-1}-{season[-2:]}"
            df = df[['Team', drtg_col, 'SEASON']]
            df.rename(columns={'Team': 'TEAM', drtg_col: 'DEF_RTG'}, inplace=True)
            all_seasons.append(df)

    all_stats = pd.concat(all_seasons, ignore_index=True)
    all_stats['TEAM'] = all_stats['TEAM'].str.replace(r'\*', '', regex=True)
    all_stats['TEAM'] = all_stats['TEAM'].str.replace('LA Clippers', 'Los Angeles Clippers')
    all_stats['TEAM'] = all_stats['TEAM'].str.replace('Portland Trail Blazers', 'Portland')

    return all_stats

def clean_features(df):
    df = df.sort_values('GAME_DATE')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    df = df[['SEASON', 'GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'PLUS_MINUS', 'FGA', 'FGM', 'MIN']]
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
    df = df.dropna()

    df['PTS_avg3'] = df['PTS'].rolling(3).mean().shift(1)
    df['REB_avg3'] = df['REB'].rolling(3).mean().shift(1)
    df['AST_avg3'] = df['AST'].rolling(3).mean().shift(1)
    df['PLUSMINUS_avg3'] = df['PLUS_MINUS'].rolling(3).mean().shift(1)

    df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])

    team_abbrev_map = {
        'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
        'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
        'POR': 'Portland', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    }

    df['OPPONENT_FULL'] = df['OPPONENT'].map(team_abbrev_map)

    team_stats = load_team_defense_data()
    df = df.merge(
        team_stats,
        how='left',
        left_on=['OPPONENT_FULL', 'SEASON'],
        right_on=['TEAM', 'SEASON']
    )
    df.drop(columns=['TEAM'], inplace=True)
    df = df.dropna()

    return df

def train_eval(df):
    features = ['PTS_avg3', 'REB_avg3', 'AST_avg3', 'PLUSMINUS_avg3', 'DEF_RTG']
    targets = ['PTS', 'REB', 'AST']

    x = df[features]
    y = df[targets]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    print("\n--- Model Evaluation ---")
    for i, stat in enumerate(targets):
        print(f"{stat}: MAE = {mae[i]:.2f}, R² = {r2[i]:.2f}")

    return model, x_test, y_test, y_pred

def predict_next_game(model, df_clean):
    latest = df_clean.iloc[-1]
    input_features = latest[['PTS_avg3', 'REB_avg3', 'AST_avg3', 'PLUSMINUS_avg3', 'DEF_RTG']].values.reshape(1, -1)
    prediction = model.predict(input_features)[0]
    return {
        'Predicted PTS': round(prediction[0], 1),
        'Predicted REB': round(prediction[1], 1),
        'Predicted AST': round(prediction[2], 1)
    }
    
def predict_against_next_opponent(model, df_clean, opponent_name, team_stats_df):
    latest = df_clean.iloc[-1]
    season = latest['SEASON']
    
    def_rtg = get_team_def_rtg_by_name(opponent_name, season, team_stats_df)
    if def_rtg is None:
        raise ValueError(f"No DEF_RTG found for {opponent_name} in {season}")
    
    input_features = pd.DataFrame([{
        'PTS_avg3': latest['PTS_avg3'],
        'REB_avg3': latest['REB_avg3'],
        'AST_avg3': latest['AST_avg3'],
        'PLUSMINUS_avg3': latest['PLUSMINUS_avg3'],
        'DEF_RTG': def_rtg
    }])

    prediction = model.predict(input_features)[0]
    return {
        'Predicted PTS': round(float(prediction[0]), 1),
        'Predicted REB': round(float(prediction[1]), 1),
        'Predicted AST': round(float(prediction[2]), 1),
        'Against': opponent_name
    }

def predict_against_opponent(player_name, opponent_team_name):

    print(f"Running prediction for {player_name} vs {opponent_team_name}")
    df = get_all_game_logs(player_name)
    df_clean = clean_features(df)
    
    team_stats_df = load_team_defense_data()
    
    model, _, _, _ = train_eval(df_clean)
    
    latest = df_clean.iloc[-1]
    season = latest['SEASON']
    
    def_rtg = get_team_def_rtg_by_name(opponent_team_name, season, team_stats_df)
    if def_rtg is None:
        raise ValueError(f"No DEF_RTG found for {opponent_team_name} in {season}")
    
    input_features = pd.DataFrame([{
        'PTS_avg3': latest['PTS_avg3'],
        'REB_avg3': latest['REB_avg3'],
        'AST_avg3': latest['AST_avg3'],
        'PLUSMINUS_avg3': latest['PLUSMINUS_avg3'],
        'DEF_RTG': def_rtg
    }])
    
    prediction = model.predict(input_features)[0]
    
    return {
        'Player': player_name,
        'Opponent': opponent_team_name,
        'Predicted PTS': round(float(prediction[0]), 1),
        'Predicted REB': round(float(prediction[1]), 1),
        'Predicted AST': round(float(prediction[2]), 1)
    }

    
def fetch_schedule():
    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch schedule data.")
    
def find_next_game(schedule, team_id):
    today = datetime.datetime.now().date()
    games = schedule['leagueSchedule']['gameDates']
    for game_date in games:
        game_day = datetime.datetime.strptime(game_date['gameDate'], '%m/%d/%Y %H:%M:%S').date()
        if game_day >= today:
            for game in game_date['games']:
                if game['homeTeam']['teamId'] == team_id or game['awayTeam']['teamId'] == team_id:
                    opponent = game['awayTeam'] if game['homeTeam']['teamId'] == team_id else game['homeTeam']
                    return {
                        'game_date': game_day,
                        'opponent_team_id': opponent['teamId'],
                        'opponent_team_name': opponent['teamName']
                    }
    return None

def get_team_def_rtg_by_name(opponent_name, season, team_stats_df):
    match = team_stats_df[
        (team_stats_df['SEASON'] == season) &
        (team_stats_df['TEAM'].str.contains(opponent_name, case=False, na=False))
    ]
    if not match.empty:
        return match['DEF_RTG'].values[0]
    else:
        raise ValueError(f"No DEF_RTG found for {opponent_name} in {season}")


if __name__ == "__main__":
    player_name = "James Harden"

    schedule = fetch_schedule()
    team_id = get_player_team(player_name)
    next_game_info = find_next_game(schedule, team_id)

    if next_game_info:
        print(f"\nNext game for {player_name}: {next_game_info['game_date']} vs {next_game_info['opponent_team_name']}")

        df = get_all_game_logs(player_name)
        df_clean = clean_features(df)

        model, _, _, _ = train_eval(df_clean)

        team_stats_df = load_team_defense_data()

        prediction = predict_against_opponent(
            player_name,
            next_game_info['opponent_team_name']
        )

        print("\n--- Next Game Prediction ---")
        print(prediction)

    else:
        print("No upcoming games found.")



