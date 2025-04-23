from nba_api.stats.endpoints import playergamelog, commonplayerinfo, commonteamroster
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.static import players, teams
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import requests
import datetime
import json
import time
    
CACHE_FILE = "team_roster_cache.json"
CACHE_DURATION = 60 * 60 * 24

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.nba.com/',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
})

TEAM_ABBREV_TO_ID = {
    'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751,
    'CHA': 1610612766, 'CHI': 1610612741, 'CLE': 1610612739,
    'DAL': 1610612742, 'DEN': 1610612743, 'DET': 1610612765,
    'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
    'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763,
    'MIA': 1610612748, 'MIL': 1610612749, 'MIN': 1610612750,
    'NOP': 1610612740, 'NYK': 1610612752, 'OKC': 1610612760,
    'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759,
    'TOR': 1610612761, 'UTA': 1610612762, 'WAS': 1610612764
}

TEAM_ABBREV_TO_FULL = {
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
    
API_KEY = "dd7jmZQzRLLrBl9dX7FzTYgRkR3uDEvzffH7btqW"
url = f"https://api.sportradar.us/nba/trial/v8/en/league/injuries.json?api_key={API_KEY}"

response = session.get(url, timeout=10)
print("Status:", response.status_code)

if response.status_code == 200:
    data = response.json()
else:
    print("Error:", response.text)

def get_player_id(name):
    player_dict = players.find_players_by_full_name(name)
    return player_dict[0]['id'] if player_dict else None

def update_team_roster_cache():
    print("Updating team roster cache...")
    rosters = {}
    failed_teams = []

    for team in teams.get_teams():
        team_id = team['id']
        team_name = team['full_name']
        try:
            print(f"Fetching roster for {team_name}...")
            roster_data = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
            for player in roster_data['PLAYER']:
                rosters[player] = team_id
        except Exception as e:
            print(f"Failed to fetch roster for {team_name}: {e}")
            failed_teams.append(team)

        time.sleep(1.5)  # 🧘 slow down to avoid getting blocked

    # Retry once for the teams that failed
    if failed_teams:
        print("\nRetrying failed teams...\n")
        for team in failed_teams:
            team_id = team['id']
            team_name = team['full_name']
            try:
                print(f"Retrying {team_name}...")
                roster_data = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
                for player in roster_data['PLAYER']:
                    rosters[player] = team_id
            except Exception as e:
                print(f"Still failed for {team_name}: {e}")
            time.sleep(1.5)

    with open(CACHE_FILE, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "rosters": rosters
        }, f)

    print("\nRoster cache update complete.")

def load_team_roster_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
            if time.time() - data["timestamp"] < CACHE_DURATION:
                return data["rosters"]
    update_team_roster_cache()
    return load_team_roster_cache()

def get_player_team(player_name):
    rosters = load_team_roster_cache()
    if player_name in rosters:
        return rosters[player_name]
    else:
        raise ValueError(f"Team ID not found for player: {player_name}")


def get_all_game_logs(player_name, start_season='2010-11', end_season='2024-25'):
    player_id = get_player_id(player_name)
    all_logs = []
    for year in range(int(start_season[:4]), int(end_season[:4]) + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        print(f"Fetching season {season}...")
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            df = gamelog.get_data_frames()[0]
            if not df.empty and 'MATCHUP' in df.columns:
                df['SEASON'] = season
                all_logs.append(df)
            else:
                print(f"Skipping season {season}: no MATCHUP column or empty log.")
        except Exception as e:
            print(f"Failed for {season}: {e}")
    return pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()

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
            df['SEASON'] = f"{int(season)-1}-{season[-2:]}"
            df = df[['Team', drtg_col, 'SEASON']]
            df.rename(columns={'Team': 'TEAM', drtg_col: 'DEF_RTG'}, inplace=True)
            all_seasons.append(df)
    all_stats = pd.concat(all_seasons, ignore_index=True)
    all_stats['TEAM'] = all_stats['TEAM'].str.replace(r'\*', '', regex=True)
    all_stats['TEAM'] = all_stats['TEAM'].str.replace('LA Clippers', 'Los Angeles Clippers')
    all_stats['TEAM'] = all_stats['TEAM'].str.replace('Portland Trail Blazers', 'Portland')
    return all_stats

def structure_injuries(raw_data):
    structured = {}

    if "teams" not in raw_data:
        raise ValueError("Invalid data format: 'teams' key not found.")

    for team_entry in raw_data["teams"]:
        team_name = f"{team_entry['market']} {team_entry['name']}"
        structured[team_name] = {}

        for player in team_entry.get("players", []):
            if "injuries" in player and player["injuries"]:
                latest_injury = player["injuries"][-1]
                status = latest_injury.get("status", "Unknown")
                structured[team_name][player["full_name"]] = status

    return structured

def get_teammate_injuries(player_name, structured_data):
    found_team = None
    for team_name, players_status in structured_data.items():
        if player_name in players_status:
            found_team = team_name
            break

    if not found_team:
        for team_name, players_status in structured_data.items():
            if player_name not in players_status:
                found_team = team_name
                break

    if not found_team:
        print(f"Warning: Could not find {player_name} in injury data. Skipping injury teammate extraction.")
        return [], []

    out_statuses = {"Out", "Out For Season"}
    d2d_statuses = {"Day To Day", "Questionable", "Doubtful", "Probable"}

    out_players = []
    d2d_players = []

    for teammate, status in structured_data[found_team].items():
        if teammate == player_name:
            continue
        if status in out_statuses:
            out_players.append(teammate)
        elif status in d2d_statuses:
            d2d_players.append(teammate)

    return out_players, d2d_players

def print_cached_rosters():
    if not os.path.exists(CACHE_FILE):
        print("No cache file found.")
        return

    with open(CACHE_FILE, "r") as f:
        data = json.load(f)

    rosters = data.get("rosters", {})
    print("\n--- Cached Rosters ---")
    for player, team_id in sorted(rosters.items()):
        print(f"{player}: {team_id}")
            
def get_player_game_dates(player_name, season='2023-24'):
    player_id = get_player_id(player_name)
    logs = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
    return set(logs['GAME_DATE'])

def build_availability_table(target_player, teammates, season='2023-24'):
    target_dates = get_player_game_dates(target_player, season)
    table = pd.DataFrame({'GAME_DATE': sorted(target_dates)})

    for teammate in teammates:
        teammate_dates = get_player_game_dates(teammate, season)
        table[teammate] = table['GAME_DATE'].apply(lambda d: 1 if d in teammate_dates else 0)

    return table

def clean_features(df):
    df = df.sort_values('GAME_DATE')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    df = df[['SEASON', 'GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'PLUS_MINUS', 'FGA', 'FGM', 'MIN']]
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
    df.dropna(inplace=True)

    df['PTS_avg3'] = df['PTS'].rolling(3).mean().shift(1)
    df['REB_avg3'] = df['REB'].rolling(3).mean().shift(1)
    df['AST_avg3'] = df['AST'].rolling(3).mean().shift(1)
    df['PLUSMINUS_avg3'] = df['PLUS_MINUS'].rolling(3).mean().shift(1)

    df['PTS_season_avg'] = df['PTS'].expanding().mean().shift(1)
    df['REB_season_avg'] = df['REB'].expanding().mean().shift(1)
    df['AST_season_avg'] = df['AST'].expanding().mean().shift(1)

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

    def get_opponent_avg_stat(row, stat):
        past_games = df[(df['OPPONENT_FULL'] == row['OPPONENT_FULL']) & (df['GAME_DATE'] < row['GAME_DATE'])]
        return past_games[stat].mean() if not past_games.empty else None

    df['PTS_vs_opp_avg'] = df.apply(lambda row: get_opponent_avg_stat(row, 'PTS'), axis=1)
    df['REB_vs_opp_avg'] = df.apply(lambda row: get_opponent_avg_stat(row, 'REB'), axis=1)
    df['AST_vs_opp_avg'] = df.apply(lambda row: get_opponent_avg_stat(row, 'AST'), axis=1)

    team_stats = load_team_defense_data()
    df = df.merge(team_stats, how='left', left_on=['OPPONENT_FULL', 'SEASON'], right_on=['TEAM', 'SEASON'])
    df.drop(columns=['TEAM'], inplace=True)
    df.dropna(inplace=True)

    return df

def train_eval(df):
    features = [
        'PTS_avg3', 'PTS_season_avg', 'PTS_vs_opp_avg',
        'REB_avg3', 'REB_season_avg', 'REB_vs_opp_avg',
        'AST_avg3', 'AST_season_avg', 'AST_vs_opp_avg',
        'PLUSMINUS_avg3', 'DEF_RTG'
    ]
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

    model.feature_names_in_ = x.columns.tolist()
    return model, x_test, y_test, y_pred

def get_team_def_rtg_by_name(opponent_name, season, team_stats_df):
    match = team_stats_df[
        (team_stats_df['SEASON'] == season) &
        (team_stats_df['TEAM'].str.contains(opponent_name, case=False, na=False))
    ]
    if not match.empty:
        return match['DEF_RTG'].values[0]
    else:
        return None


def predict_against_opponent(player_name, opponent_name, df_clean, model, team_stats_df):
    latest = df_clean.iloc[-1]
    season = latest['SEASON']
    def_rtg = get_team_def_rtg_by_name(opponent_name, season, team_stats_df)
    if def_rtg is None:
        raise ValueError(f"No DEF_RTG found for {opponent_name} in {season}")

    input_features_dict = {name: latest[name] for name in model.feature_names_in_ if name != 'DEF_RTG'}
    input_features_dict['DEF_RTG'] = def_rtg

    input_features = pd.DataFrame([input_features_dict])
    prediction = model.predict(input_features)[0]

    return {
        'Player': player_name,
        'Opponent': opponent_name,
        'Predicted PTS': round(float(prediction[0]), 1),
        'Predicted REB': round(float(prediction[1]), 1),
        'Predicted AST': round(float(prediction[2]), 1)
    }

def filter_games_without_all_teammates(df, absent_teammates, season='2024-25'):
    print(f"Filtering games where all of {absent_teammates} were absent...")

    teammate_absent_sets = []

    for teammate in absent_teammates:
        try:
            teammate_id = get_player_id(teammate)
            logs = playergamelog.PlayerGameLog(player_id=teammate_id, season=season).get_data_frames()[0]
            logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE'])
            teammate_dates = set(logs['GAME_DATE'])
            all_game_dates = pd.to_datetime(df['GAME_DATE'])
            absent_dates = set(all_game_dates) - teammate_dates
            teammate_absent_sets.append(absent_dates)
        except Exception as e:
            print(f"Failed to fetch game log for teammate {teammate}: {e}")
            return df

    if teammate_absent_sets:
        common_absent_dates = set.intersection(*teammate_absent_sets)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
        filtered_df = df[df['GAME_DATE'].isin(common_absent_dates)]
        print(f"Filtered down to {len(filtered_df)} games where all teammates were out.")
        return filtered_df
    else:
        return df


def fetch_schedule():
    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None


def find_next_game(schedule, team_id):
    today = datetime.datetime.now().date()
    games = schedule['leagueSchedule']['gameDates']
    for game_date in games:
        game_day = datetime.datetime.strptime(game_date['gameDate'], '%m/%d/%Y %H:%M:%S').date()
        if game_day >= today:
            for game in game_date['games']:
                if str(game['homeTeam']['teamId']) == str(team_id) or str(game['awayTeam']['teamId']) == str(team_id):
                    opponent = game['awayTeam'] if str(game['homeTeam']['teamId']) == str(team_id) else game['homeTeam']
                    return {
                        'game_date': game_day,
                        'opponent_team_id': opponent['teamId'],
                        'opponent_team_name': opponent['teamName']
                    }
    return None

def get_team_abbrev_from_logs_or_cache(player_id):
    try:
        gamelog_df = playergamelog.PlayerGameLog(player_id=player_id, season='2023-24').get_data_frames()[0]
        if 'MATCHUP' in gamelog_df.columns and not gamelog_df.empty:
            first_matchup = gamelog_df.iloc[0]['MATCHUP']
            team_abbrev = first_matchup.split(' ')[0]
            return team_abbrev
    except Exception as e:
        print(f"Failed to get team abbreviation from logs: {e}")

    rosters = load_team_roster_cache()
    player_name = None

    for p in players.get_active_players():
        if p["id"] == player_id:
            player_name = p["full_name"]
            break

    if player_name and player_name in rosters:
        team_id = rosters[player_name]
        for abbrev, tid in TEAM_ABBREV_TO_ID.items():
            if str(tid) == str(team_id):
                return abbrev

    raise ValueError(f"Could not find team abbreviation for player ID {player_id}.")


def get_out_teammates(player_name, structured_data):
    player_id = get_player_id(player_name)
    
    try:
        team_abbrev = get_team_abbrev_from_logs_or_cache(player_id)
    except Exception as e:
        raise ValueError(f"Failed to get team abbreviation: {e}")

    full_team_name = TEAM_ABBREV_TO_FULL.get(team_abbrev)
    if not full_team_name:
        raise ValueError(f"Could not find full team name for {team_abbrev} in TEAM_ABBREV_TO_FULL.")

    out_statuses = {"Out", "Out For Season"}
    out_teammates = []

    if full_team_name not in structured_data:
        print(f"Warning: {full_team_name} not found in injury data structure.")
        return []

    for teammate, status in structured_data[full_team_name].items():
        if teammate != player_name and status in out_statuses:
            out_teammates.append(teammate)

    return out_teammates


if __name__ == "__main__":
    player_name = "Tyler Herro"

    df = get_all_game_logs(player_name)

    if 'MATCHUP' not in df.columns:
        print("MATCHUP column missing — retrying game log fetch.")
        df = get_all_game_logs(player_name)
        if 'MATCHUP' not in df.columns:
            print(df.head())
            raise ValueError("MATCHUP column still missing after retry. Check API limits or connection.")

    structured_injuries = structure_injuries(data)
    print("Available teams in injury data:", list(structured_injuries.keys()))

    out_teammates = get_out_teammates(player_name, structured_injuries)

    df_filtered = df
    df_filtered = filter_games_without_all_teammates(df_filtered, out_teammates)

    try:
        team_id = get_player_team(player_name)
    except ValueError as e:
        print(f"Could not get team ID for {player_name} from cache: {e}")
        player_id = get_player_id(player_name)
        team_abbrev = get_team_abbrev_from_logs_or_cache(player_id)
        team_id = TEAM_ABBREV_TO_ID.get(team_abbrev)
        if not team_id:
            raise ValueError(f"Team ID could not be resolved for {team_abbrev}")

    df_clean = clean_features(df_filtered)
    if df_clean.empty:
        raise ValueError("No clean data available for training.")

    team_stats_df = load_team_defense_data()
    model, _, _, _ = train_eval(df_clean)

    schedule = fetch_schedule()
    next_game_info = find_next_game(schedule, team_id)

    if next_game_info:
        print(f"\nNext game: {next_game_info['game_date']} vs {next_game_info['opponent_team_name']}")
        prediction = predict_against_opponent(
            player_name,
            next_game_info['opponent_team_name'],
            df_clean,
            model,
            team_stats_df
        )
        print("\n--- Next Game Prediction ---")
        print(prediction)
    else:
        print("No upcoming games found.")

