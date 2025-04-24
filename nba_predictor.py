from nba_api.stats.endpoints import playergamelog, commonplayerinfo, commonteamroster
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.static import players, teams
import pandas as pd
import numpy as np
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

try:
    response = session.get(url, timeout=10)
    response.raise_for_status()  # Raise an exception for bad status codes
    data = response.json()
    print("Successfully fetched injury data")
except requests.exceptions.RequestException as e:
    print(f"Error fetching injury data: {e}")
    data = {"teams": []}  # Initialize with empty data structure

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
    
    # First try to get current season data with retries
    max_retries = 3
    current_season = '2024-25'
    
    for attempt in range(max_retries):
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=current_season,
                season_type_all_star='Regular Season',
                timeout=30
            )
            df = gamelog.get_data_frames()[0]
            if not df.empty and 'MATCHUP' in df.columns:
                df['SEASON'] = current_season
                all_logs.append(df)
                print(f"Successfully fetched {len(df)} games for {current_season}")
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                print(f"Failed to fetch current season data after {max_retries} attempts")
    
    # Combine all logs and sort by date
    if all_logs:
        combined_df = pd.concat(all_logs, ignore_index=True)
        combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'])
        combined_df = combined_df.sort_values('GAME_DATE')
        
        # Print debug info
        print(f"\nTotal games fetched: {len(combined_df)}")
        print(f"Date range: {combined_df['GAME_DATE'].min().strftime('%Y-%m-%d')} to {combined_df['GAME_DATE'].max().strftime('%Y-%m-%d')}")
        
        return combined_df
    else:
        return pd.DataFrame()

def load_team_defense_data(stats_folder='team_stats'):
    if not os.path.exists(stats_folder):
        print(f"Creating {stats_folder} directory...")
        os.makedirs(stats_folder)
        print(f"Please add team statistics Excel files to the {stats_folder} directory")
        return pd.DataFrame(columns=['TEAM', 'DEF_RTG', 'SEASON', 'PACE', 'OFF_RTG', 'EFG_PCT', 'TOV_PCT', 'OREB_PCT', 'FT_RATE'])

    all_seasons = []
    for file in os.listdir(stats_folder):
        if file.endswith(('.xlsx', '.csv')):
            try:
                season = file.split('_')[-1].split('.')[0]
                if file.endswith('.xlsx'):
                    df = pd.read_excel(os.path.join(stats_folder, file))
                else:
                    df = pd.read_csv(os.path.join(stats_folder, file))
                
                # Clean up the data
                df = df[df['Rk'].notna()]  # Keep only rows with valid Rk
                df['Team'] = df['Team'].replace({
                    'LA Clippers': 'Los Angeles Clippers',
                    'Portland Trl': 'Portland Trail Blazers',
                    'Brooklyn': 'Brooklyn Nets',
                    'Golden State': 'Golden State Warriors',
                    'LA Lakers': 'Los Angeles Lakers'
                })

                # Create a new dataframe with only the columns we need
                new_df = pd.DataFrame()
                new_df['TEAM'] = df['Team']
                new_df['DEF_RTG'] = df['DRtg']
                new_df['OFF_RTG'] = df['ORtg']
                new_df['PACE'] = df['Pace']
                new_df['EFG_PCT'] = df['eFG%']
                # Convert percentages from decimal to percentage format if needed
                new_df['TOV_PCT'] = df['TOV%'].apply(lambda x: x if x <= 100 else x/100)
                new_df['OREB_PCT'] = df['ORB%'].apply(lambda x: x if x <= 100 else x/100)
                new_df['FT_RATE'] = df['FT/FGA']
                new_df['SEASON'] = f"{int(season)-1}-{season[-2:]}"

                # Verify percentage ranges
                for col in ['TOV_PCT', 'OREB_PCT']:
                    if new_df[col].max() > 100 or new_df[col].min() < 0:
                        print(f"Warning: {col} has values outside 0-100 range: {new_df[col].min():.1f} to {new_df[col].max():.1f}")

                all_seasons.append(new_df)
                print(f"Successfully processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

    if not all_seasons:
        print(f"No valid team statistics found in {stats_folder}")
        return pd.DataFrame(columns=['TEAM', 'DEF_RTG', 'SEASON', 'PACE', 'OFF_RTG', 'EFG_PCT', 'TOV_PCT', 'OREB_PCT', 'FT_RATE'])

    all_stats = pd.concat(all_seasons, ignore_index=True)
    
    # Fill missing values with season averages
    for col in all_stats.columns:
        if col not in ['TEAM', 'SEASON']:
            season_means = all_stats.groupby('SEASON')[col].transform('mean')
            all_stats[col] = all_stats[col].fillna(season_means)
    
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
    # Print initial data sample
    print("\nInitial data sample:")
    print(df[['FGM', 'FGA']].head())
    
    df = df.sort_values('GAME_DATE')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Basic stats - keep as raw numbers, don't convert to percentages
    df = df[['SEASON', 'GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST', 'PLUS_MINUS', 'FGA', 'FGM', 'MIN']]
    df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
    
    # Convert FGM and FGA to numeric if they aren't already
    df['FGM'] = pd.to_numeric(df['FGM'], errors='coerce')
    df['FGA'] = pd.to_numeric(df['FGA'], errors='coerce')
    
    print("\nAfter numeric conversion:")
    print(df[['FGM', 'FGA']].head())
    
    # Only drop rows where all key stats are missing
    df = df.dropna(subset=['PTS', 'REB', 'AST', 'MIN'], how='all')
    print(f"\nGames after initial cleaning: {len(df)}")

    # Rolling averages with different windows - keep as raw numbers
    for window in [3, 5, 10]:
        df[f'PTS_avg{window}'] = df['PTS'].rolling(window).mean().shift(1)
        df[f'REB_avg{window}'] = df['REB'].rolling(window).mean().shift(1)
        df[f'AST_avg{window}'] = df['AST'].rolling(window).mean().shift(1)
        df[f'PLUSMINUS_avg{window}'] = df['PLUS_MINUS'].rolling(window).mean().shift(1)
        df[f'FGM_avg{window}'] = df['FGM'].rolling(window).mean().shift(1)
        df[f'FGA_avg{window}'] = df['FGA'].rolling(window).mean().shift(1)

    # Season averages - keep as raw numbers
    df['PTS_season_avg'] = df.groupby('SEASON')['PTS'].expanding().mean().reset_index(0, drop=True)
    df['REB_season_avg'] = df.groupby('SEASON')['REB'].expanding().mean().reset_index(0, drop=True)
    df['AST_season_avg'] = df.groupby('SEASON')['AST'].expanding().mean().reset_index(0, drop=True)
    df['FGM_season_avg'] = df.groupby('SEASON')['FGM'].expanding().mean().reset_index(0, drop=True)
    df['FGA_season_avg'] = df.groupby('SEASON')['FGA'].expanding().mean().reset_index(0, drop=True)

    # Only calculate percentages for shooting stats
    df['FG_PCT'] = np.where(df['FGA'] > 0, (df['FGM'] / df['FGA']) * 100, 0)
    df['FG_PCT_avg3'] = df['FG_PCT'].rolling(3).mean().shift(1)
    df['FG_PCT_season_avg'] = df.groupby('SEASON')['FG_PCT'].expanding().mean().reset_index(0, drop=True)

    print("\nField Goal Percentage Stats:")
    print(f"Average FG%: {df['FG_PCT'].mean():.1f}%")
    print(f"Max FG%: {df['FG_PCT'].max():.1f}%")
    print(f"Min FG%: {df['FG_PCT'].min():.1f}%")

    # Extract opponent and map to full team name
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
        'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    }
    
    df['OPPONENT_FULL'] = df['OPPONENT'].map(team_abbrev_map)
    if df['OPPONENT_FULL'].isna().any():
        print("\nWarning: Some opponent abbreviations could not be mapped to full team names")
        print("Unmapped opponents:", df[df['OPPONENT_FULL'].isna()]['OPPONENT'].unique())

    # Merge with team stats
    team_stats = load_team_defense_data()
    df = df.merge(team_stats, how='left', left_on=['OPPONENT_FULL', 'SEASON'], right_on=['TEAM', 'SEASON'])
    df.drop(columns=['TEAM'], inplace=True)

    # Calculate opponent-specific stats - keep as raw numbers
    def get_opponent_avg_stat(row, stat):
        past_games = df[(df['OPPONENT_FULL'] == row['OPPONENT_FULL']) & (df['GAME_DATE'] < row['GAME_DATE'])]
        if not past_games.empty:
            return past_games[stat].mean()
        return None

    for stat in ['PTS', 'REB', 'AST', 'FG_PCT']:
        df[f'{stat}_vs_opp_avg'] = df.apply(lambda row: get_opponent_avg_stat(row, stat), axis=1)

    # Print averages for verification
    print("\nKey Statistics Averages:")
    for stat in ['PTS', 'REB', 'AST', 'FG_PCT']:
        print(f"{stat}: {df[stat].mean():.1f}")
        print(f"{stat}_avg3: {df[f'{stat}_avg3'].mean():.1f}")
        print(f"{stat}_season_avg: {df[f'{stat}_season_avg'].mean():.1f}")
        print(f"{stat}_vs_opp_avg: {df[f'{stat}_vs_opp_avg'].mean():.1f}")
        print()

    # Only drop rows where all features are missing
    df = df.dropna(subset=['PTS_avg3', 'PTS_avg5', 'PTS_avg10', 'FG_PCT'], how='all')
    print(f"\nFinal number of games after cleaning: {len(df)}")

    return df

def train_eval(df):
    if df.empty or len(df) < 5:  # Check if we have enough data
        print("Insufficient data for training. Need at least 5 games.")
        return None, None, None, None
        
    current_season = df['SEASON'].iloc[-1]
    current_season_data = df[df['SEASON'] == current_season]
    
    print(f"\nTotal games available: {len(df)}")
    print(f"Current season games: {len(current_season_data)}")
    
    last_20_games = df.tail(20)
    last_10_games = df.tail(10)
    last_5_games = df.tail(5)
    
    print("\nRecent Performance:")
    print("Last 5 games averages:")
    print(f"PTS: {last_5_games['PTS'].mean():.1f}")
    print(f"REB: {last_5_games['REB'].mean():.1f}")
    print(f"AST: {last_5_games['AST'].mean():.1f}")
    print(f"FG%: {(last_5_games['FGM'].sum() / last_5_games['FGA'].sum() * 100):.1f}%")
    
    print("\nLast 10 games averages:")
    print(f"PTS: {last_10_games['PTS'].mean():.1f}")
    print(f"REB: {last_10_games['REB'].mean():.1f}")
    print(f"AST: {last_10_games['AST'].mean():.1f}")
    print(f"FG%: {(last_10_games['FGM'].sum() / last_10_games['FGA'].sum() * 100):.1f}%")
    
    print("\nLast 20 games averages:")
    print(f"PTS: {last_20_games['PTS'].mean():.1f}")
    print(f"REB: {last_20_games['REB'].mean():.1f}")
    print(f"AST: {last_20_games['AST'].mean():.1f}")
    print(f"FG%: {(last_20_games['FGM'].sum() / last_20_games['FGA'].sum() * 100):.1f}%")
    
    print("\nSeason averages:")
    print(f"PTS: {current_season_data['PTS'].mean():.1f}")
    print(f"REB: {current_season_data['REB'].mean():.1f}")
    print(f"AST: {current_season_data['AST'].mean():.1f}")
    print(f"FG%: {(current_season_data['FGM'].sum() / current_season_data['FGA'].sum() * 100):.1f}%")

    features = ['PTS_avg3', 'PTS_avg5', 'PTS_avg10', 'FG_PCT']
    targets = ['PTS', 'REB', 'AST']
    
    # Only proceed with model training if we have enough data
    if len(df) >= 10:  # Reduced minimum games requirement
        df['PTS_avg3'] = df['PTS'].rolling(3).mean()
        df['PTS_avg5'] = df['PTS'].rolling(5).mean()
        df['PTS_avg10'] = df['PTS'].rolling(10).mean()
        df['FG_PCT'] = df['FGM'] / df['FGA'] * 100
        
        # Only drop rows where all features are missing
        df = df.dropna(subset=features, how='all')
        print(f"\nGames after feature cleaning: {len(df)}")
        
        if len(df) < 10:  # Check again after dropping NaN values
            print("Not enough data after cleaning for model training")
            return None, None, None, None
            
        x = df[features]
        y = df[targets]
        
        # Use a smaller test set if we don't have enough data
        test_size = min(10, len(df) // 3)  # Use at most 10 games for testing
        train_size = len(df) - test_size
        
        x_train = x[:train_size]
        x_test = x[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        models = {}
        predictions = {}
        metrics = {}
        
        for target in targets:
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            model.fit(x_train, y_train[target])
            y_pred = model.predict(x_test)
            
            # Adjust predictions based on recent performance
            recent_avg = last_10_games[target].mean()
            pred_avg = np.mean(y_pred)
            adjustment = recent_avg / pred_avg if pred_avg > 0 else 1
            y_pred = y_pred * adjustment
            
            models[target] = model
            predictions[target] = y_pred
            metrics[target] = {
                'mae': mean_absolute_error(y_test[target], y_pred),
                'r2': r2_score(y_test[target], y_pred)
            }
        
        print("\n--- Model Evaluation ---")
        for target in targets:
            print(f"\n{target}:")
            print(f"MAE = {metrics[target]['mae']:.2f}")
            print(f"R² = {metrics[target]['r2']:.2f}")
            print(f"Predicted range: {predictions[target].min():.1f} to {predictions[target].max():.1f}")
            print(f"Actual range: {y_test[target].min():.1f} to {y_test[target].max():.1f}")
        
        combined_model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ))
        combined_model.estimators_ = [models[target] for target in targets]
        combined_model.feature_names_in_ = features
        
        return combined_model, x_test, y_test, np.column_stack([predictions[target] for target in targets])
    else:
        print("Not enough data for model training")
        return None, None, None, None

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
    df_clean = df_clean[df_clean['MIN'] >= 10]
    
    current_season = '2024-25'
    current_season_data = df_clean[df_clean['SEASON'] == current_season]
    
    if current_season_data.empty:
        print(f"\nWarning: No data found for {current_season}. Using all available data.")
        current_season_data = df_clean
    
    current_season_stats = {
        'PTS': float(current_season_data['PTS'].mean()),
        'REB': float(current_season_data['REB'].mean()),
        'AST': float(current_season_data['AST'].mean())
    }
    
    last_10_games = current_season_data.tail(10)
    recent_stats = {
        'PTS': float(last_10_games['PTS'].mean()),
        'REB': float(last_10_games['REB'].mean()),
        'AST': float(last_10_games['AST'].mean())
    }
    
    last_5_games = current_season_data.tail(5)
    very_recent_stats = {
        'PTS': float(last_5_games['PTS'].mean()),
        'REB': float(last_5_games['REB'].mean()),
        'AST': float(last_5_games['AST'].mean())
    }
    
    prediction = {
        'Player': player_name,
        'Opponent': opponent_name
    }
    
    for stat in ['PTS', 'REB', 'AST']:
        weighted_prediction = (
            0.7 * current_season_stats[stat] +
            0.2 * recent_stats[stat] +
            0.1 * very_recent_stats[stat]
        )
        
        if recent_stats[stat] > current_season_stats[stat] * 1.1:
            weighted_prediction = (
                0.5 * current_season_stats[stat] +
                0.3 * recent_stats[stat] +
                0.2 * very_recent_stats[stat]
            )
        
        prediction[f'Predicted {stat}'] = round(float(weighted_prediction), 1)
    
    return prediction

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
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching schedule: {e}")
        return None


def find_next_game(schedule, team_id):
    if not schedule or 'leagueSchedule' not in schedule or 'gameDates' not in schedule['leagueSchedule']:
        print("Invalid schedule data")
        return None

    today = datetime.datetime.now().date()
    games = schedule['leagueSchedule']['gameDates']
    
    for game_date in games:
        try:
            game_day = datetime.datetime.strptime(game_date['gameDate'], '%m/%d/%Y %H:%M:%S').date()
            if game_day >= today:
                for game in game_date.get('games', []):
                    if str(game['homeTeam']['teamId']) == str(team_id) or str(game['awayTeam']['teamId']) == str(team_id):
                        opponent = game['awayTeam'] if str(game['homeTeam']['teamId']) == str(team_id) else game['homeTeam']
                        return {
                            'game_date': game_day,
                            'opponent_team_id': opponent['teamId'],
                            'opponent_team_name': opponent['teamName']
                        }
        except (KeyError, ValueError) as e:
            print(f"Error processing game date: {e}")
            continue
            
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
    try:
        player_name = "Cade Cunningham"
        print(f"Fetching game logs for {player_name}...")
        df = get_all_game_logs(player_name)

        if df.empty:
            raise ValueError(f"No game logs found for {player_name}")

        if 'MATCHUP' not in df.columns:
            print("MATCHUP column missing — retrying game log fetch.")
            df = get_all_game_logs(player_name)
            if 'MATCHUP' not in df.columns:
                print(df.head())
                raise ValueError("MATCHUP column still missing after retry. Check API limits or connection.")

        print("Processing injury data...")
        structured_injuries = structure_injuries(data)
        print("Available teams in injury data:", list(structured_injuries.keys()))

        out_teammates = get_out_teammates(player_name, structured_injuries)
        print(f"Found {len(out_teammates)} out teammates")

        df_filtered = df
        if out_teammates:
            df_filtered = filter_games_without_all_teammates(df_filtered, out_teammates)

        print("Getting team information...")
        try:
            team_id = get_player_team(player_name)
        except ValueError as e:
            print(f"Could not get team ID for {player_name} from cache: {e}")
            player_id = get_player_id(player_name)
            team_abbrev = get_team_abbrev_from_logs_or_cache(player_id)
            team_id = TEAM_ABBREV_TO_ID.get(team_abbrev)
            if not team_id:
                raise ValueError(f"Team ID could not be resolved for {team_abbrev}")

        print("Cleaning features...")
        df_clean = clean_features(df_filtered)
        if df_clean.empty:
            raise ValueError("No clean data available for training.")

        print("Loading team defense data...")
        team_stats_df = load_team_defense_data()
        if team_stats_df.empty:
            print("Warning: No team defense data available. Predictions may be less accurate.")

        print("Training model...")
        model, _, _, _ = train_eval(df_clean)

        print("Fetching schedule...")
        schedule = fetch_schedule()
        if not schedule:
            print("Warning: Could not fetch schedule. Cannot predict next game.")
        else:
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

    except Exception as e:
        print(f"\nError: {e}")
        print("Please check the error message above and try again.")

