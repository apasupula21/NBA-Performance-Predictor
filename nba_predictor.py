from datetime import datetime, timedelta
import pickle
import os
import time
import json
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, commonteamroster
from nba_api.stats.endpoints import boxscoretraditionalv2
from nba_api.stats.static import players, teams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import requests
from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.library.parameters import SeasonAll
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

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
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}
    
API_KEY = "dd7jmZQzRLLrBl9dX7FzTYgRkR3uDEvzffH7btqW"
url = f"https://api.sportradar.us/nba/trial/v8/en/league/injuries.json?api_key={API_KEY}"

try:
    response = session.get(url, timeout=10)
    response.raise_for_status()  
    data = response.json()
    print("Successfully fetched injury data")
except requests.exceptions.RequestException as e:
    print(f"Error fetching injury data: {e}")
    data = {"teams": []}  

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

        time.sleep(1.5)  

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
    try:
        rosters = load_team_roster_cache()
        if player_name in rosters:
            return rosters[player_name]
        else:
            print(f"Team ID not found for player: {player_name}")
            return None
    except Exception as e:
        print(f"Error getting player team: {e}")
        return None

def get_all_game_logs(player_name, start_season='2010-11', end_season='2024-25'):
    player_id = get_player_id(player_name)
    if not player_id:
        print(f"Could not find player ID for {player_name}")
        return pd.DataFrame()
    
    all_logs = []
    
    start_year = int(start_season.split('-')[0])
    end_year = int(end_season.split('-')[0])
    seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year + 1)]
    
    for season in seasons:
        try:
            print(f"Fetching game logs for {player_name} in {season}...")
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season',
                timeout=60
            )
            
            df = gamelog.get_data_frames()[0]
            if not df.empty and 'MATCHUP' in df.columns:
                df['SEASON'] = season
                all_logs.append(df)
                print(f"Successfully fetched {len(df)} games for {season}")
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Failed to fetch {season} data: {e}")
            continue
    
    if all_logs:
        combined_df = pd.concat(all_logs, ignore_index=True)
        try:
            combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'], format='mixed')
        except:
            try:
                combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'], format='%B %d, %Y')
            except:
                combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'])
        
        combined_df = combined_df.sort_values('GAME_DATE')
        
        print(f"\nTotal games fetched: {len(combined_df)}")
        print(f"Date range: {combined_df['GAME_DATE'].min().strftime('%Y-%m-%d')} to {combined_df['GAME_DATE'].max().strftime('%Y-%m-%d')}")
        print(f"Seasons included: {combined_df['SEASON'].unique()}")
        
        return combined_df
    else:
        return pd.DataFrame()

def load_team_defense_data(stats_folder='team_stats', cache_duration_hours=24):
    cache_file = 'team_stats_cache.pkl'
    
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(hours=cache_duration_hours):
            try:
                with open(cache_file, 'rb') as f:
                    print("Loading team stats from cache...")
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    if not os.path.exists(stats_folder):
        print(f"Creating {stats_folder} directory...")
        os.makedirs(stats_folder)
        print(f"Please add team statistics Excel files to the {stats_folder} directory")
        return pd.DataFrame(columns=['TEAM', 'DEF_RTG', 'SEASON', 'PACE', 'OFF_RTG', 'EFG_PCT', 'TOV_PCT', 'OREB_PCT', 'FT_RATE'])

    all_seasons = []
    processed_files = set()
    
    for file in os.listdir(stats_folder):
        if file in processed_files:
            continue
            
        if file.endswith(('.xlsx', '.csv')):
            try:
                season = file.split('_')[-1].split('.')[0]
                if file.endswith('.xlsx'):
                    df = pd.read_excel(os.path.join(stats_folder, file))
                else:
                    df = pd.read_csv(os.path.join(stats_folder, file))
                
                if 'Rk' not in df.columns:
                    print(f"Skipping {file}: Missing 'Rk' column")
                    continue
                    
                df = df[df['Rk'].notna()] 
                df['Team'] = df['Team'].replace({
                    'LA Clippers': 'Los Angeles Clippers',
                    'Portland Trl': 'Portland Trail Blazers',
                    'Brooklyn': 'Brooklyn Nets',
                    'Golden State': 'Golden State Warriors',
                    'LA Lakers': 'Los Angeles Lakers'
                })

                new_df = pd.DataFrame()
                new_df['TEAM'] = df['Team']
                new_df['DEF_RTG'] = df['DRtg']
                new_df['OFF_RTG'] = df['ORtg']
                new_df['PACE'] = df['Pace']
                new_df['EFG_PCT'] = df['eFG%']
                new_df['TOV_PCT'] = df['TOV%'].apply(lambda x: x if x <= 100 else x/100)
                new_df['OREB_PCT'] = df['ORB%'].apply(lambda x: x if x <= 100 else x/100)
                new_df['FT_RATE'] = df['FT/FGA']
                new_df['SEASON'] = f"{int(season)-1}-{season[-2:]}"

                for col in ['TOV_PCT', 'OREB_PCT']:
                    if new_df[col].max() > 100 or new_df[col].min() < 0:
                        print(f"Warning: {col} has values outside 0-100 range: {new_df[col].min():.1f} to {new_df[col].max():.1f}")

                all_seasons.append(new_df)
                processed_files.add(file)
                print(f"Successfully processed {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

    if not all_seasons:
        print(f"No valid team statistics found in {stats_folder}")
        return pd.DataFrame(columns=['TEAM', 'DEF_RTG', 'SEASON', 'PACE', 'OFF_RTG', 'EFG_PCT', 'TOV_PCT', 'OREB_PCT', 'FT_RATE'])

    all_stats = pd.concat(all_seasons, ignore_index=True)
    
    for col in all_stats.columns:
        if col not in ['TEAM', 'SEASON']:
            season_means = all_stats.groupby('SEASON')[col].transform('mean')
            all_stats[col] = all_stats[col].fillna(season_means)
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(all_stats, f)
        print("Team stats cached successfully")
    except Exception as e:
        print(f"Error caching team stats: {e}")
    
    return all_stats

def structure_injuries(raw_data):
    structured_data = {}
    for team in raw_data.get('teams', []):
        team_name = team.get('name', '')
        injuries = []
        for player in team.get('players', []):
            if player.get('status', {}).get('id') == 'OUT':
                injuries.append({
                    'name': player.get('full_name', ''),
                    'status': 'OUT',
                    'description': player.get('status', {}).get('description', '')
                })
        if injuries:
            structured_data[team_name] = injuries
    return structured_data

def get_teammate_injuries(player_name, structured_data):
    team_id = get_player_team(player_name)
    if not team_id:
        return []
    
    team_info = teams.find_team_name_by_id(team_id)
    if not team_info:
        return []
    
    team_name = team_info.get('full_name', '')
    return structured_data.get(team_name, [])

def print_cached_rosters():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
            print("\nCached rosters:")
            for player, team_id in data["rosters"].items():
                team_info = teams.find_team_name_by_id(team_id)
                team_name = team_info.get('full_name', 'Unknown Team') if team_info else 'Unknown Team'
                print(f"{player}: {team_name}")

def get_player_game_dates(player_name, season='2023-24'):
    return get_all_game_logs(player_name, season, season)['GAME_DATE'].tolist()

def build_availability_table(target_player, teammates, season='2023-24'):
    target_dates = get_player_game_dates(target_player, season)
    availability = {teammate: get_player_game_dates(teammate, season) for teammate in teammates}
    return target_dates, availability

def get_head_to_head_stats(player_name, opponent_team, season='2024-25', max_retries=3):
    cache_key = f"{player_name}_{opponent_team}_{season}"
    cache_file = f'head_to_head_cache_{cache_key}.pkl'
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading head-to-head cache: {e}")
    
    try:
        player_id = get_player_id(player_name)
        if not player_id:
            return None
            
        all_logs = get_all_game_logs(player_name, season, season)
        if all_logs.empty:
            return None
            
        opponent_name = opponent_team['full_name'] if isinstance(opponent_team, dict) else opponent_team
        
        # Convert opponent name to abbreviation if it's a full name
        team_abbrev_to_full = {
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
        
        # Create reverse mapping
        full_to_abbrev = {v: k for k, v in team_abbrev_to_full.items()}
        
        # Get opponent abbreviation
        opponent_abbrev = full_to_abbrev.get(opponent_name, opponent_name)
        
        # Match using the abbreviation
        h2h_games = all_logs[all_logs['MATCHUP'].str.contains(opponent_abbrev, case=False)]
        if h2h_games.empty:
            return None
            
        stats = {
            'games_played': len(h2h_games),
            'avg_pts': h2h_games['PTS'].mean(),
            'avg_reb': h2h_games['REB'].mean(),
            'avg_ast': h2h_games['AST'].mean(),
            'avg_fg_pct': h2h_games['FG_PCT'].mean() if 'FG_PCT' in h2h_games.columns else None,
            'last_game_pts': h2h_games['PTS'].iloc[-1] if not h2h_games.empty else None,
            'last_game_reb': h2h_games['REB'].iloc[-1] if not h2h_games.empty else None,
            'last_game_ast': h2h_games['AST'].iloc[-1] if not h2h_games.empty else None
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(stats, f)
        except Exception as e:
            print(f"Error caching head-to-head stats: {e}")
        
        return stats
    except Exception as e:
        print(f"Error getting head-to-head stats: {e}")
        return None

def get_team_defensive_stats(team_name, season='2024-25'):
    try:
        team_stats_df = load_team_defense_data()
        if team_stats_df.empty:
            return None
            
        team_abbrev_to_full = {
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
        
        if team_name in team_abbrev_to_full:
            team_name = team_abbrev_to_full[team_name]
            print(f"Converted team abbreviation to full name: {team_name}")
        
        print(f"Available teams: {team_stats_df['TEAM'].unique()}")
        
        team_stats = team_stats_df[team_stats_df['TEAM'] == team_name]
        if team_stats.empty:
            print(f"No stats found for team: {team_name}")
            return None
        
        return {
            'def_rtg': team_stats['DEF_RTG'].iloc[0],
            'pace': team_stats['PACE'].iloc[0],
            'efg_pct': team_stats['EFG_PCT'].iloc[0],
            'tov_pct': team_stats['TOV_PCT'].iloc[0],
            'oreb_pct': team_stats['OREB_PCT'].iloc[0],
            'ft_rate': team_stats['FT_RATE'].iloc[0]
        }
    except Exception as e:
        print(f"Error getting team defensive stats: {e}")
        return None

def fetch_schedule(cache_duration_hours=24):
    cache_file = 'schedule_cache.pkl'
    
    if os.path.exists(cache_file):
        try:
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=cache_duration_hours):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading schedule cache: {e}")
    
    print("Fetching new schedule...")
    try:
        current_date = datetime.now()
        schedule = []
        
        for i in range(7):
            target_date = current_date + timedelta(days=i)
            try:
                scoreboard = scoreboardv2.ScoreboardV2(
                    game_date=target_date.strftime('%m/%d/%Y'),
                    league_id='00',
                    day_offset=0
                )
                
                games = scoreboard.get_data_frames()[0]
                if not games.empty:
                    for _, game in games.iterrows():
                        schedule.append({
                            'game_date': target_date,
                            'home_team_id': game['HOME_TEAM_ID'],
                            'away_team_id': game['VISITOR_TEAM_ID'],
                            'home_team_name': game['HOME_TEAM_NAME'],
                            'away_team_name': game['VISITOR_TEAM_NAME']
                        })
                
                time.sleep(1)
            except Exception as e:
                print(f"Error fetching games for {target_date.strftime('%Y-%m-%d')}: {e}")
                continue
        
        if not schedule:
            print("No games found in schedule")
            return []
        
        schedule = list({(game['game_date'], game['home_team_id'], game['away_team_id']): game 
                        for game in schedule}.values())
        schedule.sort(key=lambda x: x['game_date'])
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(schedule, f)
        except Exception as e:
            print(f"Error caching schedule: {e}")
        
        return schedule
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []

def get_team_id(team_name):
    team_dict = teams.find_teams_by_full_name(team_name)
    return team_dict[0]['id'] if team_dict else None

def get_team_abbreviation(team_name):
    team_dict = teams.find_teams_by_full_name(team_name)
    return team_dict[0]['abbreviation'] if team_dict else None

def clean_features(df, player_name):
    def extract_opponent(matchup):
        if isinstance(matchup, str):
            if ' vs. ' in matchup:
                return matchup.split(' vs. ')[1]
            elif ' @ ' in matchup:
                return matchup.split(' @ ')[1]
        return None

    def extract_team(matchup):
        if isinstance(matchup, str):
            if ' vs. ' in matchup:
                return matchup.split(' vs. ')[0]
            elif ' @ ' in matchup:
                return matchup.split(' @ ')[0]
        return None

    df_clean = df.copy()
    
    df_clean['HOME_GAME'] = df_clean['MATCHUP'].str.contains(' vs. ')
    df_clean['OPPONENT'] = df_clean['MATCHUP'].apply(extract_opponent)
    df_clean['TEAM_ABBREVIATION'] = df_clean['MATCHUP'].apply(extract_team)
    
    df_clean['GAME_DATE'] = pd.to_datetime(df_clean['GAME_DATE'])
    df_clean['DAYS_REST'] = df_clean['GAME_DATE'].diff().dt.days.fillna(3)
    df_clean['GAME_NUMBER'] = range(1, len(df_clean) + 1)
    
    for stat in ['PTS', 'REB', 'AST']:
        df_clean[f'LAST_5_GAMES_{stat}'] = df_clean[stat].rolling(window=5, min_periods=1).mean()
    
    return df_clean

def train_eval(df):
    if df.empty:
        print("No data available for training")
        return None, None, None, None
        
    required_features = [
        'HOME_GAME', 'DAYS_REST', 'GAME_NUMBER', 
        'LAST_5_GAMES_PTS', 'LAST_5_GAMES_REB', 'LAST_5_GAMES_AST',
        'OPP_DEF_RTG', 'OPP_PACE', 'OPP_EFG_PCT', 'OPP_TOV_PCT', 'OPP_OREB_PCT', 'OPP_FT_RATE',
        'H2H_GAMES_PLAYED', 'H2H_AVG_PTS', 'H2H_AVG_REB', 'H2H_AVG_AST'
    ]
    
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0
    
    X = df[required_features].fillna(0)
    y = df[['PTS', 'REB', 'AST']]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = MultiOutputRegressor(Ridge(
        alpha=1.0,
        fit_intercept=True,
        random_state=42
    ))
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions, multioutput='raw_values')
    r2 = r2_score(y_test, predictions, multioutput='raw_values')
    
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")
    
    model.scaler = scaler
    model.feature_names = required_features
    
    return model, X_test, y_test, predictions

def get_team_def_rtg_by_name(opponent_name, season, team_stats_df):
    team_stats = team_stats_df[team_stats_df['TEAM'] == opponent_name]
    if not team_stats.empty:
        return team_stats['DEF_RTG'].iloc[0]
    return None

def predict_against_opponent(player_name, opponent_name, df_clean, model, team_stats_df):
    if model is None:
        return None
    
    features = {feature: [0] for feature in model.feature_names}
    
    features['HOME_GAME'] = [True]
    features['DAYS_REST'] = [3]
    features['GAME_NUMBER'] = [len(df_clean) + 1]
    
    last_game = df_clean.iloc[-1]
    features['LAST_5_GAMES_PTS'] = [last_game['LAST_5_GAMES_PTS']]
    features['LAST_5_GAMES_REB'] = [last_game['LAST_5_GAMES_REB']]
    features['LAST_5_GAMES_AST'] = [last_game['LAST_5_GAMES_AST']]
    
    opponent_stats = get_team_defensive_stats(opponent_name)
    if not opponent_stats:
        return None
    
    features['OPP_DEF_RTG'] = [opponent_stats['def_rtg']]
    features['OPP_PACE'] = [opponent_stats['pace']]
    features['OPP_EFG_PCT'] = [opponent_stats['efg_pct']]
    features['OPP_TOV_PCT'] = [opponent_stats['tov_pct']]
    features['OPP_OREB_PCT'] = [opponent_stats['oreb_pct']]
    features['OPP_FT_RATE'] = [opponent_stats['ft_rate']]
    
    h2h_defaults = {
        'H2H_GAMES_PLAYED': 0,
        'H2H_AVG_PTS': df_clean['PTS'].mean(),
        'H2H_AVG_REB': df_clean['REB'].mean(),
        'H2H_AVG_AST': df_clean['AST'].mean()
    }
    
    h2h_stats = get_head_to_head_stats(player_name, opponent_name)
    if h2h_stats:
        for key in h2h_defaults:
            if key in h2h_stats:
                h2h_defaults[key] = h2h_stats[key]
    
    for key, value in h2h_defaults.items():
        features[key] = [value]
    
    features_df = pd.DataFrame(features)[model.feature_names]
    features_scaled = model.scaler.transform(features_df)
    prediction = model.predict(features_scaled)
    
    season_avg_pts = df_clean['PTS'].mean()
    season_avg_reb = df_clean['REB'].mean()
    season_avg_ast = df_clean['AST'].mean()
    
    weight = 0.7
    predicted_pts = weight * season_avg_pts + (1 - weight) * prediction[0][0]
    predicted_reb = weight * season_avg_reb + (1 - weight) * prediction[0][1]
    predicted_ast = weight * season_avg_ast + (1 - weight) * prediction[0][2]
    
    return {
        'Predicted PTS': predicted_pts,
        'Predicted REB': predicted_reb,
        'Predicted AST': predicted_ast,
        'Season Avg PTS': season_avg_pts,
        'Season Avg REB': season_avg_reb,
        'Season Avg AST': season_avg_ast
    }

def filter_games_without_all_teammates(df, absent_teammates, season='2024-25'):
    if not absent_teammates:
        return df
    
    filtered_df = df.copy()
    for teammate in absent_teammates:
        teammate_id = get_player_id(teammate)
        if teammate_id:
            teammate_games = get_all_game_logs(teammate, season, season)
            if not teammate_games.empty:
                teammate_dates = set(teammate_games['GAME_DATE'])
                filtered_df = filtered_df[filtered_df['GAME_DATE'].isin(teammate_dates)]
    
    return filtered_df

def find_next_game(schedule, team_id):
    if not schedule or not team_id:
        return None
    
    current_date = datetime.now()
    
    for game in schedule:
        game_date = game['game_date']
        if game_date > current_date and (game['home_team_id'] == team_id or game['away_team_id'] == team_id):
            home_team_info = teams.find_team_name_by_id(game['home_team_id'])
            away_team_info = teams.find_team_name_by_id(game['away_team_id'])
            
            if not home_team_info or not away_team_info:
                continue
                
            home_abbr = home_team_info.get('abbreviation', '') if isinstance(home_team_info, dict) else home_team_info
            away_abbr = away_team_info.get('abbreviation', '') if isinstance(away_team_info, dict) else away_team_info
            
            if not home_abbr or not away_abbr:
                continue
                
            if game['home_team_id'] == team_id:
                opponent_abbr = away_abbr
                is_home_game = True
            else:
                opponent_abbr = home_abbr
                is_home_game = False
            
            return {
                'game_date': game_date,
                'opponent_team_name': TEAM_ABBREV_TO_FULL.get(opponent_abbr, opponent_abbr),
                'is_home_game': is_home_game
            }
    
    return None

def get_team_abbrev_from_logs_or_cache(player_id):
    try:
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season='2024-25',
            season_type_all_star='Regular Season'
        )
        df = gamelog.get_data_frames()[0]
        if not df.empty and 'TEAM_ABBREVIATION' in df.columns:
            return df['TEAM_ABBREVIATION'].iloc[0]
    except:
        pass
    return None

def get_out_teammates(player_name, structured_data):
    team_id = get_player_team(player_name)
    if not team_id:
        return []
    
    team_info = teams.find_team_name_by_id(team_id)
    if not team_info:
        return []
    
    team_name = team_info.get('full_name', '')
    injuries = structured_data.get(team_name, [])
    return [injury['name'] for injury in injuries]

