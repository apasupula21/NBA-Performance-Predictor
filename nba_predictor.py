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
    max_retries = 3
    retry_delay = 5  # seconds to wait between retries
    
    # Get all seasons between start and end
    start_year = int(start_season.split('-')[0])
    end_year = int(end_season.split('-')[0])
    seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year + 1)]
    
    for season in seasons:
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1} to fetch game logs for {player_name} in {season}...")
                gamelog = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star='Regular Season',
                    timeout=60  # Increased timeout
                )
                
                # Add a delay before making the request
                time.sleep(2)
                
                df = gamelog.get_data_frames()[0]
                if not df.empty and 'MATCHUP' in df.columns:
                    df['SEASON'] = season
                    all_logs.append(df)
                    print(f"Successfully fetched {len(df)} games for {season}")
                    break
                else:
                    print(f"Received empty or invalid data frame for {season}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {season}: {e}")
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to fetch {season} data after {max_retries} attempts")
    
    if all_logs:
        combined_df = pd.concat(all_logs, ignore_index=True)
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
    
    # Check if cache exists and is recent enough
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(hours=cache_duration_hours):
            try:
                with open(cache_file, 'rb') as f:
                    print("Loading team stats from cache...")
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    # If no cache or cache is expired, process files
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
    
    # Cache the processed data
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(all_stats, f)
        print("Team stats cached successfully")
    except Exception as e:
        print(f"Error caching team stats: {e}")
    
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
        return {}

    teammate_injuries = {}
    for teammate, status in structured_data[found_team].items():
        if teammate != player_name:
            teammate_injuries[teammate] = status

    return teammate_injuries

def print_cached_rosters():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
            print(f"Cache timestamp: {datetime.datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nCached rosters:")
            for player, team_id in data['rosters'].items():
                print(f"{player}: {team_id}")
    else:
        print("No cache file found.")

def get_player_game_dates(player_name, season='2023-24'):
    df = get_all_game_logs(player_name, season, season)
    return df['GAME_DATE'].tolist() if not df.empty else []

def build_availability_table(target_player, teammates, season='2023-24'):
    target_dates = get_player_game_dates(target_player, season)
    availability = {teammate: [] for teammate in teammates}
    
    for teammate in teammates:
        teammate_dates = get_player_game_dates(teammate, season)
        for date in target_dates:
            availability[teammate].append(date in teammate_dates)
    
    return pd.DataFrame(availability, index=target_dates)

def get_head_to_head_stats(player_name, opponent_team, season='2024-25', max_retries=3):
    # Create a cache key based on player and opponent
    cache_key = f"{player_name}_{opponent_team}_{season}"
    cache_file = f'head_to_head_cache_{cache_key}.pkl'
    
    # Try to load from cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                print(f"Loading head-to-head stats from cache for {player_name} vs {opponent_team}")
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading head-to-head cache: {e}")
    
    # If not in cache, try to get from existing game logs
    try:
        player_id = get_player_id(player_name)
        if not player_id:
            return None
            
        # Get all game logs for the player
        all_logs = get_all_game_logs(player_name, season, season)
        if all_logs.empty:
            return None
            
        # Filter for games against the opponent
        opponent_name = opponent_team['full_name'] if isinstance(opponent_team, dict) else opponent_team
        h2h_games = all_logs[all_logs['MATCHUP'].str.contains(opponent_name, case=False)]
        if h2h_games.empty:
            return None
            
        # Calculate stats
        stats = {
            'games_played': len(h2h_games),
            'avg_pts': h2h_games['PTS'].mean(),
            'avg_reb': h2h_games['REB'].mean(),
            'avg_ast': h2h_games['AST'].mean(),
            'avg_fg_pct': h2h_games['FG_PCT'].mean(),
            'last_game_pts': h2h_games['PTS'].iloc[-1] if len(h2h_games) > 0 else None,
            'last_game_reb': h2h_games['REB'].iloc[-1] if len(h2h_games) > 0 else None,
            'last_game_ast': h2h_games['AST'].iloc[-1] if len(h2h_games) > 0 else None
        }
        
        # Cache the results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(stats, f)
            print(f"Cached head-to-head stats for {player_name} vs {opponent_name}")
        except Exception as e:
            print(f"Error caching head-to-head stats: {e}")
            
        return stats
        
    except Exception as e:
        print(f"Error calculating head-to-head stats: {e}")
        return None

def get_team_defensive_stats(team_name, season='2024-25'):
    try:
        print(f"Loading team defense data for {team_name}...")
        team_stats_df = load_team_defense_data()
        if team_stats_df.empty:
            print("Error: No team defense data available")
            return None
            
        # Convert team abbreviation to full name if needed
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
        
        # Check if input is an abbreviation and convert to full name
        if team_name in team_abbrev_to_full:
            team_name = team_abbrev_to_full[team_name]
            print(f"Converted team abbreviation to full name: {team_name}")
        
        # Print available teams for debugging
        print(f"Available teams: {team_stats_df['TEAM'].unique()}")
        
        team_stats = team_stats_df[
            (team_stats_df['TEAM'].str.contains(team_name, case=False)) & 
            (team_stats_df['SEASON'] == season)
        ]
        
        if team_stats.empty:
            print(f"Error: No stats found for {team_name} in {season}")
            print(f"Available seasons: {team_stats_df['SEASON'].unique()}")
            return None
        
        stats = {
            'def_rtg': team_stats['DEF_RTG'].iloc[0],
            'pace': team_stats['PACE'].iloc[0],
            'efg_pct': team_stats['EFG_PCT'].iloc[0],
            'tov_pct': team_stats['TOV_PCT'].iloc[0],
            'oreb_pct': team_stats['OREB_PCT'].iloc[0],
            'ft_rate': team_stats['FT_RATE'].iloc[0]
        }
        
        print(f"Successfully loaded stats for {team_name}: {stats}")
        return stats
    except Exception as e:
        print(f"Error in get_team_defensive_stats: {str(e)}")
        return None

def fetch_schedule(cache_duration_hours=24):
    cache_file = 'schedule_cache.pkl'
    
    # Check if cache exists and is recent enough
    if os.path.exists(cache_file):
        try:
            cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_time < timedelta(hours=cache_duration_hours):
                try:
                    with open(cache_file, 'rb') as f:
                        print("Loading schedule from cache...")
                        cached_data = pickle.load(f)
                        if isinstance(cached_data, list) and len(cached_data) > 0:
                            print(f"Successfully loaded {len(cached_data)} games from cache")
                            return cached_data
                        else:
                            print("Cache file exists but contains invalid data. Deleting cache file...")
                            os.remove(cache_file)
                except Exception as e:
                    print(f"Error loading schedule cache: {e}")
                    print("Deleting corrupted cache file...")
                    os.remove(cache_file)
        except Exception as e:
            print(f"Error checking cache file: {e}")
            print("Deleting corrupted cache file...")
            os.remove(cache_file)
    
    print("Fetching new schedule...")
    try:
        # Get current date
        current_date = datetime.now()
        
        # Initialize schedule list
        schedule = []
        
        # Get scoreboard for the next 7 days
        for i in range(7):
            target_date = current_date + timedelta(days=i)
            try:
                scoreboard = scoreboardv2.ScoreboardV2(
                    game_date=target_date.strftime('%m/%d/%Y'),
                    league_id='00',
                    day_offset='0',
                    timeout=30
                )
                
                games = scoreboard.get_data_frames()[0]
                if not games.empty:
                    for _, game in games.iterrows():
                        home_team_id = game['HOME_TEAM_ID']
                        away_team_id = game['VISITOR_TEAM_ID']
                        
                        home_team = teams.find_team_name_by_id(home_team_id)
                        away_team = teams.find_team_name_by_id(away_team_id)
                        
                        if home_team and away_team:
                            schedule.append({
                                'game_date': target_date,
                                'home_team_name': home_team,
                                'away_team_name': away_team,
                                'home_team_id': home_team_id,
                                'away_team_id': away_team_id
                            })
                
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error fetching games for {target_date.strftime('%Y-%m-%d')}: {e}")
                continue
        
        if not schedule:
            print("No games found in the schedule")
            return []
        
        # Remove duplicates and sort by date
        schedule = list({(game['game_date'], game['home_team_id'], game['away_team_id']): game 
                        for game in schedule}.values())
        schedule.sort(key=lambda x: x['game_date'])
        
        # Cache the results
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(schedule, f)
            print(f"Successfully cached {len(schedule)} games")
        except Exception as e:
            print(f"Error caching schedule: {e}")
        
        return schedule
        
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []

def get_team_id(team_name):
    team_dict = teams.find_teams_by_full_name(team_name)
    if team_dict:
        return team_dict[0]['id']
    return None

def get_team_abbreviation(team_name):
    team_abbrevs = {
        'Atlanta Hawks': 'ATL',
        'Boston Celtics': 'BOS',
        'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA',
        'Chicago Bulls': 'CHI',
        'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL',
        'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW',
        'Houston Rockets': 'HOU',
        'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL',
        'Memphis Grizzlies': 'MEM',
        'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL',
        'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK',
        'Oklahoma City Thunder': 'OKC',
        'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI',
        'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC',
        'San Antonio Spurs': 'SAS',
        'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA',
        'Washington Wizards': 'WAS'
    }
    return team_abbrevs.get(team_name, team_name)

def clean_features(df, player_name):
    if df.empty:
        return df
    
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['SEASON'] = df['SEASON'].fillna('2024-25')
    
    numeric_cols = ['MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                   'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                   'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['HOME_GAME'] = df['MATCHUP'].str.contains('vs.')
    
    def extract_opponent(matchup):
        if pd.isna(matchup):
            return None
        if 'vs.' in matchup:
            return matchup.split('vs.')[1].strip().split()[0]
        elif '@' in matchup:
            return matchup.split('@')[1].strip().split()[0]
        return None
    
    df['OPPONENT'] = df['MATCHUP'].apply(extract_opponent)
    
    def extract_team(matchup):
        if pd.isna(matchup):
            return None
        if 'vs.' in matchup:
            return matchup.split('vs.')[0].strip()
        elif '@' in matchup:
            return matchup.split('@')[0].strip()
        return None
    
    df['TEAM'] = df['MATCHUP'].apply(extract_team)
    df['TEAM_ABBREVIATION'] = df['TEAM'].apply(get_team_abbreviation)
    
    df['GAME_NUMBER'] = df.groupby('SEASON').cumcount() + 1
    
    df['LAST_5_GAMES_PTS'] = df.groupby('SEASON')['PTS'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['LAST_5_GAMES_REB'] = df.groupby('SEASON')['REB'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['LAST_5_GAMES_AST'] = df.groupby('SEASON')['AST'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    
    df['DAYS_REST'] = df['GAME_DATE'].diff().dt.days.fillna(0)
    
    return df

def train_eval(df):
    if df.empty:
        return None, None, None, None
        
    # Define all required features in the correct order
    required_features = [
        'HOME_GAME', 'DAYS_REST', 'GAME_NUMBER', 
        'LAST_5_GAMES_PTS', 'LAST_5_GAMES_REB', 'LAST_5_GAMES_AST',
        'OPP_DEF_RTG', 'OPP_PACE', 'OPP_EFG_PCT', 'OPP_TOV_PCT', 
        'OPP_OREB_PCT', 'OPP_FT_RATE',
        'H2H_GAMES_PLAYED', 'H2H_AVG_PTS', 'H2H_AVG_REB', 'H2H_AVG_AST',
        'H2H_AVG_FG_PCT', 'H2H_LAST_GAME_PTS', 'H2H_LAST_GAME_REB', 'H2H_LAST_GAME_AST'
    ]
    
    # Ensure all required features exist in the DataFrame
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0  # Initialize missing features with 0
    
    # Select only the required features in the correct order
    X = df[required_features].fillna(0)
    y = df[['PTS', 'REB', 'AST']]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Use Ridge Regression
    model = MultiOutputRegressor(Ridge(
        alpha=1.0,  # Regularization strength
        fit_intercept=True,
        random_state=42
    ))
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Calculate and print evaluation metrics
    mae = mean_absolute_error(y_test, predictions, multioutput='raw_values')
    r2 = r2_score(y_test, predictions, multioutput='raw_values')
    
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")
    
    # Store the scaler and feature names with the model for later use
    model.scaler = scaler
    model.feature_names = required_features
    
    return model, X_test, y_test, predictions

def get_team_def_rtg_by_name(opponent_name, season, team_stats_df):
    opponent_stats = team_stats_df[
        (team_stats_df['TEAM'].str.contains(opponent_name, case=False)) & 
        (team_stats_df['SEASON'] == season)
    ]
    if not opponent_stats.empty:
        return opponent_stats['DEF_RTG'].iloc[0]
    return None

def predict_against_opponent(player_name, opponent_name, df_clean, model, team_stats_df):
    if model is None:
        return None
    
    last_game = df_clean.iloc[-1]
    season = last_game['SEASON']
    
    opponent_stats = get_team_defensive_stats(opponent_name, season)
    h2h_stats = get_head_to_head_stats(player_name, opponent_name, season)
    
    if not opponent_stats:
        return None
    
    # Initialize features dictionary with default values
    features = {feature: [0] for feature in model.feature_names}
    
    # Set base features
    features['HOME_GAME'] = [True]
    features['DAYS_REST'] = [3]
    features['GAME_NUMBER'] = [last_game['GAME_NUMBER'] + 1]
    features['LAST_5_GAMES_PTS'] = [last_game['LAST_5_GAMES_PTS']]
    features['LAST_5_GAMES_REB'] = [last_game['LAST_5_GAMES_REB']]
    features['LAST_5_GAMES_AST'] = [last_game['LAST_5_GAMES_AST']]
    
    # Set opponent stats
    features['OPP_DEF_RTG'] = [opponent_stats['def_rtg']]
    features['OPP_PACE'] = [opponent_stats['pace']]
    features['OPP_EFG_PCT'] = [opponent_stats['efg_pct']]
    features['OPP_TOV_PCT'] = [opponent_stats['tov_pct']]
    features['OPP_OREB_PCT'] = [opponent_stats['oreb_pct']]
    features['OPP_FT_RATE'] = [opponent_stats['ft_rate']]
    
    # Set head-to-head stats with defaults from last game
    h2h_defaults = {
        'H2H_GAMES_PLAYED': 0,
        'H2H_AVG_PTS': last_game['PTS'],
        'H2H_AVG_REB': last_game['REB'],
        'H2H_AVG_AST': last_game['AST'],
        'H2H_AVG_FG_PCT': last_game['FG_PCT'],
        'H2H_LAST_GAME_PTS': last_game['PTS'],
        'H2H_LAST_GAME_REB': last_game['REB'],
        'H2H_LAST_GAME_AST': last_game['AST']
    }
    
    # Update with actual h2h stats if available
    if h2h_stats:
        for key in h2h_defaults:
            if key in h2h_stats:
                h2h_defaults[key] = h2h_stats[key]
    
    # Update features with h2h stats
    for key, value in h2h_defaults.items():
        features[key] = [value]
    
    # Create DataFrame with features in the correct order
    features_df = pd.DataFrame(features)[model.feature_names]
    
    # Scale the features using the stored scaler
    features_scaled = model.scaler.transform(features_df)
    
    # Get the prediction
    prediction = model.predict(features_scaled)
    
    # Get season averages for comparison
    season_avg_pts = df_clean['PTS'].mean()
    season_avg_reb = df_clean['REB'].mean()
    season_avg_ast = df_clean['AST'].mean()
    
    # Apply a weighted average between prediction and season average
    weight = 0.7  # Weight for season average
    predicted_pts = weight * season_avg_pts + (1 - weight) * prediction[0][0]
    predicted_reb = weight * season_avg_reb + (1 - weight) * prediction[0][1]
    predicted_ast = weight * season_avg_ast + (1 - weight) * prediction[0][2]
    
    return {
        'Predicted PTS': round(predicted_pts, 1),
        'Predicted REB': round(predicted_reb, 1),
        'Predicted AST': round(predicted_ast, 1),
        'Season Avg PTS': round(season_avg_pts, 1),
        'Season Avg REB': round(season_avg_reb, 1),
        'Season Avg AST': round(season_avg_ast, 1)
    }

def filter_games_without_all_teammates(df, absent_teammates, season='2024-25'):
    if not absent_teammates:
        return df
    
    season_data = df[df['SEASON'] == season]
    if season_data.empty:
        return df
    
    filtered_games = []
    for _, game in season_data.iterrows():
        game_date = game['GAME_DATE']
        all_teammates_present = True
        
        for teammate in absent_teammates:
            teammate_games = get_player_game_dates(teammate, season)
            if game_date not in teammate_games:
                all_teammates_present = False
                break
        
        if all_teammates_present:
            filtered_games.append(game)
    
    if filtered_games:
        return pd.DataFrame(filtered_games)
    return df

def find_next_game(schedule, team_id):
    if not schedule or not team_id:
        return None
    
    current_date = datetime.now()
    for game in schedule:
        if not game or 'game_date' not in game:
            continue
            
        game_date = game['game_date']
        if game_date > current_date and (game['home_team_id'] == team_id or game['away_team_id'] == team_id):
            # Get team information
            home_team_info = teams.find_team_name_by_id(game['home_team_id'])
            away_team_info = teams.find_team_name_by_id(game['away_team_id'])
            
            if not home_team_info or not away_team_info:
                continue
                
            # Extract abbreviations
            home_abbr = home_team_info.get('abbreviation', '') if isinstance(home_team_info, dict) else home_team_info
            away_abbr = away_team_info.get('abbreviation', '') if isinstance(away_team_info, dict) else away_team_info
            
            if not home_abbr or not away_abbr:
                continue
                
            # Determine opponent and game location
            if game['home_team_id'] == team_id:
                opponent_abbr = away_abbr
                is_home_game = True
            else:
                opponent_abbr = home_abbr
                is_home_game = False
                
            return {
                'game_date': game_date,
                'opponent_team_name': opponent_abbr,
                'is_home_game': is_home_game
            }
    return None

def get_team_abbrev_from_logs_or_cache(player_id):
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
        team_abbrev = player_info['TEAM_ABBREVIATION'].iloc[0]
        return team_abbrev
    except Exception as e:
        print(f"Error getting team abbreviation: {e}")
        return None

def get_out_teammates(player_name, structured_data):
    teammate_injuries = get_teammate_injuries(player_name, structured_data)
    out_teammates = [teammate for teammate, status in teammate_injuries.items() 
                    if status.lower() in ['out', 'doubtful']]
    return out_teammates

