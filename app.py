import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nba_predictor import (
    get_all_game_logs, 
    predict_against_opponent, 
    load_team_defense_data,
    clean_features,
    train_eval,
    fetch_schedule,
    find_next_game,
    get_player_team,
    get_head_to_head_stats,
    get_team_defensive_stats
)
from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail
import numpy as np
from datetime import datetime
import time
import sqlite3
import os
import unicodedata

DB_PATH = 'nba_data.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS game_logs (
            PLAYER_ID INTEGER,
            PLAYER_NAME TEXT,
            GAME_ID TEXT,
            GAME_DATE TEXT,
            SEASON TEXT,
            MATCHUP TEXT,
            TEAM_ABBREVIATION TEXT,
            PTS REAL,
            REB REAL,
            AST REAL,
            MIN REAL,
            FGM REAL,
            FGA REAL,
            FG3M REAL,
            FG3A REAL,
            FTM REAL,
            FTA REAL,
            OREB REAL,
            DREB REAL,
            STL REAL,
            BLK REAL,
            TOV REAL,
            PF REAL,
            PLUS_MINUS REAL,
            DATA_SOURCE TEXT,
            LAST_UPDATED TEXT,
            PRIMARY KEY (PLAYER_ID, GAME_ID)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS shot_data (
            PLAYER_ID INTEGER,
            PLAYER_NAME TEXT,
            SEASON TEXT,
            GAME_ID TEXT,
            SHOT_ID TEXT,
            LOC_X REAL,
            LOC_Y REAL,
            SHOT_DISTANCE REAL,
            SHOT_TYPE TEXT,
            SHOT_MADE_FLAG INTEGER,
            DATA_SOURCE TEXT,
            LAST_UPDATED TEXT,
            PRIMARY KEY (PLAYER_ID, GAME_ID, SHOT_ID)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS player_metadata (
            PLAYER_ID INTEGER PRIMARY KEY,
            PLAYER_NAME TEXT,
            ROOKIE_SEASON TEXT,
            LAST_SEASON TEXT,
            ACTIVE BOOLEAN,
            LAST_UPDATED TEXT
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS career_trajectory (
            PLAYER_ID INTEGER,
            PLAYER_NAME TEXT,
            SEASON TEXT,
            PTS REAL,
            REB REAL,
            AST REAL,
            MIN REAL,
            FGM REAL,
            FGA REAL,
            FG3M REAL,
            FG3A REAL,
            FTM REAL,
            FTA REAL,
            TOV REAL,
            STL REAL,
            BLK REAL,
            PLUS_MINUS REAL,
            GAMES_PLAYED INTEGER,
            FG_PCT REAL,
            THREE_PCT REAL,
            FT_PCT REAL,
            LAST_UPDATED TEXT,
            PRIMARY KEY (PLAYER_ID, SEASON)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS data_validation (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            TABLE_NAME TEXT,
            VALIDATION_DATE TEXT,
            RECORDS_CHECKED INTEGER,
            RECORDS_VALID INTEGER,
            RECORDS_INVALID INTEGER,
            VALIDATION_NOTES TEXT
        )
    ''')
    
    c.execute('CREATE INDEX IF NOT EXISTS idx_game_logs_player_name ON game_logs(PLAYER_NAME)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_game_logs_season ON game_logs(SEASON)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_game_logs_game_date ON game_logs(GAME_DATE)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_shot_data_player_name ON shot_data(PLAYER_NAME)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_shot_data_season ON shot_data(SEASON)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_career_trajectory_player_name ON career_trajectory(PLAYER_NAME)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_career_trajectory_season ON career_trajectory(SEASON)')
    
    conn.commit()
    conn.close()

def fix_column_names():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        c.execute("PRAGMA table_info(game_logs)")
        current_columns = {row[1]: row[1] for row in c.fetchall()}
        
        correct_columns = {
            'player_id': 'PLAYER_ID',
            'player_name': 'PLAYER_NAME',
            'game_id': 'GAME_ID',
            'game_date': 'GAME_DATE',
            'season': 'SEASON',
            'matchup': 'MATCHUP',
            'team_abbreviation': 'TEAM_ABBREVIATION',
            'pts': 'PTS',
            'reb': 'REB',
            'ast': 'AST',
            'min': 'MIN',
            'fgm': 'FGM',
            'fga': 'FGA',
            'fg3m': 'FG3M',
            'fg3a': 'FG3A',
            'ftm': 'FTM',
            'fta': 'FTA',
            'oreb': 'OREB',
            'dreb': 'DREB',
            'stl': 'STL',
            'blk': 'BLK',
            'tov': 'TOV',
            'pf': 'PF',
            'plus_minus': 'PLUS_MINUS',
            'data_source': 'DATA_SOURCE',
            'last_updated': 'LAST_UPDATED'
        }
        
        for old_name, new_name in correct_columns.items():
            if old_name in current_columns and old_name != new_name:
                c.execute(f'ALTER TABLE game_logs RENAME COLUMN {old_name} TO {new_name}')
        
        conn.commit()
        conn.close()
    except Exception:
        if 'conn' in locals():
            conn.close()

def validate_game_logs_data(df):
    """Validate game logs data before insertion."""
    if df is None or df.empty:
        return False, "Empty dataframe"
    
    column_mapping = {
        'PLAYER_ID': 'PLAYER_ID',
        'GAME_ID': 'GAME_ID',
        'GAME_DATE': 'GAME_DATE',
        'SEASON': 'SEASON',
        'PTS': 'PTS',
        'REB': 'REB',
        'AST': 'AST'
    }
    
    required_stats = ['PTS', 'REB', 'AST']
    if not all(stat in df.columns for stat in required_stats):
        return False, f"Missing required stats: {[stat for stat in required_stats if stat not in df.columns]}"
    
    if df['PTS'].min() < 0 or df['REB'].min() < 0 or df['AST'].min() < 0:
        return False, "Negative values found in stats"
    
    if df['PTS'].max() > 100 or df['REB'].max() > 50 or df['AST'].max() > 50:
        return False, "Unreasonable stat values found"
    
    return True, "Data validation passed"

def validate_shot_data(df):
    if df is None or df.empty:
        return False, "Empty dataframe"
    
    required_columns = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SHOT_MADE_FLAG']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"
    
    return True, "Data validation passed"

def update_player_metadata(player_name):
    """Update or insert player metadata."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            conn.close()
            return
        
        player_id = player_dict[0]['id']
        player_info = players.find_player_by_id(player_id)
        
        if player_info:
            rookie_season = None
            last_season = None
            active = True
            
            if 'year_start' in player_info and player_info['year_start']:
                rookie_season = f"{player_info['year_start']}-{str(int(player_info['year_start']) + 1)[-2:]}"
            
            if 'year_end' in player_info and player_info['year_end']:
                last_season = f"{player_info['year_end']}-{str(int(player_info['year_end']) + 1)[-2:]}"
            
            if 'is_active' in player_info:
                active = player_info['is_active']
            
            c.execute('''
                INSERT OR REPLACE INTO player_metadata 
                (player_id, player_name, rookie_season, last_season, active, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (player_id, player_name, rookie_season, last_season, active, datetime.now().isoformat()))
            
            conn.commit()
        
        conn.close()
    except Exception as e:
        st.error(f"Error updating player metadata: {str(e)}")
        if 'conn' in locals():
            conn.close()

def log_data_validation(table_name, records_checked, records_valid, records_invalid, notes):
    """Log data validation results."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO data_validation 
        (table_name, validation_date, records_checked, records_valid, records_invalid, validation_notes)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (table_name, datetime.now().isoformat(), records_checked, records_valid, records_invalid, notes))
    
    conn.commit()
    conn.close()

def cleanup_old_data():
    """Remove duplicate entries and clean up old data."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        DELETE FROM game_logs 
        WHERE rowid NOT IN (
            SELECT MIN(rowid) 
            FROM game_logs 
            GROUP BY player_id, game_id
        )
    ''')
    
    c.execute('''
        DELETE FROM shot_data 
        WHERE rowid NOT IN (
            SELECT MIN(rowid) 
            FROM shot_data 
            GROUP BY player_id, game_id, shot_id
        )
    ''')
    
    ten_years_ago = (datetime.now().year - 10)
    c.execute('DELETE FROM game_logs WHERE CAST(SUBSTR(season, 1, 4) AS INTEGER) < ?', (ten_years_ago,))
    c.execute('DELETE FROM shot_data WHERE CAST(SUBSTR(season, 1, 4) AS INTEGER) < ?', (ten_years_ago,))
    
    conn.commit()
    conn.close()

def get_player_data_from_db(player_name, start_season=None, end_season=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(game_logs)")
        db_columns = {row[1]: row[1] for row in cursor.fetchall()}
        
        select_columns = [col for col in db_columns.keys()]
        query = f'''
            SELECT {', '.join(select_columns)} FROM game_logs 
            WHERE PLAYER_NAME = ? 
        '''
        params = [player_name]
        
        if start_season:
            query += ' AND SEASON >= ?'
            params.append(start_season)
        if end_season:
            query += ' AND SEASON <= ?'
            params.append(end_season)
        
        query += ' ORDER BY GAME_DATE'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            return df
        return None
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

def save_player_data_to_db(df, player_name):
    if df is None or df.empty:
        return
    
    is_valid, message = validate_game_logs_data(df)
    if not is_valid:
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            conn.close()
            return
        
        player_id = player_dict[0]['id']
        
        df_to_save = df.copy()
        
        df_to_save['PLAYER_ID'] = player_id
        df_to_save['PLAYER_NAME'] = player_name
        df_to_save['DATA_SOURCE'] = 'NBA API'
        df_to_save['LAST_UPDATED'] = datetime.now().isoformat()
        
        if 'GAME_ID' not in df_to_save.columns:
            df_to_save['GAME_ID'] = df_to_save.index.astype(str)
        
        if 'TEAM_ABBREVIATION' not in df_to_save.columns:
            df_to_save['TEAM_ABBREVIATION'] = 'UNK'
            
        if 'MATCHUP' not in df_to_save.columns:
            df_to_save['MATCHUP'] = 'Unknown'
            
        if 'GAME_DATE' not in df_to_save.columns:
            df_to_save['GAME_DATE'] = pd.Timestamp.now()
            
        if 'SEASON' not in df_to_save.columns:
            df_to_save['SEASON'] = '2023-24'
        
        columns = {
            'PLAYER_ID': 'player_id',
            'PLAYER_NAME': 'player_name',
            'GAME_ID': 'game_id',
            'GAME_DATE': 'game_date',
            'SEASON': 'season',
            'MATCHUP': 'matchup',
            'TEAM_ABBREVIATION': 'team_abbreviation',
            'PTS': 'pts',
            'REB': 'reb',
            'AST': 'ast',
            'MIN': 'min',
            'FGM': 'fgm',
            'FGA': 'fga',
            'FG3M': 'fg3m',
            'FG3A': 'fg3a',
            'FTM': 'ftm',
            'FTA': 'fta',
            'OREB': 'oreb',
            'DREB': 'dreb',
            'STL': 'stl',
            'BLK': 'blk',
            'TOV': 'tov',
            'PF': 'pf',
            'PLUS_MINUS': 'plus_minus',
            'DATA_SOURCE': 'data_source',
            'LAST_UPDATED': 'last_updated'
        }
        
        df_final = pd.DataFrame()
        for db_col, api_col in columns.items():
            if api_col in df_to_save.columns:
                df_final[db_col] = df_to_save[api_col]
            elif db_col in df_to_save.columns:
                df_final[db_col] = df_to_save[db_col]
            else:
                df_final[db_col] = None
        
        existing_data = pd.read_sql_query(
            "SELECT player_id, game_id FROM game_logs WHERE player_id = ?",
            conn,
            params=[player_id]
        )
        
        if not existing_data.empty:
            existing_keys = set(zip(existing_data['player_id'], existing_data['game_id']))
            new_data = df_final[~df_final.apply(lambda x: (x['player_id'], x['game_id']) in existing_keys, axis=1)]
            
            if not new_data.empty:
                new_data.to_sql('game_logs', conn, if_exists='append', index=False)
                log_data_validation('game_logs', len(new_data), len(new_data), 0, "New data added")
        else:
            df_final.to_sql('game_logs', conn, if_exists='append', index=False)
            log_data_validation('game_logs', len(df_final), len(df_final), 0, "New data added")
        
        conn.close()
    except Exception:
        if 'conn' in locals():
            conn.close()

def get_shot_data_from_db(player_name, season):
    """Retrieve shot data from the database with enhanced error handling."""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        query = '''
            SELECT * FROM shot_data 
            WHERE player_name = ? AND season = ?
        '''
        
        df = pd.read_sql_query(query, conn, params=[player_name, season])
        conn.close()
        
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

def save_shot_data_to_db(df, player_name, season):
    if df is None or df.empty:
        return
    
    is_valid, _ = validate_shot_data(df)
    if not is_valid:
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            conn.close()
            return
        
        player_id = player_dict[0]['id']
        
        df['PLAYER_ID'] = player_id
        df['PLAYER_NAME'] = player_name
        df['SEASON'] = season
        df['DATA_SOURCE'] = 'NBA API'
        df['LAST_UPDATED'] = datetime.now().isoformat()
        
        if 'SHOT_ID' not in df.columns:
            df['SHOT_ID'] = df.index.astype(str)
        
        columns = {
            'PLAYER_ID': 'PLAYER_ID',
            'PLAYER_NAME': 'PLAYER_NAME',
            'SEASON': 'SEASON',
            'GAME_ID': 'GAME_ID',
            'SHOT_ID': 'SHOT_ID',
            'LOC_X': 'LOC_X',
            'LOC_Y': 'LOC_Y',
            'SHOT_DISTANCE': 'SHOT_DISTANCE',
            'SHOT_TYPE': 'SHOT_TYPE',
            'SHOT_MADE_FLAG': 'SHOT_MADE_FLAG',
            'DATA_SOURCE': 'DATA_SOURCE',
            'LAST_UPDATED': 'LAST_UPDATED'
        }
        
        df_to_save = df[list(columns.keys())].rename(columns=columns)
        
        df_to_save.to_sql('shot_data', conn, if_exists='append', index=False)
        
        log_data_validation('shot_data', len(df), len(df), 0, "New data added")
        
        conn.close()
    except Exception:
        if 'conn' in locals():
            conn.close()

def get_career_trajectory_from_db(player_name):
    """Retrieve career trajectory data from the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = '''
            SELECT * FROM career_trajectory 
            WHERE PLAYER_NAME = ?
            ORDER BY SEASON
        '''
        df = pd.read_sql_query(query, conn, params=[player_name])
        conn.close()
        return df if not df.empty else None
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return None

def save_career_trajectory_to_db(df, player_name):
    """Save career trajectory data to the database."""
    if df is None or df.empty:
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            conn.close()
            return
        
        player_id = player_dict[0]['id']
        
        df_to_save = df.copy()
        df_to_save['PLAYER_ID'] = player_id
        df_to_save['PLAYER_NAME'] = player_name
        df_to_save['LAST_UPDATED'] = datetime.now().isoformat()
        
        # Calculate shooting percentages
        df_to_save['FG_PCT'] = (df_to_save['FGM'] / df_to_save['FGA'] * 100)
        df_to_save['THREE_PCT'] = (df_to_save['FG3M'] / df_to_save['FG3A'] * 100)
        df_to_save['FT_PCT'] = (df_to_save['FTM'] / df_to_save['FTA'] * 100)
        
        # Delete existing records for this player
        c = conn.cursor()
        c.execute('DELETE FROM career_trajectory WHERE PLAYER_ID = ?', (player_id,))
        
        # Save new data
        df_to_save.to_sql('career_trajectory', conn, if_exists='append', index=False)
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving career trajectory data: {str(e)}")
        if 'conn' in locals():
            conn.close()

init_db()
fix_column_names()
cleanup_old_data()

st.set_page_config(layout="wide")

def normalize_name(name):
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII').lower()

@st.cache_data
def get_player_list():
    player_list = players.get_active_players()
    normalized_players = []
    for player in player_list:
        original_name = player['full_name']
        # Create multiple variations of the name for better matching
        normalized_name = normalize_name(original_name)
        # Remove special characters but keep spaces
        simplified_name = ''.join(c for c in original_name if c.isalnum() or c.isspace()).lower()
        normalized_players.append({
            'original_name': original_name,
            'normalized_name': normalized_name,
            'simplified_name': simplified_name
        })
    return sorted(normalized_players, key=lambda x: x['normalized_name'])

def find_player_by_name(search_name):
    if not search_name:
        return None
    
    # Normalize the search input
    search_name = search_name.lower()
    search_name_no_special = ''.join(c for c in search_name if c.isalnum() or c.isspace())
    
    player_list = get_player_list()
    
    # First try exact match
    for player in player_list:
        if player['normalized_name'] == search_name or player['simplified_name'] == search_name:
            return player['original_name']
    
    # Then try partial match
    matches = []
    for player in player_list:
        if (search_name in player['normalized_name'] or 
            search_name_no_special in player['simplified_name']):
            matches.append(player['original_name'])
    
    if matches:
        return matches[0]  # Return the first match
    
    return None

@st.cache_data
def get_player_rookie_and_last_season(player_name):
    normalized_name = find_player_by_name(player_name)
    if normalized_name:
        player_name = normalized_name
    player_dict = players.find_players_by_full_name(player_name)
    if not player_dict:
        return None, None
    player_id = player_dict[0]['id']
    player_info = players.find_player_by_id(player_id)
    if player_info and 'year_start' in player_info and 'year_end' in player_info:
        year_start = int(player_info['year_start'])
        year_end = int(player_info['year_end'])
        rookie_season = f"{year_start}-{str(year_start + 1)[-2:]}"
        last_season = f"{year_end}-{str(year_end + 1)[-2:]}"
        return rookie_season, last_season
    return None, None

@st.cache_data(ttl=3600)
def load_player_data(player_name, start_season=None, end_season=None):
    normalized_name = find_player_by_name(player_name)
    if normalized_name:
        player_name = normalized_name
    try:
        df = get_player_data_from_db(player_name, start_season, end_season)
        
        required_columns = ['MATCHUP', 'GAME_DATE', 'SEASON', 'PTS', 'REB', 'AST']
        if df is not None and not df.empty:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                df = None
        
        if df is None or df.empty:
            df = get_all_game_logs(player_name, '2010-11', '2024-25')
            if df.empty:
                st.error("No data found for this player")
                return None
            
            df = df.sort_values('GAME_DATE')
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            if 'SEASON' in df.columns:
                seasons = sorted(df['SEASON'].unique())
                true_start = seasons[0]
                true_end = seasons[-1]
                df = df[df['SEASON'].between(true_start, true_end)]
            
            save_player_data_to_db(df, player_name)
        
        return df
    except Exception:
        df = get_all_game_logs(player_name, '2010-11', '2024-25')
        if df.empty:
            st.error("No data found for this player")
            return None
        
        df = df.sort_values('GAME_DATE')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        if 'SEASON' in df.columns:
            seasons = sorted(df['SEASON'].unique())
            true_start = seasons[0]
            true_end = seasons[-1]
            df = df[df['SEASON'].between(true_start, true_end)]
        
        save_player_data_to_db(df, player_name)
        
        return df

@st.cache_data
def get_shot_data(player_name, season='2024-25', max_retries=3):
    normalized_name = find_player_by_name(player_name)
    if normalized_name:
        player_name = normalized_name
    shot_data = get_shot_data_from_db(player_name, season)
    
    if shot_data is None or shot_data.empty:
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            return None
        
        player_id = player_dict[0]['id']
        
        for attempt in range(max_retries):
            try:
                shot_data = shotchartdetail.ShotChartDetail(
                    player_id=player_id,
                    team_id=0,
                    season_nullable=season,
                    context_measure_simple='FGA',
                    season_type_all_star='Regular Season',
                    timeout=30
                )
                df = shot_data.get_data_frames()[0]
                
                save_shot_data_to_db(df, player_name, season)
                
                return df
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Attempt {attempt + 1} failed. Retrying...")
                    time.sleep(2)
                else:
                    st.error(f"Failed to fetch shot data after {max_retries} attempts: {e}")
                    return None
    
    return shot_data

st.title("NBA Player Performance Predictor")
st.markdown("### Visualize player stats and get performance predictions")

st.sidebar.header("Player Selection")
player_list = get_player_list()

# Create a custom search function for the selectbox
def search_players(search_term):
    if not search_term:
        return [p['original_name'] for p in player_list]
    
    search_term = search_term.lower()
    search_term_no_special = ''.join(c for c in search_term if c.isalnum() or c.isspace())
    
    matches = []
    for player in player_list:
        if (search_term in player['normalized_name'] or 
            search_term_no_special in player['simplified_name']):
            matches.append(player['original_name'])
    
    return matches if matches else ["No matches found"]

player_names = ["Select a player..."] + [p['original_name'] for p in player_list]

player_name = st.sidebar.selectbox(
    "Select Player",
    options=player_names,
    index=0,
    key="player_select",
    help="Type to search. Special characters will be matched automatically (e.g., 'Doncic' will match 'Dončić')"
)

if player_name and player_name != "Select a player...":
    normalized_name = find_player_by_name(player_name)
    if normalized_name:
        player_name = normalized_name

df = None
if player_name and player_name != "Select a player...":
    df = load_player_data(player_name)

if df is not None:
    df_clean = clean_features(df, player_name)
    
    available_seasons = sorted(df_clean['SEASON'].unique(), reverse=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Next Game & Matchup", "Game-by-Game Performance", "Shot Distance Analysis", "Career Trajectory", "Player Comparison"])
    
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Overview"
    
    if 'selected_season' not in st.session_state:
        st.session_state.selected_season = available_seasons[0]
    
    if tab1:
        st.session_state.active_tab = "Overview"
    elif tab2:
        st.session_state.active_tab = "Next Game & Matchup"
    elif tab3:
        st.session_state.active_tab = "Game-by-Game Performance"
    elif tab4:
        st.session_state.active_tab = "Shot Distance Analysis"
    elif tab5:
        st.session_state.active_tab = "Career Trajectory"
    elif tab6:
        st.session_state.active_tab = "Player Comparison"
    
    # Add season selector in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Season Selection")
    selected_season = st.sidebar.selectbox(
        "Select Season",
        options=available_seasons,
        index=available_seasons.index(st.session_state.selected_season),
        key="season_selector"
    )
    st.session_state.selected_season = selected_season
    
    if st.session_state.active_tab == "Overview":
        with tab1:
            season_data = df_clean[df_clean['SEASON'] == st.session_state.selected_season]
            
            st.subheader(f"Season Averages - {st.session_state.selected_season}")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Points", f"{season_data['PTS'].mean():.1f}")
            with col2:
                st.metric("Rebounds", f"{season_data['REB'].mean():.1f}")
            with col3:
                st.metric("Assists", f"{season_data['AST'].mean():.1f}")
            with col4:
                st.metric("Minutes", f"{season_data['MIN'].mean():.1f}")
            with col5:
                teams = season_data['TEAM_ABBREVIATION'].unique()
                if len(teams) > 1:
                    # Create a compact display for the metric
                    if len(teams) > 2:
                        team_display = f"{teams[0]} → {teams[-1]}"
                        full_sequence = " → ".join(teams)
                        st.metric("Teams", team_display, help=f"Player was traded multiple times: {full_sequence}")
                    else:
                        team_display = " → ".join(teams)
                        st.metric("Teams", team_display, help="Player was traded during the season")
                else:
                    team = teams[0] if not season_data.empty else "N/A"
                    st.metric("Team", team)
            
            st.subheader(f"Shooting Efficiency - {st.session_state.selected_season}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'FGM' in season_data.columns and 'FGA' in season_data.columns:
                    fg_pct = (season_data['FGM'].sum() / season_data['FGA'].sum() * 100)
                    st.metric("Field Goal %", f"{fg_pct:.1f}%")
            with col2:
                if 'FG3M' in season_data.columns and 'FG3A' in season_data.columns:
                    three_pct = (season_data['FG3M'].sum() / season_data['FG3A'].sum() * 100)
                    st.metric("3-Point %", f"{three_pct:.1f}%")
            with col3:
                if 'FTM' in season_data.columns and 'FTA' in season_data.columns:
                    ft_pct = (season_data['FTM'].sum() / season_data['FTA'].sum() * 100)
                    st.metric("Free Throw %", f"{ft_pct:.1f}%")
            with col4:
                if 'FGM' in season_data.columns and 'FGA' in season_data.columns:
                    efg_pct = ((season_data['FGM'].sum() + 0.5 * season_data['FG3M'].sum()) / season_data['FGA'].sum() * 100)
                    st.metric("Effective FG%", f"{efg_pct:.1f}%")
            
            st.subheader(f"Advanced Stats - {st.session_state.selected_season}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'TOV' in season_data.columns:
                    st.metric("Turnovers", f"{season_data['TOV'].mean():.1f}")
            with col2:
                if 'STL' in season_data.columns:
                    st.metric("Steals", f"{season_data['STL'].mean():.1f}")
            with col3:
                if 'BLK' in season_data.columns:
                    st.metric("Blocks", f"{season_data['BLK'].mean():.1f}")
            with col4:
                if 'PF' in season_data.columns:
                    st.metric("Fouls", f"{season_data['PF'].mean():.1f}")
            
            st.subheader(f"Season Highs - {st.session_state.selected_season}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'PTS' in season_data.columns:
                    max_pts = season_data['PTS'].max()
                    max_pts_game = season_data[season_data['PTS'] == max_pts].iloc[0]
                    game_date = max_pts_game['GAME_DATE'].strftime('%m/%d/%y')
                    st.metric("Points", f"{max_pts:.0f}", f"{game_date} {max_pts_game['MATCHUP']}")
                if 'REB' in season_data.columns:
                    max_reb = season_data['REB'].max()
                    max_reb_game = season_data[season_data['REB'] == max_reb].iloc[0]
                    game_date = max_reb_game['GAME_DATE'].strftime('%m/%d/%y')
                    st.metric("Rebounds", f"{max_reb:.0f}", f"{game_date} {max_reb_game['MATCHUP']}")
            
            with col2:
                if 'AST' in season_data.columns:
                    max_ast = season_data['AST'].max()
                    max_ast_game = season_data[season_data['AST'] == max_ast].iloc[0]
                    game_date = max_ast_game['GAME_DATE'].strftime('%m/%d/%y')
                    st.metric("Assists", f"{max_ast:.0f}", f"{game_date} {max_ast_game['MATCHUP']}")
                if 'STL' in season_data.columns:
                    max_stl = season_data['STL'].max()
                    max_stl_game = season_data[season_data['STL'] == max_stl].iloc[0]
                    game_date = max_stl_game['GAME_DATE'].strftime('%m/%d/%y')
                    st.metric("Steals", f"{max_stl:.0f}", f"{game_date} {max_stl_game['MATCHUP']}")
            
            with col3:
                if 'BLK' in season_data.columns:
                    max_blk = season_data['BLK'].max()
                    max_blk_game = season_data[season_data['BLK'] == max_blk].iloc[0]
                    game_date = max_blk_game['GAME_DATE'].strftime('%m/%d/%y')
                    st.metric("Blocks", f"{max_blk:.0f}", f"{game_date} {max_blk_game['MATCHUP']}")
                if 'FG3M' in season_data.columns:
                    max_3pm = season_data['FG3M'].max()
                    max_3pm_game = season_data[season_data['FG3M'] == max_3pm].iloc[0]
                    game_date = max_3pm_game['GAME_DATE'].strftime('%m/%d/%y')
                    st.metric("3-Pointers Made", f"{max_3pm:.0f}", f"{game_date} {max_3pm_game['MATCHUP']}")
            
            with col4:
                if 'FGM' in season_data.columns and 'FGA' in season_data.columns:
                    max_fg = season_data['FGM'].max()
                    max_fg_game = season_data[season_data['FGM'] == max_fg].iloc[0]
                    game_date = max_fg_game['GAME_DATE'].strftime('%m/%d/%y')
                    st.metric("Field Goals Made", f"{max_fg:.0f}", f"{game_date} {max_fg_game['MATCHUP']}")
                if 'FTM' in season_data.columns:
                    max_ft = season_data['FTM'].max()
                    max_ft_game = season_data[season_data['FTM'] == max_ft].iloc[0]
                    game_date = max_ft_game['GAME_DATE'].strftime('%m/%d/%y')
                    st.metric("Free Throws Made", f"{max_ft:.0f}", f"{game_date} {max_ft_game['MATCHUP']}")
            
            st.subheader(f"Advanced Metrics - {st.session_state.selected_season}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if all(col in season_data.columns for col in ['PTS', 'FGM', 'FGA', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'MIN']):
                    per = (season_data['PTS'].mean() * 1.0 + 
                          season_data['FGM'].mean() * 0.5 - 
                          season_data['FGA'].mean() * 0.5 + 
                          season_data['FTM'].mean() * 0.5 - 
                          season_data['FTA'].mean() * 0.5 + 
                          season_data['OREB'].mean() * 0.5 + 
                          season_data['DREB'].mean() * 0.5 + 
                          season_data['AST'].mean() * 0.5 + 
                          season_data['STL'].mean() * 1.0 + 
                          season_data['BLK'].mean() * 1.0 - 
                          season_data['TOV'].mean() * 1.0 - 
                          season_data['PF'].mean() * 0.5)
                    st.metric("PER", f"{per:.1f}", help="Player Efficiency Rating: A measure of per-minute production standardized such that the league average is 15.0")
            
            with col2:
                if all(col in season_data.columns for col in ['FGA', 'FTA', 'TOV', 'MIN']):
                    possessions = season_data['FGA'].sum() + 0.44 * season_data['FTA'].sum() + season_data['TOV'].sum()
                    minutes = season_data['MIN'].sum()
                    if minutes > 0:
                        possessions_per_game = (minutes / 48) * 100
                        usg_rate = (possessions / possessions_per_game) * 100
                        st.metric("Usage Rate", f"{usg_rate:.1f}%", help="Percentage of team plays used by a player while on the floor")
            
            with col3:
                if all(col in season_data.columns for col in ['PTS', 'FGA', 'FTA']):
                    fga_fta = season_data['FGA'].sum() + 0.44 * season_data['FTA'].sum()
                    if fga_fta > 0:
                        ts_pct = season_data['PTS'].sum() / (2 * fga_fta) * 100
                        st.metric("True Shooting %", f"{ts_pct:.1f}%", help="Measure of shooting efficiency that takes into account field goals, 3-point field goals, and free throws")
            
            with col4:
                if all(col in season_data.columns for col in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'PF']):
                    bpm = (season_data['PTS'].mean() * 0.274 + 
                          season_data['AST'].mean() * 0.7 + 
                          season_data['REB'].mean() * 0.5 + 
                          season_data['STL'].mean() * 0.7 + 
                          season_data['BLK'].mean() * 0.7 - 
                          season_data['TOV'].mean() - 
                          season_data['PF'].mean() * 0.4)
                    st.metric("Box Plus/Minus", f"{bpm:+.1f}", help="Box score estimate of the points per 100 possessions a player contributed above a league-average player")
            
            st.subheader(f"Impact Metrics - {st.session_state.selected_season}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'PLUS_MINUS' in season_data.columns:
                    plus_minus = season_data['PLUS_MINUS'].mean()
                    plus_minus_str = f"{plus_minus:+.1f}" if plus_minus != 0 else "0.0"
                    st.metric("Plus/Minus", plus_minus_str, help="Average point differential when the player is on the court")
            with col2:
                if all(col in season_data.columns for col in ['AST', 'TOV']):
                    ast_to = season_data['AST'].sum() / season_data['TOV'].sum() if season_data['TOV'].sum() > 0 else 0
                    st.metric("AST/TO Ratio", f"{ast_to:.1f}", help="Ratio of assists to turnovers, measuring playmaking efficiency")
            with col3:
                if all(col in season_data.columns for col in ['PTS', 'AST', 'REB']):
                    st.metric("PTS+AST+REB", f"{season_data['PTS'].mean() + season_data['AST'].mean() + season_data['REB'].mean():.1f}", help="Sum of points, assists, and rebounds per game")
            with col4:
                if all(col in season_data.columns for col in ['FGM', 'FGA']):
                    efg_pct = ((season_data['FGM'].sum() + 0.5 * season_data['FG3M'].sum()) / season_data['FGA'].sum() * 100)
                    st.metric("Effective FG%", f"{efg_pct:.1f}%", help="Field goal percentage that accounts for 3-pointers being worth more than 2-pointers")
            
            # Add Custom Stat Combinations section
            st.subheader("Custom Stat Combinations")
            st.markdown("""
            Select from predefined statistical combinations to analyze player performance.
            """)
            
            # Define preset combinations
            preset_combinations = {
                'Efficiency Score': {
                    'name': 'Efficiency Score',
                    'formula': lambda df: df['PTS'] + df['REB'] + df['AST'] - df['TOV'],
                    'formula_str': 'PTS + REB + AST - TOV',
                    'description': 'Combines scoring, rebounding, and playmaking while penalizing turnovers',
                    'format': '{:.1f}'
                },
                'Defensive Impact': {
                    'name': 'Defensive Impact',
                    'formula': lambda df: df['STL'] + df['BLK'],
                    'formula_str': 'STL + BLK',
                    'description': 'Measures defensive contributions through steals and blocks',
                    'format': '{:.1f}'
                },
                'Scoring Efficiency': {
                    'name': 'Scoring Efficiency',
                    'formula': lambda df: df['PTS'] / df['FGA'].replace(0, np.nan),
                    'formula_str': 'PTS / FGA',
                    'description': 'Points per field goal attempt',
                    'format': '{:.3f}'
                },
                'Playmaking Impact': {
                    'name': 'Playmaking Impact',
                    'formula': lambda df: df['AST'] / df['TOV'].replace(0, np.nan),
                    'formula_str': 'AST / TOV',
                    'description': 'Assist to turnover ratio',
                    'format': '{:.2f}'
                },
                'Overall Impact': {
                    'name': 'Overall Impact',
                    'formula': lambda df: df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK'] - df['TOV'],
                    'formula_str': 'PTS + REB + AST + STL + BLK - TOV',
                    'description': 'Comprehensive measure of player impact across all major categories',
                    'format': '{:.1f}'
                }
            }
            
            # Create dropdown for preset combinations
            selected_combination = st.selectbox(
                "Select Stat Combination",
                options=list(preset_combinations.keys()),
                help="Choose a predefined statistical combination to analyze"
            )
            
            # Calculate and display the selected combination
            if selected_combination:
                combination = preset_combinations[selected_combination]
                try:
                    # Calculate the stat
                    custom_stat = combination['formula'](season_data)
                    avg_value = custom_stat.mean()
                    
                    # Display the metric
                    st.metric(
                        f"{combination['name']} ({combination['formula_str']})",
                        combination['format'].format(avg_value),
                        help=combination['description']
                    )
                    
                    # Create visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=season_data['GAME_DATE'],
                        y=custom_stat,
                        mode='lines+markers',
                        name=combination['name'],
                        line=dict(width=2, color='#FF4B4B'),
                        marker=dict(size=8),
                        hovertemplate="<b>Date</b>: %{x|%b %d}<br>" +
                                    f"<b>{combination['name']}</b>: %{{y:.1f}}<br>" +
                                    "<b>Opponent</b>: %{text}<extra></extra>",
                        text=season_data['MATCHUP']
                    ))
                    
                    # Add rolling average
                    window = 5
                    rolling_avg = custom_stat.rolling(window=window, min_periods=1).mean()
                    
                    fig.add_trace(go.Scatter(
                        x=season_data['GAME_DATE'],
                        y=rolling_avg,
                        mode='lines',
                        name=f'{window}-Game Average',
                        line=dict(width=2, color='#00CC96', dash='dash'),
                        hovertemplate="<b>Date</b>: %{x|%b %d}<br>" +
                                    f"<b>{window}-Game Average</b>: %{{y:.1f}}<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        title=f"{combination['name']} Trend",
                        xaxis_title="Date",
                        yaxis_title="Value",
                        hovermode='x unified',
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error calculating custom stat: {str(e)}")
            
            st.markdown("""
            <div style="background-color: rgba(240, 242, 246, 0.05); padding: 15px; border-radius: 8px; margin-top: 20px; color: #262730;">
                <details>
                    <summary style="color: white; font-weight: bold; cursor: pointer;">Tips for Creating Custom Stats</summary>
                    <p style="color: rgba(255, 255, 255, 0.7); margin-top: 10px;">Here are some useful custom stat combinations:</p>
                    <ul style="color: rgba(255, 255, 255, 0.7);">
                        <li><b>Efficiency Score</b>: PTS + REB + AST - TOV</li>
                        <li><b>Defensive Impact</b>: STL + BLK</li>
                        <li><b>Scoring Efficiency</b>: PTS / FGA</li>
                        <li><b>Playmaking Impact</b>: AST / TOV</li>
                        <li><b>Overall Impact</b>: PTS + REB + AST + STL + BLK - TOV</li>
                    </ul>
                </details>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader(f"Home vs Away Performance - {st.session_state.selected_season}")
            
            home_games = season_data[season_data['MATCHUP'].str.contains('vs')]
            away_games = season_data[season_data['MATCHUP'].str.contains('@')]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                home_pts = home_games['PTS'].mean()
                away_pts = away_games['PTS'].mean()
                st.metric("Points (Home/Away)", f"{home_pts:.1f}/{away_pts:.1f}", f"{home_pts - away_pts:+.1f}")
            with col2:
                home_reb = home_games['REB'].mean()
                away_reb = away_games['REB'].mean()
                st.metric("Rebounds (Home/Away)", f"{home_reb:.1f}/{away_reb:.1f}", f"{home_reb - away_reb:+.1f}")
            with col3:
                home_ast = home_games['AST'].mean()
                away_ast = away_games['AST'].mean()
                st.metric("Assists (Home/Away)", f"{home_ast:.1f}/{away_ast:.1f}", f"{home_ast - away_ast:+.1f}")
            with col4:
                home_min = home_games['MIN'].mean()
                away_min = away_games['MIN'].mean()
                st.metric("Minutes (Home/Away)", f"{home_min:.1f}/{away_min:.1f}", f"{home_min - away_min:+.1f}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'FGM' in home_games.columns and 'FGA' in home_games.columns:
                    home_fg = (home_games['FGM'].sum() / home_games['FGA'].sum() * 100)
                    away_fg = (away_games['FGM'].sum() / away_games['FGA'].sum() * 100)
                    st.metric("FG% (Home/Away)", f"{home_fg:.1f}%/{away_fg:.1f}%", f"{home_fg - away_fg:+.1f}%")
            with col2:
                if 'FG3M' in home_games.columns and 'FG3A' in home_games.columns:
                    home_3pt = (home_games['FG3M'].sum() / home_games['FG3A'].sum() * 100)
                    away_3pt = (away_games['FG3M'].sum() / away_games['FG3A'].sum() * 100)
                    st.metric("3PT% (Home/Away)", f"{home_3pt:.1f}%/{away_3pt:.1f}%", f"{home_3pt - away_3pt:+.1f}%")
            with col3:
                if 'FTM' in home_games.columns and 'FTA' in home_games.columns:
                    home_ft = (home_games['FTM'].sum() / home_games['FTA'].sum() * 100)
                    away_ft = (away_games['FTM'].sum() / away_games['FTA'].sum() * 100)
                    st.metric("FT% (Home/Away)", f"{home_ft:.1f}%/{away_ft:.1f}%", f"{home_ft - away_ft:+.1f}%")
            with col4:
                if all(col in home_games.columns for col in ['FGM', 'FGA', 'FG3M']):
                    home_efg = ((home_games['FGM'].sum() + 0.5 * home_games['FG3M'].sum()) / home_games['FGA'].sum() * 100)
                    away_efg = ((away_games['FGM'].sum() + 0.5 * away_games['FG3M'].sum()) / away_games['FGA'].sum() * 100)
                    st.metric("eFG% (Home/Away)", f"{home_efg:.1f}%/{away_efg:.1f}%", f"{home_efg - away_efg:+.1f}%")
    
    model, x_test, y_test, predictions = train_eval(df_clean)
    
    with tab2:
        st.subheader("Next Game Prediction & Matchup Insights")
        if model is None:
            st.warning("Not enough data available for model training. Showing visualizations only.")
        else:
            try:
                team_id = get_player_team(player_name)
                schedule = fetch_schedule()
                next_game = find_next_game(schedule, team_id) if team_id and schedule else None
                
                if not team_id:
                    st.warning("Could not find team information for the player.")
                elif not schedule:
                    current_date = datetime.now()
                    if current_date.month in [4, 5, 6]:
                        st.info("No upcoming playoff games found in the schedule. The player's team may have been eliminated or the next game hasn't been scheduled yet.")
                    else:
                        st.info("No upcoming games found in the schedule. The NBA season may be in the offseason.")
                elif not next_game:
                    st.info("No upcoming games found in the schedule for the next 7 days.")
                else:
                    opponent = next_game.get('opponent_team_name')
                    if not opponent:
                        st.warning("Could not determine opponent information.")
                    else:
                        game_date = next_game.get('game_date', datetime.now()).strftime('%B %d, %Y')
                        location = "Home" if next_game.get('is_home_game', False) else "Away"
                        current_date = datetime.now()
                        game_type = "Playoff" if current_date.month in [4, 5, 6] else "Regular Season"
                        st.write(f"**Next Game**: {game_type} {location} vs {opponent} on {game_date}")
                        
                        team_stats_df = load_team_defense_data()
                        prediction = predict_against_opponent(player_name, opponent, season_data, model, team_stats_df)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Predicted Points", f"{prediction['Predicted PTS']:.1f}", f"{prediction['Predicted PTS'] - prediction['Season Avg PTS']:+.1f} vs season avg")
                        with col2:
                            st.metric("Predicted Rebounds", f"{prediction['Predicted REB']:.1f}", f"{prediction['Predicted REB'] - prediction['Season Avg REB']:+.1f} vs season avg")
                        with col3:
                            st.metric("Predicted Assists", f"{prediction['Predicted AST']:.1f}", f"{prediction['Predicted AST'] - prediction['Season Avg AST']:+.1f} vs season avg")
                        
                        st.markdown("---")
                        
                        st.subheader(f"Historic Performance vs {opponent}")
                        
                        h2h_stats = get_head_to_head_stats(player_name, opponent, season_data['SEASON'].iloc[0])
                        
                        if h2h_stats:
                            st.markdown("### Career vs Opponent")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Games", h2h_stats['games_played'])
                            with col2:
                                st.metric("Points", f"{h2h_stats['avg_pts']:.1f}")
                            with col3:
                                st.metric("Rebounds", f"{h2h_stats['avg_reb']:.1f}")
                            with col4:
                                st.metric("Assists", f"{h2h_stats['avg_ast']:.1f}")
                            
                            if h2h_stats['avg_fg_pct'] is not None:
                                fg_pct = h2h_stats['avg_fg_pct'] * 100
                                st.markdown(f"**Career FG% vs {opponent}**: {fg_pct:.1f}%")
                            
                            st.markdown("---")
                            
                            st.markdown("### Last Game vs Opponent")
                            
                            # Display game info
                            st.markdown(f"**{h2h_stats['last_game_date']}** - {h2h_stats['last_game_type']}")
                            st.markdown(f"**Matchup**: {h2h_stats['last_game_matchup']}")
                            
                            # Main stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if h2h_stats['last_game_pts'] is not None:
                                    st.metric("Points", f"{h2h_stats['last_game_pts']:.0f}")
                            with col2:
                                if h2h_stats['last_game_reb'] is not None:
                                    st.metric("Rebounds", f"{h2h_stats['last_game_reb']:.0f}")
                            with col3:
                                if h2h_stats['last_game_ast'] is not None:
                                    st.metric("Assists", f"{h2h_stats['last_game_ast']:.0f}")
                            
                            # Shooting stats
                            st.markdown("#### Shooting")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                fg_pct = (h2h_stats['last_game_fgm'] / h2h_stats['last_game_fga'] * 100) if h2h_stats['last_game_fga'] > 0 else 0
                                st.metric("FG%", f"{fg_pct:.1f}%", f"{h2h_stats['last_game_fgm']}/{h2h_stats['last_game_fga']}")
                            with col2:
                                fg3_pct = (h2h_stats['last_game_fg3m'] / h2h_stats['last_game_fg3a'] * 100) if h2h_stats['last_game_fg3a'] > 0 else 0
                                st.metric("3P%", f"{fg3_pct:.1f}%", f"{h2h_stats['last_game_fg3m']}/{h2h_stats['last_game_fg3a']}")
                            with col3:
                                ft_pct = (h2h_stats['last_game_ftm'] / h2h_stats['last_game_fta'] * 100) if h2h_stats['last_game_fta'] > 0 else 0
                                st.metric("FT%", f"{ft_pct:.1f}%", f"{h2h_stats['last_game_ftm']}/{h2h_stats['last_game_fta']}")
                            with col4:
                                st.metric("Minutes", f"{h2h_stats['last_game_min']:.0f}")
                            
                            # Additional stats
                            st.markdown("#### Additional Stats")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Steals", f"{h2h_stats['last_game_stl']:.0f}")
                            with col2:
                                st.metric("Blocks", f"{h2h_stats['last_game_blk']:.0f}")
                            with col3:
                                st.metric("Turnovers", f"{h2h_stats['last_game_tov']:.0f}")
                            with col4:
                                st.metric("Plus/Minus", f"{h2h_stats['last_game_plus_minus']:+.0f}")
                            
                            st.markdown("---")
                        else:
                            st.info(f"No previous matchups found against {opponent} this season.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("No upcoming games found in the schedule. The NBA season may be in the offseason.")
    
    with tab3:
        st.subheader(f"Game-by-Game Performance - {st.session_state.selected_season}")
        
        season_data = df_clean[df_clean['SEASON'] == st.session_state.selected_season]
        
        fig_games = go.Figure()
        
        fig_games.add_trace(go.Scatter(
            x=season_data['GAME_DATE'],
            y=season_data['PTS'],
            mode='lines+markers',
            name='Points',
            line=dict(width=2, color='#FF4B4B'),
            marker=dict(size=8),
            hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                        "<b>Points</b>: %{y}<br>" +
                        "<b>Opponent</b>: %{customdata[1]}<br>" +
                        "<b>Team</b>: %{customdata[2]}<extra></extra>",
            customdata=np.column_stack((
                season_data['GAME_DATE'].dt.strftime('%b %d'),
                season_data['MATCHUP'],
                season_data['TEAM_ABBREVIATION']
            ))
        ))
        
        fig_games.add_trace(go.Scatter(
            x=season_data['GAME_DATE'],
            y=season_data['REB'],
            mode='lines+markers',
            name='Rebounds',
            line=dict(width=2, color='#00CC96'),
            marker=dict(size=8),
            hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                        "<b>Rebounds</b>: %{y}<br>" +
                        "<b>Opponent</b>: %{customdata[1]}<br>" +
                        "<b>Team</b>: %{customdata[2]}<extra></extra>",
            customdata=np.column_stack((
                season_data['GAME_DATE'].dt.strftime('%b %d'),
                season_data['MATCHUP'],
                season_data['TEAM_ABBREVIATION']
            ))
        ))
        
        fig_games.add_trace(go.Scatter(
            x=season_data['GAME_DATE'],
            y=season_data['AST'],
            mode='lines+markers',
            name='Assists',
            line=dict(width=2, color='#636EFA'),
            marker=dict(size=8),
            hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                        "<b>Assists</b>: %{y}<br>" +
                        "<b>Opponent</b>: %{customdata[1]}<br>" +
                        "<b>Team</b>: %{customdata[2]}<extra></extra>",
            customdata=np.column_stack((
                season_data['GAME_DATE'].dt.strftime('%b %d'),
                season_data['MATCHUP'],
                season_data['TEAM_ABBREVIATION']
            ))
        ))
        
        window = 5
        season_data['PTS_AVG'] = season_data['PTS'].rolling(window=window, min_periods=1).mean()
        season_data['REB_AVG'] = season_data['REB'].rolling(window=window, min_periods=1).mean()
        season_data['AST_AVG'] = season_data['AST'].rolling(window=window, min_periods=1).mean()
        
        fig_games.add_trace(go.Scatter(
            x=season_data['GAME_DATE'],
            y=season_data['PTS_AVG'],
            mode='lines',
            name=f'{window}-Game PTS Avg',
            line=dict(width=2, color='#FF4B4B', dash='dash'),
            hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                        f"<b>{window}-Game PTS Avg</b>: %{{y:.1f}}<extra></extra>",
            customdata=season_data['GAME_DATE'].dt.strftime('%b %d')
        ))
        
        fig_games.add_trace(go.Scatter(
            x=season_data['GAME_DATE'],
            y=season_data['REB_AVG'],
            mode='lines',
            name=f'{window}-Game REB Avg',
            line=dict(width=2, color='#00CC96', dash='dash'),
            hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                        f"<b>{window}-Game REB Avg</b>: %{{y:.1f}}<extra></extra>",
            customdata=season_data['GAME_DATE'].dt.strftime('%b %d')
        ))
        
        fig_games.add_trace(go.Scatter(
            x=season_data['GAME_DATE'],
            y=season_data['AST_AVG'],
            mode='lines',
            name=f'{window}-Game AST Avg',
            line=dict(width=2, color='#636EFA', dash='dash'),
            hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                        f"<b>{window}-Game AST Avg</b>: %{{y:.1f}}<extra></extra>",
            customdata=season_data['GAME_DATE'].dt.strftime('%b %d')
        ))
        
        fig_games.update_layout(
            title=f"{player_name}'s Game Log - {st.session_state.selected_season}",
            xaxis_title="Date",
            yaxis_title="Stats",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600
        )
        
        st.plotly_chart(fig_games, use_container_width=True, key="game_log_chart")
        
        st.subheader("Shooting Efficiency Breakdown")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'FGM' in season_data.columns and 'FGA' in season_data.columns:
                fg_pct = (season_data['FGM'].sum() / season_data['FGA'].sum() * 100)
                st.metric("Field Goal %", f"{fg_pct:.1f}%")
            
            if 'FG3M' in season_data.columns and 'FG3A' in season_data.columns:
                three_pct = (season_data['FG3M'].sum() / season_data['FG3A'].sum() * 100)
                st.metric("3-Point %", f"{three_pct:.1f}%")
            
            if 'FTM' in season_data.columns and 'FTA' in season_data.columns:
                ft_pct = (season_data['FTM'].sum() / season_data['FTA'].sum() * 100)
                st.metric("Free Throw %", f"{ft_pct:.1f}%")
        
        with col2:
            if 'FGM' in season_data.columns and 'FGA' in season_data.columns:
                st.metric("Field Goals Made/Attempted", f"{season_data['FGM'].sum()}/{season_data['FGA'].sum()}")
            
            if 'FG3M' in season_data.columns and 'FG3A' in season_data.columns:
                st.metric("3-Pointers Made/Attempted", f"{season_data['FG3M'].sum()}/{season_data['FG3A'].sum()}")
            
            if 'FTM' in season_data.columns and 'FTA' in season_data.columns:
                st.metric("Free Throws Made/Attempted", f"{season_data['FTM'].sum()}/{season_data['FTA'].sum()}")
        
        st.subheader("Win/Loss Record by Points")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            point_threshold = st.number_input(
                "Select Point Threshold",
                min_value=0,
                max_value=100,
                value=20,
                step=1,
                help="See win/loss record when player scores above/below this many points"
            )
        
        with col2:
            above_threshold = season_data[season_data['PTS'] >= point_threshold]
            below_threshold = season_data[season_data['PTS'] < point_threshold]
            
            above_wins = len(above_threshold[above_threshold['PLUS_MINUS'] > 0])
            above_losses = len(above_threshold[above_threshold['PLUS_MINUS'] <= 0])
            below_wins = len(below_threshold[below_threshold['PLUS_MINUS'] > 0])
            below_losses = len(below_threshold[below_threshold['PLUS_MINUS'] <= 0])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**When scoring {point_threshold}+ points:**")
                st.metric("Record", f"{above_wins}-{above_losses}", 
                         f"{above_wins/(above_wins+above_losses)*100:.1f}% win rate" if (above_wins+above_losses) > 0 else "N/A")
                if above_wins + above_losses > 0:
                    st.metric("Average Plus/Minus", f"{above_threshold['PLUS_MINUS'].mean():.1f}")
            
            with col2:
                st.markdown(f"**When scoring <{point_threshold} points:**")
                st.metric("Record", f"{below_wins}-{below_losses}", 
                         f"{below_wins/(below_wins+below_losses)*100:.1f}% win rate" if (below_wins+below_losses) > 0 else "N/A")
                if below_wins + below_losses > 0:
                    st.metric("Average Plus/Minus", f"{below_threshold['PLUS_MINUS'].mean():.1f}")
            
            if 'PLUS_MINUS' in season_data.columns:
                fig_wl = go.Figure()
                
                fig_wl.add_trace(go.Bar(
                    y=['Above Threshold', 'Below Threshold'],
                    x=[above_wins, below_wins],
                    name='Wins',
                    marker_color='#00CC96',
                    orientation='h',
                    width=0.4
                ))
                
                fig_wl.add_trace(go.Bar(
                    y=['Above Threshold', 'Below Threshold'],
                    x=[above_losses, below_losses],
                    name='Losses',
                    marker_color='#EF553B',
                    orientation='h',
                    width=0.4
                ))
                
                fig_wl.update_layout(
                    title=f"Win/Loss Record When Scoring {point_threshold}+ Points",
                    barmode='group',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=300,
                    xaxis_title="Number of Games",
                    yaxis_title="",
                    bargap=0.2,
                    bargroupgap=0.1
                )
                
                st.plotly_chart(fig_wl, use_container_width=True)
        
        if 'FGM' in season_data.columns and 'FGA' in season_data.columns:
            fig_shooting = go.Figure()
            
            fig_shooting.add_trace(go.Scatter(
                x=season_data['GAME_DATE'],
                y=season_data['FGM'] / season_data['FGA'] * 100,
                mode='lines+markers',
                name='FG%',
                line=dict(width=2, color='#FF4B4B'),
                marker=dict(size=8),
                hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                            "<b>FG%</b>: %{y:.1f}%<br>" +
                            "<b>FGM/FGA</b>: %{customdata[1]}/%{customdata[2]}<extra></extra>",
                customdata=np.column_stack((
                    season_data['GAME_DATE'].dt.strftime('%b %d'),
                    season_data['FGM'],
                    season_data['FGA']
                ))
            ))
            
            if 'FG3M' in season_data.columns and 'FG3A' in season_data.columns:
                fig_shooting.add_trace(go.Scatter(
                    x=season_data['GAME_DATE'],
                    y=season_data['FG3M'] / season_data['FG3A'] * 100,
                    mode='lines+markers',
                    name='3P%',
                    line=dict(width=2, color='#00CC96'),
                    marker=dict(size=8),
                    hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                                "<b>3P%</b>: %{y:.1f}%<br>" +
                                "<b>3PM/3PA</b>: %{customdata[1]}/%{customdata[2]}<extra></extra>",
                    customdata=np.column_stack((
                        season_data['GAME_DATE'].dt.strftime('%b %d'),
                        season_data['FG3M'],
                        season_data['FG3A']
                    ))
                ))
            
            fig_shooting.update_layout(
                title="Shooting Percentage Trends",
                xaxis_title="Date",
                yaxis_title="Percentage",
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=400
            )
            
            st.plotly_chart(fig_shooting, use_container_width=True, key="shooting_trends_chart_1")
    
    with tab4:
        st.subheader(f"Shot Distance Analysis - {st.session_state.selected_season}")
        
        shot_data = get_shot_data(player_name, season=st.session_state.selected_season)
        if shot_data is not None and not shot_data.empty:
            bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
            labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40']
            
            shot_data['Distance_Range'] = pd.cut(shot_data['SHOT_DISTANCE'], bins=bins, labels=labels)
            
            distance_stats = shot_data.groupby('Distance_Range').agg({
                'SHOT_MADE_FLAG': ['count', 'mean'],
                'SHOT_DISTANCE': 'mean'
            }).reset_index()
            
            distance_stats.columns = ['Distance_Range', 'Attempts', 'FG%', 'Avg_Distance']
            distance_stats['FG%'] = distance_stats['FG%'] * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=distance_stats['Distance_Range'],
                y=distance_stats['Attempts'],
                name='Shot Attempts',
                marker_color='#636EFA',
                opacity=0.7
            ))
            
            fig.add_trace(go.Scatter(
                x=distance_stats['Distance_Range'],
                y=distance_stats['FG%'],
                name='FG%',
                yaxis='y2',
                line=dict(color='#EF553B', width=3),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title=f"{player_name}'s Shot Distribution by Distance - {st.session_state.selected_season}",
                xaxis_title="Distance from Basket (feet)",
                yaxis_title="Number of Attempts",
                yaxis2=dict(
                    title="Field Goal %",
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                avg_distance = shot_data['SHOT_DISTANCE'].mean()
                st.metric("Average Shot Distance", f"{avg_distance:.1f} ft")
            
            with col2:
                valid_ranges = distance_stats[distance_stats['Attempts'] >= 10]
                if not valid_ranges.empty:
                    best_range = valid_ranges.loc[valid_ranges['FG%'].idxmax(), 'Distance_Range']
                    best_fg_pct = valid_ranges.loc[valid_ranges['FG%'].idxmax(), 'FG%']
                    st.metric(
                        "Most Efficient Range",
                        f"{best_range} ft",
                        f"{best_fg_pct:.1f}%"
                    )
                else:
                    st.metric("Most Efficient Range", "N/A", "Not enough attempts")
            
            with col3:
                if not distance_stats.empty:
                    most_common = distance_stats.loc[distance_stats['Attempts'].idxmax(), 'Distance_Range']
                    most_attempts = distance_stats['Attempts'].max()
                    most_fg_pct = distance_stats.loc[distance_stats['Attempts'].idxmax(), 'FG%']
                    st.metric(
                        "Favorite Shooting Range",
                        f"{most_common} ft",
                        f"{most_attempts} shots"
                    )
                else:
                    st.metric("Favorite Shooting Range", "N/A", "No data")
            
            with col4:
                total_shots = len(shot_data)
                made_shots = shot_data['SHOT_MADE_FLAG'].sum()
                fg_pct = (made_shots / total_shots * 100)
                st.metric(
                    "Overall Shooting",
                    f"{fg_pct:.1f}%",
                    f"{made_shots}/{total_shots}"
                )
            
            with col5:
                made_shots_data = shot_data[shot_data['SHOT_MADE_FLAG'] == 1]
                if not made_shots_data.empty:
                    farthest_made = made_shots_data.loc[made_shots_data['SHOT_DISTANCE'].idxmax()]
                    st.metric(
                        "Farthest Shot Made",
                        f"{farthest_made['SHOT_DISTANCE']:.1f} ft",
                        f"{farthest_made['SHOT_TYPE']}"
                    )
                else:
                    st.metric("Farthest Shot Made", "N/A", "No made shots")
            
            st.subheader(f"Shot Type Analysis - {st.session_state.selected_season}")
            
            shot_data['Shot_Category'] = shot_data['SHOT_TYPE'].apply(lambda x: '3PT' if '3PT' in x else '2PT')
            shot_category_stats = shot_data.groupby('Shot_Category').agg({
                'SHOT_MADE_FLAG': ['count', 'mean'],
                'SHOT_DISTANCE': 'mean'
            }).reset_index()
            
            shot_category_stats.columns = ['Shot_Category', 'Attempts', 'FG%', 'Avg_Distance']
            shot_category_stats['FG%'] = shot_category_stats['FG%'] * 100
            
            shot_category_stats['FG%'] = shot_category_stats['FG%'].fillna(0)
            shot_category_stats['Avg_Distance'] = shot_category_stats['Avg_Distance'].fillna(0)

            fig_types = go.Figure()
            
            fig_types.add_trace(go.Pie(
                labels=shot_category_stats['Shot_Category'],
                values=shot_category_stats['Attempts'],
                hole=0.4,
                marker=dict(
                    colors=['#9B6B9E', '#E6A4B4'],
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textposition='inside',
                hovertemplate="<b>%{label}</b><br>" +
                            "Attempts: %{value}<extra></extra>"
            ))
            
            fig_types.update_layout(
                title=f"Shot Type Distribution - {st.session_state.selected_season}",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(color='white')
                ),
                height=400,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hoverlabel=dict(
                    bgcolor='rgba(0,0,0,0.8)',
                    font_size=12,
                    font_family='Arial'
                )
            )
            
            st.plotly_chart(fig_types, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style='background-color: rgba(155, 107, 158, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(155, 107, 158, 0.2);'>
                    <h4 style='color: #9B6B9E; margin: 0; font-size: 16px;'>2-Point Shooting</h4>
                    <p style='color: white; font-size: 20px; margin: 8px 0;'>
                        {:.1f}% FG
                    </p>
                    <p style='color: rgba(255, 255, 255, 0.7); margin: 0; font-size: 14px;'>
                        {} attempts
                    </p>
                    <p style='color: rgba(255, 255, 255, 0.7); margin: 4px 0 0 0; font-size: 14px;'>
                        Avg Distance: {:.1f} ft
                    </p>
                </div>
                """.format(
                    shot_category_stats[shot_category_stats['Shot_Category'] == '2PT']['FG%'].iloc[0],
                    shot_category_stats[shot_category_stats['Shot_Category'] == '2PT']['Attempts'].iloc[0],
                    shot_category_stats[shot_category_stats['Shot_Category'] == '2PT']['Avg_Distance'].iloc[0]
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background-color: rgba(230, 164, 180, 0.05); padding: 15px; border-radius: 8px; border: 1px solid rgba(230, 164, 180, 0.2);'>
                    <h4 style='color: #E6A4B4; margin: 0; font-size: 16px;'>3-Point Shooting</h4>
                    <p style='color: white; font-size: 20px; margin: 8px 0;'>
                        {:.1f}% FG
                    </p>
                    <p style='color: rgba(255, 255, 255, 0.7); margin: 0; font-size: 14px;'>
                        {} attempts
                    </p>
                    <p style='color: rgba(255, 255, 255, 0.7); margin: 4px 0 0 0; font-size: 14px;'>
                        Avg Distance: {:.1f} ft
                    </p>
                </div>
                """.format(
                    shot_category_stats[shot_category_stats['Shot_Category'] == '3PT']['FG%'].iloc[0],
                    shot_category_stats[shot_category_stats['Shot_Category'] == '3PT']['Attempts'].iloc[0],
                    shot_category_stats[shot_category_stats['Shot_Category'] == '3PT']['Avg_Distance'].iloc[0]
                ), unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background-color: rgba(240, 242, 246, 0.05); padding: 15px; border-radius: 8px; margin-top: 20px; color: #262730;">
                <details>
                    <summary style="color: white; font-weight: bold; cursor: pointer;">Shot Analysis Guide</summary>
                    <p style="color: rgba(255, 255, 255, 0.7); margin-top: 10px;">This analysis shows:</p>
                    <ul style="color: rgba(255, 255, 255, 0.7);">
                        <li><b>Shot Distribution</b>: Number of attempts and shooting percentage at different distances</li>
                        <li><b>Best Range</b>: Distance range with highest shooting percentage (minimum 10 attempts)</li>
                        <li><b>Most Common Range</b>: Distance where the player shoots most frequently</li>
                        <li><b>Shot Types</b>: Breakdown of different shot types and their success rates</li>
                    </ul>
                </details>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning(f"Shot data not available for {st.session_state.selected_season}.")
    
    with tab5:
        st.subheader("Career Trajectory")
        
        # Try to get career trajectory data from database first
        career_stats = get_career_trajectory_from_db(player_name)
        
        # If not in database, calculate and save it
        if career_stats is None:
            # Calculate career averages by season
            career_stats = df_clean.groupby('SEASON').agg({
                'PTS': 'mean',
                'REB': 'mean',
                'AST': 'mean',
                'MIN': 'mean',
                'FGM': 'mean',
                'FGA': 'mean',
                'FG3M': 'mean',
                'FG3A': 'mean',
                'FTM': 'mean',
                'FTA': 'mean',
                'TOV': 'mean',
                'STL': 'mean',
                'BLK': 'mean',
                'PLUS_MINUS': 'mean',
                'GAME_DATE': 'count'
            }).reset_index()
            
            # Rename GAME_DATE to GAMES_PLAYED
            career_stats = career_stats.rename(columns={'GAME_DATE': 'GAMES_PLAYED'})
            
            # Save to database
            save_career_trajectory_to_db(career_stats, player_name)
        
        # Sort seasons chronologically
        career_stats['SEASON'] = career_stats['SEASON'].astype(str)
        career_stats = career_stats.sort_values('SEASON')
        
        # Calculate shooting percentages if not already in database
        if 'FG_PCT' not in career_stats.columns:
            career_stats['FG_PCT'] = (career_stats['FGM'] / career_stats['FGA'] * 100)
            career_stats['THREE_PCT'] = (career_stats['FG3M'] / career_stats['FG3A'] * 100)
            career_stats['FT_PCT'] = (career_stats['FTM'] / career_stats['FTA'] * 100)
        
        # Create main stats trajectory
        fig_main = go.Figure()
        
        fig_main.add_trace(go.Scatter(
            x=career_stats['SEASON'],
            y=career_stats['PTS'],
            mode='lines+markers',
            name='Points',
            line=dict(width=2, color='#FF4B4B'),
            marker=dict(size=8),
            hovertemplate="<b>Season</b>: %{x}<br>" +
                        "<b>Points</b>: %{y:.1f}<br>" +
                        "<b>Games Played</b>: %{customdata}<extra></extra>",
            customdata=career_stats['GAMES_PLAYED']
        ))
        
        fig_main.add_trace(go.Scatter(
            x=career_stats['SEASON'],
            y=career_stats['REB'],
            mode='lines+markers',
            name='Rebounds',
            line=dict(width=2, color='#00CC96'),
            marker=dict(size=8),
            hovertemplate="<b>Season</b>: %{x}<br>" +
                        "<b>Rebounds</b>: %{y:.1f}<br>" +
                        "<b>Games Played</b>: %{customdata}<extra></extra>",
            customdata=career_stats['GAMES_PLAYED']
        ))
        
        fig_main.add_trace(go.Scatter(
            x=career_stats['SEASON'],
            y=career_stats['AST'],
            mode='lines+markers',
            name='Assists',
            line=dict(width=2, color='#636EFA'),
            marker=dict(size=8),
            hovertemplate="<b>Season</b>: %{x}<br>" +
                        "<b>Assists</b>: %{y:.1f}<br>" +
                        "<b>Games Played</b>: %{customdata}<extra></extra>",
            customdata=career_stats['GAMES_PLAYED']
        ))
        
        fig_main.update_layout(
            title=f"{player_name}'s Career Trajectory - Main Stats",
            xaxis_title="Season",
            yaxis_title="Average",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Create shooting percentages trajectory
        fig_shooting = go.Figure()
        
        fig_shooting.add_trace(go.Scatter(
            x=career_stats['SEASON'],
            y=career_stats['FG_PCT'],
            mode='lines+markers',
            name='FG%',
            line=dict(width=2, color='#FF4B4B'),
            marker=dict(size=8),
            hovertemplate="<b>Season</b>: %{x}<br>" +
                        "<b>FG%</b>: %{y:.1f}%<br>" +
                        "<b>Games Played</b>: %{customdata}<extra></extra>",
            customdata=career_stats['GAMES_PLAYED']
        ))
        
        fig_shooting.add_trace(go.Scatter(
            x=career_stats['SEASON'],
            y=career_stats['THREE_PCT'],
            mode='lines+markers',
            name='3P%',
            line=dict(width=2, color='#00CC96'),
            marker=dict(size=8),
            hovertemplate="<b>Season</b>: %{x}<br>" +
                        "<b>3P%</b>: %{y:.1f}%<br>" +
                        "<b>Games Played</b>: %{customdata}<extra></extra>",
            customdata=career_stats['GAMES_PLAYED']
        ))
        
        fig_shooting.add_trace(go.Scatter(
            x=career_stats['SEASON'],
            y=career_stats['FT_PCT'],
            mode='lines+markers',
            name='FT%',
            line=dict(width=2, color='#636EFA'),
            marker=dict(size=8),
            hovertemplate="<b>Season</b>: %{x}<br>" +
                        "<b>FT%</b>: %{y:.1f}%<br>" +
                        "<b>Games Played</b>: %{customdata}<extra></extra>",
            customdata=career_stats['GAMES_PLAYED']
        ))
        
        fig_shooting.update_layout(
            title=f"{player_name}'s Career Trajectory - Shooting Percentages",
            xaxis_title="Season",
            yaxis_title="Percentage",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        st.plotly_chart(fig_shooting, use_container_width=True)
        
        # Create advanced stats trajectory
        fig_advanced = go.Figure()
        
        fig_advanced.add_trace(go.Scatter(
            x=career_stats['SEASON'],
            y=career_stats['TOV'],
            mode='lines+markers',
            name='Turnovers',
            line=dict(width=2, color='#FF4B4B'),
            marker=dict(size=8),
            hovertemplate="<b>Season</b>: %{x}<br>" +
                        "<b>Turnovers</b>: %{y:.1f}<br>" +
                        "<b>Games Played</b>: %{customdata}<extra></extra>",
            customdata=career_stats['GAMES_PLAYED']
        ))
        
        fig_advanced.add_trace(go.Scatter(
            x=career_stats['SEASON'],
            y=career_stats['STL'],
            mode='lines+markers',
            name='Steals',
            line=dict(width=2, color='#00CC96'),
            marker=dict(size=8),
            hovertemplate="<b>Season</b>: %{x}<br>" +
                        "<b>Steals</b>: %{y:.1f}<br>" +
                        "<b>Games Played</b>: %{customdata}<extra></extra>",
            customdata=career_stats['GAMES_PLAYED']
        ))
        
        fig_advanced.add_trace(go.Scatter(
            x=career_stats['SEASON'],
            y=career_stats['BLK'],
            mode='lines+markers',
            name='Blocks',
            line=dict(width=2, color='#636EFA'),
            marker=dict(size=8),
            hovertemplate="<b>Season</b>: %{x}<br>" +
                        "<b>Blocks</b>: %{y:.1f}<br>" +
                        "<b>Games Played</b>: %{customdata}<extra></extra>",
            customdata=career_stats['GAMES_PLAYED']
        ))
        
        fig_advanced.update_layout(
            title=f"{player_name}'s Career Trajectory - Advanced Stats",
            xaxis_title="Season",
            yaxis_title="Average",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        st.plotly_chart(fig_advanced, use_container_width=True)
        
        # Add career progression metrics
        st.subheader("Career Progression Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Find the index of the selected season
            selected_season_idx = career_stats[career_stats['SEASON'] == st.session_state.selected_season].index[0]
            if selected_season_idx > 0:  # If there is a previous season
                selected_season_data = career_stats.iloc[selected_season_idx]
                prev_season_data = career_stats.iloc[selected_season_idx - 1]
                
                pts_improvement = selected_season_data['PTS'] - prev_season_data['PTS']
                delta_color = "normal" if pts_improvement > 0 else "inverse" if pts_improvement < 0 else "off"
                st.metric(
                    "Points Improvement",
                    f"{pts_improvement:+.1f}",
                    f"vs {prev_season_data['SEASON']}",
                    delta_color=delta_color
                )
            else:
                st.metric("Points Improvement", "N/A", "No previous season")
        
        with col2:
            if selected_season_idx > 0:
                fg_improvement = selected_season_data['FG_PCT'] - prev_season_data['FG_PCT']
                delta_color = "normal" if fg_improvement > 0 else "inverse" if fg_improvement < 0 else "off"
                st.metric(
                    "FG% Improvement",
                    f"{fg_improvement:+.1f}%",
                    f"vs {prev_season_data['SEASON']}",
                    delta_color=delta_color
                )
            else:
                st.metric("FG% Improvement", "N/A", "No previous season")
        
        with col3:
            if selected_season_idx > 0:
                plus_minus_improvement = selected_season_data['PLUS_MINUS'] - prev_season_data['PLUS_MINUS']
                delta_color = "normal" if plus_minus_improvement > 0 else "inverse" if plus_minus_improvement < 0 else "off"
                st.metric(
                    "Plus/Minus Improvement",
                    f"{plus_minus_improvement:+.1f}",
                    f"vs {prev_season_data['SEASON']}",
                    delta_color=delta_color
                )
            else:
                st.metric("Plus/Minus Improvement", "N/A", "No previous season")
        
        # Add peak season analysis
        st.subheader("Peak Season Analysis")
        
        peak_season = career_stats.loc[career_stats['PTS'].idxmax()]
        st.markdown(f"**Peak Scoring Season**: {peak_season['SEASON']} ({peak_season['PTS']:.1f} PPG)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Points", f"{peak_season['PTS']:.1f}")
            st.metric("Rebounds", f"{peak_season['REB']:.1f}")
            st.metric("Assists", f"{peak_season['AST']:.1f}")
        
        with col2:
            st.metric("FG%", f"{peak_season['FG_PCT']:.1f}%")
            st.metric("3P%", f"{peak_season['THREE_PCT']:.1f}%")
            st.metric("FT%", f"{peak_season['FT_PCT']:.1f}%")
        
        with col3:
            st.metric("Steals", f"{peak_season['STL']:.1f}")
            st.metric("Blocks", f"{peak_season['BLK']:.1f}")
            st.metric("Plus/Minus", f"{peak_season['PLUS_MINUS']:+.1f}")
        
        # Add career trajectory analysis
        st.subheader("Career Trajectory Analysis")
        
        # Calculate career averages (excluding SEASON column)
        numeric_columns = career_stats.select_dtypes(include=[np.number]).columns
        career_avg = career_stats[numeric_columns].mean()
        
        st.markdown("""
        <style>
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: linear-gradient(145deg, rgba(255, 75, 75, 0.1), rgba(255, 75, 75, 0.05));
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255, 75, 75, 0.2);
            transition: transform 0.2s;
        }
        .stat-card:hover {
            transform: translateY(-2px);
        }
        .stat-label {
            color: rgba(255, 255, 255, 0.7);
            font-size: 14px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-value {
            color: white;
            font-size: 24px;
            font-weight: 600;
            margin: 0;
        }
        .shooting-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 30px;
        }
        .shooting-card {
            background: linear-gradient(145deg, rgba(0, 204, 150, 0.1), rgba(0, 204, 150, 0.05));
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(0, 204, 150, 0.2);
            transition: transform 0.2s;
        }
        .shooting-card:hover {
            transform: translateY(-2px);
        }
        .shooting-label {
            color: rgba(255, 255, 255, 0.7);
            font-size: 14px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .shooting-value {
            color: white;
            font-size: 24px;
            font-weight: 600;
            margin: 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Points</div>
                <div class="stat-value">{:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Rebounds</div>
                <div class="stat-value">{:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Assists</div>
                <div class="stat-value">{:.1f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Minutes</div>
                <div class="stat-value">{:.1f}</div>
            </div>
        </div>
        """.format(
            career_avg['PTS'],
            career_avg['REB'],
            career_avg['AST'],
            career_avg['MIN']
        ), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="shooting-grid">
            <div class="shooting-card">
                <div class="shooting-label">Field Goal %</div>
                <div class="shooting-value">{:.1f}%</div>
            </div>
            <div class="shooting-card">
                <div class="shooting-label">3-Point %</div>
                <div class="shooting-value">{:.1f}%</div>
            </div>
            <div class="shooting-card">
                <div class="shooting-label">Free Throw %</div>
                <div class="shooting-value">{:.1f}%</div>
            </div>
        </div>
        """.format(
            career_avg['FG_PCT'],
            career_avg['THREE_PCT'],
            career_avg['FT_PCT']
        ), unsafe_allow_html=True)

    with tab6:
        st.subheader(f"Player Comparison - {st.session_state.selected_season}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Main Player")
            st.markdown(f"**{player_name}**")
            st.markdown("(Selected in sidebar)")
        
        with col2:
            st.subheader("Compare With")
            compare_players = st.multiselect(
                "Select Players to Compare",
                options=[p['original_name'] for p in player_list if p['original_name'] != player_name],
                default=[],
                key="compare_players_select"
            )
        
        if compare_players:
            all_players = [player_name] + compare_players
            
            player_data = {}
            for player in all_players:
                player_df = load_player_data(player)
                if player_df is not None:
                    player_data[player] = clean_features(player_df, player)
            
            if len(player_data) > 1:
                season_data_dict = {
                    player: data[data['SEASON'] == st.session_state.selected_season]
                    for player, data in player_data.items()
                }
                
                st.subheader(f"Basic Stats Comparison - {st.session_state.selected_season}")
                fig_basic = go.Figure()
                
                for i, player in enumerate(all_players):
                    if player in season_data_dict:
                        data = season_data_dict[player]
                        fig_basic.add_trace(go.Bar(
                            x=['Points', 'Rebounds', 'Assists'],
                            y=[data['PTS'].mean(), data['REB'].mean(), data['AST'].mean()],
                            name=player,
                            marker_color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
                        ))
                
                fig_basic.update_layout(
                    title=f"Season Average Comparison - {st.session_state.selected_season}",
                    xaxis_title="Stat",
                    yaxis_title="Average",
                    barmode='group',
                    bargap=0.15,
                    bargroupgap=0.1,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_basic, use_container_width=True, key="basic_stats_comparison")
                
                st.subheader(f"Advanced Stats Comparison - {st.session_state.selected_season}")
                
                stats = []
                
                for player in all_players:
                    if player in season_data_dict:
                        data = season_data_dict[player]
                        
                        player_stats = {
                            'Player': player,
                            'PTS': round(data['PTS'].mean(), 1),
                            'AST': round(data['AST'].mean(), 1),
                            'REB': round(data['REB'].mean(), 1),
                            'TOV': round(data['TOV'].mean(), 1),
                            'STL': round(data['STL'].mean(), 1),
                            'BLK': round(data['BLK'].mean(), 1)
                        }
                        
                        if 'FGM' in data.columns and 'FGA' in data.columns:
                            fg_pct = (data['FGM'].sum() / data['FGA'].sum()) * 100
                            player_stats['FG%'] = f"{round(fg_pct, 1)}%"
                        
                        if 'FG3M' in data.columns and 'FG3A' in data.columns:
                            three_pct = (data['FG3M'].sum() / data['FG3A'].sum()) * 100
                            player_stats['3P%'] = f"{round(three_pct, 1)}%"
                        
                        if 'FTM' in data.columns and 'FTA' in data.columns:
                            ft_pct = (data['FTM'].sum() / data['FTA'].sum()) * 100
                            player_stats['FT%'] = f"{round(ft_pct, 1)}%"
                        
                        stats.append(player_stats)
                
                if stats:
                    df = pd.DataFrame(stats)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.subheader(f"Game Score Comparison - {st.session_state.selected_season}")
                
                game_scores = []
                
                for player in all_players:
                    if player in season_data_dict:
                        data = season_data_dict[player]
                        
                        game_score = data['PTS']
                        if 'FGM' in data.columns:
                            game_score += 0.4 * data['FGM']
                        if 'FGA' in data.columns:
                            game_score -= 0.7 * data['FGA']
                        if 'FTM' in data.columns:
                            game_score += 0.4 * data['FTM']
                        if 'FTA' in data.columns:
                            game_score -= 0.7 * data['FTA']
                        if 'OREB' in data.columns:
                            game_score += 0.7 * data['OREB']
                        if 'DREB' in data.columns:
                            game_score += 0.3 * data['DREB']
                        if 'STL' in data.columns:
                            game_score += data['STL']
                        if 'AST' in data.columns:
                            game_score += 0.7 * data['AST']
                        if 'BLK' in data.columns:
                            game_score += 0.7 * data['BLK']
                        if 'TOV' in data.columns:
                            game_score -= data['TOV']
                        if 'PF' in data.columns:
                            game_score -= 0.4 * data['PF']
                        
                        game_scores.append({
                            'Player': player,
                            'Average Game Score': round(game_score.mean(), 1),
                            'Best Game Score': round(game_score.max(), 1)
                        })
                    
                if game_scores:
                    game_score_df = pd.DataFrame(game_scores)
                    st.dataframe(game_score_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No valid player data available for comparison.")
        else:
            st.info("Select players to compare from the dropdown above.")