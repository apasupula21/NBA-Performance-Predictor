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

st.set_page_config(layout="wide")

@st.cache_data
def get_player_list():
    player_list = players.get_active_players()
    return sorted([player['full_name'] for player in player_list])

@st.cache_data
def get_player_rookie_and_last_season(player_name):
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

@st.cache_data
def load_player_data(player_name, start_season=None, end_season=None):
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
    return df

@st.cache_data
def get_shot_data(player_name, season='2024-25', max_retries=3):
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
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)
            else:
                st.error(f"Failed to fetch shot data after {max_retries} attempts: {e}")
                return None

st.title("NBA Player Performance Predictor")
st.markdown("### Visualize player stats and get performance predictions")

player_list = ["Select a player..."] + get_player_list()

st.sidebar.header("Player Selection")
player_name = st.sidebar.selectbox("Select Player", player_list, index=0, key="player_select")

df = None
if player_name and player_name != "Select a player...":
    df = load_player_data(player_name)

if df is not None:
    df_clean = clean_features(df, player_name)
    
    available_seasons = sorted(df_clean['SEASON'].unique(), reverse=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Next Game & Matchup", "Game-by-Game Performance", "Shot Distance Analysis", "Player Comparison"])
    
    # Store the active tab in session state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Overview"
    
    # Update active tab based on user selection
    if tab1:
        st.session_state.active_tab = "Overview"
    elif tab2:
        st.session_state.active_tab = "Next Game & Matchup"
    elif tab3:
        st.session_state.active_tab = "Game-by-Game Performance"
    elif tab4:
        st.session_state.active_tab = "Shot Distance Analysis"
    elif tab5:
        st.session_state.active_tab = "Player Comparison"
    
    # Show content based on active tab
    if st.session_state.active_tab == "Overview":
        with tab1:
            season_data = df_clean[df_clean['SEASON'] == available_seasons[0]]
            
            st.subheader("Season Averages")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Points", f"{season_data['PTS'].mean():.1f}")
            with col2:
                st.metric("Rebounds", f"{season_data['REB'].mean():.1f}")
            with col3:
                st.metric("Assists", f"{season_data['AST'].mean():.1f}")
            with col4:
                st.metric("Minutes", f"{season_data['MIN'].mean():.1f}")
            
            st.subheader("Shooting Efficiency")
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
            
            st.subheader("Advanced Stats")
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
            
            st.subheader("Advanced Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate PER (corrected formula)
            with col1:
                if all(col in season_data.columns for col in ['PTS', 'FGM', 'FGA', 'FTM', 'FTA', 'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'MIN']):
                    # PER calculation (simplified version)
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
                    st.metric("PER", f"{per:.1f}")
            
            # Calculate Usage Rate
            with col2:
                if all(col in season_data.columns for col in ['FGA', 'FTA', 'TOV', 'MIN']):
                    possessions = season_data['FGA'].sum() + 0.44 * season_data['FTA'].sum() + season_data['TOV'].sum()
                    minutes = season_data['MIN'].sum()
                    if minutes > 0:
                        possessions_per_game = (minutes / 48) * 100
                        usg_rate = (possessions / possessions_per_game) * 100
                        st.metric("Usage Rate", f"{usg_rate:.1f}%")
            
            # Calculate True Shooting %
            with col3:
                if all(col in season_data.columns for col in ['PTS', 'FGA', 'FTA']):
                    fga_fta = season_data['FGA'].sum() + 0.44 * season_data['FTA'].sum()
                    if fga_fta > 0:
                        ts_pct = season_data['PTS'].sum() / (2 * fga_fta) * 100
                        st.metric("True Shooting %", f"{ts_pct:.1f}%")
            
            # Calculate Box Plus/Minus
            with col4:
                if all(col in season_data.columns for col in ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'PF']):
                    # Simplified BPM calculation
                    bpm = (season_data['PTS'].mean() * 0.274 + 
                          season_data['AST'].mean() * 0.7 + 
                          season_data['REB'].mean() * 0.5 + 
                          season_data['STL'].mean() * 0.7 + 
                          season_data['BLK'].mean() * 0.7 - 
                          season_data['TOV'].mean() - 
                          season_data['PF'].mean() * 0.4)
                    st.metric("Box Plus/Minus", f"{bpm:+.1f}")
            
            st.subheader("Impact Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if 'PLUS_MINUS' in season_data.columns:
                    plus_minus = season_data['PLUS_MINUS'].mean()
                    plus_minus_str = f"{plus_minus:+.1f}" if plus_minus != 0 else "0.0"
                    st.metric("Plus/Minus", plus_minus_str)
            with col2:
                if all(col in season_data.columns for col in ['AST', 'TOV']):
                    ast_to = season_data['AST'].sum() / season_data['TOV'].sum() if season_data['TOV'].sum() > 0 else 0
                    st.metric("AST/TO Ratio", f"{ast_to:.1f}")
            with col3:
                if all(col in season_data.columns for col in ['PTS', 'AST', 'REB']):
                    st.metric("PTS+AST+REB", f"{season_data['PTS'].mean() + season_data['AST'].mean() + season_data['REB'].mean():.1f}")
            with col4:
                if all(col in season_data.columns for col in ['FGM', 'FGA']):
                    efg_pct = ((season_data['FGM'].sum() + 0.5 * season_data['FG3M'].sum()) / season_data['FGA'].sum() * 100)
                    st.metric("Effective FG%", f"{efg_pct:.1f}%")
            
            st.subheader("Home vs Away Performance")
            
            # Split data into home and away games
            home_games = season_data[season_data['MATCHUP'].str.contains('vs')]
            away_games = season_data[season_data['MATCHUP'].str.contains('@')]
            
            # Points, Rebounds, Assists
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
            
            # Shooting Percentages
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
            
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: #262730;">
                <details>
                    <summary style="color: #262730; font-weight: bold;">Advanced Metrics Explanation</summary>
                    <ul style="color: #262730;">
                        <li><b>PER (Player Efficiency Rating)</b>: A measure of per-minute production standardized to 15.0 as league average</li>
                        <li><b>Usage Rate</b>: Estimates the percentage of team plays used by a player while on the floor</li>
                        <li><b>True Shooting %</b>: A measure of shooting efficiency that takes into account field goals, 3-point field goals, and free throws</li>
                        <li><b>Box Plus/Minus (BPM)</b>: A box score estimate of the points per 100 possessions a player contributed above a league-average player</li>
                        <li><b>Plus/Minus</b>: The point differential when the player is on the court</li>
                        <li><b>AST/TO Ratio</b>: Measures a player's ability to create assists while limiting turnovers</li>
                        <li><b>PTS+AST+REB</b>: Combined scoring, playmaking, and rebounding production</li>
                        <li><b>Effective FG%</b>: Adjusts field goal percentage to account for the fact that 3-pointers are worth more than 2-pointers</li>
                    </ul>
                </details>
            </div>
            """, unsafe_allow_html=True)
    
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
                    st.warning("Could not fetch the schedule.")
                elif not next_game:
                    st.info("No upcoming games found in the schedule. The NBA season may be in the offseason.")
                else:
                    opponent = next_game['opponent_team_name']
                    if not opponent:
                        st.warning("Could not determine opponent information.")
                    else:
                        game_date = next_game['game_date'].strftime('%B %d, %Y')
                        location = "Home" if next_game['is_home_game'] else "Away"
                        st.write(f"**Next Game**: {location} vs {opponent} on {game_date}")
                        
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
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if h2h_stats['last_game_pts'] is not None:
                                    st.metric("Points", f"{h2h_stats['last_game_pts']:.1f}")
                            with col2:
                                if h2h_stats['last_game_reb'] is not None:
                                    st.metric("Rebounds", f"{h2h_stats['last_game_reb']:.1f}")
                            with col3:
                                if h2h_stats['last_game_ast'] is not None:
                                    st.metric("Assists", f"{h2h_stats['last_game_ast']:.1f}")
                            
                            st.markdown("---")
                            
                            st.markdown("### Matchup Insights")
                            
                            team_stats = get_team_defensive_stats(opponent, season_data['SEASON'].iloc[0])
                            if team_stats:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    def_rtg = min(max(team_stats['def_rtg'], 100), 120) if team_stats['def_rtg'] is not None else 0
                                    st.metric("Defensive Rating", f"{def_rtg:.1f}")
                                    
                                    pace = team_stats['pace'] if team_stats['pace'] is not None else 0
                                    st.metric("Pace", f"{pace:.1f}")
                                
                                with col2:
                                    efg_pct = min(max(team_stats['efg_pct'] * 100, 45), 60) if team_stats['efg_pct'] is not None else 0
                                    st.metric("Opponent eFG%", f"{efg_pct:.1f}%")
                                    
                                    tov_pct = min(max(team_stats['tov_pct'] * 100, 10), 20) if team_stats['tov_pct'] is not None else 0
                                    st.metric("Opponent TOV%", f"{tov_pct:.1f}%")
                                
                                with col3:
                                    oreb_pct = min(max(team_stats['oreb_pct'] * 100, 20), 35) if team_stats['oreb_pct'] is not None else 0
                                    st.metric("Opponent OREB%", f"{oreb_pct:.1f}%")
                                    
                                    ft_rate = min(max(team_stats['ft_rate'], 0.15), 0.35) if team_stats['ft_rate'] is not None else 0
                                    st.metric("Opponent FT Rate", f"{ft_rate:.2f}")
                                
                                st.markdown("""
                                <style>
                                .metric-explanation {
                                    background-color: #f0f2f6;
                                    padding: 15px;
                                    border-radius: 5px;
                                    margin-top: 10px;
                                    color: #262730;
                                }
                                .metric-explanation summary {
                                    color: #262730;
                                    font-weight: bold;
                                    cursor: pointer;
                                }
                                .metric-explanation ul {
                                    color: #262730;
                                    margin-top: 10px;
                                }
                                .metric-explanation li {
                                    margin-bottom: 5px;
                                }
                                </style>
                                <div class="metric-explanation">
                                    <details>
                                        <summary>Understanding Defensive Metrics</summary>
                                        <ul>
                                            <li><b>Defensive Rating</b>: Points allowed per 100 possessions (lower is better, typically 100-120)</li>
                                            <li><b>Pace</b>: Possessions per 48 minutes (higher means faster-paced games)</li>
                                            <li><b>eFG%</b>: Effective field goal percentage allowed (accounts for 3-pointers being worth more, typically 45-60%)</li>
                                            <li><b>TOV%</b>: Opponent turnover percentage (higher means better at forcing turnovers, typically 10-20%)</li>
                                            <li><b>OREB%</b>: Opponent offensive rebound percentage (lower means better at defensive rebounding, typically 20-35%)</li>
                                            <li><b>FT Rate</b>: Free throw attempts per field goal attempt (lower means better at avoiding fouls, typically 0.15-0.35)</li>
                                        </ul>
                                    </details>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info("No defensive statistics available for this opponent.")
                        else:
                            st.info(f"No previous matchups found against {opponent} this season.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("No upcoming games found in the schedule. The NBA season may be in the offseason.")
    
    with tab3:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_season = st.selectbox(
                "Select Season",
                options=available_seasons,
                index=0,
                key="season_select"
            )
        
        season_data = df_clean[df_clean['SEASON'] == selected_season]
        
        st.subheader(f"Game-by-Game Performance - {selected_season}")
        
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
            title=f"{player_name}'s Game Log - {selected_season}",
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
        st.subheader("Shot Distance Analysis")
        
        shot_data = get_shot_data(player_name)
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
                title=f"{player_name}'s Shot Distribution by Distance",
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
            
            col1, col2, col3, col4 = st.columns(4)
            
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
                        f"Shoots {best_fg_pct:.1f}% from this distance"
                    )
                else:
                    st.metric("Most Efficient Range", "N/A", "Not enough shot attempts")
            
            with col3:
                if not distance_stats.empty:
                    most_common = distance_stats.loc[distance_stats['Attempts'].idxmax(), 'Distance_Range']
                    most_attempts = distance_stats['Attempts'].max()
                    most_fg_pct = distance_stats.loc[distance_stats['Attempts'].idxmax(), 'FG%']
                    st.metric(
                        "Favorite Shooting Range",
                        f"{most_common} ft",
                        f"{most_attempts} attempts, {most_fg_pct:.1f}% shooting"
                    )
                else:
                    st.metric("Favorite Shooting Range", "N/A", "No shot data available")
            
            with col4:
                total_shots = len(shot_data)
                made_shots = shot_data['SHOT_MADE_FLAG'].sum()
                fg_pct = (made_shots / total_shots * 100)
                st.metric(
                    "Overall Shooting",
                    f"{fg_pct:.1f}%",
                    f"{made_shots}/{total_shots} shots made"
                )
            
            st.subheader("Shot Type Analysis")
            
            # Aggregate data for 2PT vs 3PT shots
            shot_data['Shot_Category'] = shot_data['SHOT_TYPE'].apply(lambda x: '3PT' if '3PT' in x else '2PT')
            shot_category_stats = shot_data.groupby('Shot_Category').agg({
                'SHOT_MADE_FLAG': ['count', 'mean'],
                'SHOT_DISTANCE': 'mean'
            }).reset_index()
            
            shot_category_stats.columns = ['Shot_Category', 'Attempts', 'FG%', 'Avg_Distance']
            shot_category_stats['FG%'] = shot_category_stats['FG%'] * 100
            
            # Fill NaN values with 0
            shot_category_stats['FG%'] = shot_category_stats['FG%'].fillna(0)
            shot_category_stats['Avg_Distance'] = shot_category_stats['Avg_Distance'].fillna(0)
            
            # Create the pie chart
            fig_types = go.Figure()
            
            # Add pie chart for shot distribution
            fig_types.add_trace(go.Pie(
                labels=shot_category_stats['Shot_Category'],
                values=shot_category_stats['Attempts'],
                hole=0.4,
                marker=dict(
                    colors=['#9B6B9E', '#E6A4B4'],  # Purple for 2PT, Pink for 3PT
                    line=dict(color='white', width=2)
                ),
                textinfo='label+percent',
                textposition='inside',
                hovertemplate="<b>%{label}</b><br>" +
                            "Attempts: %{value}<extra></extra>"
            ))
            
            # Update layout
            fig_types.update_layout(
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
            
            # Add some space
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display FG% in a clean way
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
            
            # Add more space before the guide
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Move the shot analysis guide to the bottom
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
            
            # After the existing shot type analysis
            st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.warning("Shot data not available for this player.")
    
    with tab5:
        st.subheader("Player Comparison")
        
        selected_season = st.selectbox(
            "Select Season",
            options=available_seasons,
            index=0,
            key="comparison_season_select"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Main Player")
            st.markdown(f"**{player_name}**")
            st.markdown("(Selected in sidebar)")
        
        with col2:
            st.subheader("Compare With")
            compare_players = st.multiselect(
                "Select Players to Compare",
                options=[p for p in player_list if p != player_name],
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
                    player: data[data['SEASON'] == selected_season]
                    for player, data in player_data.items()
                }
                
                st.subheader("Basic Stats Comparison")
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
                    title="Season Average Comparison",
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
                
                st.subheader("Advanced Stats Comparison")
                
                # Initialize empty list for stats
                stats = []
                
                # For each player, get their stats
                for player in all_players:
                    if player in season_data_dict:
                        data = season_data_dict[player]
                        
                        # Get basic stats
                        player_stats = {
                            'Player': player,
                            'PTS': round(data['PTS'].mean(), 1),
                            'AST': round(data['AST'].mean(), 1),
                            'REB': round(data['REB'].mean(), 1),
                            'TOV': round(data['TOV'].mean(), 1),
                            'STL': round(data['STL'].mean(), 1),
                            'BLK': round(data['BLK'].mean(), 1)
                        }
                        
                        # Get shooting percentages from totals
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
                
                # Create and display DataFrame
                if stats:
                    df = pd.DataFrame(stats)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.subheader("Game Score Comparison")
                
                # Initialize empty list for game scores
                game_scores = []
                
                # For each player, calculate game score
                for player in all_players:
                    if player in season_data_dict:
                        data = season_data_dict[player]
                        
                        # Calculate game score
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
                
                # Create and display DataFrame
                if game_scores:
                    game_score_df = pd.DataFrame(game_scores)
                    st.dataframe(game_score_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No valid player data available for comparison.")
        else:
            st.info("Select players to compare from the dropdown above.") 