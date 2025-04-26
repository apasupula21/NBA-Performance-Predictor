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
    get_player_team
)
from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail
import numpy as np
from datetime import datetime
import time

st.set_page_config(layout="wide")

st.title("NBA Player Performance Predictor")
st.markdown("### Visualize player stats and get performance predictions")

@st.cache_data
def get_player_list():
    player_list = players.get_active_players()
    return sorted([player['full_name'] for player in player_list])

player_list = get_player_list()

st.sidebar.header("Player Selection")
player_name = st.sidebar.selectbox("Select Player", player_list, 
                                 index=player_list.index("James Harden") if "James Harden" in player_list else 0,
                                 key="player_select")
compare_player = st.sidebar.selectbox("Select Player to Compare (optional)", [""] + player_list,
                                    key="compare_player_select")

@st.cache_data
def load_player_data(player_name, start_season='2010-11', end_season='2024-25'):
    df = get_all_game_logs(player_name, start_season, end_season)
    if df.empty:
        st.error("No data found for this player")
        return None
    
    df = df.sort_values('GAME_DATE')
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    
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

df = load_player_data(player_name)

if df is not None:
    df_clean = clean_features(df, player_name)
    
    # Get unique seasons and sort them in reverse chronological order
    available_seasons = sorted(df_clean['SEASON'].unique(), reverse=True)
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Overview", "Game-by-Game Performance"])
    
    with tab1:
        # Use the most recent season for overview stats
        season_data = df_clean[df_clean['SEASON'] == available_seasons[0]]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Season PPG", f"{season_data['PTS'].mean():.1f}")
        with col2:
            st.metric("Season RPG", f"{season_data['REB'].mean():.1f}")
        with col3:
            st.metric("Season APG", f"{season_data['AST'].mean():.1f}")
        
        model, x_test, y_test, predictions = train_eval(df_clean)
        
        if model is None:
            st.warning("Not enough data available for model training. Showing visualizations only.")
        else:
            st.subheader("Next Game Prediction")
            
            try:
                team_id = get_player_team(player_name)
                schedule = fetch_schedule()
                next_game = find_next_game(schedule, team_id) if team_id and schedule else None
                
                if not team_id:
                    st.warning("Could not find team information for the player.")
                elif not schedule:
                    st.warning("Could not fetch the schedule.")
                elif not next_game:
                    st.warning("No upcoming games found in the schedule.")
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
                        st.metric("Predicted Points", f"{prediction['Predicted PTS']:.1f}", 
                                f"{prediction['Predicted PTS'] - prediction['Season Avg PTS']:+.1f} vs season avg")
                    with col2:
                        st.metric("Predicted Rebounds", f"{prediction['Predicted REB']:.1f}", 
                                f"{prediction['Predicted REB'] - prediction['Season Avg REB']:+.1f} vs season avg")
                    with col3:
                        st.metric("Predicted Assists", f"{prediction['Predicted AST']:.1f}", 
                                f"{prediction['Predicted AST'] - prediction['Season Avg AST']:+.1f} vs season avg")
            except Exception as e:
                st.error(f"Error fetching next game information: {str(e)}")
    
    with tab2:
        # Add season selector directly in the Game-by-Game Performance tab
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_season = st.selectbox(
                "Select Season",
                options=available_seasons,
                index=0,  # Default to most recent season
                key="season_select"
            )
        
        # Filter data for selected season
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
        
        # Add 5-game rolling averages
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
        
        st.plotly_chart(fig_games, use_container_width=True)
        
        # Show team changes for the selected season
        team_changes = season_data.groupby('TEAM_ABBREVIATION').agg({
            'GAME_DATE': ['min', 'max', 'count']
        }).reset_index()
        team_changes.columns = ['Team', 'First Game', 'Last Game', 'Games Played']
        team_changes['First Game'] = team_changes['First Game'].dt.strftime('%b %d, %Y')
        team_changes['Last Game'] = team_changes['Last Game'].dt.strftime('%b %d, %Y')
        st.markdown("**Team Changes**")
        st.dataframe(team_changes, hide_index=True)

    st.subheader("Advanced Stats Per Game")
    
    season_data['GAME_DATE'] = pd.to_datetime(season_data['GAME_DATE'])
    season_data = season_data.sort_values('GAME_DATE')
    
    game_score = season_data['PTS']
    if 'FGM' in season_data.columns:
        game_score += 0.4 * season_data['FGM']
    if 'FGA' in season_data.columns:
        game_score -= 0.7 * season_data['FGA']
    if 'FTM' in season_data.columns:
        game_score += 0.4 * season_data['FTM']
    if 'FTA' in season_data.columns:
        game_score -= 0.7 * season_data['FTA']
    if 'OREB' in season_data.columns:
        game_score += 0.7 * season_data['OREB']
    if 'DREB' in season_data.columns:
        game_score += 0.3 * season_data['DREB']
    if 'STL' in season_data.columns:
        game_score += season_data['STL']
    if 'AST' in season_data.columns:
        game_score += 0.7 * season_data['AST']
    if 'BLK' in season_data.columns:
        game_score += 0.7 * season_data['BLK']
    if 'TOV' in season_data.columns:
        game_score -= season_data['TOV']
    if 'PF' in season_data.columns:
        game_score -= 0.4 * season_data['PF']
    
    season_data['GAME_SCORE'] = game_score
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 10px; color: #262730;">
        <details>
            <summary style="color: #262730; font-weight: bold;">How is Game Score calculated?</summary>
            <p style="color: #262730;">Game Score is calculated using John Hollinger's formula:</p>
            <ul style="color: #262730;">
                <li>Points: +1.0 per point</li>
                <li>Field Goals Made: +0.4 per FGM</li>
                <li>Field Goals Attempted: -0.7 per FGA</li>
                <li>Free Throws Made: +0.4 per FTM</li>
                <li>Free Throws Attempted: -0.7 per FTA</li>
                <li>Offensive Rebounds: +0.7 per OREB</li>
                <li>Defensive Rebounds: +0.3 per DREB</li>
                <li>Steals: +1.0 per STL</li>
                <li>Assists: +0.7 per AST</li>
                <li>Blocks: +0.7 per BLK</li>
                <li>Turnovers: -1.0 per TOV</li>
                <li>Personal Fouls: -0.4 per PF</li>
            </ul>
            <p style="color: #262730;">This provides a comprehensive measure of a player's overall contribution in a game.</p>
        </details>
    </div>
    """, unsafe_allow_html=True)
    
    fig_advanced = go.Figure()
    
    fig_advanced.add_trace(go.Scatter(
        x=season_data['GAME_DATE'],
        y=season_data['GAME_SCORE'],
        mode='lines+markers',
        name='Game Score',
        line=dict(width=2, color='#FF4B4B'),
        marker=dict(size=8, color='#FF4B4B'),
        hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                    "<b>Game Score</b>: %{y:.1f}<br>" +
                    "<b>Opponent</b>: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack((
            season_data['GAME_DATE'].dt.strftime('%b %d'),
            season_data['MATCHUP']
        ))
    ))
    
    window = 5
    season_data['GAME_SCORE_AVG'] = season_data['GAME_SCORE'].rolling(window=window, min_periods=1).mean()
    
    fig_advanced.add_trace(go.Scatter(
        x=season_data['GAME_DATE'],
        y=season_data['GAME_SCORE_AVG'],
        mode='lines',
        name=f'{window}-Game Avg',
        line=dict(width=2, color='#636EFA', dash='dash'),
        hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                    f"<b>{window}-Game Avg</b>: %{{y:.1f}}<extra></extra>",
        customdata=season_data['GAME_DATE'].dt.strftime('%b %d')
    ))
    
    fig_advanced.update_layout(
        title=f"{player_name}'s Game Score",
        xaxis_title="Date",
        yaxis_title="Game Score",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        xaxis=dict(
            type='date',
            rangeslider=dict(visible=True)
        )
    )
    
    st.plotly_chart(fig_advanced, use_container_width=True)
    
    st.subheader("Game Score Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Game Score", f"{season_data['GAME_SCORE'].mean():.1f}")
        st.metric("Best Game Score", f"{season_data['GAME_SCORE'].max():.1f}")
    
    with col2:
        recent_avg = season_data['GAME_SCORE'].tail(window).mean()
        prev_avg = season_data['GAME_SCORE'].tail(window*2).head(window).mean()
        trend = recent_avg - prev_avg
        
        st.metric(f"Recent {window}-Game Average", f"{recent_avg:.1f}")
        st.metric("Trend", f"{trend:+.1f}")
    
    best_game = season_data.loc[season_data['GAME_SCORE'].idxmax()]
    st.markdown(f"**Best Performance**: {best_game['GAME_DATE'].strftime('%b %d')} vs {best_game['MATCHUP'].split()[-1]}")
    st.markdown(f"Game Score: {best_game['GAME_SCORE']:.1f} | Points: {best_game['PTS']} | Rebounds: {best_game['REB']} | Assists: {best_game['AST']}")
    
    st.subheader("Game-by-Game Performance")
    
    min_date = season_data['GAME_DATE'].min()
    max_date = season_data['GAME_DATE'].max()
    
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    with col2:
        # Get unique seasons from the data
        all_seasons = sorted(df_clean['SEASON'].unique(), reverse=True)
        if not all_seasons:  # If no seasons found, add current season
            all_seasons = ['2024-25']
        selected_season = st.selectbox(
            "Select Season",
            options=all_seasons,
            index=0
        )
    
    # Filter data based on selected season
    filtered_data = df_clean[df_clean['SEASON'] == selected_season]
    
    # Get team information for the selected season
    season_teams = filtered_data['TEAM_ABBREVIATION'].unique()
    if len(season_teams) > 0:
        team_info = " & ".join(season_teams)
        st.markdown(f"**Team(s) in {selected_season}**: {team_info}")
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = filtered_data[
            (filtered_data['GAME_DATE'].dt.date >= start_date) &
            (filtered_data['GAME_DATE'].dt.date <= end_date)
        ]
    
    if not filtered_data.empty:
        st.markdown(f"**Averages for Selected Period ({start_date.strftime('%b %d')} - {end_date.strftime('%b %d')})**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Points", f"{filtered_data['PTS'].mean():.1f}")
        with col2:
            st.metric("Rebounds", f"{filtered_data['REB'].mean():.1f}")
        with col3:
            st.metric("Assists", f"{filtered_data['AST'].mean():.1f}")
        
        fig_games = go.Figure()
        
        fig_games.add_trace(go.Scatter(
            x=filtered_data['GAME_DATE'],
            y=filtered_data['PTS'],
            mode='lines+markers',
            name='Points',
            line=dict(width=2, color='#FF4B4B'),
            marker=dict(size=8),
            hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                        "<b>Points</b>: %{y}<br>" +
                        "<b>Opponent</b>: %{customdata[1]}<br>" +
                        "<b>Team</b>: %{customdata[2]}<extra></extra>",
            customdata=np.column_stack((
                filtered_data['GAME_DATE'].dt.strftime('%b %d'),
                filtered_data['MATCHUP'],
                filtered_data['TEAM_ABBREVIATION']
            ))
        ))
        
        fig_games.add_trace(go.Scatter(
            x=filtered_data['GAME_DATE'],
            y=filtered_data['REB'],
            mode='lines+markers',
            name='Rebounds',
            line=dict(width=2, color='#00CC96'),
            marker=dict(size=8),
            hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                        "<b>Rebounds</b>: %{y}<br>" +
                        "<b>Opponent</b>: %{customdata[1]}<br>" +
                        "<b>Team</b>: %{customdata[2]}<extra></extra>",
            customdata=np.column_stack((
                filtered_data['GAME_DATE'].dt.strftime('%b %d'),
                filtered_data['MATCHUP'],
                filtered_data['TEAM_ABBREVIATION']
            ))
        ))
        
        fig_games.add_trace(go.Scatter(
            x=filtered_data['GAME_DATE'],
            y=filtered_data['AST'],
            mode='lines+markers',
            name='Assists',
            line=dict(width=2, color='#636EFA'),
            marker=dict(size=8),
            hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                        "<b>Assists</b>: %{y}<br>" +
                        "<b>Opponent</b>: %{customdata[1]}<br>" +
                        "<b>Team</b>: %{customdata[2]}<extra></extra>",
            customdata=np.column_stack((
                filtered_data['GAME_DATE'].dt.strftime('%b %d'),
                filtered_data['MATCHUP'],
                filtered_data['TEAM_ABBREVIATION']
            ))
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
            )
        )
        
        st.plotly_chart(fig_games, use_container_width=True)
        
        # Show team changes for the selected season
        team_changes = filtered_data.groupby('TEAM_ABBREVIATION').agg({
            'GAME_DATE': ['min', 'max', 'count']
        }).reset_index()
        team_changes.columns = ['Team', 'First Game', 'Last Game', 'Games Played']
        team_changes['First Game'] = team_changes['First Game'].dt.strftime('%b %d, %Y')
        team_changes['Last Game'] = team_changes['Last Game'].dt.strftime('%b %d, %Y')
        st.markdown("**Team Changes**")
        st.dataframe(team_changes, hide_index=True)
    
    if compare_player:
        st.subheader(f"Player Comparison: {player_name} vs {compare_player}")
        compare_df = load_player_data(compare_player)
        if compare_df is not None:
            compare_df_clean = clean_features(compare_df)
            compare_season_data = compare_df_clean[compare_df_clean['SEASON'] == selected_season]
            
            fig_compare = go.Figure()
            
            fig_compare.add_trace(go.Bar(
                x=['Points', 'Rebounds', 'Assists'],
                y=[season_data['PTS'].mean(), season_data['REB'].mean(), season_data['AST'].mean()],
                name=player_name,
                marker_color='#FF4B4B'
            ))
            
            fig_compare.add_trace(go.Bar(
                x=['Points', 'Rebounds', 'Assists'],
                y=[compare_season_data['PTS'].mean(), compare_season_data['REB'].mean(), compare_season_data['AST'].mean()],
                name=compare_player,
                marker_color='#636EFA'
            ))
            
            fig_compare.update_layout(
                title="Season Average Comparison",
                xaxis_title="Stat",
                yaxis_title="Average",
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1
            )
            
            st.plotly_chart(fig_compare, use_container_width=True) 