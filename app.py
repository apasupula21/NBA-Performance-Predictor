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
player_name = st.sidebar.selectbox("Select Player", player_list, index=player_list.index("Cade Cunningham") if "Cade Cunningham" in player_list else 0)
compare_player = st.sidebar.selectbox("Select Player to Compare (optional)", [""] + player_list)

@st.cache_data
def load_player_data(player_name):
    df = get_all_game_logs(player_name)
    if df.empty:
        st.error("No data found for this player")
        return None
    
    # Sort by date and ensure we have all games
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
                timeout=30  # Increase timeout to 30 seconds
            )
            df = shot_data.get_data_frames()[0]
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed. Retrying...")
                time.sleep(2)  # Wait 2 seconds before retrying
            else:
                st.error(f"Failed to fetch shot data after {max_retries} attempts: {e}")
                return None

df = load_player_data(player_name)

if df is not None:
    df_clean = clean_features(df)
    current_season = '2024-25'
    current_season_data = df_clean[df_clean['SEASON'] == current_season]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Season PPG", f"{current_season_data['PTS'].mean():.1f}")
    with col2:
        st.metric("Current Season RPG", f"{current_season_data['REB'].mean():.1f}")
    with col3:
        st.metric("Current Season APG", f"{current_season_data['AST'].mean():.1f}")
    
    st.subheader("Shooting Heatmap")
    shot_df = get_shot_data(player_name, current_season)
    
    if shot_df is not None and not shot_df.empty:
        court_fig = go.Figure()
        
        court_length = 94
        court_width = 50
        hoop_radius = 1.5
        key_width = 16
        key_length = 19
        three_point_radius = 23.75
        three_point_side_radius = 22
        
        court_fig.add_shape(type="rect", x0=-court_width/2, y0=0, x1=court_width/2, y1=court_length,
                          line=dict(color="black", width=2), fillcolor="white")
        
        court_fig.add_shape(type="rect", x0=-key_width/2, y0=0, x1=key_width/2, y1=key_length,
                          line=dict(color="black", width=2))
        
        court_fig.add_shape(type="circle", x0=-hoop_radius, y0=key_length-hoop_radius, 
                          x1=hoop_radius, y1=key_length+hoop_radius,
                          line=dict(color="black", width=2))
        
        court_fig.add_shape(type="path",
                          path=f"M {-three_point_side_radius} 0 L {-three_point_side_radius} {three_point_radius} A {three_point_radius} {three_point_radius} 0 0 0 {three_point_side_radius} {three_point_radius} L {three_point_side_radius} 0",
                          line=dict(color="black", width=2))
        
        made_shots = shot_df[shot_df['SHOT_MADE_FLAG'] == 1]
        missed_shots = shot_df[shot_df['SHOT_MADE_FLAG'] == 0]
        
        court_fig.add_trace(go.Scatter(
            x=made_shots['LOC_X'],
            y=made_shots['LOC_Y'],
            mode='markers',
            name='Made Shots',
            marker=dict(
                color='#00FF00',
                size=12,
                opacity=0.8,
                line=dict(color='black', width=1)
            ),
            customdata=np.stack((made_shots['SHOT_DISTANCE'], made_shots['SHOT_TYPE']), axis=-1),
            hovertemplate="<b>Made Shot</b><br>Distance: %{customdata[0]} ft<br>Type: %{customdata[1]}<extra></extra>"
        ))
        
        court_fig.add_trace(go.Scatter(
            x=missed_shots['LOC_X'],
            y=missed_shots['LOC_Y'],
            mode='markers',
            name='Missed Shots',
            marker=dict(
                color='#FF0000',
                size=12,
                opacity=0.8,
                line=dict(color='black', width=1)
            ),
            customdata=np.stack((missed_shots['SHOT_DISTANCE'], missed_shots['SHOT_TYPE']), axis=-1),
            hovertemplate="<b>Missed Shot</b><br>Distance: %{customdata[0]} ft<br>Type: %{customdata[1]}<extra></extra>"
        ))
        
        court_fig.update_layout(
            title=f"{player_name}'s Shot Chart - {current_season}",
            xaxis=dict(
                range=[-court_width/2, court_width/2],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            yaxis=dict(
                range=[0, court_length],
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            showlegend=True,
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(court_fig, use_container_width=True)
        
        total_shots = len(shot_df)
        made_shots_count = len(made_shots)
        fg_percentage = (made_shots_count / total_shots) * 100 if total_shots > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Shots", total_shots)
        with col2:
            st.metric("Made Shots", made_shots_count)
        with col3:
            st.metric("Field Goal %", f"{fg_percentage:.1f}%")
    
    model, x_test, y_test, predictions = train_eval(df_clean)
    
    if model is None:
        st.warning("Not enough data available for model training. Showing visualizations only.")
    else:
        st.subheader("Next Game Prediction")
        
        try:
            team_id = get_player_team(player_name)
            schedule = fetch_schedule()
            next_game = find_next_game(schedule, team_id)
            
            if next_game:
                opponent = next_game['opponent_team_name']
                st.write(f"Next Game: vs {opponent} on {next_game['game_date'].strftime('%B %d, %Y')}")
                
                team_stats_df = load_team_defense_data()
                prediction = predict_against_opponent(player_name, opponent, current_season_data, model, team_stats_df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Points", f"{prediction['Predicted PTS']:.1f}")
                with col2:
                    st.metric("Predicted Rebounds", f"{prediction['Predicted REB']:.1f}")
                with col3:
                    st.metric("Predicted Assists", f"{prediction['Predicted AST']:.1f}")
                
                h2h_games = current_season_data[current_season_data['MATCHUP'].str.contains(opponent, case=False)]
                if not h2h_games.empty:
                    st.subheader(f"Historical Performance vs {opponent}")
                    fig_h2h = go.Figure()
                    fig_h2h.add_trace(go.Bar(x=h2h_games['GAME_DATE'], y=h2h_games['PTS'], 
                                           name='Points', marker_color='#FF4B4B'))
                    fig_h2h.add_trace(go.Bar(x=h2h_games['GAME_DATE'], y=h2h_games['REB'], 
                                           name='Rebounds', marker_color='#00CC96'))
                    fig_h2h.add_trace(go.Bar(x=h2h_games['GAME_DATE'], y=h2h_games['AST'], 
                                           name='Assists', marker_color='#636EFA'))
                    
                    fig_h2h.update_layout(
                        title=f"{player_name}'s Performance vs {opponent}",
                        xaxis_title="Date",
                        yaxis_title="Stats",
                        barmode='group',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_h2h, use_container_width=True)
            else:
                st.warning("No upcoming games found in the schedule.")
        except Exception as e:
            st.error(f"Error fetching next game information: {e}")
    
    st.subheader("Advanced Stats Per Game")
    
    current_season_data['GAME_DATE'] = pd.to_datetime(current_season_data['GAME_DATE'])
    current_season_data = current_season_data.sort_values('GAME_DATE')
    
    # Calculate Game Score with full formula
    game_score = current_season_data['PTS']
    if 'FGM' in current_season_data.columns:
        game_score += 0.4 * current_season_data['FGM']
    if 'FGA' in current_season_data.columns:
        game_score -= 0.7 * current_season_data['FGA']
    if 'FTM' in current_season_data.columns:
        game_score += 0.4 * current_season_data['FTM']
    if 'FTA' in current_season_data.columns:
        game_score -= 0.7 * current_season_data['FTA']
    if 'OREB' in current_season_data.columns:
        game_score += 0.7 * current_season_data['OREB']
    if 'DREB' in current_season_data.columns:
        game_score += 0.3 * current_season_data['DREB']
    if 'STL' in current_season_data.columns:
        game_score += current_season_data['STL']
    if 'AST' in current_season_data.columns:
        game_score += 0.7 * current_season_data['AST']
    if 'BLK' in current_season_data.columns:
        game_score += 0.7 * current_season_data['BLK']
    if 'TOV' in current_season_data.columns:
        game_score -= current_season_data['TOV']
    if 'PF' in current_season_data.columns:
        game_score -= 0.4 * current_season_data['PF']
    
    current_season_data['GAME_SCORE'] = game_score
    
    # Add tooltip explaining Game Score calculation
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
    
    # Create figure for Game Score
    fig_advanced = go.Figure()
    
    # Add Game Score trace
    fig_advanced.add_trace(go.Scatter(
        x=current_season_data['GAME_DATE'],
        y=current_season_data['GAME_SCORE'],
        mode='lines+markers',
        name='Game Score',
        line=dict(width=2, color='#FF4B4B'),
        marker=dict(size=8, color='#FF4B4B'),
        hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                    "<b>Game Score</b>: %{y:.1f}<br>" +
                    "<b>Opponent</b>: %{customdata[1]}<extra></extra>",
        customdata=np.column_stack((
            current_season_data['GAME_DATE'].dt.strftime('%b %d'),
            current_season_data['MATCHUP']
        ))
    ))
    
    # Add rolling average with proper window
    window = 5
    current_season_data['GAME_SCORE_AVG'] = current_season_data['GAME_SCORE'].rolling(window=window, min_periods=1).mean()
    
    # Add rolling average trace
    fig_advanced.add_trace(go.Scatter(
        x=current_season_data['GAME_DATE'],
        y=current_season_data['GAME_SCORE_AVG'],
        mode='lines',
        name=f'{window}-Game Avg',
        line=dict(width=2, color='#636EFA', dash='dash'),
        hovertemplate="<b>Game</b>: %{customdata[0]}<br>" +
                    f"<b>{window}-Game Avg</b>: %{{y:.1f}}<extra></extra>",
        customdata=current_season_data['GAME_DATE'].dt.strftime('%b %d')
    ))
    
    # Update layout
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
    
    # Add Game Score summary
    st.subheader("Game Score Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Average Game Score", f"{current_season_data['GAME_SCORE'].mean():.1f}")
        st.metric("Best Game Score", f"{current_season_data['GAME_SCORE'].max():.1f}")
    
    with col2:
        # Calculate the most recent 5-game average
        recent_avg = current_season_data['GAME_SCORE'].tail(window).mean()
        # Calculate the previous 5-game average for trend
        prev_avg = current_season_data['GAME_SCORE'].tail(window*2).head(window).mean()
        trend = recent_avg - prev_avg
        
        st.metric(f"Recent {window}-Game Average", f"{recent_avg:.1f}")
        st.metric("Trend", f"{trend:+.1f}")
    
    # Add best performance
    best_game = current_season_data.loc[current_season_data['GAME_SCORE'].idxmax()]
    st.markdown(f"**Best Performance**: {best_game['GAME_DATE'].strftime('%b %d')} vs {best_game['MATCHUP'].split()[-1]}")
    st.markdown(f"Game Score: {best_game['GAME_SCORE']:.1f} | Points: {best_game['PTS']} | Rebounds: {best_game['REB']} | Assists: {best_game['AST']}")
    
    st.subheader("Game-by-Game Performance")
    
    # Add date range selector
    min_date = current_season_data['GAME_DATE'].min()
    max_date = current_season_data['GAME_DATE'].max()
    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = current_season_data[
            (current_season_data['GAME_DATE'].dt.date >= start_date) &
            (current_season_data['GAME_DATE'].dt.date <= end_date)
        ]
        
        # Add averages for selected date range
        st.markdown(f"**Averages for Selected Period ({start_date.strftime('%b %d')} - {end_date.strftime('%b %d')})**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Points", f"{filtered_data['PTS'].mean():.1f}")
        with col2:
            st.metric("Rebounds", f"{filtered_data['REB'].mean():.1f}")
        with col3:
            st.metric("Assists", f"{filtered_data['AST'].mean():.1f}")
        
        # Create game-by-game performance chart
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
                        "<b>Opponent</b>: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack((
                filtered_data['GAME_DATE'].dt.strftime('%b %d'),
                filtered_data['MATCHUP']
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
                        "<b>Opponent</b>: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack((
                filtered_data['GAME_DATE'].dt.strftime('%b %d'),
                filtered_data['MATCHUP']
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
                        "<b>Opponent</b>: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack((
                filtered_data['GAME_DATE'].dt.strftime('%b %d'),
                filtered_data['MATCHUP']
            ))
        ))
        
        fig_games.update_layout(
            title=f"{player_name}'s Game Log",
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
    
    if compare_player:
        st.subheader(f"Player Comparison: {player_name} vs {compare_player}")
        compare_df = load_player_data(compare_player)
        if compare_df is not None:
            compare_df_clean = clean_features(compare_df)
            compare_season_data = compare_df_clean[compare_df_clean['SEASON'] == current_season]
            
            fig_compare = go.Figure()
            
            fig_compare.add_trace(go.Bar(
                x=['Points', 'Rebounds', 'Assists'],
                y=[current_season_data['PTS'].mean(), current_season_data['REB'].mean(), current_season_data['AST'].mean()],
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