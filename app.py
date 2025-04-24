import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nba_predictor import get_all_game_logs, predict_against_opponent, load_team_defense_data
from nba_api.stats.static import players

st.set_page_config(layout="wide")

st.title("NBA Player Performance Predictor")
st.markdown("### Visualize player stats and get performance predictions")

# Sidebar for player selection
st.sidebar.header("Player Selection")
player_name = st.sidebar.text_input("Enter player name", "Cade Cunningham")

# Get player data
@st.cache_data
def load_player_data(player_name):
    df = get_all_game_logs(player_name)
    if df.empty:
        st.error("No data found for this player")
        return None
    return df

df = load_player_data(player_name)

if df is not None:
    # Filter for current season
    current_season = '2024-25'
    current_season_data = df[df['SEASON'] == current_season]
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Season PPG", f"{current_season_data['PTS'].mean():.1f}")
    with col2:
        st.metric("Current Season RPG", f"{current_season_data['REB'].mean():.1f}")
    with col3:
        st.metric("Current Season APG", f"{current_season_data['AST'].mean():.1f}")
    
    # Rolling averages chart
    st.subheader("Rolling Averages")
    rolling_window = st.slider("Rolling Window Size", 3, 15, 5)
    
    current_season_data['PTS_rolling'] = current_season_data['PTS'].rolling(rolling_window).mean()
    current_season_data['REB_rolling'] = current_season_data['REB'].rolling(rolling_window).mean()
    current_season_data['AST_rolling'] = current_season_data['AST'].rolling(rolling_window).mean()
    
    fig_rolling = go.Figure()
    fig_rolling.add_trace(go.Scatter(x=current_season_data['GAME_DATE'], y=current_season_data['PTS_rolling'], 
                                    name='Points', line=dict(color='#FF4B4B')))
    fig_rolling.add_trace(go.Scatter(x=current_season_data['GAME_DATE'], y=current_season_data['REB_rolling'], 
                                    name='Rebounds', line=dict(color='#00CC96')))
    fig_rolling.add_trace(go.Scatter(x=current_season_data['GAME_DATE'], y=current_season_data['AST_rolling'], 
                                    name='Assists', line=dict(color='#636EFA')))
    
    fig_rolling.update_layout(
        title=f"{player_name}'s {rolling_window}-Game Rolling Averages",
        xaxis_title="Date",
        yaxis_title="Average",
        hovermode='x unified'
    )
    st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Game-by-game performance
    st.subheader("Game-by-Game Performance")
    fig_games = go.Figure()
    fig_games.add_trace(go.Bar(x=current_season_data['GAME_DATE'], y=current_season_data['PTS'], 
                              name='Points', marker_color='#FF4B4B'))
    fig_games.add_trace(go.Bar(x=current_season_data['GAME_DATE'], y=current_season_data['REB'], 
                              name='Rebounds', marker_color='#00CC96'))
    fig_games.add_trace(go.Bar(x=current_season_data['GAME_DATE'], y=current_season_data['AST'], 
                              name='Assists', marker_color='#636EFA'))
    
    fig_games.update_layout(
        title=f"{player_name}'s Game-by-Game Stats",
        xaxis_title="Date",
        yaxis_title="Stats",
        barmode='group',
        hovermode='x unified'
    )
    st.plotly_chart(fig_games, use_container_width=True)
    
    # Prediction section
    st.subheader("Next Game Prediction")
    opponent = st.selectbox("Select Opponent", 
                          sorted(set(current_season_data['OPPONENT_FULL'].dropna().unique())))
    
    if opponent:
        team_stats_df = load_team_defense_data()
        prediction = predict_against_opponent(player_name, opponent, current_season_data, None, team_stats_df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Points", f"{prediction['Predicted PTS']:.1f}")
        with col2:
            st.metric("Predicted Rebounds", f"{prediction['Predicted REB']:.1f}")
        with col3:
            st.metric("Predicted Assists", f"{prediction['Predicted AST']:.1f}")
        
        # Historical performance against opponent
        h2h_games = current_season_data[current_season_data['OPPONENT_FULL'] == opponent]
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