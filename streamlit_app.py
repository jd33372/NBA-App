import pandas as pd
import numpy as np
import streamlit as st 
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Streamlit caching for better performance
@st.cache_data
def load_and_process_data():
    """Load and preprocess NBA data with caching for better performance"""
    try:
        # For deployment: CSV should be in the same directory as the app
        csv_path = "NBA_career_stats.csv"  # Place CSV in same folder as this file
        
        # Try local path first, then fallback to Downloads folder for local testing
        try:
            nba_data = pd.read_csv(csv_path)
        except FileNotFoundError:
            # Fallback for local development
            nba_data = pd.read_csv("C:/Users/james/Downloads/NBA_career_stats.csv")
            st.info("Using local development path for CSV file")
        
        # Validate required columns exist
        if 'player' not in nba_data.columns:
            st.error("Dataset must contain a 'player' column")
            return None, None
        
        # Handle missing position data
        if 'pos' not in nba_data.columns:
            nba_data['pos'] = 'Unknown'
        
        # Player Position DataFrame
        pos_df = nba_data[["player", "pos"]].reset_index(drop=True)
        
        # Normalize stats for comparison (only numerical columns)
        stats = nba_data.select_dtypes(include=[np.number])
        
        # Handle missing values before scaling
        stats = stats.fillna(stats.mean())
        
        # Initialize scaler
        scaler = StandardScaler()
        normalized_stats = scaler.fit_transform(stats)
        
        # Create normalized DataFrame
        normalized_df = pd.DataFrame(normalized_stats, columns=stats.columns, index=nba_data.index)
        
        # Combine position and normalized stats
        final_df = pd.concat([pos_df, normalized_df], axis=1)
        
        # Calculate Career Score (more robust calculation)
        numerical_columns = final_df.select_dtypes(include=[np.number]).columns
        final_df['Career Score'] = final_df[numerical_columns].sum(axis=1)
        
        return final_df, nba_data
        
    except FileNotFoundError:
        st.error("NBA_career_stats.csv file not found. Please make sure the file is in the same directory as this app.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Load data using cached function
final_df, original_data = load_and_process_data()

# Only proceed if data loaded successfully
if final_df is not None:
    
    st.title("🏀 NBA Player Comparison Tool")
    st.markdown("Find NBA players with similar career performance based on statistical analysis")
    
    st.sidebar.header("🎯 Player Selection")
    
    def get_player_input():
        player_name = st.sidebar.selectbox("Select Player", final_df['player'].unique())
        num_comparisons = st.sidebar.selectbox("Number of Similar Players", [1, 2, 3, 4, 5])
        same_position_only = st.sidebar.checkbox("Same Position Only", value=False)
        
        return player_name, num_comparisons, same_position_only
    
    def find_similar_players(target_player, num_comparisons, same_position_only=False):
        """Find similar players based on career score"""
        # Get target player data
        target_data = final_df[final_df['player'] == target_player]
        if target_data.empty:
            return pd.DataFrame()
        
        # Get all other players
        other_players = final_df[final_df['player'] != target_player].copy()
        
        # Filter by same position if requested
        if same_position_only:
            target_position = target_data['pos'].iloc[0]
            other_players = other_players[other_players['pos'] == target_position]
        
        # Find players with similar career scores
        target_score = target_data['Career Score'].iloc[0]
        other_players['score_difference'] = abs(other_players['Career Score'] - target_score)
        
        # Sort by score difference (ascending) and return top matches
        similar_players = other_players.nsmallest(num_comparisons, 'score_difference')
        
        return similar_players
        
    def display_player_comparison(target_player, similar_players, same_position_only=False):
        """Display detailed comparison between target player and similar players"""
        position_text = " (Same Position)" if same_position_only else ""
        st.subheader(f"Players Similar to {target_player}{position_text}")
        
        # Get target player info
        target_info = final_df[final_df['player'] == target_player].iloc[0]
        
        # Display target player info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target Player", target_player)
        with col2:
            st.metric("Position", target_info['pos'])
        with col3:
            st.metric("Career Score", f"{target_info['Career Score']:.2f}")
        
        # Display similar players
        if not similar_players.empty:
            st.write("### Similar Players (by Career Score):")
            
            for idx, (_, player) in enumerate(similar_players.iterrows(), 1):
                with st.expander(f"{idx}. {player['player']} - {player['pos']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Position:** {player['pos']}")
                        st.write(f"**Career Score:** {player['Career Score']:.2f}")
                    
                    with col2:
                        st.write(f"**Score Difference:** {player['score_difference']:.2f}")
                        score_similarity = max(0, 100 - (player['score_difference'] * 10))
                        st.write(f"**Score Similarity:** {score_similarity:.1f}%")
                    
                    with col3:
                        # Show some key stats if available
                        key_stats = ['pts', 'reb', 'ast', 'fg_pct'] if any(stat in original_data.columns for stat in ['pts', 'reb', 'ast', 'fg_pct']) else []
                        for stat in key_stats[:3]:  # Show max 3 stats
                            if stat in original_data.columns:
                                original_player_data = original_data[original_data['player'] == player['player']]
                                if not original_player_data.empty:
                                    st.write(f"**{stat.upper()}:** {original_player_data[stat].iloc[0]}")
        else:
            warning_msg = f"No players found with similar career scores"
            if same_position_only:
                target_pos = final_df[final_df['player'] == target_player]['pos'].iloc[0]
                warning_msg += f" in the {target_pos} position"
            st.warning(warning_msg + ".")
    
    # Main Application Logic
    player_name, num_comparisons, same_position_only = get_player_input()
    
    if st.button("🔍 Find Similar Players", type="primary"):
        with st.spinner("Finding similar players by career score..."):
            similar_players = find_similar_players(player_name, num_comparisons, same_position_only)
            display_player_comparison(player_name, similar_players, same_position_only)
    
    # Additional features
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    
    if st.sidebar.button("Show Dataset Statistics"):
        st.subheader("📈 Dataset Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Players", len(final_df))
        with col2:
            st.metric("Available Columns", len(final_df.columns))
        
        # Show position distribution
        if 'pos' in final_df.columns:
            st.subheader("Player Distribution by Position")
            pos_counts = final_df['pos'].value_counts()
            fig = px.bar(x=pos_counts.index, y=pos_counts.values, 
                        title="Player Distribution by Position",
                        labels={'x': 'Position', 'y': 'Number of Players'},
                        color=pos_counts.values,
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        # Show top players by career score
        st.subheader("🏆 Top 10 Players by Career Score")
        top_players = final_df.nlargest(10, 'Career Score')[['player', 'pos', 'Career Score']]
        st.dataframe(top_players, use_container_width=True)

else:
    st.error("❌ Failed to load NBA data. Please make sure NBA_career_stats.csv is in the same directory as this app.")
    st.info("For deployment on Streamlit Cloud, upload your CSV file to your GitHub repository.")