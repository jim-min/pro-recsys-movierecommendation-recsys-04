import streamlit as st
import os
import pandas as pd
from loader import load_ratings, load_item_info
from utils import plot_user_interactions, get_user_stats, get_temporal_stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="RecSys Data Explorer", layout="wide")

def main():
    st.title("🎬 RecSys Data Explorer")

    # --- Sidebar ---
    st.sidebar.header("Configuration")
    
    # 1. Directory Input
    default_dir = os.path.join(os.getcwd(), 'data', 'train')
    data_dir = st.sidebar.text_input("Data Directory", value=default_dir)
    
    if st.sidebar.button("Load Data"):
        st.session_state['load_clicked'] = True
        
    if 'load_clicked' not in st.session_state:
        st.info("Please verify the data directory and click 'Load Data'.")
        return

    # Load Data
    with st.spinner("Loading data..."):
        ratings_df = load_ratings(data_dir)
        item_info_df = load_item_info(data_dir)
        
    if ratings_df is None:
        return

    # Display summary metrics below title
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Users", f"{ratings_df['user'].nunique():,}")
    m_col2.metric("Total Movies", f"{ratings_df['item'].nunique():,}")
    m_col3.metric("Total Interactions", f"{len(ratings_df):,}")
    
    # 2. Global Filters (Sidebar)
    st.sidebar.subheader("Filters")
    
    # Time Range
    min_time = ratings_df['datetime'].min()
    max_time = ratings_df['datetime'].max()
    
    time_range = st.sidebar.slider(
        "Time Range",
        min_value=min_time.to_pydatetime(),
        max_value=max_time.to_pydatetime(),
        value=(min_time.to_pydatetime(), max_time.to_pydatetime())
    )
    
    # User Selection Logic
    all_users = sorted(ratings_df['user'].unique())
    
    # Range Selection
    st.sidebar.markdown("### User Selection")
    user_select_mode = st.sidebar.radio("Selection Mode", ["Specific Users", "User ID Range"])
    
    selected_users = []
    
    if user_select_mode == "Specific Users":
        # Check if we need to update from session state (logic handled by 'key')
        if 'selected_users_manual' not in st.session_state:
            st.session_state['selected_users_manual'] = all_users[:1] if len(all_users) > 0 else []
            
        selected_users = st.sidebar.multiselect(
            "Select Users", 
            all_users, 
            key="selected_users_manual"
        )
    else:
        col1, col2 = st.sidebar.columns(2)
        min_id = int(min(all_users))
        max_id = int(max(all_users))
        
        start_id = col1.number_input("Start ID", min_value=min_id, max_value=max_id, value=min_id)
        end_id = col2.number_input("End ID", min_value=min_id, max_value=max_id, value=min(min_id + 50, max_id))
        
        if start_id <= end_id:
            selected_users = [u for u in all_users if start_id <= u <= end_id]
            st.sidebar.write(f"Selected {len(selected_users)} users.")
        else:
            st.sidebar.error("Start ID must be <= End ID")
            
    use_separate = st.sidebar.checkbox("Separate Time Axes per User", value=False)

    # --- Main Content ---
    
    tab1, tab2, tab3 = st.tabs(["User Interactions & Stats", "Item Info", "Overall Statistics"])
    
    # Filter by time
    filtered_df = ratings_df[
        (ratings_df['datetime'] >= time_range[0]) & 
        (ratings_df['datetime'] <= time_range[1])
    ]
    
    # --- Tab 1: Interactions ---
    with tab1:
        st.header("Visualizations")
        
        if selected_users:
            # Filter by users
            display_df = filtered_df[filtered_df['user'].isin(selected_users)]
            
            if display_df.empty:
                st.warning("No interactions found for selected users in current time range.")
            else:
                # Plot
                fig = plot_user_interactions(display_df, selected_users, use_separate_axis=use_separate)
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats per user
                st.subheader("User Statistics")
                stats = get_user_stats(display_df, selected_users)
                st.dataframe(stats)
                
                # Temporal breakdown
                st.subheader("Temporal Distribution")
                # temp_df contains filtered data for selected users
                temp_df = get_temporal_stats(display_df, selected_users)
                
                period = st.selectbox("Group By", ['Day', 'Week', 'Month', 'Year'])
                # Use 'YS' (Year Start) for Year to align with start of year labels, 'MS' for Month Start
                period_map_resample = {'Day': 'D', 'Week': 'W', 'Month': 'MS', 'Year': 'YS'}
                
                if not temp_df.empty:
                    # Resample for continuous axis
                    # Group by User and Date, then resample
                    vis_data = []
                    
                    for user in selected_users:
                        user_df = temp_df[temp_df['user'] == user].set_index('datetime')
                        # Resample count unique indices or just size
                        resampled = user_df.resample(period_map_resample[period]).size()
                        
                        # Create a DataFrame for this user
                        u_res = resampled.reset_index(name='count')
                        u_res['user'] = user
                        vis_data.append(u_res)
                        
                    if vis_data:
                        final_vis_df = pd.concat(vis_data)
                        
                        # Convert to string for categorical plotting to ensure alignment
                        # This avoids the "Bar vs Tick" alignment issues of continuous time axes
                        # while still preserving the "Gaps" because we resampled first.
                        if period == "Year":
                            final_vis_df['time_str'] = final_vis_df['datetime'].dt.strftime('%Y')
                        elif period == "Month":
                            final_vis_df['time_str'] = final_vis_df['datetime'].dt.strftime('%Y-%m')
                        elif period == "Week":
                            final_vis_df['time_str'] = final_vis_df['datetime'].dt.strftime('%Y-%m-%d')
                        elif period == "Day":
                            final_vis_df['time_str'] = final_vis_df['datetime'].dt.strftime('%Y-%m-%d')

                        fig_temp = px.bar(final_vis_df, x='time_str', y='count', color='user', barmode='group')
                        
                        # Explicitly set type to category regarding order
                        fig_temp.update_xaxes(type='category', title='Time')
                        
                        st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("Select users to visualize.")

    # --- Tab 2: Item Info ---
    with tab2:
        st.header("Movie Information")
        
        if not item_info_df.empty:
            movie_options = item_info_df['item'].unique()
            selected_movie_id = st.selectbox("Select Movie ID", sorted(movie_options))
            
            if selected_movie_id:
                info = item_info_df[item_info_df['item'] == selected_movie_id].iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Movie ID", selected_movie_id)
                    st.write(f"**Title:** {info.get('title', 'N/A')}")
                    st.write(f"**Genres:** {info.get('genre', 'N/A')}")
                with col2:
                    st.write(f"**Director:** {info.get('director', 'N/A')}")
                    st.write(f"**Writer:** {info.get('writer', 'N/A')}")
                    st.write(f"**Year:** {info.get('year', 'N/A')}")
        else:
            st.warning("No item information found.")

    # --- Tab 3: Detailed Statistics ---
    with tab3:
        st.header("Ranking Statistics")
        from utils import get_top_k_users, get_top_k_items, get_top_k_interactions
        
        col1, col2, col3 = st.columns(3)
        k = col1.number_input("Top/Bottom K", min_value=1, value=10)
        target = col2.selectbox("Target", ["Users", "Items", "Interactions"])
        order = col3.selectbox("Order", ["Top (Most)", "Bottom (Least)"])
        
        bottom_flag = (order == "Bottom (Least)")
        
        if target == "Users":
            res = get_top_k_users(filtered_df, k, bottom_flag)
            st.write(f"**{order} {k} Users by Interaction Count**")
            
            # Button to add users to selection using callback to avoid API exception
            def add_users_to_selection():
                current_selected = set(st.session_state.get('selected_users_manual', []))
                new_users = set(res['user'].tolist())
                updated_selection = list(current_selected.union(new_users))
                st.session_state['selected_users_manual'] = updated_selection
                
            st.button("Add these users to Sidebar Selection", on_click=add_users_to_selection)

            col_table, col_chart = st.columns([1, 2])
            with col_table:
                st.dataframe(res)
            with col_chart:
                fig_rank = px.bar(res, x='user', y='count', title=f"{order} {k} Users")
                # Ensure user axis is categorical
                fig_rank.update_xaxes(type='category')
                st.plotly_chart(fig_rank, use_container_width=True)
                
        elif target == "Items":
            res = get_top_k_items(filtered_df, k, bottom_flag)
            st.write(f"**{order} {k} Items by Interaction Count**")
            
            col_table, col_chart = st.columns([1, 2])
            with col_table:
                st.dataframe(res)
            with col_chart:
                fig_rank = px.bar(res, x='item', y='count', title=f"{order} {k} Items")
                fig_rank.update_xaxes(type='category')
                st.plotly_chart(fig_rank, use_container_width=True)
                
        else:
            res = get_top_k_interactions(filtered_df, k, bottom_flag)
            st.write(f"**{order} {k} User-Item Interactions**")
            st.dataframe(res)

if __name__ == "__main__":
    main()
