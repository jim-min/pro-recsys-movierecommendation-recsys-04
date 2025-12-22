import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Optional


def plot_user_interactions(
    df: pd.DataFrame, 
    user_ids: List[int], 
    use_separate_axis: bool = False
) -> go.Figure:
    """Plots user interactions over time."""
    
    # Filter for selected users
    filtered_df = df[df['user'].isin(user_ids)].copy()
    
    if filtered_df.empty:
        return go.Figure()
        
    # Sort by time
    filtered_df = filtered_df.sort_values('datetime')
    
    # Assign colors/symbols based on user
    # Plotly express handles this automatically with 'color' and 'symbol' args
    
    if use_separate_axis:
        # Facet row by user for separate axes
        fig = px.scatter(
            filtered_df,
            x='datetime',
            y='item',
            color='user',
            symbol='user',
            facet_row='user',
            title='User Interactions Over Time (Separate Axes)',
            height=300 * len(user_ids)  # Dynamic height
        )
        # Allow independent x-axes if desired, but usually time comparison needs shared x
        # Requirement says: "time axis can be shared or separate". 
        # Facet row separates the plotting area. 
        fig.update_yaxes(matches=None) # Independent y (items)
    else:
        # Shared axis
        fig = px.scatter(
            filtered_df,
            x='datetime',
            y='item',
            color='user',
            symbol='user',
            title='User Interactions Over Time (Shared Axis)'
        )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Item ID",
        hovermode="closest"
    )
        
    return fig

def get_user_stats(df: pd.DataFrame, user_ids: List[int] = None) -> pd.DataFrame:
    """Calculates statistics for users."""
    
    if user_ids:
        target_df = df[df['user'].isin(user_ids)]
    else:
        target_df = df
        
    stats = target_df.groupby('user').agg(
        total_ratings=('item', 'count'),
        first_interaction=('datetime', 'min'),
        last_interaction=('datetime', 'max')
    ).reset_index()
    
    return stats

def get_temporal_stats(df: pd.DataFrame, user_ids: List[int]) -> pd.DataFrame:
    """Calculates temporal stats (daily, weekly, etc.) for selected users."""
    target_df = df[df['user'].isin(user_ids)].copy()
    
    if target_df.empty:
        return pd.DataFrame()

    # Resample requires a datetime index
    target_df = target_df.set_index('datetime')
    
    return target_df.reset_index() # Return with datetime column available

def get_top_k_users(df: pd.DataFrame, k: int = 10, bottom: bool = False) -> pd.DataFrame:
    """Returns top or bottom k users by interaction count."""
    counts = df['user'].value_counts()
    if bottom:
        res = counts.nsmallest(k)
    else:
        res = counts.nlargest(k)
    return res.to_frame('count').reset_index().rename(columns={'index': 'user'})

def get_top_k_items(df: pd.DataFrame, k: int = 10, bottom: bool = False) -> pd.DataFrame:
    """Returns top or bottom k items by interaction count."""
    counts = df['item'].value_counts()
    if bottom:
        res = counts.nsmallest(k)
    else:
        res = counts.nlargest(k)
    return res.to_frame('count').reset_index().rename(columns={'index': 'item'})

def get_top_k_interactions(df: pd.DataFrame, k: int = 10, bottom: bool = False) -> pd.DataFrame:
    """Returns top or bottom k user-item pairs (assuming just listing raw rows isn't helpful, 
    but since pairs are unique in implicit feedback usually, this might just mean top users AND top items.
    However, requirement says 'user interaction counts most user and item'.
    I'll interpret this as returning the top users and top items separately provided in the function above.
    Wait, requirement 18 says: 'user interaction count most user and item'. 
    If this means the pair that has most interaction, it implies re-occuring pairs.
    If implicit feedback (1 interaction per user-item), then all are 1.
    Let's assume implicit for now. If all are 1, this stat is meaningless for pairs.
    But let's check if there are duplicates.
    """
    # Check for duplicates
    counts = df.groupby(['user', 'item']).size()
    if bottom:
        res = counts.nsmallest(k)
    else:
        res = counts.nlargest(k)
    return res.to_frame('count').reset_index()
