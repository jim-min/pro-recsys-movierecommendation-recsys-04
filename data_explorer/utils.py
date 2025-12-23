import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Optional


def plot_user_interactions(
    df: pd.DataFrame, user_ids: List[int], use_separate_axis: bool = False
) -> go.Figure:
    """Plots user interactions over time."""

    # Filter for selected users
    filtered_df = df[df["user"].isin(user_ids)].copy()

    if filtered_df.empty:
        return go.Figure()

    # Sort by time
    filtered_df = filtered_df.sort_values("datetime")

    # Assign colors/symbols based on user
    # Plotly express handles this automatically with 'color' and 'symbol' args

    if use_separate_axis:
        # Facet row by user for separate axes
        fig = px.scatter(
            filtered_df,
            x="datetime",
            y="item",
            color="user",
            symbol="user",
            facet_row="user",
            title="User Interactions Over Time (Separate Axes)",
            height=300 * len(user_ids),  # Dynamic height
        )
        # Allow independent x-axes if desired, but usually time comparison needs shared x
        # Requirement says: "time axis can be shared or separate".
        # Facet row separates the plotting area.
        fig.update_yaxes(matches=None)  # Independent y (items)
    else:
        # Shared axis
        fig = px.scatter(
            filtered_df,
            x="datetime",
            y="item",
            color="user",
            symbol="user",
            title="User Interactions Over Time (Shared Axis)",
        )

    fig.update_layout(xaxis_title="Time", yaxis_title="Item ID", hovermode="closest")

    return fig


def get_user_stats(df: pd.DataFrame, user_ids: List[int] = None) -> pd.DataFrame:
    """Calculates statistics for users."""

    if user_ids:
        target_df = df[df["user"].isin(user_ids)]
    else:
        target_df = df

    stats = (
        target_df.groupby("user")
        .agg(
            total_ratings=("item", "count"),
            first_interaction=("datetime", "min"),
            last_interaction=("datetime", "max"),
        )
        .reset_index()
    )

    return stats


def get_temporal_stats(df: pd.DataFrame, user_ids: List[int]) -> pd.DataFrame:
    """Calculates temporal stats (daily, weekly, etc.) for selected users."""
    target_df = df[df["user"].isin(user_ids)].copy()

    if target_df.empty:
        return pd.DataFrame()

    # Resample requires a datetime index
    target_df = target_df.set_index("datetime")

    return target_df.reset_index()  # Return with datetime column available


def get_top_k_users(
    df: pd.DataFrame, k: int = 10, bottom: bool = False
) -> pd.DataFrame:
    """Returns top or bottom k users by interaction count."""
    counts = df["user"].value_counts()
    if bottom:
        res = counts.nsmallest(k)
    else:
        res = counts.nlargest(k)
    return res.to_frame("count").reset_index().rename(columns={"index": "user"})


def get_top_k_items(
    df: pd.DataFrame, k: int = 10, bottom: bool = False
) -> pd.DataFrame:
    """Returns top or bottom k items by interaction count."""
    counts = df["item"].value_counts()
    if bottom:
        res = counts.nsmallest(k)
    else:
        res = counts.nlargest(k)
    return res.to_frame("count").reset_index().rename(columns={"index": "item"})


def get_top_k_interactions(
    df: pd.DataFrame, k: int = 10, bottom: bool = False
) -> pd.DataFrame:
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
    counts = df.groupby(["user", "item"]).size()
    if bottom:
        res = counts.nsmallest(k)
    else:
        res = counts.nlargest(k)
    return res.to_frame("count").reset_index()


def analyze_early_account_activity(df, time_window_hours=0.5, min_ratings=5):
    """
    Analyze early account activity patterns to identify initial bulk ratings.
    계정 생성 초기에 짧은 시간 내 몰아서 평가한 패턴을 분석합니다.

    Parameters:
    -----------
    df : DataFrame
        Ratings dataframe with columns: user, item, datetime
    time_window_hours : float
        Time window (in hours) from first rating to consider as "early activity"
        Default 0.5 (30분)
    min_ratings : int
        Minimum number of ratings in time window to flag as bulk rating

    Returns:
    --------
    DataFrame with early activity statistics per user
    """
    results = []

    for user in df["user"].unique():
        user_df = df[df["user"] == user].sort_values("datetime")

        if len(user_df) == 0:
            continue

        first_rating_time = user_df["datetime"].min()
        time_window_end = first_rating_time + pd.Timedelta(hours=time_window_hours)

        # Ratings within the time window
        early_ratings = user_df[user_df["datetime"] <= time_window_end]

        # Calculate statistics
        early_count = len(early_ratings)
        total_count = len(user_df)
        early_ratio = early_count / total_count if total_count > 0 else 0

        # Time between first and last early rating (in minutes for short windows)
        if early_count > 1:
            early_duration_minutes = (
                early_ratings["datetime"].max() - early_ratings["datetime"].min()
            ).total_seconds() / 60
            early_duration_hours = early_duration_minutes / 60
        else:
            early_duration_minutes = 0
            early_duration_hours = 0

        # Flag as bulk if meets criteria
        is_bulk = early_count >= min_ratings

        # Calculate rating velocity (ratings per minute)
        if early_duration_minutes > 0:
            ratings_per_minute = early_count / early_duration_minutes
        else:
            ratings_per_minute = early_count  # All at once

        results.append(
            {
                "user": user,
                "total_ratings": total_count,
                "early_ratings": early_count,
                "early_ratio": early_ratio,
                "early_duration_minutes": early_duration_minutes,
                "early_duration_hours": early_duration_hours,
                "is_bulk_rater": is_bulk,
                "first_rating_time": first_rating_time,
                "ratings_per_minute": ratings_per_minute,
                "avg_ratings_per_hour": (
                    early_count / time_window_hours if early_count > 0 else 0
                ),
            }
        )

    return pd.DataFrame(results).sort_values("early_ratings", ascending=False)


def plot_early_activity_distribution(early_stats_df):
    """Plot distribution of early activity patterns"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Distribution of Early Ratings Count",
            "Early Ratings Ratio",
            "Early Rating Duration (minutes)",
            "Bulk Raters vs Regular Users",
        ),
    )

    # 1. Distribution of early ratings count
    fig.add_trace(
        go.Histogram(
            x=early_stats_df["early_ratings"], nbinsx=50, name="Early Ratings"
        ),
        row=1,
        col=1,
    )

    # 2. Early ratio distribution
    fig.add_trace(
        go.Histogram(x=early_stats_df["early_ratio"], nbinsx=50, name="Early Ratio"),
        row=1,
        col=2,
    )

    # 3. Duration distribution (for users with >1 early rating) - in minutes
    duration_data = early_stats_df[early_stats_df["early_duration_minutes"] > 0][
        "early_duration_minutes"
    ]
    fig.add_trace(
        go.Histogram(x=duration_data, nbinsx=50, name="Duration (min)"), row=2, col=1
    )

    # 4. Bulk vs Regular pie chart
    bulk_counts = early_stats_df["is_bulk_rater"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=["Regular Users", "Bulk Raters"],
            values=[bulk_counts.get(False, 0), bulk_counts.get(True, 0)],
            name="User Type",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800, showlegend=False, title_text="Early Account Activity Analysis"
    )
    return fig


def get_bulk_rater_details(df, early_stats_df, user_id):
    """Get detailed information about a specific user's early rating behavior"""
    user_stats = early_stats_df[early_stats_df["user"] == user_id].iloc[0]
    user_ratings = df[df["user"] == user_id].sort_values("datetime")

    first_rating_time = user_stats["first_rating_time"]
    time_window_end = first_rating_time + pd.Timedelta(hours=24)
    early_ratings = user_ratings[user_ratings["datetime"] <= time_window_end]

    return {
        "stats": user_stats,
        "early_ratings": early_ratings,
        "all_ratings": user_ratings,
    }


def plot_user_early_activity_timeline(df, user_id, time_window_hours=0.5):
    """Plot timeline of a specific user's early activity"""
    user_df = df[df["user"] == user_id].sort_values("datetime")

    if len(user_df) == 0:
        return None

    first_time = user_df["datetime"].min()
    time_window_end = first_time + pd.Timedelta(hours=time_window_hours)

    # Separate early and later ratings
    early_df = user_df[user_df["datetime"] <= time_window_end]
    later_df = user_df[user_df["datetime"] > time_window_end]

    # Calculate time window in minutes for display
    time_window_minutes = time_window_hours * 60

    fig = go.Figure()

    # Plot early ratings
    fig.add_trace(
        go.Scatter(
            x=early_df["datetime"],
            y=early_df["item"],
            mode="markers",
            name=f"Early Ratings ({len(early_df)})",
            marker=dict(size=10, color="red", symbol="circle"),
            text=[
                f"Item: {item}<br>Time: {dt}"
                for item, dt in zip(early_df["item"], early_df["datetime"])
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )

    # Plot later ratings
    if len(later_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=later_df["datetime"],
                y=later_df["item"],
                mode="markers",
                name=f"Later Ratings ({len(later_df)})",
                marker=dict(size=8, color="blue", symbol="diamond"),
                text=[
                    f"Item: {item}<br>Time: {dt}"
                    for item, dt in zip(later_df["item"], later_df["datetime"])
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Add vertical line at window end
    window_label = (
        f"{int(time_window_minutes)}분"
        if time_window_hours < 1
        else f"{time_window_hours}시간"
    )
    fig.add_vline(
        x=time_window_end,
        line_dash="dash",
        line_color="green",
        annotation_text=f"{window_label} 구간",
    )

    # Calculate and display duration info
    if len(early_df) > 1:
        duration_minutes = (
            early_df["datetime"].max() - early_df["datetime"].min()
        ).total_seconds() / 60
        duration_text = (
            f"초기 {len(early_df)}개 평점을 {duration_minutes:.1f}분 동안 작성"
        )
    else:
        duration_text = f"초기 평점: {len(early_df)}개"

    fig.update_layout(
        title=f"User {user_id} Rating Timeline<br><sub>{duration_text}</sub>",
        xaxis_title="Time",
        yaxis_title="Item ID",
        height=500,
    )

    return fig


def compare_bulk_vs_regular_users(df, early_stats_df):
    """Compare characteristics of bulk raters vs regular users"""
    bulk_users = early_stats_df[early_stats_df["is_bulk_rater"] == True][
        "user"
    ].tolist()
    regular_users = early_stats_df[early_stats_df["is_bulk_rater"] == False][
        "user"
    ].tolist()

    bulk_df = df[df["user"].isin(bulk_users)]
    regular_df = df[df["user"].isin(regular_users)]

    comparison = {
        "Metric": [
            "Number of Users",
            "Avg Ratings per User",
            "Median Ratings per User",
            "Avg Unique Items",
            "Total Ratings",
        ],
        "Bulk Raters": [
            len(bulk_users),
            bulk_df.groupby("user").size().mean() if len(bulk_users) > 0 else 0,
            bulk_df.groupby("user").size().median() if len(bulk_users) > 0 else 0,
            (
                bulk_df.groupby("user")["item"].nunique().mean()
                if len(bulk_users) > 0
                else 0
            ),
            len(bulk_df),
        ],
        "Regular Users": [
            len(regular_users),
            regular_df.groupby("user").size().mean() if len(regular_users) > 0 else 0,
            regular_df.groupby("user").size().median() if len(regular_users) > 0 else 0,
            (
                regular_df.groupby("user")["item"].nunique().mean()
                if len(regular_users) > 0
                else 0
            ),
            len(regular_df),
        ],
    }

    return pd.DataFrame(comparison)
