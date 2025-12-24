import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Optional, Dict
import time
from plotly.subplots import make_subplots
from collections import Counter
import itertools


# ==================== Existing Functions ====================


def plot_user_interactions(
    df: pd.DataFrame, user_ids: List[int], use_separate_axis: bool = False
) -> go.Figure:
    """Plots user interactions over time."""
    filtered_df = df[df["user"].isin(user_ids)].copy()

    if filtered_df.empty:
        return go.Figure()

    filtered_df = filtered_df.sort_values("datetime")

    if use_separate_axis:
        fig = px.scatter(
            filtered_df,
            x="datetime",
            y="item",
            color="user",
            symbol="user",
            facet_row="user",
            title="User Interactions Over Time (Separate Axes)",
            height=300 * len(user_ids),
        )
        fig.update_yaxes(matches=None)
    else:
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

    target_df = target_df.set_index("datetime")
    return target_df.reset_index()


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
    """Returns top or bottom k user-item pairs."""
    counts = df.groupby(["user", "item"]).size()
    if bottom:
        res = counts.nsmallest(k)
    else:
        res = counts.nlargest(k)
    return res.to_frame("count").reset_index()


def analyze_early_account_activity(
    df, time_window_hours=0.5, min_ratings=5, show_progress=False
):
    """Analyze early account activity patterns to identify initial bulk ratings."""
    results = []
    all_users = df["user"].unique()
    total_users = len(all_users)

    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

    for idx, user in enumerate(all_users):
        user_df = df[df["user"] == user].sort_values("datetime")

        if len(user_df) == 0:
            continue

        first_rating_time = user_df["datetime"].min()
        time_window_end = first_rating_time + pd.Timedelta(hours=time_window_hours)

        early_ratings = user_df[user_df["datetime"] <= time_window_end]

        early_count = len(early_ratings)
        total_count = len(user_df)
        early_ratio = early_count / total_count if total_count > 0 else 0

        if early_count > 1:
            early_duration_minutes = (
                early_ratings["datetime"].max() - early_ratings["datetime"].min()
            ).total_seconds() / 60
            early_duration_hours = early_duration_minutes / 60
        else:
            early_duration_minutes = 0
            early_duration_hours = 0

        is_bulk = early_count >= min_ratings

        if early_duration_minutes > 0:
            ratings_per_minute = early_count / early_duration_minutes
        else:
            ratings_per_minute = early_count

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

        if show_progress:
            update_interval = min(100, max(1, total_users // 100))
            if (idx + 1) % update_interval == 0 or idx == total_users - 1:
                progress = (idx + 1) / total_users
                progress_bar.progress(progress)

                elapsed_time = time.time() - start_time
                users_per_sec = (idx + 1) / elapsed_time if elapsed_time > 0 else 0
                remaining_users = total_users - (idx + 1)
                eta_seconds = (
                    remaining_users / users_per_sec if users_per_sec > 0 else 0
                )

                if eta_seconds < 60:
                    eta_str = f"{eta_seconds:.0f}s"
                elif eta_seconds < 3600:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds/3600:.1f}h"

                status_text.text(
                    f"Processing: {idx + 1:,}/{total_users:,} users ({progress*100:.1f}%) | "
                    f"Speed: {users_per_sec:.1f} users/s | ETA: {eta_str}"
                )

    if show_progress:
        progress_bar.progress(1.0)
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.1f}s"
        else:
            time_str = f"{elapsed_time/60:.1f}m"
        status_text.text(f"✅ Complete! Analyzed {total_users:,} users in {time_str}")

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
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "domain"}]],
    )

    fig.add_trace(
        go.Histogram(
            x=early_stats_df["early_ratings"], nbinsx=50, name="Early Ratings"
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(x=early_stats_df["early_ratio"], nbinsx=50, name="Early Ratio"),
        row=1,
        col=2,
    )

    duration_data = early_stats_df[early_stats_df["early_duration_minutes"] > 0][
        "early_duration_minutes"
    ]
    fig.add_trace(
        go.Histogram(x=duration_data, nbinsx=50, name="Duration (min)"), row=2, col=1
    )

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


# ==================== Distribution & Statistics Functions ====================


def dist_summary(s: pd.Series, name: str) -> pd.DataFrame:
    """Distribution summary with quantiles"""
    q = s.quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    out = pd.DataFrame(
        {
            "name": [name],
            "count": [s.shape[0]],
            "min": [int(s.min())],
            "p50": [int(q.loc[0.5])],
            "p75": [int(q.loc[0.75])],
            "p90": [int(q.loc[0.9])],
            "p95": [int(q.loc[0.95])],
            "p99": [int(q.loc[0.99])],
            "max": [int(s.max())],
            "mean": [f"{s.mean():.1f}"],
            "std": [f"{s.std():.1f}"],
        }
    )
    return out


def heavy_share(counts: pd.Series, top_percent: float):
    """Calculate share of top X% entities"""
    sorted_counts = counts.sort_values(ascending=False)
    k = max(1, int(len(sorted_counts) * (top_percent / 100.0)))
    share = sorted_counts.iloc[:k].sum() / sorted_counts.sum()
    return {
        "top_percent": top_percent,
        "k": k,
        "share": float(share),
        "top_k_sum": int(sorted_counts.iloc[:k].sum()),
        "total_sum": int(sorted_counts.sum()),
    }


def cold_user_ratio(user_counts: pd.Series, k: int) -> float:
    """Calculate ratio of users with <= k interactions"""
    return (user_counts <= k).mean()


def calculate_concentration_metrics(df):
    """Calculate concentration metrics (Gini-like) for users and items"""
    user_counts = df.groupby("user")["item"].count()
    item_counts = df.groupby("item")["user"].count()

    metrics = {}

    for pct in [0.1, 0.5, 1, 5, 10, 20]:
        user_share = heavy_share(user_counts, pct)
        item_share = heavy_share(item_counts, pct)

        metrics[f"top_{pct}_pct"] = {
            "user_share": user_share["share"],
            "item_share": item_share["share"],
        }

    user_stats = dist_summary(user_counts, "users")
    item_stats = dist_summary(item_counts, "items")

    metrics["tail_ratios"] = {
        "user_p99_p50": float(user_stats["p99"].iloc[0])
        / float(user_stats["p50"].iloc[0]),
        "item_p99_p50": float(item_stats["p99"].iloc[0])
        / float(item_stats["p50"].iloc[0]),
    }

    metrics["cold_users"] = {
        f"le_{k}": cold_user_ratio(user_counts, k) for k in [3, 10, 50, 100]
    }

    return metrics


def plot_concentration_analysis(df):
    """Plot concentration analysis for users and items"""
    user_counts = df.groupby("user")["item"].count()
    item_counts = df.groupby("item")["user"].count()

    percentiles = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]
    user_shares = []
    item_shares = []

    for pct in percentiles:
        user_shares.append(heavy_share(user_counts, pct)["share"] * 100)
        item_shares.append(heavy_share(item_counts, pct)["share"] * 100)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=percentiles,
            y=user_shares,
            mode="lines+markers",
            name="Users",
            line=dict(color="#667eea", width=3),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=percentiles,
            y=item_shares,
            mode="lines+markers",
            name="Items",
            line=dict(color="#f093fb", width=3),
        )
    )

    fig.update_layout(
        title="Concentration Curve: Top X% Entities' Share of Total Interactions",
        xaxis_title="Top X% of Entities",
        yaxis_title="Share of Total Interactions (%)",
        hovermode="x unified",
        height=500,
    )

    return fig


def plot_distribution_summary(df):
    """Plot comprehensive distribution summary"""
    user_counts = df.groupby("user")["item"].count()
    item_counts = df.groupby("item")["user"].count()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "User Interaction Distribution",
            "Item Popularity Distribution",
            "User Distribution (Log Scale)",
            "Item Distribution (Log Scale)",
        ),
    )

    fig.add_trace(go.Histogram(x=user_counts, nbinsx=50, name="Users"), row=1, col=1)

    fig.add_trace(go.Histogram(x=item_counts, nbinsx=50, name="Items"), row=1, col=2)

    fig.add_trace(
        go.Histogram(x=user_counts, nbinsx=50, name="Users (log)"), row=2, col=1
    )
    fig.update_yaxes(type="log", row=2, col=1)

    fig.add_trace(
        go.Histogram(x=item_counts, nbinsx=50, name="Items (log)"), row=2, col=2
    )
    fig.update_yaxes(type="log", row=2, col=2)

    fig.update_layout(height=800, showlegend=False)

    return fig


# ==================== Genre Analysis Functions ====================


def safe_len(x):
    """Safely get length of list"""
    if isinstance(x, list):
        return len(x)
    return 0


def analyze_genre_distribution(df, item_info_df, top_n=None):
    """Analyze genre distribution in the dataset"""
    if "genre" not in item_info_df.columns:
        return None

    merged = df.merge(item_info_df[["item", "genre"]], on="item", how="left")

    if top_n:
        popular_items = df["item"].value_counts().head(top_n).index
        merged = merged[merged["item"].isin(popular_items)]

    all_genres = []
    for genres_str in merged["genre"].dropna():
        if isinstance(genres_str, str):
            if "|" in genres_str:
                all_genres.extend(genres_str.split("|"))
            elif "," in genres_str:
                all_genres.extend(genres_str.split(","))
            else:
                all_genres.append(genres_str)

    genre_counts = Counter(all_genres)

    genre_df = pd.DataFrame(genre_counts.most_common(), columns=["Genre", "Count"])

    return genre_df


def plot_genre_distribution(genre_df, title="Genre Distribution"):
    """Plot genre distribution as bar chart"""
    if genre_df is None or genre_df.empty:
        return None

    fig = px.bar(
        genre_df.head(20),
        x="Genre",
        y="Count",
        title=title,
        labels={"Count": "Number of Ratings", "Genre": "Genre"},
    )
    fig.update_xaxes(tickangle=45)
    return fig


def analyze_genre_combinations(item_info_df, df=None):
    """
    Analyze genre combinations
    Returns DataFrame with combo, item_count, and optionally review_count
    """
    if "genre" not in item_info_df.columns:
        return None

    # Create genre combo column
    combos = []
    for idx, row in item_info_df.iterrows():
        genre_str = row["genre"]
        if isinstance(genre_str, str) and genre_str:
            if "|" in genre_str:
                genres = sorted(genre_str.split("|"))
            elif "," in genre_str:
                genres = sorted(genre_str.split(","))
            else:
                genres = [genre_str]
            combo = "|".join(genres)
        else:
            combo = "Unknown"
        combos.append(combo)

    item_info_df = item_info_df.copy()
    item_info_df["genre_combo"] = combos

    # Count items per combo
    combo_counts = item_info_df["genre_combo"].value_counts().reset_index()
    combo_counts.columns = ["genre_combo", "item_count"]

    # Add review counts if ratings provided
    if df is not None:
        merged = df.merge(item_info_df[["item", "genre_combo"]], on="item", how="left")
        review_stats = (
            merged.groupby("genre_combo")
            .agg(review_count=("user", "count"), unique_users=("user", "nunique"))
            .reset_index()
        )

        combo_counts = combo_counts.merge(review_stats, on="genre_combo", how="left")
        combo_counts["mean_review"] = (
            combo_counts["review_count"] / combo_counts["item_count"]
        )

    return combo_counts.sort_values("item_count", ascending=False)


def find_rare_but_popular_combos(combo_df, top_k=20):
    """
    Find rare genre combinations that are popular
    Uses rarity score weighted by popularity
    """
    if combo_df is None or "mean_review" not in combo_df.columns:
        return None

    # Calculate rarity score (inverse of item count)
    combo_df = combo_df.copy()
    combo_df["rarity_score"] = 1 / (combo_df["item_count"] + 1)

    # Weighted score: high mean review + rare
    combo_df["weighted_score"] = (
        combo_df["mean_review"] * combo_df["rarity_score"] * 1000
    )

    # Filter out Unknown and single-item combos
    combo_df = combo_df[
        (combo_df["genre_combo"] != "Unknown")
        & (combo_df["item_count"] < 100)  # Rare combos only
        & (combo_df["mean_review"] > combo_df["mean_review"].median())  # Above average
    ]

    return combo_df.nlargest(top_k, "weighted_score")


def plot_rare_popular_combos(rare_combos):
    """Plot rare but popular genre combinations"""
    if rare_combos is None or rare_combos.empty:
        return None

    fig = px.scatter(
        rare_combos,
        x="item_count",
        y="mean_review",
        size="review_count",
        hover_data=["genre_combo"],
        title="Rare but Popular Genre Combinations",
        labels={
            "item_count": "Number of Items (Rarity)",
            "mean_review": "Average Reviews per Item",
            "review_count": "Total Reviews",
        },
    )

    return fig


# ==================== User Segmentation Functions ====================


def segment_users(df, early_stats_df=None):
    """
    Segment users into Heavy/Medium/Light categories
    """
    user_counts = df.groupby("user")["item"].count()

    # Define thresholds (adjustable)
    q25 = user_counts.quantile(0.25)
    q75 = user_counts.quantile(0.75)

    segments = []
    for user, count in user_counts.items():
        if count >= q75:
            segment = "Heavy"
        elif count >= q25:
            segment = "Medium"
        else:
            segment = "Light"

        segments.append({"user": user, "rating_count": count, "segment": segment})

    segment_df = pd.DataFrame(segments)

    # Add bulk rater info if available
    if early_stats_df is not None:
        segment_df = segment_df.merge(
            early_stats_df[["user", "is_bulk_rater", "early_ratio"]],
            on="user",
            how="left",
        )

    return segment_df


def analyze_user_segments(segment_df, df):
    """Analyze characteristics of each user segment"""
    summary = (
        segment_df.groupby("segment")
        .agg(
            user_count=("user", "count"),
            avg_ratings=("rating_count", "mean"),
            median_ratings=("rating_count", "median"),
            min_ratings=("rating_count", "min"),
            max_ratings=("rating_count", "max"),
        )
        .reset_index()
    )

    # Add share of total ratings
    total_ratings = df.shape[0]
    for idx, row in summary.iterrows():
        segment_users = segment_df[segment_df["segment"] == row["segment"]][
            "user"
        ].tolist()
        segment_ratings = df[df["user"].isin(segment_users)].shape[0]
        summary.loc[idx, "rating_share"] = f"{segment_ratings/total_ratings*100:.1f}%"

    return summary


def plot_user_segmentation(segment_df):
    """Visualize user segmentation"""
    segment_counts = segment_df["segment"].value_counts()

    fig = go.Figure(
        data=[
            go.Pie(labels=segment_counts.index, values=segment_counts.values, hole=0.3)
        ]
    )

    fig.update_layout(title="User Segmentation Distribution", height=400)

    return fig


# ==================== Item Cold Start Functions ====================


def find_cold_start_items(df, item_info_df):
    """Find items with metadata but no ratings"""
    items_with_ratings = set(df["item"].unique())
    items_with_metadata = set(item_info_df["item"].unique())

    cold_start_items = items_with_metadata - items_with_ratings

    cold_start_df = item_info_df[item_info_df["item"].isin(cold_start_items)].copy()

    return cold_start_df


def analyze_cold_start_items(cold_start_df):
    """Analyze characteristics of cold start items"""
    if cold_start_df.empty:
        return None

    summary = {
        "total_count": len(cold_start_df),
    }

    # Genre distribution if available
    if "genre" in cold_start_df.columns:
        all_genres = []
        for genre_str in cold_start_df["genre"].dropna():
            if isinstance(genre_str, str):
                if "|" in genre_str:
                    all_genres.extend(genre_str.split("|"))
                elif "," in genre_str:
                    all_genres.extend(genre_str.split(","))
                else:
                    all_genres.append(genre_str)

        genre_counts = Counter(all_genres)
        summary["top_genres"] = dict(genre_counts.most_common(5))

    # Year distribution if available
    if "year" in cold_start_df.columns:
        year_stats = cold_start_df["year"].describe()
        summary["year_range"] = f"{int(year_stats['min'])} - {int(year_stats['max'])}"
        summary["avg_year"] = int(year_stats["mean"])

    return summary


# ==================== Temporal Analysis Functions ====================


def analyze_temporal_patterns(df):
    """Analyze temporal patterns in ratings"""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek  # 0=Monday
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    patterns = {
        "hourly": df["hour"].value_counts().sort_index(),
        "daily": df["day_of_week"].value_counts().sort_index(),
        "monthly": df["month"].value_counts().sort_index(),
        "yearly": df["year"].value_counts().sort_index(),
    }

    return patterns


def plot_temporal_heatmap(df):
    """Plot hour x day of week heatmap"""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek

    heatmap_data = df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
    heatmap_pivot = heatmap_data.pivot(
        index="day_of_week", columns="hour", values="count"
    ).fillna(0)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heatmap_pivot.index = [day_labels[i] for i in heatmap_pivot.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale="Blues",
        )
    )

    fig.update_layout(
        title="Activity Heatmap: Hour of Day vs Day of Week",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        height=400,
    )

    return fig


def plot_temporal_trends(patterns):
    """Plot temporal trends"""
    fig = make_subplots(
        rows=2, cols=2, subplot_titles=("Hourly", "Daily", "Monthly", "Yearly")
    )

    # Hourly
    fig.add_trace(
        go.Bar(x=patterns["hourly"].index, y=patterns["hourly"].values, name="Hour"),
        row=1,
        col=1,
    )

    # Daily
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig.add_trace(
        go.Bar(
            x=[day_labels[i] for i in patterns["daily"].index],
            y=patterns["daily"].values,
            name="Day",
        ),
        row=1,
        col=2,
    )

    # Monthly
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    fig.add_trace(
        go.Bar(
            x=[month_labels[i - 1] for i in patterns["monthly"].index],
            y=patterns["monthly"].values,
            name="Month",
        ),
        row=2,
        col=1,
    )

    # Yearly
    fig.add_trace(
        go.Scatter(
            x=patterns["yearly"].index,
            y=patterns["yearly"].values,
            mode="lines+markers",
            name="Year",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(height=600, showlegend=False)

    return fig


# ==================== Data Quality Check Functions ====================
def check_data_quality(
    df, df_name="DataFrame", show_progress=False, progress_bar=None, status_text=None
):
    """
    Comprehensive data quality check with progress tracking
    Returns dictionary with all quality metrics
    """
    quality_report = {
        "df_name": df_name,
        "shape": df.shape,
        "missing_values": {},
        "duplicates": {},
        "data_types": {},
        "outliers": {},
        "unique_counts": {},
    }

    total_steps = 5
    current_step = 0

    # Step 1: Missing values
    if show_progress and status_text:
        status_text.text(
            f"📊 [{df_name}] Step 1/{total_steps}: Checking missing values..."
        )

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    quality_report["missing_values"] = pd.DataFrame(
        {"Missing Count": missing, "Missing %": missing_pct}
    )

    current_step += 1
    if show_progress and progress_bar:
        progress_bar.progress(current_step / total_steps)

    # Step 2: Duplicates
    if show_progress and status_text:
        status_text.text(f"📊 [{df_name}] Step 2/{total_steps}: Checking duplicates...")

    duplicate_rows = df.duplicated().sum()
    quality_report["duplicates"] = {
        "total_duplicates": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / len(df) * 100, 2),
    }

    current_step += 1
    if show_progress and progress_bar:
        progress_bar.progress(current_step / total_steps)

    # Step 3: Data types
    if show_progress and status_text:
        status_text.text(
            f"📊 [{df_name}] Step 3/{total_steps}: Analyzing data types..."
        )

    quality_report["data_types"] = df.dtypes.to_dict()

    current_step += 1
    if show_progress and progress_bar:
        progress_bar.progress(current_step / total_steps)

    # Step 4: Unique value counts
    if show_progress and status_text:
        status_text.text(
            f"📊 [{df_name}] Step 4/{total_steps}: Counting unique values..."
        )

    for col in df.columns:
        quality_report["unique_counts"][col] = df[col].nunique()

    current_step += 1
    if show_progress and progress_bar:
        progress_bar.progress(current_step / total_steps)

    # Step 5: Numeric outliers (IQR method)
    if show_progress and status_text:
        status_text.text(f"📊 [{df_name}] Step 5/{total_steps}: Detecting outliers...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        quality_report["outliers"][col] = {
            "count": len(outliers),
            "pct": round(len(outliers) / len(df) * 100, 2),
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    current_step += 1
    if show_progress and progress_bar:
        progress_bar.progress(1.0)

    if show_progress and status_text:
        status_text.text(f"✅ [{df_name}] Quality check complete!")

    return quality_report


def plot_data_quality_summary(quality_report):
    """Visualize data quality report"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Missing Values (%)",
            "Data Types Distribution",
            "Duplicate Records",
            "Outliers by Column",
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "indicator"}, {"type": "bar"}],
        ],
    )

    # 1. Missing values bar chart
    missing_df = quality_report["missing_values"]
    missing_df = missing_df[missing_df["Missing %"] > 0]

    if not missing_df.empty:
        fig.add_trace(
            go.Bar(
                x=missing_df.index,
                y=missing_df["Missing %"],
                name="Missing %",
                marker_color="indianred",
            ),
            row=1,
            col=1,
        )

    # 2. Data types pie chart
    dtypes = pd.Series(quality_report["data_types"]).astype(str).value_counts()
    fig.add_trace(
        go.Pie(labels=dtypes.index, values=dtypes.values, name="Data Types"),
        row=1,
        col=2,
    )

    # 3. Duplicates indicator
    dup_pct = quality_report["duplicates"]["duplicate_pct"]
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=dup_pct,
            title={"text": "Duplicate %"},
            delta={"reference": 0},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkred" if dup_pct > 5 else "darkgreen"},
                "steps": [
                    {"range": [0, 1], "color": "lightgreen"},
                    {"range": [1, 5], "color": "yellow"},
                    {"range": [5, 100], "color": "lightcoral"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 5,
                },
            },
        ),
        row=2,
        col=1,
    )

    # 4. Outliers bar chart
    outliers_data = []
    for col, info in quality_report["outliers"].items():
        if info["count"] > 0:
            outliers_data.append({"Column": col, "Outlier %": info["pct"]})

    if outliers_data:
        outliers_df = pd.DataFrame(outliers_data)
        fig.add_trace(
            go.Bar(
                x=outliers_df["Column"],
                y=outliers_df["Outlier %"],
                name="Outliers %",
                marker_color="orange",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        height=800,
        showlegend=False,
        title_text=f"Data Quality: {quality_report['df_name']}",
    )

    return fig


def check_user_item_validity(df, item_info_df=None):
    """
    Check validity of user and item IDs
    """
    validity_report = {}

    # 1. User ID checks
    validity_report["user"] = {
        "total_unique": df["user"].nunique(),
        "min_id": int(df["user"].min()),
        "max_id": int(df["user"].max()),
        "negative_ids": (df["user"] < 0).sum(),
        "zero_ids": (df["user"] == 0).sum(),
    }

    # 2. Item ID checks
    validity_report["item"] = {
        "total_unique": df["item"].nunique(),
        "min_id": int(df["item"].min()),
        "max_id": int(df["item"].max()),
        "negative_ids": (df["item"] < 0).sum(),
        "zero_ids": (df["item"] == 0).sum(),
    }

    # 3. User-Item pair duplicates
    user_item_pairs = df.groupby(["user", "item"]).size()
    duplicate_pairs = user_item_pairs[user_item_pairs > 1]

    validity_report["user_item_pairs"] = {
        "total_unique_pairs": len(user_item_pairs),
        "duplicate_pairs": len(duplicate_pairs),
        "max_duplicates": int(user_item_pairs.max()) if len(user_item_pairs) > 0 else 0,
    }

    # 4. Items not in metadata (if item_info provided)
    if item_info_df is not None:
        items_in_ratings = set(df["item"].unique())
        items_in_metadata = set(item_info_df["item"].unique())

        missing_metadata = items_in_ratings - items_in_metadata
        validity_report["metadata_coverage"] = {
            "items_with_metadata": len(items_in_metadata),
            "items_without_metadata": len(missing_metadata),
            "coverage_pct": (
                round(len(items_in_metadata) / len(items_in_ratings) * 100, 2)
                if len(items_in_ratings) > 0
                else 0
            ),
        }

    return validity_report


def check_temporal_validity(df):
    """Check temporal data validity"""
    temporal_report = {}

    # 1. Datetime range
    temporal_report["range"] = {
        "min_date": df["datetime"].min(),
        "max_date": df["datetime"].max(),
        "date_span_days": (df["datetime"].max() - df["datetime"].min()).days,
    }

    # 2. Future dates (if any)
    current_time = pd.Timestamp.now()
    future_dates = df[df["datetime"] > current_time]
    temporal_report["future_dates"] = {
        "count": len(future_dates),
        "pct": round(len(future_dates) / len(df) * 100, 2),
    }

    # 3. Null timestamps
    null_timestamps = df["datetime"].isnull().sum()
    temporal_report["null_timestamps"] = {
        "count": null_timestamps,
        "pct": round(null_timestamps / len(df) * 100, 2),
    }

    # 4. Temporal distribution
    df_temp = df.copy()
    df_temp["year"] = df_temp["datetime"].dt.year
    df_temp["month"] = df_temp["datetime"].dt.month
    df_temp["hour"] = df_temp["datetime"].dt.hour

    temporal_report["distribution"] = {
        "years": df_temp["year"].value_counts().to_dict(),
        "busiest_month": int(df_temp["month"].mode()[0]) if len(df_temp) > 0 else None,
        "busiest_hour": int(df_temp["hour"].mode()[0]) if len(df_temp) > 0 else None,
    }

    return temporal_report


def check_metadata_quality(item_info_df, show_progress=False):
    """
    Check quality of metadata (directors, genres, titles, writers, years)
    """
    metadata_reports = {}

    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    # Available metadata columns
    metadata_types = {
        "title": "Titles",
        "genre": "Genres",
        "director": "Directors",
        "writer": "Writers",
        "year": "Years",
    }

    total_types = len(
        [col for col in metadata_types.keys() if col in item_info_df.columns]
    )
    current = 0

    for col, name in metadata_types.items():
        if col not in item_info_df.columns:
            continue

        if show_progress:
            status_text.text(f"📋 Checking {name}... ({current+1}/{total_types})")

        report = {
            "total_items": len(item_info_df),
            "missing_count": item_info_df[col].isnull().sum(),
            "missing_pct": round(
                item_info_df[col].isnull().sum() / len(item_info_df) * 100, 2
            ),
            "unique_values": item_info_df[col].nunique(),
        }

        # Type-specific checks
        if col == "year":
            # Year-specific validation
            valid_years = item_info_df[item_info_df[col].notna()][col]
            if len(valid_years) > 0:
                report["min_year"] = (
                    int(valid_years.min()) if not pd.isna(valid_years.min()) else None
                )
                report["max_year"] = (
                    int(valid_years.max()) if not pd.isna(valid_years.max()) else None
                )
                report["future_years"] = (valid_years > pd.Timestamp.now().year).sum()
                report["ancient_years"] = (valid_years < 1800).sum()

        elif col in ["genre", "director", "writer"]:
            # Count items with multiple values (if pipe-separated)
            non_null = item_info_df[item_info_df[col].notna()][col]
            if len(non_null) > 0:
                # Check if values are pipe-separated
                multi_value = non_null.str.contains("|", na=False).sum()
                report["items_with_multiple"] = multi_value
                report["multi_value_pct"] = round(multi_value / len(non_null) * 100, 2)

        metadata_reports[name] = report

        current += 1
        if show_progress:
            progress_bar.progress(current / total_types)

    if show_progress:
        status_text.text("✅ Metadata quality check complete!")
        progress_bar.progress(1.0)

    return metadata_reports


def generate_data_quality_summary_text(
    quality_report, validity_report, temporal_report
):
    """Generate human-readable summary"""
    summary = []

    # Overall
    summary.append(f"## 📊 Data Quality Summary: {quality_report['df_name']}")
    summary.append(
        f"**Shape**: {quality_report['shape'][0]:,} rows × {quality_report['shape'][1]} columns"
    )
    summary.append("")

    # Missing values
    missing_cols = quality_report["missing_values"][
        quality_report["missing_values"]["Missing %"] > 0
    ]
    if len(missing_cols) > 0:
        summary.append("### ⚠️ Missing Values")
        for col, row in missing_cols.iterrows():
            summary.append(
                f"- **{col}**: {row['Missing Count']:,} ({row['Missing %']:.2f}%)"
            )
    else:
        summary.append("### ✅ Missing Values: None")
    summary.append("")

    # Duplicates
    dup_count = quality_report["duplicates"]["total_duplicates"]
    dup_pct = quality_report["duplicates"]["duplicate_pct"]
    if dup_count > 0:
        summary.append(f"### ⚠️ Duplicate Rows: {dup_count:,} ({dup_pct:.2f}%)")
    else:
        summary.append("### ✅ Duplicate Rows: None")
    summary.append("")

    # Outliers
    outlier_cols = {
        k: v for k, v in quality_report["outliers"].items() if v["count"] > 0
    }
    if outlier_cols:
        summary.append("### 📌 Outliers (IQR method)")
        for col, info in outlier_cols.items():
            summary.append(
                f"- **{col}**: {info['count']:,} ({info['pct']:.2f}%) outside [{info['lower_bound']:.1f}, {info['upper_bound']:.1f}]"
            )
    else:
        summary.append("### ✅ Outliers: None detected")
    summary.append("")

    # User/Item validity
    summary.append("### 🔍 ID Validity")
    summary.append(
        f"- **Users**: {validity_report['user']['total_unique']:,} unique (ID range: {validity_report['user']['min_id']}-{validity_report['user']['max_id']})"
    )
    summary.append(
        f"- **Items**: {validity_report['item']['total_unique']:,} unique (ID range: {validity_report['item']['min_id']}-{validity_report['item']['max_id']})"
    )

    if (
        validity_report["user"]["negative_ids"] > 0
        or validity_report["user"]["zero_ids"] > 0
    ):
        summary.append(
            f"  - ⚠️ Users with ID ≤ 0: {validity_report['user']['negative_ids'] + validity_report['user']['zero_ids']}"
        )

    if (
        validity_report["item"]["negative_ids"] > 0
        or validity_report["item"]["zero_ids"] > 0
    ):
        summary.append(
            f"  - ⚠️ Items with ID ≤ 0: {validity_report['item']['negative_ids'] + validity_report['item']['zero_ids']}"
        )
    summary.append("")

    # User-Item pairs
    if validity_report["user_item_pairs"]["duplicate_pairs"] > 0:
        summary.append(
            f"### ⚠️ Duplicate User-Item Pairs: {validity_report['user_item_pairs']['duplicate_pairs']:,}"
        )
    else:
        summary.append("### ✅ User-Item Pairs: All unique")
    summary.append("")

    # Temporal
    summary.append("### 📅 Temporal Data")
    summary.append(
        f"- **Date Range**: {temporal_report['range']['min_date'].date()} to {temporal_report['range']['max_date'].date()} ({temporal_report['range']['date_span_days']} days)"
    )

    if temporal_report["future_dates"]["count"] > 0:
        summary.append(
            f"  - ⚠️ Future dates: {temporal_report['future_dates']['count']:,} ({temporal_report['future_dates']['pct']:.2f}%)"
        )

    if temporal_report["null_timestamps"]["count"] > 0:
        summary.append(
            f"  - ⚠️ Null timestamps: {temporal_report['null_timestamps']['count']:,} ({temporal_report['null_timestamps']['pct']:.2f}%)"
        )

    return "\n".join(summary)


def generate_metadata_summary_text(metadata_reports):
    """Generate human-readable metadata summary"""
    summary = []

    summary.append("## 📋 Metadata Quality Summary")
    summary.append("")

    for name, report in metadata_reports.items():
        summary.append(f"### {name}")
        summary.append(f"- **Total Items**: {report['total_items']:,}")

        if report["missing_count"] > 0:
            summary.append(
                f"- ⚠️ **Missing**: {report['missing_count']:,} ({report['missing_pct']:.2f}%)"
            )
        else:
            summary.append(f"- ✅ **Missing**: None")

        summary.append(f"- **Unique Values**: {report['unique_values']:,}")

        # Type-specific info
        if "min_year" in report:
            summary.append(
                f"- **Year Range**: {report['min_year']} - {report['max_year']}"
            )
            if report["future_years"] > 0:
                summary.append(f"  - ⚠️ Future years: {report['future_years']}")
            if report["ancient_years"] > 0:
                summary.append(
                    f"  - ⚠️ Ancient years (<1800): {report['ancient_years']}"
                )

        if "items_with_multiple" in report:
            summary.append(
                f"- **Multiple Values**: {report['items_with_multiple']:,} items ({report['multi_value_pct']:.2f}%)"
            )

        summary.append("")

    return "\n".join(summary)
