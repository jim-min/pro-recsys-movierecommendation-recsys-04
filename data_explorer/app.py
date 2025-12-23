import streamlit as st
import os
import pandas as pd
import numpy as np
from loader import load_ratings, load_item_info, load_item_attributes
from utils import (
    plot_user_interactions,
    get_user_stats,
    get_temporal_stats,
    get_top_k_users,
    get_top_k_items,
    get_top_k_interactions,
    analyze_early_account_activity,
    plot_early_activity_distribution,
    compare_bulk_vs_regular_users,
)
import plotly.express as px

st.set_page_config(page_title="RecSys Data Explorer", layout="wide")


def main():
    st.title("🎬 RecSys Data Explorer")

    # --- Sidebar ---
    st.sidebar.header("Configuration")

    # 1. Directory Input
    default_dir = os.path.join(os.getcwd(), "data", "raw", "train")
    data_dir = st.sidebar.text_input("Data Directory", value=default_dir)

    if st.sidebar.button("Load Data"):
        st.session_state["load_clicked"] = True

    if "load_clicked" not in st.session_state:
        st.info("Please verify the data directory and click 'Load Data'.")
        return

    # Load Data
    with st.spinner("Loading data..."):
        ratings_df = load_ratings(data_dir)
        item_info_df = load_item_info(data_dir)
        item_attrs = load_item_attributes(data_dir)

    if ratings_df is None:
        return

    # Display summary metrics
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Total Users", f"{ratings_df['user'].nunique():,}")
    m_col2.metric("Total Movies", f"{ratings_df['item'].nunique():,}")
    m_col3.metric("Total Interactions", f"{len(ratings_df):,}")

    # 2. Global Filters (Sidebar)
    st.sidebar.subheader("Global Filters")

    # Time Range
    min_time = ratings_df["datetime"].min()
    max_time = ratings_df["datetime"].max()

    time_range = st.sidebar.slider(
        "Time Range",
        min_value=min_time.to_pydatetime(),
        max_value=max_time.to_pydatetime(),
        value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
    )

    # --- Main Content ---

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "👥 User Subset Analysis",
            "🌐 All Users Analysis",
            "🎬 Item Subset Analysis",
            "📊 All Items Analysis",
        ]
    )

    # Filter by time
    filtered_df = ratings_df[
        (ratings_df["datetime"] >= time_range[0])
        & (ratings_df["datetime"] <= time_range[1])
    ]

    all_users = sorted(filtered_df["user"].unique())
    all_items = sorted(filtered_df["item"].unique())

    # ==================== Tab 1: User Subset Analysis ====================
    with tab1:
        st.header("👥 User Subset Analysis")
        st.markdown("Analyze specific users or a range of users in detail.")

        # User selection
        st.subheader("🎯 Select Users")
        user_select_mode = st.radio(
            "Selection Mode",
            ["Specific Users", "User ID Range", "Random Sample"],
            key="tab1_user_mode",
            horizontal=True,
        )

        selected_users = []

        if user_select_mode == "Specific Users":
            selected_users = st.multiselect(
                "Select Users",
                all_users,
                default=all_users[:5] if len(all_users) >= 5 else all_users,
                key="tab1_specific_users",
            )

        elif user_select_mode == "User ID Range":
            col1, col2 = st.columns(2)
            min_id = int(min(all_users))
            max_id = int(max(all_users))

            start_id = col1.number_input(
                "Start ID",
                min_value=min_id,
                max_value=max_id,
                value=min_id,
                key="tab1_range_start",
            )
            end_id = col2.number_input(
                "End ID",
                min_value=min_id,
                max_value=max_id,
                value=min(min_id + 50, max_id),
                key="tab1_range_end",
            )

            if start_id <= end_id:
                selected_users = [u for u in all_users if start_id <= u <= end_id]
                st.info(
                    f"✓ Selected {len(selected_users)} users (ID {start_id} to {end_id})"
                )
            else:
                st.error("Start ID must be <= End ID")

        elif user_select_mode == "Random Sample":
            col1, col2, col3 = st.columns(3)
            sample_size = col1.number_input(
                "Sample Size",
                min_value=1,
                max_value=len(all_users),
                value=min(100, len(all_users)),
                key="tab1_sample_size",
            )
            random_seed = col2.number_input(
                "Random Seed", min_value=0, value=42, key="tab1_random_seed"
            )

            if col3.button("🎲 Sample", key="tab1_sample_btn"):
                np.random.seed(random_seed)
                selected_users = list(
                    np.random.choice(all_users, size=sample_size, replace=False)
                )
                st.session_state["tab1_sampled_users"] = selected_users

            if "tab1_sampled_users" in st.session_state:
                selected_users = st.session_state["tab1_sampled_users"]
                st.success(f"✓ Using {len(selected_users)} randomly sampled users")

        st.markdown("---")

        # Visualizations
        if selected_users:
            display_df = filtered_df[filtered_df["user"].isin(selected_users)]

            if display_df.empty:
                st.warning(
                    "No interactions found for selected users in current time range."
                )
            else:
                # User interaction timeline
                st.subheader("📈 User Interaction Timeline")
                use_separate = st.checkbox(
                    "Separate Time Axes per User", value=False, key="tab1_separate_axes"
                )

                fig = plot_user_interactions(
                    display_df, selected_users, use_separate_axis=use_separate
                )
                st.plotly_chart(fig, use_container_width=True)

                # User statistics
                st.subheader("📊 User Statistics")
                stats = get_user_stats(display_df, selected_users)
                st.dataframe(stats, use_container_width=True)

                # Temporal distribution
                st.subheader("📅 Temporal Distribution")
                temp_df = get_temporal_stats(display_df, selected_users)

                period = st.selectbox(
                    "Group By", ["Day", "Week", "Month", "Year"], key="tab1_period"
                )
                period_map_resample = {
                    "Day": "D",
                    "Week": "W",
                    "Month": "MS",
                    "Year": "YS",
                }

                if not temp_df.empty:
                    vis_data = []

                    for user in selected_users:
                        user_df = temp_df[temp_df["user"] == user].set_index("datetime")
                        resampled = user_df.resample(period_map_resample[period]).size()

                        u_res = resampled.reset_index(name="count")
                        u_res["user"] = user
                        vis_data.append(u_res)

                    if vis_data:
                        final_vis_df = pd.concat(vis_data)

                        if period == "Year":
                            final_vis_df["time_str"] = final_vis_df[
                                "datetime"
                            ].dt.strftime("%Y")
                        elif period == "Month":
                            final_vis_df["time_str"] = final_vis_df[
                                "datetime"
                            ].dt.strftime("%Y-%m")
                        elif period == "Week":
                            final_vis_df["time_str"] = final_vis_df[
                                "datetime"
                            ].dt.strftime("%Y-%m-%d")
                        elif period == "Day":
                            final_vis_df["time_str"] = final_vis_df[
                                "datetime"
                            ].dt.strftime("%Y-%m-%d")

                        fig_temp = px.bar(
                            final_vis_df,
                            x="time_str",
                            y="count",
                            color="user",
                            barmode="group",
                        )
                        fig_temp.update_xaxes(type="category", title="Time")
                        st.plotly_chart(fig_temp, use_container_width=True)
        else:
            st.info("👆 Please select users to analyze using the controls above.")

    # ==================== Tab 2: All Users Analysis ====================
    with tab2:
        st.header("🌐 All Users Analysis")
        st.markdown("Analyze patterns across all users in the dataset.")

        st.info(
            f"📊 Analyzing **{len(all_users):,} users** with **{len(filtered_df):,} total interactions**"
        )

        # Early Account Activity Analysis
        st.subheader("🔍 Early Account Activity Analysis")
        st.markdown(
            """
        **가설**: 계정 생성 초기에 응집적으로 평가한 기록은 사용자의 선호도를 나타내는 중요한 정보일 수 있습니다.
        
        이 분석은 각 사용자의 **첫 평가 이후 단시간 내에 얼마나 많은 평가를 몰아서 남겼는지** 확인합니다.
        """
        )

        # Analysis parameters
        col1, col2 = st.columns(2)

        time_window_minutes = col1.slider(
            "Time Window (minutes)",
            min_value=5,
            max_value=1440,
            value=30,
            step=5,
            key="tab2_time_window",
            help="첫 평가 이후 고려할 시간 범위 (분 단위)",
        )
        time_window_hours = time_window_minutes / 60

        min_ratings = col2.number_input(
            "Minimum Ratings for Bulk Classification",
            min_value=1,
            value=5,
            key="tab2_min_ratings",
            help="해당 시간 내 최소 평가 수 (이상이면 bulk rater로 분류)",
        )

        st.info(
            f"📊 분석 설정: **{time_window_minutes}분 ({time_window_hours:.1f}시간)** 내에 **{min_ratings}개 이상** 평가한 사용자를 찾습니다."
        )

        if st.button("🔄 Run Analysis", type="primary", key="tab2_run_analysis"):
            with st.spinner("Analyzing early account activity..."):
                early_stats = analyze_early_account_activity(
                    filtered_df,
                    time_window_hours=time_window_hours,
                    min_ratings=min_ratings,
                    show_progress=True,
                )
                st.session_state["early_stats"] = early_stats
                st.session_state["time_window_hours"] = time_window_hours

        if "early_stats" in st.session_state:
            early_stats = st.session_state["early_stats"]
            time_window = st.session_state.get("time_window_hours", time_window_hours)

            # Summary metrics
            st.subheader("📊 Summary Statistics")
            sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)

            bulk_count = early_stats["is_bulk_rater"].sum()
            total_users = len(early_stats)
            avg_early_ratings = early_stats["early_ratings"].mean()
            avg_early_ratio = early_stats["early_ratio"].mean()

            sum_col1.metric(
                "Bulk Raters",
                f"{bulk_count:,}",
                f"{bulk_count/total_users*100:.1f}% of users",
            )
            sum_col2.metric("Regular Users", f"{total_users - bulk_count:,}")
            sum_col3.metric("Avg Early Ratings", f"{avg_early_ratings:.1f}")
            sum_col4.metric("Avg Early Ratio", f"{avg_early_ratio:.1%}")

            # Additional insights
            st.markdown("---")
            insight_col1, insight_col2, insight_col3 = st.columns(3)

            bulk_stats = early_stats[early_stats["is_bulk_rater"] == True]
            if len(bulk_stats) > 0:
                avg_velocity = bulk_stats["ratings_per_minute"].mean()
                max_early = bulk_stats["early_ratings"].max()
                min_duration = bulk_stats[bulk_stats["early_duration_minutes"] > 0][
                    "early_duration_minutes"
                ].min()

                insight_col1.metric(
                    "Avg Rating Velocity",
                    f"{avg_velocity:.2f} ratings/min",
                    help="Bulk raters의 평균 평가 속도",
                )
                insight_col2.metric(
                    "Max Early Ratings",
                    f"{int(max_early):,}",
                    help="한 사용자가 초기에 남긴 최대 평가 수",
                )
                insight_col3.metric(
                    "Min Duration",
                    (
                        f"{min_duration:.1f} min"
                        if min_duration == min_duration
                        else "N/A"
                    ),
                    help="가장 짧은 평가 기간",
                )

            # Distribution plots
            st.subheader("📈 Distribution Analysis")
            fig_dist = plot_early_activity_distribution(early_stats)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Comparison table
            st.subheader("⚖️ Bulk Raters vs Regular Users")
            comparison_df = compare_bulk_vs_regular_users(filtered_df, early_stats)
            st.dataframe(comparison_df, use_container_width=True)

            # Top bulk raters
            st.subheader("🏆 Top Bulk Raters")
            top_bulk = early_stats[early_stats["is_bulk_rater"] == True].nlargest(
                20, "early_ratings"
            )

            col_table, col_chart = st.columns([1, 1])

            with col_table:
                display_cols = [
                    "user",
                    "early_ratings",
                    "total_ratings",
                    "early_ratio",
                    "early_duration_minutes",
                    "ratings_per_minute",
                ]
                display_top_bulk = top_bulk[display_cols].copy()
                display_top_bulk["early_ratio"] = display_top_bulk["early_ratio"].apply(
                    lambda x: f"{x:.1%}"
                )
                display_top_bulk["early_duration_minutes"] = display_top_bulk[
                    "early_duration_minutes"
                ].apply(lambda x: f"{x:.1f}")
                display_top_bulk["ratings_per_minute"] = display_top_bulk[
                    "ratings_per_minute"
                ].apply(lambda x: f"{x:.2f}")

                st.dataframe(display_top_bulk, use_container_width=True)

            with col_chart:
                fig_top = px.bar(
                    top_bulk.head(10),
                    x="user",
                    y="early_ratings",
                    title="Top 10 Bulk Raters by Early Ratings Count",
                    labels={"early_ratings": "Early Ratings", "user": "User ID"},
                )
                fig_top.update_xaxes(type="category")
                st.plotly_chart(fig_top, use_container_width=True)

        # Overall User Rankings
        st.markdown("---")
        st.subheader("📊 User Rankings")

        col1, col2 = st.columns(2)
        k_users = col1.number_input(
            "Top/Bottom K", min_value=1, value=10, key="tab2_k_users"
        )
        order_users = col2.selectbox(
            "Order", ["Top (Most)", "Bottom (Least)"], key="tab2_order_users"
        )

        bottom_flag = order_users == "Bottom (Least)"
        res = get_top_k_users(filtered_df, k_users, bottom_flag)

        col_table, col_chart = st.columns([1, 2])
        with col_table:
            st.dataframe(res, use_container_width=True)
        with col_chart:
            fig_rank = px.bar(
                res, x="user", y="count", title=f"{order_users} {k_users} Users"
            )
            fig_rank.update_xaxes(type="category")
            st.plotly_chart(fig_rank, use_container_width=True)

    # ==================== Tab 3: Item Subset Analysis ====================
    with tab3:
        st.header("🎬 Item Subset Analysis")
        st.markdown("Analyze specific items or a range of items in detail.")

        # Item selection
        st.subheader("🎯 Select Items")
        item_select_mode = st.radio(
            "Selection Mode",
            ["Specific Items", "Item ID Range", "Random Sample"],
            key="tab3_item_mode",
            horizontal=True,
        )

        selected_items = []

        if item_select_mode == "Specific Items":
            selected_items = st.multiselect(
                "Select Items",
                all_items,
                default=all_items[:10] if len(all_items) >= 10 else all_items,
                key="tab3_specific_items",
            )

        elif item_select_mode == "Item ID Range":
            col1, col2 = st.columns(2)
            min_item_id = int(min(all_items))
            max_item_id = int(max(all_items))

            start_item_id = col1.number_input(
                "Start ID",
                min_value=min_item_id,
                max_value=max_item_id,
                value=min_item_id,
                key="tab3_range_start",
            )
            end_item_id = col2.number_input(
                "End ID",
                min_value=min_item_id,
                max_value=max_item_id,
                value=min(min_item_id + 50, max_item_id),
                key="tab3_range_end",
            )

            if start_item_id <= end_item_id:
                selected_items = [
                    i for i in all_items if start_item_id <= i <= end_item_id
                ]
                st.info(
                    f"✓ Selected {len(selected_items)} items (ID {start_item_id} to {end_item_id})"
                )
            else:
                st.error("Start ID must be <= End ID")

        elif item_select_mode == "Random Sample":
            col1, col2, col3 = st.columns(3)
            sample_size = col1.number_input(
                "Sample Size",
                min_value=1,
                max_value=len(all_items),
                value=min(100, len(all_items)),
                key="tab3_sample_size",
            )
            random_seed = col2.number_input(
                "Random Seed", min_value=0, value=42, key="tab3_random_seed"
            )

            if col3.button("🎲 Sample", key="tab3_sample_btn"):
                np.random.seed(random_seed)
                selected_items = list(
                    np.random.choice(all_items, size=sample_size, replace=False)
                )
                st.session_state["tab3_sampled_items"] = selected_items

            if "tab3_sampled_items" in st.session_state:
                selected_items = st.session_state["tab3_sampled_items"]
                st.success(f"✓ Using {len(selected_items)} randomly sampled items")

        st.markdown("---")

        # Item analysis
        if selected_items:
            display_df = filtered_df[filtered_df["item"].isin(selected_items)]

            if display_df.empty:
                st.warning(
                    "No interactions found for selected items in current time range."
                )
            else:
                # Item statistics
                st.subheader("📊 Item Statistics")
                item_stats = (
                    display_df.groupby("item")
                    .agg(
                        total_ratings=("user", "count"),
                        unique_users=("user", "nunique"),
                        first_rating=("datetime", "min"),
                        last_rating=("datetime", "max"),
                    )
                    .reset_index()
                )
                st.dataframe(item_stats, use_container_width=True)

                # Item popularity over time
                st.subheader("📈 Item Popularity Over Time")
                period = st.selectbox(
                    "Group By", ["Day", "Week", "Month", "Year"], key="tab3_period"
                )
                period_map = {"Day": "D", "Week": "W", "Month": "MS", "Year": "YS"}

                vis_data = []
                for item in selected_items:
                    item_df = display_df[display_df["item"] == item].set_index(
                        "datetime"
                    )
                    resampled = item_df.resample(period_map[period]).size()
                    i_res = resampled.reset_index(name="count")
                    i_res["item"] = item
                    vis_data.append(i_res)

                if vis_data:
                    final_vis_df = pd.concat(vis_data)

                    if period == "Year":
                        final_vis_df["time_str"] = final_vis_df["datetime"].dt.strftime(
                            "%Y"
                        )
                    elif period == "Month":
                        final_vis_df["time_str"] = final_vis_df["datetime"].dt.strftime(
                            "%Y-%m"
                        )
                    elif period == "Week":
                        final_vis_df["time_str"] = final_vis_df["datetime"].dt.strftime(
                            "%Y-%m-%d"
                        )
                    elif period == "Day":
                        final_vis_df["time_str"] = final_vis_df["datetime"].dt.strftime(
                            "%Y-%m-%d"
                        )

                    fig_item = px.bar(
                        final_vis_df,
                        x="time_str",
                        y="count",
                        color="item",
                        barmode="group",
                    )
                    fig_item.update_xaxes(type="category", title="Time")
                    st.plotly_chart(fig_item, use_container_width=True)

                # Item information details
                if not item_info_df.empty:
                    st.subheader("🎥 Item Information")
                    selected_movie = st.selectbox(
                        "Select Item for Details",
                        selected_items,
                        key="tab3_movie_select",
                    )

                    if selected_movie and selected_movie in item_info_df["item"].values:
                        info = item_info_df[
                            item_info_df["item"] == selected_movie
                        ].iloc[0]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Movie ID", selected_movie)
                            st.write(f"**Title:** {info.get('title', 'N/A')}")
                            st.write(f"**Genres:** {info.get('genre', 'N/A')}")
                        with col2:
                            st.write(f"**Director:** {info.get('director', 'N/A')}")
                            st.write(f"**Writer:** {info.get('writer', 'N/A')}")
                            st.write(f"**Year:** {info.get('year', 'N/A')}")

                        if item_attrs and str(selected_movie) in item_attrs:
                            st.write(
                                f"**Genre IDs (from attributes):** {item_attrs[str(selected_movie)]}"
                            )
        else:
            st.info("👆 Please select items to analyze using the controls above.")

    # ==================== Tab 4: All Items Analysis ====================
    with tab4:
        st.header("📊 All Items Analysis")
        st.markdown("Analyze patterns across all items in the dataset.")

        st.info(
            f"📊 Analyzing **{len(all_items):,} items** with **{len(filtered_df):,} total interactions**"
        )

        # Item Rankings
        st.subheader("🏆 Item Rankings")

        col1, col2 = st.columns(2)
        k_items = col1.number_input(
            "Top/Bottom K", min_value=1, value=10, key="tab4_k_items"
        )
        order_items = col2.selectbox(
            "Order", ["Top (Most)", "Bottom (Least)"], key="tab4_order_items"
        )

        bottom_flag = order_items == "Bottom (Least)"
        res = get_top_k_items(filtered_df, k_items, bottom_flag)

        st.write(f"**{order_items} {k_items} Items by Interaction Count**")

        col_table, col_chart = st.columns([1, 2])
        with col_table:
            st.dataframe(res, use_container_width=True)
        with col_chart:
            fig_rank = px.bar(
                res, x="item", y="count", title=f"{order_items} {k_items} Items"
            )
            fig_rank.update_xaxes(type="category")
            st.plotly_chart(fig_rank, use_container_width=True)

        # Item distribution analysis
        st.markdown("---")
        st.subheader("📈 Item Popularity Distribution")

        item_counts = filtered_df["item"].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Distribution Statistics**")
            st.metric("Total Items", f"{len(item_counts):,}")
            st.metric("Avg Ratings per Item", f"{item_counts.mean():.1f}")
            st.metric("Median Ratings per Item", f"{item_counts.median():.0f}")
            st.metric("Max Ratings (Single Item)", f"{item_counts.max():,}")
            st.metric("Min Ratings (Single Item)", f"{item_counts.min():,}")

        with col2:
            fig_dist = px.histogram(
                item_counts.values,
                nbins=50,
                title="Distribution of Ratings per Item",
                labels={"value": "Number of Ratings", "count": "Number of Items"},
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        # User-Item interaction matrix stats
        st.markdown("---")
        st.subheader("🔢 Interaction Matrix Statistics")

        res_interactions = get_top_k_interactions(filtered_df, k_items, bottom_flag)
        st.write(f"**{order_items} {k_items} User-Item Interactions**")
        st.dataframe(res_interactions, use_container_width=True)


if __name__ == "__main__":
    main()
