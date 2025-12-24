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
    dist_summary,
    heavy_share,
    cold_user_ratio,
    analyze_genre_distribution,
    plot_genre_distribution,
    calculate_concentration_metrics,
    plot_concentration_analysis,
    plot_distribution_summary,
    analyze_user_segments,
    segment_users,
    plot_user_segmentation,
    analyze_genre_combinations,
    find_rare_but_popular_combos,
    plot_rare_popular_combos,
    find_cold_start_items,
    analyze_cold_start_items,
    analyze_temporal_patterns,
    plot_temporal_heatmap,
    plot_temporal_trends,
    check_data_quality,
    plot_data_quality_summary,
    check_user_item_validity,
    check_temporal_validity,
    generate_data_quality_summary_text,
    check_metadata_quality,
    generate_metadata_summary_text,
)
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="RecSys Data Explorer", layout="wide")


# 캐싱
@st.cache_data(show_spinner=True)
def load_all_data(data_dir):
    ratings_df = load_ratings(data_dir)
    item_info_df = load_item_info(data_dir)
    item_attrs = load_item_attributes(data_dir)
    return ratings_df, item_info_df, item_attrs


@st.cache_data
def filter_by_time(df, start, end):
    return df[(df["datetime"] >= start) & (df["datetime"] <= end)]


@st.cache_data
def get_all_users_items(df):
    return (
        sorted(df["user"].unique()),
        sorted(df["item"].unique()),
    )


def main():
    st.title("🎬 RecSys Data Explorer")

    # --- Sidebar ---
    st.sidebar.header("Configuration")

    # Directory Input
    default_dir = os.path.join(os.getcwd(), "data", "raw", "train")
    data_dir = st.sidebar.text_input("Data Directory", value=default_dir)

    if st.sidebar.button("Load Data"):
        st.session_state["load_clicked"] = True

    if "load_clicked" not in st.session_state:
        st.info("Please verify the data directory and click 'Load Data'.")
        return

    # Load Data (CACHED)
    ratings_df, item_info_df, item_attrs = load_all_data(data_dir)

    if ratings_df is None:
        return

    # ---------------- Summary Metrics ----------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Users", f"{ratings_df['user'].nunique():,}")
    c2.metric("Total Movies", f"{ratings_df['item'].nunique():,}")
    c3.metric("Total Interactions", f"{len(ratings_df):,}")

    # ---------------- Global Filters ----------------
    st.sidebar.subheader("Global Filters")

    min_time = ratings_df["datetime"].min()
    max_time = ratings_df["datetime"].max()

    time_range = st.sidebar.slider(
        "Time Range",
        min_value=min_time.to_pydatetime(),
        max_value=max_time.to_pydatetime(),
        value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
    )

    # ---------------- Cached filtering ----------------
    filtered_df = filter_by_time(ratings_df, time_range[0], time_range[1])
    all_users, all_items = get_all_users_items(filtered_df)

    # ---------------- Tabs ----------------
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🔍 Data Quality",
            "👥 User Subset Analysis",
            "🌐 All Users Analysis",
            "🎬 Item Subset Analysis",
            "📊 All Items Analysis",
            "📈 Advanced EDA",
        ]
    )

    # Filter by time
    filtered_df = ratings_df[
        (ratings_df["datetime"] >= time_range[0])
        & (ratings_df["datetime"] <= time_range[1])
    ]

    all_users = sorted(filtered_df["user"].unique())
    all_items = sorted(filtered_df["item"].unique())

    # ==================== Tab 0: Data Quality ====================
    with tab0:
        st.header("🔍 Data Quality Check")
        st.markdown(
            """
        **전체 데이터셋 품질 검사**: Ratings, Metadata (Titles, Genres, Directors, Writers, Years) 모두 검증
        - **결측치**: 각 컬럼별 누락 데이터 비율
        - **중복**: 중복 레코드 수
        - **이상치**: IQR 방법으로 통계적 이상치 탐지
        - **타입 검증**: 데이터 타입 및 유효성
        - **메타데이터**: 영화 정보 완성도 체크
        """
        )

        # Quality check button
        if st.button(
            "🔄 Run Quality Check",
            type="primary",
            key="run_quality_check",
        ):

            # Initialize containers for progress tracking
            overall_progress = st.progress(0)
            overall_status = st.empty()

            with st.spinner("Analyzing data quality..."):

                # ====================  1. Ratings Data Quality ====================
                overall_status.text("📊 [1/6] Checking Ratings Data...")

                progress_bar_1 = st.progress(0)
                status_text_1 = st.empty()

                quality_report = check_data_quality(
                    filtered_df,
                    "Ratings Data",
                    show_progress=True,
                    progress_bar=progress_bar_1,
                    status_text=status_text_1,
                )

                overall_progress.progress(1 / 6)

                # ==================== 2. User/Item Validity ====================
                overall_status.text("🔍 [2/6] Validating User & Item IDs...")

                validity_report = check_user_item_validity(filtered_df, item_info_df)

                overall_progress.progress(2 / 6)

                # ==================== 3. Temporal Validity ====================
                overall_status.text("📅 [3/6] Checking Temporal Data...")

                temporal_report = check_temporal_validity(filtered_df)

                overall_progress.progress(3 / 6)

                # ==================== 4. Metadata Quality ====================
                overall_status.text("📋 [4/6] Checking Metadata Quality...")

                if not item_info_df.empty:
                    metadata_reports = check_metadata_quality(
                        item_info_df, show_progress=True
                    )
                else:
                    metadata_reports = {}

                overall_progress.progress(4 / 6)

                # ==================== 5. Item Info Quality (Detailed) ====================
                overall_status.text("📑 [5/6] Analyzing Item Info Dataset...")

                progress_bar_5 = st.progress(0)
                status_text_5 = st.empty()

                if not item_info_df.empty:
                    item_quality_report = check_data_quality(
                        item_info_df,
                        "Item Info Data",
                        show_progress=True,
                        progress_bar=progress_bar_5,
                        status_text=status_text_5,
                    )
                else:
                    item_quality_report = None

                overall_progress.progress(5 / 6)

                # ==================== 6. Save Results ====================
                overall_status.text("💾 [6/6] Finalizing reports...")

                st.session_state["quality_report"] = quality_report
                st.session_state["validity_report"] = validity_report
                st.session_state["temporal_report"] = temporal_report
                st.session_state["metadata_reports"] = metadata_reports
                st.session_state["item_quality_report"] = item_quality_report

                overall_progress.progress(1.0)
                overall_status.text("✅ Quality check complete!")

        # ==================== Display Results ====================
        if "quality_report" in st.session_state:
            quality_report = st.session_state["quality_report"]
            validity_report = st.session_state["validity_report"]
            temporal_report = st.session_state["temporal_report"]
            metadata_reports = st.session_state.get("metadata_reports", {})
            item_quality_report = st.session_state.get("item_quality_report", None)

            # ==================== Summary Section ====================
            st.markdown("---")
            st.subheader("📊 Executive Summary")

            # Create tabs for different datasets
            sum_tab1, sum_tab2, sum_tab3 = st.tabs(
                ["Ratings Data", "Metadata Summary", "Item Info Data"]
            )

            with sum_tab1:
                # Ratings summary
                summary_text = generate_data_quality_summary_text(
                    quality_report, validity_report, temporal_report
                )
                st.markdown(summary_text)

                # Quick metrics
                col1, col2, col3, col4 = st.columns(4)

                missing_total = quality_report["missing_values"]["Missing Count"].sum()
                col1.metric(
                    "Missing Values",
                    f"{missing_total:,}",
                    f"{missing_total/quality_report['shape'][0]*100:.2f}%",
                )

                col2.metric(
                    "Duplicate Rows",
                    f"{quality_report['duplicates']['total_duplicates']:,}",
                    f"{quality_report['duplicates']['duplicate_pct']:.2f}%",
                )

                outlier_total = sum(
                    [v["count"] for v in quality_report["outliers"].values()]
                )
                col3.metric("Total Outliers", f"{outlier_total:,}")

                col4.metric(
                    "Unique Pairs",
                    f"{validity_report['user_item_pairs']['total_unique_pairs']:,}",
                )

            with sum_tab2:
                # Metadata summary
                if metadata_reports:
                    metadata_summary = generate_metadata_summary_text(metadata_reports)
                    st.markdown(metadata_summary)

                    # Metadata completeness chart
                    completeness_data = []
                    for name, report in metadata_reports.items():
                        completeness_pct = 100 - report["missing_pct"]
                        completeness_data.append(
                            {
                                "Metadata": name,
                                "Completeness": completeness_pct,
                                "Missing": report["missing_pct"],
                            }
                        )

                    completeness_df = pd.DataFrame(completeness_data)

                    fig_completeness = go.Figure()

                    fig_completeness.add_trace(
                        go.Bar(
                            name="Complete",
                            x=completeness_df["Metadata"],
                            y=completeness_df["Completeness"],
                            marker_color="lightgreen",
                        )
                    )

                    fig_completeness.add_trace(
                        go.Bar(
                            name="Missing",
                            x=completeness_df["Metadata"],
                            y=completeness_df["Missing"],
                            marker_color="lightcoral",
                        )
                    )

                    fig_completeness.update_layout(
                        barmode="stack",
                        title="Metadata Completeness",
                        yaxis_title="Percentage (%)",
                        height=400,
                    )

                    st.plotly_chart(fig_completeness, use_container_width=True)
                else:
                    st.info("No metadata available")

            with sum_tab3:
                # Item info dataset quality
                if item_quality_report:
                    st.write(
                        f"**Shape**: {item_quality_report['shape'][0]:,} rows × {item_quality_report['shape'][1]} columns"
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Missing Values**")
                        missing_item = item_quality_report["missing_values"]
                        missing_item_display = missing_item[
                            missing_item["Missing %"] > 0
                        ]
                        if not missing_item_display.empty:
                            st.dataframe(missing_item_display, use_container_width=True)
                        else:
                            st.success("✅ No missing values")

                    with col2:
                        st.write("**Data Type Summary**")
                        dtype_counts = (
                            pd.Series(item_quality_report["data_types"])
                            .astype(str)
                            .value_counts()
                        )
                        dtype_counts_df = dtype_counts.reset_index()
                        dtype_counts_df.columns = ["Type", "Count"]
                        st.dataframe(dtype_counts_df, use_container_width=True)
                else:
                    st.info("No item info data available")

            # ==================== Visualizations Section ====================
            st.markdown("---")
            st.subheader("📈 Visual Analysis")

            viz_tab1, viz_tab2 = st.tabs(["Ratings Data", "Item Info Data"])

            with viz_tab1:
                fig_quality = plot_data_quality_summary(quality_report)
                st.plotly_chart(fig_quality, use_container_width=True)

            with viz_tab2:
                if item_quality_report:
                    fig_item_quality = plot_data_quality_summary(item_quality_report)
                    st.plotly_chart(fig_item_quality, use_container_width=True)
                else:
                    st.info("No visualization available")

            # ==================== Detailed Tables Section ====================
            st.markdown("---")
            st.subheader("📋 Detailed Reports")

            detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs(
                ["Missing Values", "Outliers", "Data Types", "Metadata Details"]
            )

            with detail_tab1:
                st.write("**Ratings Data - Missing Values**")
                st.dataframe(quality_report["missing_values"], use_container_width=True)

                if item_quality_report:
                    st.markdown("---")
                    st.write("**Item Info Data - Missing Values**")
                    st.dataframe(
                        item_quality_report["missing_values"], use_container_width=True
                    )

            with detail_tab2:
                st.write("**Ratings Data - Outliers (IQR Method)**")
                outlier_data = []
                for col, info in quality_report["outliers"].items():
                    outlier_data.append(
                        {
                            "Column": col,
                            "Outlier Count": info["count"],
                            "Outlier %": f"{info['pct']:.2f}%",
                            "Lower Bound": f"{info['lower_bound']:.2f}",
                            "Upper Bound": f"{info['upper_bound']:.2f}",
                        }
                    )
                st.dataframe(pd.DataFrame(outlier_data), use_container_width=True)

                if item_quality_report and item_quality_report["outliers"]:
                    st.markdown("---")
                    st.write("**Item Info Data - Outliers**")
                    outlier_data_item = []
                    for col, info in item_quality_report["outliers"].items():
                        outlier_data_item.append(
                            {
                                "Column": col,
                                "Outlier Count": info["count"],
                                "Outlier %": f"{info['pct']:.2f}%",
                                "Lower Bound": f"{info['lower_bound']:.2f}",
                                "Upper Bound": f"{info['upper_bound']:.2f}",
                            }
                        )
                    if outlier_data_item:
                        st.dataframe(
                            pd.DataFrame(outlier_data_item), use_container_width=True
                        )

            with detail_tab3:
                st.write("**Ratings Data - Data Types**")
                dtype_data = pd.DataFrame(
                    {
                        "Column": quality_report["data_types"].keys(),
                        "Data Type": [
                            str(v) for v in quality_report["data_types"].values()
                        ],
                        "Unique Values": [
                            quality_report["unique_counts"][k]
                            for k in quality_report["data_types"].keys()
                        ],
                    }
                )
                st.dataframe(dtype_data, use_container_width=True)

                if item_quality_report:
                    st.markdown("---")
                    st.write("**Item Info Data - Data Types**")
                    dtype_data_item = pd.DataFrame(
                        {
                            "Column": item_quality_report["data_types"].keys(),
                            "Data Type": [
                                str(v)
                                for v in item_quality_report["data_types"].values()
                            ],
                            "Unique Values": [
                                item_quality_report["unique_counts"][k]
                                for k in item_quality_report["data_types"].keys()
                            ],
                        }
                    )
                    st.dataframe(dtype_data_item, use_container_width=True)

            with detail_tab4:
                if metadata_reports:
                    for name, report in metadata_reports.items():
                        with st.expander(f"📋 {name}", expanded=False):
                            col1, col2, col3 = st.columns(3)

                            col1.metric("Total Items", f"{report['total_items']:,}")
                            col2.metric(
                                "Missing",
                                f"{report['missing_count']:,}",
                                f"{report['missing_pct']:.2f}%",
                            )
                            col3.metric("Unique Values", f"{report['unique_values']:,}")

                            # Type-specific metrics
                            if "min_year" in report:
                                st.write(
                                    f"**Year Range**: {report['min_year']} - {report['max_year']}"
                                )
                                if report["future_years"] > 0:
                                    st.warning(
                                        f"⚠️ Future years found: {report['future_years']}"
                                    )
                                if report["ancient_years"] > 0:
                                    st.warning(
                                        f"⚠️ Ancient years (<1800): {report['ancient_years']}"
                                    )

                            if "items_with_multiple" in report:
                                st.info(
                                    f"📊 Items with multiple values: {report['items_with_multiple']:,} ({report['multi_value_pct']:.2f}%)"
                                )
                else:
                    st.info("No metadata details available")

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
                # User Segmentation
                st.subheader("👤 User Segmentation")

                # Segment selected users
                user_counts_subset = display_df.groupby("user")["item"].count()

                # Calculate segment thresholds
                q25 = user_counts_subset.quantile(0.25)
                q75 = user_counts_subset.quantile(0.75)

                segment_info = []
                for user in selected_users:
                    count = user_counts_subset.get(user, 0)
                    if count >= q75:
                        segment = "Heavy"
                        color = "🔴"
                    elif count >= q25:
                        segment = "Medium"
                        color = "🟡"
                    else:
                        segment = "Light"
                        color = "🟢"

                    segment_info.append(
                        {
                            "User": user,
                            "Segment": f"{color} {segment}",
                            "Rating Count": count,
                        }
                    )

                segment_df_display = pd.DataFrame(segment_info)
                st.dataframe(segment_df_display, use_container_width=True)

                # Segment distribution
                segment_counts = pd.Series(
                    [s["Segment"] for s in segment_info]
                ).value_counts()

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.write("**Segment Distribution**")
                    for seg, count in segment_counts.items():
                        st.metric(seg, count)

                with col2:
                    fig_seg = go.Figure(
                        data=[
                            go.Pie(
                                labels=[s.split()[1] for s in segment_counts.index],
                                values=segment_counts.values,
                                marker=dict(colors=["#ff6b6b", "#ffd93d", "#6bcf7f"]),
                            )
                        ]
                    )
                    fig_seg.update_layout(title="Segment Distribution", height=300)
                    st.plotly_chart(fig_seg, use_container_width=True)

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
            min_value=1,
            max_value=120,
            value=30,
            step=1,
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

        st.markdown("---")
        st.subheader("👥 User Segmentation Analysis")

        if st.button("🔄 Analyze User Segments", key="analyze_segments"):
            with st.spinner("Segmenting users..."):
                # Get early stats if available
                early_stats_for_segment = st.session_state.get("early_stats", None)

                # Segment users
                segment_df = segment_users(filtered_df, early_stats_for_segment)
                st.session_state["segment_df"] = segment_df

        if "segment_df" in st.session_state:
            segment_df = st.session_state["segment_df"]

            # Summary statistics
            summary = analyze_user_segments(segment_df, filtered_df)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("**Segment Summary**")
                st.dataframe(summary, use_container_width=True)

            with col2:
                fig_seg = plot_user_segmentation(segment_df)
                st.plotly_chart(fig_seg, use_container_width=True)

            # Detailed breakdown
            st.write("**Segment Characteristics**")

            tab_seg1, tab_seg2, tab_seg3 = st.tabs(
                ["Heavy Users", "Medium Users", "Light Users"]
            )

            for tab, segment_name in zip(
                [tab_seg1, tab_seg2, tab_seg3], ["Heavy", "Medium", "Light"]
            ):
                with tab:
                    users_in_segment = segment_df[segment_df["segment"] == segment_name]
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Count", f"{len(users_in_segment):,}")
                    col_b.metric(
                        "Avg Ratings", f"{users_in_segment['rating_count'].mean():.1f}"
                    )
                    col_c.metric(
                        "Median Ratings",
                        f"{users_in_segment['rating_count'].median():.0f}",
                    )

                    if "is_bulk_rater" in users_in_segment.columns:
                        bulk_ratio = users_in_segment["is_bulk_rater"].mean()
                        st.info(
                            f"📊 **{bulk_ratio*100:.1f}%** of {segment_name} users are Bulk Raters"
                        )

                    st.write(f"**Top 10 {segment_name} Users**")
                    top_in_segment = users_in_segment.nlargest(10, "rating_count")
                    st.dataframe(
                        top_in_segment[["user", "rating_count"]],
                        use_container_width=True,
                    )

        # 🆕 Temporal Patterns
        st.markdown("---")
        st.subheader("⏰ Temporal Activity Patterns")

        if st.button("🔄 Analyze Temporal Patterns", key="analyze_temporal"):
            with st.spinner("Analyzing temporal patterns..."):
                patterns = analyze_temporal_patterns(filtered_df)
                st.session_state["temporal_patterns"] = patterns

        if "temporal_patterns" in st.session_state:
            patterns = st.session_state["temporal_patterns"]

            # Temporal trends
            fig_temporal = plot_temporal_trends(patterns)
            st.plotly_chart(fig_temporal, use_container_width=True)

            # Heatmap
            st.write("**Activity Heatmap**")
            fig_heatmap = plot_temporal_heatmap(filtered_df)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Key insights
            st.write("**📊 Temporal Insights**")

            # Peak hour
            peak_hour = patterns["hourly"].idxmax()
            st.info(
                f"🕐 **Peak Hour**: {peak_hour}:00 with {patterns['hourly'].max():,} ratings"
            )

            # Peak day
            day_labels = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            peak_day = patterns["daily"].idxmax()
            st.info(
                f"📅 **Peak Day**: {day_labels[peak_day]} with {patterns['daily'].max():,} ratings"
            )

            # Peak month
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
            peak_month = patterns["monthly"].idxmax()
            st.info(
                f"📆 **Peak Month**: {month_labels[peak_month-1]} with {patterns['monthly'].max():,} ratings"
            )

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

        if selected_items and not item_info_df.empty:
            st.markdown("---")
            st.subheader("🎭 Genre Analysis for Selected Items")

            selected_item_info = item_info_df[item_info_df["item"].isin(selected_items)]

            if not selected_item_info.empty and "genre" in selected_item_info.columns:
                # Genre distribution for selected items
                genre_df_subset = analyze_genre_distribution(
                    filtered_df[filtered_df["item"].isin(selected_items)], item_info_df
                )

                if genre_df_subset is not None and not genre_df_subset.empty:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.dataframe(genre_df_subset.head(10), use_container_width=True)

                    with col2:
                        fig_genre_subset = plot_genre_distribution(
                            genre_df_subset, "Genre Distribution - Selected Items"
                        )
                        if fig_genre_subset:
                            st.plotly_chart(fig_genre_subset, use_container_width=True)

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

        st.markdown("---")
        st.subheader("🎬 Genre Combination Analysis")

        if not item_info_df.empty:
            combo_df = analyze_genre_combinations(item_info_df, filtered_df)

            if combo_df is not None:
                st.write("**Top Genre Combinations**")

                # Display top combos
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**By Number of Items**")
                    st.dataframe(combo_df.head(20), use_container_width=True)

                with col2:
                    if "mean_review" in combo_df.columns:
                        st.write("**By Average Reviews per Item**")
                        top_by_review = combo_df.nlargest(20, "mean_review")
                        st.dataframe(top_by_review, use_container_width=True)

                # 🆕 Rare but Popular Combos
                st.markdown("---")
                st.subheader("💎 Rare but Popular Genre Combinations")
                st.markdown(
                    """
                이 분석은 **희귀하지만 인기 있는 장르 조합**을 찾습니다.
                - 아이템 수는 적지만 (희귀)
                - 평균 리뷰 수가 높은 (인기) 조합
                """
                )

                rare_popular = find_rare_but_popular_combos(combo_df, top_k=20)

                if rare_popular is not None and not rare_popular.empty:
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.dataframe(
                            rare_popular[
                                [
                                    "genre_combo",
                                    "item_count",
                                    "mean_review",
                                    "review_count",
                                ]
                            ],
                            use_container_width=True,
                        )

                    with col2:
                        fig_rare = plot_rare_popular_combos(rare_popular)
                        if fig_rare:
                            st.plotly_chart(fig_rare, use_container_width=True)

                    # Insights
                    st.info(
                        f"💡 Found **{len(rare_popular)}** rare but popular combinations"
                    )

        # 🆕 Item Cold Start Analysis
        st.markdown("---")
        st.subheader("❄️ Item Cold Start Analysis")

        cold_start_items = find_cold_start_items(filtered_df, item_info_df)

        if not cold_start_items.empty:
            st.warning(
                f"⚠️ Found **{len(cold_start_items):,} items** with metadata but NO ratings (Cold Start Items)"
            )

            cold_summary = analyze_cold_start_items(cold_start_items)

            if cold_summary:
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Cold Start Items", f"{cold_summary['total_count']:,}"
                )

                if "year_range" in cold_summary:
                    col2.metric("Year Range", cold_summary["year_range"])
                    col3.metric("Average Year", cold_summary["avg_year"])

                # Genre distribution of cold start items
                if "top_genres" in cold_summary:
                    st.write("**Top Genres in Cold Start Items**")
                    genre_data = pd.DataFrame(
                        list(cold_summary["top_genres"].items()),
                        columns=["Genre", "Count"],
                    )

                    fig_cold_genre = px.bar(
                        genre_data,
                        x="Genre",
                        y="Count",
                        title="Genre Distribution - Cold Start Items",
                    )
                    st.plotly_chart(fig_cold_genre, use_container_width=True)

            # Export cold start items
            with st.expander("📋 View Cold Start Items"):
                st.dataframe(cold_start_items.head(100), use_container_width=True)

                csv_cold = cold_start_items.to_csv(index=False)
                st.download_button(
                    label="📥 Download Cold Start Items (CSV)",
                    data=csv_cold,
                    file_name="cold_start_items.csv",
                    mime="text/csv",
                )
        else:
            st.success(
                "✅ No cold start items found - all items with metadata have been rated!"
            )

    # ==================== Tab 5: Advanced EDA ====================
    with tab5:
        st.header("📈 Advanced Exploratory Data Analysis")
        st.markdown("대규모 연산이 포함된 심화 EDA 분석")

        if "advanced_eda_results" not in st.session_state:
            st.session_state["advanced_eda_results"] = None

        st.info(
            "⚠️ Advanced EDA는 계산량이 많습니다.\n\n"
            "아래 버튼을 눌렀을 때만 분석이 시작됩니다."
        )

        if st.button("🚀 Run Advanced EDA", type="primary"):
            progress = st.progress(0)
            status = st.empty()

            status.text("📊 Computing distributions...")
            user_counts = filtered_df.groupby("user")["item"].count()
            item_counts = filtered_df.groupby("item")["user"].count()
            progress.progress(0.2)

            user_summary = dist_summary(user_counts, "User Interactions")
            item_summary = dist_summary(item_counts, "Item Ratings")

            status.text("🎯 Calculating concentration metrics...")
            metrics = calculate_concentration_metrics(filtered_df)
            progress.progress(0.4)

            status.text("📈 Plotting curves...")
            fig_dist = plot_distribution_summary(filtered_df)
            fig_conc = plot_concentration_analysis(filtered_df)
            progress.progress(0.6)

            status.text("❄️ Cold-start analysis...")
            cold_thresholds = [3, 10, 50, 100, 300, 500]
            cold_ratios = [
                cold_user_ratio(user_counts, k) * 100 for k in cold_thresholds
            ]
            progress.progress(0.8)

            status.text("🎭 Genre analysis...")
            genre_df = (
                analyze_genre_distribution(filtered_df, item_info_df)
                if not item_info_df.empty
                else None
            )
            progress.progress(1.0)

            st.session_state["advanced_eda_results"] = {
                "user_summary": user_summary,
                "item_summary": item_summary,
                "metrics": metrics,
                "fig_dist": fig_dist,
                "fig_conc": fig_conc,
                "cold_thresholds": cold_thresholds,
                "cold_ratios": cold_ratios,
                "genre_df": genre_df,
            }

            status.text("✅ Advanced EDA complete!")

        if st.session_state["advanced_eda_results"] is None:
            st.info("👆 Run Advanced EDA 버튼을 눌러 분석을 시작하세요.")
            return

        res = st.session_state["advanced_eda_results"]

        st.subheader("📊 Distribution Summary")
        st.dataframe(res["user_summary"], use_container_width=True)
        st.dataframe(res["item_summary"], use_container_width=True)
        st.plotly_chart(res["fig_dist"], use_container_width=True)

        st.markdown("---")
        st.subheader("🎯 Concentration Analysis")
        st.plotly_chart(res["fig_conc"], use_container_width=True)

        st.markdown("---")
        st.subheader("❄️ Cold Start Analysis")
        fig_cold = go.Figure(
            go.Bar(x=[f"≤ {k}" for k in res["cold_thresholds"]], y=res["cold_ratios"])
        )
        st.plotly_chart(fig_cold, use_container_width=True)

        st.markdown("---")
        st.subheader("🎭 Genre Analysis")
        if res["genre_df"] is not None and not res["genre_df"].empty:
            st.dataframe(res["genre_df"].head(20), use_container_width=True)


if __name__ == "__main__":
    main()
