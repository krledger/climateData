# -*- coding: utf-8 -*-
"""
Climate Viewer Metrics Tab
===========================
climate_viewer_tab_metrics.py
Last Updated: 2025-12-02 12:00 AEST - Fixed: tab is now display-only

Bug Fixes Applied:
- Removed duplicate smoothing (was applied here AND in main app)
- Removed duplicate alignment (was always applied, ignoring toggle)
- Tab now receives pre-processed data and only displays it
- Mode-specific transformations (Baseline, Deltas) still handled here
  as they depend on display settings, not data processing toggles

Architecture:
- Main app handles: BC toggle, Smooth toggle, Align toggle
- This tab handles: Display mode (Values/Baseline/Deltas), filtering, visualisation
"""
import pandas as pd
import altair as alt
from climate_viewer_constants import IDX_COLS_ANNUAL, IDX_COLS_SEASONAL, WARMING_TARGET, AMPLIFICATION_FACTOR, EMOJI
from climate_viewer_data_operations import apply_deltas_vs_base, apply_baseline_from_start


def render_metrics_tab(st, ns, loc_sel, scen_sel, type_sel, name_sel, season_sel, y0, y1,
                       mode, smooth, smooth_win, show_global_thresholds, show_horizon_shading,
                       short_start, mid_start, long_start, horizon_end,
                       df_all, base_df, preindustrial_baseline, BASE_LABEL):
    """
    Render the Climate Metrics tab.

    This is a DISPLAY-ONLY function. Data processing (BC, smoothing, alignment)
    is handled by the main app before data is passed here.

    This function handles:
    - Content filtering (type, name, season, location)
    - Display mode transformations (Baseline, Deltas)
    - Year range filtering
    - Chart rendering
    - Data table display
    """
    st.title(f"{EMOJI['thermometer']} Climate Metrics Viewer")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{EMOJI['map']} Locations", len(loc_sel))
    with col2:
        st.metric(f"{EMOJI['thermometer']} Scenarios", len(scen_sel))
    with col3:
        st.metric(f"{EMOJI['calendar']} Year Range", f"{y0}-{y1}")
    with col4:
        st.metric(f"{EMOJI['graph']} Mode", mode.split("(")[0].strip())

    st.caption(
        f"Locations: {', '.join(loc_sel)} | "
        f"Scenarios: {', '.join(scen_sel)} | "
        f"Type: {type_sel} | "
        f"Metrics: {', '.join(name_sel)}"
    )

    # Filter by content (type, name, season, location)
    # Year range already applied by main app, but we re-filter for display modes
    content_mask = (
            df_all["Location"].isin(loc_sel) &
            df_all["Season"].isin(season_sel) &
            (df_all["Type"] == type_sel) &
            df_all["Name"].isin(name_sel)
    )
    view = df_all[content_mask].copy()

    use_baseline = mode.startswith("Baseline")
    apply_delta = mode.startswith("Deltas")

    # Apply display mode transformations
    # Note: These are display transformations, not data processing
    if apply_delta:
        base_content_mask = (
                base_df["Location"].isin(loc_sel) &
                base_df["Season"].isin(season_sel) &
                (base_df["Type"] == type_sel) &
                base_df["Name"].isin(name_sel)
        )
        base_filtered = base_df[base_content_mask].copy()
        view = apply_deltas_vs_base(view, base_filtered)

    elif use_baseline:
        view = apply_baseline_from_start(view, baseline_year=y0)

    # Data is already filtered to year range by main app
    # No additional year filtering needed here

    view = view.dropna(subset=["Value"])

    if view.empty:
        st.warning("No data matches your current filters.  Try:")
        st.markdown("""
        - Expanding the year range
        - Selecting additional scenarios or locations
        - Choosing different metric names
        """)
        st.stop()

    use_seasonal = len(season_sel) > 1 or (
            len(season_sel) == 1 and season_sel[0] != "Annual"
    )
    idx_cols = IDX_COLS_SEASONAL if use_seasonal else IDX_COLS_ANNUAL

    # Calculate warming reference lines
    warming_references = {}  # Dict[threshold -> Dict[location -> temp]]
    DISPLAY_THRESHOLDS = [1.5, 2.0, 2.5]
    if type_sel == "Temp" and "Average" in name_sel and show_global_thresholds:
        if preindustrial_baseline:
            for threshold in DISPLAY_THRESHOLDS:
                # Regional warming = Global warming × Amplification factor
                regional_warming_at_threshold = threshold * AMPLIFICATION_FACTOR

                if use_baseline or apply_delta:
                    # In baseline/delta mode, calculate relative to display baseline
                    warming_references[threshold] = {}
                    for loc, preindustrial_temp in preindustrial_baseline.items():
                        target_absolute_temp = preindustrial_temp + regional_warming_at_threshold

                        loc_data = df_all[
                            (df_all["Location"] == loc) &
                            (df_all["Type"] == "Temp") &
                            (df_all["Name"] == "Average") &
                            (df_all["Season"] == "Annual") &
                            (df_all["Year"] == y0)
                            ]
                        if not loc_data.empty:
                            baseline_temp = loc_data["Value"].mean()
                            warming_references[threshold][loc] = target_absolute_temp - baseline_temp
                else:
                    # Absolute mode - use absolute temperature values
                    warming_references[threshold] = {
                        loc: temp + regional_warming_at_threshold
                        for loc, temp in preindustrial_baseline.items()
                    }

    # Summary statistics
    with st.expander(f"{EMOJI['chart']} Summary Statistics", expanded=False):
        summary = view.groupby(["Scenario", "Name"])["Value"].agg([
            ("Mean", "mean"),
            ("Min", "min"),
            ("Max", "max"),
            ("Change", lambda x: x.max() - x.min())
        ]).round(2)
        st.dataframe(summary, use_container_width=True)

    st.subheader(f"{EMOJI['graph']} Visualisation")

    # Render chart for each location
    for location in loc_sel:
        st.markdown(f"{EMOJI['pin']} {location}")

        location_data = view[view["Location"] == location].copy()

        if location_data.empty:
            st.info(f"No data available for {location}")
            continue

        plot_data = location_data[idx_cols + ["Data Type", "Value", "Location", "Name", "Scenario"]].copy()
        plot_data = plot_data.rename(columns={"Data Type": "Metric"})

        if idx_cols == IDX_COLS_ANNUAL:
            plot_data["X"] = plot_data["Year"].astype(int)
            x_encoding = alt.X(
                "X:Q",
                title="Year",
                axis=alt.Axis(format="d", labelFontSize=12, titleFontSize=14),
                scale=alt.Scale(domain=[int(y0), int(y1)])
            )
        else:
            season_order = ["DJF", "MAM", "JJA", "SON"]
            plot_data["Season"] = pd.Categorical(
                plot_data["Season"], categories=season_order, ordered=True
            )
            plot_data = plot_data.sort_values(["Year", "Season"])
            plot_data["X"] = (
                    plot_data["Year"].astype(int).astype(str) +
                    "-" +
                    plot_data["Season"].astype(str)
            )
            x_encoding = alt.X(
                "X:N",
                title="Year-Season",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14, labelAngle=-45),
                sort=list(plot_data["X"].unique())
            )

        selection = alt.selection_point(fields=["Metric", "Scenario"], bind="legend")

        # Calculate y-axis domain including reference lines if present
        y_min = plot_data["Value"].min()
        y_max = plot_data["Value"].max()

        # Extend domain to include all threshold lines
        for threshold, loc_temps in warming_references.items():
            if location in loc_temps:
                ref_temp = loc_temps[location]
                y_min = min(y_min, ref_temp)
                y_max = max(y_max, ref_temp)

        y_range = y_max - y_min
        y_padding = y_range * 0.05 if y_range > 0 else 1
        y_domain = [y_min - y_padding, y_max + y_padding]

        chart_layers = []

        # Time horizon shading
        if show_horizon_shading and idx_cols == IDX_COLS_ANNUAL:
            if short_start < mid_start and short_start >= y0 and mid_start <= y1:
                short_rect = alt.Chart(pd.DataFrame({
                    'start': [max(short_start, y0)],
                    'end': [min(mid_start, y1)]
                })).mark_rect(opacity=0.15, color='#3498DB').encode(
                    x=alt.X('start:Q', scale=alt.Scale(domain=[int(y0), int(y1)])),
                    x2=alt.X2('end:Q'),
                    tooltip=alt.value(f'Short period: {short_start}-{mid_start}')
                )
                chart_layers.append(short_rect)

            if mid_start < long_start and mid_start >= y0 and long_start <= y1:
                mid_rect = alt.Chart(pd.DataFrame({
                    'start': [max(mid_start, y0)],
                    'end': [min(long_start, y1)]
                })).mark_rect(opacity=0.15, color='#F39C12').encode(
                    x=alt.X('start:Q', scale=alt.Scale(domain=[int(y0), int(y1)])),
                    x2=alt.X2('end:Q'),
                    tooltip=alt.value(f'Mid period: {mid_start}-{long_start}')
                )
                chart_layers.append(mid_rect)

            if long_start < horizon_end and long_start >= y0 and horizon_end <= y1:
                long_rect = alt.Chart(pd.DataFrame({
                    'start': [max(long_start, y0)],
                    'end': [min(horizon_end, y1)]
                })).mark_rect(opacity=0.15, color='#E74C3C').encode(
                    x=alt.X('start:Q', scale=alt.Scale(domain=[int(y0), int(y1)])),
                    x2=alt.X2('end:Q'),
                    tooltip=alt.value(f'Long period: {long_start}-{horizon_end}')
                )
                chart_layers.append(long_rect)

        # Main line chart
        chart = (
            alt.Chart(plot_data)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=x_encoding,
                y=alt.Y(
                    "Value:Q",
                    title="Value",
                    scale=alt.Scale(domain=y_domain),
                    axis=alt.Axis(labelFontSize=12, titleFontSize=14)
                ),
                color=alt.Color(
                    "Scenario:N",
                    title="Scenario",
                    legend=alt.Legend(
                        labelFontSize=12,
                        titleFontSize=14,
                        labelLimit=400
                    )
                ),
                strokeDash=alt.StrokeDash(
                    "Metric:N",
                    title="Metric",
                    legend=alt.Legend(
                        labelFontSize=12,
                        titleFontSize=14,
                        labelLimit=300
                    )
                ),
                tooltip=[
                    alt.Tooltip("Scenario:N", title="Scenario"),
                    alt.Tooltip("Metric:N", title="Metric"),
                    alt.Tooltip("Value:Q", format=",.2f", title="Value"),
                    alt.Tooltip("X:N", title="Period"),
                    alt.Tooltip("Location:N", title="Location"),
                    alt.Tooltip("Name:N", title="Name"),
                ],
                opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
            )
            .add_params(selection)
        )

        chart_layers.append(chart)

        # Add threshold reference lines (1.5, 2.0, 2.5°C) with labels
        threshold_colors = {1.5: "#E74C3C", 2.0: "#F39C12", 2.5: "#8E44AD"}  # Red, Orange, Purple
        for threshold, loc_temps in warming_references.items():
            if location in loc_temps:
                ref_temp = loc_temps[location]
                ref_data = pd.DataFrame({"Value": [ref_temp], "Year": [y1], "Label": [f"{threshold}°C"]})

                # Reference line
                reference_line = (
                    alt.Chart(ref_data)
                    .mark_rule(color=threshold_colors.get(threshold, "red"), strokeDash=[5, 5], strokeWidth=2)
                    .encode(
                        y="Value:Q",
                        tooltip=alt.value(f"{threshold}°C global warming threshold")
                    )
                )
                chart_layers.append(reference_line)

                # Label at right edge of chart
                label = (
                    alt.Chart(ref_data)
                    .mark_text(
                        align="left",
                        baseline="middle",
                        dx=5,
                        fontSize=10,
                        fontWeight="bold",
                        color=threshold_colors.get(threshold, "red")
                    )
                    .encode(
                        x=alt.X("Year:Q"),
                        y=alt.Y("Value:Q"),
                        text="Label:N"
                    )
                )
                chart_layers.append(label)

        combined_chart = alt.layer(*chart_layers).properties(
            height=450
        ).configure_view(
            strokeWidth=0
        ).configure_legend(
            padding=20,
            offset=30
        )
        st.altair_chart(combined_chart, use_container_width=True)

    # Data table
    st.subheader(f"{EMOJI['page']} Data Table")

    table_view = view.copy()

    if len(scen_sel) > 1:
        table = table_view.pivot_table(
            index=idx_cols,
            columns=["Data Type", "Scenario"],
            values="Value",
            aggfunc="first"
        ).sort_index()
    else:
        table = table_view.pivot_table(
            index=idx_cols,
            columns="Data Type",
            values="Value",
            aggfunc="first"
        ).sort_index()

    st.dataframe(table, use_container_width=True, height=400)

    csv = table.to_csv()
    st.download_button(
        label=f"{EMOJI['page']} Download as CSV",
        data=csv,
        file_name=f"climate_metrics_{type_sel}_{y0}-{y1}.csv",
        mime="text/csv",
        key=f"{ns}_download"
    )