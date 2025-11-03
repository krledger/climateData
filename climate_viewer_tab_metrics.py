import pandas as pd
import altair as alt
from climate_viewer_config import IDX_COLS_ANNUAL, IDX_COLS_SEASONAL
from climate_viewer_data_operations import apply_deltas_vs_base, apply_baseline_from_start, apply_smoothing
from climate_viewer_constants import WARMING_TARGET, AMPLIFICATION_FACTOR

def render_metrics_tab(st, ns, loc_sel, scen_sel, type_sel, name_sel, season_sel, y0, y1,
                       mode, smooth, smooth_win, table_interval, show_15c_shading, show_horizon_shading,
                       short_start, mid_start, long_start, horizon_end,
                       df_all, base_df, preindustrial_baseline, BASE_LABEL):
    st.title("🌡️ Climate Metrics Viewer")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🗺️ Locations", len(loc_sel))
    with col2:
        st.metric("🌡️ Scenarios", len(scen_sel))
    with col3:
        st.metric("🗓️ Year Range", f"{y0}-{y1}")
    with col4:
        st.metric("📈 Mode", mode.split("(")[0].strip())

    st.caption(
        f"Locations: {', '.join(loc_sel)} - "
        f"Scenarios: {', '.join(scen_sel)} - "
        f"Type: {type_sel} - "
        f"Metrics: {', '.join(name_sel)}"
    )

    mask = (
            df_all["Year"].between(y0, y1) &
            df_all["Location"].isin(loc_sel) &
            df_all["Season"].isin(season_sel) &
            (df_all["Type"] == type_sel) &
            df_all["Name"].isin(name_sel)
    )
    view = df_all[mask].copy()

    use_baseline = mode.startswith("Baseline")
    apply_delta = mode.startswith("Deltas")

    if apply_delta:
        base_mask = (
                base_df["Year"].between(y0, y1) &
                base_df["Location"].isin(loc_sel) &
                base_df["Season"].isin(season_sel) &
                (base_df["Type"] == type_sel) &
                base_df["Name"].isin(name_sel)
        )
        base_filtered = base_df[base_mask].copy()

        view = apply_deltas_vs_base(view, base_filtered)

        if smooth:
            view = apply_smoothing(view, smooth_win)
    elif use_baseline:
        if smooth:
            view = apply_smoothing(view, smooth_win)
        view = apply_baseline_from_start(view)
    else:
        if smooth:
            view = apply_smoothing(view, smooth_win)

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

    warming_reference = None
    if type_sel == "Temp" and "Average" in name_sel and show_15c_shading:
        if preindustrial_baseline and not use_baseline and not apply_delta:
            # When GLOBAL warming reaches target, regional warming is amplified
            # Regional warming = Global warming Ã— Amplification factor
            # Regional temp at global target = Regional_baseline + (WARMING_TARGET Ã— AMPLIFICATION_FACTOR)
            REGIONAL_WARMING_AT_GLOBAL_TARGET = WARMING_TARGET * AMPLIFICATION_FACTOR  # = 1.724Â°C
            warming_reference = {loc: temp + REGIONAL_WARMING_AT_GLOBAL_TARGET for loc, temp in preindustrial_baseline.items()}
    with st.expander("📊 Summary Statistics", expanded=False):
        summary = view.groupby(["Scenario", "Name"])["Value"].agg([
            ("Mean", "mean"),
            ("Min", "min"),
            ("Max", "max"),
            ("Change", lambda x: x.max() - x.min())
        ]).round(2)
        st.dataframe(summary, width="stretch")

    st.subheader("📈 Visualisation")

    for location in loc_sel:
        st.markdown(f"📍 {location}")

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
                title="Year-",
                axis=alt.Axis(labelFontSize=12, titleFontSize=14, labelAngle=-45),
                sort=list(plot_data["X"].unique())
            )

        selection = alt.selection_point(fields=["Metric", "Scenario"], bind="legend")

        chart_layers = []

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

        chart = (
            alt.Chart(plot_data)
            .mark_line(point=True, strokeWidth=2.5)
            .encode(
                x=x_encoding,
                y=alt.Y(
                    "Value:Q",
                    title="Value",
                    scale=alt.Scale(zero=False),
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

        if warming_reference and location in warming_reference:
            reference_temp = warming_reference[location]
            reference_line = (
                alt.Chart(pd.DataFrame({"temp": [reference_temp]}))
                .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=2)
                .encode(
                    y=alt.Y("temp:Q"),
                    tooltip=alt.value(f"🎯 1.5°C global warming target")
                )
            )
            chart_layers.append(reference_line)

        combined_chart = alt.layer(*chart_layers).properties(height=450).interactive()
        st.altair_chart(combined_chart, use_container_width=True)

    st.subheader("📋 Data Table")

    table_view = view.copy()
    if table_interval > 1:
        anchor = int(table_view["Year"].min())
        table_view = table_view[
            ((table_view["Year"].astype(int) - anchor) % table_interval) == 0
            ]

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

    st.dataframe(table, width="stretch", height=400)

    csv = table.to_csv()
    st.download_button(
        label="📄 Download as CSV",
        data=csv,
        file_name=f"climate_metrics_{type_sel}_{y0}-{y1}.csv",
        mime="text/csv",
        key=f"{ns}_download"
    )
