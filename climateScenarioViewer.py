#!/usr/bin/env python3
"""
Climate Metrics Viewer (Streamlit)
A streamlined single-file app for browsing climate metrics with charts and tables.
Supports various metric types including Temperature, Rain, Wind, and Humidity.
"""

import os
import sys
import re
from typing import Sequence, Tuple, Literal

import pandas as pd

# ============================== CONFIG ========================================
MODE = "metrics"
PORT = 8501

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
FOLDER = os.path.join(BASE_DIR, "metricsDataFiles")

GROUP_KEYS = ["Location", "Type", "Name", "Season", "Data Type"]
IDX_COLS_ANNUAL = ["Year"]
IDX_COLS_SEASONAL = ["Year", "Season"]


# ============================ HELPERS =========================================
def resolve_folder() -> str:
    """Validate metrics folder exists."""
    if not os.path.isdir(FOLDER):
        sys.exit(f"Invalid folder (metrics root): {FOLDER}")
    return FOLDER


def dedupe_preserve_order(items):
    """Remove duplicates while preserving order."""
    seen, out = set(), []
    for x in items or []:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def slugify(s: str) -> str:
    """Convert string to URL-safe slug."""
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def parse_data_type(series: pd.Series) -> pd.DataFrame:
    """Extract Type, Name, Location from 'Data Type' column."""
    return series.str.extract(
        r"^(?P<Type>[^ ]+) \((?P<Name>[^,]+), (?P<Location>[^)]+)\)$"
    )


# ============================ UI COMPONENTS ===================================
def multi_selector(
        container,
        label: str,
        options,
        default=None,
        widget: Literal["checkbox", "toggle"] = "checkbox",
        columns: int = 1,
        key_prefix: str = "sel",
        namespace: str = None
):
    """Render multi-selection UI with checkboxes or toggles."""
    opts = dedupe_preserve_order(options or [])
    default = set(default or [])
    selected = []

    container.markdown(f"**{label}**")
    cols = container.columns(columns)
    widget_fn = container.checkbox if widget == "checkbox" else container.toggle

    for i, opt in enumerate(opts):
        key = f"{namespace + '_' if namespace else ''}{key_prefix}_{i}_{slugify(opt)}"
        with cols[i % columns]:
            if widget_fn(str(opt), value=(opt in default), key=key):
                selected.append(opt)
    return selected


# ============================ DATA TRANSFORMS =================================
def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist with proper types."""
    required = {
        "Year": "Int64",
        "Season": str,
        "Data Type": str,
        "Value": float,
        "Location": str,
        "Name": str,
        "Type": str,
        "Scenario": str
    }

    df = df.copy()

    # Add missing columns
    if "Season" not in df.columns:
        df["Season"] = "Annual"

    for col, dtype in required.items():
        if col not in df.columns:
            df[col] = pd.NA
        if dtype == str:
            df[col] = df[col].astype(str).str.strip()
        elif dtype == float:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif dtype == "Int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def apply_deltas_vs_base(view: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Calculate deltas by subtracting base scenario values."""
    join_keys = ["Year", "Season", "Data Type", "Location", "Type", "Name"]
    base_values = base[join_keys + ["Value"]].rename(columns={"Value": "BaseValue"})

    merged = view.merge(base_values, on=join_keys, how="left")
    merged["Value"] = merged["Value"] - merged["BaseValue"]

    return merged.drop(columns=["BaseValue"])


def apply_baseline_from_start(view: pd.DataFrame) -> pd.DataFrame:
    """Subtract first year's value as baseline per group."""
    if view.empty:
        return view

    keys = ["Scenario"] + GROUP_KEYS

    # Get first year for each group
    first_year = view.groupby(keys, as_index=False)["Year"].min()
    first_year = first_year.rename(columns={"Year": "FirstYear"})

    # Get baseline values (first year values)
    baseline = view.merge(first_year, on=keys, how="left")
    baseline = baseline[baseline["Year"] == baseline["FirstYear"]]
    baseline = baseline[keys + ["Value"]].rename(columns={"Value": "Baseline"})

    # Apply baseline subtraction
    result = view.merge(baseline, on=keys, how="left")
    result["Value"] = result["Value"] - result["Baseline"]

    return result.drop(columns=["Baseline"])


def apply_smoothing(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Apply rolling mean smoothing per group."""
    if window <= 1 or df.empty:
        return df

    # Ensure odd window
    window = window if window % 2 == 1 else window + 1
    min_periods = max(1, window // 2)

    group_cols = ["Scenario"] + GROUP_KEYS

    smoothed_groups = []
    for _, group in df.groupby(group_cols, dropna=False):
        group = group.sort_values("Year").copy()
        group["Value"] = group["Value"].rolling(
            window, center=True, min_periods=min_periods
        ).mean()
        smoothed_groups.append(group)

    result = pd.concat(smoothed_groups, ignore_index=True)
    return result.dropna(subset=["Value"])


# ============================ DATA LOADING ====================================
def discover_scenarios(base_folder: str) -> list[Tuple[str, str, str]]:
    """Find all scenario folders with metrics parquet files."""
    scenarios = []

    try:
        for name in sorted(os.listdir(base_folder)):
            path = os.path.join(base_folder, name)
            if not os.path.isdir(path):
                continue

            parquet_files = [
                f for f in os.listdir(path)
                if f.startswith("metrics") and f.endswith(".parquet")
            ]

            if parquet_files:
                parquet_path = os.path.join(path, sorted(parquet_files)[0])
                scenarios.append((name, path, parquet_path))
    except Exception as e:
        sys.exit(f"Error discovering scenarios: {e}")

    return scenarios


def load_metrics_file(path: str) -> pd.DataFrame:
    """Load and prepare a single metrics parquet file."""
    try:
        df = pd.read_parquet(path, engine="pyarrow")

        # Parse Data Type column
        parts = parse_data_type(df["Data Type"])
        df = pd.concat([df, parts], axis=1)

        # Apply schema
        df = ensure_schema(df)

        return df.dropna(subset=["Year"])
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")


# ============================ MAIN APP ========================================
def run_metrics_viewer() -> None:
    """Main Streamlit application."""
    try:
        import altair as alt
        import streamlit as st
    except ImportError:
        sys.exit("Missing dependencies. Run: pip install streamlit pandas pyarrow altair")

    # ============================= DISCLAIMER ================================
    if "disclaimer_accepted" not in st.session_state:
        st.session_state.disclaimer_accepted = False

    layout = "centered" if not st.session_state.disclaimer_accepted else "wide"
    st.set_page_config(page_title="Climate Metrics Viewer", layout=layout)

    if not st.session_state.disclaimer_accepted:
        st.title("🌡️ Climate Metrics Viewer")
        st.markdown("---")

        st.subheader("Data Disclaimer")
        st.markdown("""
By selecting "Accept," you acknowledge that the climate data provided through this viewer has not been independently verified and that you use any information displayed entirely at your own risk. This tool is intended for general informational and educational purposes only and should not be used as the sole basis for any decision-making. Climate projections and historical data may contain uncertainties, errors, or limitations. You are solely responsible for verifying any data before use and for determining whether this information is appropriate for your specific needs and requirements. The creators and providers of this viewer make no representations or warranties of any kind, express or implied, regarding the accuracy, completeness, timeliness, or reliability of the data presented. Under no circumstances shall the creators or providers be liable for any decisions made or actions taken based on information from this viewer.
        """)

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Accept", key="accept_disclaimer", type="primary", use_container_width=True):
                st.session_state.disclaimer_accepted = True
                st.rerun()

        st.stop()

    # ============================= DATA SETUP ================================
    @st.cache_data(show_spinner=False)
    def load_metrics_cached(path: str, mtime: float) -> pd.DataFrame:
        """Cached metrics loader."""
        _ = mtime  # Force cache invalidation on file changes
        return load_metrics_file(path)

    @st.cache_data(show_spinner=False)
    def load_minimal_metadata(
            pairs: Sequence[Tuple[str, str, float]]
    ) -> pd.DataFrame:
        """Load minimal columns for building filters."""
        frames = []
        for label, path, mtime in pairs:
            _ = mtime
            df = pd.read_parquet(
                path, engine="pyarrow",
                columns=["Year", "Season", "Data Type"]
            )
            parts = parse_data_type(df["Data Type"])
            df = pd.concat([df.drop(columns=["Data Type"]), parts], axis=1)
            df["Scenario"] = label
            frames.append(df)

        result = pd.concat(frames, ignore_index=True)
        result = ensure_schema(result)
        return result.dropna(subset=["Year"])

    # Discover scenarios
    base_folder = resolve_folder()
    scenarios = discover_scenarios(base_folder)

    if not scenarios:
        st.error(f"❌ No scenarios found in: {base_folder}")
        st.stop()

    labels = [lbl for lbl, _, _ in scenarios]
    label_to_path = {lbl: path for lbl, _, path in scenarios}
    BASE_LABEL = "SSP1-26" if "SSP1-26" in labels else labels[0]

    # Load minimal metadata for filters
    metadata_tuples = [
        (lbl, label_to_path[lbl], os.path.getmtime(label_to_path[lbl]))
        for lbl in labels
    ]
    metadata = load_minimal_metadata(metadata_tuples)

    # ============================= SIDEBAR ===================================
    ns = "metrics"

    st.sidebar.title("⚙️ Filters")

    # Location filter
    with st.sidebar.expander("📍 Locations", expanded=True):
        all_locations = sorted(metadata["Location"].dropna().unique())
        default_locs = ["Ravenswood"] if "Ravenswood" in all_locations else all_locations[:1]

        loc_sel = multi_selector(
            st, "Select locations", all_locations,
            default=default_locs, columns=1, namespace=ns, key_prefix="loc"
        )

    if not loc_sel:
        st.sidebar.warning("⚠️ Select at least one location")
        st.stop()

    # Scenario filter
    with st.sidebar.expander("🌍 Scenarios", expanded=True):
        default_scen = [BASE_LABEL] if BASE_LABEL in labels else [labels[0]]
        scen_sel = multi_selector(
            st, "Select scenarios", labels,
            default=default_scen, columns=1, namespace=ns, key_prefix="scen"
        )

    if not scen_sel:
        st.sidebar.warning("⚠️ Select at least one scenario")
        st.stop()

    # Display options
    with st.sidebar.expander("🎨 Display Options", expanded=True):
        mode = st.radio(
            "Display Mode",
            ["Values", "Baseline (start year)", f"Deltas vs {BASE_LABEL}"],
            index=0, key=f"{ns}_mode"
        )

        smooth = st.toggle("Smooth Values", value=False, key=f"{ns}_smooth")
        if smooth:
            smooth_win = st.slider(
                "Smoothing Window (years)",
                3, 21, step=2, value=9, key=f"{ns}_smooth_win"
            )
        else:
            smooth_win = 1

        table_interval = st.radio(
            "Table Interval (years)",
            [1, 2, 5, 10],
            index=1, horizontal=True, key=f"{ns}_table_int"
        )

    # Year range
    with st.sidebar.expander("📆 Year Range", expanded=True):
        yr_min, yr_max = int(metadata["Year"].min()), int(metadata["Year"].max())
        y0, y1 = st.select_slider(
            "Select range",
            options=list(range(yr_min, yr_max + 1)),
            value=(yr_min, yr_max),
            key=f"{ns}_year"
        )

    # Metric type filter
    with st.sidebar.expander("📊 Metric Type", expanded=False):
        preferred_types = ["Temp", "Rain", "Wind", "Humidity"]
        all_types = list(metadata["Type"].unique())
        type_options = [t for t in preferred_types if t in all_types] + \
                       [t for t in sorted(all_types) if t not in preferred_types]
        default_type = "Temp" if "Temp" in type_options else type_options[0]

        type_sel = st.radio(
            "Type", type_options,
            index=type_options.index(default_type),
            horizontal=True, key=f"{ns}_type"
        )

    # Metric name filter (depends on type and location)
    with st.sidebar.expander("📈 Metric Names", expanded=False):
        filtered_meta = metadata[
            (metadata["Location"].isin(loc_sel)) &
            (metadata["Type"] == type_sel)
            ]
        name_options = sorted(filtered_meta["Name"].dropna().unique())

        # Smart sorting based on type
        if type_sel == "Temp":
            priority = ["Average", "Max", "Max Day", "5-Day Avg Max", "Avg Max"]
            name_options = sorted(
                name_options,
                key=lambda n: (priority.index(n) if n in priority else 99, n)
            )
            default_names = ["Average"] if "Average" in name_options else name_options[:1]
        elif type_sel == "Humidity":
            priority = ["Average RH", "Average VPD"]
            name_options = sorted(
                name_options,
                key=lambda n: (priority.index(n) if n in priority else 99, n)
            )
            default_names = name_options[:1]
        else:
            default_names = name_options[:1]

        name_sel = multi_selector(
            st, "Select metrics", name_options,
            default=default_names, widget="toggle",
            columns=1, namespace=ns, key_prefix="name"
        )

    if not name_sel:
        st.sidebar.warning("⚠️ Select at least one metric")
        st.stop()

    # Season filter
    with st.sidebar.expander("📅 Seasons", expanded=False):
        seasons_all = ["Annual", "DJF", "MAM", "JJA", "SON"]
        available_seasons = [s for s in seasons_all if s in metadata["Season"].unique()]
        default_seasons = ["Annual"] if "Annual" in available_seasons else available_seasons

        season_sel = multi_selector(
            st, "Select seasons", available_seasons,
            default=default_seasons, columns=1,
            namespace=ns, key_prefix="season"
        )

    if not season_sel:
        st.sidebar.warning("⚠️ Select at least one season")
        st.stop()

    # Help section
    with st.sidebar.expander("ℹ️ Help"):
        st.markdown("""
        **Smoothing**: Applies a rolling average to reduce noise in the data.

        **Baseline**: Shows change from the first year in the selected range.

        **Deltas**: Shows difference from the reference scenario (SSP1-26).

        **Table Interval**: Controls row spacing in the values table.
        """)

    # ============================= MAIN CONTENT ==============================
    st.title("🌡️ Climate Metrics Viewer")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Locations", len(loc_sel))
    with col2:
        st.metric("🌍 Scenarios", len(scen_sel))
    with col3:
        st.metric("📅 Year Range", f"{y0}–{y1}")
    with col4:
        st.metric("📊 Mode", mode.split("(")[0].strip())

    st.caption(
        f"Locations: {', '.join(loc_sel)} • "
        f"Scenarios: {', '.join(scen_sel)} • "
        f"Type: {type_sel} • "
        f"Metrics: {', '.join(name_sel)}"
    )

    # Load full data for selected scenarios
    with st.spinner("Loading data..."):
        dfs = []
        for label in scen_sel:
            path = label_to_path[label]
            df = load_metrics_cached(path, os.path.getmtime(path))
            df["Scenario"] = label
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)

        # Load base scenario for delta calculation
        base_path = label_to_path[BASE_LABEL]
        base_df = load_metrics_cached(base_path, os.path.getmtime(base_path))
        base_df["Scenario"] = BASE_LABEL

    # Filter data
    mask = (
            df_all["Year"].between(y0, y1) &
            df_all["Location"].isin(loc_sel) &
            df_all["Season"].isin(season_sel) &
            (df_all["Type"] == type_sel) &
            df_all["Name"].isin(name_sel)
    )
    view = df_all[mask].copy()

    # Apply transformations
    use_baseline = mode.startswith("Baseline")
    apply_delta = mode.startswith("Deltas")

    if use_baseline:
        view = apply_baseline_from_start(view)
    elif apply_delta:
        view = apply_deltas_vs_base(view, base_df)

    if smooth:
        view = apply_smoothing(view, smooth_win)

    view = view.dropna(subset=["Value"])

    # Check for empty results
    if view.empty:
        st.warning("⚠️ No data matches your current filters. Try:")
        st.markdown("""
        - Expanding the year range
        - Selecting additional scenarios or locations
        - Choosing different metric names
        """)
        st.stop()

    # Determine index columns for pivot
    use_seasonal = len(season_sel) > 1 or (
            len(season_sel) == 1 and season_sel[0] != "Annual"
    )
    idx_cols = IDX_COLS_SEASONAL if use_seasonal else IDX_COLS_ANNUAL

    # ============================= SUMMARY STATISTICS ========================
    with st.expander("📊 Summary Statistics", expanded=False):
        summary = view.groupby(["Scenario", "Name"])["Value"].agg([
            ("Mean", "mean"),
            ("Min", "min"),
            ("Max", "max"),
            ("Change", lambda x: x.max() - x.min())
        ]).round(2)
        st.dataframe(summary, use_container_width=True)

    # ============================= CHART =====================================
    st.subheader("📈 Visualization")

    plot_data = view[idx_cols + ["Data Type", "Value", "Location", "Name", "Scenario"]].copy()
    plot_data = plot_data.rename(columns={"Data Type": "Metric"})

    # Prepare X-axis
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
            title="Year–Season",
            axis=alt.Axis(labelFontSize=12, titleFontSize=14, labelAngle=-45),
            sort=list(plot_data["X"].unique())
        )

    # Interactive selection
    selection = alt.selection_point(fields=["Metric", "Scenario"], bind="legend")

    # Build chart
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
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    # ============================= TABLE =====================================
    st.subheader("📋 Data Table")

    # Apply table interval filtering
    table_view = view.copy()
    if table_interval > 1:
        anchor = int(table_view["Year"].min())
        table_view = table_view[
            ((table_view["Year"].astype(int) - anchor) % table_interval) == 0
            ]

    # Create pivot table
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

    # Download button
    csv = table.to_csv()
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"climate_metrics_{type_sel}_{y0}-{y1}.csv",
        mime="text/csv",
        key=f"{ns}_download"
    )


# ================================== MAIN ======================================
if __name__ == "__main__":
    if MODE.lower() == "metrics":
        run_metrics_viewer()
    else:
        sys.exit('Set MODE = "metrics"')