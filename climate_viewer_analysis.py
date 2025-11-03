"""
Climate Viewer Analysis
Analysis functions for 1.5°C warming scenarios and dashboard image generation.
"""

from typing import Dict, Optional, Tuple
from io import BytesIO
import pandas as pd
import numpy as np

from climate_viewer_constants import (
    BASELINE_PERIOD,
    WARMING_TARGET,
    USE_HARDCODED_BASELINE,
    HARDCODED_AUSTRALIA_BASELINE,
    HARDCODED_GLOBAL_BASELINE,
    COLORS,
    ALL_ANALYSIS_METRICS,
)
from climate_viewer_utils import (
    calculate_preindustrial_baseline,
    find_year_at_global_warming_target,
    extract_conditions_at_year,
)


def find_15C_warming_year(
    df: pd.DataFrame,
    baseline_temp: float,
    location: str = None
) -> Tuple[Optional[int], Optional[float], pd.DataFrame]:
    """
    Find year when warming reaches 1.5°C target.
    
    Args:
        df: DataFrame with climate data
        baseline_temp: Baseline temperature for comparison
        location: Optional location filter
        
    Returns:
        Tuple of (year, warming_level, trajectory_df)
    """
    annual_temps = df[
        (df["Type"] == "Temp") &
        (df["Name"] == "Average") &
        (df["Season"] == "Annual")
    ]
    
    if location:
        annual_temps = annual_temps[annual_temps["Location"] == location]
    
    annual_temps = annual_temps.groupby("Year")["Value"].mean().reset_index()
    annual_temps["Warming"] = annual_temps["Value"] - baseline_temp
    annual_temps = annual_temps.sort_values("Year")
    
    warming_years = annual_temps[annual_temps["Warming"] >= WARMING_TARGET]
    
    if not warming_years.empty:
        year = int(warming_years.iloc[0]["Year"])
        warming = warming_years.iloc[0]["Warming"]
        return year, warming, annual_temps
    
    if not annual_temps.empty:
        max_idx = annual_temps["Warming"].idxmax()
        max_warming = annual_temps.loc[max_idx, "Warming"]
        return None, max_warming, annual_temps
    
    return None, None, pd.DataFrame()


def extract_australian_conditions(
    df: pd.DataFrame,
    year: int,
    start_year: int
) -> Dict[str, Dict[str, float]]:
    """
    Extract climate conditions for all metrics at a specific year.
    
    Args:
        df: DataFrame with climate data
        year: Target year
        start_year: Start year for calculating change
        
    Returns:
        Dictionary of conditions with changes
    """
    conditions = {}
    
    year_data = df[
        (df["Year"] == year) &
        (df["Location"] == "Australia")
    ]
    
    start_data = df[
        (df["Year"] == start_year) &
        (df["Location"] == "Australia")
    ]
    
    for metric_type, name, key, _ in ALL_ANALYSIS_METRICS:
        year_metric = year_data[
            (year_data["Type"] == metric_type) &
            (year_data["Name"] == name)
        ]
        
        start_metric = start_data[
            (start_data["Type"] == metric_type) &
            (start_data["Name"] == name)
        ]
        
        if not year_metric.empty:
            for season in ["Annual", "DJF", "MAM", "JJA", "SON"]:
                season_year = year_metric[year_metric["Season"] == season]
                season_start = start_metric[start_metric["Season"] == season]
                
                if not season_year.empty:
                    year_val = float(season_year["Value"].mean())
                    start_val = float(season_start["Value"].mean()) if not season_start.empty else np.nan
                    change = year_val - start_val if not np.isnan(start_val) else np.nan
                    
                    conditions[f"{key}_{season.lower()}"] = {
                        "value": year_val,
                        "change": change,
                        "label": f"{name} ({season})"
                    }
    
    return conditions


def analyze_15C_warming_from_data(
    scenarios: list[Tuple[str, str, str]],
    load_func,
    selected_location: str = "Australia"
) -> Dict:
    """
    Analyze when each scenario reaches 1.5°C warming.
    
    Args:
        scenarios: List of tuples (label, folder, path)
        load_func: Function to load metrics file
        selected_location: Location to analyze
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        "metadata": {
            "baseline_period": BASELINE_PERIOD,
            "warming_target": WARMING_TARGET,
            "using_hardcoded_baseline": USE_HARDCODED_BASELINE,
            "location": selected_location
        },
        "scenarios": {}
    }
    
    if USE_HARDCODED_BASELINE:
        baseline_temp = HARDCODED_AUSTRALIA_BASELINE if selected_location == "Australia" else HARDCODED_GLOBAL_BASELINE
        results["metadata"]["baseline_source"] = "Hardcoded (IPCC AR6 estimate)"
    else:
        baseline_temp = None
        for label, _, path in scenarios:
            df = load_func(path)
            baseline_temp = calculate_preindustrial_baseline(
                df, BASELINE_PERIOD, selected_location
            )
            if not np.isnan(baseline_temp):
                results["metadata"]["baseline_source"] = f"Calculated from {label}"
                break
        
        if baseline_temp is None or np.isnan(baseline_temp):
            return {
                "error": "Could not calculate baseline temperature. Set USE_HARDCODED_BASELINE = True or add historical data."
            }
    
    results["metadata"]["baseline_temperature"] = baseline_temp
    
    for label, _, path in scenarios:
        df = load_func(path)
        
        start_year = int(df["Year"].min())
        
        warming_year, warming_level, trajectory = find_15C_warming_year(
            df, baseline_temp, selected_location
        )
        
        results["scenarios"][label] = {
            "reaches_15C": warming_year is not None,
            "year": warming_year,
            "warming_level": warming_level,
            "trajectory": trajectory,
            "start_year": start_year
        }
        
        if warming_year:
            conditions = extract_australian_conditions(df, warming_year, start_year)
            results["scenarios"][label]["conditions_summary"] = conditions
            results["scenarios"][label]["years_until_15C"] = warming_year - 2025
    
    return results


def create_timeline_dataframe(results: Dict) -> pd.DataFrame:
    """
    Create timeline DataFrame showing when scenarios reach 1.5°C.
    
    Args:
        results: Analysis results dictionary
        
    Returns:
        DataFrame with timeline data
    """
    timeline_data = []
    
    for scenario, data in results.get("scenarios", {}).items():
        if data.get("reaches_15C"):
            timeline_data.append({
                "Scenario": scenario,
                "Year": data["year"],
                "Warming (°C)": round(data["warming_level"], 2),
                "Years Until": data.get("years_until_15C", "N/A")
            })
    
    if not timeline_data:
        return pd.DataFrame()
    
    return pd.DataFrame(timeline_data).sort_values("Year")


def create_dashboard_image(results: Dict, selected_location: str) -> BytesIO:
    """
    Create dashboard summary image for 1.5°C warming analysis.
    
    Args:
        results: Analysis results dictionary
        selected_location: Location name for title
        
    Returns:
        BytesIO buffer with image, or None if matplotlib not available
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return None
    
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['background'])
    
    bg_color = COLORS['panel_background']
    header_color = COLORS['header']
    warm_color = COLORS['warm']
    cool_color = COLORS['cool']
    precip_color = COLORS['precip']
    
    fig.suptitle(f'Climate Change Summary: {selected_location}',
                 fontsize=24, fontweight='bold', color=header_color, y=0.98)
    
    baseline_temp = results["metadata"]["baseline_temperature"]
    baseline_period = results["metadata"]["baseline_period"]
    
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3,
                          left=0.08, right=0.92, top=0.92, bottom=0.08)
    
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.7, f'Baseline: {baseline_period[0]}-{baseline_period[1]} ({baseline_temp:.1f}°C)',
                  ha='center', fontsize=14, style='italic', color=header_color)
    
    timeline_data = []
    temp_changes = []
    precip_changes = []
    
    for scenario, data in results["scenarios"].items():
        if data.get("reaches_15C"):
            timeline_data.append({
                'scenario': scenario,
                'year': data['year'],
                'warming': data['warming_level']
            })
            
            if "conditions_summary" in data:
                cond = data["conditions_summary"]
                
                if "tas_annual" in cond and not np.isnan(cond["tas_annual"]["change"]):
                    temp_changes.append({
                        'scenario': scenario,
                        'tas': cond["tas_annual"]["change"],
                        'days37': cond.get("days_over_37_annual", {}).get("change", 0)
                    })
                
                if "total_annual" in cond and not np.isnan(cond["total_annual"]["change"]):
                    precip_changes.append({
                        'scenario': scenario,
                        'total': cond["total_annual"]["change"],
                        'cdd': cond.get("cdd_annual", {}).get("change", 0)
                    })
    
    ax1 = fig.add_subplot(gs[1, :])
    if timeline_data:
        scenarios = [d['scenario'] for d in timeline_data]
        years = [d['year'] for d in timeline_data]
        warmings = [d['warming'] for d in timeline_data]
        
        colors_bars = [warm_color if w >= 1.5 else cool_color for w in warmings]
        bars = ax1.barh(scenarios, years, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for i, (bar, year) in enumerate(zip(bars, years)):
            ax1.text(year + 1, i, f'{year}', va='center', fontsize=11, fontweight='bold')
        
        ax1.set_xlabel('Year', fontsize=13, fontweight='bold')
        ax1.set_title('When Does Each Scenario Reach 1.5°C Warming?',
                      fontsize=15, fontweight='bold', pad=15, color=header_color)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
    else:
        ax1.text(0.5, 0.5, 'No scenarios reach 1.5°C warming',
                 ha='center', va='center', fontsize=14, style='italic')
        ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[2, 0])
    if temp_changes:
        scenarios = [d['scenario'] for d in temp_changes]
        tas = [d['tas'] for d in temp_changes]
        
        bars = ax2.bar(range(len(scenarios)), tas, color=warm_color, alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
        ax2.set_ylabel('Change (°C)', fontsize=11, fontweight='bold')
        ax2.set_title('Average Temperature Change', fontsize=13, fontweight='bold',
                      color=warm_color, pad=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        for i, (bar, val) in enumerate(zip(bars, tas)):
            ax2.text(i, val + 0.05, f'+{val:.1f}°C', ha='center', fontsize=10, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, style='italic')
        ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[2, 1])
    if temp_changes:
        scenarios = [d['scenario'] for d in temp_changes]
        days37 = [d['days37'] for d in temp_changes]
        
        bars = ax3.bar(range(len(scenarios)), days37, color=COLORS['hot'], alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax3.set_xticks(range(len(scenarios)))
        ax3.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
        ax3.set_ylabel('Change (days)', fontsize=11, fontweight='bold')
        ax3.set_title('Days Over 37°C Change', fontsize=13, fontweight='bold',
                      color=COLORS['hot'], pad=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_axisbelow(True)
        
        for i, (bar, val) in enumerate(zip(bars, days37)):
            sign = '+' if val >= 0 else ''
            ax3.text(i, val + 0.5, f'{sign}{val:.0f}', ha='center', fontsize=10, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, style='italic')
        ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[2, 2])
    if precip_changes:
        scenarios = [d['scenario'] for d in precip_changes]
        total = [d['total'] for d in precip_changes]
        
        colors_bars = [precip_color if t >= 0 else COLORS['precip_negative'] for t in total]
        bars = ax4.bar(range(len(scenarios)), total, color=colors_bars, alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax4.set_xticks(range(len(scenarios)))
        ax4.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
        ax4.set_ylabel('Change (mm)', fontsize=11, fontweight='bold')
        ax4.set_title('Total Precipitation Change', fontsize=13, fontweight='bold',
                      color=precip_color, pad=10)
        ax4.grid(axis='y', alpha=0.3, linestyle='--')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_axisbelow(True)
        
        for i, (bar, val) in enumerate(zip(bars, total)):
            sign = '+' if val >= 0 else ''
            ax4.text(i, val + (5 if val >= 0 else -5), f'{sign}{val:.0f} mm',
                     ha='center', fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, style='italic')
        ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[3, 0])
    if precip_changes:
        scenarios = [d['scenario'] for d in precip_changes]
        cdd = [d['cdd'] for d in precip_changes]
        
        bars = ax5.bar(range(len(scenarios)), cdd, color=COLORS['drought'], alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax5.set_xticks(range(len(scenarios)))
        ax5.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
        ax5.set_ylabel('Change (days)', fontsize=11, fontweight='bold')
        ax5.set_title('Consecutive Dry Days Change', fontsize=13, fontweight='bold',
                      color=COLORS['drought'], pad=10)
        ax5.grid(axis='y', alpha=0.3, linestyle='--')
        ax5.set_axisbelow(True)
        
        for i, (bar, val) in enumerate(zip(bars, cdd)):
            sign = '+' if val >= 0 else ''
            ax5.text(i, val + 0.5, f'{sign}{val:.0f}', ha='center', fontsize=10, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12, style='italic')
        ax5.axis('off')
    
    ax_footer = fig.add_subplot(gs[3, 1:])
    ax_footer.axis('off')
    disclaimer = (
        'Changes shown are relative to the start year of the scenario data.\n'
        'All metrics shown are annual averages when 1.5°C warming is reached.'
    )
    ax_footer.text(0.5, 0.5, disclaimer, ha='center', va='center',
                   fontsize=10, style='italic', color=COLORS['text_light'],
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                             edgecolor=COLORS['border'], linewidth=2))
    
    buf = BytesIO()
    plt.savefig(buf, format='jpg', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    
    return buf
