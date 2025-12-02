#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate Viewer User Guide
==========================

Comprehensive scientific documentation for the Climate Metrics Viewer.
Uses collapsible sections to balance accessibility with technical depth.
"""

import streamlit as st


def render_user_guide():
    """
    Render comprehensive user guide with collapsible sections.

    Provides layered information:
    - Basic information visible by default
    - Detailed technical explanations in expandable sections
    - Comprehensive coverage of methodologies and calculations
    """

    st.markdown("""
# 📖 Climate Metrics Viewer - Scientific User Guide

*Version 2.0 | Last Updated: 2025-11-08*

---

## 🎯 Overview

This application provides comprehensive climate projection analysis using CMIP6 data combined with 
historical observations.  The tool is designed for climate risk assessment, infrastructure planning 
and compliance with AASB S2 climate-related financial disclosure requirements.

**Key Features:**
- Historical observations from Bureau of Meteorology (AGCD)
- Future projections from CMIP6 ACCESS-CM2 climate model
- Multiple emissions scenarios (SSP1-2.6 through SSP5-8.5)
- Bias-corrected rainfall metrics for improved accuracy
- Pre-industrial baseline comparisons
- 1.5°C global warming threshold analysis
""")

    # ========================================================================
    # DATA SOURCES SECTION
    # ========================================================================

    st.markdown("---")
    st.markdown("## 📊 Data Sources and Quality")

    st.markdown("""
This application uses two primary data sources that complement each other: 
observational data for validation and climate model projections for future scenarios.
""")

    with st.expander("**Historical Observations: AGCD (Australian Gridded Climate Data)**", expanded=False):
        st.markdown("""
### Provider and Coverage
**Australian Gridded Climate Data** is maintained by the Bureau of Meteorology and provides 
high-quality gridded climate observations across Australia.

**Specifications:**
- **Provider:** Bureau of Meteorology (BOM)
- **Spatial Resolution:** 0.05° (~5 km grid cells)
- **Temporal Coverage:** 1900-present (varies by variable)
- **Update Frequency:** Daily, then aggregated to monthly and annual
- **Quality Control:** Rigorous station network quality assurance
- **Best Quality Period:** 1970-present (dense station network)

### Data Processing
1. **Station Data Collection:** Raw observations from BOM weather station network
2. **Quality Control:** Outlier detection, homogeneity testing, consistency checks
3. **Spatial Interpolation:** Barnes successive correction scheme
4. **Grid Generation:** Regular 0.05° latitude-longitude grid
5. **Validation:** Cross-validation against withheld stations

### Limitations and Considerations
- **Early Period (pre-1960):** Sparse station network reduces accuracy
- **Regional Variability:** Quality correlates with station density
- **Interpolation Effects:** Smooth fields may not capture local extremes
- **Spatial Scale:** Not suitable for sub-5 km analysis
- **Variable Coverage:** Wind and humidity have shorter records than temperature and rainfall

### Reference
[http://www.bom.gov.au/climate/data/](http://www.bom.gov.au/climate/data/)

Jones, D.A., Wang, W. and Fawcett, R. (2009), High-quality spatial climate data-sets for 
Australia, Australian Meteorological and Oceanographic Journal, 58, 233-248.
""")

    with st.expander("**Climate Projections: CMIP6 ACCESS-CM2**", expanded=False):
        st.markdown("""
### Model Description
**ACCESS-CM2** (Australian Community Climate and Earth-System Simulator - Coupled Model version 2) 
is Australia's global climate model contribution to CMIP6.

**Model Specifications:**
- **Institution:** CSIRO and Bureau of Meteorology (Australia)
- **Model Family:** ACCESS (Australian Community Climate and Earth-System Simulator)
- **Resolution:** ~250 km horizontal (~1.25° × 1.875° latitude-longitude)
- **Vertical Levels:** 85 atmospheric levels, 50 ocean levels
- **Temporal Coverage:** 1850-2100
- **Ensemble Member:** r1i1p1f1 (realisation 1, initialisation 1, physics 1, forcing 1)

### Scenarios Included

**Historical (1850-2014):**
- Model simulation of past climate using observed forcings
- Greenhouse gases, aerosols, solar variability, volcanic eruptions
- Used for model validation and bias correction calculations
- Diverges from SSP scenarios at 2015

**SSP1-2.6 (Sustainability - Low Emissions):**
- CO₂ emissions decline to net zero by 2075
- Strong mitigation, sustainable development
- Likely warming: ~1.8°C by 2100 (relative to 1850-1900)
- Paris Agreement target pathway

**SSP2-4.5 (Middle of the Road - Intermediate Emissions):**
- Current policies continue, gradual improvements
- Moderate mitigation and adaptation challenges
- Likely warming: ~2.7°C by 2100
- Reference scenario for many assessments

**SSP3-7.0 (Regional Rivalry - High Emissions):**
- Resurgent nationalism, regional focus
- High mitigation and adaptation challenges
- Likely warming: ~3.6°C by 2100
- Scenario of concern for risk assessment

**SSP5-8.5 (Fossil-Fueled Development - Very High Emissions):**
- Rapid economic growth based on fossil fuels
- High mitigation challenge, low adaptation challenge
- Likely warming: ~4.4°C by 2100
- High-end scenario for infrastructure planning

### Model Physics and Components

**Atmospheric Component (UM-GA7.1):**
- Unified Model Global Atmosphere configuration 7.1
- Sophisticated cloud microphysics
- Convection parameterisation (mass-flux scheme)
- Boundary layer turbulence
- Radiation transfer (Edwards-Slingo)

**Ocean Component (ACCESS-OM2):**
- MOM5 (Modular Ocean Model version 5)
- Tripolar grid (1° resolution)
- Sea ice model (CICE5.1)
- Ocean biogeochemistry (WOMBAT)

**Land Surface (CABLE):**
- Community Atmosphere Biosphere Land Exchange
- Soil moisture, vegetation dynamics
- Carbon and nitrogen cycles
- Fire processes

### Important Considerations

**Single Model Limitation:**
This application uses a single climate model (ACCESS-CM2) and single ensemble member 
(r1i1p1f1).  A comprehensive assessment would typically use:
- Multiple models (multi-model ensemble)
- Multiple ensemble members (quantifying internal variability)
- Pattern scaling or other techniques

**Coarse Resolution Effects:**
At ~250 km resolution, the model:
- Averages out local topographic effects
- Underestimates extreme rainfall intensities
- Cannot resolve mesoscale phenomena (thunderstorms, sea breezes)
- Smooths coastal gradients

**Systematic Biases:**
All climate models contain systematic biases due to:
- Simplified physics (parameterisations)
- Numerical approximations
- Incomplete process representation
- Resolution limitations

*These biases are addressed through bias correction (see Bias Correction section).*

### Data Access
- **CMIP6 Archive:** [https://esgf-node.llnl.gov/projects/cmip6/](https://esgf-node.llnl.gov/projects/cmip6/)
- **Copernicus Climate Data Store:** [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
- **ACCESS Model Information:** [https://www.cawcr.gov.au/research/ACCESS](https://www.cawcr.gov.au/research/ACCESS)

### References
Bi, D., et al. (2020). Configuration and spin-up of ACCESS-CM2, the new generation 
Australian Community Climate and Earth System Simulator Coupled Model. Journal of 
Southern Hemisphere Earth Systems Science, 70(1), 225-251.
""")

    with st.expander("**Data Processing and Extraction**", expanded=False):
        st.markdown("""
### Spatial Extraction Method

**Point Extraction:**
Climate model data is extracted for the grid cell containing each location of interest.  
This is a standard approach for site-specific climate analysis.

**Process:**
1. Identify target coordinates (latitude, longitude)
2. Find nearest grid cell centre in model output
3. Extract all timesteps for that grid cell
4. Apply land-sea mask (exclude ocean cells)
5. Perform unit conversions and aggregations

**Land-Sea Masking:**
Following IPCC protocols, ocean grid cells are excluded from regional land statistics:
- Uses CMIP6 model's native land fraction (sftlf variable)
- Cells with land fraction <50% are excluded
- Prevents contamination of land climate statistics with ocean values
- Critical for accurate regional temperature and precipitation trends

### Temporal Processing

**Daily to Annual:**
1. **Quality Control:** Check for missing values, outliers
2. **Daily Aggregation:** Calculate appropriate metrics (see Metrics section)
3. **Seasonal Aggregation:** 3-month periods following Southern Hemisphere definitions
4. **Annual Aggregation:** Calendar year (January-December)

**Seasonal Definitions (Southern Hemisphere):**
- **DJF (Summer):** December, January, February
- **MAM (Autumn):** March, April, May
- **JJA (Winter):** June, July, August
- **SON (Spring):** September, October, November

*Note: December is assigned to the following year (e.g. Dec 2020 → DJF 2021)*

### Unit Conversions

**Temperature:**
- Model output: Kelvin (K)
- Application display: Celsius (°C)
- Conversion: T(°C) = T(K) - 273.15

**Precipitation:**
- Model output: kg m⁻² s⁻¹ (mass flux)
- Application display: mm (millimetres)
- Conversion: mm/day = kg m⁻² s⁻¹ × 86400 s/day
- Physical meaning: 1 mm = 1 litre per square metre

**Wind Speed:**
- Model output: m s⁻¹
- Application display: m/s (metres per second)
- Conversion to km/h: km/h = m/s × 3.6
""")

    # ========================================================================
    # CLIMATE METRICS SECTION
    # ========================================================================

    st.markdown("---")
    st.markdown("## 🌡️ Climate Metrics and Calculations")

    st.markdown("""
All metrics are calculated from daily data using standard climate indices methodologies 
following ETCCDI (Expert Team on Climate Change Detection and Indices) protocols where applicable.
""")

    with st.expander("**Temperature Metrics**", expanded=False):
        st.markdown("""
### Average Temperature (Mean Temperature)

**Definition:** Mean daily temperature

**Calculation:**
```
T_mean = (T_max + T_min) / 2
```

Where:
- T_max = Daily maximum temperature (°C)
- T_min = Daily minimum temperature (°C)

**Aggregation:**
- Daily values averaged for monthly/seasonal/annual periods
- Area-weighted when combining grid cells

**Physical Meaning:**
Represents the overall thermal state of the atmosphere.  Best indicator of climate warming trends.

**Typical Values (Australia):**
- Historical (1850-1900): ~21.7°C annual mean
- Current (2015): ~23.0°C annual mean
- SSP5-8.5 (2100): ~26-28°C annual mean

**Uncertainty:**
- Observational: ±0.3°C
- Model systematic bias: ±0.5-1.0°C (correctable)
- Future projection spread: ±0.5-1.5°C (depending on scenario and timeframe)

---

### Maximum Temperature (Tmax)

**Definition:** Daily maximum temperature

**Metrics Available:**
1. **Max Day:** Highest daily maximum in period
2. **Avg Max:** Average of all daily maxima
3. **5-Day Avg Max:** Highest 5-day running mean of daily maxima

**Use Cases:**
- Heat stress assessment
- Infrastructure design temperatures
- Energy demand (cooling)
- Agricultural heat stress
- Fire danger

**Model Characteristics:**
- Models typically underestimate extreme heat by 1-2°C
- Better representation of mean maximum than individual extremes
- Bias correction improves accuracy

---

### Temperature Threshold Exceedances

**Days ≥37°C ≥3h (estimated):**

**Definition:** Estimated number of days where temperature exceeds 37°C for at least 3 hours

**Calculation Method:**
Uses daily maximum temperature as proxy:
1. Count days where T_max ≥ 37°C
2. Apply empirical correction factor (based on hourly data analysis)
3. Conservative estimate (actual hours may differ)

**Significance:**
- 37°C wet bulb: Critical heat stress threshold for humans
- Mining operations: Heat stress management trigger
- Occupational health and safety threshold
- Agricultural productivity impacts

**Days ≥40°C ≥3h (estimated):**

**Definition:** Estimated number of days where temperature exceeds 40°C for at least 3 hours

**Significance:**
- Extreme heat threshold
- Infrastructure stress (rails, roads, power lines)
- Severe health impacts
- Productivity severe limitations
- Emergency management activation

**Limitations:**
- Estimates based on daily maximum (not hourly data)
- Does not account for humidity effects
- Conservative approach to critical thresholds
""")

    with st.expander("**Rainfall/Precipitation Metrics**", expanded=False):
        st.markdown("""
### Overview of Rainfall Metrics

Rainfall metrics are provided in **two versions** for SSP scenarios:
1. **Raw model output** (e.g. "Total")
2. **Bias-corrected** (e.g. "Total (BC)")

**Recommendation:** Use bias-corrected metrics for SSP scenarios.  Bias correction significantly 
improves accuracy by accounting for systematic model errors.

---

### Total Rainfall

**Definition:** Sum of daily precipitation over the period

**Calculation:**
```
Total = Σ(daily_precipitation)
```

**Units:** mm (millimetres)

**Seasonal Patterns (Ravenswood region):**
- **DJF (Summer):** Wet season, 60-70% of annual rainfall
- **MAM (Autumn):** Transition, moderate rainfall
- **JJA (Winter):** Dry season, 5-10% of annual rainfall
- **SON (Spring):** Transition, increasing rainfall

**Bias Correction:**
Raw ACCESS-CM2 output shows systematic errors:
- **Underestimates wet season** (DJF): Model gives ~44% less than observed
- **Overestimates dry season** (JJA, MAM): Model gives ~13-19% more than observed

*See Bias Correction section for detailed methodology.*

**Typical Annual Values:**
- Historical (1850-1900): ~600-700 mm
- Current observations (2000-2020): Variable, 500-900 mm
- Future projections: High uncertainty, scenario-dependent

---

### Extreme Rainfall Metrics

**Max Day (Maximum Daily Rainfall):**

**Definition:** Highest daily precipitation total in the period

**Significance:**
- Flash flood potential
- Drainage system capacity
- Erosion risk
- Infrastructure design criteria

**Model Limitations:**
Coarse resolution (250 km) significantly underestimates intense rainfall:
- Cannot resolve convective storms
- Smooths out peak intensities
- Typical underestimation: 30-50% for extreme events

**Bias Correction:** Applied to improve accuracy

---

**Max 5-Day (Maximum 5-Day Cumulative Rainfall):**

**Definition:** Highest 5-day running sum of daily precipitation

**Calculation:**
```
For each day i:
    Sum_5day[i] = Σ(precipitation[i to i+4])
Max_5day = maximum(Sum_5day)
```

**Significance:**
- Sustained heavy rainfall events
- River flooding potential
- Soil saturation
- Landslide risk
- Multi-day accumulation impacts

**Physical Context:**
5-day accumulations better captured by models than single-day extremes due to 
spatial and temporal averaging.

---

### Wet Day Frequency Metrics

**R10mm (Days with Rainfall ≥10 mm):**

**Definition:** Count of days with precipitation ≥10 mm

**Significance:**
- Moderate rainfall events
- Effective precipitation for agriculture
- Runoff generation threshold
- General wetness indicator

**Typical Values:**
- Historical: 15-25 days per year
- High uncertainty in projections

---

**R20mm (Days with Rainfall ≥20 mm):**

**Definition:** Count of days with precipitation ≥20 mm

**Significance:**
- Heavy rainfall days
- Flood risk indicator
- Erosion potential
- Infrastructure stress events

**Typical Values:**
- Historical: 5-12 days per year
- Generally increasing in projections (more intense events)

---

### Dry Spell Metrics

**CDD (Consecutive Dry Days):**

**Definition:** Maximum length of dry spell (consecutive days with precipitation <1 mm)

**Calculation:**
```
For each day:
    If precipitation < 1 mm:
        Increment current dry spell length
    Else:
        Record dry spell length
        Reset counter to 0
CDD = maximum dry spell length in period
```

**Significance:**
- Drought indicator
- Water resource stress
- Agricultural impacts (crop stress)
- Fire danger accumulation
- Dam yield calculations

**Seasonal Patterns:**
- Longest CDD typically in JJA (winter dry season)
- Can extend 60-90+ days in dry years
- Critical for water security assessment

**Projection Trends:**
Generally increasing CDD in most scenarios, indicating more severe dry spells.

**Bias Correction:** Applied to improve accuracy of spell lengths
""")

    with st.expander("**Wind and Humidity Metrics**", expanded=False):
        st.markdown("""
### Wind Speed Metrics

**Average Wind Speed:**

**Definition:** Mean near-surface (10 m height) wind speed

**Calculation:** Scalar average of daily wind speeds

**Units:** m/s (metres per second)

**Limitations:**
- Scalar average underestimates gustiness
- 250 km resolution cannot capture local wind channelling
- Does not represent extreme gusts (typically 1.5-2× mean)

**Use Cases:**
- General wind climate characterisation
- Dust transport potential
- Evaporation estimates (combined with temperature)

---

**95th Percentile Wind Speed:**

**Definition:** Wind speed exceeded on only 5% of days (18 days per year)

**Significance:**
- Design wind speeds for temporary structures
- Wind loading on equipment
- High wind day frequency
- Operational wind limits

---

**Max Day Wind Speed:**

**Definition:** Highest daily mean wind speed in the period

**Limitations:**
- Daily mean significantly lower than instantaneous gusts
- Extreme wind events poorly captured at coarse resolution
- Recommend site-specific wind engineering assessment for critical applications

---

### Humidity Metrics

**Relative Humidity (RH):**

**Definition:** Ratio of actual vapour pressure to saturation vapour pressure

**Units:** % (percentage)

**Calculation:**
```
RH = (actual_vapour_pressure / saturation_vapour_pressure) × 100
```

Where saturation vapour pressure depends on temperature (Clausius-Clapeyron relation)

**Significance:**
- Heat stress (combined with temperature)
- Apparent temperature/heat index
- Fire danger (low RH increases fire risk)
- Equipment corrosion
- Human comfort

**Typical Values:**
- Morning: 60-80%
- Afternoon: 30-50%
- Seasonal variation: Higher in wet season (DJF)

---

**Vapour Pressure Deficit (VPD):**

**Definition:** Difference between saturation and actual vapour pressure

**Units:** kPa (kilopascals)

**Physical Meaning:**
Measures the "drying power" of the air.  High VPD indicates dry air that readily 
absorbs moisture.

**Significance:**
- Plant water stress (critical for agriculture)
- Evaporation rate
- Transpiration demand
- Fire danger component
- Irrigation scheduling

**Typical Values:**
- Low VPD (<1 kPa): Humid conditions, low evaporation
- Moderate VPD (1-2 kPa): Typical daytime conditions
- High VPD (>2 kPa): Dry conditions, high evaporation, plant stress

**Projection Trends:**
Generally increasing VPD due to warming (saturation vapour pressure increases 
exponentially with temperature).
""")

    # ========================================================================
    # BIAS CORRECTION SECTION
    # ========================================================================

    st.markdown("---")
    st.markdown("## 🔧 Bias Correction Methodology")

    st.markdown("""
Climate models contain systematic biases that must be corrected for regional applications.  
This application implements empirical bias correction using observed climate data as a reference, 
with corrections applied on-demand via the sidebar toggle.

**Key Features:**
- Optional toggle allows viewing raw or bias-corrected data
- Corrections calculated from 1971-2014 Historical vs AGCD comparison
- Additive corrections for temperature, multiplicative for rainfall
- Preserves climate change signals while correcting absolute values
- Applied to Historical and SSP scenarios (AGCD unchanged as it is observational data)
""")

    with st.expander("**Why Bias Correction is Necessary**", expanded=False):
        st.markdown("""
### Sources of Model Bias

Climate models are simplified representations of complex Earth system processes.  
Systematic biases arise from:

**1. Parameterisation Approximations:**
- Sub-grid processes (clouds, convection) are parameterised not explicitly resolved
- Parameterisations calibrated for global performance may not suit all regions
- Trade-offs between different processes introduce systematic errors

**2. Resolution Limitations:**
- 250 km grid spacing cannot resolve local topography, coastlines, land-sea contrasts
- Smooths out extreme values and local features
- Underestimates rainfall intensity due to spatial averaging
- Misses mesoscale phenomena (thunderstorms, sea breezes, orographic enhancement)

**3. Physics Simplifications:**
- Incomplete representation of all relevant processes
- Approximations in radiation, cloud microphysics, land surface interactions
- Coupled feedbacks may be biased (e.g. soil moisture-precipitation)

**4. Regional Climate Characteristics:**
- Tropical-subtropical rainfall patterns challenging to simulate
- Australian monsoon systems require accurate representation of multiple processes
- Local-scale features poorly captured at coarse resolution

### ACCESS-CM2 Identified Biases

**Temperature:**
- Australia-wide: Model runs approximately 1.3°C too warm on annual average
- Seasonal variation: Summer (DJF) warm bias ~4-7°C, Winter (JJA) shows ~2°C warm bias
- Ravenswood: Similar patterns with stronger summer warm bias

**Rainfall:**
- Northern Australia wet season: Model underestimates by 20-50% (published literature)
- Ravenswood DJF: Model underestimates by ~52% (requires 1.52× multiplier)
- Ravenswood dry season: Model overestimates slightly (requires 0.69-0.87× reduction)
- Australia-wide: Model underestimates annual rainfall by ~21%

These biases are **systematic** (consistent across multiple decades) and **significant** 
(affecting decision-making), making bias correction essential for regional applications.

### Scientific Basis for Correction

**Standard Practice:**
Bias correction is widely used in climate impact studies and is recommended by:
- IPCC Working Groups for regional impact assessments
- World Climate Research Programme (WCRP) guidelines
- CSIRO and Bureau of Meteorology for Australian applications
- Academic literature spanning 20+ years of methodological development

**Physical Justification:**
1. Models capture large-scale dynamics (pressure systems, fronts) well
2. Models capture climate change signals (trends, responses to forcing) well  
3. Models struggle with regional-scale absolute values
4. Observations provide the "ground truth" for regional climate
5. Combining model change signals with observed baseline gives best estimates

### Why This Approach is Valid

**Preserves Climate Change Signal:**
- Additive correction (temperature): Future warming added to corrected baseline
- Multiplicative correction (rainfall): Future percentage changes preserved
- Trend information from the model retained
- Only absolute values adjusted to match observations

**Uses Appropriate Reference:**
- AGCD observations: High-quality, gridded, Bureau of Meteorology data
- Calibration period 1971-2014: Maximum overlap between Historical and AGCD
- 44 years provides robust statistics for correction factors
- Covers multiple ENSO cycles and climate regimes

**Defensible for Regulatory Purposes:**
- Transparent methodology documented in code and user guide
- Traceable to published scientific literature
- Corrections empirically derived not arbitrary
- Conservative approach (significant biases only)
- Allows comparison of raw vs corrected (toggle on/off)
""")

    with st.expander("**Bias Correction Implementation**", expanded=False):
        st.markdown("""
### Calibration Period and Data

**Reference Period:** 1971-2014 (44 years)

**Data Sources:**
- **Observations:** Australian Gridded Climate Data (AGCD) from Bureau of Meteorology
  - Daily gridded observations interpolated from weather station network
  - 0.05° resolution (~5 km) regridded to ACCESS-CM2 grid
  - Quality controlled, homogenised time series

- **Model:** ACCESS-CM2 Historical scenario
  - CMIP6 historical simulation driven by observed forcings
  - 250 km native resolution
  - Ensemble member r1i1p1f1

**Why 1971-2014?**
- AGCD reliable coverage begins 1970s
- Historical scenario ends 2014
- Maximises calibration period length
- Includes diverse climate states (El Niño, La Niña, neutral)

### Correction Calculation Method

**Step 1: Calculate Mean Values Over Calibration Period**

For each metric, location and season, calculate:
```
μ_AGCD = mean(AGCD values, 1971-2014)
μ_Hist = mean(Historical values, 1971-2014)
```

**Step 2: Determine Correction Type and Calculate Factor**

**Temperature Metrics (Additive Correction):**
```
Correction = μ_AGCD - μ_Hist

Example: Australia Annual Temperature Average
μ_AGCD = 21.5°C
μ_Hist = 22.8°C
Correction = 21.5 - 22.8 = -1.3°C

Corrected_value = Raw_value + Correction
```

Additive correction preserves temperature trends:
- If model projects +2.5°C warming by 2080
- Corrected: (T_baseline - 1.3°C) + 2.5°C warming
- The 2.5°C warming signal is preserved

**Rainfall Metrics (Multiplicative Correction):**
```
Correction = μ_AGCD / μ_Hist

Example: Ravenswood DJF Total Rainfall
μ_AGCD = 580 mm
μ_Hist = 382 mm
Correction = 580 / 382 = 1.52

Corrected_value = Raw_value × Correction
```

Multiplicative correction preserves rainfall percentage changes:
- If model projects 15% increase by 2080
- Corrected: (R_baseline × 1.52) × 1.15
- The 15% increase is preserved

**Step 3: Apply Significance Threshold**

Only store corrections where bias is significant:
- **Temperature:** |Correction| > 0.5°C
- **Rainfall:** |Correction - 1.0| > 0.10 (i.e. >10% difference)

This avoids over-correcting small, potentially spurious differences.

**Step 4: Apply Corrections to Scenarios**

When bias correction toggle is **ON**:
- Historical and SSP scenarios: Apply corrections
- AGCD: No correction (it is the observational reference)

When toggle is **OFF**:
- All scenarios show raw model output

### Example: Temperature Correction in Detail

**Metric:** Temperature Average, Australia, Annual

**Calibration (1971-2014):**
- AGCD mean: 21.5°C
- Historical mean: 22.8°C  
- Bias: +1.3°C (model too warm)
- Correction: -1.3°C

**Application to SSP1-26 projection for year 2050:**
```
Raw SSP1-26 value: 23.5°C
Corrected value: 23.5°C + (-1.3°C) = 22.2°C

Model projects warming of: 23.5°C - 22.8°C = +0.7°C from historical mean
Corrected warming: 22.2°C - 21.5°C = +0.7°C from AGCD mean

→ Warming signal preserved; absolute value corrected
```

### Example: Rainfall Correction in Detail

**Metric:** Total Rainfall, Ravenswood, DJF (wet season)

**Calibration (1971-2014):**
- AGCD mean: 580 mm
- Historical mean: 382 mm
- Bias: -34% (model underestimates)
- Correction factor: 1.519

**Application to SSP3-70 projection for year 2080:**
```
Raw SSP3-70 value: 420 mm
Corrected value: 420 mm × 1.519 = 638 mm

Model projects change of: (420 - 382) / 382 = +10% from historical
Corrected change: (638 - 580) / 580 = +10% from AGCD

→ Percentage change preserved; absolute value corrected
```

### Correction Coverage

**71 corrections applied across:**
- 2 locations (Australia, Ravenswood)
- 5 seasons (DJF, MAM, JJA, SON, Annual)
- Temperature and rainfall metrics
- Only where bias is significant (>0.5°C or >10%)

**Sample corrections:**
- Australia Annual Temp Average: -1.3°C
- Ravenswood DJF Rain Total: ×1.52
- Ravenswood JJA Rain Total: ×0.69
- Australia DJF Temp Max Day: -6.9°C

Full correction table available in source code: `climate_viewer_constants.py`
""")

    with st.expander("**Scientific Validity and Limitations**", expanded=False):
        st.markdown("""
### Why This Approach is Scientifically Valid

**1. Established Methodology:**
- Delta-change bias correction is standard in climate impact studies
- Published extensively in peer-reviewed literature
- Recommended by IPCC, CSIRO, BoM for regional applications
- Used in Australian Climate Change in Australia portal

**2. Preserves Model Skill:**
- Climate change signals (trends, sensitivities) retained
- Spatial patterns of change preserved
- Temporal relationships maintained
- Only absolute baseline adjusted

**3. Appropriate for Purpose:**
- Infrastructure planning requires accurate absolute values
- Regulatory compliance needs defensible methodology
- Risk assessment benefits from bias-reduced projections
- Stakeholder communication improved with corrected values

**4. Empirically Grounded:**
- Corrections derived from actual observations not theory
- 44-year calibration period provides robust statistics
- Multiple ENSO cycles included in calibration
- Cross-validation possible (not yet implemented)

**5. Transparent and Auditable:**
- Toggle allows viewing raw vs corrected data
- Correction factors explicitly documented
- Methodology traceable in source code
- Users can verify corrections against published literature

### Limitations and Assumptions

**1. Stationarity Assumption:**

*Assumption:* Model biases remain constant under climate change.

*Reality:* Biases may evolve as climate changes.

*Impact:* 
- Temperature: Likely small impact (bias mainly from resolution/parameterisation)
- Rainfall: Potentially larger impact (convective regime changes)
- Conservative: Raw data still available for comparison

*Mitigation:* 
- Use both raw and corrected for uncertainty assessment
- Re-evaluate corrections every 5-10 years as new data available
- Document assumption in reports and decision-making

**2. Single Model:**

*Limitation:* Corrections specific to ACCESS-CM2.

*Impact:*
- Other CMIP6 models have different biases
- Multi-model ensemble would have model-specific corrections
- Model uncertainty not captured

*Mitigation:*
- ACCESS-CM2 is CSIRO's flagship model for Australia
- Documented as suitable for Australian regional applications
- Future work: Add other CMIP6 models with their own corrections

**3. Spatial Scale Mismatch:**

*Issue:* 
- AGCD: 5 km resolution
- ACCESS-CM2: 250 km resolution  
- Corrections applied at coarse scale

*Impact:*
- Local features (topography, coastline) smoothed
- Sub-grid variability not captured
- Corrections represent grid-cell average

*Mitigation:*
- Appropriate for planning-scale applications (infrastructure, portfolios)
- Not suitable for site-specific engineering design without further downscaling
- Document spatial resolution in reports

**4. Observational Uncertainty:**

*Issue:* AGCD has its own uncertainties:
- Station network density varies
- Interpolation introduces smoothing
- Quality control may remove valid extremes
- Urban heat island effects

*Impact:*
- Corrections inherit AGCD uncertainties
- Typically 5-10% for rainfall, 0.5°C for temperature
- Larger in data-sparse regions

*Mitigation:*
- AGCD is best available gridded product for Australia
- BoM quality control procedures well-documented
- Uncertainties small relative to climate change signals

**5. Temporal Sampling:**

*Issue:* 44-year calibration period (1971-2014) may not capture:
- Multi-decadal variability (PDO, IPO)
- Rare extreme events
- Long-term trends in bias structure

*Impact:*
- Correction factors have sampling uncertainty
- ±10-20% for rainfall in typical cases
- ±0.2-0.5°C for temperature

*Mitigation:*
- 44 years exceeds typical 30-year climate normals
- Multiple ENSO cycles captured
- Longest period with reliable overlap

### When to Use Bias Correction

**Use Corrected Data When:**
- Planning infrastructure with 30+ year lifespan
- Assessing climate risks for asset portfolios
- Regulatory compliance requiring best-available projections
- Stakeholder communication about absolute impacts
- Comparing future climate to present-day experience

**Use Raw Data When:**
- Understanding model physics and behaviour
- Comparing different models or scenarios
- Academic research on model performance
- Sensitivity testing to correction assumptions
- Conservative "worst case" analysis (use whichever is more severe)

**Use Both When:**
- Uncertainty assessment for decision-making
- Demonstrating range of plausible futures
- Sensitivity analysis to bias correction
- Peer review of climate risk assessment

### Comparison with Other Approaches

**Simple Mean Correction:**
- Adds/subtracts constant offset
- ✓ Simple, transparent
- ✗ May not preserve trends
- ✗ Does not address distribution shape

**Linear Scaling (This Approach for Rainfall):**
- Multiplies by ratio of means
- ✓ Preserves percentage changes
- ✓ Simple to implement and understand
- ✗ Assumes ratio constant across all values
- Used here for rainfall metrics

**Additive Delta (This Approach for Temperature):**
- Adds difference of means
- ✓ Preserves absolute changes (warming)
- ✓ Appropriate for temperature
- Used here for temperature metrics

**Quantile Mapping:**
- Matches full distributions
- ✓ Corrects distribution shape
- ✓ Improves extremes
- ✗ More complex, harder to validate
- ✗ May not preserve change signals well
- Not used here (simpler methods sufficient)

**Statistical Downscaling:**
- Develops empirical relationships with predictors
- ✓ Can achieve finer spatial resolution
- ✗ Assumes relationships stationary
- ✗ Requires extensive development
- Beyond scope of current application

### Regulatory Compliance and Due Diligence

For AASB S2 climate-related financial disclosures and infrastructure planning:

**Methodology Documentation:**
- Bias correction approach fully documented
- Traceable to published scientific literature  
- Corrections quantified and available for audit
- Assumptions and limitations explicitly stated

**Best Available Science:**
- Uses IPCC-recommended scenarios (SSP pathways)
- Employs established bias correction methodology
- Based on highest quality Australian observations (AGCD)
- Consistent with CSIRO and BoM guidance

**Defensibility:**
- Toggle allows comparison of raw vs corrected
- Transparent correction factors (not a black box)
- Conservative thresholds (only significant biases corrected)
- Uncertainty documentation provided

**Appropriate Use:**
- Suitable for planning-scale assessments (>10 km)
- Appropriate for multi-decade projections
- Valid for infrastructure and portfolio risk assessment
- Not suitable for single-site engineering design without validation
""")

    # ========================================================================
    # WARMING ANALYSIS SECTION
    # ========================================================================

    st.markdown("---")
    st.markdown("## 🌡️ Global Warming Thresholds and Regional Amplification")

    st.markdown("""
Understanding how global warming translates to regional temperature change is critical for 
risk assessment and planning.  This section explains the 1.5°C global warming threshold and 
regional amplification factors.
""")

    with st.expander("**The 1.5°C Global Warming Threshold**", expanded=False):
        st.markdown("""
### Paris Agreement Context

The Paris Agreement (2015) aims to limit global warming to:
- **Primary Goal:** Well below 2.0°C above pre-industrial levels
- **Aspirational Goal:** Limit to 1.5°C above pre-industrial levels

### Pre-Industrial Baseline Definition

**IPCC AR6 Standard:**
- **Period:** 1850-1900 (51 years)
- **Rationale:** Earliest period with sufficient global coverage
- **Global Mean Temperature:** Approximately 13.9°C

**Why 1850-1900?**
1. Pre-dates significant industrialisation impacts
2. Global station network established
3. Sufficient data for robust estimate
4. Standardises comparisons across studies

### Current Warming Status (2023-2024)

**Global Mean Temperature:**
- Current warming: Approximately 1.1-1.2°C above 1850-1900
- Rate of warming: ~0.2°C per decade (recent decades)
- Acceleration: Warming rate increasing

**Trajectory:**
- SSP1-2.6: Peaks ~1.8°C, stabilises
- SSP2-4.5: Reaches ~2.7°C by 2100
- SSP3-7.0: Reaches ~3.6°C by 2100
- SSP5-8.5: Reaches ~4.4°C by 2100

### When Will 1.5°C Be Reached?

**Scenario Timing (50th percentile, global mean):**

| Scenario | Year of 1.5°C Exceedance |
|----------|--------------------------|
| SSP1-2.6 | ~2030-2035 (temporary, then stabilises) |
| SSP2-4.5 | ~2030-2035 (permanent) |
| SSP3-7.0 | ~2030 (permanent, continues rising) |
| SSP5-8.5 | ~2025-2030 (permanent, rapid rise) |

**Key Point:** Under most scenarios, 1.5°C global warming will be reached within the next 
10-15 years.  This makes understanding regional impacts urgent.

### Impacts at 1.5°C Global Warming

**Global Scale Impacts (IPCC AR6):**
- Hot extremes: More frequent and intense
- Heavy precipitation: Increased intensity (7% per °C)
- Drought: Increased frequency in some regions
- Sea level: Committed rise continues for centuries
- Ice sheets: Accelerated melting
- Ecosystems: Widespread changes, species range shifts
- Coral reefs: 70-90% decline at 1.5°C, >99% at 2°C

**Australian Context:**
- Fire weather: Increased extreme fire danger days
- Marine heatwaves: More frequent and severe
- Great Barrier Reef: Severe bleaching events
- Agricultural zones: Shift southward
- Water resources: Increased variability
""")

    with st.expander("**Regional Warming Amplification: Why 1.15× for Ravenswood**", expanded=False):
        st.markdown("""
### The Land-Ocean Warming Contrast

**Physical Principle:**
Land surfaces warm faster than oceans due to fundamental thermodynamic differences.

**Why Land Warms More:**

1. **Heat Capacity Difference:**
   - Ocean: High heat capacity, heat penetrates deep via mixing
   - Land: Low heat capacity, heat concentrated in thin surface layer
   - Result: Same energy input produces more warming over land

2. **Evaporative Cooling:**
   - Ocean: Unlimited water for evaporation, strong cooling effect
   - Land: Limited soil moisture, less evaporative cooling
   - Result: Land has less heat dissipation mechanism

3. **Mixing Depth:**
   - Ocean: Heat mixed through upper 50-100 m (or deeper)
   - Land: Heat confined to top few metres of soil
   - Result: Energy concentrated in smaller mass on land

4. **Surface Properties:**
   - Ocean: Smooth surface, different radiation properties
   - Land: Varied surface (vegetation, soil), different albedo and emissivity
   - Result: Different energy balance

### Global Pattern: ~1.4× Amplification

**IPCC AR6 Assessment:**
- **Global land average:** Approximately 1.4× global mean warming
- **Range:** 1.3-1.6× depending on region and season
- **Confidence:** High (robust across models and observations)

**Physical Interpretation:**
If global mean warms by 1.0°C (combining land and ocean):
- Ocean component: ~0.7°C
- Land component: ~1.4°C
- Weighted average: 1.0°C global mean

### Regional Variation: Why Ravenswood is 1.15×, Not 1.4×

**Location Context:**
- **Ravenswood:** Queensland, Australia (approximately 20°S, 146°E)
- **Regional Classification:** Coastal-influenced inland region
- **Distance from Coast:** ~100-150 km from Coral Sea

**Factors Reducing Amplification Below 1.4×:**

**1. Maritime Influence:**
- Proximity to Coral Sea moderates temperature extremes
- Sea breeze penetration during summer months
- Moisture advection from ocean
- Result: Partial oceanic influence reduces pure land amplification

**2. Tropical-Subtropical Location:**
- Smaller land-ocean temperature contrast in tropics
- High humidity maintains evaporative cooling
- Convective processes distribute heat vertically
- Result: Less concentrated surface warming than mid-latitudes

**3. Monsoon Circulation:**
- Wet season (DJF) brings maritime air masses
- Enhanced evaporative cooling during wet period
- Cloud cover moderates daytime heating
- Result: Seasonal damping of temperature extremes

**4. Regional Analysis of ACCESS-CM2:**
Analysis of ACCESS-CM2 historical and SSP scenarios for the Ravenswood grid cell 
specifically shows:
- Warming ratio (regional/global): 1.12-1.18× across scenarios
- Average: 1.15× used as best estimate
- Confidence: Medium (single model, specific location)

### Mathematical Implementation

**Global Target:** 1.5°C above pre-industrial (1850-1900)

**Regional Conversion:**
```
Regional_warming_target = Global_warming_target × Amplification_factor
Regional_warming_target = 1.5°C × 1.15
Regional_warming_target = 1.725°C ≈ 1.7°C
```

**Regional Temperature Threshold:**
```
If pre-industrial regional mean (1850-1900) = 21.7°C
Then 1.5°C global warming corresponds to:
Regional_temperature = 21.7°C + 1.7°C = 23.4°C
```

**This is the reference line shown in the 1.5°C Global Warming Analysis tab.**

### Uncertainty in Amplification Factor

**Sources of Uncertainty:**

1. **Model Dependence:**
   - Different models show 1.10-1.20× for this region
   - Ensemble mean would be more robust
   - Single model (ACCESS-CM2) used here

2. **Scenario Dependence:**
   - Amplification factor varies slightly by scenario
   - Higher emissions scenarios may show different patterns
   - Non-linear effects at high warming levels

3. **Time Scale:**
   - Transient vs equilibrium response
   - Short-term variability vs long-term trend
   - Amplification may change over time

4. **Local Effects:**
   - Land use changes (clearing, urbanisation)
   - Irrigation effects
   - Urban heat island (if applicable)

**Conservative Approach:**
Using 1.15× is conservative (lower than global land average of 1.4×) and appropriate for:
- Coastal-influenced inland location
- Tropical-subtropical setting
- Risk assessment (not over-estimating impacts)
- Single model limitation acknowledgment

### Comparison to Other Regions

**Typical Amplification Factors:**

| Region Type | Typical Amplification |
|-------------|----------------------|
| Tropical coastal | 1.0-1.2× |
| **Coastal-influenced inland (Ravenswood)** | **1.15-1.2×** |
| Mid-latitude continental | 1.4-1.6× |
| High-latitude continental | 1.6-2.0× |
| Arctic (polar amplification) | 2.0-3.0× |

### Implications for Risk Assessment

**For 1.5°C Global Warming:**
- Ravenswood regional temperature: +1.7°C above 1850-1900
- Less extreme than continental interior (+2.1°C)
- But still significant impacts on heat extremes, water resources, ecosystems

**For 2.0°C Global Warming:**
- Ravenswood regional temperature: +2.3°C above 1850-1900
- Crossing additional threshold for severe impacts

**For 3.0°C Global Warming (SSP3-7.0):**
- Ravenswood regional temperature: +3.5°C above 1850-1900
- Transformative changes to climate and ecosystems

### References

- IPCC AR6 WG1 (2021), Chapter 4: Future Global Climate
- Sutton et al. (2007), Land/sea warming ratio in response to climate change
- Byrne & O'Gorman (2018), Trends in continental temperature and humidity
- Grose et al. (2020), Insights from CMIP6 for Australia's future climate
""")

    with st.expander("**Interpreting the 1.5°C Warming Visualisations**", expanded=False):
        st.markdown("""
### Dashboard Display

On charts showing annual average temperature, a **red dashed reference line** may appear.

**This line represents:**
The regional temperature corresponding to 1.5°C global warming above pre-industrial levels, 
calculated as:

```
Regional_temp_at_1.5C = Regional_baseline_1850-1900 + (1.5°C × 1.15)
Regional_temp_at_1.5C = 21.7°C + 1.725°C
Regional_temp_at_1.5C ≈ 23.4°C
```

### How to Use This Information

**1. Timing Assessment:**
Identify when each scenario crosses the reference line:
- Earlier crossing = faster warming rate
- Steeper slope = higher rate of change
- Permanent vs temporary exceedance

**2. Magnitude Assessment:**
Calculate how far above the threshold:
```
Exceedance = Actual_temperature - Reference_threshold
```
Example: If 2050 temperature is 24.2°C:
Exceedance = 24.2°C - 23.4°C = 0.8°C above 1.5°C global warming level

**3. Duration Assessment:**
Count years exceeding the threshold:
- Continuous exceedance indicates new climate state
- Temporary exceedance may revert (only SSP1-2.6)
- Duration increases with higher emissions scenarios

**4. Planning Implications:**

**Infrastructure Design:**
- Design for conditions at or above the reference line
- Consider safety factors for uncertainty
- Plan adaptation triggers

**Water Resources:**
- Assess impacts on evaporation, runoff, demand
- Plan for reduced reliability
- Design storage with climate change buffer

**Operations:**
- Heat stress management protocols
- Equipment temperature limits
- Process design temperatures

**Compliance:**
For AASB S2 climate-related financial disclosures:
- 1.5°C is referenced temperature threshold
- Must assess resilience at 1.5°C and higher scenarios
- Document adaptation strategies

### Global Warming Analysis Tab

The dedicated **1.5°C Global Warming Analysis** tab provides:

1. **Timing Table:**
   - Year each scenario reaches 1.5°C global warming
   - Regional temperature at that time
   - Exceedance duration metrics

2. **Impact Comparison:**
   - Side-by-side comparison across scenarios
   - Visual representation of timing differences
   - Quantitative metrics for each scenario

3. **Interpretation Guidance:**
   - Context for results
   - Implications for planning horizons
   - Links to detailed methodology

**Use this tab for:**
- Scenario comparison and selection
- Planning horizon establishment
- Risk threshold identification
- Compliance documentation
""")

    # ========================================================================
    # DISPLAY OPTIONS SECTION
    # ========================================================================

    st.markdown("---")
    st.markdown("## 📊 Display Options and Analysis Modes")

    with st.expander("**Display Mode: Values, Baseline, Deltas**", expanded=False):
        st.markdown("""
### Values (Absolute Mode)

**Description:** Shows actual measured or modelled values in original units.

**Display:**
- Temperature: °C
- Rainfall: mm
- Wind: m/s
- Humidity: %

**When to Use:**
- Comparing to design thresholds (e.g. Is rainfall >1000 mm?)
- Assessing absolute magnitudes for operations
- Determining if values exceed critical thresholds
- Reporting actual conditions

**Example:**
Annual average temperature in 2050 under SSP2-4.5: 24.2°C

---

### Baseline Mode (Change from Start Year)

**Description:** Shows change relative to the first year in your selected range.

**Calculation:**
```
Change(year) = Value(year) - Value(start_year)
```

**When to Use:**
- Emphasising warming or rainfall trends
- Removing absolute bias (preserves trend)
- Comparing scenarios starting from same point
- Visualising rate of change

**Example:**
If start year (2020) has 23.5°C and 2050 has 24.2°C:
Change = +0.7°C

**Scientific Advantage:**
The "delta method" or "anomaly method" is scientifically conservative because:
- Climate change signal (trends) preserved even if absolute values biased
- Models better at simulating changes than absolute values
- Reduces impact of systematic biases
- Standard method in IPCC assessments

---

### Deltas vs Reference Scenario

**Description:** Shows difference between each scenario and a reference scenario.

**Calculation:**
```
Delta(year) = Value_ScenarioX(year) - Value_Reference(year)
```

**Default Reference:** Typically SSP1-2.6 (lowest emissions pathway)

**When to Use:**
- Quantifying cost of inaction (difference between scenarios)
- Policy analysis (impact of emission choices)
- Scenario comparison (divergence over time)
- Decision support (benefit of mitigation)

**Example:**
In 2050:
- SSP1-2.6: 23.8°C
- SSP5-8.5: 25.1°C
- Delta: +1.3°C (SSP5-8.5 is 1.3°C warmer than SSP1-2.6)

**Interpretation:**
- Positive delta: Scenario is warmer/wetter than reference
- Negative delta: Scenario is cooler/drier than reference
- Growing delta: Scenarios diverging (emission choices matter more)

**Policy Relevance:**
Deltas quantify the impact of different emission pathways, critical for:
- Cost-benefit analysis of mitigation
- Adaptation planning priorities
- Climate risk disclosure (AASB S2)
- Investment decision-making
""")

    with st.expander("**Smoothing: Reducing Year-to-Year Variability**", expanded=False):
        st.markdown("""
### Purpose of Smoothing

Climate data contains two components:
1. **Forced Trend:** Long-term change due to greenhouse gases
2. **Internal Variability:** Year-to-year fluctuations (natural chaos)

Smoothing reveals the underlying trend by averaging out internal variability.

### Method: Centred Rolling Average

**Calculation:**
```
Smoothed_value(year) = mean(values[year - window/2 to year + window/2])
```

**Example with 9-year window:**
Smoothed value for 2025 = average of (2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029)

### Scientific Justification

**Why This Method:**
- Centred average (not trailing) prevents lag in trend
- Equal weighting of all years in window
- Preserves long-term trends exactly
- Does not create false trends
- Standard method in climate science

**What Smoothing Does:**
- Removes high-frequency noise
- Preserves low-frequency signal (trend)
- Reduces apparent scatter
- Makes scenario differences clearer

**What Smoothing Does NOT Do:**
- Change long-term trends or mean
- Create information not present in data
- Correct biases
- Improve accuracy (only visualisation)

### Window Selection Guidelines

**3-5 Years (Light Smoothing):**
- **Preserves:** ENSO cycles, drought-flood sequences
- **Removes:** Single-year anomalies
- **Use for:** Recent period analysis, event context

**7-11 Years (Moderate Smoothing - Recommended):**
- **Preserves:** Decadal variability, climate trends
- **Removes:** Year-to-year noise, individual events
- **Use for:** Standard climate assessment, most applications
- **Rationale:** Balances trend clarity with temporal resolution

**15-21 Years (Heavy Smoothing):**
- **Preserves:** Multi-decadal trends only
- **Removes:** Most variability including some real variations
- **Use for:** Very long-term trends, century-scale analysis

### When to Use Smoothing

**Recommended for:**
- Rainfall analysis (highly variable year-to-year)
- Long time series (>50 years)
- Trend identification
- Scenario comparison
- Policy presentation

**NOT Recommended for:**
- Extreme event analysis (need individual years)
- Short time series (<20 years)
- Threshold exceedance counts
- Recent years (edge effects)

### Edge Effects

**Issue:** Smoothing requires data before and after each point.

**Impact on Edges:**
- First few years: Incomplete window (only future data available)
- Last few years: Incomplete window (only past data available)
- Can create artifacts at start and end of series

**Mitigation:**
- Application handles edges carefully
- Reduces window size at edges if necessary
- Consider removing edge points from analysis

### Example: Rainfall Smoothing

**Raw Annual Rainfall (hypothetical):**
```
Year:     2020  2021  2022  2023  2024  2025  2026  2027  2028  2029  2030
Rainfall: 650   480   720   580   690   510   780   600   550   720   640 mm
```

**9-Year Smoothed:**
```
Year:     2020  2021  2022  2023  2024  2025  2026  2027  2028  2029  2030
Smoothed: edge  edge  625   620   635   640   640   645   648  edge  edge mm
```

Notice:
- Edge years cannot be smoothed with full window
- Middle years smoothed values between high and low years
- Overall trend of ~640 mm clearly visible
- Individual wet/dry years averaged out

### Technical Note: Implementation

**Algorithm:**
```python
def rolling_smooth(data, window=9):
    smoothed = []
    half_window = window // 2
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        smoothed.append(mean(data[start:end]))
    return smoothed
```

This ensures:
- Centred window when possible
- Reduced window at edges (not NaN)
- Consistent handling across all metrics
""")

    # ========================================================================
    # TIME HORIZONS SECTION
    # ========================================================================

    st.markdown("---")
    st.markdown("## 📅 Time Horizons and Planning Periods")

    with st.expander("**IPCC Assessment Report Time Horizons**", expanded=False):
        st.markdown("""
### Standard Climate Assessment Periods

IPCC Assessment Reports use standardised time periods for consistency across studies.

**Near-Term (2021-2040):**
- **Characteristics:** Already partly committed warming
- **Uncertainty:** Low (mostly determined by historical emissions)
- **Scenario Dependence:** Low (scenarios similar in near-term)
- **Key Driver:** Emissions to date + short-lived forcing agents
- **Confidence:** High for temperature, medium for precipitation

**Mid-Term (2041-2060):**
- **Characteristics:** Scenario divergence becomes apparent
- **Uncertainty:** Medium (emission pathway matters)
- **Scenario Dependence:** Medium to high
- **Key Driver:** Emissions in 2020-2040 period critical
- **Confidence:** Medium for most variables

**Long-Term (2081-2100):**
- **Characteristics:** Maximum scenario divergence
- **Uncertainty:** High (emission pathway dominates)
- **Scenario Dependence:** Very high
- **Key Driver:** Cumulative emissions over 21st century
- **Confidence:** Lower (more model dependence)

### Customisable Time Horizons in Application

The application allows you to define custom planning horizons relevant to your specific needs.

**Defaults:**
- **Short-Term:** 2020-2040
- **Mid-Term:** 2041-2060
- **Long-Term:** 2061-2080
- **Planning Horizon End:** 2100

**How to Customise:**
Use sidebar controls to set:
- Start year for each horizon
- End year for long-term planning
- Allows alignment with asset lifetimes, planning cycles, regulatory requirements

### Matching Horizons to Applications

**Infrastructure Planning:**

| Asset Type | Typical Lifetime | Recommended Horizon |
|------------|------------------|---------------------|
| Temporary structures | <5 years | Near-term only |
| Mobile equipment | 10-15 years | Near-term |
| Buildings | 25-50 years | Mid-term |
| Process plants | 30-40 years | Mid-term to long-term |
| Dams, tailings | 100+ years | Long-term + sensitivity analysis |

**Example: Mining Operation**
- Current operations (2025-2035): Near-term climate
- Life of mine (2025-2045): Near-term + mid-term
- Closure phase (2045-2145): Mid-term + long-term + post-2100
- Tailings storage: Multi-century assessment required

### Uncertainty Increases with Horizon

**Sources of Increasing Uncertainty:**

1. **Scenario Uncertainty:**
   - Near-term: Which pathway are we on? (Low uncertainty)
   - Long-term: Which pathway will we follow? (High uncertainty)

2. **Model Uncertainty:**
   - Near-term: Climate sensitivity matters less
   - Long-term: Climate sensitivity critical (feedback processes)

3. **Internal Variability:**
   - Constant absolute magnitude
   - Smaller fraction of signal in long-term (trend dominates)
   - Larger fraction in near-term (noise comparable to signal)

**Confidence Levels by Horizon:**

| Variable | Near-Term | Mid-Term | Long-Term |
|----------|-----------|----------|-----------|
| Temperature | Very High | High | Medium-High |
| Extreme heat | High | Medium-High | Medium |
| Mean precipitation | Medium | Low-Medium | Low |
| Extreme precipitation | Medium | Low-Medium | Low |
| Drought | Low-Medium | Low | Low |

### Time Horizon Shading on Charts

**Visual Aid:**
Charts can display coloured shading for each horizon:
- **Blue:** Short-term period
- **Orange:** Mid-term period
- **Red:** Long-term period

**Purpose:**
- Quick visual reference for planning periods
- Understand when critical thresholds may be crossed
- Align climate information with decision timeframes

**How to Enable:**
Use "Show Time Horizon Shading" checkbox in sidebar display options.
""")

    # ========================================================================
    # UNCERTAINTY SECTION
    # ========================================================================

    st.markdown("---")
    st.markdown("## ⚠️ Uncertainties and Limitations")

    st.markdown("""
All climate projections contain uncertainties.  Understanding these limitations is essential 
for informed decision-making and appropriate application of the data.
""")

    with st.expander("**Sources of Uncertainty in Climate Projections**", expanded=False):
        st.markdown("""
### Three Primary Sources

Climate projection uncertainty comes from three main sources that affect different timescales.

---

### 1. Internal Variability (Natural Climate Chaos)

**Definition:** Year-to-year and decade-to-decade fluctuations arising from the chaotic nature 
of the climate system.

**Causes:**
- Ocean circulation variations (ENSO, IPO, PDO)
- Atmospheric circulation patterns
- Ocean-atmosphere interactions
- Random weather variations

**Characteristics:**
- Irreducible uncertainty (chaotic system, not predictable beyond ~2 weeks)
- Same magnitude in all time periods
- Larger relative to signal in near-term
- Affects individual years and short periods, not long-term trends

**Example:**
Two identical climate model runs with infinitesimally different starting conditions will 
diverge within weeks to months.  Internal variability means we cannot predict whether 2030 
will be El Niño or La Niña, but we can predict the long-term warming trend.

**Quantification:**
- Estimated from ensemble members (multiple runs of same model)
- Typically ±0.2-0.3°C for decadal mean temperature
- Much larger for precipitation (±20-30% for decadal mean)

**Implications:**
- Cannot predict exact values for specific years
- Can predict multi-decadal trends with confidence
- Planning should consider range, not single value

---

### 2. Model Uncertainty (Structural and Parametric)

**Definition:** Uncertainty arising from incomplete representation of Earth system processes 
and differences between climate models.

**Causes:**

**Structural Uncertainty:**
- Different model physics schemes
- Resolution differences (grid size)
- Component interactions (atmosphere-ocean-ice-land)
- Parameterisation choices (clouds, convection, etc.)

**Parametric Uncertainty:**
- Parameter values in equations
- Empirical relationships
- Tuning choices

**Characteristics:**
- Affects all time periods
- Largest for regional precipitation
- Can be estimated from multi-model ensembles
- Some processes more uncertain than others

**Example - Cloud Feedback:**
Different models simulate cloud responses to warming differently, leading to range of 
climate sensitivities (1.8-5.6°C for 2×CO₂ in CMIP6).

**Quantification:**
- Multi-model ensemble spread: ±0.5-1.0°C for global temperature by 2100
- Regional scales: ±20-50% for precipitation
- Larger for extremes than means

**This Application:**
Uses single model (ACCESS-CM2) and single ensemble member (r1i1p1f1).
- Does not capture model uncertainty
- Representative of ACCESS-CM2 projections only
- Consider consulting multi-model ensemble for comprehensive assessment

---

### 3. Scenario Uncertainty (Future Emissions Path)

**Definition:** Uncertainty about which emission pathway humanity will follow.

**Causes:**
- Policy decisions (mitigation actions)
- Economic development pathways
- Technological change rates
- Social and behavioural factors
- Geopolitical events

**Characteristics:**
- Negligible in near-term (committed warming)
- Grows rapidly after 2040
- Dominates by 2100 (60% of total uncertainty)
- Reducible through mitigation policy

**Quantification:**
By 2100, scenario spread:
- Temperature: 2-3°C between SSP1-2.6 and SSP5-8.5
- Larger than model uncertainty or internal variability
- Represents policy choices, not physical uncertainty

**Implications:**
- Scenario selection critical for long-term planning
- Consider multiple scenarios (low, medium, high)
- Update projections as emission pathways become clearer
- Focus mitigation efforts (only reducible uncertainty source)

---

### Relative Contribution by Time Period

**Near-Term (2030):**
- Internal variability: 50%
- Model uncertainty: 40%
- Scenario uncertainty: 10%

**Mid-Term (2050):**
- Internal variability: 20%
- Model uncertainty: 40%
- Scenario uncertainty: 40%

**Long-Term (2090):**
- Internal variability: 10%
- Model uncertainty: 30%
- Scenario uncertainty: 60%

**Interpretation:**
Near-term projections are more certain (warming already committed), but cannot predict 
individual years.  Long-term projections have large scenario uncertainty, but emission 
choices determine the outcome.
""")

    with st.expander("**Confidence Levels in Climate Projections**", expanded=False):
        st.markdown("""
### IPCC Confidence Framework

IPCC uses standardised terminology to communicate confidence in findings.

**Confidence Terms (qualitative):**
- **Very High Confidence:** Robust evidence, high agreement (>90% probability)
- **High Confidence:** Consistent evidence, good agreement (66-90% probability)
- **Medium Confidence:** Some evidence, emerging agreement (33-66% probability)
- **Low Confidence:** Limited evidence, low agreement (<33% probability)

---

### Confidence by Variable and Scale

**Global Temperature:**
- Near-term warming: **Very High Confidence**
- Long-term warming direction: **Very High Confidence**
- Magnitude (given scenario): **High Confidence**

**Regional Temperature (Australia):**
- Warming direction: **Very High Confidence**
- Warming magnitude: **High Confidence** (near-term), **Medium-High** (long-term)
- Seasonal pattern changes: **Medium Confidence**

**Global Precipitation Changes:**
- Wet regions wetter, dry regions drier: **High Confidence**
- Extreme rainfall intensification: **High Confidence** (7% per °C warming)
- Overall amount: **Medium Confidence**

**Regional Precipitation (Australia):**
- Northern Australia wet season: **Medium Confidence** (likely increase in intensity)
- Southern Australia cool season: **Medium Confidence** (likely decrease)
- Local/site-specific totals: **Low Confidence** (high variability)

**Extreme Events:**
- Hot temperature extremes: **Very High Confidence** (increase in frequency and intensity)
- Heavy rainfall events: **High Confidence** (increase in intensity)
- Drought: **Medium Confidence** (region-dependent, multiple definitions)
- Tropical cyclones: **Medium Confidence** (overall decrease in number, increase in intensity)
- Extreme winds (non-tropical): **Low Confidence**

---

### What Affects Confidence?

**High Confidence Factors:**
- Physical understanding robust (e.g. thermodynamics)
- Model agreement high
- Observed trends consistent with projections
- Multiple lines of evidence
- Large-scale phenomena

**Low Confidence Factors:**
- Complex processes (e.g. convection)
- Model disagreement
- Limited observations
- Multiple competing factors
- Small-scale phenomena

---

### Implications for Decision-Making

**High Confidence Projections:**
- Can base adaptation planning on central estimate
- Still consider range for risk management
- Strong scientific basis for action

**Medium Confidence Projections:**
- Use ensemble range (e.g. 10th to 90th percentile)
- Scenario comparison critical
- Flexible/adaptive strategies appropriate

**Low Confidence Projections:**
- Consider wide range of possibilities
- Stress-test against multiple scenarios
- Avoid irreversible decisions based solely on these projections
- Monitor and update as science advances

**Example Application:**
- **Temperature-based design:** High confidence → use central projection with safety factor
- **Rainfall-based design:** Medium-low confidence → test across wide range, use conservative assumptions
""")

    with st.expander("**Limitations of This Application**", expanded=False):
        st.markdown("""
### Single Model and Ensemble Member

**Limitation:**
This application uses only ACCESS-CM2 (one climate model) and r1i1p1f1 (one ensemble member).

**Implications:**
- Does not represent model uncertainty
- May not capture full range of possible futures
- ACCESS-CM2 biases and strengths will affect all results

**Mitigation:**
- Bias correction applied for rainfall
- ACCESS-CM2 is a credible, peer-reviewed CMIP6 model
- Results generally consistent with multi-model mean
- For critical applications, consult multi-model ensembles (e.g. CMIP6 archive)

**When Multi-Model Ensemble Needed:**
- High-stakes decisions (billion-dollar infrastructure)
- Legally binding compliance (detailed due diligence)
- Publication/formal assessment
- Quantifying full uncertainty range

---

### Spatial Resolution Limitations

**Model Resolution:** ~250 km (1.25° × 1.875°)

**What This Means:**
- Local topographic features not resolved (valleys, hills, coastal effects)
- Mesoscale phenomena underrepresented (thunderstorms, sea breezes, fog)
- Extreme values underestimated (spatial averaging)
- Coastal gradients smoothed

**Example Impact:**
A thunderstorm producing 100 mm rainfall in 1 hour over 10 km² would appear in model as 
~20 mm over 24 hours across 62,500 km² grid cell.

**Implications for Application:**
- Temperature: Generally reliable at regional scale
- Rainfall totals: Reasonable for monthly/seasonal averages
- Rainfall extremes: Underestimated (bias correction helps)
- Local features: Not captured (e.g. valley cold pooling, orographic enhancement)

**When Downscaling Needed:**
- Sub-10 km scale applications
- Complex topography
- Coastal sites
- Urban areas
- Extreme event magnitude (not just frequency)

---

### Bias Correction Limitations

**Stationarity Assumption:**
Current method assumes bias structure remains constant under climate change.

**Potential Issues:**
- Model biases may change with warming
- New climate regimes may emerge
- Physical processes may shift non-linearly

**Implications:**
- Bias correction best for near to mid-term
- Long-term projections more uncertain
- Regular updates recommended as new observations available

**Quality of Historical Data:**
Bias correction relies on AGCD observations:
- Limited to 1990-2014 (data availability)
- Station density affects quality
- Early period less reliable

---

### Temporal Scope

**Historical Period:**
- Begins 1850 (model simulation)
- AGCD observations: 1900+ (variable-dependent)
- Quality improves markedly after 1960

**Future Period:**
- Extends to 2100
- Post-2100: No data (would require extended concentration pathways)

**No Sub-Daily Data:**
- Daily values only
- Cannot resolve diurnal cycle
- Hours of exceedance estimated, not calculated

---

### Variables Not Included

This application does not include:
- Hourly temperature data (only daily averages and max/min)
- Solar radiation
- Incoming longwave radiation
- Soil moisture
- Evapotranspiration (direct calculation)
- Snow cover/depth
- Atmospheric pressure
- Cloud cover
- Many other CMIP6 variables

**Why Not Included:**
- Focus on most critical variables for mining/industry
- Data size/processing constraints
- Some variables require complex post-processing
- Not all variables available from AGCD for validation

---

### Uncertainty Quantification Limitations

**No Formal Uncertainty Bounds:**
Application does not provide confidence intervals or probability distributions because:
- Single model (cannot estimate model uncertainty)
- Single ensemble member (limited internal variability quantification)
- Would require complex statistical analysis

**Qualitative Only:**
- Scenario comparison shows some spread
- Smoothing shows some year-to-year variability
- Users should consult literature for formal uncertainty quantification

---

### Downstream Impacts Not Modeled

Climate variables provided, but not:
- Hydrological modeling (runoff, streamflow, groundwater)
- Vegetation/ecosystem responses
- Fire danger indices (only input variables)
- Heat stress indices (only input variables)
- Infrastructure failure thresholds
- Economic impacts

**Users Must:**
- Apply climate data to impact models
- Consider non-climate factors
- Integrate with other information sources

---

### Regulatory Compliance Context

**AASB S2 Climate Disclosures:**
This tool provides climate projections suitable for:
- Scenario analysis (multiple SSPs available)
- Physical risk assessment
- Transition risk context (via scenario descriptions)

**However:**
- Financial materiality assessment: User's responsibility
- Integration with business model: User's responsibility
- Adaptation strategy development: Requires expertise
- Governance structure: Outside scope of tool

**Users Must Also Consider:**
- Task Force on Climate-related Financial Disclosures (TCFD) alignment
- ISSB Standards (IFRS S1, IFRS S2)
- Jurisdictional requirements (Australian context)
- Industry-specific guidance (mining sector best practices)

---

### When to Seek Expert Assistance

**Consider Consulting Climate Scientists for:**
- Multi-billion dollar decisions
- Long-lived critical infrastructure (100+ year lifetime)
- Novel applications without established methods
- High legal/regulatory scrutiny
- Publication/peer review
- Complex impact models
- Downscaling to local scale (<10 km)
- Formal uncertainty quantification
- Multi-model ensemble analysis
- Bias correction validation

**This Application is Appropriate for:**
- Preliminary assessment
- Planning-level analysis
- Scenario exploration
- Trend identification
- Education/capacity building
- Supporting detailed analysis (as input)
- Routine climate risk screening
""")

    # ========================================================================
    # BEST PRACTICES SECTION
    # ========================================================================

    st.markdown("---")
    st.markdown("## ✅ Best Practices for Climate Risk Assessment")

    with st.expander("**Scenario Selection Strategy**", expanded=False):
        st.markdown("""
### Why Multiple Scenarios?

**Single Scenario Problems:**
- Ignores emissions uncertainty
- May miss critical risks
- Creates false sense of precision
- Not aligned with IPCC or TCFD frameworks

**Multi-Scenario Approach:**
Provides understanding of:
- Sensitivity to emission pathway choices
- Range of plausible futures
- Decision robustness across scenarios
- Timing of impacts

---

### Recommended Scenario Sets

**Minimum (Quick Assessment):**
- **SSP1-2.6:** Low emissions (optimistic)
- **SSP5-8.5:** High emissions (pessimistic)

**Standard (Most Applications):**
- **SSP1-2.6:** Low emissions
- **SSP2-4.5:** Intermediate (often "business as usual")
- **SSP5-8.5:** High emissions

**Comprehensive (Detailed Analysis):**
- **SSP1-2.6:** Low emissions
- **SSP2-4.5:** Intermediate
- **SSP3-7.0:** High emissions, fragmented world
- **SSP5-8.5:** Very high emissions, fossil-fueled

---

### Matching Scenarios to Applications

**Infrastructure Planning (Long-Lived Assets):**
- Test designs against **SSP2-4.5** (central)
- Stress-test against **SSP5-8.5** (high-end)
- Consider **SSP1-2.6** for optimistic planning

**Financial Risk Disclosure (AASB S2):**
- Required: Assess resilience under **1.5°C and 2°C** global warming
- Implied scenarios: **SSP1-2.6** (1.5°C) and **SSP2-4.5** (2°C)
- Optional: Include **SSP5-8.5** for physical risk upper bound

**Adaptation Planning:**
- Focus on **SSP2-4.5** and **SSP5-8.5**
- If adaptation effective under SSP5-8.5, robust to lower scenarios
- Check timing differences (when do impacts occur?)

**Policy Analysis:**
- Compare all scenarios to show mitigation benefit
- **Deltas vs SSP1-2.6** shows cost of inaction
- Useful for benefit-cost analysis

---

### Time Horizons for Scenarios

**Near-Term (2021-2040):**
- Scenarios converge (limited divergence)
- All scenarios relevant (committed warming)
- Focus on adaptation readiness

**Mid-Term (2041-2060):**
- Scenarios begin diverging
- Central scenarios most relevant for planning
- High scenarios for stress-testing

**Long-Term (2061-2100):**
- Maximum scenario divergence
- Test full range (SSP1-2.6 to SSP5-8.5)
- Scenario selection critical for outcomes

---

### Updating Scenario Selection

**As Science Advances:**
- New IPCC reports may adjust likelihoods
- Emission trends may favour some scenarios
- Model improvements change projections

**As Policy Evolves:**
- Paris Agreement implementation
- National Determined Contributions (NDCs)
- Technology breakthroughs
- Geopolitical shifts

**Recommendation:**
Review scenario selection every 3-5 years or with major IPCC reports.
""")

    with st.expander("**Interpreting Results for Decision-Making**", expanded=False):
        st.markdown("""
### From Climate Data to Decisions

Climate projections are scientific information, not decisions.  Converting data to action 
requires additional steps:

**Step 1: Identify Decision Context**
- What decision is being made?
- What are the consequences?
- What is the decision timeframe?
- Who are the stakeholders?

**Step 2: Determine Climate Sensitivities**
- Which climate variables matter most?
- What are the critical thresholds?
- Are there tipping points or non-linearities?
- Are there interaction effects?

**Step 3: Extract Relevant Climate Information**
- Select appropriate scenarios
- Choose relevant time horizons
- Identify critical metrics
- Calculate statistics (means, extremes, trends)

**Step 4: Quantify Impacts**
- Apply climate changes to impact models
- Consider non-climate factors
- Estimate magnitude and timing
- Identify uncertainties

**Step 5: Evaluate Options**
- Assess adaptation options
- Consider costs and benefits
- Test robustness across scenarios
- Identify no-regret actions

**Step 6: Implement and Monitor**
- Make decision with best available information
- Establish monitoring triggers
- Plan for adaptive management
- Schedule periodic review

---

### Dealing with Uncertainty

**Accept Uncertainty:**
- Perfect knowledge impossible
- Some uncertainty irreducible
- Don't wait for certainty to act

**Characterise Uncertainty:**
- What do we know with high confidence?
- What is more uncertain?
- What are the bounds?

**Design for Uncertainty:**
- Robust solutions (work across scenarios)
- Flexible options (can adapt later)
- Staged implementation (learn as you go)
- Reversible choices when possible

**Example Approaches:**

**Robust Decision-Making:**
Choose options that perform "reasonably well" across all scenarios rather than "optimally" 
under one scenario.

**Adaptive Management:**
Implement in stages with decision points to adjust based on observed trends and updated 
projections.

**Real Options Analysis:**
Value flexibility to change course as uncertainty resolves.

---

### Red Flags and Reality Checks

**Warning Signs (Potential Misuse of Data):**
- Relying on single scenario
- Focusing only on far-future (ignoring near-term)
- Ignoring model limitations (spatial resolution, single model)
- Treating projections as predictions (specific years)
- Using data beyond valid spatial scale
- Not considering bias correction for rainfall
- Assuming linear extrapolation of trends
- Neglecting non-climate factors

**Reality Checks:**
- Do results make physical sense?
- Are magnitudes reasonable (compare to literature)?
- Are trends consistent across scenarios?
- Have you tested sensitivity to assumptions?
- Does it align with IPCC findings?
- Have multiple lines of evidence?

---

### Communicating Uncertainty to Stakeholders

**Different Audiences:**

**Technical Audience (Engineers, Scientists):**
- Provide quantitative ranges
- Explain methodology details
- Reference peer-reviewed sources
- Discuss limitations explicitly

**Decision-Makers (Executives, Board):**
- Focus on key metrics (not all variables)
- Present scenario range clearly
- Emphasise decision-relevant information
- Provide actionable recommendations

**General Audience (Public, Community):**
- Use visual aids (charts, maps)
- Avoid jargon
- Relate to lived experience
- Focus on impacts, not just numbers

**Regulatory/Compliance Context:**
- Follow prescribed frameworks (AASB S2, TCFD)
- Document methodology thoroughly
- Provide traceable references
- Include expert review

---

### Documentation Best Practices

**Essential to Document:**
1. **Data sources:** CMIP6 ACCESS-CM2, AGCD
2. **Scenarios used:** Which SSPs, why chosen
3. **Time periods:** Baseline, projections, horizons
4. **Spatial extent:** Grid cell(s), location(s)
5. **Metrics calculated:** Definitions, methods
6. **Bias correction:** If applied, method used
7. **Limitations:** What is NOT included
8. **Assumptions:** Key judgements made
9. **Uncertainties:** Confidence levels, caveats
10. **Version control:** When accessed, version of data/tool

**Why Documentation Matters:**
- Reproducibility
- Auditability (regulatory compliance)
- Update when new data available
- Defend decisions if challenged
- Institutional memory

**Recommendation:**
Create a "Climate Data Lineage" document for each significant assessment.
""")

    with st.expander("**AASB S2 Climate-Related Financial Disclosure Guidance**", expanded=False):
        st.markdown("""
### AASB S2 Standard Overview

**Australian Accounting Standards Board Standard 2:**  
Climate-related Financial Disclosures

**Effective Date:**
Annual reporting periods beginning on or after 1 January 2025 (Australia)

**Scope:**
Requires entities to disclose:
- Governance around climate risks and opportunities
- Strategy for managing climate risks
- Risk management processes
- Metrics and targets

---

### Physical Risk Assessment Requirements

**Scenario Analysis Requirement:**

Entities must assess resilience of strategy under:
1. **1.5°C warming scenario** (relative to pre-industrial)
2. **Higher warming scenario** (entity's choice, typically 2-4°C)

**Metrics Required:**
- Amount and percentage of assets vulnerable to physical risks
- Amount and percentage of revenue vulnerable to physical risks
- Internal carbon price (if used)
- Scope 1, 2, 3 greenhouse gas emissions

**Disclosure Required:**
- Significant climate-related risks and opportunities
- Current and anticipated impacts on business model
- Financial effects (quantified where possible)
- Resilience of strategy under different scenarios

---

### Using This Tool for AASB S2 Compliance

**What This Tool Provides:**

**1. Scenario Analysis Inputs:**
- Multiple SSP scenarios (including 1.5°C-consistent SSP1-2.6)
- Temperature projections (for 1.5°C threshold identification)
- Physical climate variables (temperature, precipitation, wind)
- Time horizon flexibility (near, medium, long-term)

**2. Physical Risk Indicators:**
- Heat exposure metrics (days >37°C, >40°C)
- Water availability trends (rainfall totals, dry spells)
- Extreme event metrics (maximum rainfall, wind)
- Trends and magnitudes of change

**3. Timing Information:**
- When thresholds crossed (e.g. 1.5°C global warming)
- Rate of change
- Short vs long-term impacts

**What This Tool Does NOT Provide:**

- Financial impact quantification (requires business context)
- Asset-specific vulnerability (requires engineering assessment)
- Adaptation strategy development (requires expert input)
- Governance structure guidance (organisational decision)
- Materiality assessment (entity-specific)

---

### Recommended Workflow for AASB S2 Physical Risk Assessment

**Phase 1: Screening Assessment**
1. Use tool to assess climate changes under SSP1-2.6 (1.5°C) and SSP5-8.5 (high warming)
2. Identify which variables show significant changes
3. Determine time horizons relevant to business
4. Document key findings

**Phase 2: Vulnerability Assessment**
1. Match climate variables to operational thresholds
   - Example: Temperature >37°C → heat stress procedures triggered
   - Example: Rainfall <500 mm/year → water supply constraints
2. Identify which assets/operations exposed
3. Quantify exposure (% of assets, revenue, etc.)
4. Assess adaptive capacity (current measures)

**Phase 3: Impact Quantification**
1. Model financial impacts (may require consultants)
   - Production losses from heat shutdowns
   - Increased costs from water scarcity
   - Infrastructure damage from extreme events
2. Estimate probability-weighted impacts
3. Calculate Net Present Value impacts
4. Quantify range (low, medium, high scenarios)

**Phase 4: Resilience Assessment**
1. Identify adaptation options
   - Engineering solutions
   - Operational changes
   - Insurance/risk transfer
2. Test strategy under different scenarios
3. Identify "no-regret" actions (beneficial under all scenarios)
4. Develop staged implementation plan

**Phase 5: Disclosure**
1. Document methodology (data sources, scenarios, methods)
2. Present quantitative results (where possible)
3. Explain qualitative factors (where quantification not feasible)
4. Describe governance and oversight
5. Establish metrics and targets for ongoing monitoring

---

### Specific Disclosure Examples

**Example 1: Heat Stress Risk**

*Context: Mining operation, 2500 employees, 24/7 operations*

**Physical Risk:**
"Analysis of ACCESS-CM2 climate projections shows days >37°C for >3 hours increasing from 
current 15 days/year to 25 days/year by 2030 (SSP1-2.6) or 35 days/year by 2030 (SSP5-8.5).  
By 2050, projections show 45 days/year (SSP1-2.6) to 75 days/year (SSP5-8.5)."

**Financial Impact:**
"Based on current heat stress protocols, additional heat days result in estimated productivity 
loss of $X million per year by 2030 and $Y million per year by 2050 (SSP2-4.5 scenario)."

**Resilience Measures:**
"Adaptation strategies under evaluation include enhanced cooling facilities ($A million), 
flexible work rosters ($B million), and heat-resilient equipment upgrades ($C million).  
These measures would reduce financial impact to $D million per year under SSP5-8.5."

---

**Example 2: Water Availability Risk**

*Context: Processing plant requires 2 ML/day water, sourced from local creek*

**Physical Risk:**
"Climate projections (ACCESS-CM2) indicate annual rainfall decreasing by 5-15% by 2040 under 
SSP2-4.5 scenario, with longer consecutive dry days (increase from 60 to 75 days).  
Concurrent temperature increase (+1.5°C) raises evaporation, further reducing water yield."

**Financial Impact:**
"Hydrological modeling indicates creek yield may be insufficient 2-3 months/year by 2040.  
Estimated impact: production curtailment ($X million per event) or alternative water sourcing 
costs ($Y million per year)."

**Resilience Measures:**
"Planned adaptation: expand on-site water storage ($Z million) to carry through dry periods, 
and water recycling improvements (reduce consumption by 30%).  Residual risk: extreme drought 
events under SSP5-8.5 may still cause curtailment."

---

### Key Considerations for Mining Sector

**Long-Lived Infrastructure:**
- Tailings storage facilities: Multi-century time horizon
- Dams and water infrastructure: 50-100+ year lifetime
- Must assess long-term scenarios (SSP3-7.0, SSP5-8.5)

**Climate-Sensitive Operations:**
- Heat stress management
- Water availability
- Flood risks
- Dust generation (wind, soil moisture)

**Closure and Post-Closure:**
- Climate impacts on closure plans
- Long-term environmental management
- Rehabilitation success under changing climate

**Supply Chain and Transportation:**
- Port access (sea level rise, cyclones)
- Road/rail climate resilience
- Workforce health and safety

---

### Materiality Assessment

**Financial Materiality Factors:**
- Magnitude of financial impact (absolute and %)
- Likelihood of occurrence
- Timeframe of impact
- Ability to adapt

**Threshold Guidance:**
- >5% EBITDA impact: Likely material
- 1-5% EBITDA impact: Possibly material (assess)
- <1% EBITDA impact: Likely not material (unless cascading effects)

**Time Horizon:**
- Near-term impacts: Higher weight in materiality
- Long-term impacts: Consider if irreversible or affects asset value

**Board-Level Question:**
"Would this information influence investor decisions about the entity?"

If yes → Material → Disclose

---

### Governance and Oversight

**Board-Level Responsibilities:**
- Oversight of climate-related risks and opportunities
- Integration into enterprise risk management
- Review and approval of scenario analysis
- Setting climate-related targets
- Monitoring progress on adaptation measures

**Management Responsibilities:**
- Conducting scenario analysis
- Quantifying physical risks
- Developing adaptation strategies
- Implementing risk management
- Regular reporting to board

**Documentation:**
- Board papers on climate risk
- Risk register entries
- Scenario analysis reports
- Adaptation strategy documents
- Monitoring dashboards

---

### External Assurance Considerations

**What May Require Assurance:**
- GHG emissions (Scope 1, 2, 3)
- Key metrics and calculations
- Scenario analysis methodology

**What Auditors Will Review:**
- Data sources and quality
- Calculation methods
- Assumptions and judgements
- Consistency with other disclosures
- Completeness of material risks

**Best Practice:**
- Document everything thoroughly
- Use credible data sources (CMIP6, AGCD)
- Apply standard methodologies
- Engage climate experts for complex assessments
- Consider pre-assurance review

---

### Resources and Standards

**Primary Standards:**
- AASB S2 (Australian standard)
- IFRS S2 (international standard)
- TCFD Recommendations (Task Force on Climate-related Financial Disclosures)

**Guidance Documents:**
- TCFD Technical Supplement on Scenario Analysis
- ISSB Educational Material
- Industry-specific guidance (mining/resources)

**Climate Science Resources:**
- IPCC Assessment Reports (AR6)
- CSIRO Climate projections
- This application (for physical risk inputs)
""")

    # ========================================================================
    # REFERENCES AND SUPPORT
    # ========================================================================

    st.markdown("---")
    st.markdown("## 📚 References and Resources")

    with st.expander("**Key Scientific Literature**", expanded=False):
        st.markdown("""
### IPCC Assessment Reports

**IPCC AR6 Working Group I (2021)**  
*Climate Change 2021: The Physical Science Basis*  
Comprehensive assessment of climate science  
[https://www.ipcc.ch/report/ar6/wg1/](https://www.ipcc.ch/report/ar6/wg1/)

**IPCC AR6 Working Group II (2022)**  
*Climate Change 2022: Impacts, Adaptation and Vulnerability*  
Assessment of climate change impacts and adaptation options  
[https://www.ipcc.ch/report/ar6/wg2/](https://www.ipcc.ch/report/ar6/wg2/)

**IPCC Special Report on 1.5°C (2018)**  
*Global Warming of 1.5°C*  
Impacts of 1.5°C vs higher warming levels  
[https://www.ipcc.ch/sr15/](https://www.ipcc.ch/sr15/)

---

### CMIP6 and Climate Modeling

**Eyring et al. (2016)**  
*Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization*  
Geoscientific Model Development, 9, 1937-1958  
[https://doi.org/10.5194/gmd-9-1937-2016](https://doi.org/10.5194/gmd-9-1937-2016)

**Bi et al. (2020)**  
*Configuration and spin-up of ACCESS-CM2*  
Journal of Southern Hemisphere Earth Systems Science, 70(1), 225-251  
[https://doi.org/10.1071/ES19040](https://doi.org/10.1071/ES19040)

**O'Neill et al. (2016)**  
*The Scenario Model Intercomparison Project (ScenarioMIP) for CMIP6*  
Geoscientific Model Development, 9, 3461-3482  
[https://doi.org/10.5194/gmd-9-3461-2016](https://doi.org/10.5194/gmd-9-3461-2016)

---

### Bias Correction Methods

**Cannon et al. (2015)**  
*Bias Correction of GCM Precipitation by Quantile Mapping*  
Climate Research, 64, 211-228  
[https://doi.org/10.3354/cr01312](https://doi.org/10.3354/cr01312)

**Maraun & Widmann (2018)**  
*Statistical Downscaling and Bias Correction for Climate Research*  
Cambridge University Press  
[https://doi.org/10.1017/9781107588783](https://doi.org/10.1017/9781107588783)

**Maraun et al. (2017)**  
*Bias Correcting Climate Change Simulations - a Critical Review*  
Current Climate Change Reports, 3, 211-220  
[https://doi.org/10.1007/s40641-017-0064-x](https://doi.org/10.1007/s40641-017-0064-x)

---

### Australian Climate Projections

**Grose et al. (2020)**  
*Insights from CMIP6 for Australia's future climate*  
Earth's Future, 8, e2019EF001469  
[https://doi.org/10.1029/2019EF001469](https://doi.org/10.1029/2019EF001469)

**Evans et al. (2014)**  
*Design of a regional climate modelling projection ensemble*  
Geoscientific Model Development, 7, 621-629  
[https://doi.org/10.5194/gmd-7-621-2014](https://doi.org/10.5194/gmd-7-621-2014)

**Jones et al. (2009)**  
*High-quality spatial climate data-sets for Australia*  
Australian Meteorological and Oceanographic Journal, 58, 233-248

---

### Regional Warming Amplification

**Sutton et al. (2007)**  
*Land/sea warming ratio in response to climate change: IPCC AR4 model results and comparison with observations*  
Geophysical Research Letters, 34, L02701  
[https://doi.org/10.1029/2006GL028164](https://doi.org/10.1029/2006GL028164)

**Byrne & O'Gorman (2018)**  
*Trends in continental temperature and humidity directly linked to ocean warming*  
Proceedings of the National Academy of Sciences, 115(19), 4863-4868  
[https://doi.org/10.1073/pnas.1722312115](https://doi.org/10.1073/pnas.1722312115)

---

### Climate Risk Assessment

**TCFD (2017)**  
*Final Report: Recommendations of the Task Force on Climate-related Financial Disclosures*  
[https://www.fsb-tcfd.org/](https://www.fsb-tcfd.org/)

**Lempert et al. (2013)**  
*Ensuring Robust Flood Risk Management in Ho Chi Minh City*  
World Bank Policy Research Working Paper 6465

**Hallegatte et al. (2012)**  
*Investment Decision Making Under Deep Uncertainty*  
World Bank Policy Research Working Paper 6193
""")

    with st.expander("**Data Access and Tools**", expanded=False):
        st.markdown("""
### Climate Data Sources

**CMIP6 Archive**  
[https://esgf-node.llnl.gov/projects/cmip6/](https://esgf-node.llnl.gov/projects/cmip6/)  
Primary archive for CMIP6 model output

**Copernicus Climate Data Store (CDS)**  
[https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)  
User-friendly interface for CMIP6 and other climate datasets

**Bureau of Meteorology Climate Data Online**  
[http://www.bom.gov.au/climate/data/](http://www.bom.gov.au/climate/data/)  
Australian observational data (AGCD and station data)

**Australian Climate Service**  
[https://www.acs.gov.au/](https://www.acs.gov.au/)  
Climate information for Australia (replacing Climate Change in Australia)

---

### Analysis Tools and Software

**Climate Data Operators (CDO)**  
[https://code.mpimet.mpg.de/projects/cdo](https://code.mpimet.mpg.de/projects/cdo)  
Command-line tools for climate data processing

**NCAR Command Language (NCL)**  
[https://www.ncl.ucar.edu/](https://www.ncl.ucar.edu/)  
Analysis and visualization of climate data

**Python Climate Libraries**
- **xarray:** [http://xarray.pydata.org/](http://xarray.pydata.org/)
- **Cartopy:** [https://scitools.org.uk/cartopy/](https://scitools.org.uk/cartopy/)
- **Pandas:** [https://pandas.pydata.org/](https://pandas.pydata.org/)
- **NumPy:** [https://numpy.org/](https://numpy.org/)

---

### Australian Resources

**CSIRO**  
[https://www.csiro.au/en/research/natural-environment/climate-change](https://www.csiro.au/en/research/natural-environment/climate-change)  
Australian climate change research

**Bureau of Meteorology**  
[http://www.bom.gov.au/climate/](http://www.bom.gov.au/climate/)  
Official Australian climate information

**NARCliM (NSW/ACT)**  
[https://climatechange.environment.nsw.gov.au/narclim](https://climatechange.environment.nsw.gov.au/narclim)  
Regional climate projections for NSW/ACT

**Queensland Future Climate Portal**  
[https://www.longpaddock.qld.gov.au/qld-future-climate/](https://www.longpaddock.qld.gov.au/qld-future-climate/)  
Queensland-specific climate projections
""")

    with st.expander("**Regulatory and Compliance Resources**", expanded=False):
        st.markdown("""
### Climate Disclosure Standards

**AASB S2 Climate-related Disclosures**  
[https://www.aasb.gov.au/](https://www.aasb.gov.au/)  
Australian Accounting Standards Board

**IFRS S2 Climate-related Disclosures**  
[https://www.ifrs.org/issued-standards/ifrs-sustainability-standards-navigator/ifrs-s2-climate-related-disclosures/](https://www.ifrs.org/issued-standards/ifrs-sustainability-standards-navigator/ifrs-s2-climate-related-disclosures/)  
International Sustainability Standards Board

**TCFD Recommendations**  
[https://www.fsb-tcfd.org/](https://www.fsb-tcfd.org/)  
Task Force on Climate-related Financial Disclosures

---

### Implementation Guidance

**TCFD Knowledge Hub**  
[https://www.tcfdhub.org/](https://www.tcfdhub.org/)  
Resources and case studies for TCFD implementation

**ISSB Educational Materials**  
[https://www.ifrs.org/supporting-implementation/supporting-materials-by-ifrs-sustainability-disclosure-standard/ifrs-s2-climate-related-disclosures/](https://www.ifrs.org/supporting-implementation/supporting-materials-by-ifrs-sustainability-disclosure-standard/ifrs-s2-climate-related-disclosures/)  
Guidance on applying IFRS S2

**ASIC Regulatory Guide 247**  
*Effective disclosure in an operating and financial review*  
Includes climate risk disclosure expectations

---

### Industry-Specific Guidance

**Minerals Council of Australia**  
Climate change resources and position papers

**International Council on Mining & Metals (ICMM)**  
*Climate Change Position Statement*  
[https://www.icmm.com/en-gb/our-work/climate-action](https://www.icmm.com/en-gb/our-work/climate-action)

**Global Reporting Initiative (GRI)**  
GRI 305: Emissions  
[https://www.globalreporting.org/](https://www.globalreporting.org/)
""")

    # ========================================================================
    # VERSION AND CONTACT
    # ========================================================================

    st.markdown("---")
    st.markdown("""
## 📞 Support and Feedback

### Application Version
**Version:** 2.0  
**Last Updated:** 2025-11-08  
**Data:** CMIP6 ACCESS-CM2 r1i1p1f1, AGCD

### Disclaimer

This tool provides climate projection information for planning and analysis purposes.  Users are 
responsible for:
- Verifying appropriateness for their specific application
- Understanding limitations and uncertainties
- Seeking expert advice for critical decisions
- Proper citation of data sources in publications
- Compliance with relevant regulations and standards

**No Warranty:**
Climate projections contain uncertainties and limitations.  This tool is provided "as is" 
without warranty of any kind, express or implied.

### Data Citation

When using data from this application, please cite:

**Climate Model Data:**
Bi, D., et al. (2020). ACCESS-CM2. CMIP6. Available at: 
https://esgf-node.llnl.gov/search/cmip6/

**Observational Data:**
Australian Bureau of Meteorology (2024). Australian Gridded Climate Data (AGCD). 
Available at: http://www.bom.gov.au/climate/data/

**Application:**
Climate Metrics Viewer Version 2.0 (2025). Ravenswood Gold Mine, Queensland, Australia.

---

*This user guide provides comprehensive technical documentation for the Climate Metrics Viewer.  
For specific applications or detailed assessments, consider consulting with climate scientists 
and risk assessment specialists.*

*Last updated: 2025-11-08*
""")


def render_help_button(st, namespace="main"):
    """
    Render help button in sidebar to open user guide.

    Parameters
    ----------
    st : streamlit module
        Streamlit instance
    namespace : str
        Namespace for session state keys
    """
    if st.button(
            "📖 Open Scientific User Guide",
            key=f"{namespace}_help_button",
            help="Open comprehensive scientific documentation",
            use_container_width=True
    ):
        st.session_state[f"{namespace}_show_guide"] = True
        st.rerun()