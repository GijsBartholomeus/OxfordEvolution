# ChicoRescalePeriod Notebook

## Overview

This notebook is a modified version of `ChicoOscillation.ipynb` that adds **oscillation detection** and **adaptive window rescaling** to include exactly 2 wavelengths when oscillations are detected.

## Key Modifications

### 1. **Oscillation Detection Function** (from PeakDetectionComparison.ipynb)
- Added `estimate_period_original()` function that detects oscillations using:
  - Peak detection with prominence-based filtering
  - Regularity assessment (CV < 0.3 for periods and amplitudes)
  - Returns detected period and coarse-grained cycle data

### 2. **Adaptive Window Rescaling** 
- New function: `simulate_and_extract_rescaled()`
- **Workflow:**
  1. Simulate the full time series
  2. Detect oscillations using `estimate_period_original()`
  3. If oscillation detected AND 2 wavelengths fit in available data:
     - Rescale window to contain exactly `2 * period` minutes
     - Extract window from end of simulation backwards
  4. If no oscillation OR insufficient data:
     - Use original windowing (as in ChicoOscillation)
  5. Coarse-grain the (possibly rescaled) window
  6. Calculate complexity

### 3. **Oscillation Info Tracking**
Each analysis returns `oscillation_info` dict containing:
- `detected`: bool - whether oscillation was found
- `period`: float or None - detected period in minutes
- `rescaled`: bool - whether window was actually rescaled
- `original_window`: (start, end) tuple
- `rescaled_window`: (start, end) tuple or None

### 4. **Batch Processing Updates**
- Worker function tracks oscillation statistics:
  - `oscillation_detected_count`: # of samples with detected oscillations
  - `rescaled_count`: # of samples where window was rescaled to 2λ
- Merge function aggregates these counts across all workers

### 5. **Visualization**
- Added new cell: "**OSCILLATION DETECTION VISUALIZATION**"
- Demonstrates the rescaling process with real examples:
  - Shows 3 oscillating examples (with period info and window rescaling)
  - Shows 2 non-oscillating examples (uses original windowing)
  - Visualizes both full time series and coarse-grained data
  - Reports detailed statistics about detected periods and window rescaling

### 6. **Statistics Reporting**
- Added to results output:
  - `🔍 Oscillations detected: X (Y% of successful)`
  - `📏 Windows rescaled to 2λ: X (Y% of oscillating)`

## Usage

Run the notebook as you would ChicoOscillation.ipynb:

1. **Testing Section**: Will demonstrate oscillation detection on wildtype and randomized samples
2. **Visualization Section**: Shows examples of detected oscillations and rescaling
3. **Main Analysis**: Processes `SAMPLING_SIZE` samples with parallel processing
4. **Results**: Reports complexity distribution + oscillation statistics

## Key Differences from ChicoOscillation

| Feature | ChicoOscillation | ChicoRescalePeriod |
|---------|------------------|-------------------|
| Oscillation Detection | ❌ None | ✅ Peak-based detection |
| Window Size | Fixed (e.g., 500 min) | Adaptive (2λ when oscillating) |
| Period Information | Not tracked | Tracked and reported |
| Complexity Calculation | Direct coarse-graining | Coarse-graining on rescaled window |

## Expected Behavior

- **For oscillating dynamics**: Window rescaled to 2 wavelengths → complexity calculated on ~2 periods
- **For non-oscillating dynamics**: Original windowing → complexity calculated on fixed time window
- **For irregular/noisy dynamics**: Oscillation detection fails regularity check → uses original windowing

## Parameters

Key parameters controlling oscillation detection (in `estimate_period_original`):
- `require_peaks=3`: Minimum peaks needed
- Prominence threshold: 5% of signal range
- Regularity thresholds: CV < 0.3 for periods and amplitudes

## Notes

- The function maintains **backwards compatibility** by keeping `simulate_and_extract_chico()` as a wrapper
- Oscillation detection is performed on the **full simulation** before windowing
- If 2 wavelengths don't fit in available data, uses original window (no partial wavelengths)
- Minimum 10 points required in rescaled window for coarse-graining

## Created From

- **Base**: `ChicoOscillation.ipynb`
- **Oscillation Detection**: `PeakDetectionComparison.ipynb` (original method)
- **Date**: 2025
