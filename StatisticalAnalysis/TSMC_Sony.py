import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
from scipy.signal import correlate

# Helper function to process data
def process_data(data, exclude_conditions=None):
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)
    df["YoY_Growth"] = df["Revenue"].pct_change(periods=4) * 100
    df = df.dropna().reset_index(drop=True)
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.quarter
    if exclude_conditions is not None:
        df = df[~exclude_conditions(df)]
    chronological_data = []
    for _, row in df.iterrows():
        year = row["Year"]
        quarter = row["Quarter"]
        growth = row["YoY_Growth"]
        chronological_data.append((year, quarter, growth))
    chronological_data.sort(key=lambda x: (x[0], x[1]))
    line_x = [item[0] + (item[1] - 1) * 0.25 for item in chronological_data]
    line_y = [item[2] for item in chronological_data]
    return line_x, line_y

# TSMC Data
tsmc_data = {
    "Date": [
        "2024-09-30", "2024-06-30", "2024-03-31", "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31",
        "2022-12-31", "2022-09-30", "2022-06-30", "2022-03-31", "2021-12-31", "2021-09-30", "2021-06-30",
        "2021-03-31", "2020-12-31", "2020-09-30", "2020-06-30", "2020-03-31", "2019-12-31", "2019-09-30",
        "2019-06-30", "2019-03-31", "2018-12-31", "2018-09-30", "2018-06-30", "2018-03-31", "2017-12-31",
        "2017-09-30", "2017-06-30", "2017-03-31", "2016-12-31", "2016-09-30", "2016-06-30", "2016-03-31",
        "2015-12-31", "2015-09-30", "2015-06-30", "2015-03-31", "2014-12-31", "2014-09-30", "2014-06-30",
        "2014-03-31", "2013-12-31", "2013-09-30", "2013-06-30", "2013-03-31", "2012-12-31", "2012-09-30",
        "2012-06-30", "2012-03-31", "2011-12-31", "2011-09-30", "2011-06-30", "2011-03-31", "2010-12-31",
        "2010-09-30", "2010-06-30", "2010-03-31", "2009-12-31", "2009-09-30", "2009-06-30", "2009-03-31"
    ],
    "Revenue": [
        10083, 7658, 7170, 8412, 6668, 5927, 6810, 7774, 9240, 8059, 7238, 5961, 5610, 4810, 4973,
        5575, 4682, 4048, 3884, 4444, 3244, 2150, 1995, 3457, 2904, 2429, 3071, 3657, 2977, 2194,
        2812, 2977, 3058, 2240, 1963, 2214, 2358, 2581, 2504, 1944, 2542, 1982, 1580, 1342, 1740,
        1736, 1350, 1524, 1652, 1412, 1131, 1043, 1044, 1246, 1241, 1340, 1470, 1266, 1053, 1018,
        932, 738, 46
    ]
}

tsmc_exclude_conditions = lambda df: (
    ((df["Year"] == 2010) & (df["Quarter"] == 1))
)
tsmc_x, tsmc_y = process_data(tsmc_data, exclude_conditions=tsmc_exclude_conditions)

# Sony Data
sony_data = {
    "Date": [
        "2024-09-30", "2024-06-30", "2024-03-31",
        "2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31",
        "2022-12-31", "2022-09-30", "2022-06-30", "2022-03-31",
        "2021-12-31", "2021-09-30", "2021-06-30", "2021-03-31",
        "2020-12-31", "2020-09-30", "2020-06-30", "2020-03-31",
        "2019-12-31", "2019-09-30", "2019-06-30", "2019-03-31",
        "2018-12-31", "2018-09-30", "2018-06-30", "2018-03-31",
        "2017-12-31", "2017-09-30", "2017-06-30", "2017-03-31",
        "2016-12-31", "2016-09-30", "2016-06-30", "2016-03-31",
        "2015-12-31", "2015-09-30", "2015-06-30", "2015-03-31",
        "2014-12-31", "2014-09-30", "2014-06-30", "2014-03-31",
        "2013-12-31", "2013-09-30", "2013-06-30", "2013-03-31",
        "2012-12-31", "2012-09-30", "2012-06-30", "2012-03-31",
        "2011-12-31", "2011-09-30", "2011-06-30", "2011-03-31",
        "2010-12-31", "2010-09-30", "2010-06-30", "2010-03-31",
        "2009-12-31", "2009-09-30", "2009-06-30", "2009-03-31"
    ],
    "Revenue": [
        19468, 19275, 23208,
        25483, 19518, 21635, 23551,
        24232, 19814, 17799, 19527,
        26676, 21561, 20537, 20530,
        25887, 19867, 18311, 16069,
        22661, 19737, 17524, 18997,
        21376, 19645, 17973, 17826,
        23784, 18563, 16723, 17658,
        20668, 16722, 15662, 15177,
        21507, 15773, 14820, 18187,
        21213, 17445, 17920, 19277,
        22979, 18117, 17300, 18436,
        22391, 20573, 19180, 19517,
        23370, 20454, 18456, 19046,
        27238, 20881, 18663, 18442,
        24325, 18458, 16665, 15552
    ]
}


sony_exclude_conditions = lambda df: (
    ((df["Year"] == 2010) & (df["Quarter"] == 1))
)
sony_x, sony_y = process_data(sony_data, exclude_conditions=sony_exclude_conditions)


# --- Correlation of Direction of YoY Growth with Lag Analysis ---

# Interpolate the data to a common timeline
common_time = np.intersect1d(tsmc_x, sony_x)
tsmc_interp = np.interp(common_time, tsmc_x, tsmc_y)
sony_interp = np.interp(common_time, sony_x, sony_y)

# Calculate the direction of change (1 for up, -1 for down, 0 for no change)
tsmc_direction = np.sign(np.diff(tsmc_interp))
sony_direction = np.sign(np.diff(sony_interp))

# Cross-correlation for direction analysis
lags = np.arange(-len(tsmc_direction) + 1, len(tsmc_direction))
cross_corr = correlate(tsmc_direction, sony_direction, mode='full')
max_corr_idx = np.argmax(cross_corr)
best_lag = lags[max_corr_idx]
max_corr = cross_corr[max_corr_idx]

# Adjust the direction arrays based on the lag
if best_lag > 0:
    # Sony lags TSMC: shift Sony forward
    sony_shifted = np.pad(sony_direction, (best_lag, 0), mode='constant', constant_values=0)[:len(tsmc_direction)]
    tsmc_shifted = tsmc_direction
elif best_lag < 0:
    # Sony leads TSMC: shift TSMC forward
    tsmc_shifted = np.pad(tsmc_direction, (-best_lag, 0), mode='constant', constant_values=0)[:len(sony_direction)]
    sony_shifted = sony_direction
else:
    # No lag adjustment needed
    tsmc_shifted = tsmc_direction
    sony_shifted = sony_direction

# Calculate Pearson correlation for the aligned direction arrays
valid_indices = ~np.isnan(tsmc_shifted) & ~np.isnan(sony_shifted)
direction_correlation, direction_p_value = pearsonr(tsmc_shifted[valid_indices], sony_shifted[valid_indices])

# Calculate lagged Pearson correlation for original YoY growth data
if best_lag > 0:
    # Sony lags TSMC
    lagged_tsmc = tsmc_interp[:-best_lag]
    lagged_sony = sony_interp[best_lag:]
elif best_lag < 0:
    # Sony leads TSMC
    lagged_tsmc = tsmc_interp[-best_lag:]
    lagged_sony = sony_interp[:best_lag]
else:
    # No lag adjustment needed
    lagged_tsmc = tsmc_interp
    lagged_sony = sony_interp

# Valid indices for lagged Pearson correlation
valid_indices_lagged = ~np.isnan(lagged_tsmc) & ~np.isnan(lagged_sony)
lagged_correlation, lagged_p_value = pearsonr(lagged_tsmc[valid_indices_lagged], lagged_sony[valid_indices_lagged])

# Print results
print(f"Best Lag: {best_lag} Quarters")
print(f"Maximum Cross-Correlation for Directions: {max_corr:.2f}")
print(f"Direction Correlation: {direction_correlation:.2f}")
print(f"P-value for Direction Correlation: {direction_p_value:.2e}")
print(f"Lagged Pearson Correlation: {lagged_correlation:.2f}")
print(f"P-value for Lagged Pearson Correlation: {lagged_p_value:.2e}")

# --- Plotting Direction Correlation with Lag ---
plt.figure(figsize=(14, 8))

# Plot the shifted directions for visualization
plt.plot(common_time[1:], tsmc_shifted, label="TSMC Direction of Change (Shifted)", color="blue", linestyle="--")
plt.plot(common_time[1:], sony_shifted, label="Sony Direction of Change (Shifted)", color="green", linestyle="--")

# Labels and legend
plt.title(f"Direction of YoY Growth Changes (TSMC vs. Sony, Lag = {best_lag} Quarters)")
plt.xlabel("Time (Years)")
plt.ylabel("Direction (1 = Up, -1 = Down, 0 = No Change)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Cross-Correlation Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(lags, cross_corr, label="Cross-Correlation (Directions)", color="purple")
plt.axvline(best_lag, color="red", linestyle="--", label=f"Best Lag = {best_lag}")
plt.title("Cross-Correlation of Direction of YoY Growth Changes")
plt.xlabel("Lag (Quarters)")
plt.ylabel("Cross-Correlation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting YoY Growth of TSMC and Sony
plt.figure(figsize=(14, 8))

# Plot interpolated YoY growth data
plt.plot(common_time, tsmc_interp, label="TSMC YoY Growth", color="blue", linestyle="--")
plt.plot(common_time, sony_interp, label="Sony YoY Growth", color="green", linestyle="--")

# Overlay original actual data points
plt.scatter(tsmc_x, tsmc_y, color="blue", label="TSMC Actual Data Points", alpha=0.5)
plt.scatter(sony_x, sony_y, color="green", label="Sony Actual Data Points", alpha=0.5)

# Add labels and legend
plt.title("YoY Growth of TSMC and Sony")
plt.xlabel("Time (Years)")
plt.ylabel("YoY Growth (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()