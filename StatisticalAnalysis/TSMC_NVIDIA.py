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

# NVIDIA Data
nvidia_data = {
    "Date": [
        "2024-10-31", "2024-07-31", "2024-04-30", "2024-01-31",
        "2023-10-31", "2023-07-31", "2023-04-30", "2023-01-31",
        "2022-10-31", "2022-07-31", "2022-04-30", "2022-01-31",
        "2021-10-31", "2021-07-31", "2021-04-30", "2021-01-31",
        "2020-10-31", "2020-07-31", "2020-04-30", "2020-01-31",
        "2019-10-31", "2019-07-31", "2019-04-30", "2019-01-31",
        "2018-10-31", "2018-07-31", "2018-04-30", "2018-01-31",
        "2017-10-31", "2017-07-31", "2017-04-30", "2017-01-31",
        "2016-10-31", "2016-07-31", "2016-04-30", "2016-01-31",
        "2015-10-31", "2015-07-31", "2015-04-30", "2015-01-31",
        "2014-10-31", "2014-07-31", "2014-04-30", "2014-01-31",
        "2013-10-31", "2013-07-31", "2013-04-30", "2013-01-31",
        "2012-10-31", "2012-07-31", "2012-04-30", "2012-01-31",
        "2011-10-31", "2011-07-31", "2011-04-30", "2011-01-31",
        "2010-10-31", "2010-07-31", "2010-04-30", "2010-01-31",
        "2009-10-31", "2009-07-31", "2009-04-30", "2009-01-31"
    ],
    "Revenue": [
        22347, 19075, 17319, 14002,
        10789, 7165, 2524, 1682,
        1007, 877, 2202, 3279,
        2969, 2730, 2237, 1795,
        1697, 1055, 1083, 1096,
        1019, 663, 449, 372,
        1126, 1216, 1352, 1127,
        944, 737, 601, 760,
        693, 370, 298, 303,
        301, 132, 237, 293,
        275, 226, 213, 226,
        204, 167, 143, 242,
        309, 196, 127, 174,
        249, 226, 203, 226,
        151, -129, 195, 182,
        157, -61, -180, -127
    ]
}

nvidia_exclude_conditions = lambda df: (
    ((df["Year"] == 2023) & (df["Quarter"] >= 3)) |
    ((df["Year"] == 2024) & (df["Quarter"] <= 2)) |
    ((df["Year"] == 2010) & (df["Quarter"] <= 3)) |
    ((df["Year"] == 2011) & (df["Quarter"] == 3))
)
nvidia_x, nvidia_y = process_data(nvidia_data, exclude_conditions=nvidia_exclude_conditions)


common_time = np.intersect1d(tsmc_x, nvidia_x)
tsmc_interp = np.interp(common_time, tsmc_x, tsmc_y)
nvidia_interp = np.interp(common_time, nvidia_x, nvidia_y)

# --- Correlation of Direction of YoY Growth ---

# Calculate the direction of change (1 for up, -1 for down, 0 for no change)
tsmc_direction = np.sign(tsmc_interp)
nvidia_direction = np.sign(nvidia_interp)

# --- Cross-Correlation Analysis for Direction ---
# Calculate cross-correlation to find lag
lags = np.arange(-len(tsmc_direction) + 1, len(tsmc_direction))
cross_corr = correlate(tsmc_direction, nvidia_direction, mode='full')
max_corr_idx = np.argmax(cross_corr)
max_corr = cross_corr[max_corr_idx]
best_lag_max = lags[max_corr_idx]

# Print results
print(f"Best Lag: {best_lag_max} Quarters")
print(f"Maximum Cross-Correlation for Direction: {max_corr:.2f}")

# Shift arrays based on lag
if best_lag_max > 0:
    # NVIDIA lags TSMC: shift NVIDIA forward
    nvidia_shifted = np.pad(nvidia_direction, (best_lag_max, 0), mode='constant', constant_values=0)[:len(tsmc_direction)]
    tsmc_shifted = tsmc_direction
elif best_lag_max < 0:
    # NVIDIA leads TSMC: shift TSMC forward
    tsmc_shifted = np.pad(tsmc_direction, (-best_lag_max, 0), mode='constant', constant_values=0)[:len(nvidia_direction)]
    nvidia_shifted = nvidia_direction
else:
    # No lag adjustment needed
    tsmc_shifted = tsmc_direction
    nvidia_shifted = nvidia_direction

# Calculate Pearson correlation for aligned directions
valid_indices = ~np.isnan(tsmc_shifted) & ~np.isnan(nvidia_shifted)
direction_correlation, direction_p_value = pearsonr(tsmc_shifted[valid_indices], nvidia_shifted[valid_indices])

# Print results
print(f"Direction Correlation: {direction_correlation:.2f}")
print(f"P-value for Direction Correlation: {direction_p_value:.2e}")

# --- Plotting Direction Correlation ---
plt.figure(figsize=(14, 8))

# Plot the shifted directions for visualization
plt.plot(common_time, tsmc_shifted, label="TSMC Direction of Change (Shifted)", color="blue", linestyle="--")
plt.plot(common_time, nvidia_shifted, label="NVIDIA Direction of Change (Shifted)", color="green", linestyle="--")

# Labels and legend
plt.title(f"Direction of YoY Growth Changes (TSMC vs. NVIDIA, Lag = {best_lag_max} Quarters)")
plt.xlabel("Time (Years)")
plt.ylabel("Direction (1 = Up, -1 = Down, 0 = No Change)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Cross-Correlation Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(lags, cross_corr, label="Cross-Correlation (Direction)", color="purple")
plt.axvline(best_lag_max, color="red", linestyle="--", label=f"Best Lag = {best_lag_max}")
plt.title("Cross-Correlation of Direction of YoY Growth Changes")
plt.xlabel("Lag (Quarters)")
plt.ylabel("Cross-Correlation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
