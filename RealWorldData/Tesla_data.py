import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# Data
data = {
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
    ],
    "Revenue": [
        25182, 25500, 21301,
        25167, 23350, 24927, 23329,
        24318, 21454, 16934, 18756,
        17719, 13757, 11958, 10389,
        10744, 8771, 6036, 5985,
        7384, 6303, 6350, 4541,
        7226, 6824, 4002, 3409,
        3288, 2985, 2790, 2696,
        2285, 2298, 1270, 1147,
        1214, 937, 955, 940,
        957, 852, 769, 621,
        615, 431, 405, 562,
        306, 50, 27, 30,
        39, 58, 58, 49,
        36, 31, 28, 21,
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date").reset_index(drop=True)

# Calculate Quarterly YoY Growth
df["YoY_Growth"] = df["Revenue"].pct_change(periods=4) * 100

# Drop NaN values
df = df.dropna().reset_index(drop=True)

# Extract year and quarter
df["Year"] = df["Date"].dt.year
df["Quarter"] = df["Date"].dt.quarter

# Pivot to group by Year and Quarter
pivot_df = df.pivot(index="Year", columns="Quarter", values="YoY_Growth")
print("Year-Over-Year (YoY) Growth Percentages by Quarter:")
print(pivot_df)

# Flatten pivot_df into a chronological list of (year, quarter, value)
chronological_data = []
for year in pivot_df.index:
    for q in pivot_df.columns:
        val = pivot_df.loc[year, q]
        if not np.isnan(val):
            chronological_data.append((year, q, val))

# Sort by year and quarter
chronological_data.sort(key=lambda x: (x[0], x[1]))

# Exclude year 2013 and Q4 of 2012
chronological_data = [item for item in chronological_data if item[0] != 2013 and not (item[0] == 2012 and item[1] == 4)]

# Convert year/quarter into x_positions: x = Year + (Quarter-1)*0.25
line_x = [item[0] + (item[1]-1)*0.25 for item in chronological_data]
line_y = [item[2] for item in chronological_data]

# Remove the first point
line_x = line_x[1:]
line_y = line_y[1:]

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the line and markers
ax.plot(line_x, line_y, color='black', marker='o', linewidth=1)

# Formatting
ax.set_title("Year-Over-Year (YoY) Quarterly Revenue Growth (No first point, no bars, no 2013, exclude 2012 Q4)")
ax.set_xlabel("Time (Year with quarters as fractions)")
ax.set_ylabel("YoY Growth (%)")
ax.set_ylim(-80, 160)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show only whole years on x-axis
years = sorted(set([int(x) for x in line_x]))
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], rotation=45)

plt.tight_layout()
plt.show()
