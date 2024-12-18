import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
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

# Filter out Q3/Q4 of 2023, Q1/Q2 of 2024, Q1-Q3 of 2010, and Q3 of 2011
exclude_conditions = (
    ((df["Year"] == 2023) & (df["Quarter"] >= 3)) |
    ((df["Year"] == 2024) & (df["Quarter"] <= 2)) |
    ((df["Year"] == 2010) & (df["Quarter"] <= 3)) |
    ((df["Year"] == 2011) & (df["Quarter"] == 3))
)
df = df[~exclude_conditions]

# Pivot to group by Year and Quarter
pivot_df = df.pivot(index="Year", columns="Quarter", values="YoY_Growth")

# Flatten pivot_df into chronological list
chronological_data = []
for year in pivot_df.index:
    for q in pivot_df.columns:
        val = pivot_df.loc[year, q]
        if not np.isnan(val):
            chronological_data.append((year, q, val))

# Sort by year and quarter
chronological_data.sort(key=lambda x: (x[0], x[1]))

# Convert year/quarter into x_positions
line_x = [item[0] + (item[1]-1)*0.25 for item in chronological_data]
line_y = [item[2] for item in chronological_data]

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the line and markers
ax.plot(line_x, line_y, color='black', marker='o', linewidth=1)

# Formatting
ax.set_title("NVIDIA Year-Over-Year (YoY) Quarterly Revenue Growth")
ax.set_xlabel("Time")
ax.set_ylabel("YoY Growth (%)")
ax.set_ylim(-100, 220)
ax.axhline(0, color="black", linestyle="--", linewidth=1)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Custom x-tick labeling
years = sorted(set([int(x) for x in line_x]))
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], rotation=45)

plt.tight_layout()
plt.show()
