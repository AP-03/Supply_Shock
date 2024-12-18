import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

def plot_tsmc_growth():

    # Data
    data = {
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

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date").reset_index(drop=True)

    # Calculate Quarterly YoY Growth (compare current quarter to same quarter last year)
    df["YoY_Growth"] = df["Revenue"].pct_change(periods=4) * 100

    # Drop NaN values (due to missing data for first few quarters)
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

    # Convert year/quarter into x_positions
    # We'll represent time continuously: For each year, quarters are spaced evenly.
    # For simplicity: x = Year + (Quarter-1)*0.25
    line_x = [item[0] + (item[1]-1)*0.25 for item in chronological_data]
    line_y = [item[2] for item in chronological_data]

    # Remove the first point (Q1 of 2010)
    # Identify Q1 of 2010 in chronological_data:
    # The earliest data point should be 2009 Q4 or 2010 Q1 depending on data.
    # Let's just drop the first element from both line_x and line_y:
    line_x = line_x[1:]
    line_y = line_y[1:]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the line and markers
    ax.plot(line_x, line_y, color='black', marker='o', linewidth=1)

    # Formatting
    ax.set_title("TSMC Year-Over-Year (YoY) Quarterly Revenue Growth")
    ax.set_xlabel("Time")
    ax.set_ylabel("YoY Growth (%)")
    ax.set_ylim(-50, 100)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Create a custom x-tick labeling: Show only whole years
    years = sorted(set([int(x) for x in line_x]))
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years], rotation=45)

    plt.tight_layout()
    plt.show()

    # Save to CSV
    yoy_data = pd.DataFrame(line_x, columns=['Time'])
    yoy_data['YoY_Growth'] = line_y
    yoy_data.to_csv("tsmc_yoy_growth.csv", index=False)

plot_tsmc_growth()