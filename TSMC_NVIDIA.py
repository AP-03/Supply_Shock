import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import pearsonr


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

# Calculate Pearson Correlation
correlation, p_value = pearsonr(tsmc_interp, nvidia_interp)

# Print results
print(f"Pearson Correlation: {correlation:.2f}")
print(f"P-value: {p_value:.2e}")

# # Supply-Demand Model Function
# def supply_demand_odes(t, y, params):
#     S, D = y  # Supply and Demand
#     r, K, alpha, gamma, beta, delta, m = (
#         params["r"], params["K"], params["alpha"], params["gamma"],
#         params["beta"], params["delta"], params["m"]
#     )

#     # ODEs
#     dS_dt = r * S * (1 - S / K) - (alpha * S / (1 + gamma * S)) * D
#     dD_dt = (beta * S / (1 + delta * D)) * D - m * D

#     return [dS_dt, dD_dt]

# # Parameters for the ODE model
# params = {
#     "r": 0.07,       # Supply growth rate
#     "K": 2500,       # Carrying capacity
#     "alpha": 1e-4,   # Supply depletion rate
#     "gamma": 1e-4,   # Economies of scale
#     "beta": 5e-5,    # Demand growth rate
#     "delta": 0.01,   # Demand saturation factor
#     "m": 0.0005      # Demand decay rate
# }

# # Initial Conditions and Time Span
# initial_conditions = [1200, 500]  # Initial supply and demand
# time_span = (0, len(common_time))  # Simulate over common time
# time_eval = np.linspace(*time_span, len(common_time))

# # Solve the ODEs
# solution = solve_ivp(
#     supply_demand_odes, time_span, initial_conditions, t_eval=time_eval, args=(params,)
# )
# simulated_supply, simulated_demand = solution.y

# # Normalize Simulated Data for Comparison
# simulated_supply = simulated_supply / max(simulated_supply) * max(tsmc_interp)
# simulated_demand = simulated_demand / max(simulated_demand) * max(nvidia_interp)

# # Plot Simulated and Actual Results
# plt.figure(figsize=(14, 8))

# # Actual TSMC
# plt.plot(common_time, tsmc_interp, label="TSMC Actual YoY Growth", color="blue", linestyle="--")

# # Actual NVIDIA
# plt.plot(common_time, nvidia_interp, label="NVIDIA Actual YoY Growth", color="green", linestyle="--")

# # Simulated TSMC Supply
# plt.plot(common_time, simulated_supply, label="Simulated TSMC Supply", color="blue", linestyle="-")

# # Simulated NVIDIA Demand
# plt.plot(common_time, simulated_demand, label="Simulated NVIDIA Demand", color="green", linestyle="-")

# plt.title("TSMC and NVIDIA YoY Growth vs Simulated Supply and Demand")
# plt.xlabel("Time (Years)")
# plt.ylabel("Values (Normalized)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()



# # Plotting the results
# plt.figure(figsize=(14, 8))

# # TSMC
# plt.plot(tsmc_x, tsmc_y, label="TSMC Actual YoY Growth (Dashed)", color="blue", linestyle="--")

# # NVIDIA
# plt.plot(nvidia_x, nvidia_y, label="NVIDIA Actual YoY Growth (Dashed)", color="green", linestyle="--")

# plt.title("TSMC and NVIDIA YoY Growth (Actual Data)")
# plt.xlabel("Time (Years)")
# plt.ylabel("YoY Growth (%)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()