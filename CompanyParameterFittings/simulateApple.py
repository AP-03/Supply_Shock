import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
from scipy.interpolate import interp1d


# ODE function defining the system of supply, demand, and price
def system_ODE(state, t, r_s, r_d, beta_s, beta_d, shock_mean, shock_std, noise_std, gamma, S_pre_shock,alpha,supply_shock):
    S, D, P = state  # Unpack the state vector
    
    #supply_shock = round(np.random.normal(shock_mean, shock_std),2)  # Non-positive supply shock
    # Generate shocks and noise
    demand_shock = np.random.normal(shock_mean, shock_std)  # Demand shock
    price_noise = np.random.normal(0, noise_std)  # Noise in price adjustment
    demand_noise = np.random.normal(0, noise_std)  # Noise in price adjustment
    supply_noise = np.random.normal(0, noise_std)  # Noise in price adjustment
    # Supply ODE
    if t >= 10 and t < 35:
        dS_dt = (r_s * S * (1 - S / K) + ((alpha_s * S) / (1 + gamma_s * S)) * D) - S * 1.1 + S
    elif t >= 35 and t < 45:
        dS_dt = (r_s * S * (1 - S / K) + ((alpha_s * S) / (1 + gamma_s * S)) * D) - S * 1.5 + S
    elif t >= 55:
        dS_dt = (r_s * S * (1 - S / K) + ((alpha_s * S) / (1 + gamma_s * S)) * D) - S * 3 + S
    else:
        dS_dt = (r_s * S * (1 - S / K) + ((alpha_s * S) / (1 + gamma_s * S)) * D) + D
    # Demand ODE
    dD_dt = (r_d * D*beta_s*S)/(1+gamma_d*D) - D*beta_d*(P-100)#+0.1*demand_noise# + demand_shock

    # Price ODE
    dP_dt = alpha*(D - S)/P#+ 0.001*price_noise  # Price adjusts based on supply-demand imbalance

    return [dS_dt, dD_dt, dP_dt]

# Parameters for simulation
T = 60  # Total time in months
n_points = 1000  # Number of time points
t = np.linspace(0, T, n_points)  # Time grid
state0 = [5000, 5000, 100]  # Initial conditions

# Monte Carlo simulation
all_supply = []
all_demand = []
all_price = []
r_s = 0.15
r_d = 0.2
beta_s =0.1
beta_d =0.3
shock_mean =0.8
shock_std =0.5
noise_std =0.02
gamma       =0.2
n_simulations   =100
S_pre_shock=state0[0]
alpha=0.01
alpha_s=0.2
gamma_s=0.15
gamma_d=0.15
K=5000
supply_shock = round(np.random.normal(shock_mean, shock_std),2)  # Non-positive supply shock

 # Non-positive supply shock
for _ in range(n_simulations):
    # Solve the system using odeint
    result = odeint(system_ODE, state0, t, args=(r_s, r_d, beta_s, beta_d, shock_mean, shock_std, noise_std, gamma, S_pre_shock,alpha,supply_shock))


    # Extract supply, demand, and price from the result
    supply = result[:, 0]
    demand = result[:, 1]
    price = result[:, 2]
    
    # Update pre-shock supply level for recovery
    S_pre_shock = max(S_pre_shock, max(supply))
    
    # Store results
    all_supply.append(supply)
    all_demand.append(demand)
    all_price.append(price)

# Convert results to numpy arrays
all_supply = np.array(all_supply)
all_demand = np.array(all_demand)
all_price = np.array(all_price)

# Calculate the averages and standard deviations
avg_supply = np.mean(all_supply, axis=0)/50
avg_demand = np.mean(all_demand, axis=0)/50
avg_price = np.mean(all_price, axis=0)

# Calculate the percentage change of supply, demand, and price
percent_change_supply = 100 * (avg_supply[1:] - avg_supply[:-1]) / avg_supply[:-1]
percent_change_demand = 100 * (avg_demand[1:] - avg_demand[:-1]) / avg_demand[:-1]
percent_change_price = 100 * (avg_price[1:] - avg_price[:-1]) / avg_price[:-1]
# Ensure valid data for initial points
#for i in range(len(percent_change_supply)-1):
#    percent_change_supply[i+1]=percent_change_supply[i+1]+percent_change_supply[i]
#    percent_change_demand[i+1]=percent_change_demand[i+1]+percent_change_demand[i]
percent_change_supply=percent_change_supply[::2]
percent_change_demand=percent_change_demand[::2]

# Plot results
# Plot percentage changes in supply and demand
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(t[0::2], percent_change_supply, label="Percentage Change in Supply", color="green")
ax1.plot(t[0::2], percent_change_demand, label="Percentage Change in Demand", color="blue")
ax1.set_xlabel("Time (Months)")
ax1.set_ylabel("Revenue growth rate (%)")
ax1.set_title("Simulation Demonstration: Percentage changes in Supply and Demand")
ax1.legend()
ax1.grid()

plt.show()


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

# Apple Data
apple_data = {
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
        94930, 85777, 90753,
        119575, 89498, 81797, 94836,
        117154, 90146, 82959, 97278,
        123945, 83360, 81434, 89584,
        111439, 64698, 59685, 58313,
        91819, 64040, 53809, 58015,
        84310, 62900, 53265, 61137,
        88293, 52579, 45408, 52896,
        78351, 46852, 42358, 50557,
        75872, 51501, 49605, 58010,
        74599, 42123, 37432, 45646,
        57594, 37472, 35323, 43603,
        54512, 35966, 35023, 39186,
        46333, 28270, 28571, 24667,
        26741, 20343, 15700, 13499,
        15683, 12207, 9734, 9084
    ]
}

apple_exclude_conditions = lambda df: (
    ((df["Year"] == 2010) & (df["Quarter"] == 1))
)
apple_x, apple_y = process_data(apple_data, exclude_conditions=apple_exclude_conditions)


common_time = np.intersect1d(tsmc_x, apple_x)
tsmc_interp = np.interp(common_time, tsmc_x, tsmc_y)
apple_interp = np.interp(common_time, apple_x, apple_y)

# Ensure simulated data matches `common_time`
simulated_time = np.linspace(common_time[0], common_time[-1], len(percent_change_supply))
supply_interp_func = interp1d(simulated_time, percent_change_supply, kind='linear', fill_value="extrapolate")
demand_interp_func = interp1d(simulated_time, percent_change_demand, kind='linear', fill_value="extrapolate")

# Interpolate simulated data to align with `common_time`
aligned_supply = supply_interp_func(common_time)
aligned_demand = demand_interp_func(common_time)

# Normalize the simulated data to the YoY Growth scale
normalized_supply = (aligned_supply / np.max(aligned_supply)) * 100
normalized_demand = ((aligned_demand / np.max(aligned_demand)) * 100) / 2.5

# Plot combined data
plt.figure(figsize=(14, 8))

# Plot interpolated YoY growth data
# plt.plot(common_time, tsmc_interp, label="TSMC YoY Growth", color="blue", linestyle="--")
plt.plot(common_time, apple_interp, label="Apple YoY Growth", color="green", linestyle="--")

# Overlay simulated normalized growth
plt.plot(common_time, normalized_supply, label="Simulated Supplier Revenue Growth (%)", color="red", linestyle="--")
plt.plot(common_time, normalized_demand + 40, label="Simulated Consumer Revenue Growth (%)", color="orange", linestyle="--")

# Add labels, title, and legend
plt.title("Combined Actual and Simulated YoY Growth: TSMC and Apple")
plt.xlabel("Time (Years)")
plt.ylabel("YoY Growth (%)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()