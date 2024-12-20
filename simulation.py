import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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
    if t>=10 and t<=30:
        damping_factor = 1 / (1 + S)
        dS_dt = (r_s * S*(1-S/K) + ((alpha_s *S)/(1+gamma_s*S))*D) - S*supply_shock#+0.8*supply_noise# - abs(supply_shock) #+ gamma * max(0, S_pre_shock - S)#+0.5*supply_noise
        #print(dS_dt)
    else:
        dS_dt = (r_s * S*(1-S/K) + ((alpha_s *S)/(1+gamma_s*S))*D)#+0.1*supply_noise# - abs(supply_shock) #+ gamma * max(0, S_pre_shock - S)#+0.5*supply_noise
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
beta_d =0.05
shock_mean =0.8
shock_std =0.4
noise_std =0.02
gamma       =0.2
n_simulations   =100
S_pre_shock=state0[0]
alpha=0.01
alpha_s=0.2
gamma_s=0.2
gamma_d=0.15
K=5000
worst=100000
best=0

 # Non-positive supply shock
for _ in range(n_simulations):
    # Solve the system using odeint
    supply_shock = round(np.random.normal(shock_mean, shock_std),2)  # Non-positive supply shock
    result = odeint(system_ODE, state0, t, args=(r_s, r_d, beta_s, beta_d, shock_mean, shock_std, noise_std, gamma, S_pre_shock,alpha,supply_shock))
    if supply_shock<worst:
        worst=supply_shock
        supply_worst = result[:, 0]
        demand_worst = result[:, 1]
        price_worst = result[:, 2]

    if supply_shock>best:
        best=supply_shock
        supply_best = result[:, 0]
        demand_best = result[:, 1]
        price_best = result[:, 2]

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

percent_change_supply_best = 100 * (supply_best[1:] - supply_best[:-1]) / supply_best[:-1]
percent_change_demand_best = 100 * (demand_best[1:] - demand_best[:-1]) / demand_best[:-1]
percent_change_price_best = 100 * (price_best[1:] - price_best[:-1]) / price_best[:-1]

percent_change_supply_worst = 100 * (supply_worst[1:] - supply_worst[:-1]) / supply_worst[:-1]
percent_change_demand_worst = 100 * (demand_worst[1:] - demand_worst[:-1]) / demand_worst[:-1]
percent_change_price_worst = 100 * (price_worst[1:] - price_worst[:-1]) / price_worst[:-1]

percent_change_supply_worst=percent_change_supply_worst[::2]
percent_change_demand_worst=percent_change_demand_worst[::2]
percent_change_demand_best=percent_change_demand_best[::2]
percent_change_supply_best=percent_change_supply_best[::2]

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
for i in range(len(percent_change_supply)-1):
    if i%2==0:
        percent_change_supply[i]=percent_change_supply[i]+percent_change_supply[i+1]
        percent_change_demand[i]=percent_change_demand[i]+percent_change_demand[i+1]
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

# Plot percentage change in price
fig, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(t, avg_price, label="Price Index", color="red")
ax2.set_xlabel("Time (Months)")
ax2.set_ylabel("Price Index")
ax2.set_title("Simulation Demonstration: Percentage Change in Price")
ax2.legend()
ax2.grid()

# Plot supply and demand over time
fig, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(t, avg_supply, label="Supplier Revenue", color="green")
ax3.plot(t, avg_demand, label="Consumer Revenue", color="blue")
ax3.set_xlabel("Time (Months)")
ax3.set_ylabel("Revenue Index")
ax3.set_title("Simulation Demonstration: Supply shocks affect on Consumers")
ax3.legend()
ax3.grid()

plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(t[0::2], percent_change_supply_best, label="Percentage Change in Supply", color="green")
ax1.plot(t[0::2], percent_change_demand_best, label="Percentage Change in Demand", color="blue")
ax1.set_xlabel("Time (Months)")
ax1.set_ylabel("Revenue growth rate (%)")
ax1.set_title("Simulation Demonstration: Percentage changes in Supply and Demand worst")
ax1.legend()
ax1.grid()

# Plot percentage change in price
fig, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(t, price_best, label="Price Index", color="red")
ax2.set_xlabel("Time (Months)")
ax2.set_ylabel("Price Index")
ax2.set_title("Simulation Demonstration: Percentage Change in Price worst")
ax2.legend()
ax2.grid()

# Plot supply and demand over time
fig, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(t, supply_best, label="Supplier Revenue", color="green")
ax3.plot(t, demand_best, label="Consumer Revenue", color="blue")
ax3.set_xlabel("Time (Months)")
ax3.set_ylabel("Revenue")
ax3.set_title("Simulation Demonstration: Supply shocks affect on Consumers worst")
ax3.legend()
ax3.grid()

plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(t[0::2], percent_change_supply_worst, label="Percentage Change in Supply", color="green")
ax1.plot(t[0::2], percent_change_demand_worst, label="Percentage Change in Demand", color="blue")
ax1.set_xlabel("Time (Months)")
ax1.set_ylabel("Revenue growth rate (%)")
ax1.set_title("Simulation Demonstration: Percentage changes in Supply and Demand best")
ax1.legend()
ax1.grid()

# Plot percentage change in price
fig, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(t, price_worst, label="Price Index", color="red")
ax2.set_xlabel("Time (Months)")
ax2.set_ylabel("Price Index")
ax2.set_title("Simulation Demonstration: Percentage Change in Price best")
ax2.legend()
ax2.grid()

# Plot supply and demand over time
fig, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(t, supply_worst, label="Supplier Revenue", color="green")
ax3.plot(t, demand_worst, label="Consumer Revenue", color="blue")
ax3.set_xlabel("Time (Months)")
ax3.set_ylabel("Revenue")
ax3.set_title("Simulation Demonstration: Supply shocks affect on Consumers best")
ax3.legend()
ax3.grid()

plt.show()

