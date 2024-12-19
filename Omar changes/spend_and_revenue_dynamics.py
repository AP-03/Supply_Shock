import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pandas as pd

# # Load the data
# time_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Year number (now 10 years)
# spending_data = np.array([18.8, 21.4, 23.5, 27.18, 30.89, 30.39, 38.75, 41.39, 53.62, 68.27])  # Real spending data
# revenue_data = np.array([57.225, 47.694, 35.774, 33.694, 32.977, 29.257, 26.571, 24.140, 20.014, 17.426])  # Real revenue data (now 10 years)

# #Reverse the revenue data
# revenue_data = revenue_data[::-1]

apple_yoy = pd.read_csv('apple_yoy_growth.csv')
tsmc_yoy = pd.read_csv("tsmc_yoy_growth.csv")

spending_data = np.array(apple_yoy['YoY_Growth'])
revenue_data = np.array(tsmc_yoy['YoY_Growth'])

# Define the oscillatory ODE system
def spending_revenue_dynamics_osc(t, y, a, K, b, c, d, e, f, omega):
    S, R = y  # S: Consumer Spending, R: Supplier Revenue
    dS_dt = a * S * (1 - S / K) - b * S * R + c * np.sin(omega * t)  # Spending with periodic forcing
    dR_dt = d * S * R - e * R + f * np.cos(omega * t)  # Revenue with periodic forcing
    return [dS_dt, dR_dt]

# Solve the ODEs
def solve_model_osc(params, t_span, y0):
    a, K, b, c, d, e, f, omega = params
    sol = solve_ivp(
        spending_revenue_dynamics_osc,
        t_span,
        y0,
        args=(a, K, b, c, d, e, f, omega),
        t_eval=np.linspace(t_span[0], t_span[1], t_span[1])  # Match number of time points
    )
    return sol.t, sol.y

# Objective function to minimize (error between real data and simulated results)
def objective(params, t_span, y0, real_spending_data, real_revenue_data):
    time, results = solve_model_osc(params, t_span, y0)
    simulated_spending, simulated_revenue = results  # Results contain both spending and revenue
    
    # Select the first row (consumer spending) and the second row (supplier revenue)
    simulated_spending = simulated_spending[0]
    simulated_revenue = simulated_revenue[1]
    
    # Calculate the squared error
    error_S = np.sum((real_spending_data - simulated_spending) ** 2)
    error_R = np.sum((real_revenue_data - simulated_revenue) ** 2)
    
    return error_S + error_R  # Total squared error


# Initial conditions and parameters
t_span = [0, 10]  # Time span (e.g., quarters)
y0 = [spending_data[0], revenue_data[0]]      # Initial spending and revenue levels
initial_params = [  # [a, K, b, c, d, e, f, omega]
    0.05,      # a: Growth rate for spending (5% growth rate)
    500,       # K: Carrying capacity for spending (e.g., $500B maximum)
    0.0001,    # b: Interaction term (spending reduces revenue slightly, scaled down)
    10,        # c: Forcing amplitude for spending (e.g., $10B fluctuations)
    0.00005,   # d: Interaction term (spending increases revenue, scaled down)
    0.05,      # e: Decay rate for revenue (5% decay rate)
    5,        # f: Forcing amplitude for revenue (e.g., $10B fluctuations)
    1.5        # omega: Frequency of oscillation (1.5 cycles per year)
]


# Perform the parameter fitting (minimizing the objective function)
result = minimize(objective, initial_params, args=(t_span, y0, spending_data, revenue_data), method="Nelder-Mead")


# Extract the optimized parameters
optimized_params = result.x
print("Optimized Parameters:", optimized_params)

# Solve the model with optimized parameters
time, results = solve_model_osc(optimized_params, t_span, y0)
spending_fitted, revenue_fitted = results

# Plot real data vs fitted results
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(spending_data)), spending_data, 'bo-', label="Real Spending")
plt.plot(np.arange(len(spending_fitted)), spending_fitted, 'b--', label="Fitted Spending") #x axis is time, number of quarters/year depending on the data
plt.plot(np.arange(len(revenue_data)), revenue_data, 'go-', label="Real Revenue (Reversed)")
plt.plot(np.arange(len(revenue_fitted)), revenue_fitted, 'g--', label="Fitted Revenue")
plt.xlabel("Time (Years)")
plt.ylabel("Values")
plt.title("Fitting Consumer Spending and Supplier Revenue")
plt.legend()
plt.grid()
plt.show()
