import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the data
tsmc_yoy = pd.read_csv('tsmc_yoy_growth.csv')
apple_yoy = pd.read_csv('apple_yoy_growth.csv')

time_data = np.array(apple_yoy['Time'])  # Year number (now 10 years)
spending_data = np.array(apple_yoy['YoY_Growth'])  # Real spending data
revenue_data = np.array(tsmc_yoy['YoY_Growth'])  # Real revenue data (now 10 years)

print(time_data, spending_data, revenue_data)


# System of ODEs with Gaussian shock for supply-demand dynamics
def supply_demand_dynamics(t, y, r, K_S, alpha, delta, gamma, beta, m, shock_std_dev):
    S, D = y
    # Generate a random shock (Gaussian distributed noise)
    shock = np.random.normal(0, shock_std_dev)  # Mean 0, standard deviation `shock_std_dev`
    dS_dt = r * S * (1 - (S/K_S)) - (alpha * S * D) / (1 + (gamma*S)) + shock  # Spending dynamics with shock
    dD_dt = (beta * S * D) / (1 + (delta*D)) - (m * D)  # Demand dynamics with shock (fixed order of operations)
    return [dS_dt, dD_dt]

# Solve the ODEs with given parameters
def solve_model(params, shock_std_dev):
    r, K_S, alpha, delta, gamma, beta, m = params
    S0 = spending_data[0]
    D0 = revenue_data[0]  # Assuming D is analogous to revenue
    sol = solve_ivp(
        supply_demand_dynamics, 
        [time_data[0], time_data[-1]], 
        [S0, D0], 
        args=(r, K_S, alpha, delta, gamma, beta, m, shock_std_dev), 
        t_eval=time_data  # Ensure we evaluate only 10 points matching the data length
    )
    return sol.y  # Returns simulated [S, D] over time

# Objective function to minimize (error between model and data)
def objective(params, shock_std_dev):
    S_sim, D_sim = solve_model(params, shock_std_dev)
    error_S = np.sum((spending_data - S_sim) ** 2)
    error_D = np.sum((revenue_data - D_sim) ** 2)
    return error_S + error_D  # Total squared error

# Initial parameter guesses: [r, K_S, alpha, delta, gamma, beta, m]
initial_guess = [0.1, 100, 1, 2, 10, 1, 0.1]
shock_std_dev = 1  # Standard deviation of the shock (controls the intensity of randomness)

# Perform optimization (minimizing the objective function)
result = minimize(objective, initial_guess, args=(shock_std_dev,), method="Nelder-Mead")
best_params = result.x
print("Optimized Parameters:", best_params)

# Solve the model with optimized parameters
S_sim, R_sim = solve_model(best_params, shock_std_dev)

# Plot real data vs simulated results
plt.figure(figsize=(10, 6))
plt.plot(time_data, spending_data, 'bo-', label="Real Spending")
plt.plot(time_data, S_sim, 'b--', label="Simulated Spending with Shock")
plt.plot(time_data, revenue_data, 'go-', label="Real Revenue (Reversed)")
plt.plot(time_data, R_sim, 'g--', label="Simulated Revenue with Shock")
plt.xlabel("Time")
plt.ylabel("Values (Billions)")
plt.title("Real vs Simulated Spending and Revenue with Gaussian Shock")
plt.legend()
plt.grid()
plt.show()

