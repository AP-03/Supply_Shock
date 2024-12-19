import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Load data
apple_yoy = pd.read_csv('apple_yoy_growth.csv')
tsmc_yoy = pd.read_csv("tsmc_yoy_growth.csv")

spending_data = np.array(apple_yoy['YoY_Growth'])
revenue_data = np.array(tsmc_yoy['YoY_Growth'])

# Define the semiconductor revenue growth ODE with noise
def semiconductor_revenue_growth(t, y, a, b, c, omega):
    """
    Second-order ODE for modeling semiconductor revenue growth dynamics with noise.

    Parameters:
    - t: Time (years).
    - y: Array [R, dR/dt], where R is revenue growth and dR/dt is its rate of change.
    - a: Damping coefficient (controls how quickly oscillations decay).
    - b: Restoring force coefficient (controls tendency to revert to equilibrium).
    - c: Amplitude of periodic forcing (captures external cyclic influences).
    - omega: Angular frequency of the periodic forcing (cycles/year).
    - noise_amplitude: Amplitude of the random noise added to the system.

    Returns:
    - dydt: Array [dR/dt, d^2R/dt^2], the first and second derivatives of R.
    """
    R, dR_dt = y  # Revenue growth (R) and its first derivative (dR/dt)
    
   
    
    # Compute the second derivative 
    d2R_dt2 = -a * dR_dt - b * R + c * np.sin(omega * t) 
    
    return [dR_dt, d2R_dt2]


# Solve the ODEs
def solve_model_osc(params, t_span, y0):
    a, b, c, omega = params
    sol = solve_ivp(
        semiconductor_revenue_growth,  # Correct ODE function
        t_span,
        y0,
        args=(a, b, c, omega),
        t_eval=np.linspace(t_span[0], t_span[1], len(spending_data))  # Ensure t_eval matches data length
    )
    return sol.t, sol.y

# Objective function to minimize (error between real data and simulated results)
def objective(params, t_span, y0, real_spending_data):
    time, results = solve_model_osc(params, t_span, y0)
    simulated_spending = results[1]  # Results contain both spending and revenue
    
    # Calculate the squared error
    error_S = np.sum((real_spending_data[:len(simulated_spending)] - simulated_spending[:len(real_spending_data)]) ** 2)
    
    return error_S   # Total squared error

# Initial conditions and parameters
t_span = [0, 100]  # Time span based on the length of the data
y0 = [0, spending_data[0]]  # Initial spending and revenue levels, with zero derivative
initial_params = [  # [a, b, c, omega]
    0.5,  # a: Damping coefficient (growth rate)
    0.1,  # b: Restoring force coefficient (interaction term)
    20,  # c: Forcing amplitude for spending
    0.5  # omega: Frequency of oscillation (cycles per year)
]

print("Simulation running...")

# Perform the parameter fitting (minimizing the objective function)
result = minimize(objective, initial_params, args=(t_span, y0, spending_data), method="Nelder-Mead")

# Extract the optimized parameters
optimized_params = result.x
print("Optimized Parameters:", optimized_params)

# Solve the model with optimized parameters
time, results = solve_model_osc(optimized_params, t_span, y0)
spending_fitted = results[1]  # Fitted spending values



for i in range(len(spending_fitted)):
    # Generate a random noise value for the current time step
    noise = 10 * np.random.normal(0, 1)
    spending_fitted[i] += noise

# Plot real data vs fitted results
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(spending_data)), spending_data, 'bo-', label="Real Spending")
plt.plot(time, spending_fitted, 'bx-', label="Fitted Spending")  # Adjusted to use 'time' for fitted data
# Optionally, you can add revenue data and plot as well
# plt.plot(np.arange(len(revenue_data)), revenue_data, 'go-', label="Real Revenue")
# plt.plot(np.linspace(t_span[0], t_span[1], len(revenue_fitted)), revenue_fitted, 'g--', label="Fitted Revenue")
plt.xlabel("Time (Years)")
plt.ylabel("Values")
plt.title("Fitting Consumer Spending and Supplier Revenue")
plt.legend()
plt.grid()
plt.show()
