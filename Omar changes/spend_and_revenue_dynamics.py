import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Load the data
time_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Year number (now 10 years)
spending_data = np.array([18.8, 21.4, 23.5, 27.18, 30.89, 30.39, 38.75, 41.39, 53.62, 68.27])  # Real spending data
revenue_data = np.array([57.225, 47.694, 35.774, 33.694, 32.977, 29.257, 26.571, 24.140, 20.014, 17.426])  # Real revenue data (now 10 years)

# Reverse the revenue data
revenue_data = revenue_data[::-1]
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
        t_eval=np.linspace(t_span[0], t_span[1])
    )
    return sol.t, sol.y

# Initial conditions and parameters
t_span = [0, 100]  # Time span (e.g., quarters)
y0 = [100, 50]      # Initial spending and revenue levels
params = [0.05, 500, 0.02, 10, 0.01, 0.05, 100, 0.5]  # [a, K, b, c, d, e, f, omega]

# Solve the model
time, results = solve_model_osc(params, t_span, y0)
spending, revenue = results

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time, spending, label="Consumer Spending", color="b")
plt.plot(time, revenue, label="Supplier Revenue", color="g")
plt.xlabel("Time (Quarters)")
plt.ylabel("Values")
plt.title("Oscillatory Consumer Spending vs Supplier Revenue Dynamics")
plt.legend()
plt.grid()
plt.show()
