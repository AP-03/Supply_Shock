import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(42)

params = {
    # Existing parameters
    "r": {"TSMC": 0.07, "Samsung Foundry": 0.04},
    "K": {"TSMC": 2500, "Samsung Foundry": 1800}, 
    "alpha": 1e-4,
    "gamma": 1e-4,
    "beta": {
        "Tesla": 5e-5,
        "Toyota": 2e-5,
        "Apple": 1e-5,
        "AWS": 3e-5
    },
    "delta": 0.01,
    "m": 0.0005,
    "eta": 0.5,
    "delta_p": 0.005,
    "rho": 0.1,
    "P0": 1.0,
    "nu": 0.05,
    "xi": 0.01,
    "shock_mean": -5,
    "shock_std": 1,
    "event_prob": 0.1,
    "P_max": 5e4,
    
    # New parameters for Chen-inspired modifications
    "lambda_calvo": 0.3,       # Probability of adjusting supply orders each step
    "noise_weight": 0.2,       # Weight of noise in demand forecasting
    "forecast_noise": 0.05     # Std dev of forecast noise
}

num_months = 48
def generate_shocks(num_months, event_prob, mean, std, min_val, max_val):
    shocks = []
    for month in range(num_months):
        if np.random.rand() < event_prob:
            val = np.clip(np.random.normal(mean, std), min_val, max_val)
            shocks.append((month, val))
        else:
            shocks.append((month, 0))
    return shocks

monthly_shocks = generate_shocks(num_months, params["event_prob"], params["shock_mean"], params["shock_std"], -10, 0)

# States (y):
# y = [S_TSMC, S_Samsung, D_Tesla, D_Toyota, D_Apple, D_AWS, P, I, D_f]
# D_f: forecasted future total demand (a new state)

def supply_demand_model(t, y, params, monthly_shocks):
    (S_TSMC, S_Samsung, D_Tesla, D_Toyota, D_Apple, D_AWS, P, I, D_f) = y
    
    r, K, alpha, gamma = params["r"], params["K"], params["alpha"], params["gamma"]
    beta, delta, m = params["beta"], params["delta"], params["m"]
    eta, delta_p = params["eta"], params["delta_p"]
    rho, P0, nu, xi = params["rho"], params["P0"], params["nu"], params["xi"]
    P_max = params["P_max"]
    lambda_calvo = params["lambda_calvo"]
    noise_weight = params["noise_weight"]
    forecast_noise = params["forecast_noise"]

    month_index = int(t // 30)
    if 0 <= month_index < len(monthly_shocks):
        shock = monthly_shocks[month_index][1]
    else:
        shock = 0

    # Compute totals
    D_total = D_Tesla + D_Toyota + D_Apple + D_AWS
    S_total = S_TSMC + S_Samsung

    # Inventory dynamics (unchanged)
    dI_dt = rho*(P - P0) - nu*I

    # Demand Equations (unchanged)
    def demand_eq(D, beta_val):
        denom = max(1 + delta*D, 1e-9)
        return beta_val*(S_total/denom)*D - m*D

    dD_Tesla_dt = demand_eq(D_Tesla, beta["Tesla"])
    dD_Toyota_dt = demand_eq(D_Toyota, beta["Toyota"])
    dD_Apple_dt = demand_eq(D_Apple, beta["Apple"])
    dD_AWS_dt   = demand_eq(D_AWS,   beta["AWS"])

    # Price dynamics (unchanged except for cap)
    dP_dt = eta * (D_total - S_total) - delta_p * P
    if P > P_max and dP_dt > 0:
        dP_dt = 0

    # --- Chen-inspired changes below ---

    # Demand Forecast Update:
    # Forecast future total demand based on current D_total and noise.
    # D_f'(t) = move D_f towards (1-noise_weight)*D_total plus noise
    forecast_target = (1 - noise_weight)*D_total
    noise = np.random.normal(0, forecast_noise)
    forecast_target += noise_weight * noise
    # Adjust D_f toward forecast_target smoothly:
    # We'll model this as a simple ODE: dD_f/dt = (forecast_target - D_f)
    dD_f_dt = (forecast_target - D_f)

    # Now use D_f instead of actual D in supply calculations:
    # Compute "optimal" supply changes based on D_f rather than actual D_total.
    # For TSMC (serves Tesla & Toyota):
    # Approximate D_sum_TSMC with a fraction of D_f (e.g., Tesla+Toyota share).
    # To keep it simple, distribute D_f proportionally:
    # Tesla+Toyota demand fraction:
    frac_TSMC = (D_Tesla + D_Toyota) / (D_total + 1e-9)
    frac_TSMC = min(max(frac_TSMC,0),1)
    D_sum_TSMC_forecast = D_f * frac_TSMC

    # For Samsung (serves Apple & AWS)
    frac_Samsung = (D_Apple + D_AWS) / (D_total + 1e-9)
    frac_Samsung = min(max(frac_Samsung,0),1)
    D_sum_Samsung_forecast = D_f * frac_Samsung

    # Compute the "optimal" supply growth (as if adjusting fully)
    S_TSMC_opt = (r["TSMC"] * S_TSMC * (1 - S_TSMC / K["TSMC"])
                  + xi*I
                  - (alpha * S_TSMC / (1 + gamma * S_TSMC)) * D_sum_TSMC_forecast
                  + shock)

    S_Samsung_opt = (r["Samsung Foundry"] * S_Samsung * (1 - S_Samsung / K["Samsung Foundry"])
                     + xi*I
                     - (alpha * S_Samsung / (1 + gamma * S_Samsung)) * D_sum_Samsung_forecast
                     + shock)

    # Calvo-style partial adjustment:
    # Instead of directly dS/dt = S_opt, we do a partial move towards S_opt:
    # dS_TSMC/dt = lambda_calvo*(S_TSMC_opt - 0) since previously we used dS/dt form
    # Actually, originally we used dS/dt as the supply ODE. Now we treat S_opt as a rate:
    # Let's interpret S_TSMC_opt as a "target growth rate". We blend old behavior:
    # New approach: S_opt is like a desired instantaneous change. Calvo means we apply only fraction:
    dS_TSMC_dt = lambda_calvo * S_TSMC_opt
    dS_Samsung_dt = lambda_calvo * S_Samsung_opt

    return [dS_TSMC_dt, dS_Samsung_dt,
            dD_Tesla_dt, dD_Toyota_dt, dD_Apple_dt, dD_AWS_dt,
            dP_dt, dI_dt, dD_f_dt]

# Adjust initial conditions:
# Add D_f (forecast) to state variables, start equal to D_total (700+600+400+550=2250)
y0 = [1200, 900, 700, 600, 400, 550, 1.0, 20.0, 2250.0]

t_span = (0, 730)
t_eval = np.linspace(t_span[0], t_span[1], 300)

solution = solve_ivp(
    supply_demand_model, t_span, y0, args=(params, monthly_shocks),
    t_eval=t_eval, method="RK45", atol=1e-7, rtol=1e-7
)

print("Success:", solution.success)
print("Message:", solution.message)

# Extract variables
S_TSMC = solution.y[0]
S_Samsung = solution.y[1]
D_Tesla = solution.y[2]
D_Toyota = solution.y[3]
D_Apple = solution.y[4]
D_AWS = solution.y[5]
P = solution.y[6]
I = solution.y[7]
D_f = solution.y[8]

# Normalize supply/demand/price based on initial
norm_indices = [0,1,2,3,4,5,6] # S_TSMC, S_Samsung, D_Tesla, D_Toyota, D_Apple, D_AWS, P
norm_factors = solution.y[norm_indices,0]
normalized_SD = solution.y[norm_indices,:] / norm_factors[:,None]

fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

# Suppliers & Consumers (normalized)
axs[0].plot(solution.t, normalized_SD[0], label="TSMC Supply (norm)")
axs[0].plot(solution.t, normalized_SD[1], label="Samsung Supply (norm)")
axs[0].plot(solution.t, normalized_SD[2], label="Tesla Demand (norm)")
axs[0].plot(solution.t, normalized_SD[3], label="Toyota Demand (norm)")
axs[0].plot(solution.t, normalized_SD[4], label="Apple Demand (norm)")
axs[0].plot(solution.t, normalized_SD[5], label="AWS Demand (norm)")
axs[0].set_ylabel("Normalized Values")
axs[0].set_title("Suppliers and Consumers (Normalized) - With Lag and Partial Adjustments")
axs[0].legend()
axs[0].grid(True)

# Price (normalized)
price_initial = P[0]
normalized_price = P / price_initial
axs[1].plot(solution.t, normalized_price, label="Price (norm)", color="black")
axs[1].set_ylabel("Normalized Price")
axs[1].set_title("Price Dynamics (Normalized)")
axs[1].legend()
axs[1].grid(True)

# Inventory
axs[2].plot(solution.t, I, label="Inventory (I)", color="purple")
axs[2].set_ylabel("Inventory")
axs[2].set_title("Inventory Dynamics (Absolute)")
axs[2].legend()
axs[2].grid(True)

# Forecast vs Actual Demand
D_total = D_Tesla + D_Toyota + D_Apple + D_AWS
axs[3].plot(solution.t, D_total, label="Actual D_total", color="green")
axs[3].plot(solution.t, D_f, label="Forecasted D_f", color="red", linestyle="--")
axs[3].set_xlabel("Time (days)")
axs[3].set_ylabel("Demand")
axs[3].set_title("Actual vs. Forecasted Demand")
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()
