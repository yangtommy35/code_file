tom = 1+ 1
print(tom)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

# Set random seed for reproducibility
np.random.seed(2)

# Set model parameters according to the improved parameterization
r_bar = 0.07 / 252 
kappa = 0.015     
J0 = 3* kappa     # J0 parameter - you can adjust this manually
a = 0.99           
chi = 0.05      # Updated: memory decay parameter χ for h(u) = χe^(-χu)
mu_bar = 0.5       
s0 = 0.00011        
sigma_t = 0         
nu = 0             
eta_0 = 0.019*1.5
K = 1.0  # Fertility scaling factor - you can adjust this manually

# Memory management parameters
max_memory_days = 252  # Maximum days to retain for fertility calculations
window_size = max_memory_days  # Rolling window size

# Simulation days
T = 10000  # About 40 years of trading days

# Initialize arrays
log_prices = np.zeros(T)          # Log prices log(S_t)
returns = np.zeros(T)             # Log returns r_t = log(S_t) - log(S_t-1)
total_jump_impacts = np.zeros(T)  # Total jump impact J_t
expected_jump_impacts = np.zeros(T) # Expected jump impact (compensator) κΛ_t
lambda_t = np.zeros(T)            # Total jump intensity
mu_t = np.zeros(T)                # Baseline jump intensity
eta_t = np.zeros(T)               # Self-excitation parameter
X_t = np.zeros(T)                 # Mispricing indicator
I_t = np.zeros(T)                 # Exogenous jump count
O_t = np.zeros(T)                 # Endogenous jump count
N_t = np.zeros(T)                 # Total jump count
past_influences = np.zeros(T)     # Store past jump influence for diagnostics
branching_ratio = np.zeros(T)     # Store branching ratio calculations
sqrt_variance = np.zeros(T)       # Store √E[(r_{t-1} - r̄)²] values
SES = np.zeros(T)                 # Store Self-Excitement Strength
ReLU_X_t = np.zeros(T)            # Store ReLU(X_t/s0) values
jump_surprise = np.zeros(T)       # Store (J_t - κλ_t) values
# Store all fertility data
all_fertility_values = []
all_fertility_days = []

# Diagnostic variables
max_fertility = 0
max_past_influence = 0
max_poisson_param = 0
max_jump_size = 0

# Use a rolling window to store jump data
# Each element contains (time, jump_sizes_list, fertility_list)
jump_history = deque(maxlen=max_memory_days)

# Set initial price and mispricing
log_prices[0] = np.log(100)  # Initial price 100
X_t[0] = 0.00               # Initial mispricing
eta_t[0] = eta_0              # Initial eta_t value

# Updated memory kernel function using exponential decay
def generate_memory_kernel_exponential(chi, max_lag):
    """
    Generate memory kernel function h(u) = χe^(-χu)
    
    Parameters:
        chi (float): Memory decay parameter
        max_lag (int): Maximum lag
        
    Returns:
        np.array: Memory kernel function values
    """
    u_values = np.arange(1, max_lag + 1)  # u = 1, 2, 3, ..., max_lag
    return chi * np.exp(-chi * u_values)

# Use exponential memory kernel instead of geometric
memory_kernel = generate_memory_kernel_exponential(chi, max_memory_days)

# Print the first 10 values of the memory kernel
print("First 10 values of exponential memory kernel h(u) = χe^(-χu):")
print(memory_kernel[:10])
print(f"Chi parameter: {chi}")
print(f"Sum of memory kernel (first {max_memory_days} terms): {np.sum(memory_kernel):.6f}")

# Exponential calculation function without safety limits
def safe_exp(x):
    """
    Calculate exponential without artificial limits
    
    Parameters:
        x (float or np.array): Input value
        
    Returns:
        float or np.array: exp(x)
    """
    return np.exp(x)

# Updated fertility calculation using J0 instead of 3*kappa
def calculate_fertility(jump_sizes, kappa, a, K=1.0, J0=None, debug=False):
    """
    Calculate fertility for each jump f_j^i = K * exp(J_j^i/J0)
    
    Parameters:
        jump_sizes (np.array): Array of jump sizes
        kappa (float): Average jump size
        a (float): EWMA smoothing parameter
        K (float): Fertility scaling parameter (default=1.0)
        J0 (float, optional): Fertility scale parameter (uses J0 = 3*kappa if None)
        debug (bool): Whether to print diagnostic information
        
    Returns:
        np.array: Array of fertility values
    """
    fertility_factor = K 
    # Use J0 parameter directly
    denominator = J0 if J0 is not None else 3 * kappa
    
    if debug and len(jump_sizes) > 0:
        max_jump = np.max(jump_sizes)
        max_exponent = max_jump / denominator
        print(f"  K: {K:.6f}")
        print(f"  Max jump: {max_jump:.6f}")
        print(f"  J0 (denominator): {denominator:.6f}")
        print(f"  Max exponent: {max_exponent:.6f}")
        print(f"  exp(max_exponent): {np.exp(max_exponent):.6f}")
        print(f"  fertility_factor: {fertility_factor:.6f}")
    
    # Calculate fertility: f(J_k) = K * exp(J_k/J0)
    jump_size_factor = jump_sizes / denominator
    exp_values = safe_exp(jump_size_factor)
    fertility_values = fertility_factor * exp_values
    
    if debug and len(fertility_values) > 0:
        print(f"  Max fertility: {np.max(fertility_values):.6f}")
        print(f"  Min fertility: {np.min(fertility_values):.6f}")
        print(f"  Avg fertility: {np.mean(fertility_values):.6f}")
    
    return fertility_values

# Simulation process
print("Starting simulation...")
for t in range(1, T):
    # Calculate mispricing indicator - equation (X_t)
    # X_t = (1-a)(r_{t-1} - r̄) + aX_{t-1}
    X_t[t] = (1-a)*(returns[t-1] - r_bar) + a*X_t[t-1]
    
    # Calculate time-varying self-excitation parameter
    # η_t = η_0 * (1 + ReLU(X_t/s0))
    eta_t[t] = eta_0 * (1 + max(0, X_t[t]/s0))
    ReLU_X_t[t] = max(0, X_t[t]/s0)  # Save ReLU part for analysis
    
    # Calculate baseline intensity - equation (μ_t) (with ν=0)
    # μ_t = max(0, νX_t + μ̄) = μ̄
    mu_t[t] = mu_bar  # Since ν=0, μ_t = μ̄
    
    # Generate exogenous jumps - equation (I_t)
    # I_t ~ Pois(μ_t)
    I_t[t] = np.random.poisson(mu_t[t])
    
    # Calculate endogenous jumps - equation (O_t)
    # O_t ~ Pois(η_t ∑_{j=1}^{t-1} h_{t-j} ∑_{i=1}^{N_j} f_j^i)
    if t > 1:
        # Calculate past jump influence using exponential memory kernel
        past_influence = 0
        
        # List for diagnostics
        influence_by_day = []
        
        # Iterate through jump history
        for idx, (past_time, jumps, fertility) in enumerate(jump_history):
            lag = t - past_time
            if lag <= max_memory_days:
                # h(u) = χe^(-χu) where u = lag
                memory_effect = memory_kernel[lag-1]
                daily_influence = memory_effect * np.sum(fertility)
                past_influence += daily_influence
                
                # Collect diagnostic information
                if t % 1000 == 0:
                    influence_by_day.append((lag, daily_influence, past_time))
        
        # Store past influence for diagnostics
        past_influences[t] = past_influence
        
        # Calculate Poisson parameter
        poisson_param = eta_t[t] * past_influence
        
        # Update diagnostic variables
        max_past_influence = max(max_past_influence, past_influence)
        max_poisson_param = max(max_poisson_param, poisson_param)
        
        # Generate endogenous jumps
        try:
            O_t[t] = np.random.poisson(poisson_param)
        except (ValueError, OverflowError) as e:
            print(f"Warning at day {t}: Poisson parameter too large ({poisson_param:.2e}), setting O_t[t] = 0")
            print(f"Error: {e}")
            O_t[t] = 0
        
        # Print diagnostic information every 1000 days or when issues occur
        if t % 1000 == 0 or poisson_param > 1e10:
            print(f"\n=== Day {t} Diagnostic Information ===")
            print(f"  X_t: {X_t[t]:.6f}")
            print(f"  eta_t: {eta_t[t]:.6f}")
            print(f"  Past influence: {past_influence:.6e}")
            print(f"  Poisson parameter: {poisson_param:.6e}")
            print(f"  Memory kernel decay (chi): {chi}")
            print(f"  Maximum recorded fertility: {max_fertility:.6e}")
            print(f"  Maximum recorded jump size: {max_jump_size:.6f}")
            
            # Print days with largest contribution to past influence
            if influence_by_day:
                influence_by_day.sort(key=lambda x: x[1], reverse=True)
                print("  Top 5 days contributing to past influence:")
                for lag, influence, day_index in influence_by_day[:5]:
                    print(f"    {lag} days ago (t={day_index}): {influence:.6e}")
    else:
        O_t[t] = 0
        past_influences[t] = 0
    
    # Total jumps - equation (N_t)
    # N_t = I_t + O_t
    N_t[t] = I_t[t] + O_t[t]
    
    # Generate individual jump sizes and calculate fertility
    if N_t[t] > 0:
        # Generate N_t[t] exponentially distributed jump sizes
        # J_t^i ~ (1/κ)e^(-J_t^i/κ)
        individual_jump_sizes = np.random.exponential(kappa, int(N_t[t]))
        
        # Record maximum jump size
        current_max_jump = np.max(individual_jump_sizes) if len(individual_jump_sizes) > 0 else 0
        max_jump_size = max(max_jump_size, current_max_jump)
        
        # Calculate fertility for each jump using updated formula
        # f_j^i = K * e^(J_j^i/J0) with J0 = 3κ
        debug_fertility = (t % 1000 == 0)
        fertility_values = calculate_fertility(individual_jump_sizes, kappa, a, K, J0, debug=debug_fertility)

        # Record fertility data
        for f_val in fertility_values:
            all_fertility_values.append(f_val)
            all_fertility_days.append(t)
        
        # Record maximum fertility
        current_max_fertility = np.max(fertility_values) if len(fertility_values) > 0 else 0
        max_fertility = max(max_fertility, current_max_fertility)
        
        # Store current time's jump data in history
        jump_history.appendleft((t, individual_jump_sizes, fertility_values))
        
        # Calculate total jump impact
        # J_t = ∑_{i=1}^{N_t} J_t^i
        total_jump_impacts[t] = np.sum(individual_jump_sizes)
    else:
        # If no jumps, still add empty record to maintain correct time indexing
        jump_history.appendleft((t, np.array([]), np.array([])))
        total_jump_impacts[t] = 0
    
    # Calculate total jump intensity - equation (λ_t)
    # λ_t = μ_t + η_t ∑_{j=1}^{t-1} h_{t-j} ∑_{i=1}^{N_j} f_j^i
    lambda_t[t] = mu_t[t] + eta_t[t] * past_influences[t] if t > 1 else mu_t[t]
    
    # Calculate expected jump impact (compensator)
    # E[J_t|F_{t-1}] = κ * Λ_t where Λ_t = ∫_{t-1}^t λ(s) ds ≈ λ_t
    expected_jump_impacts[t] = kappa * lambda_t[t]
    
    # Calculate jump surprise (J_t - κλ_t)
    jump_surprise[t] = total_jump_impacts[t] - expected_jump_impacts[t]
    
    # Generate returns - equation (r_t)
    # r_t = r̄ + σ_tε_t - (J_t - κλ_t)
    # Since sigma_t = 0, we remove the normal component
    returns[t] = r_bar - jump_surprise[t]
    
    # Update log prices
    log_prices[t] = log_prices[t-1] + returns[t]
    
    # Calculate √E[(r_{t-1} - r̄)²] and branching ratio
    if t >= 252:  # Use at least 1 year of data
        # Calculate variance estimate (population variance, divide by n not n-1)
        returns_window = returns[max(0, t-252):t]
        variance_estimate = np.sum((returns_window - r_bar)**2) / len(returns_window)
        sqrt_variance[t] = np.sqrt(variance_estimate)  # √E[(r_{t-1} - r̄)²]
        
        # Calculate Self-Excitement Strength (SES)
        # SES = √((1-a)/(1+a)) * √E[(r_{t-1} - r̄)²] / (s0√(2π))
        SES[t] = np.sqrt((1-a)/(1+a)) * sqrt_variance[t] / (s0 * np.sqrt(2*np.pi))
        
        # Calculate branching ratio - according to improved formula in paper
        # n = η_0 * (J0/(J0-κ)) * K * (1 + SES)
        branching_ratio[t] = eta_0 * (J0/(J0-kappa)) * K * (1 + SES[t])
    
    # Print status every 1000 days
    if t % 1000 == 0:
        print(f"Day {t}: X_t={X_t[t]:.4f}, μ_t={mu_t[t]:.4f}, η_t={eta_t[t]:.4f}, λ_t={lambda_t[t]:.4f}")

print("Simulation completed!")

# Print final diagnostic information
print("\nFinal Diagnostic Information:")
print(f"Maximum fertility: {max_fertility:.6e}")
print(f"Maximum jump size: {max_jump_size:.6f}")
print(f"Maximum past influence: {max_past_influence:.6e}")
print(f"Maximum Poisson parameter: {max_poisson_param:.6e}")
print(f"Memory decay parameter (chi): {chi}")
print(f"J0 parameter: {J0:.6f}")

# Create results dataframe
results = pd.DataFrame({
    'Log_Price': log_prices,
    'Price': np.exp(log_prices),
    'Return': returns,
    'Jumps_Count': N_t,
    'Total_Jump_Impact': total_jump_impacts,
    'Expected_Jump_Impact': expected_jump_impacts,
    'Jump_Surprise': jump_surprise,
    'Lambda': lambda_t,
    'Mu': mu_t,
    'Eta': eta_t,
    'Mispricing': X_t,
    'ReLU_X_t': ReLU_X_t, 
    'Past_Influence': past_influences,
    'Branching_Ratio': branching_ratio,
    'Sqrt_Variance': sqrt_variance,
    'SES': SES
})

# Calculate Branching Rate Simplification
print("Calculating Branching Rate Simplification...")

# Corrected formula: n = self-excitation component / (self-excitation component + μ)
self_excitation_component = past_influences * eta_t
branching_rate_simplified = np.zeros(T)

for t in range(T):
    denominator = self_excitation_component[t] + mu_t[t]
    if denominator > 0:  # Avoid division by zero
        branching_rate_simplified[t] = self_excitation_component[t] / denominator
    else:
        branching_rate_simplified[t] = 0

# Add new indicator to results DataFrame
results['Branching_Rate_Simplified'] = branching_rate_simplified

# Create a combined price and jump intensity plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.semilogy(results['Price'], 'k-', label='Price')
# Add mean return trend line
trend_line = 100 * np.exp(r_bar * np.arange(T))
ax1.semilogy(trend_line, 'gray', linestyle='--', label='Expected Growth', alpha=0.7)
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Log Price')
ax1.grid(True, which="both", ls="-", alpha=0.2)
ax1.tick_params(axis='y', labelcolor='black')

# Create secondary y-axis for jump intensity
ax2 = ax1.twinx()
ax2.plot(lambda_t, 'b-', label='λ', alpha=0.2)
ax2.set_ylabel('Jump Intensity (λ)', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Price (log scale) and Jump Intensity')
plt.tight_layout()
plt.savefig('price_and_intensity_combined.png', dpi=300)
plt.show()

# Create 10 charts, each showing 1000 days
time_ranges = [
    (0, 1000, "1-1000"),
    (1000, 2000, "1001-2000"),
    (2000, 3000, "2001-3000"),
    (3000, 4000, "3001-4000"),
    (4000, 5000, "4001-5000"),
    (5000, 6000, "5001-6000"),
    (6000, 7000, "6001-7000"),
    (7000, 8000, "7001-8000"),
    (8000, 9000, "8001-9000"),
    (9000, 10000, "9001-10000")
]

for start_idx, end_idx, label in time_ranges:
    # Create a combined plot showing the specified range of days
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot price data
    x_range = range(end_idx - start_idx)
    ax1.semilogy(x_range, results['Price'][start_idx:end_idx], 'k-', label='Price')
    
    # Add mean return trend line starting from the initial price of this period
    if start_idx == 0:
        initial_price = 100  # Initial price at day 0
    else:
        initial_price = results['Price'][start_idx-1]  # Price at day before start
    
    trend_line = initial_price * np.exp(r_bar * np.arange(end_idx - start_idx))
    ax1.semilogy(x_range, trend_line, 'gray', linestyle='--', label='Expected Growth', alpha=0.7)
    
    # Configure primary y-axis
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Log Price')
    ax1.grid(True, which="both", ls="-", alpha=0.7)
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Create secondary y-axis for jump intensity
    ax2 = ax1.twinx()
    ax2.plot(x_range, lambda_t[start_idx:end_idx], 'b-', label='λ', alpha=0.2)
    ax2.set_ylabel('Jump Intensity (λ)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Set x-axis ticks to show actual days
    tick_positions = [0, 200, 400, 600, 800, 999]
    tick_labels = [str(start_idx + pos) for pos in tick_positions]
    plt.xticks(tick_positions, tick_labels)
    
    plt.title(f'Price (log scale) and Jump Intensity - Days {label}')
    plt.tight_layout()
    plt.savefig(f'price_and_intensity_{label}.png', dpi=300)
    plt.show()

# =================== First chart: First 5 indicators ===================
plt.figure(figsize=(16, 16))

# Chart 1 - Jump Count
plt.subplot(5, 1, 1)
active_days = N_t > 0
days = np.arange(T)
if np.any(active_days):
    scatter = plt.scatter(days[active_days], N_t[active_days], 
                         c=N_t[active_days], cmap='plasma', s=25, alpha=0.8, edgecolors='black', linewidth=0.5)
plt.title('Active Days Only (Jump Count > 0)')
plt.ylabel('Jump Count')
plt.grid(True, alpha=0.3)

# Chart 2 - Mispricing
plt.subplot(5, 1, 2)
plt.plot(X_t, 'r-')
plt.title('Mispricing (X_t)')
plt.ylabel('X_t')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Chart 3 - Self-Excitation Parameter
plt.subplot(5, 1, 3)
plt.plot(eta_t, 'g-')
plt.title('Self-Excitation Parameter (η_t)')
plt.ylabel('η_t')
plt.grid(True)

# Chart 4 - Fertility Values
plt.subplot(5, 1, 4)
if len(all_fertility_values) > 0:
    scatter = plt.scatter(all_fertility_days, all_fertility_values, 
                         c=all_fertility_values, cmap='coolwarm', s=8, alpha=0.8, edgecolors='navy', linewidth=0.3)
plt.title('Individual Fertility Values')
plt.ylabel('Fertility')
plt.grid(True, alpha=0.3)

# Chart 5 - Jump Intensity Components
plt.subplot(5, 1, 5)
plt.plot(lambda_t, 'r-', label='Total Intensity (λ)')
plt.plot(mu_t, 'b--', label='Baseline Intensity (μ)')
plt.plot(past_influences * eta_t, 'g-', alpha=0.5, label='Self-Excitation Component')
plt.title('Jump Intensity Components')
plt.ylabel('Intensity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model_indicators_part1.png', dpi=300)
plt.show()

# =================== Second chart: Last 5 indicators ===================
plt.figure(figsize=(16, 20))

# Chart 1 - Returns
plt.subplot(5, 1, 1)
plt.plot(returns, 'g-')
plt.title('Log Returns')
plt.ylabel('Returns')
plt.grid(True)

# Chart 2 - Jump Surprise (J_t - κλ_t)
plt.subplot(5, 1, 2)
plt.plot(jump_surprise, 'red', linewidth=1.0, alpha=0.8)
plt.title('Jump Surprise (J_t - κλ_t)')
plt.ylabel('J_t - κλ_t')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Zero Line')
plt.legend()
plt.grid(True)

# Chart 3 - SES
plt.subplot(5, 1, 3)
plt.plot(SES, 'orange')
plt.title('Self-Excitation Strength (SES)')
plt.ylabel('SES')
plt.grid(True)

# Chart 4 - Branching Ratio
plt.subplot(5, 1, 4)
plt.plot(branching_ratio, 'purple')
plt.title('Branching Ratio')
plt.ylabel('Branching Ratio')
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Critical=1')
plt.legend()
plt.grid(True)

# Chart 5 - Branching Rate Simplification
plt.subplot(5, 1, 5)
plt.plot(branching_rate_simplified, 'darkblue', linewidth=1.5)

# Use average self-excitation parameter to calculate a more stable mean
average_self_excitation = np.mean(branching_rate_simplified)
average_branching_rate = average_self_excitation

plt.axhline(y=average_branching_rate, color='orange', linestyle='--', alpha=0.7, 
            label=f'Avg Self-Excitation = {average_branching_rate:.3f}')

plt.title('Branching Ratio Simplification \nn = Self-Excitation / (Self-Excitation + μ)')
plt.ylabel('Simplified BR')
plt.xlabel('Days')
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='100% Self-Excitation')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.1)

plt.tight_layout()
plt.savefig('model_indicators_part2.png', dpi=300)
plt.show()