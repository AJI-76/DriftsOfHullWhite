import numpy as np
import matplotlib.pyplot as plt

# Set a fixed random seed for reproducibility
np.random.seed(42)

# Shared Parameters
flt_long_term_mean = 0.05     # Long-term mean interest rate
flt_relaxation = 0.1          # Mean reversion speed
flt_sigma = 0.04              # Volatility
flt_amplitude = 0.02          # Amplitude for harmonic forcing
flt_frequency = 2.0* np.pi / 22  # Frequency for harmonic forcing
num_paths = 1000              # Number of simulated paths
num_steps = 5000               # Number of time steps
dt = 0.01                     # Time step size

# Time vector
time = np.linspace(0, num_steps * dt, num_steps + 1)

# Vectorized Hull-White Model
def hw_model_vectorized():
    r = np.zeros((num_paths, num_steps + 1))
    r[:, 0] = 0.02  # Initial interest rate
    for t in range(1, num_steps + 1):
        dr = flt_relaxation * (flt_long_term_mean - r[:, t - 1]) * dt + \
             flt_sigma * np.random.normal(0, 1, num_paths) * np.sqrt(dt)
        r[:, t] = r[:, t - 1] + dr
    return r

# Vectorized Hull-White Harmonic Forcing Model
def hw_harmonic_model_vectorized():
    r = np.zeros((num_paths, num_steps + 1))
    r[:, 0] = 0.02  # Initial interest rate
    for t in range(1, num_steps + 1):
        harmonic_term = flt_amplitude * np.sin(flt_frequency * time[t - 1])
        dr = flt_relaxation * (flt_long_term_mean - r[:, t - 1] + harmonic_term) * dt + \
             flt_sigma * np.random.normal(0, 1, num_paths) * np.sqrt(dt)
        r[:, t] = r[:, t - 1] + dr
    return r

# Plotting function for mean path only
def plot_mean_path(mean_path, title):
    plt.figure(figsize=(10, 5))
    plt.plot(mean_path, color='blue', linewidth=2)
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Interest Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Simulate and plot for Standard Hull-White Model
paths_hw = hw_model_vectorized()
mean_path_hw = np.mean(paths_hw, axis=0)
plot_mean_path(mean_path_hw, "Mean Interest Rate Path (Hull-White Model)")

# Simulate and plot for Hull-White Harmonic Forcing Model
paths_hw_harmonic = hw_harmonic_model_vectorized()
mean_path_hw_harmonic = np.mean(paths_hw_harmonic, axis=0)
plot_mean_path(mean_path_hw_harmonic, "Mean Interest Rate Path (Hull-White Harmonic Forcing)")


