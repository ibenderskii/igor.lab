import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Model + "bare" statistics
# ------------------------

N = 44  # chain length

# These should come from your J=0 simulation (mean and std of contacts)
mu0    = 15.0     # <m> at J=0
sigma0 = 2.0     # std(m) at J=0
sigma02 = sigma0**2

# Cooperative model parameters
eps0 = 1.0       # scale of temperature-dependent contact strength
Tc   = 2.0       # LCST (in same units as your T)
J    = 0.05      # cooperativity parameter (keep modest so Gaussian stays stable)

# Temperatures to sample (should straddle Tc)
T_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# Max contacts we'll consider
m_max  = 40
m_vals = np.arange(0, m_max + 1)

plt.figure(figsize=(7,5))

for T in T_list:
    beta = 1.0 / T

    # Temperature-dependent contact strength (LCST-like):
    # negative below Tc, positive above Tc.
    eps_T = eps0 * (T - Tc)

    # Denominator for m_* and sigma_eff
    denom = 1.0 - beta * J * sigma02
    if denom <= 0:
        print(f"Skipping T={T}: 1 - beta*J*sigma0^2 <= 0 (Gaussian breaks).")
        continue

    # Mean and variance in interacting (cooperative) ensemble
    m_star = (mu0 + beta * eps_T * sigma02) / denom
    sigma_eff2 = sigma02 / denom
    sigma_eff  = np.sqrt(sigma_eff2)

    # Clamp m_star to [0, m_max] for sanity in plotting
    m_star_clamped = np.clip(m_star, 0, m_max)

    print(f"T={T:4.2f}: eps(T)={eps_T:5.2f}, m_star≈{m_star:6.2f} "
          f"(clamped {m_star_clamped:5.2f}), sigma_eff≈{sigma_eff:4.2f}")

    # Discrete Gaussian over m = 0..m_max, centered at m_star
    P_unnorm = np.exp(-(m_vals - m_star)**2 / (2.0 * sigma_eff2))

    # Normalize
    Z_disc = P_unnorm.sum()
    P = P_unnorm / Z_disc

    # Plot
    plt.plot(m_vals, P, marker='o', linestyle='-', label=f"T = {T:g}")

plt.xlabel("Number of favorable contacts m")
plt.ylabel("Probability P(m | T)")
plt.title(f"Expected Contact Distributions for N = {N}")
plt.legend()
plt.tight_layout()
plt.show()
