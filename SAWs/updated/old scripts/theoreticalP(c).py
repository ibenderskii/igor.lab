import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def m_star(beta, T, mu0, sigma0, h, s):
    # From completing the square:
    # m*(beta) = mu0 - beta * sigma0^2 * (h - T s)
    return mu0 - beta * (sigma0**2) * (h - T*s)

def sigma_eff2(beta, T, sigma0):
    # For a Gaussian prior and linear energy tilt, variance stays sigma0^2.
    # Replace this if your model gives beta-dependent broadening.
    return 0.5*(sigma0**2)*(T/343)**2

def gaussian_pdf(m, mu, var):
    return (1.0 / np.sqrt(2*np.pi*var)) * np.exp(-(m-mu)**2/(2*var))

def plot_overlaid_curves(
    temps,
    mu0=5.095,
    sigma0=2.803,
    h=534.759,
    s=1.55743,
    m_min=0,
    m_max=40,
    n_m=2000,
):
    m = np.linspace(m_min, m_max, n_m)

    fig, ax = plt.subplots(figsize=(8,5))

    norm = Normalize(vmin=min(temps), vmax=max(temps))
    sm = ScalarMappable(norm=norm, cmap="coolwarm")
    sm.set_array([])

    for T in temps:
        beta = 1.0 / T
        mu = m_star(beta, T, mu0, sigma0, h, s)
        var = sigma_eff2(beta, T, sigma0)
        p = gaussian_pdf(m, mu, var)
        ax.plot(m, p, color=sm.to_rgba(T), linewidth=2)

    ax.set_xlabel("m (contacts)")
    ax.set_ylabel("P(m | T)")
    ax.set_title("Gaussian approximation for P(m | T) overlaid across T")
    ax.set_xlim(m_min, m_max)

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("T")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    temps = np.linspace(280, 360, 32)  # example temperatures
    plot_overlaid_curves(
        temps,
        mu0=5.095,
    sigma0=2.803,
    h=534.759,
    s=1.55743,
        m_min=0,
        m_max=30
    )