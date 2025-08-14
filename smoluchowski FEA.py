import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Problem parameters
# ---------------------------
R = 10.0        # outer radius
Nr = 200        # number of radial points
dr = R / (Nr-1) # radial step size
dt = 0.0005     # time step
Nt = 2000       # number of time steps

# Define radial grid
r = np.linspace(0, R, Nr)

# Example: diffusion coefficient D(r)
def D(r):
    return 1.0 + 0.5 * np.exp(- (r-3)**2 / 2.0)

# Example: potential U(r) [optional]
def U(r):
    return 0.0 * r  # flat potential

# ---------------------------
# Initial condition
# ---------------------------
p = np.exp(- (r-2)**2 / 0.2)   # Gaussian around r=2
p /= np.trapz(4*np.pi*r**2*p, r)  # normalize

# ---------------------------
# Precompute coefficients
# ---------------------------
D_vals = D(r)
U_vals = U(r)
dUdr = np.gradient(U_vals, dr)

# Finite difference time evolution (explicit Euler)
for n in range(Nt):
    # Compute flux J_r using spherical form
    dpdr = np.gradient(p, dr)
    J = -D_vals * (dpdr + p * dUdr)  # radial flux

    # Divergence in spherical coords: (1/r^2) d/dr (r^2 J)
    divJ = np.gradient(r**2 * J, dr) / (r**2 + 1e-12)

    p_new = p - dt * divJ

    # No-flux BCs: dp/dr = 0 at r=0 and r=R
    p_new[0] = p_new[1]
    p_new[-1] = p_new[-2]

    # Renormalize to keep total probability = 1
    p_new /= np.trapz(4*np.pi*r**2*p_new, r)

    p = p_new

    # Plot every few steps
    if n % 200 == 0:
        plt.plot(r, p, label=f't={n*dt:.3f}')

plt.xlabel("r")
plt.ylabel("p(r)")
plt.legend()
plt.show()
