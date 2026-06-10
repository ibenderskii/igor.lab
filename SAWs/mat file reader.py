import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


# ---------------------------
# 1) Load the MATLAB .mat file
# ---------------------------
mat_path = "250623_HDVE_PNIPAM_22kDa_2wtpct_D2O_Ti31_100mW_1195V_1LO_Gr1060nm_ZZZZ_r1_proc.mat"
mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

# ---------------------------
# 2) Pull out the experimental arrays we need
# ---------------------------
Ti = float(mat["Ti"])   # °C
Tf = float(mat["Tf"])   # °C

w3 = np.asarray(mat["w3"], dtype=float)              # frequency axis (cm^-1), shape (64,)
Y2d = np.asarray(mat["AlltLO"], dtype=float)         # shape (nTemps, 64); you can swap to AlltDPP if desired

nTemps = Y2d.shape[0]

# If the file doesn't store an explicit temperature vector (it doesn't), create one:
T_exp = np.linspace(Ti, Tf, nTemps)                  # °C, shape (nTemps,)

# ---------------------------
# 3) Convert the 2D spectra into a 1D "melting curve"
#    Choose a target frequency and take a small window average around it.
# ---------------------------
target_cm1 = 1630.0           # change this if you want (e.g. 1609, 1656, 1581)
half_window_cm1 = 3.0         # average over target ± 3 cm^-1 (helps with noise)

mask = (w3 >= target_cm1 - half_window_cm1) & (w3 <= target_cm1 + half_window_cm1)
if not np.any(mask):
    # fallback: nearest single index
    idx = int(np.argmin(np.abs(w3 - target_cm1)))
    y_exp = Y2d[:, idx]
    freq_used = w3[idx]
else:
    y_exp = Y2d[:, mask].mean(axis=1)
    freq_used = target_cm1

# Optional: normalize experiment to 0-1 so it overlays nicely with many simulations
def norm01(y):
    y = np.asarray(y, dtype=float)
    return (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y))

y_exp_n = norm01(y_exp)

# ---------------------------
# 4) Your simulation data
# ---------------------------
# You said your simulation data is in a python array.
# Here are the two common cases:

# Case A: you already have simulation values on the SAME temperature grid as experiment:
# y_sim = your_sim_array_of_length_nTemps

# Case B: you have simulation temperatures + values:
# T_sim = your_temperature_array
# y_sim = your_sim_values_array

# ---- EDIT THIS PART ----
# Example placeholders (replace with your real arrays):
y_sim = None        # <-- replace with your numpy array
T_sim = None        # <-- replace if your sim has its own temperature axis
# ------------------------

if y_sim is None:
    raise ValueError("Set y_sim to your simulation array before running.")

y_sim = np.asarray(y_sim, dtype=float)

# Normalize simulation too (optional but usually helpful)
y_sim_n = norm01(y_sim)

# Decide which x-axis to use for sim
if T_sim is None:
    if y_sim.shape[0] != nTemps:
        raise ValueError(
            f"Your y_sim has length {y_sim.shape[0]}, but experiment has {nTemps} temps. "
            "Either resample/interpolate, or provide T_sim + y_sim."
        )
    T_sim_plot = T_exp
else:
    T_sim_plot = np.asarray(T_sim, dtype=float)
    if T_sim_plot.shape[0] != y_sim.shape[0]:
        raise ValueError("T_sim and y_sim must be the same length.")

# ---------------------------
# 5) Plot
# ---------------------------
plt.figure()
plt.plot(T_exp, y_exp_n, "o", label=f"Experiment (AlltLO, ~{freq_used:.1f} cm$^{{-1}}$)")
plt.plot(T_sim_plot, y_sim_n, "-", label="Simulation")

plt.xlabel("Temperature (°C)")
plt.ylabel("Normalized signal (0–1)")
plt.title("PNIPAM melting curve: experiment vs simulation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
