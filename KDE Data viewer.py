import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- 1) Load the pickle safely (Windows path fix) ---
pkl_path = Path(r"C:\Users\ibend\data\rg_kde_data_44mer.pkl")  # r-string avoids \U escapes
if not pkl_path.exists():
    raise FileNotFoundError(f"File not found: {pkl_path}")

with pkl_path.open("rb") as f:
    kde_data = pickle.load(f)

# --- 2) Make temperature keys easier to use (coerce strings -> ints where possible) ---
def coerce_temp_keys(d):
    out = {}
    for k, v in d.items():
        try:
            out[int(k)] = v  # convert "350" -> 350
        except (ValueError, TypeError):
            out[k] = v
    return out

kde_by_temp = coerce_temp_keys(kde_data)
print("Available temps:", sorted([k for k in kde_by_temp.keys() if isinstance(k, (int, float, np.integer))]))

# --- 3) Single plot for a temperature ---
temp = 293
if temp not in kde_by_temp:
    raise KeyError(f"Temperature {temp} not found; available: {sorted(kde_by_temp.keys())}")

rg = np.asarray(kde_by_temp[temp]['rg_points'])
kde = np.asarray(kde_by_temp[temp]['kde_values'])

plt.figure()
plt.plot(rg, kde)
plt.xlabel('Rg (nm)')
plt.ylabel('Density')
plt.title(f'KDE at {temp} K')
plt.tight_layout()
plt.show()

# --- 4) Compare multiple temperatures ---
temps_to_plot = [300, 350, 400]
plt.figure()
for T in temps_to_plot:
    if T not in kde_by_temp:
        print(f"[warn] Temp {T} not found; skipping.")
        continue
    rg = np.asarray(kde_by_temp[T]['rg_points'])
    kde = np.asarray(kde_by_temp[T]['kde_values'])
    # Optional: sort by rg in case data isn’t ordered
    order = np.argsort(rg)
    plt.plot(rg[order], kde[order], label=f'{T} K')

plt.legend()
plt.xlabel('Rg (nm)')
plt.ylabel('Density')
plt.title('Rg KDE vs Temperature')
plt.tight_layout()
plt.show()