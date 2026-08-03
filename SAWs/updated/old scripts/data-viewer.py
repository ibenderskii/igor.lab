


#| Key | Shape | Description |
#|-----|-------|-------------|
#| `temps` | (64,) | Temperatures in K, sorted |
#| `rg_centers` | (149,) | Rg histogram bin centers (nm) |
#| `ct_centers` | (149,) | Contact histogram bin centers |
#| `rg_hists` | (64, 149) | P(Rg) at each temperature |
#| `ct_hists` | (64, 149) | P(contacts) at each temperature |


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = np.load("remd_distributions_30mer.npz")
temps = data["temps"]
norm = plt.Normalize(temps.min(), temps.max())
cmap = cm.coolwarm

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for i, T in enumerate(temps):
    color = cmap(norm(T))
    ax1.plot(data["rg_centers"], data["rg_hists"][i], color=color, alpha=0.7, lw=1)
    ax2.plot(data["ct_centers"], data["ct_hists"][i], color=color, alpha=0.7, lw=1)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
plt.colorbar(sm, ax=ax1).set_label("T (K)")
plt.colorbar(sm, ax=ax2).set_label("T (K)")
ax1.set_xlabel("Rg (nm)"); ax1.set_ylabel("P(Rg)")
ax2.set_xlabel("Contacts"); ax2.set_ylabel("P(contacts)")
plt.tight_layout()
plt.show()

