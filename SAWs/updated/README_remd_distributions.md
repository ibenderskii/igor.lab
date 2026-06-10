# remd_distributions_30mer.npz

Temperature-dependent distributions of Rg and hydrophobic contacts for a PNIPAM 30-mer from demuxed REMD simulations (64 replicas, 280–360 K, OPLS-AA/SPC/E).

## Definitions

**Contacts (isopropyl hydrophobic contacts):** How much the hydrophobic sidechains are clustering together. For each of the 30 residues, we take the geometric center of the 10-atom isopropyl group (CH(CH₃)₂ + hydrogens). Then we evaluate a switching function over all 435 pairs:

```
s(r) = (1 - (r/0.7)^6) / (1 - (r/0.7)^12)
```

This gives ~1 for pairs closer than 0.7 nm and ~0 for pairs farther apart. The total contact number is the sum over all pairs. More contacts = more collapsed.

## Contents

```python
data = np.load("remd_distributions_30mer.npz")
```

| Key | Shape | Description |
|-----|-------|-------------|
| `temps` | (64,) | Temperatures in K, sorted |
| `rg_centers` | (149,) | Rg histogram bin centers (nm) |
| `ct_centers` | (149,) | Contact histogram bin centers |
| `rg_hists` | (64, 149) | P(Rg) at each temperature |
| `ct_hists` | (64, 149) | P(contacts) at each temperature |

## Quick start

```python
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
```
