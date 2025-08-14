
from __future__ import annotations
import numpy as np, argparse, matplotlib.pyplot as plt
from numba import njit
from kde_utils import load_kde_as_u0 

# ------------------------------------------------------------------
# 1. parameters
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--R', type=float, default=1.0, help='outer radius')
parser.add_argument('--Nr', type=int,   default=1000, help='# radial nodes (incl. r=0)')
parser.add_argument('--tmax', type=float, default=80, help='total simulation time')
parser.add_argument('--cfl',  type=float, default=0.4, help='Courant safety factor (<1)')
parser.add_argument('--profile', choices=['gaussian','shell', 'given'], default='given',
                    help='initial condition')
parser.add_argument('--D_profile', choices=['const','tanh', 'inv', 'inv2'], default='tanh',
                    help='radial dependence of D(r)')
args = parser.parse_args()


# ------------------------------------------------------------------
# 2. radial‑dependent diffusion coefficient
# ------------------------------------------------------------------
R, Nr = args.R, args.Nr
r     = np.linspace(0.0, R, Nr)
f = 4*np.pi*r**2
dr    = r[1] - r[0]
D_cap = 1e8

if args.D_profile == 'const':
    D_node = np.full(Nr, 1.0)
elif args.D_profile == 'tanh':   # tanh: slow core, faster shell
    D_shell = 0.002
    D_core  = 0.0001
    r_c     = 0.2
    w       = 0.1
    D_node = D_shell - 0.5*(D_shell - D_core)*(1 - np.tanh((r - r_c)/w))
elif args.D_profile == 'inv': 
     D_node = 1/np.maximum(r,1e-6)
elif args.D_profile == 'inv2':
    D_node = 1/(np.maximum(r,1e-6)**2)

D_node = np.minimum(D_node, D_cap)

# harmonic‑mean diffusion at interfaces (length Nr‑1)
D_if = 2*D_node[:-1]*D_node[1:] / (D_node[:-1] + D_node[1:])
r_if = 0.5*(r[:-1] + r[1:])

# pick dt from  CFL:
Dt_max =   args.cfl * dr**2 / np.max(D_node)
Nt     = int(np.ceil(args.tmax / Dt_max))
dt     = args.tmax / Nt
print(f"Using dt={dt:.3e}, Nt={Nt}, dr={dr:.3e}")

# ------------------------------------------------------------------
# 3. initial condition u(r,0)
# ------------------------------------------------------------------
if args.profile == 'gaussian':
    p0 = np.exp(-((r-0.2)/0.1)**2)
    

elif args.profile == 'shell':  # thin shell near 0.6 R
    p0 = np.exp(-((r-0.6*R)/0.02)**2)
elif args.profile == 'given':
    pkl_path = r"C:\Users\ibend\data\rg_kde_data_44mer.pkl"
    T_sel = 297
    p0 = load_kde_as_u0(pkl_path, T_sel, r, R)

# normalise probability 
u =  4* np.pi*(r**2) *p0
u_norm = np.trapezoid(u, r)
u /= u_norm


# ------------------------------------------------------------------
# 4.  matrices 
# ------------------------------------------------------------------
a  = np.zeros(Nr)  # sub‑diag implicit
b  = np.zeros(Nr)  # main diag implicit
c  = np.zeros(Nr)  # super diag implicit

aE = np.zeros(Nr)  # explicit 
bE = np.zeros(Nr)
cE = np.zeros(Nr)

for j in range(1, Nr-1):
    Ajm = 0.5*dt * (r_if[j-1]**2 * D_if[j-1]) / (dr**2 * (r[j]**2))
    Ajp = 0.5*dt * (r_if[j]  **2 * D_if[j]  ) / (dr**2 * (r[j]**2))
    a[j]   = -Ajm
    c[j]   = -Ajp
    b[j]   = 1 + Ajm + Ajp
    aE[j]  =  Ajm
    cE[j]  =  Ajp
    bE[j]  = 1 - (Ajm + Ajp)

# --- symmetry BC at r=0 (j=0): u_{-1}=u_1 --------------------------

Ajp   = 1.5*dt * (D_if[0]) / ((dr**2) )  # avoid 0/0
a[0] = 0.0
b[0]  = 1 + Ajp
c[0]  = -Ajp
aE[0] = 0.0
bE[0] = 1 - Ajp
cE[0] =  Ajp





# --- absorbing BC at outer wall r=R (j=Nr-1): u=0 ------------------
a[-1] = 0.0; b[-1] = 1.0; c[-1] = 0.0
bE[-1]= 1.0

# ------------------------------------------------------------------
# 5. Thomas solver
# ------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray): 
    n = len(b)
    cp = np.empty(n-1, dtype=np.float64)
    dp = np.empty(n, dtype=np.float64)
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i]*cp[i-1]
        if denom == 0:
            raise ZeroDivisionError
        cp[i] = c[i]/denom
        dp[i] = (d[i] - a[i]*dp[i-1]) / denom
    dp[-1] = (d[-1] - a[-1]*dp[-2]) / (b[-1] - a[-1]*cp[-2])
    x = np.empty(n, dtype=np.float64)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

# ------------------------------------------------------------------
# 6.  integration loop
# ------------------------------------------------------------------
profiles, ts = [u.copy()], [0.0]
probs = [u.copy()]
rhs = np.empty_like(u)
for n in range(Nt):
    rhs[1:-1] = (bE[1:-1]*u[1:-1] +
                 aE[1:-1]*u[:-2]   +
                 cE[1:-1]*u[2:])      # no wrap-around
    rhs[0]  = bE[0]*u[0] + cE[0]*u[1]
    rhs[-1] = 0.0                        # absorbing outer BC    This causes problems because renormalizing divides by a smaller and smaller thing, because some probaility leaks out each time. 
    u = solve_tridiagonal(a, b, c, rhs)
                        
    u[-1] = 0.0                          # absorbing at R

    
    
    # If i want i can normalise prob with 
    u /= np.trapezoid(u, r)

    if (n+1) % max(1, Nt//10) == 0:
        profiles.append(u.copy())
        ts.append((n+1)*dt)
        pr=f*u.copy()
        probs.append(pr)

# ------------------------------------------------------------------
# 7. plot --------------------------------------------------------
plt.figure(figsize=(6,4))
for prof, tcur in zip(profiles, ts):
    plt.plot(r, prof, label=f"t={tcur:.3f}")
plt.xlabel('radius r'); plt.ylabel('spherical prob density p(r,t)')
plt.title('Spherical diffusion with D(r)')
plt.legend(); plt.tight_layout(); plt.show()


plt.figure(figsize=(6,4))
for prob, tcur in zip(probs, ts):
    plt.plot(r, prob, label=f"t={tcur:.3f}")
plt.xlabel('radius r'); plt.ylabel('Radial Prob P(r,t)')
plt.title('Spherical diffusion with D(r)')
plt.legend(); plt.tight_layout(); plt.show()




