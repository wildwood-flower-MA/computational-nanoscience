import numpy as np
import matplotlib.pyplot as plt
import kwant
from kwant.wraparound import wraparound, plot_2d_bands

def makesystem_magnetic(p, q, t=1.0, plot: bool = False):

    def onsite(site):
        return 4*t
    
    def h_x(site1, site2):
        return -t

    # A = (0, Bx, 0)
    def h_y(site1, site2):
        x = site1.pos[0]
        B = 2*np.pi*p/q
        return -t*np.exp(-1j*B*x)
    
    lat = kwant.lattice.square(a=1, norbs=1)
    sym = kwant.TranslationalSymmetry((q, 0), (0, 1))
    system = kwant.Builder(sym)
    for x in range(q):
        system[lat(x, 0)] = onsite

    system[kwant.builder.HoppingKind((1, 0), lat, lat)] = h_x
    system[kwant.builder.HoppingKind((0, 1), lat, lat)] = h_y
    system_wrapped = wraparound(system).finalized()

    if plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        plot_2d_bands(system_wrapped, k_x=51, k_y=51, ax=ax)
        
        ax.set_title(rf"Bands for $\phi=2\pi\ {p}/{q}$")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_zlabel(r"$E$") # type: ignore
        plt.show()

    return system_wrapped

q = 201
p_linspace =  np.arange(1, q, step = 1)
energies = np.zeros((q,q))
for idxp, p in enumerate(p_linspace):

    syst = makesystem_magnetic(p=p, q=q, t=1, plot = False)
    params_k0 = {'k_x': 0, 'k_y': 0}
    H_k0 = syst.hamiltonian_submatrix(params=params_k0, sparse=False) # type: ignore
    evals = np.linalg.eigvalsh(H_k0)
    energies[idxp, :] = evals

fig, ax = plt.subplots(1,1, figsize = (4,4))
ax.plot(energies, ',', color = 'k') #, markersize = 0.4)
ax.set_ylabel("E")
fig.suptitle(r"$B = 2\pi\ p/q$ ")
ax.set_xlabel("p")
ax.set_xlim(0,q)
plt.savefig("magnetic_system.png")
plt.show()