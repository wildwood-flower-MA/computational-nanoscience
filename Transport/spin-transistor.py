import kwant
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from typing import Tuple
from matplotlib.ticker import ScalarFormatter
from copy import deepcopy
import cmath

plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10) 

def eV2au(eV: float):
    return 0.03674932587122423*eV

def nm2au(nm: float):
    return 18.89726133921252*nm

def T2au(B: float):
    return 4.254382E-6*B

d_up = np.array([[1 , 0],
                [0 , 0]], dtype = complex)

d_dn = np.array([[0 , 0],
                [0 , 1]], dtype = complex)

s_0 = np.array([[1 , 0],
                [0 , 1]], dtype = complex)

s_x = np.array([[0 , 1],
                [1 , 0]], dtype = complex)

s_y = np.array([[0 , -1j],
                [1j , 0]], dtype = complex)

s_z = np.array([[1 , 0],
                [0 , -1]], dtype = complex)

def disperssion(system: kwant.builder.FiniteSystem | kwant.builder.InfiniteSystem,\
                pm: dict, nr_lead: int,\
                k_max: float, n_k_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    
    '''
    Calculate disperssion relation in a lead (nr_lead).
    '''

    bands = kwant.physics.Bands(system.leads[nr_lead])

    ks = np.linspace(-k_max*pm['dx'], k_max*pm['dx'], n_k_steps)
    Es = [bands(k) for k in ks]

    return (ks/pm['dx'], np.asarray(Es))

def plot_disperssion(disperssion: Tuple[np.ndarray, np.ndarray],
                    y_lim: list, name: str = "_") -> None:
    
    ks, energies = disperssion

    if(y_lim[0] == y_lim[1]):
        y_lim[0] = np.min(energies)/eV2au(1.0)
        y_lim[1] = np.max(energies)/eV2au(1.0)
    
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(ks*nm2au(1.0), energies/eV2au(1.0), color = 'k')
    ax.set_xlabel(r'$k_x$ (1/nm)')
    ax.set_ylabel(r'$E$ (eV)')
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.grid(True)

    fig.tight_layout()
    plt.savefig(name + ".pdf")
    plt.show()

def wavefunction(system: kwant.builder.FiniteSystem | kwant.builder.InfiniteSystem, energy: float,
                onsite = np.identity(2), inject = 0, mode = 0, name: str = "_") -> None:
    
    psi = kwant.wave_function(system, energy)(inject)
    density_operator = kwant.operator.Density(system, onsite)
    density = density_operator(psi[mode])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(r'$|\psi|^2$', fontsize=10)

    rysunek = kwant.plotter.density(syst=system, density=density, ax=ax, show=False, cmap = 'Greys')
    formatter = ScalarFormatter(useMathText=True)
    cbar0 = fig.colorbar(rysunek, ax=ax, cmap = 'Grays')
    cbar0.formatter = formatter
    cbar0.update_ticks()
    cbar0.ax.tick_params(labelsize=10)
    cbar0.ax.yaxis.offsetText.set_fontsize(10)
    cbar0.formatter.set_powerlimits((0, 0))
    cbar0.mappable.set_clim(vmin=np.min(density), vmax=np.max(density))

    plt.savefig(name + ".pdf")
    plt.show()

def conductance(system: kwant.builder.FiniteSystem | kwant.builder.InfiniteSystem,
                energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    G = []
    for energy in energies:
        S = kwant.smatrix(system, energy, in_leads = (0,1), out_leads = (0,1))
        G.append(S.transmission(1,0))
    
    return (energies, np.asarray(G))

def conductance_spin(system: kwant.builder.FiniteSystem | kwant.builder.InfiniteSystem,
                energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    G_uu, G_ud, G_dd, G_du = [], [], [], []
    for energy in energies:
        S = kwant.smatrix(system, energy, in_leads = (0,1), out_leads = (0,1))
        G_uu.append(S.transmission((1, 0), (0, 0)))
        G_ud.append(S.transmission((1, 0), (0, 1)))
        G_dd.append(S.transmission((1, 1), (0, 1)))
        G_du.append(S.transmission((1, 1), (0, 0)))
    
    return (energies, np.asarray(G_uu), np.asarray(G_ud), np.asarray(G_dd), np.asarray(G_du))


def plot_conductance(conductance: Tuple[np.ndarray, np.ndarray],
                    name: str = "_") -> None:

    es, G = conductance
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(es/eV2au(1.0), G, color = 'k')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True)
    ax.set_xlabel(r'$E$ (eV)')
    ax.set_ylabel(r'$G$ (2e$^2$/h)')
    fig.tight_layout()
    plt.savefig(name + ".pdf")
    plt.show()

def plot_B(B: Callable, pm: dict) -> None:

    x_vals = np.arange(0, pm['L'] + pm['dx'], pm['dx'])
    y_vals = np.arange(-pm['W']/2, pm['W']/2 + pm['dx'], pm['dx'])
    X, Y = np.meshgrid(x_vals, y_vals)

    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            bx, by, bz = B(X[i, j], Y[i, j])
            Bx[i, j] = bx/T2au(1.0)
            By[i, j] = by/T2au(1.0)
            Bz[i, j] = bz/T2au(1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(Bx, extent=[0, pm['L'], -pm['W']/2, pm['W']/2], aspect='auto', origin='lower')
    axes[0].set_title('$B_x$')
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(By, extent=[0, pm['L'], -pm['W']/2, pm['W']/2], aspect='auto', origin='lower')
    axes[1].set_title('$B_y$')
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(Bz, extent=[0, pm['L'], -pm['W']/2, pm['W']/2], aspect='auto', origin='lower')
    axes[2].set_title('$B_z$')
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.tight_layout()
    plt.show()

def makesystem(pm: dict, B: Callable,
            soc_map: Callable = lambda x_i, x_j : 1.0) -> kwant.builder.FiniteSystem | kwant.builder.InfiniteSystem:

    t = .5/pm['m_eff']/pm['dx']**2
    t_SO = 0.5*pm['alpha']/pm['dx']

    def onsite(site) -> float:

        x, y = site.pos
        B_x, B_y, B_z = B(x,y)

        return 4*t*s_0  + 0.5*pm['mu_b']*pm['g_factor']*(B_x*s_x + B_y*s_y + B_z*s_z)
    
    def hopping(site_i, site_j) -> float:

        x_i, y_i = site_i.pos
        x_j, y_j = site_j.pos

        so_i = 1j*t_SO*s_y*np.sign(y_i - y_j)
        so_j = -1j*t_SO*s_x*np.sign(x_i - x_j)

        return -t*s_0 + (so_i + so_j)*soc_map(x_i, x_j)
    
    def hopping_lead(site_i, site_j) -> float:

        return -t*s_0
    
    lat = kwant.lattice.square(pm['dx'], norbs = 2)

    def shape(pos: tuple) -> bool:
        (x,y) = pos

        return (0 <= x <= pm['L'] and -pm['W']/2 <= y <= pm['W']/2)
    
    def lead_shape(pos: tuple) -> bool:
        (x,y) = pos

        return (-pm['W']/2 <= y <= pm['W']/2)

    system = kwant.Builder()
    system[lat.shape(shape, (0, 0))] = onsite
    system[(lat.neighbors())] = hopping
    
    lead_left = kwant.Builder(kwant.TranslationalSymmetry((-pm['dx'], 0)),
                        conservation_law = np.array([[1,0],[0,2]]))
    lead_left[lat.shape(lead_shape, (0, 0))] = onsite
    lead_left[(lat.neighbors())] =  hopping
    system.attach_lead(lead_left)

    lead_right = kwant.Builder(kwant.TranslationalSymmetry((pm['dx'], 0)),
                        conservation_law = np.array([[1,0],[0,2]]))
    lead_right[lat.shape(lead_shape, (pm['L'], 0))] = onsite
    lead_right[(lat.neighbors())] =  hopping
    system.attach_lead(lead_right)

    system = system.finalized()

    return system

pm = {

    'dx': nm2au(4.0), # lattice constant

    'L': nm2au(2000.0), # system length
    'W': nm2au(100.0), # system width
    'm_eff': 0.014, # effective mass

    'alpha' : 0.0, # Rashba SOC constant
    'g_factor' : -50, # Lande g-factor
    'mu_b' : 0.5 # Bohr magneton

}

# B = 0
sys = makesystem(pm, lambda x, y: (0, 0, 0))
disp = disperssion(sys, pm, nr_lead = 0, k_max = 0.15/nm2au(1.0), n_k_steps = 50)
plot_disperssion(disp, y_lim=[0, 0.1], name = "disperssion_B0")

# pole B w kierunku x
sys = makesystem(pm, lambda x, y: (T2au(1.0), 0, 0))
disp = disperssion(sys, pm, nr_lead = 0, k_max = 0.15/nm2au(1.0), n_k_steps = 50)
plot_disperssion(disp, y_lim=[0, 0.1], name = "disperssion_Bx")

# pole B w kierunku y
sys = makesystem(pm, lambda x, y: (0, T2au(1.0), 0))
disp = disperssion(sys, pm, nr_lead = 0, k_max = 0.15/nm2au(1.0), n_k_steps = 50)
plot_disperssion(disp, y_lim=[0, 0.1], name = "disperssion_By")

# pole B w kierunku z
sys = makesystem(pm, lambda x, y: (0, 0, T2au(1.0)))
disp = disperssion(sys, pm, nr_lead = 0, k_max = 0.15/nm2au(1.0), n_k_steps = 50)
plot_disperssion(disp, y_lim=[0, 0.1], name = "disperssion_Bz")

energies = np.linspace(0, eV2au(50.0e-3), 40)
cond = conductance(sys, energies)
plot_conductance(cond, name = "conductance_Bz")

def make_B(B_y: float):

    def B(x,y):
        
        if (0.2*pm['L'] < x  < 0.8*pm['L']):
            return (0, B_y, T2au(.1))
        else:
            return (0, 0, T2au(.1))
    
    return B

linsp_B_y = np.linspace(0, T2au(1.0), 100)
G_uu, G_ud, G_dd, G_du = [], [], [], []
for B_y in linsp_B_y:

    sys = makesystem(pm, make_B(B_y))
    es, g_uu, g_ud, g_dd, g_du = conductance_spin(sys, np.array([eV2au(5.0e-3)]))
    
    G_uu.append(g_uu[0])
    G_ud.append(g_ud[0])
    G_dd.append(g_dd[0])
    G_du.append(g_du[0])


fig, axs = plt.subplots(1, 2, figsize = (8, 4))
axs[0].plot(linsp_B_y/T2au(1.0), G_uu, label = r"$G_{|\uparrow \rangle \leftarrow |\uparrow \rangle}$")
axs[0].plot(linsp_B_y/T2au(1.0), G_du, label = r"$G_{|\downarrow \rangle \leftarrow |\uparrow \rangle}$")
axs[0].grid(True)
axs[0].set_xlabel(r'$B$ (T)')
axs[0].legend(frameon=False)
axs[0].set_ylabel(r'$G_{x \leftarrow |\uparrow \rangle}$ (2e$^2$/h)')

axs[1].plot(linsp_B_y/T2au(1.0), G_dd, label = r"$G_{|\downarrow \rangle \leftarrow |\downarrow \rangle}$")
axs[1].plot(linsp_B_y/T2au(1.0), G_ud, label = r"$G_{|\uparrow \rangle \leftarrow |\downarrow \rangle}$")
axs[1].grid(True)
axs[1].set_xlabel(r'$B$ (T)')
axs[1].legend(frameon=False)
axs[1].set_ylabel(r'$G_{x \leftarrow |\downarrow \rangle}$ (2e$^2$/h)')

fig.tight_layout()
plt.savefig("conductance_B_5meV.pdf")
plt.show()

B = make_B(T2au(.6))
sys = makesystem(pm, B)

wavefunction(sys, eV2au(5.0e-3), onsite = d_up, inject = 0, mode = 0, name = "psi_spin_up")
wavefunction(sys, eV2au(5.0e-3), onsite = d_dn, inject = 0, mode = 0, name = "psi_spin_dn")
wavefunction(sys, eV2au(5.0e-3), onsite = s_x, inject = 0, mode = 0, name = "psi_spin_x")
wavefunction(sys, eV2au(5.0e-3), onsite = s_y, inject = 0, mode = 0, name = "psi_spin_y")
wavefunction(sys, eV2au(5.0e-3), onsite = s_z, inject = 0, mode = 0, name = "psi_spin_z")

##############################################################
################### ferromagnetyczne paski ###################
##############################################################

def local_dict(dictionary: dict, key: str, value: float) -> dict:

    new_dict = deepcopy(dictionary)
    new_dict[key] = value
    return new_dict

def make_B_helikalne(pm: dict):
    
    def B_helikalne(x,y):

        B_x = pm['B_h']*np.sin(2*np.pi*(x - pm['x_0'])/pm['a'])
        B_y = 0
        B_z = pm['B_h']*np.cos(2*np.pi*(x - pm['x_0'])/pm['a']) + pm['B_ext']

        return (B_x, B_y, B_z)
    
    return B_helikalne

def conductance_B(pm: dict, energy: float, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    G = []
    for b in B:
        pm_local = local_dict(pm, 'B_ext', b)
        
        B_helikalne = make_B_helikalne(pm_local)
        sys = makesystem(pm, B_helikalne)

        S = kwant.smatrix(makesystem(pm_local, B_helikalne),
                        energy, in_leads = (0,1), out_leads = (0,1))
        G.append(S.transmission(0,1))
    
    return (B, np.asarray(G))

def plot_conductance_B(conductance_B: Tuple[np.ndarray, np.ndarray],
                    name: str = "_") -> None:

    B, G = conductance_B
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(-B/T2au(1.0), G, color = 'k')
    ax.grid(True)
    ax.set_xlabel(r'$B_{ext}$ (T)')
    ax.set_ylabel(r'$G$ (2e$^2$/h)')

    fig.tight_layout()
    plt.savefig(name + ".pdf")
    plt.show()

pm = {

    'dx': nm2au(1.0), # lattice constant

    'L': nm2au(1000.0), # system length
    'W': nm2au(30.0), # system width
    'm_eff': 0.1, # effective mass

    'B_h': T2au(50.0e-3), # helical B field magnitude
    'x_0': nm2au(1000.0)/2, # helical B field parameter
    'a' : nm2au(1000.0), # helical B field parameter
    'B_ext': T2au(0.0), # external B field in "z" direction

    'alpha' : 0.0, # Rashba SOC constant
    'g_factor' : 200, # Lande g-factor
    'mu_b' : 0.5 # Bohr magneton

}

B_helikalne = make_B_helikalne(pm)
plot_B(B_helikalne, pm)

fig, ax = plt.subplots(figsize = (5,4))

linsp = np.linspace(0, pm['L'], 100)

Bx, By, Bz = B_helikalne(linsp, 0)

ax.plot(linsp/nm2au(1.0), Bx/T2au(1.0), label = r'B$_x$', color = 'r')
ax.plot(linsp/nm2au(1.0), np.zeros_like(linsp), label = r'B$_y$', color = 'b')
ax.plot(linsp/nm2au(1.0), Bz/T2au(1.0), label = r'B$_z$', color = 'k')
ax.plot(linsp/nm2au(1.0), - 50e-3 + Bz/T2au(1.0), '--', alpha = 0.5, label = r'B$_z$ + B$_{\text{ext, mid}}$', color = 'k')
ax.plot(linsp/nm2au(1.0), - 100e-3 + Bz/T2au(1.0), '--', alpha = 0.25, label = r'B$_z$ + B$_{\text{ext, max}}$', color = 'k')
ax.set_xlabel('x (nm)')
ax.set_ylabel('B (T)')
ax.legend(frameon = False)
plt.tight_layout()
plt.savefig("B_xyz.pdf")
plt.show()

sys = makesystem(pm, B_helikalne)
disp = disperssion(sys, pm, 0, 0.4/nm2au(1.0), 50)
plot_disperssion(disp, y_lim = [0, 40.0e-3], name = '2zadanie_dyspersja')

disp = disperssion(sys, pm, 0, 0.1/nm2au(1.0), 50)
plot_disperssion(disp, y_lim = [3.0e-3, 5.0e-3], name = '2zadanie_dyspersja_blizej')

linsp_B = np.linspace(0, -T2au(100e-3), 50)
#cond = conductance_B(pm, energy = eV2au(3.75e-3), B = linsp_B)
plot_conductance_B(cond, "G_B")

##############################################################
# ----------- ODDZIALYWANIE SPIN-ORBITA
##############################################################

pm = {

    'dx': nm2au(4.0), # lattice constant

    'L': nm2au(800.0), # system length
    'W': nm2au(100.0), # system width
    'm_eff': 0.014, # effective mass

    'alpha' : 50.0*eV2au(1.0e-3)*nm2au(1.0), # Rashba SOC constant
    'g_factor' : -50, # Lande g-factor
    'mu_b' : 0.5 # Bohr magneton

}

def B(x,y):
    return (0, 0, 0)

sys = makesystem(pm, B)
disp = disperssion(sys, pm, 0, 0.13/nm2au(1.0), 100)
plot_disperssion(disp, y_lim = [0, 30e-3], name = "disperssion_Rashba")

es = np.linspace(0, eV2au(50.0e-3), 60)

cond = conductance(sys, es)
plot_conductance(cond, "conductance_SOC")

G_uu, G_ud, G_dd, G_du = [], [], [], []
linsp_alpha = np.linspace(0, 50.0*nm2au(1.0)*eV2au(1.0e-3), 30)
for alpha in linsp_alpha:
    pm_local = local_dict(pm, 'alpha', alpha)
    sys = makesystem(pm_local, B, lambda x_i, x_j : 0.2*pm['L'] < x_i < 0.8*pm['L'] and 0.2*pm['L'] < x_j < 0.8*pm['L'])
    es, g_uu, g_ud, g_dd, g_du = conductance_spin(sys, np.array([eV2au(5.0e-3)]))
    G_uu.append(g_uu[0])
    G_ud.append(g_ud[0])
    G_du.append(g_du[0])
    G_dd.append(g_dd[0])

linsp_alpha /= nm2au(1.0)*eV2au(1.0e-3)
fig, ax = plt.subplots(1, 2, figsize = (8, 4))
ax[0].plot(linsp_alpha, G_uu, label = r'T$_{uu}$', color = 'r')
ax[0].plot(linsp_alpha, G_du, label = r'T$_{du}$', color = 'b')
ax[1].plot(linsp_alpha, G_ud, label = r'T$_{ud}$', color = 'r')
ax[1].plot(linsp_alpha, G_dd, label = r'T$_{dd}$', color = 'b')
fig.tight_layout()
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)
ax[0].set_ylabel('T')
ax[0].grid()
ax[1].grid()
ax[0].set_xlabel(r'$\alpha$ (meVnm)')
ax[1].set_xlabel(r'$\alpha$ (meVnm)')
plt.savefig("transmission_spinorbit.pdf")
plt.show()

fig, ax = plt.subplots(1, 3, figsize = (12, 4))
for idx, p in enumerate([0.2, 0.4, 1]):
    G_u = 0.5*((1 + p)*np.array(G_uu) + (1 - p)*np.array(G_ud))
    G_d = 0.5*((1 + p)*np.array(G_du) + (1 - p)*np.array(G_dd))
    G_sum = 0.5*((1+p)*G_u + (1-p)*G_d)
    ax[idx].plot(linsp_alpha, G_u, label = r'G$_u$', color = 'r')
    ax[idx].plot(linsp_alpha, G_d, label = r'G$_d$', color = 'b')
    ax[idx].plot(linsp_alpha, G_sum, label = r'G$_{suma}$', color = 'k')
    ax[idx].set_title(f'P = {p}')
    ax[idx].legend(frameon=False)
    ax[idx].grid()
    ax[idx].set_xlabel(r'$\alpha$ (meVnm)')

ax[0].set_ylabel(r'G (e$^2$/$\hbar$)')
plt.savefig("conductance_spinorbit.pdf")
plt.show()

pm_local = local_dict(pm, 'alpha', 18*nm2au(1.0)*eV2au(1.0e-3))
sys = makesystem(pm_local, B, lambda x_i, x_j : 0.2*pm['L'] < x_i < 0.8*pm['L'] and 0.2*pm['L'] < x_j < 0.8*pm['L'])
wavefunction(sys, eV2au(5.0e-3), onsite = d_up, inject = 0, mode = 0, name = "zmiana_spinu_d_up")
wavefunction(sys, eV2au(5.0e-3), onsite = d_dn, inject = 0, mode = 0, name = "zmiana_spinu_d_dn")
wavefunction(sys, eV2au(5.0e-3), onsite = s_x, inject = 0, mode = 0, name = "zmiana_spinu_s_x")
wavefunction(sys, eV2au(5.0e-3), onsite = s_y, inject = 0, mode = 0, name = "zmiana_spinu_s_y")
wavefunction(sys, eV2au(5.0e-3), onsite = s_z, inject = 0, mode = 0, name = "zmiana_spinu_s_z")

pm_local = local_dict(pm, 'alpha', 37.5*nm2au(1.0)*eV2au(1.0e-3))
sys = makesystem(pm_local, B, lambda x_i, x_j : 0.2*pm['L'] < x_i < 0.8*pm['L'] and 0.2*pm['L'] < x_j < 0.8*pm['L'])
wavefunction(sys, eV2au(5.0e-3), onsite = d_up, inject = 0, mode = 0, name = "nie_zmiana_spinu_d_up")
wavefunction(sys, eV2au(5.0e-3), onsite = d_dn, inject = 0, mode = 0, name = "nie_zmiana_spinu_d_dn")
wavefunction(sys, eV2au(5.0e-3), onsite = s_x, inject = 0, mode = 0, name = "nie_zmiana_spinu_s_x")
wavefunction(sys, eV2au(5.0e-3), onsite = s_y, inject = 0, mode = 0, name = "nie_zmiana_spinu_s_y")
wavefunction(sys, eV2au(5.0e-3), onsite = s_z, inject = 0, mode = 0, name = "nie_zmiana_spinu_s_z")