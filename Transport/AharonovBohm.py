# symulacje efektu Aharonova-Bohma w nanopierscieniu

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

def local_dict(dictionary: dict, key: str, value: float) -> dict:

    new_dict = deepcopy(dictionary)
    new_dict[key] = value
    return new_dict

def makesystem_square(pm: dict) -> kwant.builder.FiniteSystem:

    '''
    provide system parameters as a dictionary:

    'dx' - lattice constant,
    'L' - system length,
    'W' - system width,
    'm_eff' - effective mass,

    'V_0' - electric potential bias
    'x_0' - positioning of electric potential gaussian in the "x" direction
    'y_0' - positioning of electric potential gaussian in the "y" direction

    'B_z" - magnetic field in the "z" direction
    '''

    t = .5/pm['m_eff']/pm['dx']**2

    def V(site) -> float:
        x, y = site.pos

        return pm['V_0']*np.exp(( 
                                -(x - pm['x_0'])**2 - (y - pm['y_0'])**2
                                )/pm['sigma']**2
                                )

    def onsite(site) -> float:

        x, y = site.pos

        return 4*t + V(site)
    
    def hopping(site1, site2) -> float:

        x0, y0 = site1.pos
        x1, y1 = site2.pos

        return -t*np.exp(-0.5j*pm['B_z']*(y1+y0)*(x1-x0))
    
    lat = kwant.lattice.square(pm['dx'], norbs = 1)

    def shape(pos: tuple) -> bool:
        (x,y) = pos

        return (0 <= x <= pm['L'] and -pm['W']/2 <= y <= pm['W']/2)
    
    def lead_shape(pos: tuple) -> bool:
        (x,y) = pos

        return (-pm['W']/2 <= y <= pm['W']/2)

    system = kwant.Builder()
    system[lat.shape(shape, (0, 0))] = onsite
    system[(kwant.builder.HoppingKind((1,0), lat, lat))] = hopping
    system[(kwant.builder.HoppingKind((0,1), lat, lat))] = hopping
    
    lead = kwant.Builder(kwant.TranslationalSymmetry((-pm['dx'], 0)))
    lead[lat.shape(lead_shape, (0, 0))] = onsite
    lead[(kwant.builder.HoppingKind((1,0), lat, lat))] = hopping
    lead[(kwant.builder.HoppingKind((0,1), lat, lat))] = hopping

    system.attach_lead(lead)
    system.attach_lead(lead.reversed())

    system = system.finalized()

    return system

def makesystem_AB(pm: dict) -> kwant.builder.FiniteSystem:

    '''
    provide system parameters as a dictionary:

    'dx' - lattice constant,
    'm_eff' - effective mass,
    'L' - system length,
    'W' - system width,
    'r' - inner ring radius,
    'R' - outer ring radius,

    'B_z" - magnetic field in the "z" direction
    '''

    t = .5/pm['m_eff']/pm['dx']**2

    def onsite(site) -> float:

        x, y = site.pos

        return 4*t
    
    def hopping(site1, site2) -> float:

        x0, y0 = site1.pos
        x1, y1 = site2.pos

        return -t*cmath.exp(-0.5j*pm['B_z']*(y1 + y0)*(x1 - x0))

    def hopping_lead(site1, site2) -> float:

        return -t
    
    lat = kwant.lattice.square(pm['dx'], norbs = 1)

    def ring(pos):
        (x,y) = pos
        
        r2 = x**2 + y**2
        inside_ring  = pm['r']**2 <= r2 <= pm['R']**2
        inside_pre_leads = (r2 > pm['R']**2 and abs(x) <= pm['L']/2 and abs(y) <= pm['W']/2)

        return inside_ring or inside_pre_leads
    
    def lead_shape(pos):
        (x,y) = pos
        
        return abs(y) <= pm['W']/2

    system = kwant.Builder()

    system[lat.shape(ring, (-(pm['r']+1), 0))] = onsite
    system[(kwant.builder.HoppingKind((-1,0), lat, lat))] = hopping
    system[(kwant.builder.HoppingKind((0,-1), lat, lat))] = hopping

    l_lead = kwant.Builder(kwant.TranslationalSymmetry((-pm['dx'], 0)))
    l_lead[lat.shape(lead_shape, (0, 0))] = onsite
    l_lead[lat.neighbors()] = hopping_lead

    #l_lead[(kwant.builder.HoppingKind((1,0), lat, lat))] = hopping # hopping_lead
    #l_lead[(kwant.builder.HoppingKind((0,1), lat, lat))] = hopping # hopping_lead

    r_lead = kwant.Builder(kwant.TranslationalSymmetry((pm['dx'],0)))
    r_lead[lat.shape(lead_shape,(0,0))] = onsite
    r_lead[lat.neighbors()] = hopping_lead

    system.attach_lead(l_lead)
    system.attach_lead(r_lead)

    #system.attach_lead(lead)
    #system.attach_lead(lead.reversed())

    system.eradicate_dangling()
    system = system.finalized()

    return system

def disperssion(system: kwant.builder.FiniteSystem,\
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

def wavefunction_current(system: kwant.builder.FiniteSystem, energy: float,
                        inject = 0, name: str = "_") -> None:

    def sum_over_modes(operator: kwant.operator.Density | kwant.operator.Current,\
                        psi: np.ndarray) -> np.ndarray:
        
        val_total = operator(psi[0])
        for mode in range(1,len(psi)):
            val_mode = psi[mode]
            val_total += operator(val_mode)
        
        return val_total
    
    psi = kwant.wave_function(sys = system, energy = energy)(inject)
    density = sum_over_modes(kwant.operator.Density(system), psi)
    current = sum_over_modes(kwant.operator.Current(system), psi)

    kgs = {
        'fig_size': (4,4),
        'relwidth': 0.05,
        'show': False,
        'colorbar': False,
        'syst': system
        }
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].set_title(r'$|\psi|^2$', fontsize=10)
    axes[1].set_title(r'$J$', fontsize=10)

    density_plot = kwant.plotter.density(density = density, ax = axes[0], cmap = 'Grays', **kgs)
    current_plot = kwant.plotter.current(current = current, ax = axes[1], cmap = 'Reds', **kgs)

    formatter = ScalarFormatter(useMathText=True)
    
    cbar0 = fig.colorbar(density_plot, ax=axes[0], cmap = 'Grays')
    cbar0.formatter = formatter
    cbar0.update_ticks()
    cbar0.ax.tick_params(labelsize=10)
    cbar0.ax.yaxis.offsetText.set_fontsize(10)
    cbar0.formatter.set_powerlimits((0, 0))
    cbar0.mappable.set_clim(vmin=np.min(density), vmax=np.max(density))

    cbar1 = fig.colorbar(current_plot, ax=axes[1], cmap = 'Reds')
    cbar1.formatter = formatter
    cbar1.update_ticks()
    cbar1.ax.tick_params(labelsize=10)
    cbar1.ax.yaxis.offsetText.set_fontsize(10)
    cbar1.formatter.set_powerlimits((0, 0))
    cbar1.mappable.set_clim(vmin=np.min(current), vmax=np.max(current))

    plt.savefig(name + ".pdf")
    plt.show()

def conductance(system: kwant.builder.FiniteSystem, energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    G = []
    for energy in energies:
        S = kwant.smatrix(system, energy, in_leads = (0,1), out_leads = (0,1))
        G.append(S.transmission(1,0))
    
    return (energies, np.asarray(G))

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

def conductance_B(pm: dict, energy: float, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    G = []
    for b in B:
        S = kwant.smatrix(makesystem_AB(local_dict(pm, 'B_z', b)),
                        energy, in_leads = (0,1), out_leads = (0,1))
        G.append(S.transmission(0,1))
    
    return (B, np.asarray(G))

def plot_conductance_B(conductance_B: Tuple[np.ndarray, np.ndarray],
                    name: str = "_") -> None:

    B, G = conductance_B
    fig, ax = plt.subplots(figsize = (4,4))
    ax.plot(B/T2au(1.0), G, color = 'k')
    ax.grid(True)
    ax.set_xlabel(r'$B$ (T)')
    ax.set_ylabel(r'$G$ (2e$^2$/h)')

    fig.tight_layout()
    plt.savefig(name + ".pdf")
    plt.show()

pm = {

    'dx': nm2au(1.0), # lattice constant

    'L': nm2au(80.0), # system length
    'W': nm2au(60.0), # system width
    'm_eff': 0.014, # effective mass

    'V_0': eV2au(50.0e-3), # electric potential bias
    'sigma': nm2au(10.0), # electric potential gaussian sigma parameter
    'x_0': nm2au(40.0), # positioning of electric potential gaussian in the "x" direction
    'y_0': nm2au(0.0), # positioning of electric potential gaussian in the "y" direction

    'B_z': T2au(0.0) # magnetic field in the "z" direction

}

system = makesystem_square(pm)
kwant.plot(system, site_color=lambda site: system.hamiltonian(site,site),
        fig_size=(10,5), colorbar=False, show=False, num_lead_cells=2); plt.show()

disp = disperssion(system, pm, nr_lead = 0, k_max = 0.3/nm2au(1.0), n_k_steps = 60)
plot_disperssion(disp, y_lim = [0, 0.16], name = "potential_disperssion")

cond = conductance(system, np.linspace(0, eV2au(150.0e-3), 40))
plot_conductance(cond, name = "potential_G")

wavefunction_current(system, energy = eV2au(30.0e-3), inject = 0, name = "V_30meV")
wavefunction_current(system, energy = eV2au(50.0e-3), inject = 0, name = "V_50meV")
wavefunction_current(system, energy = eV2au(100.0e-3), inject = 0, name = "V_100meV")

pm = {

    'dx': nm2au(2.0), # lattice constant

    'L': nm2au(300.0), # system length
    'W': nm2au(200.0), # system width
    'm_eff': 0.014, # effective mass

    'V_0': eV2au(50.0e-2), # electric potential bias
    'sigma': nm2au(10.0), # electric potential gaussian sigma parameter
    'x_0': nm2au(150.0), # positioning of electric potential gaussian in the "x" direction
    'y_0': nm2au(100.0), # positioning of electric potential gaussian in the "y" direction

    'B_z': T2au(2.0) # magnetic field in the "z" direction

}

pm['W'] = nm2au(80.0)
system = makesystem_square(pm)
dis = disperssion(system, pm, 0, 0.6/nm2au(1.0), 50) 
plot_disperssion(dis, y_lim = [0.0, 0.2], name = "disperssion_B_80nm")

pm['W'] = nm2au(200.0)
system = makesystem_square(pm)
dis = disperssion(system, pm, 0, 0.6/nm2au(1.0), 50) 
plot_disperssion(dis, y_lim = [0.0, 0.2], name = "disperssion_B_200nm")

#cond = conductance(system, np.linspace(0, eV2au(150.0e-3), 40))
plot_conductance(cond, name = "magnetic_G")

wavefunction_current(system, energy = eV2au(0.0125), inject = 0, name = "QHE_up")
wavefunction_current(system, energy = eV2au(0.0125), inject = 1, name = "QHE_down")

pm = {

    'dx': nm2au(2.0), # lattice constant
    'm_eff': 0.014, # effective mass
    'L': nm2au(2500.0), # system length
    'W': nm2au(30.0), # system width
    'R': nm2au(630.0), # outer ring radius
    'r': nm2au(600.0), # inner ring radius

    'B_z': T2au(2.0) # magnetic field in the "z" direction

}

system = makesystem_AB(pm)
disp = disperssion(system, pm, nr_lead = 0, k_max = 0.5/nm2au(1.0), n_k_steps = 60)
plot_disperssion(disp, y_lim = [0, 0.5], name = "disp")
print(np.min(disp[1])/eV2au(1))

kwant.plot(system); plt.show()

#wavefunction_current(system, energy = eV2au(0.2), inject = 0)

#cond = conductance(system, eV2au(np.linspace(0.0, 0.3, 15)))
#plot_conductance(cond)

#cond_B = conductance_B(pm, energy = eV2au(0.03), B = np.linspace(0, T2au(10e-3), 50))
#plot_conductance_B(cond_B, name = "G_B")

pm['B_z'] = T2au(0.004)
system = makesystem_AB(pm)
wavefunction_current(system, energy = eV2au(0.03), inject = 0, name = 'psi_Bmax')

pm['B_z'] = T2au(0.002)
system = makesystem_AB(pm)
wavefunction_current(system, energy = eV2au(0.03), inject = 0, name = 'psi_Bmin')