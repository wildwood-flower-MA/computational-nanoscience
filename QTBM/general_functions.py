from types import SimpleNamespace
import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt

def eV2au(eV: float):
    return 0.036749325871*eV

def nm2au(nm: float):
    return 18.897261339*nm

def K2au(K: float):
    return K/315775.13

def make_potential_QPC(pm: SimpleNamespace) -> Callable[[float, float], float]:

    ''' returns a callable defining a QPC electric
        potential at a given point in space (x, y),
        defined itself by the system parameters:
        V_gates, sigma_x and sigma_y '''

    def potential_QPC(x: float, y: float) -> float:

        exp1 = np.exp(-((x/pm.sigma_x)**2 + ((y - pm.y_min)/pm.sigma_y)**2)**2)
        exp2 = np.exp(-((x/pm.sigma_x)**2 + ((y - pm.y_max)/pm.sigma_y)**2)**2)

        return -0.035*pm.V_gates*(exp1 + exp2)
    
    return potential_QPC

def plot_potential_QPC(pm: SimpleNamespace, name: str = "_"):

    potential_QPC = make_potential_QPC(pm)

    dx = (pm.y_max - pm.y_min)/(pm.N_y + 1)
    y_values = np.array([pm.y_min + i*dx for i in range(1, pm.N_y + 1)])
    pm.N_x = int((pm.x_max - pm.x_min)/dx)
    x_values = np.array([pm.x_min + i*dx for i in range(1, pm.N_x + 1)])

    X, Y = np.meshgrid(x_values, y_values)
    Z = np.vectorize(potential_QPC)(X, Y)

    plt.figure(figsize=(5, 4))
    plt.imshow(Z/eV2au(1.0), extent=(pm.x_min/nm2au(1.0), pm.x_max/nm2au(1.0), 
                                    pm.y_min/nm2au(1.0), pm.y_max/nm2au(1.0)), 
                                    origin='lower', aspect='auto', cmap='winter_r')
    plt.colorbar(label='el. potential (eV)')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.tight_layout()
    if name != "_":
        plt.savefig(name + ".pdf")
    plt.show()

def plot_psi(pm: SimpleNamespace, psi: np.ndarray, name: str = "_") -> None:

    psipsi = np.abs(psi)**2
    plt.figure(figsize=(5, 4))
    plt.imshow(psipsi, extent=(pm.x_min/nm2au(1.0), pm.x_max/nm2au(1.0), 
                                pm.y_min/nm2au(1.0), pm.y_max/nm2au(1.0)), 
                                origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(label=r'$|\psi|^2$')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.tight_layout()
    if name != "_":
        plt.savefig(name + ".pdf")
    plt.show()

def plot_psi_sum(pm: SimpleNamespace, psi: list, name: str = "_") -> None:

    psipsi = np.abs(psi[0])**2
    for psi_n in psi[1:]:
        psipsi += np.abs(psi_n)**2
        
    plt.figure(figsize=(5, 4))
    plt.imshow(psipsi, extent=(pm.x_min/nm2au(1.0), pm.x_max/nm2au(1.0), 
                                pm.y_min/nm2au(1.0), pm.y_max/nm2au(1.0)), 
                                origin='lower', aspect='auto', cmap='inferno')
    plt.colorbar(label=r'$|\psi|^2$')
    plt.xlabel('x (nm)')
    plt.ylabel('y (nm)')
    plt.tight_layout()
    if name != "_":
        plt.savefig(name + ".pdf")
    plt.show()