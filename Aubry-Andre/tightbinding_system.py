import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, Callable
from scipy.sparse import csc_matrix
from scipy.optimize import curve_fit
from scipy.sparse.linalg import eigsh

def makesystem(pm: dict, plot_onsite: bool = False, name: str | None = None):

    fun = pm["periodic_function"]
    N = pm["Theta_denominator"]
    k = pm["k"]
    Theta = pm["Theta_numerator"]/pm["Theta_denominator"]

    hamiltonian = np.zeros((N, N), dtype = complex)
    for n in range(N):

        hamiltonian[n, n] = 2.0*pm["V2"]*fun(2.0*np.pi*n*Theta + pm['nu'])
        
        if n > 0:
            hamiltonian[n, n-1] = pm["V1"]
        if n < N - 1:
            hamiltonian[n, n+1] = pm["V1"]

    hamiltonian[0, -1] = pm["V1"]*np.exp(-1j*k)
    hamiltonian[-1, 0] = pm["V1"]*np.exp(1j*k)

    if plot_onsite:
        fig, ax = plt.subplots(1,1,figsize=(4,2))
        lx = np.linspace(0, pm["Theta_denominator"], num = 10000, endpoint=True)
        onsite = 2.0*pm["V2"]*fun(2.0*np.pi*lx*Theta + pm['nu'])
        ax.plot(np.arange(0, N, step=1), np.real(np.diag(hamiltonian)), 'o', color= 'k', markersize = 0.1)
        ax.plot(lx, onsite, linestyle='dotted', color='r', linewidth=0.1)
        ax.set_ylabel(r"$\mathcal{O}_n$")
        ax.set_xlabel("n")
        #fig.suptitle(rf"$\vartheta$ = p/q = {pm["Theta_numerator"]}/{pm["Theta_denominator"]}, $\nu$ = {pm['nu']}")
        fig.tight_layout()
        if name != None:
            plt.savefig(name + ".pdf")
        plt.show()

    return hamiltonian

def makesystem_Biddle(pm: dict, plot_onsite: bool = False):

    fun = pm["periodic_function"]
    if not "alpha" in pm:
        Theta = pm["Theta_numerator"]/pm["Theta_denominator"]
    else:
        Theta = pm["alpha"]
    N = pm["Theta_denominator"]
    k = pm["k"]
    if not "p" in pm:
        pm["p"] = 0

    hamiltonian = np.zeros((N, N), dtype = complex)
    for n in range(N):
        for n_prime in range(N):
            if n == n_prime:
                hamiltonian[n, n] = pm["V2"]*fun(2.0*np.pi*n*Theta + pm['nu'])
            else:
                distance = np.min([np.abs(n_prime - n), N - np.abs(n_prime - n)]) # np.abs(n_prime - n) 
                hamiltonian[n, n_prime] = pm["V1"]*np.exp(-pm["p"]*distance)

    hamiltonian[0, -1] = pm["V1"]*np.exp(-pm["p"]*1)*np.exp(-1j*k)
    hamiltonian[-1, 0] = pm["V1"]*np.exp(-pm["p"]*1)*np.exp(1j*k)

    if plot_onsite:
        fig, ax = plt.subplots(1,1,figsize=(4,2))
        ax.plot(np.real(np.diag(hamiltonian)), 'o', color= 'k', markersize = 0.1)
        ax.set_ylabel("V")
        ax.set_xlabel("n")
        fig.tight_layout()
        plt.show()

    return hamiltonian

def spectrum(pm: dict, makesystem: Callable[[dict, bool], np.ndarray],
            E_range: Tuple[float, float] | None = None,
            plot: bool = False, step: float = 1,
            name: str | None = None) -> Tuple[np.ndarray, np.ndarray]:

    '''
    returns a tuple of (theta_range, array of energies for each theta)
    '''

    dof = pm.get("degrees_of_freedom", 1)
    Theta_denominator = pm["Theta_denominator"]
    Theta_numerators = np.arange(start = 0, stop = Theta_denominator/dof + step, step = step)
    energies = np.zeros((len(Theta_numerators), Theta_denominator*dof))
    thetas = np.zeros(len(Theta_numerators))
    
    for i, Theta_numerator in enumerate(Theta_numerators):

        Theta = Theta_numerator/Theta_denominator
        pm["Theta_numerator"] = Theta_numerator

        h = makesystem(pm, False)
        E = eigh(h, eigvals_only=True, subset_by_value=E_range)
    
        thetas[i] = Theta
        energies[i] = E

    if plot:
        fig, ax = plt.subplots(1,1, figsize= (5,5))
        for th, en in zip(thetas, energies):
            ax.plot(np.ones_like(en)*th, en, ',', markersize = 0.01, color = 'k')
        #ax.set_yticks([])
        #ax.set_xticks([])
        ax.set_xlabel(r"$\vartheta$ = p/q")
        ax.set_ylabel(r"E")
        ax.set_xlim((thetas[0], thetas[-1])) #type:ignore
        ax.set_ylim((np.min(energies), np.max(energies))) #type:ignore
        plt.savefig(name + ".png" if name != None else "wynik.png", dpi = 500)
        plt.show()
    
    return thetas, energies

def CrankNicolson(psi_0: np.ndarray, hamiltonian: np.ndarray, dt: float, n_t: int):

    I = np.eye(len(psi_0))
    # B*psi(t+1) = A*psi(t) => psi(t+1) = B^{-1}*A*psi(t)
    A = I - 0.5j*dt*hamiltonian
    B = I + 0.5j*dt*hamiltonian
    cn_operator = np.linalg.inv(B) @ A
    
    t = np.zeros(n_t)
    psi_t = np.zeros((n_t, len(psi_0)), dtype = complex)
    psi_t[0] = psi_0
    for t_step in range(1, n_t):
        psi_t[t_step, :] = cn_operator @ psi_t[t_step - 1, :]
        t[t_step] = dt*t_step

    return t, psi_t



######################################################
########## MULTIFRAKTALNY WYMIAR UOGOLNIONY ##########
######################################################

def md(x, q, power_of_2point3_minmax = (4, 10),
                        nsteps = 20, plot = False,
                        histrange: Tuple[float, float] | None = None) -> Tuple[float, float]:

    if histrange == None:
        histrange = (x.min(), x.max())

    epsilon = np.array([1.5**(-n) for n in np.linspace(power_of_2point3_minmax[0],
                                                        power_of_2point3_minmax[1],
                                                        nsteps)])
    n_bins_list = np.unique([int((histrange[1] - histrange[0])/e) for e in epsilon])

    N = len(x)
    histograms = [np.histogram(x, bins=n_bins, range=histrange) for n_bins in n_bins_list]
    mu = [H[H > 0]/N for H, _ in histograms]
    Z = np.array([np.sum(np.array(mu_i)**q) for mu_i in mu])

    x = np.log10(epsilon)
    y = np.log10(Z)
    popt, pcov = curve_fit(lambda vals, a, const: a*vals + const, x, y)

    tau = popt[0]
    const = popt[1]
    error = np.sqrt(np.diag(pcov))[0]

    if plot:
        fig, ax = plt.subplots(1,1,figsize = (2,1))
        ax.set_axis_off()
        ax.plot(x, y, '-', color = 'k')
        ax.plot(x, tau*x + const, '--', color = 'r')
        plt.show()

    return tau, error

def multifractal_dimension(thetas, energies, q: int = 1,
                        power_of_2point3_minmax = (4, 10), nsteps = 20, plot: bool = False, errorbar: bool = True,
                        histrange: Tuple[float, float] | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    wymiar korelacyjny w funkcji theta
    """
    dimension_mg = np.zeros(len(thetas))
    dimension_mg_error = np.zeros(len(thetas))

    for i, energies_th in enumerate(energies):

        dimension_mg[i], dimension_mg_error[i] = md(energies_th, q,
                                                    power_of_2point3_minmax,
                                                    nsteps, False, histrange)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        if errorbar:
            ax.plot(thetas, dimension_mg, 'o-', color='k', linewidth=1, markersize = 0.4)
            ax.errorbar(thetas, dimension_mg, yerr = dimension_mg_error,
                        fmt='o', color = 'r', markersize = 0.5, elinewidth=0.5)
        else:
            ax.plot(thetas, dimension_mg, '-', color='k', linewidth=1)
        ax.set_xlabel(r"$\Theta = p/q$")
        ax.set_ylabel(rf"D$_{{{q}}}$")
        ax.set_ylim(dimension_mg.min(),  1.1*dimension_mg.max())
        ax.set_xlim(thetas.min(), thetas.max())
        ax.grid(True, color = 'g', linestyle = '--', alpha = 0.4)
        plt.show()

    return thetas, dimension_mg, dimension_mg_error

######################################################
#### WSPOLCZYNNIK GAMMA TOULESSA - cos nie dziala ####
######################################################

def bands(pm: dict, n_k: int = 100, n_e: int | None = None, E_range: Tuple[float, float] | None = None, plot: bool = False):
    
    """
    relacja dyspersji E(k).
    """
    
    pm_local = deepcopy(pm)
    brillouin_zone = (-np.pi, np.pi)
    k_values = np.linspace(*brillouin_zone, num = n_k)
    N, V1, V2 = pm["Theta_denominator"], pm["V1"], pm["V2"]
    energies_k = []
    
    for k in k_values:
        
        pm_local["k"] = k
        h = makesystem(pm_local, False)
        if n_e == None:
            E = eigh(h, eigvals_only=True)
        else:
            h_sparse = csc_matrix(h)
            E = eigsh(h_sparse, k=n_e, sigma=0, which='LM', return_eigenvectors=False)
            E = np.sort(E)
        energies_k.append(E)

    energies_k = np.array(energies_k)
    
    if plot:
        plt.figure(figsize=(6, 5))
        for band_idx in range(N):
            plt.plot(k_values, energies_k[:, band_idx], color='k', linewidth = 0.5)
        plt.xlabel(r"Wave vector $k$")
        plt.ylabel(r"Energy $E$")
        plt.title(f"Dispersion E(k) for p={N}, V1={V1}, V2={V2}")
        plt.xlim(*brillouin_zone)
        if E_range != None:
            plt.ylim(*E_range) # type: ignore
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return k_values, energies_k

def gamma(pm: dict, energies_k: np.ndarray, energies_to_calculate_for: np.ndarray | None = None, n_bins: int | None = None):

    # D J Thouless 1972 J. Phys. C: Solid State Phys. 5 77

    def dos(pm: dict, energies_k: np.ndarray, n_bins: int | None = None):

        energies = energies_k.flatten()
        _dos, _energies = np.histogram(energies, n_bins if n_bins != None else pm["Theta_denominator"])

        return (_energies[1:] + _energies[:-1])/2, _dos/len(energies)

    _energies, _dos = dos(pm, energies_k, n_bins)
    nonzero = _dos != 0
    _energies = _energies[nonzero]
    _dos = _dos[nonzero]

    def _gamma(E):
        return np.sum(np.log(np.abs(E - _energies))*_dos) - np.log(pm["V1"]) # dE/dE

    if energies_to_calculate_for == None:
        energies_to_calculate_for = np.linspace(energies_k.min(), energies_k.max(), 100)

    return np.array([_gamma(e) for e in energies_to_calculate_for]) # type: ignore

def gamma_but_different(pm: dict, energies_k: np.ndarray, energies_to_calculate_for: np.ndarray | None = None):

    # D J Thouless 1972 J. Phys. C: Solid State Phys. 5 77 (eq. 5)

    N = len(energies_k)
    def _gamma(E):
        return np.sum(np.log(np.abs(energies_k - E)))/(N-1) - np.log(pm["V1"])

    return np.array([_gamma(e) for e in energies_to_calculate_for]) # type: ignore