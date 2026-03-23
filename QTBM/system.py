import numpy as np
from typing import Callable, Tuple
from types import SimpleNamespace
from general_functions import *

def makesystem_infinite(pm: SimpleNamespace, x_0 = None) -> Tuple[Callable, Callable]:

    '''
    creates two Callables:
    - returning self energies to wavevector
    - returning propagating modes to energy
    '''

    # (1) creates a function defining QPC potential
    # at a transverse cross-section at x = 0
    two_dimensional_potential_QPC = make_potential_QPC(pm)
    if x_0 == None: x_0 = pm.x_min
    potential_QPC = lambda y: two_dimensional_potential_QPC(x_0, y)

    # (2) defines the x = 0 transverse cross-section of the system
    dx = (pm.y_max - pm.y_min)/(pm.N_y + 1)
    linspace_y = np.array([pm.y_min + i*dx for i in range(1, pm.N_y + 1)])

    # (4) defines a discretization constants of the
    # time-independent Schroedinger equation
    alpha = 0.5/pm.m_eff/dx**2

    # (5) creates a hamiltonian matrix of the x = x_0
    # transverse cross-section of the system
    hamiltonian_y = np.zeros((pm.N_y, pm.N_y), dtype = complex)

    # (6) defines the x = x_0 cross-section hamiltonian's...
    for idx_y, y in enumerate(linspace_y):

        # ...onsite elements...
        hamiltonian_y[idx_y, idx_y] = 4.0*alpha + potential_QPC(y)

        # ...and hopping elements
        if idx_y < len(linspace_y) - 1:
            hamiltonian_y[idx_y + 1, idx_y] = -alpha
            hamiltonian_y[idx_y, idx_y + 1] = -alpha
    
    # (7) a function defining the disperssion relation 
    # E(k) in a translationally invariant channel created
    # by extending the x = 0 cross-section to infinity
    def E_k(k: float) -> np.ndarray:

        # (7.1) plane wave of an electron in x direction
        plane_wave = np.exp(1j*k*dx)

        # (7.2) energy operator of a system with a given
        # eigenvector taking Bloch's theorem into account
        x_invariant_hamiltonian = hamiltonian_y - alpha*(plane_wave + plane_wave**(-1))*np.identity(pm.N_y)

        # (7.3) diagonalizing self equation of the operator (7.2)
        self_energies = np.linalg.eigvalsh(x_invariant_hamiltonian)

        return self_energies
    
    # (8) a function calculating eigen modes
    # in a translationally invariant channel created
    # by extending the x = 0 cross-section to infinity
    def k_E(E: float):

        # (8.1) creating the matrices of an equation: Aφ - λBφ = 0
        # that connects two following y-cross-sections in the x direction
        A, B = np.zeros((2*pm.N_y, 2*pm.N_y), dtype=complex), np.zeros((2*pm.N_y, 2*pm.N_y), dtype=complex)

        # (8.2) defining the helper tau matrix (hopping matrix)
        tau = -alpha*np.identity(pm.N_y, dtype = complex)

        # (8.3) defining 0,1 matrix element of A
        A[:pm.N_y, pm.N_y:] = np.identity(pm.N_y, dtype = complex)

        # (8.4) defining 1,0 matrix element of A
        A[pm.N_y:, :pm.N_y] = -tau

        # (8.5) defining 1,1 matrix element of A
        A[pm.N_y:, pm.N_y:] = E*np.identity(pm.N_y, dtype = complex) - hamiltonian_y

        # (8.6) defining 0,0 matrix element of B
        B[:pm.N_y, :pm.N_y] = np.identity(pm.N_y, dtype = complex)

        # (8.7) defining 1,1 matrix element of B
        B[pm.N_y:, pm.N_y:] = tau.conj().T

        # (8.8) diagonalizing the equation (B^-1)Aφ - λφ = 0
        self_lambdas, self_vectors = np.linalg.eig(np.linalg.inv(B) @ A)
        self_vectors = self_vectors.T

        # (8.9) finding propagating modes
        propagating_ks, propagating_lambdas, propagating_vectors, propagating_velocities = [], [], [], []
        for idx, lmbda in enumerate(self_lambdas):

            # checking the λ = 1 condition
            if np.isclose(np.abs(lmbda), 1.0):

                lmbda /= np.abs(lmbda)
                
                # mode's wavevector
                k = np.angle(lmbda)/dx
                propagating_ks.append(k)

                # λ value
                propagating_lambdas.append(lmbda)

                # transverse mode
                u = self_vectors[idx, :pm.N_y]
                u /= np.sqrt(np.vdot(u, u)) # normalising
                propagating_vectors.append(u)

                # velocity
                v = -2.0*dx*np.imag(lmbda*(u.conj().T @ tau.conj().T @ u))
                propagating_velocities.append(v)

        # szacher-macher
        plus_idxs = [idx for idx, velocity in enumerate(propagating_velocities) if velocity >= 0] # "+" velocities
        propagating_ks = [propagating_ks[idx] for idx in plus_idxs]
        propagating_lambdas = [propagating_lambdas[idx] for idx in plus_idxs]
        propagating_vectors = [propagating_vectors[idx]for idx in plus_idxs]
        propagating_velocities = [propagating_velocities[idx] for idx in plus_idxs]
        propagating_ks += [-k for k in propagating_ks]
        propagating_lambdas += [lmbda**(-1) for lmbda in propagating_lambdas]
        propagating_vectors += propagating_vectors
        propagating_velocities += [-v for v in propagating_velocities]
                
        return [propagating_ks, propagating_lambdas, propagating_vectors, propagating_velocities]

    return (E_k, k_E)

def makesystem_hamiltonian(pm: SimpleNamespace) -> np.ndarray:

    '''
    returns a matrix of a 2D system hamiltonian
    '''

    # (1) creates a function defining QPC potential in 2DEG plane
    potential_QPC = make_potential_QPC(pm)

    # (2) defines x and y coordinates of system nodes
    dx = (pm.y_max - pm.y_min)/(pm.N_y + 1)
    linspace_y = np.array([pm.y_min + i*dx for i in range(1, pm.N_y + 1)])

    pm.N_x = int((pm.x_max - pm.x_min)//dx)
    linspace_x = np.array([pm.x_min + i*dx for i in range(1, pm.N_x + 1)])

    # (3) defines a discretization constants of the
    # time-independent Schroedinger equation
    alpha = 0.5/pm.m_eff/dx**2

    # (4) defines a hopping matrix in the x direction
    tau_x = -alpha*np.identity(pm.N_y, dtype = complex)

    # (5) creating a matrix of a system hamiltonian
    hamiltonian = np.zeros((pm.N_x*pm.N_y, pm.N_x*pm.N_y), dtype = complex)
    for idx_x, x in enumerate(linspace_x):

        # (5.1) creates a hamiltonian matrix of the
        # transverse cross-section of the system at certain x
        hamiltonian_y = np.zeros((pm.N_y, pm.N_y), dtype = complex)

        # (5.2) defines a certain x cross-section hamiltonian's...
        for idx_y, y in enumerate(linspace_y):

            # ...onsite elements...
            hamiltonian_y[idx_y, idx_y] = 4.0*alpha + potential_QPC(x, y)

            # ...and hopping elements
            if idx_y < len(linspace_y) - 1:
                hamiltonian_y[idx_y + 1, idx_y] = -alpha
                hamiltonian_y[idx_y, idx_y + 1] = -alpha

        # (5.3) filling matrix of a system hamiltonian
        iterator = slice(idx_x*pm.N_y, (idx_x + 1)*pm.N_y)
        iterator_plus = slice((idx_x + 1)*pm.N_y, (idx_x + 2)*pm.N_y)
        hamiltonian[iterator, iterator] = hamiltonian_y
        if idx_x < len(linspace_x) - 1:
            hamiltonian[iterator_plus, iterator] = tau_x.conj().T
            hamiltonian[iterator, iterator_plus] = tau_x
    
    return hamiltonian