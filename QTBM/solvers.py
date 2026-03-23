import numpy as np
from typing import Tuple
from types import SimpleNamespace
from general_functions import *
from system import *

def makesystem_psi(pm: SimpleNamespace, E: float, plot_disperssion = False):

    '''
    returns system wavefunctions
    (list of wavefunctions associated with following modes - to be fair)
    '''

    # (1) calculating a matrix of a system hamiltonian
    hamiltonian = makesystem_hamiltonian(pm).astype(complex)

    # (2) calculating modes in an infinite system (lead)
    E_k, k_E = makesystem_infinite(pm)
    modes_k, modes_lambda, modes_psi, modes_v  = k_E(E)
    if(len(modes_k) == 0):
        modes_k, modes_lambda, modes_psi, modes_v  = k_E(0.5*(E_k(0)[1] + E_k(0)[0]))

    # (2.1) sorting by direction
    modes_lambda_plus = np.array(modes_lambda[:len(modes_lambda)//2], dtype = complex)
    modes_lambda_minus = np.array(modes_lambda[len(modes_lambda)//2:], dtype = complex)
    modes_psi = np.array(modes_psi[:len(modes_psi)//2], dtype = complex)

    # (3) defines a hopping matrix in the x direction
    dx = (pm.y_max - pm.y_min)/(pm.N_y + 1)
    alpha_x = 0.5/pm.m_eff/dx**2
    tau_x = -alpha_x*np.identity(pm.N_y, dtype = complex)

    # (4) defines delta vectors
    delta_plus = (1.0 - 1.0/modes_lambda_plus) # Δ_{+,n}
    delta_minus = (1.0 - 1.0/modes_lambda_minus) # Δ_{-,n}

    # (5) defines alpha and beta matrices
    alpha_matrix, beta_matrix = np.zeros((pm.N_y, pm.N_y), dtype=complex), np.zeros((pm.N_y, pm.N_y), dtype=complex)
    for mu in range(pm.N_y):
        for ni in range(pm.N_y):
            for n in range(len(modes_k)//2):
                alpha_matrix[mu, ni] += np.conj(modes_psi[n, ni])*modes_psi[n, mu]*(1.0 - 1.0/modes_lambda_minus[n])  # α elements
                beta_matrix[mu, ni]  += np.conj(modes_psi[n, ni])*modes_psi[n, mu]*(1.0 - modes_lambda_plus[n]) # β backscatter elements

    # (6) defines matrix equation linear combination matrix
    hamiltonian -= E*np.identity(pm.N_x*pm.N_y, dtype = complex)
    hamiltonian[:pm.N_y, :pm.N_y] += tau_x - (tau_x @ alpha_matrix) # = np.identity(pm.N_y, dtype = complex) # test
    hamiltonian[-pm.N_y:, -pm.N_y:] += tau_x.conj().T - (tau_x.conj().T @ beta_matrix)

    # (7) creating and solving matrix equation
    psi = []
    for mode in range(len(modes_k)//2):

        # (7.1) linear combination coefficients of modes at the entrance to the system
        c_in = np.zeros(len(modes_k)//2)
        c_in[mode] = 1.0

        # (7.2) defining a right hand side vector of a matrix equation first element (e- source)
        v = np.zeros(pm.N_x*pm.N_y, dtype=complex)
        for n in range(len(modes_k)//2):
            v[:pm.N_y] += c_in[n]*(delta_plus[n] - delta_minus[n])*(tau_x @ modes_psi[n]) # = modes_psi[n] # test

        # (7.3) solving matrix equation
        psi_m = np.linalg.solve(hamiltonian, v)
        #psi_m /= np.sqrt(np.vdot(psi_m, psi_m)) #*dx # normalising
        psi.append(psi_m.reshape(pm.N_x, pm.N_y).T)

    # (9*) plotting disperssion at will
    if plot_disperssion == True:
        plot_modes = 4
        E_k_QPC, k_E_QPC = makesystem_infinite(pm, x_0 = 0)
        print(f"Lowest energy in the lead: E_lead = {np.min(E_k(0))/eV2au(1.0)} eV")
        print(f"Lowest energy in the QPC: E_QPC = {np.min(E_k_QPC(0))/eV2au(1.0)} eV")
        Es, Es_QPC = [], []
        for k in np.linspace(-np.pi/dx, np.pi/dx, 100):
            Es.append(E_k(k))
            Es_QPC.append(E_k_QPC(k))
        linsp = np.linspace(-np.pi/dx, np.pi/dx, 100)*nm2au(1.0)
        es = np.array(Es)/eV2au(1.0); es_QPC = np.array(Es_QPC)/eV2au(1.0);
        plt.figure(figsize = (4,4)); plt.xlabel('k (1/nm)'); plt.ylabel('E (eV)')
        plt.plot(linsp, es_QPC, color='Gray'); plt.plot(linsp, es, color='k');
        plt.plot(linsp, np.ones_like(linsp)*E/eV2au(1.0), color='r',
        label = f'E = {E/eV2au(1.0):.2} eV'); plt.legend(frameon = False);
        E0 = E_k(0)/eV2au(1.0); plt.ylim(E0[0], E0[plot_modes]);
        modes2_k, *modes2  = k_E(eV2au(E0[plot_modes])); k_3 = np.max(modes2_k)*nm2au(1.0)
        plt.xlim(-k_3, k_3); plt.show()

    if(len(modes_k) == 0):
        return (np.asarray(psi)*0.0).tolist()
    else:
        return psi

def transmission(pm: SimpleNamespace, energies: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    dx = (pm.y_max - pm.y_min)/(pm.N_y + 1)
    E_k, k_E = makesystem_infinite(pm)
    Ts, Rs = [], []
    E_min = np.min(E_k(0))

    for E in energies:

        # (1) if there is no propagating modes, transmission
        # is equal to 0 and reflection is equal to 1
        if E <= E_min:
            Ts.append(0.0)
            Rs.append(1.0)
            continue

        # (2) calculating propagating modes in the system lead
        *modes, modes_psi, modes_v = k_E(E)
        n_modes = len(modes_psi)//2
        modes_psi = modes_psi[:n_modes] # modes "+"
        modes_v = modes_v[:n_modes] # velocities "+"

        # (3) calculating wavefunctions inside a system
        psis = makesystem_psi(pm, E)

        # (4) calculating transmission and reflection matrix
        T_matrix = np.zeros((n_modes, n_modes))
        R_matrix = np.zeros((n_modes, n_modes))
        for i in range(n_modes):
            for j in range(n_modes):
                
                psi_i = psis[i]

                c_out_ij = np.vdot(modes_psi[j], psi_i[:,0]) - (1.0 if i == j else 0.0)
                d_out_ij = np.vdot(modes_psi[j], psi_i[:,-1])

                T_matrix[i,j] = np.abs(modes_v[i]/modes_v[j])*np.abs(d_out_ij)**2
                R_matrix[i,j] = np.abs(modes_v[i]/modes_v[j])*np.abs(c_out_ij)**2

        Ts.append(np.sum(T_matrix))
        Rs.append(np.sum(R_matrix))

    return (np.array(Ts), np.array(Rs))