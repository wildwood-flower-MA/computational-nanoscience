import numpy as np
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)

with open("temp.dat", "r") as temp:
    data = np.loadtxt("temp.dat")

    x_values = data[:, 0]
    energies = [data[:, i] for i in range(1, 11)]
    analytical_energies = [data[:, i] for i in range(11, 21)]

    plt.figure(figsize=(10, 6))
    for i, energy in enumerate(energies):
        plt.plot(x_values, energy, label=f"{i}")
    plt.xlabel("h_bar omega_x [meV]", fontsize=16)
    plt.ylabel("E [meV]", fontsize=16)
    plt.title("Energie 10 najniższych stanów", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for i, energy in enumerate(analytical_energies):
        plt.plot(x_values, energy, linestyle='--', label=f"{i}")
    plt.xlabel("h_bar omega_x [meV]", fontsize=16)
    plt.ylabel("E [meV]", fontsize=16)
    plt.title("Energie 10 najniższych stanów: analitycznie", fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
