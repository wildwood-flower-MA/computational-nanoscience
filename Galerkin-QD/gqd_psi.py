import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mgimg
import matplotlib.animation as animation
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20)

with open("temp.dat", "r") as temp:
    first_line = temp.readline().strip()
n, a, dx, alpha_x, alpha_y = map(float, first_line.split())
plansza_raw = np.loadtxt("temp.dat", skiprows = 1).transpose()

X, Y = np.meshgrid(np.linspace(-a, a, int(100)), np.linspace(-a, a, int(100)))
plt.contourf(X, Y, plansza_raw, levels=50, cmap='Greys_r')
plt.xticks([-a,0,a]); plt.yticks([-a,0,a])
plt.xlabel("x [nm]", fontsize=20)
plt.ylabel("y [nm]", fontsize=20)
colbar = plt.colorbar()# label=r'|$\psi$|$^2$')
colbar.ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
colbar.ax.yaxis.label.set_size(20); plt.tight_layout(); plt.show()

