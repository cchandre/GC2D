###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D                             ##
###################################################################################################

import numpy as np
import csv
from gc2ds_classes import GC2Ds
from pyhamsys import Parameters

## Parameters
A = 0.6
M = 25

Ntraj = 10
n_max = 50

n_data = 200
n_process = 100

params = Parameters(
    step=0.1,
    display=True,
    solver='RK45',
    extension=False,
    check_energy=True,
    projection=None,
    tol=1e+4
)

gc = GC2Ds({"A": A, "M": M})
z0 = gc.initial_conditions(Ntraj, kind="random", seed=27)
t_eval = 2 * np.pi * np.arange(n_max)

sol = gc.integrate(z0, t_eval, params=params)

val_h = np.array([gc.hamiltonian(t, y) for t, y in zip(sol.t, sol.y.T)]) + sol.k

energy_vs_time = val_h - val_h[0]

gc.save_data(params=params, filename="energy_vs_time", h=energy_vs_time)
