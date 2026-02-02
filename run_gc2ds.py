###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D                             ##
###################################################################################################

import numpy as np
import multiprocessing as mp
import csv
from gc2ds_classes import GC2Ds

## Parameters
A = 0.6
M = 25

Ntraj = 500
n_max = 500

n_data = 200
n_process = 100

default_time_step = 0.1
solver = 'BM4'
default_omega = None
projection = 'midpoint'

parameters = {"A": A, "M": M}
gc = GC2Ds(parameters)
z0 = gc.initial_conditions(Ntraj, kind="random")
t_eval = 2 * np.pi * np.arange(n_max)

## Computation of the Lyapunov exponent
# lyap = gc.compute_lyapunov(2 * np.pi * n_max, z0, reortho_dt=1, tol=1e-10, solver='RK45')
# print(lyap)

## Plot of the Poincaré section
# sol = gc.integrate(z0, t_eval, timestep=default_time_step, solver=solver, omega=default_omega, extension=True, projection=projection, tol=1e-10, max_iter=100, check_energy=True)
# gc.plot_sol(sol, wrap=True)
# gc.plot_sol(sol)

mode = 'step'

if mode == 'omega':
    param_list = np.logspace(-2, 2, n_data)  
elif mode == 'step':
    param_list = np.logspace(-2, 0, n_data)[::-1]  

def run_one(param):
    step = param if mode == 'step' else default_time_step
    om = param if mode == 'omega' else default_omega 
    sol = gc.integrate(z0, t_eval, timestep=step, omega=om, display=False, solver=solver, extension=True, check_energy=True, projection=projection, tol=1e-10, max_iter=100)
    print(f"{mode} = {param:.3e}   error = {sol.err / Ntraj}  CPU_time = {int(sol.cpu_time)}s with projection = {sol.projection} and proj_dist = {sol.proj_dist}  ")
    return {"A": A, "M": M, "Ntraj": Ntraj, "n_max": n_max, "solver": solver, "timesep": sol.step, "omega": om, 
        "error": sol.err / Ntraj, "cpu_time": sol.cpu_time, "projection": sol.projection, "proj_dist": sol.proj_dist}

if __name__ == '__main__':
    output_file = f"{mode}_results.csv"
    fieldnames = ['A', 'M', 'Ntraj', 'n_max', 'solver', 'timesep', 'omega', 'error', 'cpu_time', 'projection', 'proj_dist']
    with mp.Pool(processes=n_process) as pool:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in pool.imap_unordered(run_one, param_list):
                writer.writerow(result)
                csvfile.flush()