###################################################################################################
##                      Parameters: https://github.com/cchandre/GC2D_intranet                    ##
###################################################################################################

import numpy as xp
import multiprocessing as mp
from gc2ds_classes import GC2Ds

# potential
A = 0.6
M = 25

# parameters
Ntraj = 500
n_max = 500

n_data = 200
n_process = 100

default_time_step = 0.1
default_omega = 10
solver = 'BM4'  

parameters = {"A": A, "M": M}
hs = GC2Ds(parameters)
z0 = hs.initial_conditions(Ntraj, type="random")

t_eval = 2 * xp.pi * xp.arange(n_max)

#lyap = hs.compute_lyapunov(2 * xp.pi * n_max, z0, reortho_dt=1, tol=1e-10, solver='RK45')
#print(lyap)

# Plot of the Poincaré section
#sol = hs.integrate(z0, t_eval, timestep=5e-2, solver='BM4', extension=True)
#hs.plot_sol(sol, wrap=True)
#hs.plot_sol(sol)

parameters.update({"Ntraj": Ntraj, "n_max": n_max, "solver": solver})

mode = 'omega'

if mode == 'omega':
    param_list = xp.logspace(-2, 2, n_data)  
    parameters.update({"mode": mode, "omega": param_list, "timestep": default_time_step})
elif mode == 'step':
    param_list = xp.logspace(-2, 0, n_data)[::-1]  
    parameters.update({"mode": mode, "omega": default_omega, "timestep": param_list})
else:
    raise ValueError("Mode must be 'omega' or 'step'")

def run_one(param):
    step = param if mode == 'step' else default_time_step
    om = param if mode == 'omega' else default_omega 
    sol = hs.integrate(z0, t_eval, timestep=step, omega=om, display=False, solver=solver, extension=True, check_energy=True)
    print(f"{mode} = {param:.3e}   error = {sol.err / Ntraj}  dist_copy = {sol.dist_copy}  CPU_time = {int(sol.cpu_time)}s")
    return (sol.step, sol.err / Ntraj, sol.dist_copy, sol.cpu_time)

if __name__ == '__main__':
    with mp.Pool(processes=n_process) as pool:
        results = pool.map(run_one, param_list)

    sorted_results = sorted(results, key=lambda pair: pair[0])
    hs.save_data(sorted_results, params=parameters)
