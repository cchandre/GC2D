# Guiding-Center dynamics in plasma physics

- [`gc2ds_classes.py`](https://github.com/cchandre/GC2D/blob/main/gc2ds_classes.py): contains the GC classes and main functions defining the GC dynamics

- [`run_gc2ds.py`](https://github.com/cchandre/GC2D/blob/main/run_gc2ds.py): example of a run file to reproduce the computations done in [REF]

Once [`run_gc2ds.py`](https://github.com/cchandre/GC2D/blob/main/run_gc2ds.py) has been edited with the relevant parameters, run the file as 
```sh
python3 run_gc2ds.py
```
or 
```sh
nohup python3 -u run_gc2ds.py &>gc2d.out < /dev/null &
```
The list of Python packages and their version are specified in [`requirements.txt`](https://github.com/cchandre/GC2D/blob/main/requirements.txt)
___
###  Main parameters of the class GC2Ds

- *A*: Amplitude of the electrostatic potential
- *M*: Number of modes in the electrostatic potential
- *seed*: Seed for the random phases of the electrostatic potential (optional; default=27)

#### Example 
```python
params = {"A": 1.0, "M": 16, "seed": 42}
gc = GC2Ds(params)
z0 = gc.initial_conditions(100, type="random")
```

### Key methods

Since GC2Ds is s subclass of HamSys (from the python package [`pyhamsys`](https://pypi.org/project/pyhamsys/)), it inherits all its methods, including:

- `integrate`: Integrate numerically the trajectories of the system defined by the element of the class GC2Ds from the initial conditions defined by the function `initial_conditions`. 

- `compute_lyapunov`: Compute the Lyapunov spectrum. 

In addition, GC2Ds has the following key methods:

- `initial_conditions`: Generate starting (x, y) positions—random or on a regular grid.

- `y_dot`: Time derivative of positions for integration.

- `k_dot`: Scalar diagnostic of the potential field.

- `potential`: Potential value at time t and position z=(x, y), and its first and second derivatives, obtained by specifying (*dx*, *dy*).

- `hamiltonian`: Total Hamiltonian (sum of the potentials for each trajectory).

- `y_dot_lyap`: Extended system (equations of motion and tangent flow) for Lyapunov-exponent calculations.

- `plot_sol`: 2-D plot of a solution obtained by the function `integrate`.

- `save_data`: Save simulation results to a `.mat` file with metadata.

#### `initial_conditions`

Generate initial 2-D coordinates for trajectories on a periodic domain.

#### Usage

``` python
initial_conditions(
    n_traj: int = 1,
    x: tuple[float, float] | None = None,
    y: tuple[float, float] | None = None,
    type: str = "fixed",
    seed: int | None = None
) -> xp.ndarray
```

#### Parameters

-   **n_traj**: Number of points. For `"fixed"`, rounded to a perfect square for a square grid.
-   **x**, **y**: (min, max) ranges; default `(0, 2π)`.
-   **type**: `"random"` for uniform random samples, `"fixed"` for a regular grid.
-   **seed**: Random seed when `type="random"`.

#### Returns

1-D `xp.ndarray` of length `2*n_traj`: all x's followed by all y's.

#### Example

``` python
z0 = gc.initial_conditions(50, type="random", seed=123)
z0 = gc.initial_conditions(100, x=(0, np.pi), y=(0, np.pi), type="fixed")
```


---
Reference: 

For more information: <cristel.chandre@cnrs.fr>
