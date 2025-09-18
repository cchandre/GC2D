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

### Main functions and attributes

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

-   **n_traj**: Number of points. For `"fixed"`, rounded to a perfect
    square for a square grid.\
-   **x**, **y**: (min, max) ranges; default `(0, 2π)`.\
-   **type**: `"random"` for uniform random samples, `"fixed"` for a
    regular grid.\
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
