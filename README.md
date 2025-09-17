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
####  Main parameters

- *A*: Amplitude of the electrostatic potential
- *M*: Number of modes in the electrostatic potential
- *seed*: Seed for the random phases of the electrostatic potential (optional; default=27)

#### Main functions


---
Reference: 

For more information: <cristel.chandre@cnrs.fr>
