# Gravitational N-Body Simulation

`nbody_cpu.cpp` contains an OpenMP implementation for the N-body problem. Execution using a single thread is interpreted as the sequential implementation. `nbody_gpu.cu` contains a CUDA implementation for the N-body problem. `visualize_simulation.py` generates a plot of execution time against body and thread count for each implementation, and a gif animating the simulation for the first twenty generated bodies.

## Features
Sequential, OpenMP, and CUDA implementations

Automated benchmarking pipeline

Scaling visualization and animated trajectory output

Deterministic initialization for reproducible experiments

Reflective boundary conditions

## Requirements
`build_nbody_simulation.sh` expects a virtual environment named `venv` in the present working directory upon execution.

### Install packages:
```bash
pip install -r requirements.txt
```

## Usage
### Run code:
 ```
g++ -fopenmp -o nbody_cpu nbody_cpu.cpp
nvcc -o nbody_gpu nbody_gpu.cu
sudo bash build_nbody_simulation.sh
 ```
