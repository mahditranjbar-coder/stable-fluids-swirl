# Stable Fluids Swirl

An interactive 2D smoke simulation built with NumPy and Matplotlib. It uses a
semi-Lagrangian **Stable Fluids** solver to visualize density and vorticity as
flow moves around a circular obstacle.

This is an educational real-time simulation, not a validated CFD solver. The
obstacle and open-flow boundary conditions are approximations on a collocated
grid, so the output should not be used for engineering calculations.

## Features

- Interactive ink and velocity injection with the mouse
- Side-by-side density and vorticity views
- Optional circular obstacle and background flow
- Importable, headless numerical core with no GUI side effects
- Configurable CLI and automated tests

## Install and run

Python 3.10 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
swirl-fluid
```

You can also run it directly:

```bash
python swirl.py
```

Controls:

- Left-drag: add ink and velocity
- Right-drag: add velocity only
- `Space`: pause or resume
- `R`: reset

For all options:

```bash
swirl-fluid --help
```

Example:

```bash
swirl-fluid --grid-size 128 --background-speed 0.25 --obstacle-radius 0.12
```

## Use the solver from Python

```python
from swirl import SimulationConfig, StableFluid2D

simulation = StableFluid2D(
    SimulationConfig(grid_size=64, background_flow=False, obstacle=False)
)
simulation.add_density(32, 32, amount=100.0, radius=2)
simulation.add_velocity(32, 32, amount_x=0.2, amount_y=0.0, radius=2)

for _ in range(100):
    simulation.step()

curl = simulation.vorticity()
divergence = simulation.divergence()
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Numerical method

Each time step adds sources, diffuses velocity, projects it toward a
divergence-free field, advects velocity, projects again, then diffuses and
advects density. Advection uses backward semi-Lagrangian tracing with bilinear
interpolation. Pressure and diffusion equations use Jacobi iteration.

## License

GNU General Public License v3.0 (`GPL-3.0-only`).
