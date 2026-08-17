"""Interactive 2D smoke and vorticity demo using a Stable Fluids solver.

The numerical core depends only on NumPy and can be imported without opening a
GUI. Run ``python swirl.py`` (or the installed ``swirl-fluid`` command) to open
the Matplotlib application.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Numerical and scene parameters for :class:`StableFluid2D`."""

    grid_size: int = 96
    dt: float = 0.05
    diffusion: float = 0.0
    viscosity: float = 5e-5
    solver_iterations: int = 24
    density_decay: float = 0.12
    background_flow: bool = True
    background_speed: float = 0.35
    background_ink_rate: float = 40.0
    obstacle: bool = True
    obstacle_radius_fraction: float = 0.15

    def __post_init__(self) -> None:
        if self.grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.diffusion < 0.0 or self.viscosity < 0.0:
            raise ValueError("diffusion and viscosity cannot be negative")
        if self.solver_iterations < 1:
            raise ValueError("solver_iterations must be positive")
        if self.density_decay < 0.0:
            raise ValueError("density_decay cannot be negative")
        if self.background_speed < 0.0 or self.background_ink_rate < 0.0:
            raise ValueError("background flow parameters cannot be negative")
        if not 0.0 <= self.obstacle_radius_fraction < 0.5:
            raise ValueError("obstacle_radius_fraction must be in [0, 0.5)")


class StableFluid2D:
    """A compact, deterministic Stable Fluids simulation on a square grid.

    Arrays include a one-cell ghost boundary, so public grid coordinates are
    integers from 1 through ``grid_size``. Velocity is expressed in domain
    widths per second; density is an arbitrary visualization scalar.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self.n = self.config.grid_size
        self.shape = (self.n + 2, self.n + 2)

        self.u = self._zeros()
        self.v = self._zeros()
        self.u_source = self._zeros()
        self.v_source = self._zeros()
        self.density = self._zeros()
        self.density_source = self._zeros()
        self.solid = self._make_obstacle_mask()

        self._pressure = self._zeros()
        self._divergence = self._zeros()
        self._work_u = self._zeros()
        self._work_v = self._zeros()
        self._work_density = self._zeros()

    def _zeros(self) -> FloatArray:
        return np.zeros(self.shape, dtype=np.float64)

    def _make_obstacle_mask(self) -> NDArray[np.bool_]:
        mask = np.zeros(self.shape, dtype=bool)
        if not self.config.obstacle or self.config.obstacle_radius_fraction == 0:
            return mask

        center = (self.n + 1) / 2.0
        y, x = np.indices(self.shape)
        radius = self.n * self.config.obstacle_radius_fraction
        mask[:] = (x - center) ** 2 + (y - center) ** 2 <= radius**2
        return mask

    def reset(self) -> None:
        """Clear velocity, density, and pending sources."""

        for field in (
            self.u,
            self.v,
            self.u_source,
            self.v_source,
            self.density,
            self.density_source,
        ):
            field.fill(0.0)

    def add_density(self, x: int, y: int, amount: float, radius: int = 1) -> None:
        """Add a circular density source centered on a public grid coordinate."""

        self._splat(self.density_source, x, y, amount, radius)

    def add_velocity(
        self,
        x: int,
        y: int,
        amount_x: float,
        amount_y: float,
        radius: int = 1,
    ) -> None:
        """Add a circular velocity source centered on a grid coordinate."""

        self._splat(self.u_source, x, y, amount_x, radius)
        self._splat(self.v_source, x, y, amount_y, radius)

    def _splat(
        self,
        field: FloatArray,
        x: int,
        y: int,
        amount: float,
        radius: int,
    ) -> None:
        if radius < 0:
            raise ValueError("radius cannot be negative")
        x = int(np.clip(x, 1, self.n))
        y = int(np.clip(y, 1, self.n))
        x0, x1 = max(1, x - radius), min(self.n, x + radius)
        y0, y1 = max(1, y - radius), min(self.n, y + radius)
        xs = np.arange(x0, x1 + 1)[:, None]
        ys = np.arange(y0, y1 + 1)[None, :]
        sigma = max(0.75, radius / 1.5)
        weights = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma**2))
        weights[(xs - x) ** 2 + (ys - y) ** 2 > radius**2] = 0.0
        total = float(weights.sum())
        if total:
            field[x0 : x1 + 1, y0 : y1 + 1] += amount * weights / total

    def _set_boundary(self, boundary_type: int, field: FloatArray) -> None:
        n = self.n
        if self.config.background_flow:
            if boundary_type == 1:
                field[0, 1 : n + 1] = 2.0 * self.config.background_speed - field[
                    1, 1 : n + 1
                ]
                field[n + 1, 1 : n + 1] = field[n, 1 : n + 1]
            else:
                field[0, 1 : n + 1] = field[1, 1 : n + 1]
                field[n + 1, 1 : n + 1] = field[n, 1 : n + 1]
        elif boundary_type == 1:
            field[0, 1 : n + 1] = -field[1, 1 : n + 1]
            field[n + 1, 1 : n + 1] = -field[n, 1 : n + 1]
        else:
            field[0, 1 : n + 1] = field[1, 1 : n + 1]
            field[n + 1, 1 : n + 1] = field[n, 1 : n + 1]

        if boundary_type == 2:
            field[1 : n + 1, 0] = -field[1 : n + 1, 1]
            field[1 : n + 1, n + 1] = -field[1 : n + 1, n]
        else:
            field[1 : n + 1, 0] = field[1 : n + 1, 1]
            field[1 : n + 1, n + 1] = field[1 : n + 1, n]

        field[0, 0] = 0.5 * (field[1, 0] + field[0, 1])
        field[0, n + 1] = 0.5 * (field[1, n + 1] + field[0, n])
        field[n + 1, 0] = 0.5 * (field[n, 0] + field[n + 1, 1])
        field[n + 1, n + 1] = 0.5 * (field[n, n + 1] + field[n + 1, n])

    def _linear_solve(
        self,
        boundary_type: int,
        field: FloatArray,
        previous: FloatArray,
        coefficient: float,
        denominator: float,
    ) -> None:
        n = self.n
        if coefficient == 0.0:
            field[:] = previous
            field[self.solid] = 0.0
            self._set_boundary(boundary_type, field)
            return

        reciprocal = 1.0 / denominator
        for _ in range(self.config.solver_iterations):
            field[1 : n + 1, 1 : n + 1] = (
                previous[1 : n + 1, 1 : n + 1]
                + coefficient
                * (
                    field[0:n, 1 : n + 1]
                    + field[2 : n + 2, 1 : n + 1]
                    + field[1 : n + 1, 0:n]
                    + field[1 : n + 1, 2 : n + 2]
                )
            ) * reciprocal
            field[self.solid] = 0.0
            self._set_boundary(boundary_type, field)

    def _diffuse(
        self,
        boundary_type: int,
        field: FloatArray,
        previous: FloatArray,
        rate: float,
    ) -> None:
        coefficient = self.config.dt * rate * self.n**2
        self._linear_solve(
            boundary_type,
            field,
            previous,
            coefficient,
            1.0 + 4.0 * coefficient,
        )

    def _advect(
        self,
        boundary_type: int,
        destination: FloatArray,
        source: FloatArray,
        velocity_x: FloatArray,
        velocity_y: FloatArray,
    ) -> None:
        n = self.n
        scaled_dt = self.config.dt * n
        i = np.arange(1, n + 1)[:, None]
        j = np.arange(1, n + 1)[None, :]
        x = np.clip(
            i - scaled_dt * velocity_x[1 : n + 1, 1 : n + 1], 0.5, n + 0.5
        )
        y = np.clip(
            j - scaled_dt * velocity_y[1 : n + 1, 1 : n + 1], 0.5, n + 0.5
        )

        i0 = np.floor(x).astype(np.intp)
        j0 = np.floor(y).astype(np.intp)
        i1 = i0 + 1
        j1 = j0 + 1
        sx = x - i0
        sy = y - j0

        destination[1 : n + 1, 1 : n + 1] = (
            (1.0 - sx)
            * ((1.0 - sy) * source[i0, j0] + sy * source[i0, j1])
            + sx * ((1.0 - sy) * source[i1, j0] + sy * source[i1, j1])
        )
        destination[self.solid] = 0.0
        self._set_boundary(boundary_type, destination)

    def _project(self) -> None:
        n = self.n
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0
        div = self._divergence
        pressure = self._pressure
        div.fill(0.0)
        pressure.fill(0.0)
        div[1 : n + 1, 1 : n + 1] = -0.5 * (
            self.u[2 : n + 2, 1 : n + 1] - self.u[0:n, 1 : n + 1]
            + self.v[1 : n + 1, 2 : n + 2] - self.v[1 : n + 1, 0:n]
        ) / n
        div[self.solid] = 0.0
        self._set_boundary(0, div)
        self._set_boundary(0, pressure)
        self._linear_solve(0, pressure, div, 1.0, 4.0)

        self.u[1 : n + 1, 1 : n + 1] -= 0.5 * n * (
            pressure[2 : n + 2, 1 : n + 1] - pressure[0:n, 1 : n + 1]
        )
        self.v[1 : n + 1, 1 : n + 1] -= 0.5 * n * (
            pressure[1 : n + 1, 2 : n + 2] - pressure[1 : n + 1, 0:n]
        )
        self.u[self.solid] = 0.0
        self.v[self.solid] = 0.0
        self._set_boundary(1, self.u)
        self._set_boundary(2, self.v)

    def _velocity_step(self) -> None:
        dt = self.config.dt
        self.u += dt * self.u_source
        self.v += dt * self.v_source

        self._work_u[:] = self.u
        self._work_v[:] = self.v
        self._diffuse(1, self.u, self._work_u, self.config.viscosity)
        self._diffuse(2, self.v, self._work_v, self.config.viscosity)
        self._project()

        self._work_u[:] = self.u
        self._work_v[:] = self.v
        self._advect(1, self.u, self._work_u, self._work_u, self._work_v)
        self._advect(2, self.v, self._work_v, self._work_u, self._work_v)
        self._project()

    def _density_step(self) -> None:
        self.density += self.config.dt * self.density_source
        self._work_density[:] = self.density
        self._diffuse(0, self.density, self._work_density, self.config.diffusion)
        self._work_density[:] = self.density
        self._advect(0, self.density, self._work_density, self.u, self.v)
        if self.config.density_decay:
            self.density *= np.exp(-self.config.density_decay * self.config.dt)
        self.density[self.solid] = 0.0

    def _inject_background(self) -> None:
        if not self.config.background_flow:
            return
        self.u[1, 1 : self.n + 1] = self.config.background_speed
        center = self.n // 2 + 1
        half_width = max(2, self.n // 5)
        self.density_source[
            1, center - half_width : center + half_width + 1
        ] += self.config.background_ink_rate

    def step(self) -> None:
        """Advance the simulation by one configured time step."""

        self._inject_background()
        self._velocity_step()
        if self.config.background_flow:
            self.u[1, 1 : self.n + 1] = self.config.background_speed
        self._density_step()
        self.u_source.fill(0.0)
        self.v_source.fill(0.0)
        self.density_source.fill(0.0)

    def divergence(self) -> FloatArray:
        """Return divergence on the interior grid, useful for diagnostics."""

        n = self.n
        result = 0.5 * n * (
            self.u[2 : n + 2, 1 : n + 1] - self.u[0:n, 1 : n + 1]
            + self.v[1 : n + 1, 2 : n + 2] - self.v[1 : n + 1, 0:n]
        )
        return np.where(self.solid[1:-1, 1:-1], 0.0, result)

    def vorticity(self) -> FloatArray:
        """Return scalar curl, ``dv/dx - du/dy``, on the interior grid."""

        n = self.n
        result = 0.5 * n * (
            self.v[2 : n + 2, 1 : n + 1] - self.v[0:n, 1 : n + 1]
            - self.u[1 : n + 1, 2 : n + 2]
            + self.u[1 : n + 1, 0:n]
        )
        return np.where(self.solid[1:-1, 1:-1], 0.0, result)


class SwirlApp:
    """Matplotlib front end for :class:`StableFluid2D`."""

    def __init__(
        self,
        simulation: StableFluid2D,
        force_scale: float = 120.0,
        ink_amount: float = 300.0,
        brush_radius: int = 2,
        interval_ms: int = 30,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Circle

        self.simulation = simulation
        self.force_scale = force_scale
        self.ink_amount = ink_amount
        self.brush_radius = brush_radius
        self.paused = False
        self._last_mouse: tuple[float, float] | None = None
        self._button: int | None = None

        n = simulation.n
        self.figure, (self.density_axis, self.vorticity_axis) = plt.subplots(
            1, 2, figsize=(11, 5.5)
        )
        self.density_axis.set_title("Density")
        self.vorticity_axis.set_title("Vorticity")
        for axis in (self.density_axis, self.vorticity_axis):
            axis.set(xlabel="x", ylabel="y", xlim=(0, n), ylim=(0, n))

        self.density_image = self.density_axis.imshow(
            simulation.density[1:-1, 1:-1].T,
            origin="lower",
            extent=(0, n, 0, n),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        self.vorticity_image = self.vorticity_axis.imshow(
            simulation.vorticity().T,
            origin="lower",
            extent=(0, n, 0, n),
            cmap="seismic",
            vmin=-1.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        if simulation.config.obstacle:
            center = n / 2.0
            radius = n * simulation.config.obstacle_radius_fraction
            for axis in (self.density_axis, self.vorticity_axis):
                axis.add_patch(
                    Circle((center, center), radius, color="black", zorder=3)
                )

        self.figure.suptitle(
            "Left-drag: ink + force   Right-drag: force   Space: pause   R: reset",
            fontsize=10,
        )
        self.figure.tight_layout()
        self.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.animation = FuncAnimation(
            self.figure,
            self._update,
            interval=interval_ms,
            blit=False,
            cache_frame_data=False,
        )

    def _grid_coordinate(self, x: float, y: float) -> tuple[int, int]:
        n = self.simulation.n
        return int(np.clip(np.floor(x) + 1, 1, n)), int(
            np.clip(np.floor(y) + 1, 1, n)
        )

    def _on_press(self, event: object) -> None:
        if getattr(event, "inaxes", None) not in (
            self.density_axis,
            self.vorticity_axis,
        ):
            return
        x, y = getattr(event, "xdata", None), getattr(event, "ydata", None)
        if x is None or y is None:
            return
        self._last_mouse = (x, y)
        raw_button = getattr(event, "button", None)
        self._button = int(raw_button) if raw_button is not None else None
        if self._button == 1:
            i, j = self._grid_coordinate(x, y)
            self.simulation.add_density(i, j, self.ink_amount, self.brush_radius)

    def _on_release(self, _event: object) -> None:
        self._last_mouse = None
        self._button = None

    def _on_motion(self, event: object) -> None:
        if self._last_mouse is None or self._button is None:
            return
        if getattr(event, "inaxes", None) not in (
            self.density_axis,
            self.vorticity_axis,
        ):
            return
        x, y = getattr(event, "xdata", None), getattr(event, "ydata", None)
        if x is None or y is None:
            return

        previous_x, previous_y = self._last_mouse
        i, j = self._grid_coordinate(x, y)
        scale = self.force_scale / self.simulation.n
        self.simulation.add_velocity(
            i,
            j,
            scale * (x - previous_x),
            scale * (y - previous_y),
            self.brush_radius,
        )
        if self._button == 1:
            self.simulation.add_density(i, j, self.ink_amount, self.brush_radius)
        self._last_mouse = (x, y)

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key == " ":
            self.paused = not self.paused
        elif key and key.lower() == "r":
            self.simulation.reset()

    def _update(self, _frame: int) -> tuple[object, object]:
        if not self.paused:
            self.simulation.step()

        density = self.simulation.density[1:-1, 1:-1].T
        self.density_image.set_array(density)
        positive_density = density[density > 0.0]
        density_max = (
            max(1.0, float(np.percentile(positive_density, 99.5)))
            if positive_density.size
            else 1.0
        )
        self.density_image.set_clim(0.0, density_max)

        vorticity = self.simulation.vorticity().T
        self.vorticity_image.set_array(vorticity)
        vorticity_max = max(1.0, float(np.percentile(np.abs(vorticity), 99.5)))
        self.vorticity_image.set_clim(-vorticity_max, vorticity_max)
        return self.density_image, self.vorticity_image

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--viscosity", type=float, default=5e-5)
    parser.add_argument("--diffusion", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--density-decay", type=float, default=0.12)
    parser.add_argument("--no-background-flow", action="store_true")
    parser.add_argument("--background-speed", type=float, default=0.35)
    parser.add_argument("--background-ink-rate", type=float, default=40.0)
    parser.add_argument("--no-obstacle", action="store_true")
    parser.add_argument("--obstacle-radius", type=float, default=0.15)
    parser.add_argument("--force-scale", type=float, default=120.0)
    parser.add_argument("--ink-amount", type=float, default=300.0)
    parser.add_argument("--brush-radius", type=int, default=2)
    parser.add_argument(
        "--interval", type=int, default=30, help="animation interval in ms"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SimulationConfig(
        grid_size=args.grid_size,
        dt=args.dt,
        diffusion=args.diffusion,
        viscosity=args.viscosity,
        solver_iterations=args.iterations,
        density_decay=args.density_decay,
        background_flow=not args.no_background_flow,
        background_speed=args.background_speed,
        background_ink_rate=args.background_ink_rate,
        obstacle=not args.no_obstacle,
        obstacle_radius_fraction=args.obstacle_radius,
    )
    app = SwirlApp(
        StableFluid2D(config),
        force_scale=args.force_scale,
        ink_amount=args.ink_amount,
        brush_radius=args.brush_radius,
        interval_ms=args.interval,
    )
    app.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
