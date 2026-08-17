import numpy as np
import pytest

from swirl import SimulationConfig, StableFluid2D, main


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(grid_size=4)
    with pytest.raises(ValueError):
        SimulationConfig(dt=0.0)
    with pytest.raises(ValueError):
        SimulationConfig(obstacle_radius_fraction=0.5)


def test_step_is_finite_and_clears_sources() -> None:
    simulation = StableFluid2D(
        SimulationConfig(grid_size=24, background_flow=False, obstacle=False)
    )
    simulation.add_density(12, 12, 100.0, radius=2)
    simulation.add_velocity(12, 12, 0.4, -0.2, radius=2)
    simulation.step()

    assert np.isfinite(simulation.density).all()
    assert np.isfinite(simulation.u).all()
    assert np.isfinite(simulation.v).all()
    assert simulation.density.sum() > 0.0
    assert not simulation.density_source.any()
    assert not simulation.u_source.any()
    assert not simulation.v_source.any()


def test_obstacle_remains_empty() -> None:
    simulation = StableFluid2D(
        SimulationConfig(grid_size=32, background_flow=False, obstacle=True)
    )
    simulation.density[:] = 10.0
    simulation.u[:] = 1.0
    simulation.v[:] = -1.0
    simulation.step()

    assert not simulation.density[simulation.solid].any()
    assert not simulation.u[simulation.solid].any()
    assert not simulation.v[simulation.solid].any()


def test_projection_reduces_divergence() -> None:
    simulation = StableFluid2D(
        SimulationConfig(
            grid_size=32,
            background_flow=False,
            obstacle=False,
            solver_iterations=60,
        )
    )
    rng = np.random.default_rng(7)
    simulation.u[1:-1, 1:-1] = rng.normal(0.0, 0.05, (32, 32))
    simulation.v[1:-1, 1:-1] = rng.normal(0.0, 0.05, (32, 32))
    simulation._set_boundary(1, simulation.u)
    simulation._set_boundary(2, simulation.v)
    before = np.linalg.norm(simulation.divergence())

    simulation._project()
    after = np.linalg.norm(simulation.divergence())

    assert after < before


def test_reset_clears_state() -> None:
    simulation = StableFluid2D(SimulationConfig(grid_size=16))
    simulation.u.fill(1.0)
    simulation.v.fill(2.0)
    simulation.density.fill(3.0)
    simulation.reset()

    assert not simulation.u.any()
    assert not simulation.v.any()
    assert not simulation.density.any()


def test_help_does_not_start_gui() -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
