import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.algs.dac import DAC
from hydrax.tasks.pendulum import Pendulum


@pytest.mark.parametrize("num_segments", [1, 2, 3])
def test_open_loop(num_segments: int) -> None:
    """Use divide-and-conquer MPPI for open-loop pendulum swingup."""
    task = Pendulum()
    opt = DAC(
        task,
        num_samples=32,
        noise_level=0.1,
        temperature=0.01,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
        num_segments=num_segments,
    )

    # The horizon's knots and control steps are split across the k segments.
    assert sum(opt.seg_knots) == opt.num_knots
    assert len(opt.seg_knots) == num_segments
    assert opt.seg_steps == opt.ctrl_steps // num_segments

    jit_opt = jax.jit(opt.optimize)

    state = mjx.make_data(task.model)
    params = opt.init_params()

    for _ in range(100):
        params, _ = jit_opt(state, params)

    # The warm-started spline covers the full horizon with num_knots knots.
    assert params.mean.shape == (opt.num_knots, task.model.nu)

    knots = params.mean[None]
    tk = jnp.linspace(0.0, opt.plan_horizon, opt.num_knots)
    tq = jnp.linspace(0.0, opt.plan_horizon - opt.dt, opt.ctrl_steps)
    controls = opt.interp_func(tq, tk, knots)

    # Roll out the solution, check that it's good enough
    _, final_rollout = jax.jit(opt.eval_rollouts)(
        task.model, state, controls, knots
    )
    total_cost = jnp.sum(final_rollout.costs[0])
    assert total_cost <= 9.0


def test_invalid_num_segments() -> None:
    """num_segments must be positive and no larger than num_knots."""
    task = Pendulum()
    kwargs = dict(
        num_samples=8,
        noise_level=0.1,
        temperature=0.01,
        num_knots=4,
    )
    with pytest.raises(ValueError):
        DAC(task, num_segments=0, **kwargs)
    with pytest.raises(ValueError):
        DAC(task, num_segments=5, **kwargs)  # more segments than knots


if __name__ == "__main__":
    for k in (1, 2, 3):
        test_open_loop(k)
    test_invalid_num_segments()
