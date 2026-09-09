import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.algs.nuts import NUTS
from hydrax.tasks.pendulum import Pendulum


def test_open_loop() -> None:
    """Use NUTS Boltzmann sampling for open-loop pendulum swingup."""
    task = Pendulum()
    opt = NUTS(
        task,
        num_samples=4,
        temperature=0.1,
        step_size=0.05,
        num_mcmc_steps=6,
        num_warmup=2,
        init_noise=0.2,
        max_num_doublings=3,
        plan_horizon=1.0,
        spline_type="zero",
        num_knots=11,
    )

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


def test_invalid_warmup() -> None:
    """num_warmup must be in [0, num_mcmc_steps) and num_mcmc_steps positive."""
    task = Pendulum()
    kwargs = dict(
        num_samples=4,
        temperature=0.1,
        step_size=0.05,
        num_knots=4,
    )
    with pytest.raises(ValueError):
        NUTS(task, num_mcmc_steps=0, **kwargs)
    with pytest.raises(ValueError):
        NUTS(task, num_mcmc_steps=3, num_warmup=3, **kwargs)  # warmup >= steps


if __name__ == "__main__":
    test_open_loop()
    test_invalid_warmup()
