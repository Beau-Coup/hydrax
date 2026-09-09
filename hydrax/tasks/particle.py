from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task

# Add some stuff for LQR to be able to compute a "value function"


@dataclass
class LQRModel:
    A: jax.Array
    B: jax.Array
    Q: jax.Array
    R: jax.Array


def lqr_from_model(
    model: mujoco.MjModel, control_penalty: float = 1.0
) -> LQRModel:
    mass = float(model.body_mass[model.body("pointmass").id])
    damping = model.dof_damping
    gear = model.actuator_gear

    a = jnp.zeros((4, 4)).at[([0, 1], [2, 3])].set(1.0)
    a = a.at[([2, 3], [2, 3])].set(-jnp.asarray(damping[:2]) / mass)
    b = jnp.zeros((4, 2))
    b = b.at[([2, 3], [0, 1])].set(jnp.asarray(gear[:2, 0]) / mass)

    return LQRModel(A=a, B=b, Q=jnp.eye(4), R=jnp.eye(2) * control_penalty)


def solve_lqr(model: LQRModel) -> jax.Array:

    A = model.A
    B = model.B
    Q = model.Q
    R = model.R

    def _step(carry, _):
        p = carry
        y = B.T @ p @ A
        res = jnp.linalg.solve(R + B.T @ p @ B, y)
        new_p = Q + A.T @ p @ A - y.T @ res
        return new_p, None

    solved_P, _ = jax.lax.scan(_step, jnp.eye(4), length=1000)
    return solved_P


def discretize(model: LQRModel, dt: jax.Array = jnp.array(0.01)) -> LQRModel:
    a_disc = jax.scipy.linalg.expm(model.A * dt)
    b_disc = (
        dt
        * (
            jnp.eye(model.A.shape[0])
            + model.A * dt * 0.5
            + model.A @ model.A * dt * dt / 6.0
        )
        @ model.B
    )
    return LQRModel(A=a_disc, B=b_disc, Q=model.Q * dt, R=model.R * dt)


class Particle(Task):
    """A velocity-controlled planar point mass chases a target position."""

    def __init__(self, lqr: LQRModel | None = None, impl: str = "jax") -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/particle/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["pointmass"], impl=impl)
        self.pointmass_id = mj_model.site("pointmass").id

        if lqr is None:
            lqr = lqr_from_model(mj_model)
        self.lqr = discretize(lqr, jnp.array(self.dt))
        self.p = solve_lqr(self.lqr)

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) encourages target tracking."""
        x = jnp.concatenate(
            [
                state.site_xpos[self.pointmass_id, :2] - state.mocap_pos[0, :2],
                state.qvel,
            ]
        )
        state_cost = x.T @ self.lqr.Q @ x
        control_cost = control.T @ self.lqr.R @ control

        return state_cost + control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        x = jnp.concatenate(
            [
                state.site_xpos[self.pointmass_id, :2] - state.mocap_pos[0, :2],
                state.qvel,
            ]
        )
        return x.T @ self.p @ x

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomly perturb the actuator gains."""
        multiplier = jax.random.uniform(
            rng, self.model.actuator_gainprm[:, 0].shape, minval=0.9, maxval=1.1
        )
        new_gains = self.model.actuator_gainprm[:, 0] * multiplier
        new_gains = self.model.actuator_gainprm.at[:, 0].set(new_gains)
        return {"actuator_gainprm": new_gains}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly shift the measured particle position."""
        shift = jax.random.uniform(rng, (2,), minval=-0.01, maxval=0.01)
        return {"qpos": data.qpos + shift}
