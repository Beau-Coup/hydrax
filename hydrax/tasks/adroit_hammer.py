import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class AdroitHammer(Task):
    """Door open task for Adroit hand."""

    def __init__(self, impl: str = "jax") -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/adroit_hand/adroit_hammer.xml"
        )
        # The default is a timestep of 0.002 and 50 ls_iterations
        # self.frame_skip = 5 is the classic env setting
        # self.dt = 0.01
        # self.

        self._model_names = MujocoModelNames(mj_model)
        mj_model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"] + 1,
            :3,
        ] = np.array([10, 0, 0])
        mj_model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"] + 1,
            :3,
        ] = np.array([1, 0, 0])
        mj_model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"] + 1,
            :3,
        ] = np.array([0, -10, 0])
        mj_model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"] + 1,
            :3,
        ] = np.array([0, -1, 0])

        super().__init__(mj_model, trace_sites=["tool"], impl=impl)

        self.sparse_reward = False
        self.target_obj_site_id = self._model_names.site_name2id["S_target"]
        self.grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.obj_body_id = self._model_names.body_name2id["Object"]
        self.tool_site_id = self._model_names.site_name2id["tool"]
        self.goal_site_id = self._model_names.site_name2id["nail_goal"]
        self.target_body_id = self._model_names.body_name2id["nail_board"]

        # self.act_mean = jnp.mean(self.model.actuator_ctrlrange, axis=1)
        # self.act_rng = jnp.array(
        #     0.5
        #     * (
        #         self.model.actuator_ctrlrange[:, 1]
        #         - self.model.actuator_ctrlrange[:, 0]
        #     )
        # )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        hamm_pos = state.xpos[self.obj_body_id].ravel()
        palm_pos = state.site_xpos[self.grasp_site_id].ravel()
        head_pos = state.site_xpos[self.tool_site_id].ravel()
        nail_pos = state.site_xpos[self.target_obj_site_id].ravel()
        goal_pos = state.site_xpos[self.goal_site_id].ravel()

        goal_distance = jnp.linalg.norm(nail_pos - goal_pos)
        goal_achieved = goal_distance < 0.01
        reward = jnp.where(goal_achieved, 10.0, -0.1)

        # override reward if not sparse reward
        if not self.sparse_reward:
            # get the palm to the hammer handle
            reward = 0.1 * jnp.linalg.norm(palm_pos - hamm_pos)
            # take hammer head to nail
            reward = reward - jnp.linalg.norm(head_pos - nail_pos)
            # make nail go inside
            reward = reward + 10 * jnp.linalg.norm(nail_pos - goal_pos)
            # velocity penalty
            reward = reward - 1e-2 * jnp.linalg.norm(state.qvel.ravel())

            reward += jnp.where(
                jnp.logical_and(hamm_pos[2] > 0.04, head_pos[2] > 0.04), 2, 0
            )

            reward += jnp.where(goal_distance < 0.02, 25, 0)
            reward += jnp.where(goal_distance < 0.01, 75, 0)

        return reward

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    # TODO: This is where moving the board goes...
    # def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
    #     """Randomize the friction parameters."""
    #     n_geoms = self.model.geom_friction.shape[0]
    #     multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
    #     new_frictions = self.model.geom_friction.at[:, 0].set(
    #         self.model.geom_friction[:, 0] * multiplier
    #     )
    #     return {"geom_friction": new_frictions}

    # def domain_randomize_data(
    #     self, data: mjx.Data, rng: jax.Array
    # ) -> Dict[str, jax.Array]:
    #     """Randomly perturb the measured base position and velocities."""
    #     rng, q_rng, v_rng = jax.random.split(rng, 3)
    #     q_err = 0.01 * jax.random.normal(q_rng, (7,))
    #     v_err = 0.01 * jax.random.normal(v_rng, (6,))

    #     qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
    #     qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

    #     return {"qpos": qpos, "qvel": qvel}

    def make_data(self) -> mjx.Data:
        """Create a new state object with extra constraints allocated."""
        return super().make_data(naconmax=10000, njmax=200)
