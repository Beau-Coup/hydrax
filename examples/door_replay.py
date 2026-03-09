import os

import minari

os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
import mediapy as media
import mujoco
from mujoco import mjx


class MJXEnv:
    def __init__(self, xml_path: str):
        """
        Initializes a mirrored MuJoCo and MJX environment from an XML file.

        Args:
            xml_path: Path to the MuJoCo MJCF (.xml) file.
        """
        # 1. Load the standard MuJoCo model (CPU)
        # This is required for rendering and as a template for MJX
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)

        # 2. Initialize standard MuJoCo data (CPU)
        # Useful for syncing states back for visualization
        self.mj_data = mujoco.MjData(self.mj_model)

        # 3. Transfer the model to MJX (GPU/TPU/JAX)
        # mjx.put_model optimizes the model structure for XLA/JAX operations
        self.mjx_model = mjx.put_model(self.mj_model)

        # 4. Initialize MJX data (GPU/TPU/JAX)
        # This creates the JAX-native state structure
        self.mjx_data = mjx.make_data(self.mj_model)

    def _replay_action(self, state, action):
        pass

    def _replay_step(self, state, action):
        pass

    def step(self, state, action):
        state = mjx.step(self.data, action)
        return state


def main():
    # 1. Load Dataset using Minari
    # Note: 'D4RL/door/human-v2' is the Minari ID for the classic D4RL door dataset
    dataset_id = "D4RL/door/human-v2"

    print(f"Loading Minari dataset: {dataset_id}...")
    try:
        dataset = minari.load_dataset(dataset_id)
    except FileNotFoundError:
        print("Dataset not found locally. Downloading...")
        minari.download_dataset(dataset_id)
        dataset = minari.load_dataset(dataset_id)

    # 2. Setup MuJoCo and MJX
    # We recover the environment to get the exact MJX-compatible model
    env = dataset.recover_environment()
    mj_model = env.unwrapped.model
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)

    # 3. Extract Trajectory Data
    # Minari datasets are organized by episodes. We'll take the first one.
    episode = dataset.iterate_episodes().__next__()

    # In Minari (D4RL-style), qpos and qvel are stored in 'infos'
    qpos_raw = jnp.array(episode.infos["qpos"])
    qvel_raw = jnp.array(episode.infos["qvel"])
    num_steps = qpos_raw.shape[0]

    # 4. Define JAX/MJX Replay Function
    @jax.jit
    def replay_step(qpos, qvel):
        """Pure MJX function to set state and compute forward kinematics."""
        d = mjx.make_data(mjx_model)
        d = d.replace(qpos=qpos, qvel=qvel)
        # Updates global positions (xpos) and contact points
        return mjx.forward(mjx_model, d)

    # Use vmap to process the entire episode in parallel on the GPU
    print(f"Replaying {num_steps} steps in MJX...")
    replayed_batch = jax.vmap(replay_step)(qpos_raw, qvel_raw)

    # 5. Rendering with Mediapy
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames = []

    print("Rendering video...")
    for i in range(num_steps):
        # Slice one state from our replayed batch
        # We use jax.tree_util to handle the MJX Data structure
        state = jax.tree_util.tree_map(lambda x: x[i], replayed_batch)

        # Sync GPU state back to CPU MjData for the renderer
        mjx.get_data(mj_model, mj_data, state)

        # Update scene and render
        renderer.update_scene(mj_data, camera="fixed")
        frames.append(renderer.render())

    # 6. Save/Show Video
    output_path = "adroit_minari_mjx.mp4"
    media.write_video(output_path, frames, fps=30)
    print(f"Success! Video saved to {output_path}")


if __name__ == "__main__":
    main()
