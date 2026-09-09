import argparse
import os
import sys
import time
from copy import deepcopy

# On a headless machine (no monitor) we cannot open an OpenGL window, so select
# the EGL offscreen backend before importing mujoco. This must happen before the
# first `import mujoco` for it to take effect.
if "--headless" in sys.argv or "-H" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from hydrax import ROOT
from hydrax.algs import (
    DAC as DAC,
    MPPI as MPPI,
    NUTS as NUTS,
    PredictiveSampling,
)
from hydrax.simulation.asynchronous import run_interactive as run_async
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.humanoid_standup import HumanoidStandup
from hydrax.utils.video import VideoRecorder

"""
Run an interactive simulation of the humanoid standup task.
"""


def run_headless(
    controller,
    mj_model,
    mj_data,
    duration,
    frequency=50,
    width=720,
    height=480,
):
    """Run the simulation without a viewer and save a video.

    This mirrors `hydrax.simulation.deterministic.run_interactive`, but does not
    open a MuJoCo viewer window, so it works on machines with no display/monitor.
    It renders offscreen (requires an offscreen GL backend such as EGL) and
    always saves a video to `hydrax/recordings/`.

    Args:
        controller: The controller instance (includes the task definition).
        mj_model: The MuJoCo model to use for simulation.
        mj_data: A MuJoCo data object containing the initial system state.
        duration: How long to simulate, in seconds.
        frequency: The requested control (replanning) frequency in Hz.
        width: Width of the recorded video in pixels.
        height: Height of the recorded video in pixels.
    """
    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt

    # Create a data structure for the controller to run rollouts from.
    mjx_data = controller.task.make_data()
    mjx_data = mjx_data.replace(
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat,
    )

    # Initialize and warm up the controller
    policy_params = controller.init_params()
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    print("Jitting the controller...")
    st = time.time()
    policy_params, _ = jit_optimize(mjx_data, policy_params)
    policy_params, _ = jit_optimize(mjx_data, policy_params)
    print(f"Time to jit: {time.time() - st:.3f} seconds")

    # Set up the offscreen renderer and video recorder
    mj_model.vis.global_.offwidth = width
    mj_model.vis.global_.offheight = height
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    recorder = VideoRecorder(
        output_dir=os.path.join(ROOT, "recordings"),
        width=width,
        height=height,
        fps=actual_frequency,
    )
    if not recorder.start():
        print("Could not start the video recorder; aborting headless run.")
        renderer.close()
        return

    num_replans = max(int(duration / step_dt), 1)
    print(
        f"Running headless for {duration:.1f}s "
        f"({num_replans} steps at {actual_frequency:.1f} Hz)..."
    )
    for step in range(num_replans):
        # Set the start state for the controller
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )

        # Do a replanning step
        policy_params, _ = jit_optimize(mjx_data, policy_params)

        # Query the control spline at the sim frequency
        sim_dt = mj_model.opt.timestep
        tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + mj_data.time
        knots = policy_params.mean[None, ...]
        us = np.asarray(jit_interp_func(tq, policy_params.tk, knots))[0]

        # Simulate the system between replanning steps
        for i in range(sim_steps_per_replan):
            mj_data.ctrl[:] = np.array(us[i])
            mujoco.mj_step(mj_model, mj_data)

        # Render and record one frame per replanning step
        renderer.update_scene(mj_data)
        recorder.add_frame(renderer.render().tobytes())

        print(f"Step {step + 1}/{num_replans}", end="\r")

    print("")
    recorder.stop()
    renderer.close()


def _ground_penetration(mj_data, floor_id):
    """Deepest penetration (>=0 m) of any geom into the floor plane."""
    max_pen = 0.0
    for i in range(mj_data.ncon):
        c = mj_data.contact[i]
        if floor_id in (c.geom1, c.geom2) and c.dist < 0.0:
            max_pen = max(max_pen, -c.dist)
    return max_pen


def randomize_initial_pose(mj_model, mj_data, rng, max_attempts=100, tol=1e-3):
    """Set a random fallen base pose that does not clip the ground plane.

    Rejection sampling: repeatedly sample a random base orientation/height
    (joints stay at the standing keyframe) and keep the first pose whose geoms
    do not penetrate the floor, checked via mj_forward + contact distances.
    """
    floor_id = mj_model.geom("floor").id
    stand_qpos = mj_model.keyframe("stand").qpos
    penetration = 0.0
    for attempt in range(1, max_attempts + 1):
        mj_data.qpos[:] = stand_qpos

        # Random tilt about a random horizontal axis (which way it falls) ...
        fall_dir = rng.uniform(0.0, 2.0 * np.pi)
        axis = np.array([np.cos(fall_dir), np.sin(fall_dir), 0.0])
        angle = rng.uniform(np.deg2rad(45.0), np.deg2rad(135.0))
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, axis, angle)
        mj_data.qpos[3:7] = quat

        # ... and a random drop height.
        mj_data.qpos[2] = rng.uniform(0.3, 0.79)

        mujoco.mj_forward(mj_model, mj_data)
        penetration = _ground_penetration(mj_data, floor_id)
        if penetration <= tol:
            print(f"Sampled clip-free initial pose in {attempt} attempt(s).")
            return

    # Fallback: lift the base by the last measured penetration so the (flat)
    # floor is cleared, rather than looping forever.
    mj_data.qpos[2] += penetration + tol
    mujoco.mj_forward(mj_model, mj_data)
    print(
        f"Warning: no clip-free pose in {max_attempts} tries; "
        f"lifted base by {penetration + tol:.3f} m."
    )


# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run an interactive simulation of humanoid (G1) standup."
    )
    parser.add_argument(
        "-a",
        "--asynchronous",
        action="store_true",
        help="Use asynchronous simulation",
        default=False,
    )
    parser.add_argument(
        "--warp",
        action="store_true",
        help="Whether to use the (experimental) MjWarp backend.",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--save-video",
        action="store_true",
        help="Save a video of the simulation to hydrax/recordings/ "
        "(deterministic mode only).",
        default=False,
    )
    parser.add_argument(
        "-H",
        "--headless",
        action="store_true",
        help="Run without a viewer (no monitor required) and save a video "
        "to hydrax/recordings/. Uses offscreen (EGL) rendering.",
        default=False,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="How long to simulate in headless mode, in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for initial-pose randomization. "
        "Omit for a new random pose each run.",
    )
    args = parser.parse_args()

    # Define the task (cost and dynamics)
    task = HumanoidStandup(impl="warp" if args.warp else "jax")

    samples = 128
    knot_dt = 0.15
    knots = 4
    # Set up the controller
    ctrl = DAC(
        task,
        num_samples=samples,
        noise_level=0.3,
        temperature=0.1,
        num_randomizations=4,
        plan_horizon=knot_dt * knots,
        spline_type="zero",
        num_knots=knots,
        num_segments=3,
    )
    ctrl = MPPI(
        task,
        num_samples=samples,
        noise_level=0.3,
        temperature=0.1,
        num_randomizations=4,
        plan_horizon=knots * knot_dt,
        spline_type="zero",
        num_knots=knots,
    )
    ctrl = PredictiveSampling(
        task,
        num_samples=samples,
        noise_level=0.3,
        num_randomizations=4,
        plan_horizon=knots * knot_dt,
        spline_type="zero",
        num_knots=knots,
    )
    # NUTS samples the Boltzmann cost distribution with gradient-based MCMC, so
    # it needs the differentiable "jax" MJX backend (it will raise if the task
    # was built with --warp, whose rollout is not differentiable). It is also
    # far heavier per step than the others, hence only a handful of chains.
    ctrl = NUTS(
        task,
        num_samples=8,
        temperature=0.1,
        step_size=0.01,
        num_mcmc_steps=4,
        num_warmup=1,
        max_num_doublings=3,
        init_noise=0.3,
        num_randomizations=4,
        plan_horizon=knots * knot_dt,
        spline_type="zero",
        num_knots=knots,
    )

    # Define the model used for simulation (stiffer contact parameters)
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = 0.01
    mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

    # Set the initial state so the robot falls and needs to stand back up.
    # The base orientation and height are randomized so each run starts from a
    # different fallen pose; the joints stay at the standing keyframe.
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = mj_model.keyframe("stand").qpos

    seed = (
        args.seed if args.seed is not None else np.random.SeedSequence().entropy
    )
    rng = np.random.default_rng(seed)
    print(f"Initial-pose seed: {seed}")

    # Rejection-sample the random pose so nothing clips through the ground.
    randomize_initial_pose(mj_model, mj_data, rng)

    # Run the simulation
    if args.headless:
        print("Running headless simulation")
        run_headless(
            ctrl,
            mj_model,
            mj_data,
            duration=args.duration,
            frequency=50,
        )
    elif args.asynchronous:
        print("Running asynchronous simulation")

        if args.save_video:
            print(
                "Warning: video saving is only supported in deterministic "
                "mode; ignoring --save-video."
            )

        # Tighten up the simulator parameters, since it's running on CPU and
        # therefore won't slow down the planner
        mj_model.opt.timestep = 0.005
        mj_model.opt.iterations = 100
        mj_model.opt.ls_iterations = 50
        mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

        run_async(
            ctrl,
            mj_model,
            mj_data,
        )
    else:
        print("Running deterministic simulation")
        run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            show_traces=False,
            record_video=args.save_video,
        )
