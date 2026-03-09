import argparse
from copy import deepcopy

import mujoco

from hydrax.algs import CEM, MPPI, PredictiveSampling
from hydrax.simulation.asynchronous import run_interactive as run_async
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.adroit_hammer import AdroitHammer

"""
Run an interactive simulation of the humanoid standup task.
"""

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

    subparsers = parser.add_subparsers(
        dest="algorithm", help="Sampling algorithm (choose one)"
    )
    subparsers.add_parser("ps", help="Predictive Sampling")
    subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
    subparsers.add_parser("cem", help="Cross-Entropy Method")

    args = parser.parse_args()

    # Define the task (cost and dynamics)
    task = AdroitHammer(impl="warp" if args.warp else "jax")

    # Set up the controller
    if args.algorithm == "ps" or args.algorithm is None:
        print("Running predictive sampling")
        ctrl = PredictiveSampling(
            task,
            num_samples=128,
            noise_level=0.3,
            spline_type="zero",
            plan_horizon=0.6,
            num_knots=4,
        )
    elif args.algorithm == "mppi":
        print("Running MPPI")
        ctrl = MPPI(
            task,
            num_samples=128,
            noise_level=0.3,
            temperature=0.1,
            num_randomizations=4,
            spline_type="zero",
            plan_horizon=0.6,
            num_knots=4,
        )
    elif args.algorithm == "cem":
        print("Running CEM")
        ctrl = CEM(
            task,
            num_samples=128,
            num_elites=3,
            sigma_start=0.5,
            sigma_min=0.1,
            spline_type="zero",
            plan_horizon=0.6,
            num_knots=4,
        )
    else:
        parser.error("Other algorithms not implemented for this example!")

    # Define the model used for simulation
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = 0.01
    mj_data = mujoco.MjData(mj_model)

    # Run the interactive simulation
    if args.asynchronous:
        print("Running asynchronous simulation")

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
        )
