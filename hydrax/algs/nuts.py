from typing import Literal, Tuple

import blackjax
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class NUTSParams(SamplingParams):
    """Policy parameters for NUTS Boltzmann sampling.

    Same as SamplingParams, but with a different name for clarity. The step size
    and inverse mass matrix are fixed constructor hyperparameters, so no extra
    sampler state needs to be carried between planning steps.

    Attributes:
        tk: The knot times of the control spline.
        mean: The applied control spline knots (the lowest-cost NUTS sample).
        rng: The pseudo-random number generator key.
    """


class NUTS(SamplingBasedController):
    """Sample the Boltzmann cost distribution directly with NUTS.

    Every other sampling-based controller in hydrax is gradient-free: it
    perturbs a mean control spline with Gaussian noise and reweights the
    rollouts by the Boltzmann weight wᵢ ∝ exp(-J(Uᵢ)/λ). This controller instead
    draws samples *from* the Boltzmann distribution p(U) ∝ exp(-J(U)/λ) using
    the No-U-Turn Sampler (NUTS) from ``blackjax``. Because the MJX rollout is
    fully differentiable, exact gradients of the trajectory cost are available
    for the Hamiltonian dynamics.

    A batch of ``num_samples`` parallel NUTS chains is advanced for
    ``num_mcmc_steps`` steps each; the first ``num_warmup`` steps of every chain
    are discarded as burn-in. The remaining samples are rolled out and the
    single lowest-cost knot set is applied (a MAP-like choice, as in
    ``PredictiveSampling``).

    Note: each NUTS step takes several leapfrog steps, and every leapfrog step
    is a reverse-mode gradient through the whole rollout. This is far heavier
    per ``optimize`` call than MPPI/CEM, so keep the sampling budget modest.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        temperature: float,
        step_size: float,
        num_mcmc_steps: int = 4,
        num_warmup: int = 1,
        init_noise: float = 0.1,
        inverse_mass: float = 1.0,
        max_num_doublings: int = 6,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of parallel NUTS chains to run.
            temperature: The temperature λ of the Boltzmann target
                         p(U) ∝ exp(-J(U)/λ). Higher values flatten the target.
            step_size: The leapfrog step size for the NUTS integrator.
            num_mcmc_steps: The number of NUTS steps to take per chain.
            num_warmup: The number of leading steps discarded as burn-in. Must
                        be strictly less than num_mcmc_steps.
            init_noise: The scale of Gaussian jitter used to spread the chain
                        initializations around the warm-started mean.
            inverse_mass: The (diagonal) inverse-mass-matrix scale for NUTS.
            max_num_doublings: The maximum number of trajectory doublings per
                               NUTS step. Bounds the leapfrog steps (and hence
                               the rollout gradients) to at most 2^d, keeping
                               the per-step compute predictable for MPC.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combine costs from different randomizations
                           when scoring samples. Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.num_samples = num_samples
        self.temperature = temperature
        self.step_size = step_size
        self.num_mcmc_steps = num_mcmc_steps
        self.num_warmup = num_warmup
        self.init_noise = init_noise
        self.inverse_mass = inverse_mass
        self.max_num_doublings = max_num_doublings

        if num_mcmc_steps < 1:
            raise ValueError("num_mcmc_steps must be greater than 0!")
        if not 0 <= num_warmup < num_mcmc_steps:
            raise ValueError(
                "num_warmup must satisfy 0 <= num_warmup < num_mcmc_steps, got "
                f"num_warmup={num_warmup}, num_mcmc_steps={num_mcmc_steps}."
            )

        # NUTS differentiates through the rollout, which the MJX "jax" backend
        # supports but the MjWarp backend does not (its rollout is a
        # non-differentiable FFI call into Warp kernels). Fail early with a
        # clear message instead of a cryptic autodiff error at trace time.
        impl = getattr(self.task.model, "impl", "jax")
        impl_name = getattr(impl, "value", impl)
        if impl_name != "jax":
            raise ValueError(
                "NUTS differentiates through the rollout, which is only "
                "supported by the 'jax' MJX backend. The task uses "
                f"impl={impl_name!r}, whose rollout is a non-differentiable "
                "FFI call. Re-create the task with impl='jax' to use NUTS."
            )

    def init_params(
        self, initial_knots: jax.Array | None = None, seed: int = 0
    ) -> NUTSParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        return NUTSParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def _rollout_cost(
        self, model: mjx.Model, state: mjx.Data, controls: jax.Array
    ) -> jax.Array:
        """Total (differentiable) cost of a single control sequence.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀ (unbatched).
            controls: The control sequence, (H, nu).

        Returns:
            The scalar total cost J = Σₜ dt·ℓ(xₜ, uₜ) + ϕ(x_T).
        """

        def _step(x: mjx.Data, u: jax.Array) -> Tuple[mjx.Data, jax.Array]:
            x = x.replace(ctrl=u)
            x = mjx.step(model, x)
            return x, self.dt * self.task.running_cost(x, u)

        final_state, costs = jax.lax.scan(_step, state, controls)
        return jnp.sum(costs) + self.task.terminal_cost(final_state)

    def _total_cost(
        self,
        knots: jax.Array,
        state: mjx.Data,
        tk: jax.Array,
        tq: jax.Array,
    ) -> jax.Array:
        """Cost of one knot set, averaged over domain randomizations.

        The mean over randomizations keeps the log-density smooth and
        differentiable regardless of the configured risk strategy (the risk
        strategy is applied only when scoring the final samples).
        """
        controls = self.interp_func(tq, tk, knots[None])[0]  # (H, nu)
        if self.num_randomizations > 1:
            # axis_size is given explicitly because a task that does not
            # randomize its model leaves randomized_axes all-None, so the
            # batch size cannot be inferred from the mapped model leaves.
            costs = jax.vmap(
                self._rollout_cost,
                in_axes=(self.randomized_axes, None, None),
                axis_size=self.num_randomizations,
            )(self.model, state, controls)
            return jnp.mean(costs)
        return self._rollout_cost(self.model, state, controls)

    def optimize(
        self, state: mjx.Data, params: NUTSParams
    ) -> Tuple[NUTSParams, Trajectory]:
        """Sample the Boltzmann distribution with NUTS and apply the best knots.

        Args:
            state: The initial state x₀.
            params: The current policy parameters.

        Returns:
            Updated policy parameters (mean set to the lowest-cost sample).
            The rollouts of the sampled candidates (for visualization).
        """
        nu = self.task.model.nu
        dim = self.num_knots * nu

        # Warm-start the spline by advancing knot times by the current time and
        # re-evaluating the old spline at the new knot times (as in the base).
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(new_tk, params.tk, params.mean[None])[0]
        tq = jnp.linspace(new_tk[0], new_tk[-1], self.ctrl_steps)

        # Boltzmann log-density over flat knot vectors, p(U) ∝ exp(-J(U)/λ).
        def logdensity(z: jax.Array) -> jax.Array:
            knots = z.reshape(self.num_knots, nu)
            cost = self._total_cost(knots, state, new_tk, tq)
            return -cost / self.temperature

        kernel = blackjax.nuts(
            logdensity,
            step_size=self.step_size,
            inverse_mass_matrix=self.inverse_mass * jnp.ones(dim),
            max_num_doublings=self.max_num_doublings,
        )

        rng, init_rng, mcmc_rng, dr_rng = jax.random.split(params.rng, 4)

        # Spread chain initializations around the warm-started mean, keeping one
        # chain exactly at the mean as a warm-start anchor.
        base = new_mean.reshape(dim)
        noise = self.init_noise * jax.random.normal(
            init_rng, (self.num_samples, dim)
        )
        positions = (base[None] + noise).at[0].set(base)
        init_states = jax.vmap(kernel.init)(positions)

        # Advance all chains for num_mcmc_steps, collecting sample positions.
        def _scan_step(
            states: blackjax.mcmc.hmc.HMCState, step_rng: jax.Array
        ) -> Tuple[blackjax.mcmc.hmc.HMCState, jax.Array]:
            keys = jax.random.split(step_rng, self.num_samples)
            states, _ = jax.vmap(kernel.step)(keys, states)
            return states, states.position

        _, positions_hist = jax.lax.scan(
            _scan_step,
            init_states,
            jax.random.split(mcmc_rng, self.num_mcmc_steps),
        )
        # positions_hist: (num_mcmc_steps, num_samples, dim)

        # Drop burn-in and flatten into candidate knot sets.
        candidates = positions_hist[self.num_warmup :].reshape(
            -1, self.num_knots, nu
        )
        candidates = jnp.clip(candidates, self.task.u_min, self.task.u_max)

        # Score the candidates (respecting the risk strategy) and apply the
        # lowest-cost knot set.
        rollouts = self.rollout_with_randomizations(
            state, new_tk, candidates, dr_rng
        )
        costs = jnp.sum(rollouts.costs, axis=1)
        best = jnp.argmin(costs)
        best_knots = candidates[best]

        params = params.replace(tk=new_tk, mean=best_knots, rng=rng)
        return params, rollouts

    def sample_knots(self, params: NUTSParams) -> Tuple[jax.Array, NUTSParams]:
        """Sample control knots for interface compliance.

        ``optimize`` is overridden and drives the NUTS sampler directly, so this
        method is not used in the main planning path. It provides a coherent
        Gaussian proposal around the mean to satisfy the base-class interface.
        """
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (self.num_samples, self.num_knots, self.task.model.nu),
        )
        controls = params.mean + self.init_noise * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: NUTSParams, rollouts: Trajectory
    ) -> NUTSParams:
        """Select the lowest-cost knot set (matches the applied action)."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        best = jnp.argmin(costs)
        return params.replace(mean=rollouts.knots[best])
