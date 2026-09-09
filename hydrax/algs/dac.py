from typing import Any, Callable, Literal, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task


@dataclass
class DACParams(SamplingParams):
    """Policy parameters for divide-and-conquer path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """


def uniform_band_probs(costs: jax.Array, band: float) -> jax.Array:
    """Uniform probabilities over the costs within ``band`` of the minimum cost.

    Args:
        costs: The per-sample costs, (num_samples,).
        band: The width of the band above the lowest cost, in raw cost units.

    Returns:
        Probabilities that are equal for every in-band sample and zero for the
        rest, (num_samples,).
    """
    costs = jnp.nan_to_num(costs, nan=jnp.inf, posinf=jnp.inf)
    min_cost = jnp.min(costs)
    # Everything within the band is equally likely; the minimum is always in
    # the band, so at least one sample has non-zero probability.
    mask = costs <= min_cost + band
    # FIXME: even with uniform noise, the probabilities of optimality are not uniform.
    return mask / jnp.sum(mask)


class DAC(SamplingBasedController):
    """Divide-and-conquer path integral control.

    The planning horizon is split into ``num_segments`` (``k``) equal segments,
    processed as a sequential chain. The first segment is optimized from the
    current state; the terminal states of those rollouts are then resampled
    according to a Boltzmann distribution over their costs, and the next segment
    is optimized starting from those distinct states. This repeats at each of
    the ``k - 1`` interior boundaries, and the per-segment means are concatenated
    into the full-horizon mean.

    ``k = 1`` recovers plain MPPI; ``k = 2`` splits the horizon in half.
    """

    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        num_segments: int = 2,
        resample_band: float = 1.0,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
            num_segments: The number of segments k to divide the horizon into.
                          k=1 is plain MPPI, k=2 splits the horizon in half.
            resample_band: The width of the band used to resample terminal
                           states at segment boundaries. Absolute, in raw cost
                           units (the summed cost of a segment), and therefore
                           task-dependent: too small collapses onto the single
                           best sample, too large degenerates to uniform
                           resampling over every sample.
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
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature
        self.resample_band = resample_band

        if num_segments < 1:
            raise ValueError("num_segments must be greater than 0!")
        if resample_band <= 0:
            raise ValueError(
                f"resample_band must be positive, got {resample_band}."
            )
        if num_knots < num_segments:
            raise ValueError(
                "num_knots must be >= num_segments (need at least one knot per "
                f"segment), got num_knots={num_knots}, "
                f"num_segments={num_segments}."
            )

        # Split the horizon (and its knots / control steps) into k segments.
        # Distribute any leftover knots to the later segments so that k=2
        # matches the previous first/second half split (e.g. 5 knots -> [2, 3]).
        self.num_segments = num_segments
        base, rem = divmod(num_knots, num_segments)
        self.seg_knots = [
            base + (1 if i >= num_segments - rem else 0)
            for i in range(num_segments)
        ]
        self.seg_steps = self.ctrl_steps // num_segments

    def init_params(
        self, initial_knots: jax.Array | None = None, seed: int = 0
    ) -> DACParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        return DACParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def optimize(self, state: mjx.Data, params: Any) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Segment boundaries along the (advanced) planning horizon.
        t0 = state.time
        boundaries = t0 + jnp.linspace(
            0.0, self.plan_horizon, self.num_segments + 1
        )

        # Knot times for each segment. Interior segments drop their shared right
        # endpoint (it is the next segment's left knot) so the concatenation
        # stays strictly monotonic and has no duplicate knot times.
        tks = []
        for i, n in enumerate(self.seg_knots):
            if i < self.num_segments - 1:
                tk = jnp.linspace(boundaries[i], boundaries[i + 1], n + 1)[:-1]
            else:
                tk = jnp.linspace(boundaries[i], boundaries[i + 1], n)
            tks.append(tk)
        new_tk = jnp.concatenate(tks)

        # Warm-start the spline by re-evaluating the old spline at the new knot
        # times, then split the mean into per-segment knots.
        new_mean = self.interp_func(new_tk, params.tk, params.mean[None, ...])[
            0
        ]
        means = []
        offset = 0
        for n in self.seg_knots:
            means.append(new_mean[offset : offset + n])
            offset += n

        def broadcast_leaf(x):
            return jnp.broadcast_to(
                x, (self.num_randomizations, self.num_samples) + x.shape
            )

        carry = jax.tree_util.tree_map(
            broadcast_leaf, self._get_state_carry(state)
        )

        def rollout_fn(carry, tk, knots, rng):
            return self.rollout_from_states(carry, state, tk, knots, rng)

        # Optimize each segment in sequence, resampling the terminal states at
        # every interior boundary before handing them to the next segment.
        seg_means = []
        rollouts = None
        for i in range(self.num_segments):
            params = params.replace(tk=tks[i], mean=means[i])
            params, rollouts, final_states = self._run(
                params, tks[i], self.seg_knots[i], rollout_fn, carry
            )
            # params.mean is the cost-weighted average knots for this segment
            # (update_params, as in MPPI).
            seg_means.append(params.mean)
            if i < self.num_segments - 1:
                params, carry = self._resample_carry(
                    params, rollouts, final_states
                )

        # Concatenate the per-segment means into the full-horizon mean, which is
        # warm-started next step and queried by get_action.
        full_mean = jnp.concatenate(seg_means, axis=0)
        params = params.replace(tk=new_tk, mean=full_mean)

        return params, rollouts

    def _resample_carry(
        self,
        params: DACParams,
        rollouts: Trajectory,
        final_states: Any,
    ) -> Tuple[DACParams, Any]:
        """Resample terminal states uniformly over a band of the best cost.

        Every terminal state whose segment cost is within ``resample_band`` of
        the lowest one is equally likely to seed the next segment's rollouts;
        the rest are discarded. Uses systematic (low-variance) resampling.

        Args:
            params: The current policy parameters (its rng is advanced).
            rollouts: The finished segment's rollouts, costs (num_samples, H+1).
            final_states: The segment's terminal state carry,
                (num_randomizations, num_samples, ...).

        Returns:
            Updated policy parameters (new rng).
            The resampled terminal state carry.
        """
        per_sample_cost = jnp.sum(rollouts.costs, axis=-1)
        probs = uniform_band_probs(per_sample_cost, self.resample_band)

        rng, resample_rng = jax.random.split(params.rng)
        params = params.replace(rng=rng)

        start_locs = (
            jnp.linspace(0, 1, num=self.num_samples, endpoint=False)
            + jax.random.uniform(resample_rng) / self.num_samples
        )
        choices = jnp.searchsorted(
            jnp.cumsum(probs), start_locs, method="scan_unrolled"
        )

        def _data_map(data):
            # Resample along the sample axis (axis 1); leaves are
            # (num_randomizations, num_samples, ...).
            if data.ndim < 2:
                return data
            return data[:, choices]

        carry = jax.tree.map(_data_map, final_states)

        return params, carry

    def _run(
        self,
        params: DACParams,
        tk: jax.Array,
        num_knots: int,
        rollout_fn: Callable,
        state: mjx.Data,
    ) -> Tuple[DACParams, Trajectory, Any]:
        knots, params = self.sample_knots(params, num_knots)
        knots = jnp.clip(knots, self.task.u_min, self.task.u_max)

        rng, dr_rng = jax.random.split(params.rng)
        rollouts, final_states = rollout_fn(state, tk, knots, dr_rng)
        params = params.replace(rng=rng)

        params = self.update_params(params, rollouts)

        return params, rollouts, final_states

    # Fields that fully define the resettable simulator state. We pass these
    # (rather than the full mjx.Data) between phases so the MjWarp backend never
    # receives a per-sample batched `_impl` buffer as a vmapped input; the
    # solver buffers are scratch that mjx.step recomputes from qpos each step.
    _STATE_FIELDS = ("qpos", "qvel", "act", "time", "mocap_pos", "mocap_quat")

    def _get_state_carry(self, data: mjx.Data) -> dict[str, jax.Array]:
        """Extract the resettable state fields from an mjx.Data."""
        return {f: getattr(data, f) for f in self._STATE_FIELDS}

    def rollout_from_states(
        self,
        carry: dict[str, jax.Array],
        base: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        rng: jax.Array,
    ) -> Tuple[Trajectory, Any]:
        """Roll out from per-sample start states.

        Args:
            carry: Resettable state fields, batched
                (num_randomizations, num_samples, ...).
            base: A single (unbatched) mjx.Data providing the buffer shapes and
                non-state fields; the carry fields are written onto it.
            tk: The knot times of the control spline, (num_knots,).
            knots: The control spline knots, (num_samples, num_knots, nu).
            rng: Unused.

        Returns:
            A Trajectory with costs combined over domains via the risk strategy.
            The terminal state carry, (num_randomizations, num_samples, ...).
        """
        tq = jnp.linspace(tk[0], tk[-1], self.seg_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_samples, H, nu)

        def _rollout_sample(
            model: mjx.Model,
            sample_carry: dict[str, jax.Array],
            sample_controls: jax.Array,
            sample_knots: jax.Array,
        ) -> Tuple[mjx.Data, Trajectory]:
            # Reconstruct a full Data from the shared base + this sample's
            # state. `base` is unbatched (shared), so its `_impl` buffers are
            # never fed to warp as a batched input.
            state = base.replace(**sample_carry)
            _, final_state, rollouts = self._eval_rollout(
                model, state, sample_controls, sample_knots
            )
            return final_state, rollouts

        def _rollout_domain(
            model: mjx.Model, domain_carry: dict[str, jax.Array]
        ) -> Tuple[mjx.Data, Trajectory]:
            # domain_carry: (num_samples, ...), one randomized model.
            final_states, rollouts = jax.vmap(
                _rollout_sample, in_axes=(None, 0, 0, 0)
            )(model, domain_carry, controls, knots)
            return final_states, rollouts

        # Outer vmap over domain randomizations (model + per-domain start
        # states), inner vmap over samples.
        final_states, rollouts = jax.vmap(
            _rollout_domain, in_axes=(self.randomized_axes, 0)
        )(self.model, carry)
        # final_states: (num_randomizations, num_samples, ...)
        # rollouts.costs: (num_randomizations, num_samples, H+1)

        rollouts = self._combine_rollouts(rollouts)  # costs (num_samples, H+1)
        final_carry = self._get_state_carry(final_states)

        return rollouts, final_carry

    def _combine_rollouts(self, rollouts: Trajectory) -> Trajectory:
        """Combine per-domain rollout costs using the risk strategy."""
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        controls = rollouts.controls[0]  # identical over randomizations
        knots = rollouts.knots[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st
        return rollouts.replace(
            costs=costs, controls=controls, knots=knots, trace_sites=trace_sites
        )

    def _eval_rollout(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
    ) -> Tuple[mjx.Data, mjx.Data, Trajectory]:
        """Roll out a single control sequence and compute its costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x0 (unbatched).
            controls: The control sequence, (H, nu).
            knots: The control spline knots, (num_knots, nu).

        Returns:
            The states (stacked) experienced during the rollout.
            A Trajectory object containing the control, costs, and trace sites.
        """

        def _step(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            x = x.replace(ctrl=u)
            x = mjx.step(model, x)  # step model + compute site positions
            cost = self.dt * self.task.running_cost(x, u)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, sites)

        final_state, (states, costs, trace_sites) = jax.lax.scan(
            _step, state, controls
        )
        final_cost = self.task.terminal_cost(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return (
            states,
            final_state,
            Trajectory(
                controls=controls,
                knots=knots,
                costs=costs,
                trace_sites=trace_sites,
            ),
        )

    def sample_knots(
        self, params: DACParams, num_knots: int
    ) -> Tuple[jax.Array, DACParams]:
        """Sample control knots from a Normal centered at the mean (MPPI-style).

        The mean has shape (num_knots, nu) for the current half of the horizon.
        """
        rng, sample_rng = jax.random.split(params.rng)

        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                num_knots,
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * noise

        return controls, params.replace(rng=rng)

    def update_params(
        self, params: DACParams, rollouts: Trajectory
    ) -> DACParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)

        return params.replace(mean=mean)
