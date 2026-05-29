"""Component for optimizing stimulus selection using JAX.

This module leverages the evosax library for high-performance, JAX-native
Differential Evolution. Its primary function is to determine the optimal coil
configuration for the next stimulus, aiming to maximize the expected information
gain for localizing a response source.

The core logic is JIT-compiled for performance.
"""

# Standard library imports
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import torch
from scipy.stats.qmc import Sobol

# JAX and evosax for high-performance optimization
import jax
import jax.numpy as jnp
from evosax.algorithms import DifferentialEvolution as DE
from jax.scipy.special import logsumexp

# Local application imports
from .efield import EFieldModel
from .model import ResponseModel
from .optimizer_constraints import get_no_constraints, get_5coil_strain_constraints

# Type Aliases for clarity
Tensor = torch.Tensor
Array = jnp.ndarray

# Note: Helper functions are defined at the module level in a pure functional
# style to be compatible with JAX's JIT (Just-In-Time) compilation.

def _create_objective_function(
    efield_basis_means_jax: Array,
    efield_basis_jax: Array,
    null_dir_basis_jax: Array,
    prob_lim_jax: Array,
    efield_set_jax: Array,
    max_current_slope_jax: Array,
    efield_limit: float,
    strain_func_jax: Optional[Callable[[Array], Array]],
    strain_limit: float,
    constraint_penalty: float,
    param_keys: List[str]
) -> Callable[[Array, Array, Tuple], Array]:
    """Creates and JIT-compiles the objective function for optimization.

    This factory function captures static data (like E-field bases) and returns
    a JIT-compiled function that evaluates the "goodness" of a stimulus. The
    objective is to maximize the expected change in source probabilities,
    penalizing solutions that violate hardware constraints.

    Args:
        efield_basis_means_jax: Mean E-field vectors across the region of interest.
        efield_basis_jax: PCA basis for the E-field.
        null_dir_basis_jax: Reference direction in the 2D projected subspace.
        prob_lim_jax: Min/max limits for clamping probabilities.
        efield_set_jax: The complete set of E-field maps for all coils.
        max_current_slope_jax: Maximum current slope (dI/dt) for each coil.
        efield_limit: Maximum allowed E-field strength for subject comfort.
        strain_func_jax: JAX function for calcualting strain constraints.
        strain_limit: The maximum allowed pulse strain.
        constraint_penalty: Penalty weight for violating the strain limit.
        param_keys: Ordered list of response model parameter names.

    Returns:
        A JIT-compiled and vectorized function ready for evaluating a population
        of stimulus candidates.
    """
    # JAX-compatible helper functions, defined locally to capture scope.
    def vector_angle_jax(v1: Array, v2: Array) -> Array:
        """Calculates the 2D vector angle between two batches of vectors."""
        angle_v1 = jnp.arctan2(v1[..., 1], v1[..., 0])
        angle_v2 = jnp.arctan2(v2[..., 1], v2[..., 0])
        angle_diff = angle_v2 - angle_v1
        return angle_diff % (2 * jnp.pi)

    def efield_to_subspace_jax(efield: Array) -> Array:
        """Projects a 3D E-field vector into the learned 2D subspace."""
        new_centered = efield - efield_basis_means_jax
        return jnp.einsum('vd,vdt->vt', new_centered, efield_basis_jax)

    def efield_to_mag_and_angle_jax(efield: Array) -> Tuple[Array, Array]:
        """Calculates E-field magnitude and its angle in the 2D subspace."""
        e_mag = jnp.linalg.norm(efield, axis=-1)
        e_transformed = efield_to_subspace_jax(efield)
        e_transformed_norm = e_transformed / jnp.linalg.norm(e_transformed, axis=-1, keepdims=True)
        angles = vector_angle_jax(null_dir_basis_jax, e_transformed_norm)
        return e_mag, angles

    def directional_sigmoid_jax(e_mag: Array, angles: Array, params: Dict) -> Array:
        """JAX-native implementation of the response probability model."""
        directional_translation = (
            params['rf_translate'] +
            params['rf_fourier1'] * jnp.cos(angles) +
            params['rf_fourier2'] * jnp.sin(angles) +
            params['rf_fourier3'] * jnp.cos(angles * 2) +
            params['rf_fourier4'] * jnp.sin(angles * 2)
        )
        inner_term = jnp.clip(-params['rf_slope'] * (e_mag - directional_translation), -16, 16)
        response_prob = e_mag / (e_mag + jnp.exp(inner_term))
        return jnp.clip(response_prob, prob_lim_jax[0], prob_lim_jax[1])

    def safe_log_jax(tensor: Array) -> Array:
        """JAX-native version of the clamped logarithm for numerical stability."""
        return jnp.log(jnp.clip(tensor, prob_lim_jax[0], prob_lim_jax[1]))

    def _one_step_posterior_update_jax(log_source_prob_jax: Array, param_dict_jax: Dict, e_total: Array) -> Tuple[Array, Array, Array]:
        """Calculates the posterior source probabilities for both possible trial outcomes."""
        e_mag, angles = efield_to_mag_and_angle_jax(e_total)
        responsivity = directional_sigmoid_jax(e_mag, angles, param_dict_jax)

        # Case 1: Response is y=1 (success)
        log_responsivity = safe_log_jax(responsivity)
        numerator1 = log_source_prob_jax + log_responsivity
        log_prob_y1 = logsumexp(numerator1, axis=-1)
        log_source_prob1 = numerator1 - log_prob_y1

        # Case 2: Response is y=0 (failure)
        log_one_minus_responsivity = jnp.log(-jnp.expm1(log_responsivity)) # Numerically stable log(1-x)
        numerator0 = log_source_prob_jax + log_one_minus_responsivity
        log_prob_y0 = logsumexp(numerator0, axis=-1)
        log_source_prob0 = numerator0 - log_prob_y0

        return log_source_prob0, log_source_prob1, log_prob_y1

    # Main objective function for a single candidate solution. This will be vectorized.
    def single_objective_fun(population: Array, log_source_prob_jax: Array, param_tuple_jax: Tuple):
        """Evaluates the objective function for a single stimulus configuration."""
        param_dict_jax = dict(zip(param_keys, param_tuple_jax))
        efield_normalized = efield_set_jax * max_current_slope_jax[:, None, None]
        e_total = jnp.einsum('cvd,c->vd', efield_normalized, population)

        log_post0, log_post1, log_prob_y1 = _one_step_posterior_update_jax(log_source_prob_jax, param_dict_jax, e_total)
        prob_y1 = jnp.exp(log_prob_y1)

        # Objective: Maximize expected change in source probabilities.
        source_prob = jnp.exp(log_source_prob_jax)
        expected_change = ((1 - prob_y1) * jnp.abs(source_prob - jnp.exp(log_post0)) +
                           prob_y1 * jnp.abs(source_prob - jnp.exp(log_post1))).mean()
        objective = -expected_change # Minimize negative for maximization

        penalty = 0.0
        # Add penalty for violating the strain constraint.
        if strain_func_jax is not None:
            pulse_strain = strain_func_jax(population)
            strain_violation = jax.nn.relu(pulse_strain - strain_limit)
            penalty += constraint_penalty * strain_violation**2

        # Add penalty for violating the E-field strength limit.
        efield_max = jnp.max(jnp.linalg.norm(e_total, axis=-1))
        efield_violation = jax.nn.relu(efield_max - efield_limit)
        efield_penalty = constraint_penalty * efield_violation**2
        penalty += efield_penalty

        return objective + penalty

    # Vectorize across the population for efficient, parallel evaluation.
    return jax.jit(jax.vmap(single_objective_fun, in_axes=(0, None, None)))

def _optimizer_step(carry: Tuple, _: None, strategy: DE, strategy_params: Dict, jitted_objective: Callable) -> Tuple[Tuple, Dict]:
    """Executes one step of the Differential Evolution optimization.

    This function is designed to be used with `jax.lax.scan` for efficient,
    JIT-compiled looping over optimization generations.

    Args:
        carry: A tuple containing the optimizer state, random key, and static
               data (log probabilities, model parameters).
        _: Placeholder for the scan's sequence data (unused).
        strategy: The evosax Differential Evolution strategy instance.
        strategy_params: Hyperparameters for the DE strategy.
        jitted_objective: The pre-compiled objective function.

    Returns:
        A tuple containing the updated carry and a dictionary of metrics for the step.
    """
    state, key, log_source_prob_jax, param_tuple_jax = carry
    key, key_ask, key_tell = jax.random.split(key, 3)

    # Ask the optimizer for a new population of candidate solutions.
    population, state = strategy.ask(key_ask, state, strategy_params)
    population = jnp.clip(population, -1.0, 1.0) # Ensure solutions are within [-1, 1] range

    # Evaluate the fitness of the population.
    fitness = jitted_objective(population, log_source_prob_jax, param_tuple_jax)

    # Tell the optimizer the results to update its state.
    state, metrics = strategy.tell(key_tell, population, fitness, state, strategy_params)

    return (state, key, log_source_prob_jax, param_tuple_jax), metrics


class StimulusOptimizer:
    """Determines the optimal stimulus for the next trial via numerical optimization.

    This class orchestrates the optimization process by initializing the JAX-based
    Differential Evolution algorithm, managing its state, and running the
    optimization loop to find the coil settings that maximize information gain.

    Attributes:
        device (torch.device): The PyTorch device used for tensor operations.
        jax_device (jax.Device): The JAX device used for optimization.
        losses (Tensor): Stores the best fitness from each generation of the
            last optimization run.
        optimizer_settings (Dict): Hyperparameters for the DE algorithm.
        n_coils (int): Number of stimulator coils.
        strain_func (Optional[Callable]): JAX-compatible function to calculate
            pulse strain, or None if no constraint.
        strain_limit (Optional[float]): The maximum allowed pulse strain, or
            None if no constraint.
        param_keys (List[str]): Sorted list of response model parameter names.
        max_current_slope (Tensor): The maximum dI/dt for each coil.
        efield_limit (float): Maximum allowed E-field strength for subject comfort.
        jitted_objective (Callable): The JIT-compiled objective function.
        seed (int): Seed for random number generation.
        sobol_sampler (Sobol): Sobol sequence generator for initializing the
            optimizer population.
        jitted_step (Optional[Callable]): The JIT-compiled optimizer step
            function (created on first call).
        strategy (Optional[DE]): The evosax Differential Evolution strategy
            instance (created on first call).
        strategy_params (Optional[Dict]): The evosax strategy hyperparameters
            (created on first call).
        jitted_constraint_func (Callable): The JIT-compiled function to apply
            strain constraints to a solution.
        key (jax.Array): The main JAX random number generator key for the
    optimizer instance, ensuring stochasticity between runs.
    """

    def __init__(self, efield_model: EFieldModel, response_model: ResponseModel, stimulator: Dict, device: torch.device):
        """Initializes the StimulusOptimizer and JIT-compiles its components.
        
        Args:
            efield_model: The E-field physics model component.
            response_model: The statistical response model component.
            stimulator: A dictionary containing stimulator coil parameters.
            device: The PyTorch device ('cpu' or 'cuda') for computation.
        """
        self.device = device
        # Set JAX device based on the PyTorch device
        if device.type == 'cpu':
            self.jax_device = jax.devices('cpu')[0]
        else:
            self.jax_device = jax.devices()[0] # Default to first available device (usually GPU)
        self.losses: Tensor = torch.tensor([0])

        # Tuned hyperparameters for Differential Evolution
        self.optimizer_settings = {
            'popsize': 512,
            'num_generations': 200,
            'constraint_penalty': 1e2,
            'crossover_rate': 0.09,
            'differential_weight': 0.812	
        }

        self.n_coils = efield_model.efield_set.shape[0]
        
        # Set stimulator constraints if specified
        match stimulator["constraint_type"]:
            case "None":
                [self.strain_func, self.strain_limit] = get_no_constraints()
            case "5coil_strain":
                [self.strain_func, self.strain_limit] = get_5coil_strain_constraints()

        self.param_keys = sorted(response_model.param_dict.keys())
        self.max_current_slope = stimulator['max_current_slope'].to(self.device)

        # Set limit maximum E-field strength for subject confort
        self.efield_limit = 200  # V/m

        # Create and JIT-compile the objective function once during initialization.
        self.jitted_objective = _create_objective_function(
            efield_basis_means_jax=jax.device_put(efield_model.efield_basis_means.cpu().numpy().astype(np.float32), self.jax_device),
            efield_basis_jax=jax.device_put(efield_model.efield_basis.cpu().numpy().astype(np.float32), self.jax_device),
            null_dir_basis_jax=jax.device_put(efield_model.null_dir_basis.squeeze().cpu().numpy().astype(np.float32), self.jax_device),
            prob_lim_jax=jax.device_put(response_model.prob_lim, self.jax_device),
            efield_set_jax=jax.device_put(efield_model.efield_set.cpu().numpy().astype(np.float32), self.jax_device),
            max_current_slope_jax=jax.device_put(stimulator['max_current_slope'].cpu().numpy().astype(np.float32).flatten(), self.jax_device),
            efield_limit=jax.device_put(self.efield_limit, self.jax_device),
            strain_func_jax=self.strain_func,
            strain_limit=self.strain_limit,
            constraint_penalty=self.optimizer_settings['constraint_penalty'],
            param_keys=self.param_keys
        )

        self.seed = 0
        self.sobol_sampler = Sobol(d=self.n_coils, scramble=True, seed=self.seed)

        self.jitted_step = None
        self.strategy = None
        self.strategy_params = None

        # JIT-compile the JAX-native constraint function
        self.jitted_constraint_func = jax.jit(self._constraint_solution_strain_jax)

        # Initialize a main JAX randomization key
        self.key = jax.random.key(self.seed)

    def get_or_create_jitted_step(self):
        """
        On first call, creates and JIT-compiles the single-step optimizer function.
        Subsequently, fetches the created function handle.
        """ 
        if not self.jitted_step:
            # Initialize the Differential Evolution strategy.
            dummy_population = jnp.ones(self.n_coils) * 0.2 # Arbitrary value within bounds
            self.strategy = DE(
                population_size=self.optimizer_settings['popsize'],
                solution=dummy_population,
            )
            self.strategy_params = self.strategy.default_params.replace(
                crossover_rate=self.optimizer_settings['crossover_rate'],
                differential_weight=self.optimizer_settings['differential_weight'],
            )

            self.jitted_step = jax.jit(
                partial(
                    _optimizer_step,
                    strategy=self.strategy,
                    strategy_params=self.strategy_params,
                    jitted_objective=self.jitted_objective
                )
            )
        return self.jitted_step

    def choose_next_stimulus(self, response_model: ResponseModel) -> Tensor:
        """Finds the optimal stimulus to maximize expected information gain.

        This method runs the full JAX-based Differential Evolution optimization
        process and returns the best stimulus found.

        Note: The first time this function is called, it will be slower
        as it triggers JAX's JIT compilation for the optimization step.

        Args:
            response_model: The response model, providing the current source
                probabilities and fitted parameters.

        Returns:
            A tensor representing the optimal coil weightings (dI/dt) for the
            next stimulus.
        """
        jitted_step = self.get_or_create_jitted_step()

        settings = self.optimizer_settings
        
        opt_key, self.key = jax.random.split(self.key)

        # Convert dynamic data (which changes each trial) to JAX arrays.
        log_source_prob_jax = jax.device_put(response_model.log_source_prob.cpu().numpy().astype(np.float32), self.jax_device)
        param_tuple_jax = tuple(jax.device_put(response_model.param_dict[k]['values'].cpu().numpy().astype(np.float32), self.jax_device) for k in self.param_keys)

        # Initialize population using a Sobol sequence for quasi-random, uniform coverage of the search space.
        initial_population = self.sobol_sampler.random(n=settings['popsize']) * 2 - 1
        initial_population = jax.device_put(initial_population, self.jax_device)

        # Constraint initial population
        initial_population = self.jitted_constraint_func(initial_population)

        initial_fitness = self.jitted_objective(initial_population, log_source_prob_jax, param_tuple_jax)

        # Initialize the optimizer state.
        init_key, loop_key = jax.random.split(opt_key)
        state = self.strategy.init(init_key, initial_population, initial_fitness, self.strategy_params)

        # Run the optimization loop efficiently using jax.lax.scan.
        initial_carry = (state, loop_key, log_source_prob_jax, param_tuple_jax)
        scan_xs = jnp.arange(settings['num_generations'])
        final_carry, metrics_log = jax.lax.scan(jitted_step, initial_carry, scan_xs)

        # Block until computation is finished to get results.
        final_state = final_carry[0]
        best_solution_jax = final_state.best_solution
        best_solution_jax.block_until_ready()
        
        # Apply constraints to the final solution
        best_solution_jax_constrained = self.jitted_constraint_func(best_solution_jax)

        metrics_log['best_fitness'].block_until_ready()
        self.losses = torch.tensor(np.array(metrics_log['best_fitness']))

        final_x = torch.tensor(np.array(best_solution_jax_constrained), device=self.device)

        # Scale solution from [-1, 1] range to the true dI/dt range
        x_true_range = final_x * self.max_current_slope

        return x_true_range

    def _constraint_solution_strain_jax(self, x: Array) -> Array:
        """JAX-native function to ensures a stimulus solution does not exceed
        the maximum strain limit.

        If a solution's calculated strain is over the limit, it is analytically
        scaled down to be just within the boundary, assuming strain is a 
        homogeneous function of degree 2.

        This function is JIT-compiled in __init__ for performance.

        Args:
            x: An array of stimulus solutions, shape (n_solutions, n_coils) or (n_coils,).

        Returns:
            The input array, scaled down if necessary to meet the strain constraint.
        """

        if self.strain_limit is None:
            return x
        
        # Ensure input is at least 2D (batch, n_coils) for vmap
        was_1d = (x.ndim == 1)
        x_batch = jnp.atleast_2d(x)

        # Vmap the strain function to apply to each solution in the batch
        vmapped_strain_func = jax.vmap(self.strain_func)
        solution_strain = vmapped_strain_func(x_batch)  # Shape: (batch,)

        # Add epsilon to prevent division by zero if strain/limit is 0
        epsilon = 1e-9
        
        # S_old / L
        strain_relation = solution_strain / (self.strain_limit + epsilon) # Shape: (batch,)
        
        # Find solutions that violate the constraint (Strain_old > Limit)
        violation_mask = strain_relation > 1.0

        # Calculate the scaling factor k = sqrt(Limit / Strain_old)
        # We know k^2 = Limit / Strain_old = 1.0 / strain_relation
        # So, k = sqrt(1.0 / strain_relation)
        scaling_factor_k = jnp.sqrt(1.0 / (strain_relation + epsilon)) # Shape: (batch,)

        # Apply scaling k only to violators, others get scaling=1.0
        final_scaling = jnp.where(
            violation_mask,
            scaling_factor_k,
            1.0
        )

        # Apply scaling by broadcasting
        x_scaled = x_batch * final_scaling[:, None]

        # Return in original shape
        if was_1d:
            return x_scaled[0]
        else:
            return x_scaled