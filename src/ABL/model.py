"""Component for the statistical response model, parameter fitting, and updates.

This module provides the ResponseModel class, which defines the statistical model
of neural responses, handles parameter fitting against experimental data, and
performs Bayesian updates of source probabilities.
"""

# Standard library imports
import copy
import math
from typing import Any, Callable, Dict, List, Optional
import gc

# Third-party imports
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import gelman_rubin, effective_sample_size

#import pyro
#import pyro.distributions as dist
#import pyro.poutine as poutine
#from pyro.infer import MCMC, NUTS
#from pyro.infer.reparam import AutoReparam
#from pyro.infer.autoguide import init_to_value
#from pyro.infer import Predictive
#from pyro.ops.stats import gelman_rubin, effective_sample_size

# Local application imports
from . import _internal
from .data_handler import ExperimentData
from .efield import EFieldModel
from .mesh import CorticalMesh

# Type Aliases
Tensor = torch.Tensor

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from numpyro.diagnostics import gelman_rubin, effective_sample_size

def numpyro_response_model(_e_mag_jax: jnp.ndarray, cos_a_jax: jnp.ndarray, sin_a_jax: jnp.ndarray, 
                           cos_2a_jax: jnp.ndarray, sin_2a_jax: jnp.ndarray, log_emag_jax: jnp.ndarray, 
                           _responses_jax: jnp.ndarray):
    """Pure JAX/NumPyro probabilistic model, defined outside the class to ensure 
    compatibility with JAX tracing and avoid stateful 'self' side-effects."""
    
    n_points, n_trials = _e_mag_jax.shape

    with numpyro.plate("points", n_points, dim=-2):
        # Priors
        rf_slope = numpyro.sample("rf_slope", dist.Uniform(1e-3, 1.0))
        rf_translate = numpyro.sample("rf_translate", dist.Normal(75.0, 50.0))
        rf_fourier1 = numpyro.sample("rf_fourier1", dist.Normal(0.0, 20.0))
        rf_fourier2 = numpyro.sample("rf_fourier2", dist.Normal(0.0, 20.0))
        rf_fourier3 = numpyro.sample("rf_fourier3", dist.Normal(0.0, 20.0))
        rf_fourier4 = numpyro.sample("rf_fourier4", dist.Normal(0.0, 20.0))

        with numpyro.plate("trials", n_trials, dim=-1):
            # Direction-dependent translation model
            directional_translation = (
                rf_translate +
                rf_fourier1 * cos_a_jax +
                rf_fourier2 * sin_a_jax +
                rf_fourier3 * cos_2a_jax +
                rf_fourier4 * sin_2a_jax
            )

            # Sigmoid response function
            inner_term = -rf_slope * (_e_mag_jax - directional_translation)
            logits = log_emag_jax - inner_term

            # Condition the model
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=_responses_jax)

class ResponseModel:
    """Contains the logic for the statistical model of neural responses.

    This class manages the parameters of the response function and updates the posterior
    distribution of the neural source location via Bayes' rule. For real-time inference,
    parameter are fitted with maximum-likelihood approach, but for offline analysis, MCMC
    sampling can be used to account for parameter uncertainty.

    Attributes:
        mesh (CorticalMesh): The cortical mesh data.
        device (torch.device): The real-time computation device.
        jax_device (torch.device): (jax.Device): The JAX device used for likelihood sampling.
        param_dict (Dict): Dictionary of response function parameters.
        response_func (Optional[Callable]): The response probability model function.
        log_source_prob (Optional[Tensor]): Posterior log source probabilities.
        prior_log_source_prob (Optional[Tensor]): Prior log source probabilities.
        prob_lim (List[float]): Clamping limits for probabilities to avoid
            numerical instability (e.g., log(0)).
        opt_settings (Dict): Optimization settings for parameter fitting.
        regularization_lambda (float): L2 regularization weight.
        losses (Tensor): Stores the loss from each iteration of the last
            optimization run.
        mcmc_settings (Dict): Hyperparameters for the MCMC sampling kernel.
    """

    def __init__(self, mesh: CorticalMesh, device: torch.device):
        """Initializes the ResponseModel.

        Args:
            mesh: The cortical mesh component.
            device: The computation device.
        """
        self.mesh = mesh
        self.device = device
        if device.type == 'cpu':
            self.jax_device = jax.devices('cpu')[0]
        else:
            self.jax_device = jax.devices()[0] # Default to first available device (usually GPU)

        self.param_dict: Dict[str, Dict[str, Any]] = {}
        self.response_func: Optional[Callable] = None
        self.log_source_prob: Optional[Tensor] = None
        self.prior_log_source_prob: Optional[Tensor] = None
        self.prob_lim: List[float] = [1e-6, 1.0 - 1e-6]
        self.opt_settings: Dict[str, Any] = {}
        self.regularization_lambda: float = 0.0
        self.losses: Tensor = torch.tensor([0])
        self.posterior_samples: Dict[str, Tensor] = {}
        self.mcmc_settings: Dict[str, Any] = {
            'num_samples': 200,
            'warmup_steps': 200,
            'num_chains': 4, # Increase for better diagnostics, at the cost of memory/compute
        }

        self.initialize_response_function_parameters()
        self._initialize_optimizer_settings()
        self._define_response_function()
        self.initialize_probabilities()

    def load_data(self, data_dict: Dict):
        """Loads model parameters from a dictionary (e.g., from HDF5).

        Args:
            data_dict: A dictionary containing 'param_dict' with saved parameter values.
        """

        # Re-initialize to ensure all keys and functions exist, then overwrite values
        self.initialize_response_function_parameters()
        
        if 'posterior_samples' in data_dict:
            self.posterior_samples = {
                k: v.to(self.device) for k, v in data_dict['posterior_samples'].items()
            }
            self._calculate_point_estimates()
        elif 'param_dict' in data_dict:
            loaded_param_dict = data_dict['param_dict']
            for k, v in loaded_param_dict.items():
                if k in self.param_dict:
                    self.param_dict[k]['values'] = v['values'].to(self.device)

    def initialize_probabilities(self):
        """Sets initial source priors."""
        prior_source_prob = _internal._utils.get_prior(self.mesh.vertices[self.mesh.ROI, :]).to(self.device)

        self.prior_log_source_prob = self._safe_log(prior_source_prob)
        self.log_source_prob = copy.deepcopy(self.prior_log_source_prob)

    def initialize_response_function_parameters(self):
        """Sets up the parameter dictionary for the response function."""
        n_points = len(self.mesh.ROI)
        default_priors = {'rf_slope': 0.05, 'rf_translate': 75, 'rf_fourier1': 0, 'rf_fourier2': 0, 'rf_fourier3': 0, 'rf_fourier4': 0}

        # Parameter conversion functions are used for constrained optimization.
        # They map unconstrained values optimized by Adam to valid parameter ranges.
        param_conv_funcs = {
            'rf_slope': [lambda x: (x.clamp(1e-6, 1 - 1e-6) * 100).log(), lambda x: (x.exp() / 100).clamp(1e-6, 1 - 1e-6)],
            'default': [lambda x: x / 100, lambda x: x * 100]
        }

        param_defs = {k: (v, param_conv_funcs.get(k, param_conv_funcs['default'])) for k, v in default_priors.items()}
        self.param_dict = {
            k: {'values': val * torch.ones((n_points,), device=self.device), 'conversion_func': conv}
            for k, (val, conv) in param_defs.items()
        }

    def fit_model_to_data(self, data_handler: ExperimentData, efield_model: EFieldModel):
        """Fits the response function model to the accumulated trial data.

        This method uses an Adam optimizer to minimize the negative log-likelihood
        of the observed responses, thereby finding the best-fit parameters for
        the response function.
        """
        opt_settings = self.opt_settings['general']
        responses = data_handler.get_active_responses()
        efields = data_handler.get_active_efield_trials()
        e_mag, angles = efield_model.efield_to_mag_and_angle(efields)
        trial_number = data_handler.trial_number

        # Prepare parameters for optimization by transforming them into an unconstrained space
        opt_param_dict = {
            k: {'values': v['conversion_func'][0](v['values']).requires_grad_(), 'conversion_func': v['conversion_func']}
            for k, v in self.param_dict.items()
        }
        optimizer, scheduler = self._setup_optimizer(opt_param_dict, opt_settings, trial_number)

        losses, _ = self._run_optimization_loop(optimizer, scheduler, opt_param_dict, e_mag, angles, responses, opt_settings)
        self.losses = losses

        self._store_fit_results(opt_param_dict)

    def fit_distribution_model_to_data(self, data_handler: ExperimentData, efield_model: EFieldModel):
        """Runs the No U-turn sampler (NUTS) to infer parameter uncertainty."""
        jax_device = self.jax_device

        # Format variables
        responses = data_handler.get_active_responses().cpu().numpy().astype(np.float32)
        efields = data_handler.get_active_efield_trials()
        e_mag_pt, angles_pt = efield_model.efield_to_mag_and_angle(efields)
        e_mag = e_mag_pt.cpu().numpy().astype(np.float32)
        angles = angles_pt.cpu().numpy().astype(np.float32)

        _angles = np.transpose(angles, (1, 0))
        _e_mag = np.transpose(e_mag, (1, 0))
        _responses = np.expand_dims(responses, 0)

        # Precompute operations and move to JAX
        cos_a_jax = jax.device_put(np.cos(_angles), jax_device)
        sin_a_jax = jax.device_put(np.sin(_angles), jax_device)
        cos_2a_jax = jax.device_put(np.cos(_angles * 2), jax_device)
        sin_2a_jax = jax.device_put(np.sin(_angles * 2), jax_device)
        log_emag_jax = jax.device_put(np.log(_e_mag + 1e-9), jax_device)
        _e_mag_jax = jax.device_put(_e_mag, jax_device)
        _responses_jax = jax.device_put(_responses, jax_device)

        target_keys = [
            'rf_slope', 'rf_translate', 'rf_fourier1', 
            'rf_fourier2', 'rf_fourier3', 'rf_fourier4'
        ]

        # Initialize from MLE estimates
        init_values = {
            k: jax.device_put(
                self.param_dict[k]['values'].cpu().numpy().astype(np.float32)[:, None], 
                jax_device
            )
            for k in target_keys if k in self.param_dict
        }
        init_strategy = init_to_value(values=init_values)

        num_chains = self.mcmc_settings.get('num_chains', 2)
        
        # Initialize NUTS
        nuts_kernel = NUTS(numpyro_response_model, init_strategy=init_strategy)
        mcmc = MCMC(
            nuts_kernel,
            num_samples=self.mcmc_settings['num_samples'],
            num_warmup=self.mcmc_settings['warmup_steps'],
            num_chains=num_chains,
            chain_method='vectorized',
            progress_bar=True
        )

        print(f"\n--- Running MCMC on {self.jax_device.platform.upper()} ({num_chains} Chains) ---")
        
        rng_key = jax.random.PRNGKey(0)
        
        # Run the sampler
        mcmc.run(rng_key, _e_mag_jax, cos_a_jax, sin_a_jax, cos_2a_jax, sin_2a_jax, log_emag_jax, _responses_jax)

        samples = mcmc.get_samples(group_by_chain=True)
        self.posterior_samples = {}
        
        for k in target_keys:
            if k in samples:
                chain_data = samples[k]
                
                # Remove the empty plate dimension: (num_chains, num_samples, n_points, 1) -> (num_chains, num_samples, n_points)
                if chain_data.ndim == 4:
                    chain_data = jnp.squeeze(chain_data, axis=-1)
                
                # Flatten chains and samples together: (total_samples, n_points)
                # Convert explicitly to NumPy -> PyTorch to sever JAX ties
                flattened_data = np.array(chain_data.reshape(-1, chain_data.shape[-1]))
                self.posterior_samples[k] = torch.from_numpy(flattened_data).unsqueeze(-1).cpu()

        self._calculate_point_estimates()

        # Memory clean-up
        del cos_a_jax, sin_a_jax, cos_2a_jax, sin_2a_jax, log_emag_jax, _e_mag_jax, _responses_jax
        del init_values, mcmc, nuts_kernel, samples

    def update_source_probabilities(self, data_handler: ExperimentData, efield_model: EFieldModel):
        """Computes the Bayesian posterior update of source probabilities.

        This method applies Bayes' rule in log-space to update the probability
        of each vertex being the source, given all observed responses.
        p(H|y) = p(H) * p(y|H) / p(y)
        where H = 'Point is source' and y = 'Observed responses'.
        """
        efields = data_handler.get_active_efield_trials()
        e_mag, angles = efield_model.efield_to_mag_and_angle(efields)

        responsivity = self.response_func(e_mag, angles, self.param_dict)
        log_responsivity = self._safe_log(responsivity)

        # Calculate total log-likelihood for all trials
        responses = data_handler.get_active_responses().view(-1, 1)
        log_lik_trials = (1 - responses) * self._safe_log(1 - responsivity) + responses * log_responsivity
        log_lik_total = log_lik_trials.sum(0)

        # Apply Bayes' formula: log(posterior) = log(prior) + log(likelihood) - log(marginal)
        nominator = self.prior_log_source_prob + log_lik_total
        norm_term = nominator.logsumexp(-1)
        new_log_source_prob = nominator - norm_term

        if new_log_source_prob.isnan().any():
            print("Warning: Numerical error resulted in NaN source probabilities!")
        self.log_source_prob = new_log_source_prob

    def update_distributional_source_probabilities(self, data_handler: ExperimentData, efield_model: EFieldModel, batch_size: int = 500):
        """Computes the Bayesian posterior update of source probabilities.

        This marginalizes over the parameter uncertainty by computing the 
        expected likelihood of the data across all MCMC parameter samples.
        Computation is batched over n_points to prevent Out-Of-Memory (OOM) errors.
        """
        if not self.posterior_samples:
            print("Warning: No posterior samples exist. Run `fit_distribution_model_to_data` first.")
            return
        
        # Make sure point estimates are calculated
        self._calculate_point_estimates()

        responses = data_handler.get_active_responses().cpu()
        efields = data_handler.get_active_efield_trials()
        e_mag, angles = efield_model.efield_to_mag_and_angle(efields)
        e_mag = e_mag.cpu()
        angles = angles.cpu()

        num_samples = self.posterior_samples['rf_slope'].shape[0]
        n_points = e_mag.shape[1]

        # Responses to (1, 1, n_trials)
        _responses = responses.view(1, 1, -1)
        
        # Pre-allocate the expected log likelihood tensor for all points
        expected_log_lik = torch.zeros(n_points, device='cpu')

        # Process the points in batches to bound peak memory usage
        for i in range(0, n_points, batch_size):
            end_i = min(i + batch_size, n_points)

            # Reshape and slice inputs for the current batch: (1, batch_size, n_trials)
            _e_mag_batch = e_mag[:, i:end_i].transpose(0, 1).unsqueeze(0)
            _angles_batch = angles[:, i:end_i].transpose(0, 1).unsqueeze(0)

            # Extract parameter samples for the batch: (num_samples, batch_size, 1)
            rf_slope_batch = self.posterior_samples['rf_slope'][:, i:end_i, :].cpu()
            rf_translate_batch = self.posterior_samples['rf_translate'][:, i:end_i, :].cpu()
            rf_fourier1_batch = self.posterior_samples['rf_fourier1'][:, i:end_i, :].cpu()
            rf_fourier2_batch = self.posterior_samples['rf_fourier2'][:, i:end_i, :].cpu()
            rf_fourier3_batch = self.posterior_samples['rf_fourier3'][:, i:end_i, :].cpu()
            rf_fourier4_batch = self.posterior_samples['rf_fourier4'][:, i:end_i, :].cpu()

            # Vectorized calculation for the batch -> (num_samples, batch_size, n_trials)
            directional_translation = (
                rf_translate_batch +
                rf_fourier1_batch * torch.cos(_angles_batch) +
                rf_fourier2_batch * torch.sin(_angles_batch) +
                rf_fourier3_batch * torch.cos(_angles_batch * 2) +
                rf_fourier4_batch * torch.sin(_angles_batch * 2)
            )

            inner_term = -rf_slope_batch * (_e_mag_batch - directional_translation)
            logits = torch.log(_e_mag_batch + 1e-9) - inner_term

            log_responsivity = F.logsigmoid(logits)
            log_1_minus_responsivity = F.logsigmoid(-logits)

            log_lik_trials = _responses * log_responsivity + (1 - _responses) * log_1_minus_responsivity
            log_lik_total = log_lik_trials.sum(dim=-1)

            # Marginalize using log-sum-exp trick and store in the pre-allocated tensor
            expected_log_lik[i:end_i] = torch.logsumexp(log_lik_total, dim=0) - math.log(num_samples)

            # Eagerly free memory of large intermediate tensors
            del directional_translation, inner_term, logits, log_responsivity, log_1_minus_responsivity, log_lik_trials, log_lik_total

        # Apply Bayes using the fully assembled expected log likelihood
        nominator = self.prior_log_source_prob.cpu() + expected_log_lik
        norm_term = nominator.logsumexp(-1)
        new_log_source_prob = nominator - norm_term

        if new_log_source_prob.isnan().any():
            print("Warning: Numerical error resulted in NaN source probabilities!")
            
        self.log_source_prob = new_log_source_prob.to(self.device)

    def _initialize_optimizer_settings(self):
        """Defines hyperparameters for the optimization procedures."""
        self.opt_settings = {
            # Tuned hyperparameters for the Adam optimizer and LR scheduler
            'general': {'lr': 0.001, 'gamma': 0.904562, 'global_gamma': 0.999769, 'clip_grad': 0.1, 'betas': [0.832902, 0.706571], 'opt_steps': 150},
            'rf_slope': {'lr': 0.003403}, 'rf_translate': {'lr': 0.000548},
            'rf_fourier1': {'lr': 0.001622}, 'rf_fourier2': {'lr': 0.001622},
            'rf_fourier3': {'lr': 0.001622}, 'rf_fourier4': {'lr': 0.001622}
        }
        self.regularization_lambda = 0.018787

    def _define_response_function(self):
        """Defines the directional sigmoid response function model."""
        def directional_sigmoid(e_mag, angles, params):
            # The translation term is direction-dependent, modeled by a Fourier series
            directional_translation = (
                params['rf_translate']['values'] +
                params['rf_fourier1']['values'] * torch.cos(angles) +
                params['rf_fourier2']['values'] * torch.sin(angles) +
                params['rf_fourier3']['values'] * torch.cos(angles * 2) +
                params['rf_fourier4']['values'] * torch.sin(angles * 2)
            )
            inner_term = torch.clamp(-params['rf_slope']['values'] * (e_mag - directional_translation), -16, 16)
            response_prob = e_mag / (e_mag + torch.exp(inner_term))
            return torch.clamp(response_prob, self.prob_lim[0], self.prob_lim[1])

        self.response_func = directional_sigmoid

    def _setup_optimizer(self, opt_param_dict: dict, opt_settings: dict, trial_number: int):
        """Configures the Adam optimizer and learning rate scheduler."""
        lr_default = opt_settings['lr']
        global_gamma = opt_settings.get('global_gamma', 1.0)
        lr_decay_factor = global_gamma ** (trial_number - 1)

        # Set up parameter groups with potentially different learning rates
        opt_params = []
        for k, v in opt_param_dict.items():
            param_settings = self.opt_settings.get(k, {})
            base_lr = param_settings.get('lr', lr_default)
            lr = base_lr * lr_decay_factor  # Apply global decay
            opt_params.append({'params': v['values'], 'lr': lr})
            v['values'].register_hook(lambda grad: torch.clamp(grad, -opt_settings['clip_grad'], opt_settings['clip_grad']))

        optimizer = torch.optim.Adam(opt_params, lr=lr_default * lr_decay_factor, betas=opt_settings['betas'], foreach=True)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: opt_settings['gamma'] ** (1 / opt_settings['opt_steps']))
        return optimizer, scheduler

    def _run_optimization_loop(self, optimizer, scheduler, opt_param_dict, e_mag, angles, responses, opt_settings):
        """Executes the fitting loop until convergence or max steps are reached."""
        losses = torch.zeros((opt_settings['opt_steps'],))
        converged = False
        conv_buffer = []
        step_i = 0

        while not converged and step_i < opt_settings['opt_steps']:
            optimizer.zero_grad()
            # Convert parameters from unconstrained space back to their native range for loss calculation
            back_converted_params = {k: {'values': v['conversion_func'][1](v['values'])} for k, v in opt_param_dict.items()}

            responsivity = self.response_func(e_mag, angles, back_converted_params)
            likelihood = responsivity * responses.view(-1, 1) + (1 - responsivity) * (1 - responses).view(-1, 1)
            loss = -likelihood.log().mean()

            # Add L2 regularization to prevent overfitting
            reg = sum(self.regularization_lambda * (v['values']**2).mean() for v in opt_param_dict.values())
            loss += reg

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses[step_i] = loss.item()
            # Convergence check: stop if loss is stable for several steps
            if step_i > 0 and torch.abs(losses[step_i] - losses[step_i - 1]) < 1e-6:
                conv_buffer.append(1)
                if len(conv_buffer) >= 6:
                    converged = True
            else:
                conv_buffer = []
            step_i += 1

        return losses[:step_i], converged

    def _store_fit_results(self, opt_param_dict: dict):
        """Converts optimized parameters back to their native range and stores them."""
        final_params = {
            k: {'values': v['conversion_func'][1](v['values']).detach()}
            for k, v in copy.deepcopy(opt_param_dict).items()
        }
        for key, value_dict in final_params.items():
            self.param_dict[key]['values'] = value_dict['values']

    def _safe_log(self, tensor: Tensor) -> Tensor:
        """Clamped log operation to prevent numerical errors like log(0)."""
        return torch.clamp(tensor, self.prob_lim[0], self.prob_lim[1]).log()

    def _safe_exp(self, tensor: Tensor) -> Tensor:
        """Clamped exponential operation to prevent numerical errors."""
        min_val = math.log(self.prob_lim[0])
        max_val = math.log(self.prob_lim[1])
        return torch.clamp(tensor, min_val, max_val).exp()

    def _calculate_point_estimates(self):
        """Calculates the mean of the posterior samples to provide a point estimate."""
        self.param_dict = {}
        for k, v in self.posterior_samples.items():
            # v has shape (num_samples, n_points, 1). Mean and squeeze to get (n_points,)
            expected_value = v.mean(dim=0).squeeze(-1)
            self.param_dict[k] = {'values': expected_value}