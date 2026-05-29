"""Module for defining optimization constraints.

This module provides functions that return JAX-native penalty
function for use in the StimulusOptimizer. This allows constraints
to be defined externally and injected into the optimizer.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple

# Type Aliases
Array = jnp.ndarray

# Factory Function for Strain Constraint
def get_5coil_strain_constraints() -> Tuple[Callable, float]:
    """Returns the JAX constraint function and constraint limit."""

    def _calculate_strain(weights: Array) -> Array:
        """Strain calculation for specific 5-coil stimulator."""
        weights_abs = jnp.abs(weights)

        # Calculate coil 5 strain
        term1_5 = jnp.sqrt(0.9 * weights_abs[0]**2 + weights_abs[1]**2) * 0.2
        term2_5 = jnp.sqrt(0.9 * weights_abs[2]**2 + weights_abs[3]**2)
        coil_5_strain = (term1_5 + term2_5) * weights_abs[4]

        # Calculate coil 1 strain
        term1_1 = 0.9 * weights_abs[1]
        term2_1 = 0.7 * jnp.sqrt(weights_abs[2]**2 + 0.9 * weights_abs[3]**2)
        term3_1 = 0.2 * weights_abs[4]
        coil_1_strain = (term1_1 + term2_1 + term3_1) * weights_abs[0]

        # Find the maximum of the two strain values
        pulse_strain = jnp.maximum(coil_5_strain, coil_1_strain)

        return pulse_strain

    # Calculate and capture the strain limit, JAX-native
    max_strain_vector_jax = jnp.array([0.0, 0.0, 0.0, 0.8, 0.8])
    strain_limit = float(_calculate_strain(max_strain_vector_jax))

    return _calculate_strain, strain_limit

# Factory Function for No Constraint (default)
def get_no_constraints() -> Tuple[None, None]:
    return None, None