"""Component for managing experimental trial data.

This module provides the ExperimentData class, which is responsible for
storing, managing, and providing access to all trial-related information,
such as recorded E-fields and responses.
"""

# Standard library imports
from typing import Any, Dict, List, Optional

# Third-party imports
import torch

# Type Aliases
Tensor = torch.Tensor


class ExperimentData:
    """Manages the history of an experiment, including all trial data.

    This class handles dynamic allocation of memory for trial data, ensuring
    that the buffers for E-fields and responses can grow as the experiment
    progresses.

    Attributes:
        device (torch.device): The PyTorch device for computation ('cpu' or 'cuda').
        trial_buffer_size (int): Current allocated size for trial data tensors.
        trial_number (int): The current trial number (1-indexed for new trials).
        responses (Optional[Tensor]): Tensor of recorded responses for each trial.
        efield_trials (Optional[Tensor]): Tensor of E-fields used in each trial.
    """

    def __init__(self, device: torch.device):
        """Initializes the ExperimentData component.

        Args:
            device (torch.device): The computation device.
        """
        self.device = device
        self.trial_buffer_size: int = 150
        self.trial_number: int = 0
        self.responses: Optional[Tensor] = None
        self.efield_trials: Optional[Tensor] = None

    def initialize_trials(self, ROI_size: int):
        """Initializes or resets data structures for a new session.

        This method resets the trial counter and pre-allocates tensors for
        responses and E-fields to an initial buffer size.

        Args:
            ROI_size: The number of vertices in the region of interest,
                used to determine the shape of the E-field tensor.
        """
        self.trial_buffer_size = 150
        self.trial_number = 0
        self.responses = torch.zeros((self.trial_buffer_size,), dtype=torch.float, device=self.device)

        shape = (self.trial_buffer_size, ROI_size, 3)
        self.efield_trials = torch.zeros(shape, device=self.device)

    def load_data(self, data_dict: Dict):
        """Loads trial data from a dictionary, typically from an HDF5 file.

        Args:
            data_dict: A dictionary containing measurement data, such as
                'efield_trials' and 'responses'.
        """
        if 'efield_trials' in data_dict:
            self.efield_trials = data_dict['efield_trials'].to(self.device)
        if 'responses' in data_dict:
            self.responses = data_dict['responses'].to(self.device)

        # Ensure consistency and set trial number
        if self.efield_trials is not None and self.responses is not None:
            assert self.efield_trials.shape[0] == self.responses.shape[0], \
                "Mismatch between number of E-field trials and responses."
            self.trial_number = self.responses.shape[0]

    def get_active_responses(self) -> Tensor:
        """Returns the slice of recorded responses from the buffer."""
        return self.responses[:self.trial_number]

    def get_active_efield_trials(self) -> Tensor:
        """Returns the slice of recorded E-field trials from the buffer."""
        return self.efield_trials[:self.trial_number]

    def advance_trial_counter(self):
        """Increments the trial counter to move to the next trial."""
        self.trial_number += 1

    def record_response_trial(self, response: bool):
        """Stores the response for the current trial.

        If the buffer is full, it will be expanded automatically.

        Args:
            response: The response to record for the current trial.
        """
        if self.trial_number > self.trial_buffer_size:
            self._append_trial_buffers()
        # trial_number is 1-indexed for the *next* trial, so we write to index trial_number - 1
        self.responses[self.trial_number - 1] = float(response)

    def _record_efield_trial(self, efield: Tensor):
        """Stores the E-field for the current trial.

        Args:
            efield: The E-field tensor to record.
        """
        if self.trial_number > self.efield_trials.shape[0]:
            self._append_trial_buffers()
        # trial_number is 1-indexed for the *next* trial, so we write to index trial_number - 1
        self.efield_trials[self.trial_number - 1, :, :] = efield

    def _append_trial_buffers(self):
        """Expands the E-field and response tensors to accommodate more trials."""
        current_len = self.responses.shape[0]
        # Grow the buffer by a fixed amount
        new_len = current_len + self.trial_buffer_size

        # Expand E-field trials tensor
        new_efields = torch.empty((new_len, *self.efield_trials.shape[1:]), device=self.device)
        new_efields[:current_len, :, :] = self.efield_trials
        self.efield_trials = new_efields

        # Expand responses tensor
        new_responses = torch.empty((new_len,), dtype=torch.float, device=self.device)
        new_responses[:current_len] = self.responses
        self.responses = new_responses

        self.trial_buffer_size = new_len
