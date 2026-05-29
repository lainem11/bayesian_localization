"""Module for data preprocessing and loading utilities.

This module contains functions to load and parse data from various file
formats, such as JSON for stimulator configurations, .mat files for E-field
data, and custom .bin/.csv formats for head models.
"""
import json
import pickle
import torch

def get_stimulator_from_file(filename: str, constraint_type: str = "None") -> dict:
    """Loads and validates stimulator data from a JSON file.

    It ensures that the necessary keys ('max_current_slope' or 'max_voltage'
    and 'inductance') are present and converts all numerical data to tensors.
    Hardware constraint can be optionally specified.

    Args:
        filename: The path to the JSON stimulator file.
        constraint_type: Identifier for stimulator hardware constraint type.

    Returns:
        A dictionary containing stimulator parameters as PyTorch tensors, or
        None if the file is invalid or cannot be read.
    """
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading stimulator file: {e}")
        return None

    stimulator_dict = {}
    for key, value in data.items():
        try:
            stimulator_dict[key] = torch.atleast_1d(torch.tensor(value))
        except (TypeError, ValueError):
            print(f"Warning: Could not convert key '{key}' to a tensor. Skipping.")

    # Calculate max_current_slope if not provided
    if 'max_current_slope' not in stimulator_dict:
        print("Error: Stimulator file must contain 'max_current_slope'.")
        return None
    
    # Add constraint string
    stimulator_dict['constraint_type'] = constraint_type

    return stimulator_dict

def load_example_model(filename: str) -> tuple:
    """Loads bundled example E-field and cortex data from a pickle file.

    Args:
        filename: Path to the pickle file produced by examples/build_example_data.py.

    Returns:
        A tuple containing:
        - E (torch.Tensor): The E-field tensor, shape (n_coils, n_vertices, 3).
        - cortex (dict): A dictionary with 'vertices' and 'faces' of the mesh.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    E = data['efield_set']
    ROI = torch.arange(E.shape[1])
    cortex = {'vertices': data['vertices'], 'faces': data['faces']}
    return E, ROI, cortex

def load_example_trials(filename: str) -> tuple:
    """Loads bundled example E-field trial and response data from a pickle file.

    Args:
        filename: Path to the pickle file containing trial data.

    Returns:
        A tuple containing:
        - efield_trials (torch.Tensor): E-field tensors used per trial, shape (n_trials, n_vertices, 3).
        - responses (torch.Tensor): Recorded responses per trial, shape (n_trials,).
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['efield_trials'], data['responses']