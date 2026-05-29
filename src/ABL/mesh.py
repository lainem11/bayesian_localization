"""Component for managing cortical mesh geometry.

This module provides the CorticalMesh class, a data-centric container for holding
and managing the geometric properties of the brain mesh, such as vertices,
faces, and normals.
"""

# Standard library imports
from typing import Dict, Optional

# Third-party imports
import torch

# Local application imports
from . import _internal

# Type Aliases
Tensor = torch.Tensor


class CorticalMesh:
    """A data-centric class for brain mesh geometry.

    This class stores the geometric properties of the cortical mesh and computes
    derived quantities like vertex normals and a suitable camera position for
    visualization.

    Attributes:
        vertices (Tensor): Vertex coordinates, shape (n_vertices, 3).
        faces (Tensor): Mesh triangulations, shape (n_triangulations, 3).
        ROI (Tensor): Region-Of-Interest vertex indices.
        vertex_normals (Tensor): Normalized vertex normals, shape (n_vertices, 3).
        average_normal (Tensor): The average vertex normal in the ROI.
        camera (Dict): A dictionary defining the camera position for 3D plots.
        coarse_mesh_inds (Tensor): Subsampled mesh vertex indices for efficient plotting.
    """

    def __init__(self, cortex: Dict[str, Tensor], ROI: Optional[Tensor] = None):
        """Initializes the CorticalMesh component.

        Args:
            cortex: A dictionary with 'vertices' and 'faces' tensors.
            ROI: A tensor of indices defining the region of interest.
                If None, the entire mesh is used as the ROI.
        """
        self.vertices: Tensor = cortex["vertices"].float()
         # Center the mesh
        self.vertices = self.vertices - torch.mean(self.vertices, dim=0)

        self.faces: Tensor = cortex["faces"].long()
        self.ROI: Tensor = torch.arange(self.vertices.shape[0]) if ROI is None else ROI

        # Calculate and normalize vertex normals
        self.vertex_normals = _internal._utils.calculate_node_normals(self.vertices, self.faces)

        # Calculate the average normal vector for the ROI
        mean_normal = torch.nanmean(self.vertex_normals[self.ROI, :], 0)
        self.average_normal: Tensor = mean_normal / torch.linalg.norm(mean_normal)

        # Set up default camera and subsample mesh for plotting
        self.set_camera()
        self.coarse_mesh_inds: Tensor = self.subsample_pos(self.vertices[self.ROI, :], 0.008)

    def set_camera(self) -> Dict:
        """Sets the 3D plot camera to predetermined position."""
        self.camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0.0, y=0.0, z=0.0),
            eye=dict(x=-0.75, y=0.0, z=1.0),
        )
    
    def subsample_pos(self, nodes: torch.Tensor, spacing: float) -> torch.Tensor:
        """
        Selects a subset of nodes that are at least `spacing` distance apart.

        Implements a simple greedy algorithm:
        1. Selects the first node (index 0).
        2. Iterates through the remaining nodes.
        3. Selects a node if it is farther than `spacing` from *all*
        previously selected nodes.

        Args:
            nodes (torch.Tensor): A tensor of node coordinates, shape (num_nodes, n_dim).
            spacing (float): The minimum distance required between selected nodes.

        Returns:
            torch.Tensor: A 1D tensor of indices for the selected nodes.
        """
        # Number of nodes
        num_nodes = nodes.shape[0]
        
        # Initialize mask to track selected nodes
        selected = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Start with the first node
        selected[0] = True
        
        # Iterate through nodes and select those spaced apart by 'spacing'
        for i in range(1, num_nodes):
            # Compute distance to all previously selected nodes
            selected_nodes = nodes[selected]
            distances = ((selected_nodes - nodes[i,:])**2).sum(-1).sqrt()
            
            # Select node if it is sufficiently far from all selected nodes
            if torch.all(distances > spacing):
                selected[i] = True
        
        # Get indices of selected nodes
        sparse_indices = torch.nonzero(selected, as_tuple=True)[0]
        return sparse_indices
