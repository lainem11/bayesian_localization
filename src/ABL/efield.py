"""Component for E-field physics modeling and transformations.

This module provides the EFieldModel class, which encapsulates the physics
of electric fields, including dimensionality reduction via PCA and transformations
between 3D space and the learned 2D subspace.
"""

# Standard library imports
from typing import Dict, Optional, Tuple

# Third-party imports
import torch

# Local application imports
from . import _internal
from .mesh import CorticalMesh

# Type Aliases
Tensor = torch.Tensor


class EFieldModel:
    """Encapsulates the physics of E-fields and their transformations.

    This class manages the E-field basis set from different stimulator coils
    and performs Principal Component Analysis (PCA) to create a lower-dimensional
    representation of the E-field at each vertex.

    Attributes:
        device (torch.device): The PyTorch device for computation.
        mesh (CorticalMesh): The associated cortical mesh object.
        efield_set (Optional[Tensor]): E-field vectors for each coil, with a
            shape of (n_coils, n_vertices, 3).
        efield_basis (Optional[Tensor]): Orthonormal basis from PCA of E-fields.
            Shape is (n_vertices, 3, 2).
        efield_basis_means (Optional[Tensor]): Mean E-field values used for PCA.
            Shape is (n_vertices, 3).
        null_dir_basis (Optional[Tensor]): 2D vectors representing the E-field
            null direction in the PCA subspace.
    """

    def __init__(self, efield_set: Optional[Tensor], mesh: CorticalMesh, device: torch.device):
        """Initializes the EFieldModel.

        Args:
            efield_set: E-field vectors for each coil.
            mesh: The cortical mesh component.
            device: The computation device.
        """
        self.device = device
        self.mesh = mesh
        self.efield_set = efield_set
        self.efield_basis: Optional[Tensor] = None
        self.efield_basis_means: Optional[Tensor] = None
        self.null_dir_basis: Optional[Tensor] = None

        if self.efield_set is not None:
            self.efield_set = self.efield_set.to(self.device)
            self.apply_pca(self.efield_set)

    def load_data(self, data_dict: Dict):
        """Loads E-field subspace data from a dictionary (e.g., from HDF5).

        Args:
            data_dict: A dictionary containing 'efield_subspace' and/or 'efield_set' data.
        """
        if 'efield_subspace' in data_dict:
            efield_subspace = data_dict['efield_subspace']
            self.efield_basis = efield_subspace['efield_basis'].to(self.device)
            self.null_dir_basis = efield_subspace['null_dir_basis'].to(self.device)
            self.efield_basis_means = efield_subspace['efield_basis_means'].to(self.device)
        if 'efield_set' in data_dict:
            self.efield_set = data_dict['efield_set'].to(self.device)

    def apply_pca(self, efield_set: Tensor):
        """Applies PCA to find a 2D subspace, then re-orients the per-vertex
        basis to a globally consistent right-handed frame.

        `torch.pca_lowrank` returns each principal component with arbitrary
        sign, leaving the per-vertex (V0, V1) frame free to reflect and rotate
        in-plane. We discard PCA's choice of (V0, V1) and rebuild the frame
        from two world-axis anchors:
          * the plane normal is flipped to point toward +Z (so V0 x V1 is
            closer to +Z than -Z), and
          * V0 is the in-plane component of world Y = [0, 1, 0].
        V1 is then `plane_normal x V0`, making (V0, V1, plane_normal) right-
        handed. `atan2(y, x)` in this frame measures CCW rotation as viewed
        from +Z, consistently across every vertex.

        Args:
            efield_set: Input tensor with a shape of (n_coils, n_vertices, 3).
        """
        e_reshaped = efield_set.transpose(0, 1)
        means = e_reshaped.mean(dim=1)
        e_centered = e_reshaped - means[:, None, :]

        _, _, V = torch.pca_lowrank(e_centered, q=2)

        # Plane normal, anchored to +Z. A world-axis anchor is invariant across
        # neighboring vertices, unlike a centroid-based outward direction which
        # can flip sign for anti-parallel PCA normals when the plane is near
        # edge-on to the centroid ray.
        plane_normal = torch.cross(V[..., 0], V[..., 1], dim=-1)
        ref_z = torch.tensor([0.0, 0.0, 1.0], device=V.device)
        normal_sign = torch.where(
            (plane_normal * ref_z).sum(dim=-1, keepdim=True) < 0, -1.0, 1.0
        )
        plane_normal = plane_normal * normal_sign

        # Anchor V0 to world Y projected onto the plane.
        ref_y = torch.tensor([0.0, 1.0, 0.0], device=V.device)
        ref_dot_n = (plane_normal * ref_y).sum(dim=-1, keepdim=True)
        new_V0 = ref_y - ref_dot_n * plane_normal
        new_V0 = new_V0 / new_V0.norm(dim=-1, keepdim=True)

        # V1 completes the right-handed frame (new_V0, new_V1, plane_normal).
        new_V1 = torch.cross(plane_normal, new_V0, dim=-1)
        new_V1 = new_V1 / new_V1.norm(dim=-1, keepdim=True)

        V = torch.stack([new_V0, new_V1], dim=-1)

        # null_dir_basis is the world-Y projection in the new basis; by
        # construction this is ~[1, 0] at every vertex.
        null_dir = ref_y.repeat(efield_set.shape[1], 1)
        null_dir_transformed = (null_dir[:, None, :] @ V)
        null_dir_transformed /= torch.linalg.norm(null_dir_transformed, dim=-1, keepdim=True)

        self.efield_basis = V
        self.null_dir_basis = null_dir_transformed.transpose(0, 1)
        self.efield_basis_means = means

    def efield_to_mag_and_angle(self, efield: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculates E-field magnitude and its angle in the 2D PCA subspace.

        Args:
            efield: A tensor of E-field vectors, with a shape of (n_trials, n_vertices, 3).

        Returns:
            A tuple containing:
                - e_mag (Tensor): E-field magnitudes, shape (n_trials, n_vertices).
                - angles (Tensor): E-field angles in radians, shape (n_trials, n_vertices).
        """
        e_mag = torch.norm(efield, dim=-1)
        angles = self.efield_to_angles(efield)
        return e_mag, angles

    def efield_to_angles(self, efield: Tensor) -> Tensor:
        """Converts batched 3D E-field vectors to angles in the 2D PCA subspace.

        Args:
            efield: Input tensor with a shape of (B, N, 3), where B is
                the batch size and N is the number of vertices.

        Returns:
            A tensor of angles in radians, with a shape of (B, N).
        """
        e_transformed = self.efield_to_subspace(efield)
        norm = torch.linalg.norm(e_transformed, dim=-1, keepdim=True)
        e_transformed_norm = e_transformed / norm.clamp(min=1e-9)
        return _internal._utils.vector_angle(self.null_dir_basis, e_transformed_norm)

    def efield_to_subspace(self, efield: Tensor) -> Tensor:
        """Transforms batched 3D E-field data into the 2D PCA subspace.

        Args:
            efield: Input tensor with a shape of (B, N, 3).

        Returns:
            The transformed tensor in the 2D subspace, with a shape of (B, N, 2).
        """
        # Center the data, then project it onto the basis vectors
        new_centered = efield.transpose(0, 1)
        new_transformed = new_centered @ self.efield_basis
        return new_transformed.transpose(0, 1)

    def angles_to_directions(self, angles: Tensor, vertices: Tensor) -> Tensor:
        """Reconstructs 3D E-field directions from angles in the PCA subspace.

        Args:
            angles: Tensor of angles, shape (n_vertices, n_angles).
            vertices: Tensor of vertex indices, shape (n_vertices,).

        Returns:
            A tensor of reconstructed 3D directions.
        """
        # Convert angles back to 2D vectors in the subspace
        dir_transformed = _internal._utils.angle_to_vector(self.null_dir_basis[0, vertices], angles)
        basis = self.efield_basis[vertices]
        means = self.efield_basis_means[vertices]
        # Project back to 3D and add the mean back
        dir_reconstructed = (dir_transformed[:, :, None, :] @ basis.transpose(-2, -1)) + means[:, None, :]
        return dir_reconstructed.unsqueeze(-2)
