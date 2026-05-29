import numpy as np
import torch
import h5py
import os
from scipy.special import lambertw
from typing import List, Tuple, Optional, Dict, Any, Iterable

def set_random_seed(seed: int = 10):
    """
    Sets the random seed for NumPy and PyTorch to ensure reproducibility.

    This function sets the seed for:
    - numpy.random
    - torch.manual_seed (for CPU)
    - torch.cuda.manual_seed_all (for GPU)
    - Disables cuDNN benchmarking and enables deterministic algorithms
      for reproducible results in CUDA operations.

    Args:
        seed (int): The integer value to use as the random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def vector_angle(U: torch.Tensor, v: torch.Tensor, directional: bool = True) -> torch.Tensor:
    """
    Computes the angle between two vectors or batches of vectors.

    Supports 2D and 3D vectors.

    Args:
        U (torch.Tensor): First vector(s). Shape (n_dim,), (n_vertices, n_dim),
                          or (batch, n_vertices, n_dim). n_dim must be 2 or 3.
        v (torch.Tensor): Second vector(s). Must be broadcastable to match U's shape.
        directional (bool): Only applicable for 2D vectors (n_dim=2).
            - If True: Computes the signed angle from U to v in the
              counter-clockwise direction (range [0, 2*pi)).
            - If False: Computes the smallest angle between the vectors
              (range [0, pi]).
            For 3D vectors, this argument is ignored and the smallest angle is computed.

    Returns:
        torch.Tensor: Tensor of angles in radians. Shape (n_vertices,) or (batch, n_vertices,).
    """
    EPS = torch.tensor(1e-8)    # Small value to avoid division by 0

    if (U == 0).all(dim=-1,keepdim=True).any() or (v == 0).all(dim=-1,keepdim=True).any():
        return torch.zeros_like(v).sum(-1)

    if v.shape[-1] == 2:
        if directional:
            angle_U = torch.atan2(U[..., 1], U[..., 0])
            angle_v = torch.atan2(v[..., 1], v[..., 0])

            angle_diff = angle_v - angle_U
            angle = angle_diff % (2 * torch.pi)

            return angle
        else:
            dot_product = (U * v).sum(dim=-1)

            norm_U = torch.linalg.norm(U, dim=-1)
            norm_v = torch.linalg.norm(v, dim=-1)

            cosine = dot_product / (norm_U * norm_v + EPS)
            cosine = torch.clamp(cosine, -1.0, 1.0)

            angle = torch.acos(cosine)

            return angle
    else:
        if len(v.shape) == 1:
            if len(U.shape) == 3:
                v = v[None,None].expand_as(U)
            elif len(U.shape) == 2:
                v = v[None].expand_as(U)
            elif len(U.shape) == 1:
                v = v.expand_as(U)

        EPS = torch.tensor(EPS)
        dot_product = (U * v).sum(dim=-1)
        norm_U = torch.sqrt((U**2).sum(dim=-1))
        norm_v = torch.sqrt((v**2).sum(dim=-1))

        cosine = dot_product / (norm_U * norm_v + EPS)
        cosine = torch.clamp(cosine, -1.0, 1.0)

        angle = torch.acos(cosine)
    
        return angle
    
def angle_to_vector(v: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Rotates a 2D reference vector `v` by a given `angle` (counter-clockwise).

    Given a batch of 2D vectors `v` and a corresponding batch of angles,
    computes the new vectors `U` that result from rotating `v` by `angle`.

    Args:
        v (torch.Tensor): A reference direction vector of shape (..., 2).
                          Commonly (n_vertices, n_angles, 2).
        angle (torch.Tensor): A tensor of shape (..., 1) or (...) containing angles
                              in radians to rotate `v` by. Must be
                              broadcastable with `v`.

    Returns:
        torch.Tensor: The rotated direction vectors `U`, shape (..., 2).
    """
    # Normalize
    v = v / torch.norm(v, dim=-1, keepdim=True)

    # Extract components of v
    vx = v[..., 0]
    vy = v[..., 1]

    # Compute the components of U using rotation matrix
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    Ux = cos_theta * vx - sin_theta * vy
    Uy = sin_theta * vx + cos_theta * vy

    # Stack the components to form the direction vectors
    U = torch.stack((Ux, Uy), dim=-1)

    return U

def calculate_node_normals(vertices: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """
    Calculate vertex normals for a triangulated 3D mesh.

    Computes normals by averaging the normals of adjacent faces.
    It also ensures all normals point "outwards" relative to the
    mesh centroid.

    Args:
        vertices (torch.Tensor): Tensor of shape (N, 3) of vertex coordinates.
        triangles (torch.Tensor): Tensor of shape (M, 3) of triangle indices,
                                  where each row contains 3 indices into `vertices`.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) of normalized vertex normals.
    """
    # Calculate centroid of the mesh
    mesh_centroid = torch.mean(vertices, dim=0)

    # Compute face normals
    v1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    v2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    face_normals = torch.cross(v1, v2, dim=-1)
    
    # Normalize face normals
    face_norms = torch.linalg.norm(face_normals, dim=1, keepdim=True)
    # Avoid division by zero for degenerate faces (zero area)
    face_norms[face_norms == 0] = 1.0
    face_normals /= face_norms

    # Accumulate face normals for each vertex
    vertex_normals = torch.zeros_like(vertices)
    for i in range(3):  # Loop over each vertex of the triangle
        vertex_normals.index_add_(0, triangles[:, i], face_normals)

    # Normalize the vertex normals
    norms = torch.linalg.norm(vertex_normals, dim=1)

    # Handle zero-norm normals (e.g., isolated vertices not in any triangle)
    zero_norm_inds = torch.where(norms == 0)[0]
    if zero_norm_inds.numel() > 0:
        # For isolated vertices, use the vector from the centroid as the normal
        v_out_isolated = vertices[zero_norm_inds] - mesh_centroid
        v_out_isolated_norms = torch.linalg.norm(v_out_isolated, dim=1, keepdim=True)
        # Handle case where vertex IS the centroid
        v_out_isolated_norms[v_out_isolated_norms == 0] = 1.0 
        vertex_normals[zero_norm_inds] = v_out_isolated / v_out_isolated_norms
        norms[zero_norm_inds] = 1.0 # Mark as non-zero for normalization

    # Normalize all non-zero normals
    norms[norms == 0] = 1.0  # Avoid any division by zero
    vertex_normals /= norms.unsqueeze(1)

    # Enforce outward direction using the centroid
    v_out = vertices - mesh_centroid
    dot_products = torch.einsum('ij,ij->i', vertex_normals, v_out)
    flip_mask = dot_products < 0
    vertex_normals[flip_mask] *= -1

    return vertex_normals

def normpdf(x: torch.Tensor, mean: float, sigma: float) -> torch.Tensor:
    """
    Compute the normal (Gaussian) probability density function.
    
    Args:
        x (torch.Tensor): The input values (e.g., distances).
        mean (float): The mean (center) of the Gaussian distribution.
        sigma (float): The standard deviation (spread) of the distribution.
    
    Returns:
        torch.Tensor: Probability density values corresponding to `x`.
    """
    coeff = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))
    exponent = torch.exp(-0.5 * ((x - mean) / sigma) ** 2)
    return coeff * exponent

def form_gaussian_density(pos: torch.Tensor, POI: int, steepness: float) -> torch.Tensor:
    """
    Calculates a Gaussian density centered around a point of interest (POI).

    The density is normalized to have a maximum value of 1.

    Args:
        pos (torch.Tensor): Tensor of spatial coordinates, shape (n_points, n_dim).
        POI (int): The index (row in `pos`) of the point of interest.
        steepness (float): The standard deviation (sigma) of the Gaussian.

    Returns:
        torch.Tensor: A 1D tensor of density values, shape (n_points,).
    """
    distance_to_POI = ((pos[POI,:]-pos)**2).sum(dim=-1).sqrt()

    density = normpdf(distance_to_POI, 0.0, steepness)
    density = density / density.max()
    return density

def compute_bounds(k: torch.Tensor, u: torch.Tensor, ci: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute confidence interval bounds using the Lambert W function.

    This function is used to translate response probabilities (confidence
    intervals) back into stimulus magnitude bounds.

    Args:
        k (torch.Tensor): Slope parameter for the response function.
        u (torch.Tensor): Directional translation estimate.
        ci (torch.Tensor): A 1D tensor of confidence interval thresholds
                           (e.g., [0, 0.25, 0.5, 0.75, 1.0]).

    Returns:
        List[Tuple[torch.Tensor, torch.Tensor]]:
            A list of (lower_bound, upper_bound) tuples. Each bound
            is a tensor with the same shape as `k` and `u`.
    """
    INFINTE_BOUND_MULTIPLIER = 10  # Last bound (infinity) is approximated as finite for visualization

    # Clamp k*u before exponentiating to prevent float overflow
    exp_max = 600
    exp_ku = torch.exp(torch.clamp((k * u).double(), max=exp_max))

    bounds = []
    previous_upper = None
    for i in range(len(ci) - 2):    # Leave out the last interval which extends to infinity
        if i == 0:
            lower_bound = torch.real(lambertw(-k * exp_ku * ci[i] / (ci[i] - 1))) / k
        else:
            lower_bound = previous_upper
        upper_bound = torch.real(lambertw(-k * exp_ku * ci[i + 1] / (ci[i + 1] - 1))) / k
        bounds.append((lower_bound.float(), upper_bound.float()))
        previous_upper = upper_bound

    # Calculate last bound
    lower_bound = previous_upper
    upper_bound = torch.full_like(lower_bound, INFINTE_BOUND_MULTIPLIER * torch.max(u))
    bounds.append((lower_bound, upper_bound))
    return bounds

def compute_half_probability_bounds(k: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Computes the stimulus magnitude corresponding to a 50% response probability.

    This is a convenience wrapper for `compute_bounds` using intervals [0, 0.5, 1].

    Args:
        k (torch.Tensor): Slope parameter for the response function.
        u (torch.Tensor): Directional translation estimate.

    Returns:
        torch.Tensor: The stimulus magnitude (upper bound of the first interval)
                      at which the response probability is 0.5.
    """
    return compute_bounds(k,u,torch.tensor([0,0.5,1]))[0][1]

def get_compass_rose_coords(
    all_vertices: torch.Tensor, 
    roi_indices: torch.Tensor, 
    efield_basis: Optional[torch.Tensor], 
    null_dir_basis: Optional[torch.Tensor], 
    roi_vertex_index: int, 
    radius: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Calculates the 3D coordinates for the compass rose geometry.

    Args:
        all_vertices (torch.Tensor): All mesh vertices (N, 3).
        roi_indices (torch.Tensor): Indices of the Region of Interest (M,).
        efield_basis (Optional[torch.Tensor]): E-field basis vectors (M, 3, 2).
        null_dir_basis (Optional[torch.Tensor]): Null direction basis (1, M, 2).
        roi_vertex_index (int): The ROI index of the vertex.
        radius (float): The radius of the compass rose in 3D space.

    Returns:
        A dictionary containing the x, y, and z coordinates (as np.ndarray)
        for the line trace, or an empty dict if basis is None.
    """
    if efield_basis is None or null_dir_basis is None:
        return {}

    # Setup: Get position, basis vectors, and null direction
    center = all_vertices[roi_indices[roi_vertex_index]].cpu()
    basis = efield_basis[roi_vertex_index].cpu()
    null_dir_2d = null_dir_basis[0, roi_vertex_index].cpu()
    nan_separator = torch.full((1, 3), float('nan')) # Used to create gaps in a single line trace

    # Lift the compass slightly off the mesh for visibility
    normal_vec = torch.cross(basis[:, 0], basis[:, 1], dim=-1)
    normal_vec /= torch.norm(normal_vec)
    lift_distance = radius * 0.5
    lifted_center = center + lift_distance * normal_vec

    # 0-Degree Reference Marker (a small circle)
    marker_radius = radius * 0.15
    zero_deg_dir_3d = null_dir_2d @ basis.T
    marker_center_3d = lifted_center + radius * zero_deg_dir_3d
    theta_marker = torch.linspace(0, 2 * np.pi, 31)
    marker_points_2d = torch.stack([torch.cos(theta_marker), torch.sin(theta_marker)], dim=1)
    zero_deg_marker_3d = marker_center_3d + marker_radius * (marker_points_2d @ basis.T)

    # Compass Border (a circle with a gap for the marker)
    gap_angle_rad = 2 * torch.asin(torch.tensor(marker_radius / radius)) * 1.2
    zero_angle_rad = torch.atan2(null_dir_2d[1], null_dir_2d[0])
    start_angle, end_angle = zero_angle_rad + gap_angle_rad / 2, zero_angle_rad - gap_angle_rad / 2 + 2 * np.pi
    theta_border = torch.linspace(start_angle, end_angle, 100)
    border_points_2d = torch.stack([torch.cos(theta_border), torch.sin(theta_border)], dim=1)
    border_points_3d = lifted_center + radius * (border_points_2d @ basis.T)

    # Rotation Direction Indicator (a tangential arrow)
    indicator_angle_rad = np.pi / 2
    tip_angle_tensor = torch.tensor([indicator_angle_rad])
    tip_vec_2d = angle_to_vector(null_dir_2d.unsqueeze(0), tip_angle_tensor).squeeze() # Use _utils.angle_to_vector
    tangent_vec, radial_vec = torch.tensor([-tip_vec_2d[1], tip_vec_2d[0]]), tip_vec_2d
    arrow_len, arrow_width = 0.3, 0.25
    base1 = tip_vec_2d - arrow_len * tangent_vec + arrow_width * radial_vec
    base2 = tip_vec_2d - arrow_len * tangent_vec - arrow_width * radial_vec
    indicator_points_2d = torch.stack([base1, tip_vec_2d, base2])
    indicator_points_3d = lifted_center + radius * (indicator_points_2d @ basis.T)

    # Combine all line segments into a single array for one trace
    line_points = torch.cat([
        border_points_3d, nan_separator, zero_deg_marker_3d,
        nan_separator, indicator_points_3d
    ], dim=0)

    return {
        'line_x': line_points[:, 0].numpy(),
        'line_y': line_points[:, 1].numpy(),
        'line_z': line_points[:, 2].numpy(),
    }

def get_rf_plot_data(param_dict: Dict[str, Any], e_mags: torch.Tensor, angles: torch.Tensor, responses: torch.Tensor, point_number: int) -> Dict[str, Any]:
    """
    Calculates all necessary data for a response function (RF) polar plot.

    This function can be called by both core.py (to pre-compute) and
    plotting_app.py (if needed for dynamic updates). It computes the
    probability bounds and collates the raw response data for plotting.

    Args:
        param_dict (Dict[str, Any]): Dictionary containing model parameters
                                     (e.g., 'rf_translate', 'rf_slope').
        e_mags (torch.Tensor): Tensor of stimulus magnitudes for all trials.
        angles (torch.Tensor): Tensor of stimulus angles for all trials.
        responses (torch.Tensor): Tensor of binary responses (0 or 1) for all trials.
        point_number (int): The index of the vertex/point to compute plot data for.

    Returns:
        Dict[str, Any]: A dictionary containing all data needed for the plot:
            - 'bounds': List of (lower, upper) bound tuples.
            - 'responses_r': List of magnitudes for positive responses.
            - 'responses_theta': List of angles for positive responses.
            - 'no_responses_r': List of magnitudes for negative responses.
            - 'no_responses_theta': List of angles for negative responses.
    """
    COL_STEPS = 5

    angles_rad = torch.linspace(0, 2 * torch.pi, 100)
    directional_translation = (
        param_dict['rf_translate']['values'][point_number].cpu() +
        param_dict['rf_fourier1']['values'][point_number].cpu() * torch.cos(angles_rad) +
        param_dict['rf_fourier2']['values'][point_number].cpu() * torch.sin(angles_rad) +
        param_dict['rf_fourier3']['values'][point_number].cpu() * torch.cos(angles_rad * 2) +
        param_dict['rf_fourier4']['values'][point_number].cpu() * torch.sin(angles_rad * 2)
    )
    k = param_dict['rf_slope']['values'][point_number].cpu()
    
    ci = torch.linspace(0.0, 1.0, COL_STEPS + 1)
    bounds = compute_bounds(k, directional_translation, ci)

    # Package everything needed for the plot
    plot_data = {
        'bounds': bounds, # List of lower/upper bounds
        'responses_r': e_mags[responses == 1, point_number].numpy().tolist(),
        'responses_theta': angles[responses == 1, point_number].numpy().tolist(),
        'no_responses_r': e_mags[responses == 0, point_number].numpy().tolist(),
        'no_responses_theta': angles[responses == 0, point_number].numpy().tolist()
    }
    return plot_data

def get_prior(pos: torch.Tensor, POI: Optional[int] = None, summit_slope: Optional[float] = None) -> torch.Tensor:
    """
    Set uninformative or informative prior source probabilities.

    Args:
        pos (torch.Tensor): Tensor of shape (n_ROI_vertices, 3) of vertex coordinates.
        POI (Optional[int]): If provided, centers an informative Gaussian
                             prior at this vertex index. If None, an
                             uninformative (flat) prior is returned.
        summit_slope (Optional[float]): The standard deviation (sigma) for the
                                        Gaussian prior. If None and POI is set,
                                        it is automatically calculated based on
                                        the spatial extent of `pos`.

    Returns:
        torch.Tensor: Tensor of shape (n_ROI_vertices,) of prior source
                      probabilities (sums to 1).
    """
    WIDTH_SCALAR = 1/2  # Gaussian steepness in relation to ROI width

    if POI is None:
        # Uninformative flat prior
        n_points = len(pos)
        prior = torch.ones(n_points) / n_points
    else:
        if summit_slope is None:
            # Calculate gaussian size based on (approximate) ROI width
            min_x = torch.argmin(pos[:,0])
            max_x = torch.argmax(pos[:,0])
            min_y = torch.argmin(pos[:,1])
            max_y = torch.argmax(pos[:,1])
            dist_x = torch.norm(pos[min_x,:]-pos[max_x,:],dim=-1)
            dist_y = torch.norm(pos[min_y,:]-pos[max_y,:],dim=-1)
            if dist_x > dist_y:
                max_dist = dist_x
            else:
                max_dist = dist_y
            summit_slope = max_dist * WIDTH_SCALAR

        density = form_gaussian_density(pos,POI,summit_slope)
        prior = density / density.sum()
    return prior

def format_ticks(vals: Iterable[float]) -> List[str]:
    """
    Formats a list of numeric tick values into human-readable strings.

    Uses scientific notation for very small numbers and appropriate
    precision for others.

    Args:
        vals (Iterable[float]): A list or iterable of float values.

    Returns:
        List[str]: A list of formatted string labels.
    """
    tick_list = []
    for val in vals:
        abval = abs(val)
        if abval == 0:
            tick_list.append("0")
        elif abval > 1:
            tick_list.append(f"{val:.0f}")
        elif (abval < 1) and (abval > 0.00999999):
            tick_list.append(f"{val:.2f}")
        elif (abval < 0.00999999) and (abval > 0.000999999):
            tick_list.append(f"{val:.3f}")
        elif (abval < 0.000999999):
            tick_list.append(f"{val:.0e}")
    return tick_list

def compute_cumulative_prob(source_prob: np.ndarray) -> np.ndarray:
    """For each vertex, return the fraction of total probability mass in
    vertices with probability <= this vertex's probability (CDF from below).
    Hotspot -> 1.0, diffuse background -> ~0.0."""
    total = source_prob.sum()
    if total == 0:
        return np.zeros_like(source_prob, dtype=float)
    sorted_idx = np.argsort(source_prob)
    cumsum = np.cumsum(source_prob[sorted_idx]) / total
    cum_prob = np.empty_like(source_prob, dtype=float)
    cum_prob[sorted_idx] = cumsum
    return cum_prob


def mesh_ind_to_ROI_ind(mesh_ind: int, ROI: torch.Tensor) -> Optional[int]:
    """
    Transforms a full mesh vertex index to its corresponding ROI index.

    The Region of Interest (ROI) is a subset of the full mesh. This
    function finds the position of a global `mesh_ind` within the
    `ROI` index tensor.

    Args:
        mesh_ind (int): The vertex index from the full mesh.
        ROI (torch.Tensor): A 1D tensor of indices that define the
                            Region of Interest (a subset of the full mesh).

    Returns:
        Optional[int]: The index of `mesh_ind` *within* the `ROI` tensor.
                       Returns None if `mesh_ind` is not in the `ROI`.
    """
    ind = torch.where(ROI == mesh_ind)[0]
    if ind.numel() == 0:
        print("Point is not in the region-of-interest!")
        return None
    else:
        return ind.item()

def load_from_hdf5(file_path: str) -> Dict[str, Any]:
    """
    Load an HDF5 file into a nested dictionary, converting arrays to torch tensors.

    Recursively traverses the HDF5 file structure. Datasets are loaded
    as torch.Tensors, and groups are loaded as nested dictionaries.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        Dict[str, Any]: A nested dictionary containing the HDF5 file contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If any other error occurs during file loading.
    """
    def _load_group(h5_group: h5py.Group) -> Dict[str, Any]:
        """Helper function to recursively load HDF5 groups"""
        result = {}
        for key in h5_group.keys():
            item = h5_group[key]
            if isinstance(item, h5py.Group):
                # If it's a group, recursively load its contents
                result[key] = _load_group(item)
            elif isinstance(item, h5py.Dataset):
                # Get the data as a numpy array first
                data = item[()]
                # Convert to torch tensor if it was originally a tensor-compatible type
                if isinstance(data, np.ndarray):
                    result[key] = torch.from_numpy(data)
                else:
                    # For scalar values or other types, keep as is
                    result[key] = data
                    # Convert numpy scalar types to Python native types if possible
                    if hasattr(data, 'item'):
                        result[key] = data.item()
        return result

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Open the HDF5 file and load its contents
    try:
        with h5py.File(file_path, 'r') as f:
            return _load_group(f)
    except Exception as e:
        raise Exception(f"Error loading HDF5 file: {str(e)}")

