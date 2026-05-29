"""Component for visualization using Plotly and Dash.

This module provides the Visualizer class, which is responsible for creating,
updating, and serving an interactive Dash application for real-time
experiment visualization. It is designed to be managed by the LocalizationSession.
"""

# Standard library imports
import atexit
import json
import os
import subprocess
import sys
import sysconfig
import time
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.subplots import make_subplots

# Local application imports
from ._internal import _utils

# Type Aliases for clarity
Tensor = torch.Tensor
Figure = go.Figure


class Visualizer:
    """Manages all plotting and runs the interactive Dash application.

    This class encapsulates all Plotly and Dash functionalities. It creates
    the main figure, launches a Dash server in a separate process to display it,
    and handles real-time updates by writing data to JSON files that the Dash
    app monitors.

    Attributes:
        save_dir (str): The directory for saving figures, logs, and update files.
        dash_process (Optional[subprocess.Popen]): The running Dash server process.
        figure (Optional[Figure]): The main Plotly Figure object.
        last_fig_mod_time (float): Timestamp of the last figure file modification,
            used to detect and reload external changes (e.g., user interactions).
        plot_dict (Dict[int, Tuple[str, List[int]]]): Maps a plot's configuration
            index to its type and the list of its trace indices in the figure.
        rf_axis_name (str): Polar axis name (e.g. 'polar') of the single
            response-function plot, used for layout updates.
        rf_colorbar_trace_index (int): Index of the invisible trace that
            carries the response-probability colorbar.
        _cleanup_registered (bool): A flag to ensure the atexit cleanup
            handler is registered only once.
        stdout_file: File handle for the Dash process's standard output log.
        stderr_file: File handle for the Dash process's standard error log.
    """
    def __init__(self):
        """Initializes the Visualizer instance."""
        self.save_dir: str = ""
        self.dash_process: Optional[subprocess.Popen] = None
        self.figure: Optional[Figure] = None
        self.last_fig_mod_time: float = 0.0
        self.plot_dict: Dict[int, Tuple[str, List[int]]] = {}
        self.rf_axis_name: str = 'polar'
        self.rf_colorbar_trace_index: int = -1
        self._cleanup_registered: bool = False
        self.stdout_file = None
        self.stderr_file = None

    def start_plotter(self):
        """Launches the Dash application as a separate, non-blocking process.

        This method registers a cleanup function to stop the server on exit,
        then starts the server. It redirects the server's stdout and stderr
        to log files in the session's save directory.

        Raises:
            ValueError: If the save directory has not been set or does not exist.
            RuntimeError: If the Dash server process fails to start.
        """
        if not self._cleanup_registered:
            atexit.register(self.stop_plotter)
            self._cleanup_registered = True

        # Do not start a new process if one is already running
        if self.dash_process and self.dash_process.poll() is None:
            return

        if not self.save_dir or not os.path.isdir(self.save_dir):
            raise ValueError(f"The save directory '{self.save_dir}' does not exist.")

        # Path to the internal Dash application module
        module_path = "ABL._internal._plotting_app"
        self.stdout_file = open(os.path.join(self.save_dir, "dash_stdout.log"), "w")
        self.stderr_file = open(os.path.join(self.save_dir, "dash_stderr.log"), "w")

        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(current_file_dir))
        src_root = os.path.join(project_dir, "src")
        site_packages_path = sysconfig.get_paths()['purelib']

        env = os.environ.copy()

        python_path_list = [site_packages_path,src_root]

        new_pythonpath = os.pathsep.join(python_path_list)

        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = new_pythonpath + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = new_pythonpath

            self.dash_process = subprocess.Popen(
                [sys.executable, "-m", module_path, self.save_dir],
                stdout=self.stdout_file,
                stderr=self.stderr_file,
                env=env,
            )

        time.sleep(2)  # Wait briefly to allow the server to start

        if self.dash_process.poll() is not None:
            raise RuntimeError("Dash process terminated unexpectedly. Check logs in the save directory.")
        print("Plotting on http://127.0.0.1:8050/")

    def stop_plotter(self):
        """Stops the running Dash server process and closes log files."""
        if self.dash_process and self.dash_process.poll() is None:
            self.dash_process.terminate()
            try:
                self.dash_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.dash_process.kill()
                self.dash_process.wait()

        if hasattr(self, 'stdout_file') and self.stdout_file and not self.stdout_file.closed:
            self.stdout_file.close()
        if hasattr(self, 'stderr_file') and self.stderr_file and not self.stderr_file.closed:
            self.stderr_file.close()
        self.dash_process = None

    def plot(self, plot_data: Dict[str, Any], re_initialize: bool):
        """Creates or updates the main figure using the provided data.

        Args:
            plot_data: A dictionary containing all necessary data for plotting.
            re_initialize: If True, creates a new figure from scratch. If False,
                updates the data of the existing figure.
        """
        if re_initialize:
            self.figure = self._initialize_figure(plot_data)
            # Write the initial full figure to a JSON file for the Dash app
            filepath = os.path.join(self.save_dir, 'figure.plotly')
            pio.write_json(self.figure, filepath)
            self.last_fig_mod_time = os.stat(filepath).st_mtime
        else:
            self._update_figure_data(plot_data)

    def save_figure(self, filename: str, scale: int, plot_data: Dict[str, Any]):
        """Saves the current Plotly figure as a high-resolution PDF image.

        Args:
            filename: The base name for the output file (e.g., "final_figure").
            scale: The image scaling factor for higher resolution.
        """
        fig = self._initialize_figure(plot_data)
        try:
            output_path = os.path.join(self.save_dir, f"{filename}.pdf")    # PDF preseves colors well
            fig.write_image(output_path, scale=scale)
            print(f"Figure saved to {output_path}")
        except Exception as e:
            print(f"Error saving figure: {e}")

    def _initialize_figure(self, plot_data: Dict[str, Any]):
        """Builds the complete initial Plotly figure, including layout and all traces.

        This method orchestrates the creation of the entire figure by:
        1. Defining the plot configurations.
        2. Calculating the subplot layout.
        3. Adding all specified plot traces (meshes, lines, etc.).
        4. Applying final styling and annotations.
        5. Saving the complete figure object to a file for the Dash app.

        Args:
            plot_data: A dictionary with all data required for the initial plot.
        """
        plot_config = self._get_plot_config(plot_data)
        layout_config = self._calculate_figure_layout(plot_config, plot_data)

        fig = go.Figure(make_subplots(**layout_config['subplot_args']))

        # Add Traces
        self.plot_dict = {}
        col_idx = 1
        for i, config in enumerate(plot_config):
            plot_type = config['type']
            data = self._prepare_plot_data(config, plot_data)

            plot_function_map = {
                'mesh': self._add_mesh_plot,
                'arrow_mesh': self._add_arrow_mesh_plot,
                'loss': self._add_loss_plot,
                'rf': self._add_rf_plot
            }

            if plot_type in plot_function_map:
                plot_inds = plot_function_map[plot_type](fig, data, config, layout_config, plot_data, col_idx)
                self.plot_dict[i] = (plot_type, plot_inds)
                col_idx += 1

        self._apply_figure_styling(fig, layout_config, plot_data)
        return fig

    def _update_figure_data(self, plot_data: Dict[str, Any]):
        """Generates and saves a JSON payload with only the data that has changed.

        This method avoids redrawing the entire figure. It checks if the figure
        file was modified externally (e.g., by user interaction) and reloads it.
        Then, it generates a lightweight JSON file containing only the updated
        data and layout properties, which the Dash app applies to the figure.

        Args:
            plot_data: A dictionary with fresh data to update the plots.
        """
        filepath = os.path.join(self.save_dir, 'figure.plotly')
        update_filepath = os.path.join(self.save_dir, 'figure_update.json')

        update_payload = {"data_updates": [], "layout_updates": []}

        # Reload figure if it was modified to sync camera state, etc.
        self._reload_figure_if_modified(filepath)

        plot_config = self._get_plot_config(plot_data)

        # Generate update payloads for each plot type
        for data_index, (plot_type, plot_indices) in self.plot_dict.items():
            config = plot_config[data_index]
            data = self._prepare_plot_data(config, plot_data)

            update_function_map = {
                'mesh': self._get_mesh_update_payload,
                'arrow_mesh': self._get_arrow_mesh_update_payload,
                'loss': self._get_loss_update_payload,
                'rf': self._get_rf_update_payload
            }

            if plot_type in update_function_map:
                updates = update_function_map[plot_type](plot_indices, data, config, plot_data)
                if plot_type == 'rf':
                    # RF updates can return both data and layout changes
                    update_payload["data_updates"].extend(updates.get("data_updates", []))
                    update_payload["layout_updates"].extend(updates.get("layout_updates", []))
                else:
                    update_payload["data_updates"].extend(updates)

        self._update_annotations_payload(update_payload, plot_data)

        with open(update_filepath, 'w') as f:
            json.dump(update_payload, f)

    def _get_plot_config(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Defines the configuration for all plots in the figure.

        Each dictionary in the returned list specifies a plot's type, data source,
        and title, forming the blueprint for figure construction.

        Args:
            plot_data: The main data dictionary from the session.

        Returns:
            A list of dictionaries, where each configures one plot component.
        """
        # Pad loss tensors to the same length for plotting
        next_action_loss = plot_data['optimizer_losses']
        param_update_loss = plot_data['response_model_losses']
        max_len = max(next_action_loss.size(0), param_update_loss.size(0))

        def pad_loss(loss_tensor, length):
            pad_size = length - loss_tensor.size(0)
            if pad_size > 0:
                # Pad with the last value to extend the line flat
                return torch.nn.functional.pad(loss_tensor, (0, pad_size), 'constant', loss_tensor[-1])
            return loss_tensor

        loss_data = torch.stack((
            pad_loss(next_action_loss, max_len),
            pad_loss(param_update_loss, max_len)
        ), dim=0)

        return [
            {'type': 'mesh', 'data_key': 'source_prob', 'title': 'Source probability'},
            {'type': 'rf', 'data_key': 'param_dict', 'title': ['']},
            {'type': 'arrow_mesh', 'data_key': 'latest_efield', 'title': 'Latest E-field (V/m)'},
            #{'type': 'loss', 'data': loss_data, 'title': ['E-field optimization', 'Model fitting']},
        ]

    def _prepare_plot_data(self, config: Dict[str, Any], plot_data: Dict[str, Any]) -> Any:
        """Extracts and formats data for a specific plot from the main data package.

        For mesh plots, this function also pads data defined on the ROI to the
        full cortical mesh size, ensuring it can be mapped correctly to the vertices.

        Args:
            config: The configuration dictionary for the specific plot.
            plot_data: The main data dictionary from the session.

        Returns:
            The prepared data (e.g., a tensor or dict) ready for plotting.
        """
        if config['type'] == 'loss':
            raw_data = config['data']
        else:
            raw_data = plot_data[config['data_key']]

        if config['type'] == 'rf':
            # RF data is a dict of tensors; convert to float CPU tensors
            return {
                key: {'values': value['values'].float().cpu()}
                for key, value in raw_data.items()
            }

        # For other plot types, data is assumed to be a single tensor
        data = raw_data.float().cpu()

        if config['type'] in ['mesh', 'arrow_mesh']:
            # Pad ROI data to the full mesh size
            is_vector_data = (config['type'] == 'arrow_mesh')
            target_shape = plot_data['mesh_vertices'].shape if is_vector_data else (plot_data['mesh_vertices'].shape[0],)
            padded_data = torch.zeros(target_shape)
            padded_data[plot_data['mesh_ROI']] = data
            return padded_data

        return data

    def _calculate_figure_layout(self, plot_config: List[Dict[str, Any]], plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates figure layout specifications based on the plot configuration.

        Fixed 1-row layout with one column per plot, in plot_config order.

        Args:
            plot_config: The list of plot configuration dictionaries.
            plot_data: The main data dictionary (unused; kept for signature compatibility).

        Returns:
            A dictionary containing all layout parameters for `make_subplots`.
        """
        n_rows = 1
        n_cols = len(plot_config)

        type_to_spec = {
            'mesh': {'type': 'surface'},
            'arrow_mesh': {'type': 'surface'},
            'loss': {'type': 'xy', 'secondary_y': True},
            'rf': {'type': 'scatterpolar'},
        }

        specs = [[None] * n_cols]
        titles = [""] * n_cols
        for col_idx, config in enumerate(plot_config):
            specs[0][col_idx] = type_to_spec[config['type']]
            if config['type'] == 'rf':
                titles[col_idx] = ''
            elif config['type'] == 'loss':
                titles[col_idx] = 'Optimization convergence'

        hor_spacing = 0.03
        col_width = (1.0 - (n_cols - 1) * hor_spacing) / n_cols
        cbar_locs = [(i * (col_width + hor_spacing)) + col_width / 2.0 for i in range(n_cols)]

        return {
            "n_rows": n_rows, "n_cols": n_cols,
            "row_height": 1.0, "vert_spacing": 0.08,
            "cbar_len": col_width,
            "cbar_locs": cbar_locs,
            "subplot_args": {
                "rows": n_rows, "cols": n_cols,
                "horizontal_spacing": hor_spacing,
                "vertical_spacing": 0.08,
                "subplot_titles": titles,
                "specs": specs
            }
        }

    def _add_mesh_plot(self, fig: Figure, data: Tensor, config: Dict, layout_config: Dict, plot_data: Dict[str, Any], col_idx: int) -> List[int]:
        """Adds a 3D cortical mesh plot to the figure.

        This can include the mesh colored by intensity, markers for the true
        source, and a 'compass rose' to indicate E-field directionality.

        Args:
            fig: The main `go.Figure` object to add traces to.
            data: The intensity data tensor to map onto the mesh vertices.
            config: The configuration dictionary for this plot.
            layout_config: The main layout configuration dictionary.
            plot_data: The main data dictionary from the session.
            col_idx: The column index for this subplot.

        Returns:
            A list of indices for all traces added by this function.
        """
        trace_indices = []
        source = plot_data.get('source')
        mesh_pos = plot_data['mesh_vertices']
        mesh_tri = plot_data['mesh_faces']

        is_source_prob = (config['title'] == 'Source probability')
        mesh_name = 'source_prob_mesh' if is_source_prob else 'efield_mesh'

        if is_source_prob:
            intensity = _utils.compute_cumulative_prob(np.asarray(data))
            colorscale = 'Viridis'
            cmin, cmax = 0.0, 1.0
            tickvals = [0.0, 0.5, 1.0]
            ticktext = ['0%', '50%', '100%']
            cbar_title = 'Cumulative probability'
        else:
            intensity = data
            colorscale = 'Turbo'
            cmin = 0.0
            cmax = data.max().item()
            tickvals = [0, cmax / 2, cmax]
            ticktext = _utils.format_ticks(tickvals)
            cbar_title = config['title']

        # Add main mesh trace with intensity coloring
        fig.add_trace(go.Mesh3d(
            x=mesh_pos[:, 0], y=mesh_pos[:, 1], z=mesh_pos[:, 2],
            i=mesh_tri[:, 0], j=mesh_tri[:, 1], k=mesh_tri[:, 2],
            intensity=intensity,
            colorscale=colorscale,
            intensitymode='vertex',
            flatshading=False,
            showscale=True,
            name=mesh_name,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                len=layout_config['cbar_len'],
                x=layout_config['cbar_locs'][col_idx-1],
                y=1.0,
                xanchor='center',
                title=dict(text=cbar_title, side='top', font=dict(size=26)),
                bgcolor='rgba(255, 255, 255, 0.0)',
                thickness=25,
                outlinecolor='#000000',
                outlinewidth=2,
                tickvals=tickvals,
                ticktext=ticktext,
            )
        ), row=1, col=col_idx)
        trace_indices.append(len(fig.data) - 1)

        # Add marker for the true source location, if it exists
        if config['title'] == 'Source probability' and source is not None:
            source_pos = mesh_pos[plot_data['mesh_ROI'][source['index']]]
            fig.add_trace(go.Scatter3d(
                x=source_pos[:, 0], y=source_pos[:, 1], z=source_pos[:, 2],
                mode='markers',
                marker=dict(size=10, color='red', sizemode='diameter'),
                showlegend=True,
                legend='legend2',
                name='source',
                opacity=1
            ), row=1, col=col_idx)
            trace_indices.append(len(fig.data) - 1)

        # Add invisible anchor points to force wider clipping planes (fixes visual artifacts when saving figure)
        vertices = plot_data['mesh_vertices']
        min_bounds = vertices.min(dim=0).values.numpy()
        max_bounds = vertices.max(dim=0).values.numpy()

        # Add 25% padding to the bounding box
        padding = 0.25 * (max_bounds - min_bounds)
        corners = np.array([min_bounds - padding, max_bounds + padding])

        fig.add_trace(go.Scatter3d(
            x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
            mode='markers',
            marker=dict(size=0, color='white', opacity=0), # Invisible
            showlegend=False,
            hoverinfo='skip', # Don't show up on hover
            name='clipping_anchor'
        ), row=1, col=col_idx)
        trace_indices.append(len(fig.data) - 1)

        # Add Compass Rose to indicate E-field basis at the likeliest source (must be last trace index)
        if config['title'] == 'Source probability':
            likeliest_source_index = torch.argmax(plot_data['source_prob']).item()
            compass_traces = self._create_compass_rose_traces(fig, likeliest_source_index, plot_data, col_idx)
            trace_indices.extend(compass_traces)

        return trace_indices

    def _create_compass_rose_traces(self, fig: go.Figure, vertex_index: int, plot_data: Dict[str, Any], col_idx: int, row_idx: int = 1) -> List[int]:
        """Generates and adds Plotly traces for a 'compass rose' visualization.

        The compass rose is a 3D indicator placed on the mesh surface to show
        the 2D E-field basis (orientation and rotation direction) at a point.

        Args:
            fig: The main `go.Figure` object.
            vertex_index: The ROI index of the vertex to place the compass on.
            plot_data: The main data dictionary.
            col_idx: The subplot column index.
            row_idx: The subplot row index.

        Returns:
            A list of indices for the traces added.
        """
        if plot_data['efield_basis'] is None:
            return []

        trace_indices = []
        coords = _utils.get_compass_rose_coords(
            plot_data['mesh_vertices'],
            plot_data['mesh_ROI'],
            plot_data['efield_basis'],
            plot_data['null_dir_basis'],
            vertex_index
        )
        if not coords:
            return []

        # Add a single scatter trace containing all line segments for the compass
        fig.add_trace(go.Scatter3d(
            x=coords['line_x'], y=coords['line_y'], z=coords['line_z'],
            mode='lines',
            line=dict(color='white', width=10),
            opacity=0.3,
            hoverinfo='none',  # Make the trace non-interactive
            showlegend=False,
            name='compass_line'
        ), row=row_idx, col=col_idx)
        trace_indices.append(len(fig.data) - 1)

        return trace_indices

    def _add_arrow_mesh_plot(self, fig: Figure, data: Tensor, config: Dict, layout_config: Dict, plot_data: Dict[str, Any], col_idx: int) -> List[int]:
        """Adds a 3D mesh plot with cones representing a vector field (e.g., E-field).

        Args:
            fig: The main `go.Figure` object.
            data: The vector data tensor (N, 3) to map onto the mesh.
            config: The configuration dictionary for this plot.
            layout_config: The main layout configuration dictionary.
            plot_data: The main data dictionary from the session.
            col_idx: The column index for this subplot.

        Returns:
            A list of indices for all traces added.
        """
        # First, add the base mesh colored by vector magnitude
        mesh_trace_inds = self._add_mesh_plot(fig, torch.norm(data, dim=-1), config, layout_config, plot_data, col_idx)

        data_norm = torch.norm(data, dim=-1, keepdim=True)
        data_normalized = data / data_norm
        data_normalized[~torch.isfinite(data_normalized)] = 0 # Handle potential division by zero

        ROI_pos = plot_data['mesh_vertices'][plot_data['mesh_ROI']]
        coarse_inds = plot_data['mesh_coarse_inds'] # Use a coarse subset for clarity

        # Add cones to represent vector directions
        fig.add_trace(go.Cone(
            x=ROI_pos[coarse_inds, 0], y=ROI_pos[coarse_inds, 1], z=ROI_pos[coarse_inds, 2],
            u=data_normalized[coarse_inds, 0], v=data_normalized[coarse_inds, 1], w=data_normalized[coarse_inds, 2],
            sizemode='absolute', sizeref=0.5, showscale=False,
            colorscale=[[0, 'black'], [1, 'black']],
            lighting=dict(ambient=0, diffuse=1, fresnel=0, specular=0, roughness=1)
        ), row=1, col=col_idx)
        mesh_trace_inds.append(len(fig.data) - 1)

        return mesh_trace_inds

    def _add_loss_plot(self, fig: Figure, data: Tensor, config: Dict, layout_config: Dict, plot_data: Dict[str, Any], col_idx: int) -> List[int]:
        """Adds a 2D line plot for optimization loss curves.

        Args:
            fig: The main `go.Figure` object.
            data: A tensor containing loss curves for different optimizations.
            config: The configuration dictionary for this plot.
            layout_config: The main layout configuration dictionary.
            plot_data: The main data dictionary from the session.
            col_idx: The column index for this subplot.

        Returns:
            A list of indices for all traces added.
        """
        plot_inds = []
        titles = config['title']

        # E-field optimization loss on the primary y-axis
        fig.add_trace(go.Scatter(
            x=np.arange(data.shape[-1]), y=data[0, :].numpy(),
            mode='lines+markers', name=titles[0], line=dict(color="#636EFA")
        ), secondary_y=False, row=1, col=col_idx)
        plot_inds.append(len(fig.data) - 1)

        # Model fitting loss on the secondary y-axis
        fig.add_trace(go.Scatter(
            x=np.arange(data.shape[-1]), y=data[1, :].numpy(),
            mode='lines+markers', name=titles[1], line=dict(color="#EF553B")
        ), secondary_y=True, row=1, col=col_idx)
        plot_inds.append(len(fig.data) - 1)

        return plot_inds

    def _create_polar_style(self) -> Dict[str, Any]:
        """Build the shared confidence-band color/angle style used by the RF plot."""
        COL_STEPS = 5
        ALPHA = 0.8
        ci = torch.linspace(0.0, 1.0, COL_STEPS + 1)
        colors = px.colors.sample_colorscale('teal', torch.linspace(0.0, 0.7, COL_STEPS + 1).numpy())
        colors.reverse()
        rgba_colors = [c.replace('rgb', 'rgba').replace(')', f', {ALPHA})') for c in colors]
        disc_c_scale = []
        for k in range(COL_STEPS):
            disc_c_scale.append([k / COL_STEPS, rgba_colors[k]])
            disc_c_scale.append([(k + 1) / COL_STEPS, rgba_colors[k]])
        angles_rad = torch.linspace(0, 2 * torch.pi, 100)
        theta_vals = torch.cat((angles_rad, angles_rad.flip(0), angles_rad[0:1])).numpy()
        return {'ci': ci, 'colors': rgba_colors, 'disc_c_scale': disc_c_scale, 'theta_vals': theta_vals}

    def _add_estimated_rf_traces(self, fig: Figure, rf_data: Dict, style: Dict, row: int, col: int, meta_id: str) -> List[int]:
        """Add the 5 confidence-band traces and 2 scatter traces (no-resp / resp) for one RF plot."""
        trace_indices = []
        meta_data = {'id': meta_id}
        for i, (lower, upper) in enumerate(rf_data['bounds']):
            r_vals = torch.cat((lower, upper.flip(0), lower[0:1])).numpy()
            fig.add_trace(go.Scatterpolar(
                r=r_vals, theta=style['theta_vals'], thetaunit='radians', fill='toself',
                legendgroup='polar', showlegend=False, line=dict(width=0.5, color=style['colors'][i]),
                fillcolor=style['colors'][i], meta=meta_data
            ), row=row, col=col)
            trace_indices.append(len(fig.data) - 1)
        for name, r_key, theta_key, color in [
            ('No Response', 'no_responses_r', 'no_responses_theta', 'rgba(217, 83, 25, 0.8)'),
            ('Response', 'responses_r', 'responses_theta', 'rgba(0, 114, 189, 0.8)')
        ]:
            r_vals = rf_data[r_key] or [None]
            theta_vals = rf_data[theta_key] or [None]
            fig.add_trace(go.Scatterpolar(
                r=r_vals, theta=theta_vals, thetaunit='radians',
                name=name, showlegend=True, mode='markers',
                marker=dict(color=color, size=20, line=dict(color='black', width=0.5)),
                meta=meta_data
            ), row=row, col=col)
            trace_indices.append(len(fig.data) - 1)
        return trace_indices

    def _add_rf_plot(self, fig: Figure, param_dict: Dict, config: Dict, layout_config: Dict, plot_data: Dict[str, Any], col_idx: int) -> List[int]:
        """Adds a single response-function polar plot at (row=1, col=col_idx).

        The plot shows the estimated RF (colored confidence bands + trial scatter)
        at the likeliest source vertex by default. If a simulated source exists,
        the true RF contour is overlaid in black using `source['rf_params']`.

        Args:
            fig: The main `go.Figure` object.
            param_dict: The dictionary of response function parameters.
            config: The configuration dictionary for this plot.
            layout_config: The main layout configuration dictionary.
            plot_data: The main data dictionary from the session.
            col_idx: The column index for this subplot.

        Returns:
            A list of indices for all traces added (estimated traces first,
            then optional true-contour traces).
        """
        TRUE_REGION_COLOR = 'rgba(0, 0, 0, 0.6)'

        e_mags = plot_data['e_mags_cpu']
        angles = plot_data['angles_cpu']
        responses = plot_data['active_responses'].cpu()
        source = plot_data.get('source')

        likeliest_source_index = torch.argmax(plot_data['source_prob']).item()

        style = self._create_polar_style()
        rf_data = _utils.get_rf_plot_data(param_dict, e_mags, angles, responses, likeliest_source_index)

        row, col = 1, col_idx
        trace_indices = self._add_estimated_rf_traces(fig, rf_data, style, row, col, meta_id='rf_plot')

        # Record polar axis for later range updates
        self.rf_axis_name = fig.data[trace_indices[0]].subplot

        max_r = 100
        if e_mags.any():
            max_r = e_mags[:, likeliest_source_index].max().item() * 1.1
        fig.layout[self.rf_axis_name].radialaxis.range = [0, max_r]

        # True RF contour overlay (uses index 0 of simulated source's rf_params; multi-source not supported)
        if source:
            true_rf_data = _utils.get_rf_plot_data(source['rf_params'], e_mags, angles, responses, 0)
            for lower, upper in true_rf_data['bounds']:
                r_vals = torch.cat((lower, upper.flip(0), lower[0:1])).numpy()
                fig.add_trace(go.Scatterpolar(
                    r=r_vals, theta=style['theta_vals'], thetaunit='radians', legendgroup='polar',
                    showlegend=False, line=dict(width=2, color=TRUE_REGION_COLOR),
                    meta={'id': 'rf_true'}
                ), row=row, col=col)
                trace_indices.append(len(fig.data) - 1)

        # Invisible trace carrying the response-probability colorbar
        fig.add_trace(go.Scatterpolar(
            r=style['ci'].numpy(), theta=torch.zeros_like(style['ci']).numpy(), thetaunit='radians',
            showlegend=False, mode='markers',
            marker=dict(
                opacity=0.0,
                colorbar=dict(
                    orientation='h',
                    x=0.5, xanchor='center',
                    y=1.05, yanchor='bottom',
                    len=0.3,
                    thickness=25,
                    tickvals=[0.0, 0.5, 1.0],
                    ticktext=['0%', '50%', '100%'],
                    outlinecolor='#000000', outlinewidth=2,
                    title=dict(text='Response probability', side='top'),
                ),
                colorscale=style['disc_c_scale'], cmin=0, cmax=1
            ),
            meta={'id': 'rf_colorbar'}
        ), row=row, col=col)
        self.rf_colorbar_trace_index = len(fig.data) - 1

        return trace_indices

    def _apply_figure_styling(self, fig: Figure, layout_config: Dict, plot_data: Dict[str, Any]):
        """Applies final layout updates, styling, and annotations to the figure.

        Args:
            fig: The main `go.Figure` object.
            layout_config: The main layout configuration dictionary.
            plot_data: The main data dictionary from the session.
        """
        n_rows, n_cols = layout_config['n_rows'], layout_config['n_cols']
        font_size = 26
        bg_color = 'rgba(255, 255, 255, 1)'
        total_width = n_cols * 800
        total_height = n_rows * 1000

        fig.update_layout(
            height=total_height, width=total_width,
            paper_bgcolor=bg_color, plot_bgcolor=bg_color,
            legend=dict(yanchor='top', y=0.985, xanchor='center', x=0.6, itemsizing='constant', orientation='h', bgcolor='rgba(0,0,0,0)'),
            font=dict(size=font_size),
            margin=dict(t=total_height * 0.25),
            yaxis=dict(tickformat='.1e',color="#636EFA",nticks=5),
            yaxis2=dict(tickformat='.1e',color="#EF553B",overlaying="y",tickmode="sync"),
            uirevision='preserve'  # Preserves UI state (like camera) on update
        )

        fig.update_polars(
            bgcolor=bg_color,
            angularaxis=dict(color='gray', gridcolor='gray', showline=True, linewidth=2, linecolor='black', tickfont_color='black', rotation=180),
            radialaxis=dict(showline=False, gridcolor='gray', layer='above traces', nticks=4, color='black', angle=0, tickfont_color='rgb(0,0,0)')
        )

        camera = plot_data['mesh_camera']

        fig.update_scenes(
            camera=camera,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=bg_color,
            aspectmode='data'
        )

        # Apply flat matte lighting with a fixed light position
        fig.update_traces(
            selector=dict(type='mesh3d'),
            colorbar=dict(orientation='h', tickangle=0),
            lightposition_x=100, lightposition_y=-100, lightposition_z=0,
            lighting_ambient=0.7, lighting_diffuse=0.3, lighting_fresnel=0.1, lighting_roughness=0.8, lighting_specular=0.4
        )

        # Set colorbar locations and widths to fit
        mesh_traces_indices = [i for i, trace in enumerate(fig.data) if trace.type == 'mesh3d']
        scene_keys = ["scene","scene2"]
        for i, trace_idx in enumerate(mesh_traces_indices):
            scene_key = scene_keys[i]
            scene_domain = fig.layout[scene_key].domain.x

            # Calculate width and center from the actual domain
            subplot_width = scene_domain[1] - scene_domain[0]
            x_center = scene_domain[0] + subplot_width / 2.0

            # Update the colorbar of the specific trace
            fig.data[trace_idx].colorbar.x = x_center
            fig.data[trace_idx].colorbar.len = subplot_width * 0.66

        # Shrink the polar subplot to 0.75x (equal padding on all sides)
        polar_domain_x = list(fig.layout[self.rf_axis_name].domain.x)
        polar_domain_y = list(fig.layout[self.rf_axis_name].domain.y)
        POLAR_SCALE = 0.75
        x_pad = (polar_domain_x[1] - polar_domain_x[0]) * (1 - POLAR_SCALE) / 2
        y_pad = (polar_domain_y[1] - polar_domain_y[0]) * (1 - POLAR_SCALE) / 2
        fig.layout[self.rf_axis_name].domain.x = [polar_domain_x[0] + x_pad, polar_domain_x[1] - x_pad]
        fig.layout[self.rf_axis_name].domain.y = [polar_domain_y[0] + y_pad, polar_domain_y[1] - y_pad]

        # Position the RF colorbar: centered on the polar column, level with mesh colorbars (y=1.0)
        polar_center_x = (polar_domain_x[0] + polar_domain_x[1]) / 2.0
        polar_col_width = polar_domain_x[1] - polar_domain_x[0]

        if 0 <= self.rf_colorbar_trace_index < len(fig.data):
            cb = fig.data[self.rf_colorbar_trace_index].marker.colorbar
            cb.x = polar_center_x
            cb.xanchor = 'center'
            cb.y = 1.0
            cb.yanchor = 'bottom'
            cb.len = polar_col_width * 0.66
            cb.orientation = 'h'

        # Position legend2 (source marker) at the bottom-right of the first mesh subplot
        scene1_domain_x = fig.layout['scene'].domain.x
        scene1_domain_y = fig.layout['scene'].domain.y
        fig.update_layout(
            legend2=dict(
                x=scene1_domain_x[1]*0.85,
                y=0.975,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(0,0,0,0)',
                itemsizing='constant',
            )
        )

        # Adjust subplot title positions for better spacing.
        # Annotation y was placed at the original (pre-shrink) domain top, so search using that value.
        polar_axes_keys = [key for key in fig.layout if key.startswith('polar')]
        TOLERANCE = 1e-4
        for axis_key in polar_axes_keys:
            subplot = fig.layout[axis_key]
            subplot_height = subplot.domain.y[1] - subplot.domain.y[0]
            subplot_x_center = subplot.domain.x[0] + (subplot.domain.x[1] - subplot.domain.x[0]) / 2

            for annot in fig.layout.annotations:
                if abs(annot.x - subplot_x_center) < TOLERANCE and abs(annot.y - polar_domain_y[1]) < TOLERANCE:
                    new_y_position = subplot.domain.y[1] + (subplot_height * 0.075)
                    annot.y = new_y_position
                    break

        # Add custom annotations
        trial_text = f"Responses: {int(plot_data['active_responses'].sum().item())}/{plot_data['trial_number']}"
        fig.add_annotation(name='pulse_counter', text=trial_text, xref='paper', yref='paper', x=0.0, y=1.0 + 0.33 / n_rows, showarrow=False, font=dict(size=30))

        fig.update_annotations(font=dict(family='Helvetica', size=font_size+6))

    def _reload_figure_if_modified(self, filepath: str):
        """Checks if the figure file was modified externally and reloads it.

        This is crucial for preserving user interactions like camera zoom/pan.
        The Dash app saves the figure state to file on interaction, and this
        function detects that change and loads it back into the Python object.

        Args:
            filepath: The path to the 'figure.plotly' JSON file.
        """
        if not os.path.isfile(filepath): return

        current_mod_time = os.stat(filepath).st_mtime
        if current_mod_time != self.last_fig_mod_time:
            try:
                self.figure = pio.read_json(filepath)
                self.last_fig_mod_time = current_mod_time
                print("ABL: Reloaded figure from file due to external modification.")
            except Exception as e:
                print(f"ABL: Error reloading figure: {e}")

    def _get_mesh_update_payload(self, trace_indices: List[int], data: Tensor, config: Dict, plot_data: Dict[str, Any]) -> List[Dict]:
        """Generates the update payload for a 3D mesh plot.

        This includes updating the intensity data, color bar, and the position
        of the compass rose.

        Args:
            trace_indices: List of trace indices associated with this plot.
            data: The new intensity data for the mesh.
            config: The plot configuration dictionary.
            plot_data: The main data dictionary from the session.

        Returns:
            A list of dictionary objects for the JSON update payload.
        """
        is_source_prob = (config['title'] == 'Source probability')
        if is_source_prob:
            intensity = _utils.compute_cumulative_prob(data.numpy())
            payload = [
                {"trace_index": trace_indices[0], "key": "intensity", "value": intensity.tolist()},
            ]
        else:
            cmax_val = data.max().item()
            tickvals = [0, cmax_val / 2, cmax_val]
            ticktext = _utils.format_ticks(tickvals)
            payload = [
                {"trace_index": trace_indices[0], "key": "intensity", "value": data.numpy().tolist()},
                {"trace_index": trace_indices[0], "key": "cmax", "value": cmax_val},
                {"trace_index": trace_indices[0], "key": "colorbar.tickvals", "value": tickvals},
                {"trace_index": trace_indices[0], "key": "colorbar.ticktext", "value": ticktext},
            ]

        # Check if a compass rose trace exists and update its position
        if len(trace_indices) > 1 and self.figure and len(self.figure.data) > trace_indices[-1] and 'compass_line' in self.figure.data[trace_indices[-1]].name:
            compass_vertex = self._resolve_view_vertex(plot_data)
            compass_trace_indices = trace_indices[-1:] # The last trace is the compass line
            compass_payload = self._get_compass_rose_update_payload(compass_vertex, plot_data, compass_trace_indices)
            payload.extend(compass_payload)

        return payload

    def _get_compass_rose_update_payload(self, vertex_index: int, plot_data: Dict[str, Any], trace_indices: List[int]) -> List[Dict]:
        """Generates the update payload for the compass rose line trace.

        Args:
            vertex_index: The new ROI index for the compass rose position.
            plot_data: The main data dictionary from the session.
            trace_indices: The list of trace indices for the compass parts.

        Returns:
            A list of dictionary objects for the JSON update payload.
        """
        if not trace_indices:
            return []

        coords = _utils.get_compass_rose_coords(
            plot_data['mesh_vertices'],
            plot_data['mesh_ROI'],
            plot_data['efield_basis'],
            plot_data['null_dir_basis'],
            vertex_index
        )
        if not coords:
            return []

        # Update the x, y, z coordinates of the combined line trace
        payload = [
            {"trace_index": trace_indices[0], "key": "x", "value": coords['line_x'].tolist()},
            {"trace_index": trace_indices[0], "key": "y", "value": coords['line_y'].tolist()},
            {"trace_index": trace_indices[0], "key": "z", "value": coords['line_z'].tolist()},
        ]
        return payload

    def _get_arrow_mesh_update_payload(self, trace_indices: List[int], data: Tensor, config: Dict, plot_data: Dict[str, Any]) -> List[Dict]:
        """Generates the update payload for an arrow mesh plot.

        This updates both the mesh intensity (vector magnitude) and the cone
        directions.

        Args:
            trace_indices: List of trace indices for the mesh and cones.
            data: The new vector data tensor.
            config: The plot configuration dictionary.
            plot_data: The main data dictionary from the session.

        Returns:
            A list of dictionary objects for the JSON update payload.
        """
        e_mag = torch.norm(data, dim=-1)
        data_normalized = data / e_mag.unsqueeze(-1)
        data_normalized[~torch.isfinite(data_normalized)] = 0

        # First, update the underlying mesh intensity
        payload = self._get_mesh_update_payload([trace_indices[0]], e_mag, config, plot_data)

        # Then, update the u, v, w direction vectors for the cones
        cone_trace_index = trace_indices[-1]
        coarse_inds = plot_data['mesh_coarse_inds']
        payload.extend([
            {"trace_index": cone_trace_index, "key": "u", "value": data_normalized[coarse_inds, 0].numpy().tolist()},
            {"trace_index": cone_trace_index, "key": "v", "value": data_normalized[coarse_inds, 1].numpy().tolist()},
            {"trace_index": cone_trace_index, "key": "w", "value": data_normalized[coarse_inds, 2].numpy().tolist()},
        ])
        return payload

    def _get_loss_update_payload(self, trace_indices: List[int], data: Tensor, config: Dict, plot_data: Dict[str, Any]) -> List[Dict]:
        """Generates the update payload for a loss plot.

        Args:
            trace_indices: List of trace indices for the loss curves.
            data: The new loss data tensor.
            config: The plot configuration dictionary.
            plot_data: The main data dictionary from the session.

        Returns:
            A list of dictionary objects for the JSON update payload.
        """
        return [
            {"trace_index": trace_indices[0], "key": "x", "value": np.arange(data.shape[-1]).tolist()},
            {"trace_index": trace_indices[0], "key": "y", "value": data[0, :].numpy().tolist()},
            {"trace_index": trace_indices[1], "key": "x", "value": np.arange(data.shape[-1]).tolist()},
            {"trace_index": trace_indices[1], "key": "y", "value": data[1, :].numpy().tolist()},
        ]

    def _resolve_view_vertex(self, plot_data: Dict[str, Any]) -> int:
        """Resolves which ROI vertex the RF plot should display.

        Reads `view_state.json` from `self.save_dir`. In `auto` mode (default),
        returns the likeliest source vertex. In `pinned` mode, returns the
        stored vertex, falling back to likeliest if the index is invalid.
        """
        likeliest = int(torch.argmax(plot_data['source_prob']).item())
        n_roi = plot_data['source_prob'].shape[0]
        state_path = os.path.join(self.save_dir, 'view_state.json')
        if not os.path.isfile(state_path):
            return likeliest
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            return likeliest
        if state.get('mode') == 'pinned':
            vertex = state.get('vertex')
            if isinstance(vertex, int) and 0 <= vertex < n_roi:
                return vertex
        return likeliest

    def _get_rf_update_payload(self, trace_indices: List[int], param_dict: Dict, config: Dict, plot_data: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Generates the update payload for the single RF polar plot.

        Updates the 5 confidence bands and 2 scatter traces (no-response /
        response) at the resolved view vertex. The true-contour overlay (if
        present) is static and not updated here.

        Args:
            trace_indices: Trace indices belonging to the RF plot. The first 5
                are confidence bands, the next 2 are scatter traces, and any
                further indices are true-contour traces (not updated).
            param_dict: The RF model parameters.
            config: The plot configuration dictionary (unused).
            plot_data: The main data dictionary from the session.

        Returns:
            A dictionary containing lists of data and layout updates.
        """
        e_mags = plot_data['e_mags_cpu']
        angles = plot_data['angles_cpu']
        responses = plot_data['active_responses'].cpu()

        view_vertex = self._resolve_view_vertex(plot_data)
        rf_data = _utils.get_rf_plot_data(param_dict, e_mags, angles, responses, view_vertex)

        n_bands = len(rf_data['bounds'])
        data_updates = []
        for i, (lower, upper) in enumerate(rf_data['bounds']):
            r_vals = torch.cat((lower, upper.flip(0), lower[0:1])).numpy().tolist()
            data_updates.append({"trace_index": trace_indices[i], "key": "r", "value": r_vals})

        no_resp_idx = trace_indices[n_bands]
        resp_idx = trace_indices[n_bands + 1]
        data_updates.append({"trace_index": no_resp_idx, "key": "r", "value": rf_data['no_responses_r']})
        data_updates.append({"trace_index": no_resp_idx, "key": "theta", "value": rf_data['no_responses_theta']})
        data_updates.append({"trace_index": resp_idx, "key": "r", "value": rf_data['responses_r']})
        data_updates.append({"trace_index": resp_idx, "key": "theta", "value": rf_data['responses_theta']})

        max_r = 100
        if e_mags.any():
            max_r = e_mags[:, view_vertex].max().item() * 1.1
        layout_updates = [{"key": f"{self.rf_axis_name}.radialaxis.range", "value": [0, max_r]}]

        return {"data_updates": data_updates, "layout_updates": layout_updates}

    def _update_annotations_payload(self, payload: Dict, plot_data: Dict[str, Any]):
        """Adds annotation updates to the main payload dictionary.

        This function finds specific annotations by name (e.g., 'pulse_counter')
        in the figure object and generates the appropriate update payload entry.

        Args:
            payload: The main update payload dictionary to add to.
            plot_data: The dictionary containing data for the update.
        """
        trial_number = plot_data['trial_number']
        active_responses = plot_data['active_responses']
        trial_text = f"Responses: {int(active_responses.sum().item())}/{trial_number}"

        if not self.figure: return

        # Find annotations by name and generate layout updates
        for i, annot in enumerate(self.figure.layout.annotations):
            if annot.name == 'pulse_counter':
                payload["layout_updates"].append({"key": f"annotations[{i}].text", "value": trial_text})
            elif trial_number > 0 and annot.name == 'prior_guide':
                # Hide the "Click to add prior" text after the first trial
                payload["layout_updates"].append({"key": f"annotations[{i}].text", "value": ""})


    def create_interactive_trial_viewer(self, plot_data: Dict[str, Any], efield_trials: Tensor) -> Figure:
        """Creates an interactive figure with a slider to scroll through E-field trials.

        This method generates a single mesh instance and uses Plotly Frames to update
        both the mesh intensity (E-field magnitude) and the cones (E-field direction).
        It uses the full E-field vectors provided in `efield_trials` rather than
        reconstructing them.

        Args:
            plot_data: The main data dictionary from the session.
            efield_trials: A Tensor of shape (n_trials, n_roi_vertices, 3) containing
                           the E-field vectors for each trial.

        Returns:
            A plotly.graph_objects.Figure with a slider and play button.
        """
        if efield_trials is None or len(efield_trials) == 0:
            print("No trial data available to plot.")
            return go.Figure()

        # Ensure data is on CPU
        efield_trials = efield_trials.cpu()
        responses = plot_data.get('active_responses').cpu()
        vertices = plot_data['mesh_vertices']
        faces = plot_data['mesh_faces']
        roi_indices = plot_data['mesh_ROI']
        coarse_inds = plot_data['mesh_coarse_inds']

        n_trials = efield_trials.shape[0]

        # Calculate Magnitudes and Normalized Direction Vectors
        # shape: (n_trials, n_roi_vertices)
        e_mags = torch.norm(efield_trials, dim=-1)
        cmax_val = e_mags.max().item()

        # shape: (n_trials, n_roi_vertices, 3)
        # Handle division by zero for vectors with 0 magnitude
        e_vecs = torch.zeros_like(efield_trials)
        mask = e_mags > 1e-9
        e_vecs[mask] = efield_trials[mask] / e_mags[mask].unsqueeze(-1)

        # Pre-calculate Cone Positions (Static across trials)
        # We assume coarse_inds are indices into the ROI array
        cone_pos = vertices[roi_indices][coarse_inds]

        # --- Helper to create a Cone trace for a specific trial ---
        def create_cone_trace(trial_idx):
            # Select vectors for the subset of vertices (coarse_inds)
            vecs = e_vecs[trial_idx, coarse_inds]

            return go.Cone(
                x=cone_pos[:, 0], y=cone_pos[:, 1], z=cone_pos[:, 2],
                u=vecs[:, 0], v=vecs[:, 1], w=vecs[:, 2],
                sizemode='absolute', sizeref=0.5, showscale=False,
                colorscale=[[0, 'black'], [1, 'black']],
                lighting=dict(ambient=0, diffuse=1, fresnel=0, specular=0, roughness=1),
                name='Direction'
            )

        # 1. Prepare Initial Data (Trial 0)
        # Map ROI magnitude data to the full mesh
        initial_intensity = torch.zeros(vertices.shape[0])
        initial_intensity[roi_indices] = e_mags[0]

        response_text = "True" if responses[0] > 0.5 else "False"
        initial_title = f"Trial 1/{n_trials} - Response: {response_text}"

        data_traces = [go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            intensity=initial_intensity,
            colorscale='Viridis',
            intensitymode='vertex',
            cmin=0, cmax=cmax_val,
            showscale=True,
            colorbar=dict(title='E-field (V/m)'),
            name='Magnitude'
        )]

        initial_cone = create_cone_trace(0)
        if initial_cone:
            data_traces.append(initial_cone)

        fig = go.Figure(data=data_traces)

        # 2. Create Frames for every trial
        frames = []
        for i in range(n_trials):
            # Update Mesh Intensity
            frame_intensity = torch.zeros(vertices.shape[0])
            frame_intensity[roi_indices] = e_mags[i]

            frame_data = [go.Mesh3d(intensity=frame_intensity)]

            # Update Cone Directions
            cone_trace = create_cone_trace(i)
            if cone_trace:
                frame_data.append(cone_trace)

            r_text = "True" if responses[i] > 0.5 else "False"

            frames.append(go.Frame(
                data=frame_data,
                name=f"frame{i}",
                layout=go.Layout(title_text=f"Trial {i + 1}/{n_trials} - Response: {r_text}")
            ))

        fig.frames = frames

        # 3. Add Slider and Play Button Control
        fig.update_layout(
            title=initial_title,
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Trial: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"frame{k}"],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
                        ],
                        "label": str(k + 1),
                        "method": "animate"
                    } for k in range(n_trials)
                ]
            }],
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )

        # Apply camera if available
        if 'mesh_camera' in plot_data and plot_data['mesh_camera']:
            fig.update_scenes(camera=plot_data['mesh_camera'])

        # Clean up axis
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)

        return fig