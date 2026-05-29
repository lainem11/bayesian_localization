import dash
from dash import dcc, html, Input, Output, State, Patch, no_update
import plotly.io as pio
import os
import torch
import sys
import h5py
import json
import re
import numpy as np
from typing import Dict
from . import _utils

# We expect one argument: the path to the save directory.
if len(sys.argv) != 2:
    print("Error: This script requires exactly one argument: <save_directory_path>", file=sys.stderr)
    sys.exit(1)

savedir = sys.argv[1]

if not os.path.isdir(savedir):
    print(f"Error: The provided directory '{savedir}' does not exist.", file=sys.stderr)
    sys.exit(1)
        
# Define file paths
fig_filepath = os.path.join(savedir, 'figure.plotly')
update_filepath = os.path.join(savedir, 'figure_update.json')
experiment_filepath = os.path.join(savedir, 'experiment_data.hdf5')
view_state_filepath = os.path.join(savedir, 'view_state.json')

# Constants
COL_STEPS = 5
PROB_PLOT_INDEX = 0

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    # Graph container is position:relative so the button can be overlaid with position:absolute.
    # The polar subplot is the 2nd of 3 columns in a ~2400px-wide figure; left ~1150px centers on it.
    html.Div(
        style={'position': 'relative', 'display': 'inline-block'},
        children=[
            dcc.Graph(
                id='live-update-graph',
                figure={},
                config={'staticPlot': False, 'displayModeBar': True}
            ),
            dcc.Store(id='last-update-time', data=0),
            html.Button(
                'Reset to likeliest source',
                id='reset-view-btn',
                n_clicks=0,
                style={
                    'position': 'absolute',
                    'bottom': '12px',
                    'left': '1150px',
                    'transform': 'translateX(-50%)',
                    'zIndex': '1000',
                    'padding': '6px 14px',
                    'fontSize': '18px',
                    'cursor': 'pointer',
                }
            ),
        ],
    ),
    # A slower interval for the initial, heavy figure load.
    dcc.Interval(
        id='initial-interval',
        interval=3000,
        n_intervals=0,
        disabled=False
    ),
    # A faster interval for the lightweight patch updates.
    dcc.Interval(
        id='update-interval',
        interval=1000,
        n_intervals=0,
        disabled=True
    )
])

def set_nested_value(patch_obj, path_str, value):
    """
    Sets a value in a nested Patch object using a path string.
    Handles both dictionary keys (e.g., 'radialaxis') and list indices (e.g., '[0]').
    """
    path_parts = re.split(r'\.|(\[\d+\])', path_str)
    path_parts = [part for part in path_parts if part]

    current_level = patch_obj
    for part in path_parts[:-1]:
        if part.startswith('[') and part.endswith(']'):
            index = int(part[1:-1])
            current_level = current_level[index]
        else:
            current_level = current_level[part]
    
    last_part = path_parts[-1]
    if last_part.startswith('[') and last_part.endswith(']'):
        index = int(last_part[1:-1])
        current_level[index] = value
    else:
        current_level[last_part] = value

@app.callback(
    Output('live-update-graph', 'figure'),
    Output('last-update-time', 'data'),
    Output('initial-interval', 'disabled'),
    Output('update-interval', 'disabled'),
    [Input('initial-interval', 'n_intervals'),
     Input('update-interval', 'n_intervals')],
    State('live-update-graph', 'figure'),
    State('last-update-time', 'data'),
    prevent_initial_call=True
)
def update_graph_live(n_initial, n_update, current_figure, last_update_time):
    """
    This callback handles both the initial load and subsequent patch updates.
    """
    if not current_figure or not current_figure.get('data'):
        if os.path.exists(fig_filepath):
            try:
                print("APP: Performing initial figure load.")
                new_fig = pio.read_json(fig_filepath)
                new_fig['layout']['uirevision'] = 'preserve'
                new_time = os.stat(fig_filepath).st_mtime
                return new_fig, new_time, True, False
            except Exception as e:
                print(f"APP: Error during initial load: {e}")
                return no_update, last_update_time, False, True
        else:
            return no_update, last_update_time, False, True

    if os.path.exists(update_filepath):
        try:
            new_mod_time = os.stat(update_filepath).st_mtime
            if new_mod_time > last_update_time:
                print(f"APP: Applying patch from {new_mod_time}.")
                with open(update_filepath, 'r') as f:
                    updates = json.load(f)

                patched_figure = Patch()
                for update in updates.get('data_updates', []):
                    set_nested_value(patched_figure['data'][update['trace_index']], update['key'], update['value'])
                for update in updates.get('layout_updates', []):
                    set_nested_value(patched_figure['layout'], update['key'], update['value'])

                return patched_figure, new_mod_time, no_update, no_update
        except Exception as e:
            print(f"APP: Error applying patch: {e}")
            return no_update, last_update_time, no_update, no_update

    return no_update, last_update_time, no_update, no_update

def _find_rf_plot_index(fig: Dict) -> int:
    """Return the trace index of the first estimated-RF band trace (meta.id == 'rf_plot')."""
    for i, trace in enumerate(fig['data']):
        if trace.get('meta', {}).get('id') == 'rf_plot':
            return i
    return -1


def _build_rf_patch(fig: Dict, data_dict: Dict, roi_vertex: int) -> Dict:
    """Build the figure_update.json payload that re-points the RF plot and compass to roi_vertex."""
    updates = {'data_updates': [], 'layout_updates': []}

    # Compass rose
    compass_keys = ['cortex', 'ROI', 'efield_subspace']
    if all(key in data_dict for key in compass_keys):
        np_coords = _utils.get_compass_rose_coords(
            data_dict['cortex']['vertices'],
            data_dict['ROI'],
            data_dict['efield_subspace'].get('efield_basis'),
            data_dict['efield_subspace'].get('null_dir_basis'),
            roi_vertex
        )
        coords = {k: v.tolist() for k, v in np_coords.items()} if np_coords else {}
        if coords:
            for i, trace in enumerate(fig['data']):
                if trace.get('name', '') == 'compass_line':
                    updates['data_updates'].append({'trace_index': i, 'key': 'x', 'value': coords['line_x']})
                    updates['data_updates'].append({'trace_index': i, 'key': 'y', 'value': coords['line_y']})
                    updates['data_updates'].append({'trace_index': i, 'key': 'z', 'value': coords['line_z']})
                    break

    # RF traces
    plot_index = _find_rf_plot_index(fig)
    if plot_index != -1:
        rf_plot_data = _utils.get_rf_plot_data(
            data_dict['param_dict'], data_dict['e_mags'], data_dict['angles'], data_dict['responses'], roi_vertex
        )
        for j, (lower, upper) in enumerate(rf_plot_data['bounds']):
            r_vals = torch.cat((lower, upper.flip(0), lower[0:1])).numpy()
            updates['data_updates'].append({'trace_index': plot_index + j, 'key': 'r', 'value': r_vals.tolist()})

        no_responses_idx = plot_index + len(rf_plot_data['bounds'])
        responses_idx = no_responses_idx + 1
        updates['data_updates'].append({'trace_index': no_responses_idx, 'key': 'r', 'value': rf_plot_data['no_responses_r']})
        updates['data_updates'].append({'trace_index': no_responses_idx, 'key': 'theta', 'value': rf_plot_data['no_responses_theta']})
        updates['data_updates'].append({'trace_index': responses_idx, 'key': 'r', 'value': rf_plot_data['responses_r']})
        updates['data_updates'].append({'trace_index': responses_idx, 'key': 'theta', 'value': rf_plot_data['responses_theta']})

        axis_name = fig['data'][plot_index]['subplot']
        max_r = data_dict['e_mags'][:, roi_vertex].max().item() * 1.1
        updates['layout_updates'].append({'key': f"{axis_name}.radialaxis.range", 'value': [0, max_r]})

    return updates


def _likeliest_roi_vertex_from_fig(fig: Dict) -> int:
    """Recover the likeliest-source ROI index from the source-prob mesh intensity in the figure."""
    for i, trace in enumerate(fig['data']):
        if trace.get('name', '') == 'source_prob_mesh':
            intensity = np.asarray(trace.get('intensity', []))
            if intensity.size == 0:
                return -1
            mesh_idx = int(np.argmax(intensity))
            try:
                data_dict = _utils.load_from_hdf5(experiment_filepath)
            except Exception:
                return -1
            if 'ROI' not in data_dict:
                return -1
            roi_idx = _utils.mesh_ind_to_ROI_ind(mesh_idx, data_dict['ROI'])
            return -1 if roi_idx is None else int(roi_idx)
    return -1


def _write_view_state(mode: str, vertex: int = None) -> None:
    """Write the current view-mode state for the server to read on its next update."""
    state = {'mode': mode}
    if vertex is not None:
        state['vertex'] = int(vertex)
    try:
        with open(view_state_filepath, 'w') as f:
            json.dump(state, f)
    except Exception as e:
        print(f"APP: Error writing view state: {e}")


def _write_update_payload(updates: Dict) -> None:
    if not (updates['data_updates'] or updates['layout_updates']):
        return
    try:
        with open(update_filepath, 'w') as f:
            json.dump(updates, f)
    except Exception as e:
        print(f"APP: Error writing update file: {e}")


@app.callback(
    Output('live-update-graph', 'figure', allow_duplicate=True),
    Input('live-update-graph', 'clickData'),
    State('live-update-graph', 'figure'),
    prevent_initial_call=True
)
def update_sub_plot_on_click(clickData, fig):
    """
    Handles user clicks by generating an update file. This callback does not
    modify the figure directly. Instead, it writes the specific changes to
    `figure_update.json`, which is then picked up by the `update-interval`
    callback. When responses already exist, the click also writes
    `view_state.json` so the server pins subsequent RF updates to that vertex.
    """
    if clickData is None or not fig:
        return no_update

    points = clickData['points'][0]
    curve_number = points['curveNumber']
    trace_name = fig['data'][curve_number].get('name', '')

    if trace_name != 'source_prob_mesh':
        return no_update

    point_number = points['pointNumber']
    data_dict = _utils.load_from_hdf5(experiment_filepath)

    if 'ROI' not in data_dict:
        raise ValueError("Missing required key in file: ROI")
    ROI = data_dict['ROI']
    ROI_point_number = _utils.mesh_ind_to_ROI_ind(point_number, ROI)
    print(f"APP: Click detected on mesh vertex: {point_number}, ROI vertex: {ROI_point_number}")
    if ROI_point_number is None:
        return no_update

    is_recorded = 'responses' in data_dict and data_dict['responses'].numel()

    if not is_recorded:
        print("APP: Defining new prior probability distribution based on click location.")
        if 'cortex' not in data_dict:
            raise ValueError("Missing required keys in file: cortex")

        prior_probs = _utils.get_prior(data_dict['cortex']['vertices'][ROI, :], ROI_point_number)

        full_cortex_probs = torch.zeros(data_dict['cortex']['vertices'].shape[0])
        full_cortex_probs[ROI] = prior_probs
        cum_prob = _utils.compute_cumulative_prob(full_cortex_probs.numpy())

        # Source-prob mesh has static cmax=1.0 / fixed percent ticks; only intensity changes.
        updates = {
            'data_updates': [{'trace_index': PROB_PLOT_INDEX, 'key': 'intensity', 'value': cum_prob.tolist()}],
            'layout_updates': [],
        }

        with h5py.File(experiment_filepath, 'a') as f:
            if 'informed_prior' in f:
                del f['informed_prior']
            f.create_dataset('informed_prior', data=prior_probs.numpy())

        _write_update_payload(updates)
    else:
        print("APP: Probing the response function at clicked point.")
        required_keys = ['param_dict', 'e_mags', 'angles', 'responses']
        if any(key not in data_dict for key in required_keys):
            raise ValueError("Missing required keys for RF plot update.")

        updates = _build_rf_patch(fig, data_dict, ROI_point_number)
        _write_update_payload(updates)
        _write_view_state('pinned', ROI_point_number)

    # ALWAYS return no_update, as the other callback will handle the patching.
    return no_update


@app.callback(
    Output('live-update-graph', 'figure', allow_duplicate=True),
    Input('reset-view-btn', 'n_clicks'),
    State('live-update-graph', 'figure'),
    prevent_initial_call=True
)
def reset_view(n_clicks, fig):
    """Reset to auto-follow likeliest source and snap the polar plot there immediately."""
    if not n_clicks or not fig:
        return no_update

    _write_view_state('auto')

    try:
        data_dict = _utils.load_from_hdf5(experiment_filepath)
    except Exception as e:
        print(f"APP: Reset: failed to load experiment data: {e}")
        return no_update

    if 'responses' not in data_dict or data_dict['responses'].numel() == 0:
        # No trials yet; nothing to patch on the polar plot.
        return no_update

    likeliest_roi = _likeliest_roi_vertex_from_fig(fig)
    if likeliest_roi < 0:
        return no_update

    required_keys = ['param_dict', 'e_mags', 'angles', 'responses']
    if any(key not in data_dict for key in required_keys):
        return no_update

    updates = _build_rf_patch(fig, data_dict, likeliest_roi)
    _write_update_payload(updates)
    return no_update

if __name__ == '__main__':
    app.run(debug=False)
