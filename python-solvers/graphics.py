
"""
Script with graphic functions.
- history_2d:
    Plots 2D metric training history over iteration steps using Plotly.

- history_3d:
    Plots a 3D spatial coordinate trajectory (X, Y, Z) and ground station
    locations.

- split_curve_1d:
    Iteratively splits a 1D curve into piecewise linear segments by finding
    the point of maximum squared error in each iteration.

- split_curve_3d:
    Iteratively splits a 3D curve into piecewise linear segments by finding
    the point of maximum squared error in each iteration.

TO DO:
    Signal-to-Noise / Variance-Based Ratio treshold

"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def split_curve_1d(data:np.ndarray, max_splits:int=5,
                   threshold:float|None=None):
    """
    Iteratively splits a 1D curve into piecewise linear segments by finding
    the point of maximum squared error in each iteration.
    Helper function to reduce the number of dots to save memory and
    computations in 1d history plots.

    Parameters
    ----------
    cost_story : numpy.ndarray
        Input data as a vector (N,).
    max_splits : int, default=3
        Maximum number of splits (break points) to insert.
    threshold : float, optional
        If specified, early stopping criterion based on signal to noise ratio
        between the data (avg(data)/std(data)) and the approximation (
        avg(approximation)/std(approximation)) ratio computed after at least
        five iteration:
            |snr_appx / snr_data -1| <= threshold & step >= 5

    Returns
    -------
    split_indices : numpy.ndarray
        Indices of the split points (sorted, including start and end).

    """
    iters = np.arange(len(data))
    snr_data = np.mean(data)/np.std(data)

    # Keep track of split indices in sorted order (starts with endpoints
    # [0, N-1])
    split_indices = [0, len(data) - 1]

    # Initial line approximation between start and end
    approx_curve = np.interp(iters, split_indices, data[split_indices])
    for step in range(max_splits):
        # Calculate squared error cost curve
        cost = (data - approx_curve) ** 2

        # Find maximum cost point and where to split
        max_cost = np.max(cost)
        candidates = np.argwhere(cost == max_cost).flatten()
        if len(candidates) > 1:
            pos = np.random.choice(candidates)
        else:
            pos = candidates[0]

        # Insert new split point while maintaining sorted order
        split_indices.append(pos)
        split_indices.sort()

        # Update piecewise linear approximation across all current segments
        approx_curve = np.interp(iters, split_indices, data[split_indices])
        if step >= 5:
            snr_appx = np.mean(approx_curve)/np.std(approx_curve)
            var_c = np.abs(snr_appx/snr_data-1)
            if threshold is not None and var_c <= threshold:
                break

    return np.array(split_indices)

def split_curve_3d(coords_3d:np.ndarray, max_splits:int=5,
                   threshold:float|None=None):
    """
    Iteratively splits a 3D curve into piecewise linear segments by finding
    the point of maximum squared error in each iteration.
    Helper function to reduce the number of dots to save memory and
    computations in 3d history plots.

    Parameters
    ----------
    coords_3d : numpy.ndarray
        Array of shape (N, 3) containing [X, Y, Z] points.
    max_splits : int
        Maximum number of split points to insert.
    threshold : float, optional
        If specified, early stopping criterion based on signal to noise ratio
        between the data (avg(data)/std(data)) and the approximation (
        avg(approximation)/std(approximation)) ratio computed after at least
        five iteration:
            all(|snr_appx / snr_data -1| <= threshold) & step >= 5

    Returns
    -------
    split_indices : numpy.ndarray
        Indices of the chosen split points (sorted).

    """
    iters = np.arange(len(coords_3d))
    snr_data = np.mean(coords_3d, axis=0)/np.std(coords_3d, axis=0)
    split_indices = [0, len(coords_3d)-1]

    approx_curve = np.column_stack((
        np.interp(iters, split_indices, coords_3d[split_indices, 0]),
        np.interp(iters, split_indices, coords_3d[split_indices, 1]),
        np.interp(iters, split_indices, coords_3d[split_indices, 2])))

    snr_appx = np.mean(approx_curve, axis=0)/np.std(approx_curve, axis=0)
    for step in range(0, max_splits, 1):
        cost = np.sum((coords_3d - approx_curve) ** 2, axis=1)
        max_cost = np.max(cost)
        candidates = np.argwhere(cost == max_cost).flatten()
        if len(candidates) > 1:
            pos = np.random.choice(candidates)
        else:
            pos = candidates[0]

        split_indices.append(pos)
        split_indices.sort()

        approx_curve = np.column_stack((
            np.interp(iters, split_indices, coords_3d[split_indices, 0]),
            np.interp(iters, split_indices, coords_3d[split_indices, 1]),
            np.interp(iters, split_indices, coords_3d[split_indices, 2])))

        if step >= 5:
            snr_appx = np.mean(approx_curve, axis=0)/np.std(approx_curve, axis=0)
            var_c = np.abs(snr_appx / snr_data - 1)
            if np.all(var_c) <= threshold:
                break

    return split_indices

def history_2d(history:np.ndarray|list[float]|list[list[float]],
               label:str|list[str],
               title:str|None=None,
               log_y:bool|list[bool]=False,
               xlabel:str='Steps',
               ylabel:str='Metric',
               filename:str|None=None) -> go.Figure:
    """
    Plots 2D metric training history over iteration steps using Plotly.
    Generates a single line chart for 1D history arrays or stacked vertically
    shared-x subplots for 2D history arrays (where columns represent separate
    metrics).

    Parameters
    ----------
    history : numpy.ndarray or list
        Metric history data across iteration steps.
        - If 1D: shape `(N,)` representing `N` sequential steps for a single
          metric.
        - If 2D: shape `(N, M)` where `N` is steps and `M` is the number of
          metrics/subplots.
    label : str or list
        Label(s) for the plotted metric traces.
        - If string: Used as trace name (and y-axis title for 1D data).
        - If list of str: Used as individual subplot titles and y-axis labels
          for multi-metric histories.
    title : str, optional
        Overall title for the plot layout. The default is None.
    log_y : bool or list, optional
        Controls logarithmic scale for the y-axis. The default is False.
        - If bool: Applied uniformly across all metric plots.
        - If list of bool: Applied element-wise per corresponding metric
          subplot.
    xlabel : str, optional
        Title for the shared x-axis (displayed on the bottom subplot). The
        default is 'Steps'.
    ylabel : str, optional
        Title for the y-axis when `history` is 1D and `label` is passed as a
        string. The default is 'Metric'.
    filename : str or None, optional
        Output file path to save the interactive figure as an HTML file.
        If provided without a `.html` extension, `.html` will be appended
        automatically.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The constructed interactive Plotly figure object containing the
        metric trace(s).

    Examples
    --------
    Plotting a single metric (1D data):

    >>> cost_story = [0.9, 0.5, 0.2, 0.05]
    >>> fig = history_2d(cost_story,
    ...                  label='RMSE',
    ...                  title='Model Loss',
    ...                  filename='training_loss')

    Plotting multiple metrics with individual log-scale controls (2D data):

    >>> history = np.random.rand(100, 2)
    >>> fig = history_2d(
    ...     history,
    ...     label=['Depth', 'Timing'],
    ...     log_y=[True, False],
    ...     title='Training Metrics',
    ...     filename='../img/depth_and_timing_hist')

    """
    history = np.asarray(history)
    x = np.arange(len(history))

    tight_margins = dict(l=20, r=20, t=50 if title else 20, b=20)
    if history.ndim == 1:
        if len(history) > 1_001:
            if log_y:
                approx = split_curve_1d(np.log(history), 1001, 1e-4)
            else:
                approx = split_curve_1d(history, 1001, 1e-4)

            sub_hist = history[approx]
            x = x[approx]
            
        else:
            sub_hist = history
            

        fig = go.Figure()

        trace_name = label[0] if isinstance(label, list) else label
        is_log = log_y[0] if isinstance(log_y, list) else log_y

        fig.add_trace(go.Scatter(
            x=x, 
            y=sub_hist, 
            mode='lines+markers',
            name=trace_name))

        fig.update_layout(
            title=title,
            xaxis=dict(title=xlabel, showgrid=True),
            yaxis=dict(title=ylabel if isinstance(label,
                                    list) else label,
                       showgrid=True,
                       type='log' if is_log else 'linear',
                       exponentformat='power'),

            hovermode='x unified',
            margin=tight_margins)

    else:
        num_metrics = history.shape[1]

        fig = make_subplots(
            rows=num_metrics, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.01,
            subplot_titles=label if isinstance(label,
                list) else [f'{label} {i+1}' for i in range(num_metrics)])

        for i in range(num_metrics):
            # Extract the correct name from the label list
            trace_name = label[i] if isinstance(label,
                list) and len(label) > i else f'Metric {i+1}'

            if isinstance(log_y, list):
                is_log = log_y[i] if len(log_y) > i else False
            else:
                is_log = log_y

            fig.add_trace(go.Scatter(x=x,
                                     y=history[:, i],
                                     mode='lines+markers',
                                     name=trace_name), row=i+1, col=1)

            # Apply y-axis label for each specific subplot
            fig.update_yaxes(title_text=trace_name,
                             showgrid=True,
                             type='log' if is_log else 'linear',
                             exponentformat='power',
                             row=i+1, col=1)

        # Apply x-axis label only to the bottom subplot
        fig.update_xaxes(title_text=xlabel, showgrid=True, row=num_metrics,
                         col=1)

        fig.update_layout(
            title=title,
            hovermode='x unified',
            height=300*num_metrics,
            showlegend=False,
            margin=tight_margins)

    if type(filename) == str:
        if len(filename) > 0:
            if filename.endswith('.html'):
                save_path = filename
            else:
                save_path = f'{filename}.html'

            fig.write_html(save_path)

    fig.show()
    return fig

def history_3d(history:np.ndarray|list[list[float]],
               stations:np.ndarray|list[list[float]]|None=None,
               station_labels:list[str]|None=None,
               title:str|None=None,
               filename:str|None=None,
               height:int=1000) -> go.Figure:
    """
    Plots a 3D spatial coordinate trajectory (X, Y, Z) and ground station
    locations.

    Parameters
    ----------
    history : numpy.ndarray or list
        3D coordinate trajectory history over N steps, where columns
        represent [X, Y, Z].
    stations : numpy.ndarray or list
        Coordinates of S static stations/receivers, where columns represent
        [X, Y, Z].
    station_labels : list or None, optional
        Custom names or labels for each station. If None, stations are
        labeled as "Station 1", "Station 2", etc.
    title : str or None, optional
        Overall title for the plot layout.
    filename : str or None, optional
        Output file path to save the interactive figure as an HTML file.
        If provided without a `.html` extension, `.html` will be appended
        automatically.
    height : int, optional
        Height of the figure in pixels. The default is 1000.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The constructed interactive 3D Plotly figure object.

    Examples
    --------
    >>> trajectory = np.random.randn(100, 3).cumsum(axis=0)
    >>> station_coords = [[0, 0, 0], [10, 10, 0]]
    >>> fig = history_3d(
    ...     trajectory,
    ...     stations=station_coords,
    ...     station_labels=['Base Alpha', 'Base Beta'],
    ...     title='Target 3D Tracking',
    ...     filename='tracking_3d'
    ... )

    """
    history = np.asarray(history)
    if history.ndim != 2 or history.shape[1] != 3:
        raise ValueError(
            "`history` must be a 2D array-like structure with shape (N, 3).")

    steps = np.arange(0, len(history), 1).astype(str)
    tight_margins = dict(l=20, r=20, t=50 if title else 20, b=20)

    fig = go.Figure()

    # Trajectory Path
    if len(history) > 1_001:
        indexes = split_curve_3d(history, 1_001, 1e-4)
        sub_hist = history[indexes]
        steps = indexes
    else:
        sub_hist = history

    fig.add_trace(go.Scatter3d(x=sub_hist[:, 0], y=sub_hist[:, 1],
                               z=sub_hist[:, 2],
                               mode='lines+markers',
                               name='Trajectory',
                               customdata=steps,
                               hovertemplate=('<b>Step %{customdata}</b><br>'
                                              'X: %{x:.2f}<br>'
                                              'Y: %{y:.2f}<br>'
                                              'Z: %{z:.2f}<extra></extra>'),
                               marker=dict(size=3),
                               line=dict(width=4)))

    # Ground Station Markers
    if stations is not None:
        stations = np.asarray(stations)
        if station_labels is None:
            station_labels = [f'Station {i+1}' for i in range(len(stations))]

        fig.add_trace(
            go.Scatter3d(x=stations[:, 0],
                         y=stations[:, 1],
                         z=stations[:, 2],
                         mode='markers+text',
                         name='Stations',
                         text=station_labels,
                         textposition='top center',
                         hovertemplate=('<b>%{text}</b><br>'
                                        'X: %{x:.2f}<br>'
                                        'Y: %{y:.2f}<br>'
                                        'Z: %{z:.2f}<extra></extra>'),
                         marker=dict(size=7, color='red', symbol='diamond')))

    fig.update_layout(title=title,
                      height=height,
                      margin=tight_margins,
                      scene=dict(aspectmode='data',
                                 xaxis=dict(title='X', showgrid=True),
                                 yaxis=dict(title='Y', showgrid=True),
                                 zaxis=dict(title='Z', showgrid=True),
                                 camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))))

    if type(filename) == str:
        if len(filename) > 0:
            if filename.endswith('.html'):
                save_path = filename
            else:
                save_path = f'{filename}.html'

            fig.write_html(save_path)

    fig.show()
    return fig
