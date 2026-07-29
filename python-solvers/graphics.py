
"""
Script with graphic functions.

- split_curve_1d:
    Iteratively splits a 1D curve into piecewise linear segments by finding
    the point of maximum squared error in each iteration.

- split_curve_3d:
    Iteratively splits a 3D curve into piecewise linear segments by finding
    the point of maximum squared error in each iteration.

- history_2d:
    Plots 2D metric training history over iteration steps using Plotly.

- history_3d:
    Plots a 3D spatial coordinate trajectory (X, Y, Z) and ground station
    locations.

- solver_compare:
    Plots earthquake hypocenter solutions across different solvers in 3D
    space.
    
"""

import numpy as np
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
        steps = steps[indexes]
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

def solver_compare(solvers:dict,
                   stations:np.ndarray|list|None=None,
                   station_labels:np.ndarray|list|None=None,
                   title:str|None="Earthquake Hypocenter Solver Comparison",
                   colorscale:str="Viridis",
                   filename:str|None=None,
                   height:int=1000) -> go.Figure:
    """
    Plots earthquake hypocenter solutions across different solvers in 3D
    space.

    Parameters
    ----------
    solvers : dict
        Nested dictionary where keys are solver names and values are dicts
        containing 'X', 'Y', 'Z', 't', and 'RMSE'.
    stations : numpy.ndarray or list or None, optional
        Coordinates of ground stations [[X, Y, Z], ...]. The default is None.
    station_labels : numpy.ndarray or list or None, optional
        Custom names for each station. The default is None.
    title : str or None, optional
        Plot title. The default is "Earthquake Hypocenter Solver Comparison".
    colorscale : str, optional
        Plotly color scale for RMSE values. The default is 'Viridis'.
    filename : str or None, optional
        Path to export as an interactive HTML file. The default is None.
    height : int, optional
        Figure height in pixels. The default is 1000.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The constructed interactive 3D Plotly figure object.

    """
    fig = go.Figure()
    hov_tamp_sta = ("<b>%{text}</b>"
                    "<br>X: %{x:.2f}"
                    "<br>Y: %{y:.2f}"
                    "<br>Z: %{z:.2f}<extra></extra>")

    # Map solver names to distinct Plotly 3D marker symbols
    solver_names = list(solvers.keys())
    poss_symbols = ['circle', 'square', 'diamond', 'cross', 'x',
                    'triangle-up', 'star', 'star-triangle-down', 'bowtie']

    solver_symbols = {}
    for i, name in enumerate(solver_names):
        solver_symbols[name] = poss_symbols[i % len(poss_symbols)]

    # Extract all RMSE values to set global color limits across traces
    all_rmse = [vals['RMSE'] for vals in solvers.values()]
    min_rmse, max_rmse = min(all_rmse), max(all_rmse)

    # Add a separate trace for each solver so users can toggle them in the legend
    for i, (name, vals) in enumerate(solvers.items()):
        show_colorbar = (i == 0)

        x, y, z = vals['X'], vals['Y'], vals['Z']
        t, rmse = vals['t'], vals['RMSE']

        # Format custom hover text
        hover_template = (f"<b>Solver: {name}</b><br>"
                          f"X: {x:.6f}<br>"
                          f"Y: {y:.6f}<br>"
                          f"Z: {z:.6f}<br>"
                          f"Time (t): {t:.8f} s<br>"
                          f"RMSE: {rmse:.8f}"
                          "<extra></extra>")

        fig.add_trace(
            go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', name=name,
                         hovertemplate=hover_template,
                         marker=dict(size=6, symbol=solver_symbols[name],
                                     color=[rmse], colorscale=colorscale,
                                     cmin=min_rmse,
                                     cmax=max_rmse,
                                     colorbar=dict(title=dict(
                                         text="RMSE Misfit", side="top"),
                                         orientation="h", x=0.45,
                                         xanchor="center", y=-0.05,
                                         yanchor="top", thickness=15, len=0.60
                                      ) if show_colorbar else None,
                                     opacity=0.9)))

    # Optional Ground / Seismic Stations
    if stations is not None:
        stations = np.asarray(stations)
        if station_labels is None:
            station_labels = [f"Station {i+1}" for i in range(len(stations))]

        fig.add_trace(go.Scatter3d(x=stations[:, 0], y=stations[:, 1],
            z=stations[:, 2], mode='markers+text', name='Seismic Stations',
            text=station_labels, textposition='top center',
            hovertemplate=hov_tamp_sta,
            marker=dict(size=7, color='crimson', symbol='diamond')))

    # Layout Configuration
    tight_margins = dict(l=20, r=20, t=50 if title else 20, b=20)
    fig.update_layout(title=title, height=height, margin=tight_margins,
        legend=dict(title=dict(text="Solvers"), itemsizing='constant',
                    yanchor="middle", y=0.5, xanchor="left", x=1.02),
        scene=dict(aspectmode='data',
                   xaxis=dict(title='X (km)', showgrid=True),
                    yaxis=dict(title='Y (km)', showgrid=True),
                    zaxis=dict(title='Z (Depth, km)', showgrid=True),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))))

    # Export to HTML if path provided
    if type(filename) == str:
        if len(filename) > 0:
            if filename.endswith('.html'):
                save_path = filename
            else:
                save_path = f'{filename}.html'

            fig.write_html(save_path)

    fig.show()
    return fig

def build_column(vector:np.ndarray|list, sep_before:bool=True,
                 min_width:int|None=None) -> list:
    """
    Formats a 1D vector of elements into a centered ASCII table column with
    borders.

    Parameters
    ----------
    vector : numpy.ndarray or list
        Array or list of elements (strings, numbers) to be formatted as rows
        in a single ASCII column.
    sep_before : bool, optional
        If True, includes a left border line '| ' before element text. If
        False, starts with a single space ' '. The default is True.
    min_width : int or None, optional
        Minimum character width for the contents of the column. If elements
        are shorter than min_width, padding spaces are added. The default is
        None.

    Returns
    -------
    column : list
        List of formatted string lines representing the ASCII column,
        including top, bottom, and row separator lines.

    """
    length_v = np.zeros(len(vector), dtype=int)
    for i in range(len(vector)):
        length_v[i] = len(str(vector[i]))

    if type(min_width) == int:
        if np.max(length_v) > min_width:
            max_len = np.max(length_v)
        else:
            max_len = min_width

    else:
        max_len = np.max(length_v)

    if sep_before:
        pre_sep = '| '
        ex_lin = '+'+'-'*(max_len+2)+'+'
    else:
        pre_sep = ' '
        ex_lin = '-'*(max_len+2)+'+'

    column = []
    column.append(ex_lin)
    for i in range(len(vector)):
        if length_v[i] == max_len:
            column.append(pre_sep+str(vector[i])+' |')

        elif (max_len-length_v[i])%2 == 0:
            half = (max_len-length_v[i])//2
            column.append(pre_sep+' '*half+str(vector[i])+' '*half+' |')

        elif (max_len-length_v[i])%2 == 1:
            half = (max_len-length_v[i])//2
            column.append(pre_sep+' '*half+str(vector[i])+' '*(half+1)+' |')

        column.append(ex_lin)

    return column

def build_tables(data:dict, min_width:int|None=None, rounding:int=8) -> list:
    """
    Constructs and prints an ASCII table comparing model performance metrics,
    automatically sorted by the final metric.

    Parameters
    ----------
    data : dict
        Nested dictionary where keys are model names (str) and values are
        dicts mapping metric names (str) to their numerical error/loss
        values. Example: {'Model A': {'X': 300.12, 'RMSE': 0.18}}
    min_width : int or None, optional
        Minimum column width in characters passed to `build_column`. The
        default is None.
    rounding : int, optional
        Number of decimal places to round metric values. The default is 8.

    Returns
    -------
    models_col : list
        List of string lines representing the full formatted ASCII table.

    """
    names = np.array(list(data.keys()))
    heads = ['Models'] + list(data[names[0]].keys())
    errors = [list(res.values()) for res in list(data.values())]
    errors = np.array(errors).T

    # Rank models by the last metric (ascending)
    rank = np.argsort(errors[-1])
    errors = np.round(errors, rounding)
    names = names[rank]
    errors = errors[:, rank]

    power = int(np.log10(np.max(errors[-1])))-1
    minimp = abs(int(np.log10(np.min(errors[-1]))))
    heads[-1] = heads[-1] + ' (*10^'+str(power)+')'
    errors[-1] *= 10**abs(power)
    errors[-1] = np.round(errors[-1], minimp)

    max_name_len = max([len(str(n)) for n in names]) if len(names) > 0 else 0
    header_model_len = len(heads[0])
    
    base_min_width = min_width if isinstance(min_width, int) else 0
    models_width = max(base_min_width, max_name_len, header_model_len)

    # Build the model names column with guaranteed width for header
    models_col = build_column(names, sep_before=True, min_width=models_width)
    metrics_col = []
    for i in range(len(errors)):
        if len(errors[i]) > 0:
            max_err_len = max([len(str(val)) for val in errors[i]])
        else:
            max_err_len = 0

        header_metric_len = len(heads[i + 1])
        metric_width = max(base_min_width, max_err_len, header_metric_len)
        metrics_col.append(build_column(errors[i], sep_before=False,
                                        min_width=metric_width))

    head_line = ''

    # Models header cell
    len_col = len(models_col[0]) - 4  # Available inner space
    len_h = len(heads[0])
    padding = len_col - len_h
    half = padding // 2
    extra = padding % 2
    head_line += '| '+' '*half+heads[0]+' '*(half+extra)+' |'

    # Metrics header cells
    for i in range(len(heads) - 1):
        len_h = len(heads[i + 1])
        len_col = len(metrics_col[i][0]) - 3  # Available inner space
        padding = len_col - len_h
        half = padding // 2
        extra = padding % 2
        head_line += ' '+' '*half+heads[i+1]+' '*(half+extra)+' |'

    for i in range(len(errors)):
        for j in range(len(models_col)):
            models_col[j] += metrics_col[i][j]

    models_col = [models_col[0]] + [head_line] + models_col
    for lin in models_col:
        print(lin)

    return models_col
