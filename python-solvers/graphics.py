
"""
Script with graphic functions.
- history_2d:
    Plots 2D metric training history over iteration steps using Plotly.

- history_3d:
    Plots a 3D spatial coordinate trajectory (X, Y, Z) and ground station
    locations.

"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        If provided without a `.html` extension, `.html` will be appended automatically.

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
        fig = go.Figure()

        trace_name = label[0] if isinstance(label, list) else label
        is_log = log_y[0] if isinstance(log_y, list) else log_y

        fig.add_trace(go.Scatter(
            x=x, 
            y=history, 
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
    fig.add_trace(go.Scatter3d(x=history[:, 0], y=history[:, 1],
                               z=history[:, 2],
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
