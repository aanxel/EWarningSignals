# Basic Algorithms Libraries
import numpy as np
# Graph Visualization and Map Representation Libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Base class of early warning signals with the fold_change mease implemented
from earlywarningsignals.signals import EWarningGeneral

def plot_landscape(series, projections=[(0, 0), (0, 90), (0, 60), (-20, 60), (0, 150), (-20, 150)],
                   x_label='Country', y_label='Time Interval', z_label='L-DNM', title=None,
                   fold_changes=-1, fold_steps=1, fold_percentage=0.8, colors=('c', 'r'), fig_size=(15, 20), dpi=200):
    """
    Display the data of any 2D matrix as a 3D bars plot. Also, it incorporates the possibility of mark
    the tipping points based on the fold-change measure.

    :param numpy [[float]] series: 2D data matrix of data to be represented. The number of rows represent the countries,
        and the number of columns the time interval of study.
    :param [(int, int)] projections: List with the projections or views in with the series will be represented in 3D.
    :param string x_label: Label value for the X Axis. It represents the countries.
    :param string y_label: Label value for the Y Axis. It represents the early warning marker.
    :param string z_label: Label value for the Z Axis. It represents the time series intervals (weeks, days...).
    :param string title: The title of the resulting plot.
    :param float fold_changes: Quantity of change between one sample and the next one needed for the calculation of the
        tipping points based on the fold-change measure. Only used if its values is greater than zero.
    :param int fold_steps: Number of samples from the original one to be compared for the calculation of the
        tipping points based on the fold-change measure. Only used if <fold_changes> is greater than zero.
    :param int fold_percentage: Minimum percentage of time series to fulfill the fold change measure. Only used
        if fold_changes is greater than zero.
    :param (string, string) colors: The first value represent the default value for the 3D bars, while the second string
        represent the value for the tipping points intervals.
    :param (int, int) fig_size: Figure size configuration.
    :param int dpi: Dots per inch (DPI) of the resultant figure.

    """
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    if len(projections) == 0:
        raise Exception('There must be at least one projection. For example <projections> = [(0, 0), (0, 90)]')

    subplot_value = 321
    axs = []
    for _ in projections:
        ax = fig.add_subplot(subplot_value, projection='3d')
        axs.append(ax)
        subplot_value += 1

    x = []
    y = []
    dz = []
    for i, _x in enumerate(series):
        for j, _y in enumerate(_x):
            x.append(i)
            y.append(j)
            dz.append(_y)

    z = np.zeros(len(x))
    dx = np.ones(len(x))
    dy = np.ones(len(x))
    if fold_changes is None or fold_changes <= 0:
        color = [colors[0]] * len(x)
    else:
        tipping_points = EWarningGeneral.k_fold_changes_multiple(series, fold_changes, fold_steps, fold_percentage)
        color = np.tile(np.where(tipping_points == 1, colors[1], colors[0]), series.shape[0])
    for ax, (elv, azim) in zip(axs, projections):
        ax.bar3d(x, y, z, dx, dy, dz, color)
        ax.view_init(ax.elev + elv, ax.azim + azim)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if z_label:
            ax.set_zlabel(z_label)

    if title:
        plt.title(title)
