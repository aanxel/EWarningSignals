import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from earlywarningsignals.signals.general import EWarningGeneral


# def plot_chat(series, interval=None, series_2=None, x_label=None, y_label=None, y_label_2=None,
#               fold_changes=-1, fold_steps=1,
#               legends=None, legends_2=None, legend_position='upper center', title=None, fig_size=(16, 5), dpi=200):
#     if series.shape[0] <= 0:
#         raise Exception('To plot a chart the must be at least one time series.')
#     if interval is None:
#         if len(series.shape) > 1:
#             interval = list(range(series[0].shape[0]))
#         else:
#             interval = list(range(series.shape[0]))
#
#     if fig_size and dpi:
#         plt.figure(figsize=fig_size, dpi=dpi)
#     plt.xticks(rotation=45)
#     if title:
#         plt.title(title)
#     if y_label:
#         plt.ylabel(y_label)
#     if x_label:
#         plt.xlabel(x_label)
#
#     lines = []
#     # plt.yscale('linear')
#     if len(series.shape) > 1:
#         if type(legends) != list:
#             legends = [None] * (series.shape[0])
#             fold_changes = [-1] * (series.shape[0])
#             fold_steps = [1] * (series.shape[0])
#         for serie, legend, fold_change, fold_step in zip(series, legends, fold_changes, fold_steps):
#             line, = plt.plot(interval, serie, marker='.', markersize=8, label=legend)
#             lines.append(line)
#             if fold_change >= 0:
#                 tipping_points = np.where(EWarningGeneral.k_fold_changes(
#                     series=serie, k_fold=fold_change, step=fold_step) == 1)
#                 plt.scatter(interval[tipping_points], serie[tipping_points], marker='*', s=16**2, alpha=0.6)
#     elif len(series.shape) == 1:
#         line, = plt.plot(interval, series, marker='.', markersize=8, label=legends)
#         lines.append(line)
#         if fold_changes >= 0:
#             tipping_points = np.where(EWarningGeneral.k_fold_changes(
#                 series=series, k_fold=fold_changes, step=fold_steps) == 1)
#             plt.scatter(interval[tipping_points], series[tipping_points], marker='*', s=16 ** 2, alpha=0.6)
#
#     if series_2 is not None and len(series_2.shape) >= 1:
#         plt.twinx()  # https://matplotlib.org/3.5.1/gallery/subplots_axes_and_figures/two_scales.html
#         plt.tick_params(axis='y', colors='darkred')
#         if len(series_2.shape) == 1:
#             line, = plt.plot(interval, series_2, color='darkred', markersize=6, label=legends_2)
#             lines.append(line)
#         else:
#             if type(legends_2) != list:
#                 legends_2 = [None] * (series_2.shape[0])
#             for serie, legend in zip(series_2, legends_2):
#                 line, = plt.plot(interval, serie, color='darkred', colormarkersize=6, label=legend)
#                 lines.append(line)
#     if y_label_2:
#         plt.ylabel(y_label_2, color='darkred')
#
#     plt.tight_layout()
#
#     if ((legends is not None and len(series.shape) == 1) or (
#             len(series.shape) > 1 and any(legend for legend in legends))) or (
#             series_2 is not None and ((legends_2 is not None and len(series_2.shape) == 1) or
#                                       (len(series_2.shape) > 1 and any(legend for legend in legends_2)))):
#         plt.legend(handles=lines, loc=legend_position)


def plot_chat(series, interval=None, series_2=None, x_label=None, y_label=None, y_label_2=None, title=None,
              fold_changes=None, fold_steps=None,
              legends=None, legends_2=None, legend_position='upper center', fig_size=(16, 5), dpi=200):
    """
    This function let you plot a group of time series on the left Y Axis to be compared with another time series with
    complete different scale on le right Y Axis. Additionally, it let you configure the basic parameters of the
    resultant plot like title, axis names and related. Finally, it implements a fold change measure to detect the
    individual tipping points for each time series.

    :param numpy [[float]] series: Group of time series to be plotted on the left Axis. This will contain the early
        warning markers.
    :param numpy [datetime] interval: List of days corresponding to the interval of study. In case of None values from
        zero will be generated.
    :param numpy [[float]] series_2: Time series to be plotted on the right Axis. This will contain the evolution of
        the new daily COVID confirmed cases.
    :param string x_label: Label value for the X Axis. It represents the countries.
    :param string y_label: Label value for the Y left Axis. It represents the early warning marker.
    :param string y_label_2: Label value for the Y right Axis. It represents the new daily COVID confirmed cases.
    :param numpy [float] fold_changes: Quantity of change between one sample and the next one needed for the calculation
        of the tipping points based on the fold-change measure. If is None, the fold change measure won't be calculated.
    :param numpy [int] fold_steps: Number of samples from the original one to be compared for the calculation of the
        tipping points based on the fold-change measure. Only used if <fold_changes> is greater than zero.
    :param legends: List of names to be shown in the legend for each time series in the series parameter.
    :param legends_2: Name to be shown in the legend for the series_2 parameter.
    :param legend_position: Position that the legend will be taken based on mathplotlib options.
    :param string title: The title of the resulting plot.
    :param (int, int) fig_size: Figure size configuration.
    :param int dpi: Dots per inch (DPI) of the resultant figure.
    """
    if len(series.shape) != 2:
        raise Exception('The parameter series must have shape 2, even if you intend to only print one time series.')
    if interval is None:
        interval = list(range(series[0].shape[0]))
    lines = []

    if fig_size and dpi:
        plt.figure(figsize=fig_size, dpi=dpi)
    plt.xticks(rotation=45)
    if title:
        plt.title(title)
    if y_label:
        plt.ylabel(y_label)
    if x_label:
        plt.xlabel(x_label)

    if legends is None:
        legends = [None] * series.shape[0]
    if fold_changes is None:
        fold_changes = [-1] * series.shape[0]
        fold_steps = [1] * series.shape[0]
    elif fold_steps is None:
        fold_steps = [1] * series.shape[0]
    for serie, legend, fold_change, fold_step in zip(series, legends, fold_changes, fold_steps):
        line, = plt.plot(interval, serie, marker='.', markersize=8, label=legend)
        lines.append(line)
        if fold_change >= 0:
            tipping_points = np.where(EWarningGeneral.k_fold_changes(
                series=serie, k_fold=fold_change, step=fold_step) == 1)
            plt.scatter(interval[tipping_points], serie[tipping_points], marker='*', s=16**2, alpha=0.6)

    if series_2 is not None:
        plt.twinx()  # https://matplotlib.org/3.5.1/gallery/subplots_axes_and_figures/two_scales.html
        plt.tick_params(axis='y', colors='darkred')
        line, = plt.plot(interval, series_2, color='darkred', markersize=6, label=legends_2)
        lines.append(line)
    if y_label_2:
        plt.ylabel(y_label_2, color='darkred')
    plt.tight_layout()

    if legends[0] is not None or legends_2 is not None:
        plt.legend(handles=lines, loc=legend_position)


def plot_chart_multiple(series, interval=None, series_2=None, x_label=None, y_label=None, y_label_2=None,
                        fold_change=-1, fold_step=1, fold_percentage=1,
                        legends=None, legends_2=None, legend_position='upper center',
                        title=None, fig_size=(16, 5), dpi=200):
    """
    This function let you plot a group of time series on the left Y Axis to be compared with another time series with
    complete different scale on le right Y Axis. Additionally, it let you configure the basic parameters of the
    resultant plot like title, axis names and related. Finally, and the spect that makes the difference between this
    function and plot_chart() is that the fold change measure takes into account all the time series simultaneously.

    :param numpy [[float]] series: Group of time series to be plotted on the left Axis. This will contain the early
        warning markers.
    :param numpy [datetime] interval: List of days corresponding to the interval of study. In case of None values from
        zero will be generated.
    :param numpy [[float]] series_2: Time series to be plotted on the right Axis. This will contain the evolution of
        the new daily COVID confirmed cases.
    :param string x_label: Label value for the X Axis. It represents the countries.
    :param string y_label: Label value for the Y left Axis. It represents the early warning marker.
    :param string y_label_2: Label value for the Y right Axis. It represents the new daily COVID confirmed cases.
    :param float fold_change: Quantity of change between one sample and the next one needed for the calculation of the
        tipping points based on the fold-change measure. Only used if its values is greater than zero.
    :param int fold_step: Number of samples from the original one to be compared for the calculation of the
        tipping points based on the fold-change measure. Only used if <fold_changes> is greater than zero.
    :param int fold_percentage: Minimum percentage of time series to fulfill the fold change measure. Only used
        if fold_changes is greater than zero.
    :param legends: List of names to be shown in the legend for each time series in the series parameter.
    :param legends_2: Name to be shown in the legend for the series_2 parameter.
    :param legend_position: Position that the legend will be taken based on mathplotlib options.
    :param string title: The title of the resulting plot.
    :param (int, int) fig_size: Figure size configuration.
    :param int dpi: Dots per inch (DPI) of the resultant figure.
    """
    if len(series.shape) != 2:
        raise Exception('To plot a chart the must be at least two time series.')
    if interval is None:
        interval = list(range(series[0].shape[0]))
    lines = []

    if fig_size and dpi:
        plt.figure(figsize=fig_size, dpi=dpi)
    plt.xticks(rotation=45)
    if title:
        plt.title(title)
    if y_label:
        plt.ylabel(y_label)
    if x_label:
        plt.xlabel(x_label)

    if legends is None:
        legends = [None] * series.shape[0]
    if fold_change >= 0:
        tipping_points = np.where(EWarningGeneral.k_fold_changes_multiple(
            series=series, k_fold=fold_change, step=fold_step, rate_compare=fold_percentage) == 1)
    for serie, legend in zip(series, legends):
        line, = plt.plot(interval, serie, marker='.', markersize=8, label=legend)
        lines.append(line)
        if fold_change >= 0:
            plt.scatter(interval[tipping_points], serie[tipping_points], marker='*', s=16 ** 2, alpha=0.6)

    if series_2 is not None:
        plt.twinx()  # https://matplotlib.org/3.5.1/gallery/subplots_axes_and_figures/two_scales.html
        plt.tick_params(axis='y', colors='darkred')
        line, = plt.plot(interval, series_2, color='darkred', markersize=6, label=legends_2)
        lines.append(line)
    if y_label_2:
        plt.ylabel(y_label_2, color='darkred')

    if legends[0] is not None or legends_2 is not None:
        plt.legend(handles=lines, loc=legend_position)

    plt.tight_layout()
