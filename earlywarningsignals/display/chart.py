import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from earlywarningsignals.signals.general import EWarningGeneral


def plot_chat(series, interval=None, series_2=None, x_label=None, y_label=None, y_label_2=None,
              fold_changes=-1, fold_steps=1,
              legends=None, legends_2=None, legend_position='upper center', title=None, fig_size=(16, 5), dpi=200):
    if series.shape[0] <= 0:
        raise Exception('To plot a chart the must be at least one time series.')
    if interval is None:
        if len(series.shape) > 1:
            interval = list(range(series[0].shape[0]))
        else:
            interval = list(range(series.shape[0]))

    if fig_size and dpi:
        plt.figure(figsize=fig_size, dpi=dpi)
    plt.xticks(rotation=45)
    if title:
        plt.title(title)
    if y_label:
        plt.ylabel(y_label)
    if x_label:
        plt.xlabel(x_label)

    lines = []
    # plt.yscale('linear')
    if len(series.shape) > 1:
        if type(legends) != list:
            legends = [None] * (series.shape[0])
            fold_changes = [-1] * (series.shape[0])
            fold_steps = [1] * (series.shape[0])
        for serie, legend, fold_change, fold_step in zip(series, legends, fold_changes, fold_steps):
            line, = plt.plot(interval, serie, marker='.', markersize=8, label=legend)
            lines.append(line)
            if fold_change >= 0:
                tipping_points = np.where(EWarningGeneral.k_fold_changes(
                    series=serie, k_fold=fold_change, step=fold_step) == 1)
                plt.scatter(interval[tipping_points], serie[tipping_points], marker='*', s=16**2, alpha=0.6)
    elif len(series.shape) == 1:
        line, = plt.plot(interval, series, marker='.', markersize=8, label=legends)
        lines.append(line)
        if fold_changes >= 0:
            tipping_points = np.where(EWarningGeneral.k_fold_changes(
                series=series, k_fold=fold_changes, step=fold_steps) == 1)
            plt.scatter(interval[tipping_points], series[tipping_points], marker='*', s=16 ** 2, alpha=0.6)

    if series_2 is not None and len(series_2.shape) >= 1:
        plt.twinx()  # https://matplotlib.org/3.5.1/gallery/subplots_axes_and_figures/two_scales.html
        plt.tick_params(axis='y', colors='darkred')
        if len(series_2.shape) == 1:
            line, = plt.plot(interval, series_2, color='darkred', markersize=6, label=legends_2)
            lines.append(line)
        else:
            if type(legends_2) != list:
                legends_2 = [None] * (series_2.shape[0])
            for serie, legend in zip(series_2, legends_2):
                line, = plt.plot(interval, serie, color='darkred', colormarkersize=6, label=legend)
                lines.append(line)
    if y_label_2:
        plt.ylabel(y_label_2, color='darkred')

    plt.tight_layout()

    if ((legends is not None and len(series.shape) == 1) or (
            len(series.shape) > 1 and any(legend for legend in legends))) or (
            series_2 is not None and ((legends_2 is not None and len(series_2.shape) == 1) or
                                      (len(series_2.shape) > 1 and any(legend for legend in legends_2)))):
        plt.legend(handles=lines, loc=legend_position)
