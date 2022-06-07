from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from earlywarningsignals.__init__ import COVID_CRIDA_CUMULATIVE, ACTUAL_DIR, DATA_DIR
from earlywarningsignals.signals.flight_adjacencies import *
from earlywarningsignals.display import chart
from earlywarningsignals.display import map

from earlywarningsignals.signals import EWarningLDNM, EWarningSpecific, general


def main():
    ew = EWarningLDNM(start_date=pd.to_datetime('2020-02-15', format='%Y-%m-%d'),
                      end_date=pd.to_datetime('2020-03-15', format='%Y-%m-%d'),
                      # countries=['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'UA'],
                      window_size=0)
    ew.check_windows()

    print(f'Adjacencies: ({ew.adjacencies.shape})')
    print(f'Networks: ({ew.networks.shape})')
    print(f'landscape: ({ew.landscape_dnm().shape})')


def main1():
    std = np.std(np.array([DATE_01_01_2020, DATE_02_01_2020, DATE_03_01_2020, DATE_04_01_2020, DATE_05_01_2020,
                           DATE_06_01_2020, DATE_07_01_2020]))
    print(f'Std: {std}')
    print(f'Total Flights DATE_01_01_2020: {np.sum(DATE_01_01_2020)}')
    print(f'Total Flights DATE_02_01_2020: {np.sum(DATE_02_01_2020)}')
    print(f'Total Flights DATE_03_01_2020: {np.sum(DATE_03_01_2020)}')
    print(f'Total Flights DATE_04_01_2020: {np.sum(DATE_04_01_2020)}')
    print(f'Total Flights DATE_05_01_2020: {np.sum(DATE_05_01_2020)}')
    print(f'Total Flights DATE_06_01_2020: {np.sum(DATE_06_01_2020)}')
    print(f'Total Flights DATE_07_01_2020: {np.sum(DATE_07_01_2020)}')
    print(f'Total flights Average: {np.sum(DATE_AVERAGE_7)}')
    print(f'% of std: {std / np.sum(DATE_01_01_2020) * 100}')

    countries = general.COUNTRIES_DEFAULT

    map.plot_map(net=DATE_AVERAGE_7, net_countries=countries, weights=(0.025, 0.5), fig_size=(20, 20),
                 nodes_size=20 ** 2, undirected=True,
                 title='Frecuencia de Vuelos entre los países del Consejo de Europa (previo a la pandemia del COVID)')
    plt.show()


def main2():
    ew = EWarningSpecific(cumulative_data=False, square_root_data=False, progress_bar=False,
                          start_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                          end_date=pd.to_datetime('2020-05-01', format='%Y-%m-%d'))
    ew.check_windows()
    density_0 = ew.density()

    new_cases = np.sum(ew.data[:, ew.window_size-1:], axis=0)

    # ew = EWarningSpecific(cumulative_data=False, square_root_data=True, progress_bar=False)
    # ew.check_windows()
    # density_1 = ew.density()

    # ew = EWarningSpecific(cumulative_data=True, square_root_data=False, progress_bar=False)
    # ew.check_windows()
    # density_2 = ew.density()
    #
    # ew = EWarningSpecific(cumulative_data=True, square_root_data=True, progress_bar=False)
    # ew.check_windows()
    # density_3 = ew.density()

    interval = np.arange(np.datetime64(ew.start_date), np.datetime64(ew.end_date + timedelta(days=1)),
                         dtype='datetime64[D]')

    chart.plot_chat(series=density_0,
                    interval=interval, series_2=new_cases, fold_changes=2, fold_steps=5,
                    x_label='Intervalo de tiempo',
                    y_label='Network Density', y_label_2='Casos diarios confirmados COVID',
                    legends='densidad', legends_2='contagios COVID')
    plt.savefig("test.png", bbox_inches='tight')
    plt.show()


def main21():
    ew = EWarningSpecific(cumulative_data=False, square_root_data=False, progress_bar=False)
    ew.check_windows()
    clustering_0 = ew.clustering_coefficient()

    new_cases = np.sum(ew.data[:, ew.window_size-1:], axis=0)

    ew = EWarningSpecific(cumulative_data=False, square_root_data=True, progress_bar=False)
    ew.check_windows()
    clustering_1 = ew.clustering_coefficient()

    interval = np.arange(np.datetime64(ew.start_date), np.datetime64(ew.end_date + timedelta(days=1)),
                         dtype='datetime64[D]')

    chart.plot_chat(series=np.stack((clustering_0, clustering_1), axis=0),
                    interval=interval, series_2=new_cases, fold_changes=[3, 3], fold_steps=[7, 7],
                    title='Variants of Clustering Coefficient for JHU Dataset',
                    x_label='Timestamps interval', y_label='Clustering Coefficient',
                    y_label_2='Daily confirmed Covid cases',
                    legends=['clustering_coefficient_0', 'clustering_coefficient_1'], legends_2='Daily Covid cases')
    plt.show()


def main3():
    countries = ['DE', 'ES', 'FR', 'GB', 'IT']
    net = [[0, 0.2, 0.5, 0.3, 0.1], [0.4, 0, 0.6, 0.8, 0.2], [0.9, 0.1, 0, 0.2, 0.3], [0.5, 0.2, 0.1, 0, 0.9],
           [0.1, 0.7, 0.3, 0.4, 0]]
    net1 = [[0, 0.1, 0, 0, 0], [0, 0, 0, 0.8, 0], [0, 0, 0, 0, 0], [0.7, 0.2, 0, 0, 0],
            [0, 0, 0, 0, 0]]

    map.plot_map(net=net, net_countries=countries, x_lim=(-10, 20), y_lim=(35, 60), weights=(20, 0.6),
                 nodes_size=40**2, undirected=True,
                 title='Representación de un grafo sobre el mapa de España, Francia, Italia, Alemania y Reino Unido (Diagonal Superior)')

    map.plot_map(net=net1, net_countries=countries, x_lim=(-10, 20), y_lim=(35, 60), weights=(20, 0.6),
                 nodes_size=40 ** 2, undirected=False,
                 title='Mapa de España, Francia, Italia, Alemania y Reino Unido, sin conexiones en Italia ni en Francia')

    plt.show()


def main4():
    countries = ['DE', 'ES', 'FR', 'GB', 'IT']
    # net = [[0, 0, 0, 1, 1],
    #        [0, 0, 1, 1, 1],
    #        [0, 0, 0, 1, 1],
    #        [0, 0, 0, 0, 1],
    #        [0, 0, 0, 0, 0]]
    net = [[0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1],
           [0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0]]

    map.plot_map(net=net, net_countries=countries, x_lim=(-10, 20), y_lim=(35, 60), weights=(10, 0.6),
                 nodes_size=40 ** 2, undirected=True)
    # plt.savefig("test.png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main2()
