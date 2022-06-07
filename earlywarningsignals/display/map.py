# Data Structures and basic Algorithms Libraries
import pandas as pd
# Graph Visualization and Map Representation Libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geopandas as gpd
# Time and Progress Bar Libraries
from datetime import timedelta
# Generic Python Libraries
import os

# All global variables of the library
from earlywarningsignals.__init__ import COUNTRY_INFO


def plot_map(net, net_countries, title=None, x_label='Longitud', y_label='Latitud', fig_size=(15, 15),
             x_lim=(-30, None), y_lim=(30, None), population_file=COUNTRY_INFO, weights=(0, 1), nodes_size=None,
             undirected=True):

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.rename(columns={'iso_a3': 'ISO-3166-Alpha3'}, inplace=True)
    world.loc[world['name'] == 'Norway', 'ISO-3166-Alpha3'] = 'NOR'
    world.loc[world['name'] == 'France', 'ISO-3166-Alpha3'] = 'FRA'

    countryInfo = pd.read_csv(population_file)

    world = world.merge(countryInfo[['ISO-3166-Alpha2', 'ISO-3166-Alpha3']], on='ISO-3166-Alpha3', how='inner')
    world = world.reindex(
        columns=['pop_est', 'continent', 'name', 'ISO-3166-Alpha2', 'ISO-3166-Alpha3', 'gdp_md_est', 'geometry'])

    with plt.style.context(("seaborn", "ggplot")):
        # Plot world
        world = world[world['ISO-3166-Alpha2'].isin(net_countries)]
        world.plot(figsize=fig_size, edgecolor="grey", color="white")

        # Plot nodes
        if nodes_size:
            for node in net_countries:
                plt.scatter(countryInfo[countryInfo['ISO-3166-Alpha2'] == node]['Long'],
                            countryInfo[countryInfo['ISO-3166-Alpha2'] == node]['Lat'],
                            c='black', s=nodes_size, alpha=0.5)

        # Plot edges
        for i, x in enumerate(net):
            for j, y in enumerate(x):
                if j > i:
                    plt.plot([countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[i]]['Long'],
                              countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[j]]['Long']],
                             [countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[i]]['Lat'],
                              countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[j]]['Lat']],
                             linewidth=y * weights[0], color='blue', alpha=weights[1])
        if not undirected:
            for i, x in enumerate(net):
                for j, y in enumerate(x):
                    if j < i:
                        plt.plot([countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[i]]['Long'],
                                  countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[j]]['Long']],
                                 [countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[i]]['Lat'],
                                  countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[j]]['Lat']],
                                 linewidth=y * weights[0], color='red', alpha=weights[1])

        if title:
            plt.title(title)
        if y_label:
            plt.ylabel(y_label, fontsize=20)
        if x_label:
            plt.xlabel(x_label, fontsize=20)
        if x_lim:
            plt.xlim(x_lim)
        if y_lim:
            plt.ylim(y_lim)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
