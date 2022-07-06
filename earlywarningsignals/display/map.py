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


def plot_map(net, net_countries, title=None, x_label='Longitude', y_label='Latitude', fig_size=(15, 15),
             x_lim=(-30, None), y_lim=(30, None), population_file=COUNTRY_INFO, edge_weights=(0, 1), nodes_size=None,
             undirected=True):
    """
    Display the data of a matrix as a graph.

    :param numpy [[float]] net: 2D matrix with the edges weights of the graph to be represented.
    :param [string] net_countries: Ordered list with the ISO-3166 Alpha2 references of the countries corresponding with
        each row or column of the net parameter.
    :param string title: The title of the resulting plot.
    :param string x_label: Label value for the X Axis. It represents the longitude.
    :param string y_label: Label value for the Y Axis. It represents the latitude.
    :param (int, int) fig_size: Figure size configuration.
    :param (int, int) x_lim: Tuple that represent the bottom and upper limit respectively for the X Axis. In case of a
        None for any value of the tuple it won't establish any limit.
    :param (int, int) y_lim: Tuple that represent the left-most and right-most limit respectively for the Y Axis.
        In case of a None for any value of the tuple it won't establish any limit.
    :param string population_file: Location of the file containing additional information of each country. In this
            case the important column is the population. This file must have this structure:

            Country         | ISO-3166-Alpha2  | ISO-3166-Alpha3  | population    | Lat           | Long
            Spain           | ES               | ESP              | 46754783      | -3.74922      | 40.463667
            .               | .                | .                | .             | .             | .
            .               | .                | .                | .             | .             | .
            .               | .                | .                | .             | .             | .
            Australia       | AU               | AUS              | 25459700      | 133.0         | -25.0

            In this case the order of the columns doesn't matter. But for this special case it is mandatory to have at
            least the second column with the same header ISO-3166-Alpha2 which will contain the country reference in the
            ISO-3166-Alpha2 format, and the fourth column, also with the same column name population, which will
            contain the last known official population of the specified country.
            The rest columns are optional, first column indicates the popular name of the country,
            the third column will have its ISO-3166-Alpha3 reference, the fifth column will have its latitude and
            the sixth and last column will have its longitude.
    :param (float, float) edge_weights: Tuple that configure the display of the edges in the graph. The first value will
        determine how wide the line will be [0, inf], and the second one determine how transparent it is [0,1].
    :param float nodes_size: Determines the size of the circle that will represent each node or country [0, inf].
        In case of None, the nodes will not be plotted as a circle and only the edges will be displayed.
    :param bool undirected: Boolean that determines if the graph is undirected or directed. In case of True, the graph
        is considered as undirected which means that the net parameter is symmetrical so both halves are identical, but
        for its representation the top right half is taken. In case of False, both halves of the matrix will be plotted
        but the top right one will be in blue and the bottom left in red.
    """

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
                             linewidth=y * edge_weights[0], color='blue', alpha=edge_weights[1])
        if not undirected:
            for i, x in enumerate(net):
                for j, y in enumerate(x):
                    if j < i:
                        plt.plot([countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[i]]['Long'],
                                  countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[j]]['Long']],
                                 [countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[i]]['Lat'],
                                  countryInfo[countryInfo['ISO-3166-Alpha2'] == net_countries[j]]['Lat']],
                                 linewidth=y * edge_weights[0], color='red', alpha=edge_weights[1])

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
