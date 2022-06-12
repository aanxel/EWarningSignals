# Data Structures and basic Algorithms Libraries
import pandas as pd
import numpy as np
# Graph and Network auxiliary Libraries
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci
# Time Libraries
from datetime import timedelta
# Generic Python Libraries
import warnings
import sys

# All global variables of the library
from earlywarningsignals.__init__ import *
# Original class to be extended
from earlywarningsignals.signals import EWarningGeneral
import earlywarningsignals.signals.general as general

# Default Class Parameters
THRESHOLD_DEFAULT = 0.5
CUMULATIVE_DATA_DEFAULT = False
SQUARE_ROOT_DATA = True


# https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class EWarningSpecific(EWarningGeneral):
    """
    Specialization of the EWarningGeneral general class for the generation of early warning signals and markers
    to early detect outbreaks.
    Notes: It assumes that every country has the same number of reports and that there is no gap between the first date
    with covid reports and the last one. Also, it assumes tha all countries have the same date for the first report,
    and hence all countries have the same date for its last report. (All things has been proved)
    """

    def __init__(self, start_date=general.START_DATE_DEFAULT,
                 end_date=general.END_DATE_DEFAULT,
                 covid_file=general.COVID_FILE_DEFAULT, countries=general.COUNTRIES_DEFAULT,
                 window_size=general.WINDOW_SIZE_DEFAULT, correlation=general.CORRELATION_DEFAULT,
                 threshold=THRESHOLD_DEFAULT, cumulative_data=CUMULATIVE_DATA_DEFAULT,
                 square_root_data=SQUARE_ROOT_DATA, static_adjacency=general.STATIC_ADJACENCY_DEFAULT,
                 progress_bar=general.PROGRESS_BAR_DEFAULT):
        """
        Main constructor for the Class that receive all possible parameters.

        :param pandas datetime start_date: First date of the range of days of interest.
        :param pandas datetime end_date: Last date of the range of days of interest.
        :param string covid_file: Location of the file containing the database. This file must have this structure:

            ISO-3166-Alpha2 | Country/Region | Lat        | Long     | 1/22/20 | 1/22/20 | 1/22/20 | ..... | 9/2/20
            ES              | Spain          | 40.463667  | -3.74922 | 0       | 0       | 0       | ..... | 479554
            .               | .              | .          | .        | .       | .       | .       | .     | .
            .               | .              | .          | .        | .       | .       | .       | .     | .
            .               | .              | .          | .        | .       | .       | .       | .     | .
            FR              | France         | 46.227638  | 2.213749 | 0       | 0       | 2       | ..... | 313730

            The first row defines the content of each column and must be identical at least for the first four values.
            The first column will have the ISO-3166-Alpha2 of a country, the second column will have its popular name,
            the third column will have its latitude and the fourth will have its longitude.
            In the fifth column there will be the first date with reports of cumulative covid cases, and the next
            columns will be the continuation of the previous one.
            The format of each date must be month/day/year and values cannot have zeros on its left. For example the
            second of April 2021, should be 4/2/21.
        :param [string] countries: List of countries to take into account in the ISO-3166-Alpha2 format
            (2 letters by country).
        :param int window_size: Size of the window to shift between start_date and end_date.
        :param string correlation: Type of correlation to use for each window between each pair of countries.
            List of possible correlation values:
                 - "pearson": Pearson Correlation
                 - "spearman": Spearman Correlation
                 - "kendall":Kendall Correlation
                 - any other value: Pearson Correlation
        :param float threshold: Value from which it is determined that the correlation between two countries is high
            enough establishing a connection.
        :param bool cumulative_data: Boolean that determines whether to use cumulative confirmed covid cases (True) over
            the time or new daily cases of confirmed covid cases (True).
        :param bool square_root_data: Boolean that determines whether to apply the square root to each confirmed covid
            case value to smooth the results.
        :param numpy [[float]] static_adjacency: Static adjacency for each graph.
        :param bool progress_bar: Boolean that determines whether a progress bar will be showing the progression.

        :raises:
            DateOutRangeException: If start_date is greater than end_date or the database doesn't contain it. If there
                aren't enough dates for the window size. If there aren't enough dates for a non window size
                configuration.
            CountryUndefinedException: If there are less than two selected countries. If any country inside the
                countries list isn't contain in the database.
        """
        super().__init__(start_date=start_date, end_date=end_date, covid_file=covid_file, countries=countries,
                         window_size=window_size, correlation=correlation, static_adjacency=static_adjacency,
                         progress_bar=progress_bar)
        self.cumulative_data = cumulative_data
        self.square_root_data = square_root_data
        self.threshold = threshold

        self.networks_unweighted = None

    def check_windows(self):
        """
        Specialization of the main method, where it makes sure that all the data is correctly imported and transformed
        to subsequently generate the corresponding networks matrices with its adjacencies for each instance of time
        between the start and end date. In case that there isn't enough reports previous to the start date to fill
        the window size, it shifts the start date enough dates to fulfill it. Thanks to the specialization it also
        generates a list of unweighted networks based on the original networks list and the corresponding threshold.
        """
        super().check_windows()
        self.networks_unweighted = self.generate_unweighted()

    def transform_data(self, start_date_window):
        """
        Transform the original data of cumulative confirmed covid cases to its desired form. In this specialized case,
        depending on the class properties cumulative_data and square_root_data, it will leave the covid confirmed cases
        as cumulative data (True) or will make the discrete difference to contain the daily new confirmed cases of covid
        (False); and will apply the square root to each value of covid cases to smooth it, respectively.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date, which will be as
            many days prior to the real start date of study as the size of the windows minus one.

        :return: Matrix with the transformed data. Each Row represents a country, and each Column contains the cases
        from the first date of study until the end date.
        :rtype: numpy [[float]]
        """
        transformed_dataframe = self.data_dataframe.copy()

        with warnings.catch_warnings(record=True):
            if self.square_root_data:
                transformed_dataframe.iloc[:, 4:] = transformed_dataframe.iloc[:, 4:] ** (1 / 2)
            if not self.cumulative_data:
                transformed_dataframe.iloc[:, 4:] = transformed_dataframe.iloc[:, 4:].diff(axis=1)
                transformed_dataframe.iloc[:, 4].fillna(0, inplace=True)
        transformed_dataframe.reset_index(drop=True, inplace=True)

        return transformed_dataframe[[(start_date_window + timedelta(days=i)).strftime('%' + self.os_date_format + 'm/%'
                                                                                       + self.os_date_format + 'd/%y')
                                      for i in range((self.end_date - start_date_window).days + 1)]].to_numpy()

    def generate_unweighted(self):
        """
        Generates an unweighted adjacency matrix for each instant of study between the start date and the end date.
        Each matrix is obtained by checking in the same time corresponding correlation network if the correlation
        coefficient between each pair of nodes is greater than the threshold property of the class.

        :return: List of the unweighted adjacency matrices for each temporal instant from the start date to
            the end date.
        :rtype: numpy [[[int]]]
        """
        unweighted = []
        if self.threshold == 'GC':  # Giant Component - @TODO
            for network in self.networks:
                g = nx.Graph(network)
                # https://stackoverflow.com/questions/26105764/how-do-i-get-the-giant-component-of-a-networkx-graph
                gc = max(nx.connected_components(g), key=len)
                gc_network = np.zeros(shape=network.shape, dtype=np.int8)
                for i in range(network.shape[0]):
                    if i in gc:
                        for j in range(network.shape[1]):
                            if j in gc and j > i:
                                gc_network[i, j] = 1
                                gc_network[j, i] = 1
                unweighted.append(gc_network)
        else:
            for network in self.networks:
                unweighted.append((network > self.threshold).astype(int))
        return np.array(unweighted)

    def density(self):
        """
        Calculates the early warning signals based on the network density.

        :return: List of all the values of the densities of each network between the established dates.
        :rtype: numpy [float]
        """
        densities = []
        for netUnweighted, netAdjacency in zip(self.networks_unweighted, self.adjacencies):
            # g = nx.Graph(netUnweighted)
            # densities.append(nx.density(g))
            densities.append(np.count_nonzero(netUnweighted == 1) / np.count_nonzero(netAdjacency > 0.0001))
        return np.array(densities)

    def clustering_coefficient(self):
        """
        Calculates the early warning signals based on the clustering coefficient of the network.

        :return: List of all the values of the clustering coefficients of each network between the established dates.
        :rtype: numpy [float]
        """
        clusterings = []
        for netUnweighted in self.networks_unweighted:
            g = nx.Graph(netUnweighted)
            clusterings.append(nx.average_clustering(g) / 2)
        return np.array(clusterings)

    def assortativity_coefficient(self):
        """
        Calculates the early warning signals based on the degree assortativity coefficient of the network.
        This method implements the equation of Newman 2002.

        :return: List of all the values of the degree assortativity coefficients of each network between
            the established dates.
        :rtype: numpy [float]
        """
        assortativities = []
        for netUnweighted in self.networks_unweighted:
            g = nx.Graph(netUnweighted)
            with warnings.catch_warnings(record=True):  # Avoid unnecessary warnings
                assortativities.append(nx.degree_assortativity_coefficient(g))

        return np.array(assortativities)

    def number_edges(self):
        """
        Calculates the early warning signals based on the number of edges inside the network.

        :return: List of all the values of the number of edges inside each network between the established dates.
        :rtype: numpy [int]
        """
        edges = []
        for netUnweighted in self.networks_unweighted:
            g = nx.Graph(netUnweighted)
            edges.append(nx.number_of_edges(g))
        return np.array(edges)

    def prs(self, pupulation_file=COUNTRY_INFO):
        """
        Calculates the early warning signals based on the Preparedness Risk Score (PRS) inside the network.

        :param string pupulation_file: Location of the file containing additional information of each country. In this
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

        :return: List of all the values of the Preparedness Risk Score (PRS) of each network
            between the established dates.
        :rtype: numpy [int]
        """
        prs_s = []
        population = pd.read_csv(pupulation_file)
        countries_population = np.array(
            population.loc[population['ISO-3166-Alpha2'].isin(self.data_dataframe['ISO-3166-Alpha2'].to_list())]
                      .sort_values('ISO-3166-Alpha2')['population'].to_list(), dtype=np.int64)
        for t, netUnweighted in enumerate(self.networks_unweighted):
            covid_cases = np.array(self.data_original[:, t + self.window_size - 1], dtype=np.int64)
            susceptible_cases = countries_population - covid_cases
            prs_s.append(np.dot(np.dot(susceptible_cases, netUnweighted.astype(dtype=np.int64)),
                                susceptible_cases[np.newaxis].T)[0])
        return np.array(prs_s, dtype=np.int64)

    def srs(self):
        pass

    def forman_ricci_curvature(self):
        """
        Calculates the early warning signals based on the average of the Forman Ricci Curvature of all network's edges.

        :return: List of all the values of the average Forman Ricci Curvature of all network's edges between
            the established dates.
        :rtype: numpy [float]
        """
        forman_ricci_curvatures = []
        with HiddenPrints():
            for netUnweighted in self.networks_unweighted:
                g = nx.Graph(netUnweighted)
                frc = FormanRicci(g)
                frc.compute_ricci_curvature()
                forman_ricci_curvatures.append(np.nan if g.number_of_edges() == 0 else sum(
                    [fc for (_, _, fc) in frc.G.edges(data='formanCurvature')]) / g.number_of_edges())
        return np.array(forman_ricci_curvatures)
