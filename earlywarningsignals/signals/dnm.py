# Data Structures and basic Algorithms Libraries
import pandas as pd
import numpy as np
from scipy import stats
# Graph and Network auxiliary Libraries
import networkx as nx
# Time and Progress Bar Libraries
from tqdm import tqdm
from datetime import timedelta
# Generic Python Libraries
import warnings

# Original class to be extended
from earlywarningsignals.signals import EWarningGeneral
import earlywarningsignals.signals.general as general
# Dedicated Exceptions for the Library
from earlywarningsignals.signals.exceptions import DateOutRangeException, CountryUndefinedException

# Default Class Parameters
WINDOW_SIZE_DEFAULT = 0
CUMULATIVE_DATA_DEFAULT = False


class EWarningDNM(EWarningGeneral):
    """
    Specialization of the EWarningGeneral general class for the generation of early warning signals and markers
    to early detect outbreaks.
    Notes: It assumes that every country has the same number of reports and that there is no gap between the first date
    with covid reports and the last one. Also, it assumes tha all countries have the same date for the first report,
    and hence all countries have the same date for its last report. (All things has been proved)
    """

    def __init__(self, start_date=general.START_DATE_DEFAULT, end_date=general.END_DATE_DEFAULT,
                 covid_file=general.COVID_FILE_DEFAULT, countries=general.COUNTRIES_DEFAULT,
                 window_size=WINDOW_SIZE_DEFAULT, correlation=general.CORRELATION_DEFAULT,
                 cumulative_data=CUMULATIVE_DATA_DEFAULT, static_adjacency=general.STATIC_ADJACENCY_DEFAULT,
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
        :param bool cumulative_data: Boolean that determines whether to use cumulative confirmed covid cases (True) over
            the time or new daily cases of confirmed covid cases (True).
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

    def check_dates(self):
        """
        Assures that the user establish a start date of study previous to the end date. Also, it assures that the
        database contains reports of covid confirmed cases for both dates, which means that it also will contain reports
        for all the dates in the interval between the selected dates of study. Finally, it checks that the interval
        between the start date and the end date is equal or greater than the windows size. The new incorporation is that
        it checks if there are enough dates in case that no window size is selected (window_size = 0 or
        window_size = None).

        :raises:
            DateOutRangeException: If start_date is greater than end_date or the database doesn't contain it. If there
                aren't enough dates for the window size. If there aren't enough dates for a non window size
                configuration.
        """
        super().check_dates()
        if (self.end_date - self.start_date).days < 3 - 1 and self.window_size == 0:
            raise DateOutRangeException('The interval between <start_date> and <end_date> must be at least of 3 days.')

    def transform_data(self, start_date_window):
        """
        Transform the original data of cumulative confirmed covid cases to its desired form. In this specialized case,
        depending on the class property cumulative_data, it will leave the covid confirmed cases as cumulative data
        (True) or will make the discrete difference to contain the daily new confirmed cases of covid (False).

        :param pandas datetime start_date_window: Start date corresponding to the first window's date, which will be
            as many days prior to the real start date of study as the size of the windows minus one.

        :return: Matrix with the transformed data. Each Row represents a country, and each Column contains
            the cases from the first date of study until the end date.
        :rtype: [[int]]
        """
        transformed_dataframe = self.data_dataframe.copy()

        with warnings.catch_warnings(record=True):
            if not self.cumulative_data:
                transformed_dataframe.iloc[:, 4:] = transformed_dataframe.iloc[:, 4:].diff(axis=1)
                transformed_dataframe.iloc[:, 4].fillna(0, inplace=True)
        transformed_dataframe.reset_index(drop=True, inplace=True)

        return transformed_dataframe[[(start_date_window + timedelta(days=i)).strftime('%' + self.os_date_format + 'm/%'
                                                                                       + self.os_date_format + 'd/%y')
                                      for i in range((self.end_date - start_date_window).days + 1)]].to_numpy()

    def check_windows(self):
        """
        Main method, that makes sure that all the data is correctly imported and transformed to subsequently generate
        the corresponding networks matrices with its adjacencies for each instance of time between the start and
        end date. In case that there isn't enough reports previous to the start date to fill the window size, it shifts
        the start date enough dates to fulfill it. In case that the window size is fixed to 0 it will use all possible
        past data between the data of study and the start date, also shifting in case of need.
        """
        rest_days = (self.start_date - pd.to_datetime(self.data_dataframe.columns[4], format='%m/%d/%y')).days
        if self.window_size is None or self.window_size == 0:
            if rest_days >= 2:
                start_date_window = self.start_date - timedelta(days=2)
            else:
                start_date_window = self.start_date - timedelta(days=rest_days)
                self.start_date += timedelta(2 - rest_days)
            self.data_original = self.import_data(start_date_window)
            self.data = self.transform_data(start_date_window)
            self.adjacencies = self.generate_adjacencies_no_window(start_date_window)
            self.networks = self.generate_networks_no_window(start_date_window)
        else:
            self.window_size += 1
            if rest_days >= self.window_size:
                start_date_window = self.start_date - timedelta(self.window_size - 1)
                self.data_original = self.import_data(start_date_window)
                self.data = self.transform_data(start_date_window)
                self.adjacencies = self.generate_adjacencies(start_date_window)
                self.networks = self.generate_networks(start_date_window)
            else:
                start_date_window = self.start_date - timedelta(rest_days)
                self.data_original = self.import_data(start_date_window)
                self.data = self.transform_data(start_date_window)
                self.adjacencies = self.generate_adjacencies(start_date_window)
                self.networks = self.generate_networks(start_date_window)
                self.start_date += timedelta(self.window_size - 1 - rest_days)
            self.window_size -= 1
        self.networks = np.multiply(self.networks, self.adjacencies[1:])

    def window_to_network(self, window_t0, window_t1, adjacency_t0, adjacency_t1):
        """
        Transform the data of the confirmed covid cases of the two fixed windows with one date of difference between
        them to the graph matrix of the network, where the edges represent the coefficient correlation between
        its pair of nodes, and the nodes represent each country.

        :param [[float]] window_t0: Data of the confirmed covid cases in a fixed period of time, where the Rows
            represent each country and the Columns represent each date from the latest to the new ones.
        :param [[float]] window_t1: Data of the confirmed covid cases in a fixed period of time, where the Rows
            represent each country and the Columns represent each date from the latest to the new ones.
        :param [[int]] adjacency_t0: Adjacency matrix as a 2d int array of the window_t0. In this case is not needed,
            so it will be ignored.
        :param [[int]] adjacency_t1: Adjacency matrix as a 2d int array of the window_t1. In this case is not needed,
            so it will be ignored.

        :return: The network's matrix created with the data of the two fixed time windows.
        :rtype: numpy [[float]]
        """
        network = np.zeros((window_t0.shape[0], window_t0.shape[0]))

        for i, (x_t0, x_t1) in enumerate(zip(window_t0, window_t1)):
            sd_i_t0 = stats.tstd(x_t0)
            sd_i_t1 = stats.tstd(x_t1)
            for j, (y_t0, y_t1) in enumerate(zip(window_t0, window_t1)):
                if j > i:
                    sd_j_t0 = stats.tstd(y_t0)
                    sd_j_t1 = stats.tstd(y_t1)
                    cc_t0 = self.calculate_correlation(x_t0, y_t0)
                    cc_t1 = self.calculate_correlation(x_t1, y_t1)
                    cc_t = abs(cc_t1) - abs(cc_t0)
                    sd_t = (sd_i_t1 + sd_j_t1) / 2 - (sd_i_t0 + sd_j_t0) / 2
                    network[i, j] = abs(cc_t) * abs(sd_t)
                    network[j, i] = network[i, j]
        return np.nan_to_num(network)

    def generate_networks_no_window(self, start_date_window):
        """
        Generates a correlation matrix for each instant of study between the start date and the end date. This means
        that for every pair of windows containing the confirmed covid cases for each possible pair of countries,
        are used to calculate its correlation coefficient which will determinate the weight of the edge that
        connects them both in the graph. The new incorporation is that for each network it is required a total of two
        windows for each country instead of one. This method is oriented for instances with no window size,
        which is the same as window size equal to zero.

        :param start_date_window: Start date corresponding to the first window's date, which will be as many days prior
            to the real start date of study as the size of the windows minus one.

        :return: List of the correlation matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[float]]]
        """
        networks = []
        i = 0
        if self.progress_bar:
            pbar = tqdm(total=(self.end_date + timedelta(days=1) - start_date_window).days)
            while start_date_window + timedelta(days=2) <= self.end_date:
                window_t1 = self.data[:, :i + 2 + 1]
                window_t0 = window_t1[:, :-1]
                # print(f't-1: {window_t0}')
                # print(f't: {window_t1}')
                networks.append(self.window_to_network(window_t0, window_t1,
                                                       self.adjacencies[i], self.adjacencies[i + 1]))
                start_date_window += timedelta(days=1)
                i += 1
                pbar.update(1)
            pbar.close()
        else:
            while start_date_window + timedelta(days=2) <= self.end_date:
                window_t1 = self.data[:, :i + 2 + 1]
                window_t0 = window_t1[:, :-1]
                # print(f't-1: {window_t0}')
                # print(f't: {window_t1}')
                networks.append(self.window_to_network(window_t0, window_t1,
                                                       self.adjacencies[i], self.adjacencies[i + 1]))
                start_date_window += timedelta(days=1)
                i += 1
        return np.array(networks)

    def generate_adjacencies_no_window(self, start_date_window):
        """
        Generates an adjacency matrix for each instant of study between the start date and the end date. By default,
        the matrix generated represents a complete graph, which means that each node can be connected to every other
        node except itself. This means that all adjacency matrices will be filled with 1's except the main diagonal
        (top-left to bottom-right) that will be filled with 0's. This class will have one adjacency more than networks,
        because each network is compose of two different windows. This method is oriented for instances with no window
        size, which is the same as window size equal to zero.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date.

        :return: List of the adjacency matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[int]]]
        """
        adjacencies = super().generate_adjacencies(start_date_window + timedelta(days=2))
        return np.array(adjacencies)

    def generate_networks(self, start_date_window):
        """
        Generates a correlation matrix for each instant of study between the start date and the end date. This means
        that for every pair of windows containing the confirmed covid cases for each possible pair of countries,
        are used to calculate its correlation coefficient which will determinate the weight of the edge that connects
        them both in the graph. The new incorporation is that for each network it is required a total of two windows
        for each country instead of one. This method is oriented for instances with window size greater than zero.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date, which will be
            as many days prior to the real start date of study as the size of the windows minus one.

        :return: List of the correlation matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[float]]]
        """
        networks = []
        i = 0
        if self.progress_bar:
            pbar = tqdm(total=((self.end_date + timedelta(days=2)) -
                               (start_date_window + timedelta(days=self.window_size))).days)
            while start_date_window + timedelta(days=self.window_size) <= self.end_date + timedelta(days=1):
                window = self.data[:, i:i + self.window_size]
                window_t0 = window[:, :-1]
                window_t1 = window[:, 1:]
                # print(f't-1: {window_t0}')
                # print(f't: {window_t1}')
                networks.append(self.window_to_network(window_t0, window_t1,
                                                       self.adjacencies[i], self.adjacencies[i + 1]))
                start_date_window += timedelta(days=1)
                i += 1
                pbar.update(1)
            pbar.close()
        else:
            while start_date_window + timedelta(days=self.window_size) <= self.end_date + timedelta(days=1):
                window = self.data[:, i:i + self.window_size]
                window_t0 = window[:, :-1]
                window_t1 = window[:, 1:]
                # print(f't-1: {window_t0}')
                # print(f't: {window_t1}')
                networks.append(self.window_to_network(window_t0, window_t1,
                                                       self.adjacencies[i], self.adjacencies[i + 1]))
                start_date_window += timedelta(days=1)
                i += 1
        return np.array(networks)

    def generate_adjacencies(self, start_date_window):
        """
        Generates an adjacency matrix for each instant of study between the start date and the end date. By default,
        the matrix generated represents a complete graph, which means that each node can be connected to every other
        node except itself. This means that all adjacency matrices will be filled with 1's except the main diagonal
        (top-left to bottom-right) that will be filled with 0's. This class will have one adjacency more than networks,
        because each network is compose of two different windows. This method is oriented for instances with window size
        greater than zero.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date, which will be
            as many days prior to the real start date of study as the size of the windows minus one.

        :return: List of the adjacency matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[int]]]
        """
        self.window_size -= 1
        adjacencies = super().generate_adjacencies(start_date_window)
        self.window_size += 1
        return np.array(adjacencies)

    def mst_dnm(self):
        """
        Calculates the early warning signals based on the Minimum Spanning Tree - Dynamic Network Marker (MST-DNM).

        :return: List of all the values of the Minimum Spanning Tree - Dynamic Network Marker (MST-DNM) of each network
            between the established dates.
        :rtype: numpy [float]
        """
        mst_dnm_s = []
        for network in self.networks:
            network[network < np.float64(1.0e-12)] = 0.0
            g = nx.Graph(network)
            edges = nx.minimum_spanning_tree(g).edges(data=True)
            mst_dnm_s.append(sum([w['weight'] for (_, _, w) in edges]))
        return np.array(mst_dnm_s)

    def sp_dnm(self, paths=[('NO', 'IT'), ('IE', 'UA'), ('IS', 'AZ'), ('PT', 'FI')]):
        """
        Calculates the early warning signals based on the Shortest Path - Dynamic Network Marker (SP-DNM).

        :param [(string, string)] paths: List of the pair of countries from which the shortest path will be searched.
            This pair of countries will also be lists but in this case of size two, where the first element is a
            ISO-3166-Alpha2 of the origin country and the second one is another ISO-3166-Alpha2 reference of the
            destination country.

        :return: List of all the values of the Shortest Path - Dynamic Network Marker (SP-DNM) of each network between
        the established dates.
        :rtype: numpy [[float]]

        :raises:
            CountryUndefinedException: If any ISO-3166-Alpha2 references of the parameter paths isn't contained on
                the Class, or it is incorrect.
        """
        sp_dnm_s = []
        paths_ids = []
        for (origen, destination) in paths:
            if origen not in self.countries or destination not in self.countries:
                raise CountryUndefinedException('Some ISO-3166-Alpha2 references for the paths are incorrect '
                                                'or not established in the Class.')
            paths_ids.append((self.data_dataframe.loc[self.data_dataframe["ISO-3166-Alpha2"] == origen].index[0],
                              self.data_dataframe.loc[self.data_dataframe["ISO-3166-Alpha2"] == destination].index[0]))
            sp_dnm_s.append([])
        for network in self.networks:
            network[network < np.float64(1.0e-12)] = 0.0
            g = nx.Graph(network)
            for i, (origen, destination) in enumerate(paths_ids):
                try:
                    path_length = nx.shortest_path_length(g, source=origen, target=destination, weight='weight')
                    sp_dnm_s[i].append(path_length)
                except nx.NetworkXNoPath:
                    sp_dnm_s[i].append(0)
        return np.array([np.array(i) for i in sp_dnm_s])
