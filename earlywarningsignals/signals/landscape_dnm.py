# Data Structures and basic Algorithms Libraries
import numpy as np
from scipy import stats

# Original class to be extended
from earlywarningsignals.signals import EWarningDNM
import earlywarningsignals.signals.general as general
import earlywarningsignals.signals.dnm as dnm
# Time and Progress Bar Libraries
from tqdm import tqdm
from datetime import timedelta
import multiprocessing as mp


class EWarningLDNM(EWarningDNM):
    """
    Specialization of the EWarningDNM class for the generation of early warning signals and markers to early detect
    outbreaks.
    Notes: It assumes that every country has the same number of reports and that there is no gap between the first date
    with covid reports and the last one. Also, it assumes tha all countries have the same date for the first report,
    and hence all countries have the same date for its last report. (All things has been proved)
    """

    def __init__(self, start_date=general.START_DATE_DEFAULT, end_date=general.END_DATE_DEFAULT,
                 covid_file=general.COVID_FILE_DEFAULT, countries=general.COUNTRIES_DEFAULT,
                 window_size=dnm.WINDOW_SIZE_DEFAULT, correlation=general.CORRELATION_DEFAULT,
                 cumulative_data=dnm.CUMULATIVE_DATA_DEFAULT, static_adjacency=general.STATIC_ADJACENCY_DEFAULT,
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
        self.l_dnm_s = []
        super().__init__(start_date=start_date, end_date=end_date, covid_file=covid_file, countries=countries,
                         window_size=window_size, correlation=correlation, cumulative_data=cumulative_data,
                         static_adjacency=static_adjacency, progress_bar=progress_bar)

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
                networks.append(self.window_to_network(window_t0, window_t1, i))
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
                networks.append(self.window_to_network(window_t0, window_t1, i))
                start_date_window += timedelta(days=1)
                i += 1
        return np.array(networks)

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
                networks.append(self.window_to_network(window_t0, window_t1, i))
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
                networks.append(self.window_to_network(window_t0, window_t1, i))
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
        greater than zero. For this class instantiation, it also generates an empty array with the needed shape for the
        storage of the early warning signals.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date, which will be
            as many days prior to the real start date of study as the size of the windows minus one.

        :return: List of the adjacency matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[int]]]
        """
        adjacencies = super().generate_adjacencies(start_date_window)
        self.l_dnm_s = np.empty((len(self.countries), adjacencies.shape[0] - 1))
        return adjacencies

    def generate_adjacencies_no_window(self, start_date_window):
        """
        Generates an adjacency matrix for each instant of study between the start date and the end date. By default,
        the matrix generated represents a complete graph, which means that each node can be connected to every other
        node except itself. This means that all adjacency matrices will be filled with 1's except the main diagonal
        (top-left to bottom-right) that will be filled with 0's. This class will have one adjacency more than networks,
        because each network is compose of two different windows. This method is oriented for instances with no window
        size, which is the same as window size equal to zero. For this class instantiation, it also generates an empty
        array with the needed shape for the storage of the early warning signals.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date.

        :return: List of the adjacency matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[int]]]
        """
        adjacencies = super().generate_adjacencies_no_window(start_date_window)
        self.l_dnm_s = np.empty((len(self.countries), adjacencies.shape[0] - 1))

        return adjacencies

    def parallel_window_to_network(self, node, window_t0, window_t1, adjacency):
        """
        Due to the slow performance of this specific early warning marker, it has been necessary to parallelize
        the code, specifically the function window_to_network. For this task, a function to calculate the
        Landscape - Dynamic Network Marker (L-DNM) for a specific node and window it has been created.

        :param int node: Position of the node of study.
        :param numpy [[float]] window_t0: Data of the confirmed covid cases in a fixed period of time, where the Rows
            represent each country and the Columns represent each date from the latest to the new ones.
        :param numpy [[float]] window_t1: Data of the confirmed covid cases in a fixed period of time, where the Rows
            represent each country and the Columns represent each date from the latest to the new ones.
        :param numpy [[[int]]] adjacency: List of all the adjacency matrices in the class.

        :return: The Landscape - Dynamic Network Marker (L-DNM) for a specific node and window.
        :rtype: float
        """
        # https://www.machinelearningplus.com/python/parallel-processing-python/
        sd = 0
        cc_in = 0
        cc_out = 0
        nodes_in = [node] + np.where(adjacency[node] > 0)[0].tolist()
        # Average Differential Standard Deviation of nodes in local network
        for i in nodes_in:
            sd += abs(stats.tstd(window_t1[i]) - stats.tstd(window_t0[i]))
        sd /= len(nodes_in)

        for i, (x_t0, x_t1) in enumerate(zip(window_t0, window_t1)):
            for j, (y_t0, y_t1) in enumerate(zip(window_t0, window_t1)):
                # Average Differential Correlation Coefficient within local network
                if i in nodes_in and j in nodes_in:
                    cc_in += abs(self.calculate_correlation(x_t1, y_t1) - self.calculate_correlation(x_t0, y_t0))
                # Average Differential Correlation Coefficient between a node
                # inside the local network and an outside node
                elif i in nodes_in or j in nodes_in:
                    cc_out += abs(self.calculate_correlation(x_t1, y_t1) - self.calculate_correlation(x_t0, y_t0))
        cc_in /= len(nodes_in) * len(nodes_in)
        cc_out /= len(nodes_in) * len(nodes_in)
        # Return landscape value for node
        return sd * (cc_in + cc_out)

    def window_to_network(self, window_t0, window_t1, index):
        """
        Transform the data of the confirmed covid cases of the two fixed windows with one date of difference between
        them to the graph matrix of the network, where the edges represent the coefficient correlation between
        its pair of nodes, and the nodes represent each country. For this instantiation of the class, it also
        precalculates the early warning signals based on the Landscape - Dynamic Network Marker (L-DNM).

        :param numpy [[float]] window_t0: Data of the confirmed covid cases in a fixed period of time, where the Rows
            represent each country and the Columns represent each date from the latest to the new ones.
        :param numpy [[float]] window_t1: Data of the confirmed covid cases in a fixed period of time, where the Rows
            represent each country and the Columns represent each date from the latest to the new ones.

        :return: The network's matrix created with the data of the two fixed time windows.
        :rtype: numpy [[float]]
        """
        network = np.zeros((window_t0.shape[0], window_t0.shape[0]))
        # for node, _ in enumerate(window_t1):
        #     sd = 0
        #     cc_in = 0
        #     cc_out = 0
        #     nodes_in = [node] + np.where(self.adjacencies[index + 1][node] > 0)[0].tolist()
        #     # Average Differential Standard Deviation of nodes in local network
        #     for i, (x_t0, x_t1) in enumerate(zip(window_t0, window_t1)):
        #         if i in nodes_in:
        #             sd += abs(stats.tstd(x_t1) - stats.tstd(x_t0))
        #
        #     for i, (x_t0, x_t1) in enumerate(zip(window_t0, window_t1)):
        #         for j, (y_t0, y_t1) in enumerate(zip(window_t0, window_t1)):
        #             # Average Differential Correlation Coefficient within local network
        #             cc_t0 = self.calculate_correlation(x_t0, y_t0)
        #             cc_t1 = self.calculate_correlation(x_t1, y_t1)
        #             if i in nodes_in and j in nodes_in:
        #                 cc_in += abs(cc_t1 - cc_t0)
        #             # Average Differential Correlation Coefficient between a node
        #             # inside the local network and an outside node
        #             elif i in nodes_in or j in nodes_in:
        #                 cc_out += abs(cc_t1 - cc_t0)
        #     cc_in /= len(nodes_in) * len(nodes_in)
        #     cc_out /= len(nodes_in) * len(nodes_in)
        #     # Save index
        #     self.l_dnm_s[node][index] = sd * (cc_in + cc_out)

        pool = mp.Pool(mp.cpu_count())
        dnm_index = pool.starmap(self.parallel_window_to_network,
                                 [(node, window_t0, window_t1, self.adjacencies[index + 1])
                                  for node in range(len(self.countries))])
        pool.close()

        self.l_dnm_s[:, index] = np.array(dnm_index)

        # For maintaining some values in the general network
        for node in range(len(self.countries)):
            x_t0 = window_t0[node]
            x_t1 = window_t1[node]
            for j, (y_t0, y_t1) in enumerate(zip(window_t0, window_t1)):
                if j > node:
                    cc_t0 = self.calculate_correlation(x_t0, y_t0)
                    cc_t1 = self.calculate_correlation(x_t1, y_t1)

                    network[node, j] = abs(abs(cc_t1) - abs(cc_t0))
                    network[j, node] = network[node, j]
        return np.nan_to_num(network)

    def landscape_dnm(self):
        """
        Calculates the early warning signals based on the Landscape - Dynamic Network Marker (L-DNM).

        :return: List of all the values of the Landscape - Dynamic Network Marker (L-DNM) of each
            network between the established dates.
        :rtype: numpy [[float]]
        """
        return self.l_dnm_s
