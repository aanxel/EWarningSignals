# Data Structures and basic Algorithms Libraries
import numpy as np
from scipy import stats

# Original class to be extended
from earlywarningsignals.signals import EWarningDNM
import earlywarningsignals.signals.general as general
import earlywarningsignals.signals.dnm as dnm


class EWarningLDNM(EWarningDNM):
    """
    Specialization of the EWarningDNM class for the generation of early warning signals and markers to early detect
    outbreaks.
    Notes: It assumes that every country has the same number of reports and that there is no gap between the first date
    with covid reports and the last one. Also, it assumes tha all countries have the same date for the first report,
    and hence all countries have the same date for its last report. (All things has been proved)
    """

    def __init__(self, start_date=general.START_DATE_DEFAULT,
                 end_date=general.END_DATE_DEFAULT,
                 covid_file=general.COVID_FILE_DEFAULT, countries=general.COUNTRIES_DEFAULT,
                 window_size=dnm.WINDOW_SIZE_DEFAULT, correlation=general.CORRELATION_DEFAULT,
                 cumulative_data=dnm.CUMULATIVE_DATA_DEFAULT, progress_bar=general.PROGRESS_BAR_DEFAULT):
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
                         progress_bar=progress_bar)

    # def import_data(self, start_date_window):
    #     """
    #     Import for every country, the original cumulative data of confirmed covid cases between the established dates
    #     of study taking into account the size of the window. In addition, it creates the variable that will collect
    #     the landscape signal markers.
    #
    #     :param string start_date_window: Start date corresponding to the first window's date, which will be as many days
    #         prior to the real start date of study as the size of the windows minus one.
    #
    #     :return: Matrix with the original cumulative data of confirmed covid cases. Each Row represents
    #         a country, and each Column contains the cases from the first date of study until the end date.
    #     :rtype: [[int]]
    #     """
    #     data = super().import_data(start_date_window)
    #     self.l_dnm_s = [[] for _ in range(data.shape[0])]
    #     return data

    def generate_adjacencies(self, start_date_window):
        adjacencies = super().generate_adjacencies(start_date_window)
        self.l_dnm_s = np.empty((len(self.countries), adjacencies.shape[0] - 1))
        self.l_dnm_s[:] = np.nan

        return adjacencies

    def generate_adjacencies_no_window(self, start_date_window):
        adjacencies = super().generate_adjacencies_no_window(start_date_window)
        self.l_dnm_s = np.empty((len(self.countries), adjacencies.shape[0] - 1))
        self.l_dnm_s[:] = np.nan

        return adjacencies

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
        :param [[int]] adjacency_t1: Adjacency matrix as a 2d int array of the window_t1. Used to detect all the local
            networks.

        :return: The network's matrix created with the data of the two fixed time windows.
        :rtype: [[float]]
        """
        network = np.zeros((window_t0.shape[0], window_t0.shape[0]))
        for node, _ in enumerate(window_t1):
            sd = 0
            cc_in = 0
            cc_out = 0
            nodes_in = [node] + np.where(adjacency_t1[node] == 1)[0].tolist()
            # Average Differential Standard Deviation of nodes in local network
            for i, (x_t0, x_t1) in enumerate(zip(window_t0, window_t1)):
                if i in nodes_in:
                    sd += abs(stats.tstd(x_t1) - stats.tstd(x_t0))
            sd /= len(nodes_in)

            for i, (x_t0, x_t1) in enumerate(zip(window_t0, window_t1)):
                for j, (y_t0, y_t1) in enumerate(zip(window_t0, window_t1)):
                    # Average Differential Correlation Coefficient within local network
                    cc_t0 = self.calculate_correlation(x_t0, y_t0)
                    cc_t1 = self.calculate_correlation(x_t1, y_t1)
                    if i in nodes_in and j in nodes_in:
                        cc_in += abs(cc_t1 - cc_t0)
                    # Average Differential Correlation Coefficient between a node
                    # inside the local network and an outside node
                    elif i in nodes_in or j in nodes_in:
                        cc_out += abs(cc_t1 - cc_t0)
            cc_in /= len(nodes_in) * len(nodes_in)
            cc_out /= len(nodes_in) * len(nodes_in)
            # Save index
            self.l_dnm_s[node][np.where(np.isnan(self.l_dnm_s[node]))[0][0]] = sd * (cc_in + cc_out)

            # For maintaining some values in the general network
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
        :rtype: [[float]]
        """
        return np.array([np.array(l_dnm_node) for l_dnm_node in self.l_dnm_s])
