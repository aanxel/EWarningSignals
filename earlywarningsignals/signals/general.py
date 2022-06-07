# Data Structures and basic Algorithms Libraries
import pandas as pd
import numpy as np
from scipy import stats
# Graph Visualization and Map Representation Libraries
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import geopandas as gpd
# Time and Progress Bar Libraries
from tqdm import tqdm
from datetime import timedelta
# Generic Python Libraries
import warnings
import pickle
import sys

# All global variables of the Library
from earlywarningsignals.__init__ import *
# Dedicated Exceptions for the Library
from earlywarningsignals.signals.exceptions import DateOutRangeException, CountryUndefinedException

# Default Class Parameters
START_DATE_DEFAULT = pd.to_datetime('2020-02-15', format='%Y-%m-%d')
END_DATE_DEFAULT = pd.to_datetime('2020-09-15', format='%Y-%m-%d')
COVID_FILE_DEFAULT = COVID_JHU_CUMULATIVE
COUNTRIES_DEFAULT = ['AL', 'AD', 'AM', 'AT', 'AZ', 'BE', 'BA', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'GE',
                     'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'LV', 'LI', 'LT', 'LU', 'MT', 'MC', 'ME', 'NL', 'MK', 'NO',
                     'PL', 'PT', 'MD', 'RO', 'SM', 'RS', 'SK', 'SI', 'ES', 'SE', 'CH', 'GB', 'TR', 'UA']
WINDOW_SIZE_DEFAULT = 14
CORRELATION_DEFAULT = 'pearson'

# Default Visualization Parameters
PROGRESS_BAR_DEFAULT = True


class EWarningGeneral:
    """
    General Class for generating early warning signals and markers to early detect outbreaks.
    Notes: It assumes that every country has the same number of reports and that there is no gap between the first date
    with covid reports and the last one. Also, it assumes tha all countries have the same date for the first report,
    and hence all countries have the same date for its last report. (All things has been proved)
    """

    def __init__(self, start_date=START_DATE_DEFAULT, end_date=END_DATE_DEFAULT,
                 covid_file=COVID_FILE_DEFAULT, countries=COUNTRIES_DEFAULT,
                 window_size=WINDOW_SIZE_DEFAULT, correlation=CORRELATION_DEFAULT,
                 progress_bar=PROGRESS_BAR_DEFAULT):
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
        :param bool progress_bar: Boolean that determines whether a progress bar will be showing the progression.

        :raises:
            DateOutRangeException: If start_date is greater than end_date or the database doesn't contain it. If there
                aren't enough dates for the window size. If there aren't enough dates for a non window size
                configuration.
            CountryUndefinedException: If there are less than two selected countries. If any country inside the
                countries list isn't contain in the database.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.countries = sorted(set(countries))  # Sorted list without repetitions
        self.window_size = window_size
        self.correlation = correlation

        self.progress_bar = progress_bar

        self.networks = None
        self.adjacencies = None
        self.data_dataframe = None
        self.data_original = None
        self.data = None

        self.data_dataframe = self.import_dataframe(covid_file)

        self.check_dates()
        self.check_countries()
        # https://docs.python.org/3/library/sys.html#sys.platform
        self.os_date_format = '#' if sys.platform == 'win32' or sys.platform == 'cygwin' else '-'

    def import_dataframe(self, covid_file):
        """
        Transform a CSV representing the complete database to a pandas dataframe.

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

        :return: The complete database as a dataframe.
        :rtype pandas dataframe
        """
        data = pd.read_csv(covid_file)
        data = data.loc[data['ISO-3166-Alpha2'].isin(self.countries)].sort_values('ISO-3166-Alpha2')
        data.reset_index(drop=True, inplace=True)
        return data

    def check_dates(self):
        """
        Assures that the user establish a start date of study previous to the end date. Also, it assures that the
        database contains reports of covid confirmed cases for both dates, which means that it also will contain reports
        for all the dates in the interval between the selected dates of study. Finally, it checks that the interval
        between the start date and the end date is equal or greater than the windows size.

        :raises:
            DateOutRangeException: If startDate is greater than endDate or the database doesn't contain it. If there
                aren't enough dates for the window size.
        """
        if self.start_date > self.end_date:
            raise DateOutRangeException('<start_date> must be older than <end_date>.')
        max_date = self.data_dataframe.columns[-1]
        min_date = self.data_dataframe.columns[4]
        if self.start_date < pd.to_datetime(min_date, format='%m/%d/%y') \
                or self.end_date > pd.to_datetime(max_date, format='%m/%d/%y'):
            raise DateOutRangeException(f'Dates out of range. [{min_date} , {max_date}] (month/day/year)')
        if self.window_size > 1 \
                and (self.end_date - pd.to_datetime(min_date, format='%m/%d/%y')).days < self.window_size - 1:
            raise DateOutRangeException('The interval between the first report date in the database and the '
                                        '<end_date> must be equal or greater than <window_size>.')

    def check_countries(self):
        """
        Assures that there are at least two selected countries. And also assures, that all the selected countries
        are contained in the database.

        :raises:
            CountryUndefinedException: If there are less than two selected countries. If any country inside
                the countries list isn't contain in the database.
        """
        countries_set = set(self.countries)
        if len(countries_set) < 2:
            raise CountryUndefinedException('There must be at least two different ISO-3166-Alpha2 country '
                                            'references in <countries> and must be contained in the database.')
        countries_db = set(self.data_dataframe['ISO-3166-Alpha2'].unique())
        if not countries_set.issubset(countries_db):
            raise CountryUndefinedException('All ISO-3166-Alpha2 country references in <countries> must exist and be '
                                            'contained in the database. Errors: '
                                            f'{sorted(countries_set.difference(countries_db))}')

    def import_data(self, start_date_window):
        """
        Import for every country, the original cumulative data of confirmed covid cases between the established dates
        of study taking into account the size of the window.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date, which will be
            as many days prior to the real start date of study as the size of the windows minus one.

        :return: Matrix with the original cumulative data of confirmed covid cases. Each Row represents a country,
            and each Column contains the cases from the first date of study until the end date.
        :rtype: numpy [[int]]
        """
        return self.data_dataframe[[(start_date_window + timedelta(days=i)).strftime('%' + self.os_date_format + 'm/%' +
                                                                                     self.os_date_format + 'd/%y')
                                    for i in range((self.end_date - start_date_window).days + 1)]].to_numpy()

    def transform_data(self, start_date_window):
        """
        Transform the original data of cumulative confirmed covid cases to its desired form. In this general case, it
        returns a copy of the same data matrix.

        :param start_date_window: Start date corresponding to the first window's date, which will be as many days prior
            to the real start date of study as the size of the windows minus one.

        :return: Matrix with the transformed data. Each Row represents a country, and each Column contains the cases
            from the first date of study until the end date.
        :rtype: numpy [[float]]
        """
        return self.data_dataframe[[(start_date_window + timedelta(days=i)).strftime('%' + self.os_date_format + 'm/%' +
                                                                                     self.os_date_format + 'd/%y')
                                    for i in range((self.end_date - start_date_window).days + 1)]].to_numpy()

    def check_windows(self):
        """
        Main method, that makes sure that all the data is correctly imported and transformed to subsequently generate
        the corresponding networks matrices with its adjacencies for each instance of time between the start and
        end date. In case that there isn't enough reports previous to the start date to fill the window size, it shifts
        the start date enough dates to fulfill it.
        """
        rest_days = (self.start_date - pd.to_datetime(self.data_dataframe.columns[4], format='%m/%d/%y')).days
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

    def calculate_correlation(self, x, y):
        """
        Computes the correlation coefficient between two arrays. Depending on the value established on the class
            property correlation different types of correlation will be used. List of possible correlation values:
                - "pearson": Pearson Correlation
                - "spearman": Spearman Correlation
                - "kendall":Kendall Correlation
                - any other value: Pearson Correlation

        :param [float] x: First data array.
        :param [float] y: Second data array.

        :return: The computed correlation coefficient, or zero in case it returns NaN.
        :rtype: float
        """
        with warnings.catch_warnings(record=True):
            if self.correlation == 'pearson':
                cc = stats.pearsonr(x, y)[0]
            elif self.correlation == 'spearman':
                cc = stats.spearmanr(x, y)[0]
            elif self.correlation == 'kendall':
                cc = stats.kendalltau(x, y)[0]
            else:
                cc = stats.pearsonr(x, y)[0]  # https://realpython.com/numpy-scipy-pandas-correlation-python/
        return 0 if np.isnan(cc) else cc

    def window_to_network(self, window):
        """
        Transform the data of the confirmed covid cases in a fixed window time to the graph matrix of the network,
        where the edges represent the coefficient correlation between its pair of nodes, and the nodes represent each
        country.

        :param [[float]] window: Data of the confirmed covid cases in a fixed period of time, where the Rows represent
            each country and the columns represent each date from the latest to the new ones.

        :return: The network's matrix created with the data of a fixed time window.
        :rtype: numpy [[float]]
        """
        network = np.zeros((window.shape[0], window.shape[0]))

        for i, x in enumerate(window):
            for j, y in enumerate(window):
                if j > i:
                    cc = self.calculate_correlation(x.tolist(), y.tolist())
                    network[i, j] = cc
                    network[j, i] = cc
        return np.nan_to_num(network)

    def generate_networks(self, start_date_window):
        """
        Generates a correlation matrix for each instant of study between the start date and the end date. This means
        that for every pair of windows containing the confirmed covid cases for each possible pair of countries,
        are used to calculate its correlation coefficient which will determinate the weight of the edge that
        connects them both in the graph.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date, which will be as
            many days prior to the real start date of study as the size of the windows minus one.

        :return: List of the correlation matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[float]]]
        """
        networks = []
        i = 0
        if self.progress_bar:
            pbar = tqdm(total=((self.end_date + timedelta(days=2)) -
                               (start_date_window + timedelta(days=self.window_size))).days)
            while start_date_window + timedelta(days=self.window_size) <= self.end_date + timedelta(days=1):
                # print(window)
                networks.append(self.window_to_network(self.data[:, i:i + self.window_size]))
                start_date_window += timedelta(days=1)
                i += 1
                pbar.update(1)
            pbar.close()
        else:
            while start_date_window + timedelta(days=self.window_size) <= self.end_date + timedelta(days=1):
                # print(window)
                networks.append(self.window_to_network(self.data[:, i:i + self.window_size]))
                start_date_window += timedelta(days=1)
                i += 1
        return np.array(networks)

    def generate_adjacencies(self, start_date_window):
        """
        Generates an adjacency matrix for each instant of study between the start date and the end date. By default,
        the matrix generated represents a complete graph, which means that each node can be connected to every other
        node except itself. This means that all adjacency matrices will be filled with 1's except the main diagonal
        (top-left to bottom-right) that will be filled with 0's.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date, which will be as
            many days prior to the real start date of study as the size of the windows minus one.

        :return: List of the adjacency matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[int]]]
        """
        adjacencies = []
        adjacency = np.ones((len(self.countries), len(self.countries)))
        np.fill_diagonal(adjacency, 0)
        day = start_date_window
        while day + timedelta(days=self.window_size) <= self.end_date + timedelta(days=1):
            adjacencies.append(adjacency)
            day += timedelta(days=1)
        return np.array(adjacencies)

    def save(self, name):
        """
        Generate and save a serialization of the constructed class with all its data to be recovered any time in the
        future.

        :param string name: Destination path of the serializable object to be saved.
        """
        with open(name, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(name):
        """
        Generate the corresponding Python Class object from a serializable file.

        :param string name: Location path of the serializable object to be transformed back into a Python Class object.

        :return: The new constructed class obtained form the serializable object.
        """
        with open(name, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def k_fold_changes(series, k_fold, step=1):
        """
        Generates the tipping points of a time series based on fold change. This measure established a tipping point if
        a value increments k_fold times from one sample to the next one.

        :param numpy [float] series: Time series for finding the tipping points
        :param float k_fold: Quantity of change between one sample and the next one.

        :return: A numpy array with the same shape as the original time series but filled with 0 except the tipping
            points which have 1.
        :rtype: numpy [int]
        """
        tipping_points = np.zeros(series.shape)

        for i in range(len(series) - step):
            for j in range(1, step + 1):
                if series[i + j] > k_fold * series[i] and series[i] != 0:
                    tipping_points[i + j] = 1

        return tipping_points
