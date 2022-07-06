# Basic Algorithms Library
import numpy as np
# Time Library
from datetime import timedelta

# Original class to be extended
from earlywarningsignals.signals import EWarningLDNM
import earlywarningsignals.signals.general as general
import earlywarningsignals.signals.dnm as dnm
# Import adjacency matrices based on the flight frequency for the COUNTRIES_DEFAULT
import earlywarningsignals.signals.flight_adjacencies as flight_adjacencies
# Dedicated Exceptions for the Library
from earlywarningsignals.signals.exceptions import CountryUndefinedException


class EWarningLDNMDynamic(EWarningLDNM):
    """
    Specialization of the EWarningLDNM class for the implementation of dynamic adjacency matrices based on the flight
    frequency of the window.
    Notes: It assumes that every country has the same number of reports and that there is no gap between the first date
    with covid reports and the last one. Also, it assumes tha all countries have the same date for the first report,
    and hence all countries have the same date for its last report. (All things has been proved)
    """

    def __init__(self, start_date=general.START_DATE_DEFAULT, end_date=general.END_DATE_DEFAULT,
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
                countries list isn't contain in the database. If the list of countries specified are not the same as the
                46 members of the Council of Europe.
        """
        if set(countries) != set(general.COUNTRIES_DEFAULT):
            raise CountryUndefinedException('This class only works with if <countries> is equal to the list '
                                            'of country members in the Council of Europe.')

        static_adjacency = np.zeros(shape=(len(countries), len(countries)))
        super().__init__(start_date=start_date, end_date=end_date, covid_file=covid_file, countries=countries,
                         window_size=window_size, correlation=correlation, cumulative_data=cumulative_data,
                         static_adjacency=static_adjacency, progress_bar=progress_bar)

    def generate_adjacencies_no_window(self, start_date_window):
        """
        Generates an adjacency matrix for each instant of study between the start date and the end date. The adjacency
        matrix for each time instant will be the flight frequency of its window slice, which means that will average
        between the actual instant of study and as many days prior as the windows size. The adjacency matrix will not
        only contain 1's and 0's, instead the sum of all values will add to 1.
        This class will have one adjacency more than networks, because each network is compose of two different windows.
        This method is oriented for instances with no window size, which is the same as window size equal to zero.
        For this class instantiation, it also generates an empty array with the needed shape for the storage of the
        early warning signals.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date.

        :return: List of the adjacency matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[float]]]
        """
        adjacencies = []
        adjacency = np.zeros(shape=(len(self.countries), len(self.countries)))
        adjacency += vars(flight_adjacencies)['DATE_{}_{}_{}'.format(
            str(start_date_window.day) if start_date_window.day >= 10 else ('0' + str(start_date_window.day)),
            str(start_date_window.month) if start_date_window.month >= 10 else ('0' + str(start_date_window.month)),
            str(start_date_window.year))]
        start_date_window += timedelta(days=1)
        adjacency += vars(flight_adjacencies)['DATE_{}_{}_{}'.format(
            str(start_date_window.day) if start_date_window.day >= 10 else ('0' + str(start_date_window.day)),
            str(start_date_window.month) if start_date_window.month >= 10 else ('0' + str(start_date_window.month)),
            str(start_date_window.year))]
        start_date_window += timedelta(days=1)
        adjacencies.append(adjacency / (np.sum(adjacency) / 2 if np.sum(adjacency) > 0 else 1))
        while start_date_window <= self.end_date:
            adjacency += vars(flight_adjacencies)['DATE_{}_{}_{}'.format(
                str(start_date_window.day) if start_date_window.day >= 10 else ('0' + str(start_date_window.day)),
                str(start_date_window.month) if start_date_window.month >= 10 else ('0' + str(start_date_window.month)),
                str(start_date_window.year))]
            adjacencies.append(adjacency / (np.sum(adjacency) / 2 if np.sum(adjacency) > 0 else 1))
            start_date_window += timedelta(days=1)

        adjacencies = np.array(adjacencies)
        self.l_dnm_s = np.empty((len(self.countries), adjacencies.shape[0] - 1))

        return adjacencies

    def generate_adjacencies(self, start_date_window):
        """
        Generates an adjacency matrix for each instant of study between the start date and the end date. The adjacency
        matrix for each time instant will be the flight frequency of its window slice, which means that will average
        between the actual instant of study and as many days prior as the windows size. The adjacency matrix will not
        only contain 1's and 0's, instead the sum of all values will add to 1.
        This class will have one adjacency more than networks, because each network is compose of two different windows.
        This method is oriented for instances with window size greater than zero. For this class instantiation,
        it also generates an empty array with the needed shape for the storage of the early warning signals.

        :param pandas datetime start_date_window: Start date corresponding to the first window's date, which will be
            as many days prior to the real start date of study as the size of the windows minus one.

        :return: List of the adjacency matrices for each temporal instant from the start date to the end date.
        :rtype: numpy [[[float]]]
        """
        window_size_tmp = self.window_size - 1
        adjacencies = []
        day = start_date_window
        while day + timedelta(days=window_size_tmp) <= self.end_date + timedelta(days=1):
            adjacency = np.zeros(shape=(len(self.countries), len(self.countries)))
            for i in range(window_size_tmp):
                day_tmp = day + timedelta(days=i)
                adjacency += vars(flight_adjacencies)['DATE_{}_{}_{}'.format(
                    str(day_tmp.day) if day_tmp.day >= 10 else ('0' + str(day_tmp.day)),
                    str(day_tmp.month) if day_tmp.month >= 10 else ('0' + str(day_tmp.month)),
                    str(day_tmp.year))]
            adjacencies.append(adjacency / (np.sum(adjacency) / 2 if np.sum(adjacency) > 0 else 1))
            day += timedelta(days=1)

        adjacencies = np.array(adjacencies)
        self.l_dnm_s = np.empty((len(self.countries), adjacencies.shape[0] - 1))

        return adjacencies
