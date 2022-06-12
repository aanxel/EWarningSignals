import unittest

import numpy as np
import pandas as pd

from earlywarningsignals import COVID_CRIDA_CUMULATIVE
from earlywarningsignals.signals import EWarningGeneral
from earlywarningsignals.signals.exceptions import DateOutRangeException, CountryUndefinedException


class TestEWarningGeneral(unittest.TestCase):
    """
    Unittest Class used to test the class EWarningGeneral.
    """

    def test_check_dates_1(self):
        """
        Tests that the method check_dates() from the EWarningGeneral Class properly throws Exception when the dates are
        wrong. The first test will check that the start_date is not grater than the end_date.
        """
        with self.assertRaises(DateOutRangeException) as context:
            EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                            start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                            end_date=pd.to_datetime('2020-01-15', format='%Y-%m-%d'))
        self.assertEqual(str(context.exception), '<start_date> must be older than <end_date>.')

    def test_check_dates_2(self):
        """
        Tests that the method check_dates() from the EWarningGeneral Class properly throws Exception when the dates are
        wrong. This test will check that the start_date has information for each Country in the database.
        """
        with self.assertRaises(DateOutRangeException) as context:
            EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                            start_date=pd.to_datetime('2020-01-20', format='%Y-%m-%d'),
                            end_date=pd.to_datetime('2020-01-25', format='%Y-%m-%d'))
        self.assertEqual(str(context.exception), 'Dates out of range. [1/22/20 , 10/13/20] (month/day/year)')

    def test_check_dates_3(self):
        """
        Tests that the method check_dates() from the EWarningGeneral Class properly throws Exception when the dates are
        wrong. This test will check that the endDate has information for each Country in the database.
        """
        with self.assertRaises(DateOutRangeException) as context:
            EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                            start_date=pd.to_datetime('2020-01-25', format='%Y-%m-%d'),
                            end_date=pd.to_datetime('2020-10-15', format='%Y-%m-%d'))
        self.assertEqual(str(context.exception), 'Dates out of range. [1/22/20 , 10/13/20] (month/day/year)')

    def test_check_dates_4(self):
        """
        Tests that the method check_dates() from the EWarningGeneral Class is properly executed with no errors.
        """
        start_date = pd.to_datetime('2020-01-22', format='%Y-%m-%d')
        end_date = pd.to_datetime('2020-03-01', format='%Y-%m-%d')

        ew = EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE, start_date=start_date, end_date=end_date)

        self.assertEqual(ew.start_date, start_date)
        self.assertEqual(ew.end_date, end_date)

    def test_check_dates_5(self):
        """
        Tests that the method check_dates() from the EWarningGeneral Class properly throws Exception when the dates are
        wrong. This test will check that the interval of dates between the start date and the end date are equal or
        greater than the window size.
        """
        with self.assertRaises(DateOutRangeException) as context:
            EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                            start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                            end_date=pd.to_datetime('2020-02-03', format='%Y-%m-%d'))
        self.assertEqual(str(context.exception), 'The interval between the first report date in the database and the '
                                                 '<end_date> must be equal or greater than <window_size>.')

    def test_check_countries_1(self):
        """
        Tests that the method check_countries() from the EWarningGeneral Class properly throws Exception when the list
        of country's references in the ISO-3166-Alpha2 format is incorrect. This test check that there are at least two
        different countries in the countries list.
        """
        with self.assertRaises(CountryUndefinedException) as context:
            EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                            start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                            end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                            countries=['ES', 'ES'])
        self.assertEqual(str(context.exception), 'There must be at least two different ISO-3166-Alpha2 country '
                                                 'references in <countries> and must be contained in the database.')

    def test_check_countries_2(self):
        """
        Tests that the method check_countries() from the EWarningGeneral Class properly throws Exception when the list
        of country's references in the ISO-3166-Alpha2 format is incorrect. This test check that there are more than two
        countries in the countries list.
        """
        with self.assertRaises(CountryUndefinedException) as context:
            EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                            start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                            end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                            countries=['ES'])
        self.assertEqual(str(context.exception), 'There must be at least two different ISO-3166-Alpha2 country '
                                                 'references in <countries> and must be contained in the database.')

    def test_check_countries_3(self):
        """
        Tests that the method check_countries() from the EWarningGeneral Class properly throws Exception when the list
        of country's references in the ISO-3166-Alpha2 format is incorrect. This test check that incorrect country
        references in the ISO-3166-Alpha2 format not contained in the database are detected.
        """
        with self.assertRaises(CountryUndefinedException) as context:
            EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                            start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                            end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                            countries=['ES', 'FR', 'AL', 'GB', 'ERROR'])
        self.assertEqual(str(context.exception), 'All ISO-3166-Alpha2 country references in <countries> must exist '
                                                 'and be contained in the database. Errors: [\'ERROR\']')

    def test_check_windows_1(self):
        """
        Tests that the method check_windows() from the EWarningGeneral Class properly changes the start_date based on
        the dates in the database. It assures that there are enough previous information and dates to the start_date
        based on the windows size. First test will use the first date with information.
        """
        ew = EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                             start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                             end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                             progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, pd.to_datetime('2020-02-04', format='%Y-%m-%d'))

    def test_check_windows_2(self):
        """
        Tests that the method check_windows() from the EWarningGeneral Class properly changes the start_date based on
        the dates in the database. It assures that there are enough previous information and dates to the start_date
        based on the windows size. First test will use a date between the first possible date and as many days passed
        as the window size.
        """
        ew = EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                             start_date=pd.to_datetime('2020-01-26', format='%Y-%m-%d'),
                             end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                             progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, pd.to_datetime('2020-02-04', format='%Y-%m-%d'))

    def test_check_windows_3(self):
        """
        Tests that the method check_windows() from the EWarningGeneral Class properly changes the start_date based on
        the dates in the database. It assures that there are enough previous information and dates to the start_date
        based on the windows size. Last test will use a date that doesn't need any changes.
        """
        ew = EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                             start_date=pd.to_datetime('2020-02-10', format='%Y-%m-%d'),
                             end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                             progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, pd.to_datetime('2020-02-10', format='%Y-%m-%d'))

    def test_check_windows_4(self):
        """
        Tests that the method check_windows() from the EWarningGeneral Class properly changes the start_date based on
        the dates in the database. It assures that there are enough previous information and dates to the start_date
        based on the windows size. Same to test_check_windows_1() with different window_size.
        """
        static_adjacency = np.ones(shape=(len(['ES', 'FR', 'AL', 'GB']), len(['ES', 'FR', 'AL', 'GB'])))
        np.fill_diagonal(static_adjacency, 0)
        ew = EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                             start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                             end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                             countries=['ES', 'FR', 'AL', 'GB'], window_size=7,
                             static_adjacency=static_adjacency, progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, pd.to_datetime('2020-01-28', format='%Y-%m-%d'))

    def test_check_windows_5(self):
        """
        Tests that the method check_windows() from the EWarningGeneral Class properly changes the start_date based on
        the dates in the database. It assures that there are enough previous information and dates to the start_date
        based on the windows size. Same to test_check_windows_2() with different window_size.
        """
        static_adjacency = np.ones(shape=(len(['ES', 'FR', 'AL', 'GB']), len(['ES', 'FR', 'AL', 'GB'])))
        np.fill_diagonal(static_adjacency, 0)
        ew = EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                             start_date=pd.to_datetime('2020-01-26', format='%Y-%m-%d'),
                             end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                             countries=['ES', 'FR', 'AL', 'GB'], window_size=7,
                             static_adjacency=static_adjacency, progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, pd.to_datetime('2020-01-28', format='%Y-%m-%d'))

    def test_check_windows_6(self):
        """
        Tests that the method check_windows() from the EWarningGeneral Class properly changes the start_date based on
        the dates in the database. Same to test_check_windows_3() with different window_size.
        """
        static_adjacency = np.ones(shape=(len(['ES', 'FR', 'AL', 'GB']), len(['ES', 'FR', 'AL', 'GB'])))
        np.fill_diagonal(static_adjacency, 0)
        ew = EWarningGeneral(covid_file=COVID_CRIDA_CUMULATIVE,
                             start_date=pd.to_datetime('2020-02-10', format='%Y-%m-%d'),
                             end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                             countries=['ES', 'FR', 'AL', 'GB'], window_size=7,
                             static_adjacency=static_adjacency, progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, pd.to_datetime('2020-02-10', format='%Y-%m-%d'))


if __name__ == '__main__':
    unittest.main()
