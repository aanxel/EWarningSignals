import unittest
import pandas as pd
import numpy as np
from datetime import timedelta

from earlywarningsignals import COVID_CRIDA_CUMULATIVE
from earlywarningsignals.signals import EWarningDNM
from earlywarningsignals.signals.exceptions import DateOutRangeException, CountryUndefinedException


class MyTestCase(unittest.TestCase):
    """
    Unittest Class used to test the class EWarningDNM.
    """

    def test_check_dates_1(self):
        """
        Tests that the method check_dates() from the EWarningDNM Class properly throws Exception when the dates are
        wrong. The test will check that the interval of days between start date and the end date is not greater or equal
        to three when no window size or window size equal to zero.
        """
        with self.assertRaises(DateOutRangeException) as context:
            EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                        start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                        end_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                        window_size=0, cumulative_data=False)
        self.assertEqual(str(context.exception), 'The interval between <start_date> and <end_date> must be at least of '
                                                 '3 days.')

    def test_check_dates_2(self):
        """
        Tests that the method check_dates() from the EWarningDNM Class properly throws Exception when the dates are
        wrong. This test will check that the interval of dates between the start date and the end date are equal or
        greater than the window size.
        """
        with self.assertRaises(DateOutRangeException) as context:
            EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                        start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                        end_date=pd.to_datetime('2020-02-03', format='%Y-%m-%d'),
                        window_size=14, cumulative_data=False)
        self.assertEqual(str(context.exception), 'The interval between the first report date in the database '
                                                 'and the <end_date> must be equal or greater than <window_size>.')

    def test_check_dates_3(self):
        """
        Tests that the method check_dates() from the EWarningDNM Class is properly executed with no errors when none
        window size or window size equal to zero.
        """
        start_date = pd.to_datetime('2020-01-22', format='%Y-%m-%d')
        end_date = pd.to_datetime('2020-02-22', format='%Y-%m-%d')

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE, start_date=start_date, end_date=end_date,
                         window_size=0, cumulative_data=False)

        self.assertEqual(ew.start_date, start_date)
        self.assertEqual(ew.end_date, end_date)

    def test_check_dates_4(self):
        """
        Tests that the method check_dates() from the EWarningDNM Class is properly executed with no errors when window
        size is greater than zero.
        """
        start_date = pd.to_datetime('2020-01-22', format='%Y-%m-%d')
        end_date = pd.to_datetime('2020-03-01', format='%Y-%m-%d')

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE, start_date=start_date, end_date=end_date,
                         window_size=7, cumulative_data=False)

        self.assertEqual(ew.start_date, start_date)
        self.assertEqual(ew.end_date, end_date)

    def test_check_windows_1(self):
        """
        Tests that the method check_windows() from the EWarningDNM Class properly changes the start date based on the
        dates in the database. It assures that there are enough previous information and dates to the start date based
        on the windows size. This test will use the first date of the database and none window size or window size equal
        to zero.
        """
        start_date = pd.to_datetime('2020-01-22', format='%Y-%m-%d')
        end_date = pd.to_datetime('2020-02-10', format='%Y-%m-%d')

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE, start_date=start_date, end_date=end_date,
                         window_size=0, cumulative_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, start_date + timedelta(2))
        self.assertEqual(ew.end_date, end_date)
        self.assertEqual(ew.adjacencies.shape[0], ew.networks.shape[0] + 1)

    def test_check_windows_2(self):
        """
        Tests that the method check_windows() from the EWarningDNM Class properly changes the start date based on the
        dates in the database. It assures that there are enough previous information and dates to the start date based
        on the windows size. This test will use the first date ont the database and none window size greater than zero.
        """
        start_date = pd.to_datetime('2020-01-22', format='%Y-%m-%d')
        end_date = pd.to_datetime('2020-02-10', format='%Y-%m-%d')

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE, start_date=start_date, end_date=end_date,
                         window_size=14, cumulative_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, start_date + timedelta(14))
        self.assertEqual(ew.end_date, end_date)
        self.assertEqual(ew.adjacencies.shape[0], ew.networks.shape[0] + 1)

    def test_check_windows_3(self):
        """
        Tests that the method check_windows() from the EWarningDNM Class properly changes the start date based on the
        dates in the database. It assures that there are enough previous information and dates to the start date based
        on the windows size. This test will use a date between the first possible date the next three days, because
        it is the minimum interval when no window size or window size equal to zero.
        """
        start_date = pd.to_datetime('2020-01-23', format='%Y-%m-%d')
        end_date = pd.to_datetime('2020-03-01', format='%Y-%m-%d')

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE, start_date=start_date, end_date=end_date,
                         window_size=0, cumulative_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, start_date + timedelta(1))
        self.assertEqual(ew.end_date, end_date)
        self.assertEqual(ew.adjacencies.shape[0], ew.networks.shape[0] + 1)

    def test_check_windows_4(self):
        """
        Tests that the method check_windows() from the EWarningDNM Class properly changes the start date based on the
        dates in the database. It assures that there are enough previous information and dates to the start date based
        on the windows size. This test will use a date between the first possible date and as many days passed as the
        window size, when this is greater than zero.
        """
        start_date = pd.to_datetime('2020-01-26', format='%Y-%m-%d')
        end_date = pd.to_datetime('2020-02-25', format='%Y-%m-%d')

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE, start_date=start_date, end_date=end_date,
                         window_size=14, cumulative_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, start_date + timedelta(10))
        self.assertEqual(ew.end_date, end_date)
        self.assertEqual(ew.adjacencies.shape[0], ew.networks.shape[0] + 1)

    def test_check_windows_5(self):
        """
        Tests that the method check_windows() from the EWarningDNM Class properly changes the start date based on the
        dates in the database. It assures that there are enough previous information and dates to the start date based
        on the windows size. This test will use a date that doesn't need any changes and no window size or
        window size equal to zero.
        :return:
        """
        start_date = pd.to_datetime('2020-02-10', format='%Y-%m-%d')
        end_date = pd.to_datetime('2020-03-01', format='%Y-%m-%d')

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE, start_date=start_date, end_date=end_date,
                         window_size=0, cumulative_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, start_date)
        self.assertEqual(ew.end_date, end_date)
        self.assertEqual(ew.adjacencies.shape[0], ew.networks.shape[0] + 1)

    def test_check_windows_6(self):
        """
        Tests that the method check_windows() from the EWarningDNM Class properly changes the start date based on the
        dates in the database. It assures that there are enough previous information and dates to the start date based
        on the windows size. Last test will use a date that doesn't need any changes and no window size is greater than
        zero.
        """
        start_date = pd.to_datetime('2020-02-10', format='%Y-%m-%d')
        end_date = pd.to_datetime('2020-02-28', format='%Y-%m-%d')

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE, start_date=start_date, end_date=end_date,
                         window_size=7, cumulative_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual(ew.start_date, start_date)
        self.assertEqual(ew.end_date, end_date)
        self.assertEqual(ew.adjacencies.shape[0], ew.networks.shape[0] + 1)

    def test_mst_dnm_1(self):
        """
        Tests that the method mst_dnm() from the EWarningDNM returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and windows size is set to zero. Rest of parameters are being changed to assure
        its correctness.
        """
        mst_dnm_s = [0., 0., 0., 0.051305183464283, 0.053836049047137, 0.029628194681969, 0.001414068969903,
                     0.069203167857410, 0.030282027403214, 0.000859827077797, 0.010225545191142, 0.010092508014319,
                     0.000486070437841, 0.000425388885128, 0.000416036617870, 0.010130064916214, 0.001087398192265,
                     0.000293022974542, 0.000192865489876, 0.000153737215378, 0.000131426842813, 0.000113102434099,
                     0.000147967728692, 0.000127479995431, 0.000081769446742, 0.000100149137835, 0.000065542585953,
                     0.000052277103250, 0.010397299490541, 0.013620638913456, 0.014977828959442, 0.002274412369611,
                     0.024562614365076, 0.038631069651736, 0.050467273266932, 0.036249302444126, 0.042147201938300,
                     0.271143299140212]

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                         window_size=0, correlation='pearson', cumulative_data=False,
                         progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.mst_dnm()], [round(x, 10) for x in mst_dnm_s])

    def test_mst_dnm_2(self):
        """
        Tests that the method mst_dnm() from the EWarningDNM returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and windows size is set to a positive value. Rest of parameters are being changed to
        assure its correctness.
        """
        mst_dnm_s = [0.036629472661698, 0., 0.136291419271724, 0.175107261563823, 0.008576622364481, 0.764310386897689,
                     0.049981768723534, 0.048432272385102, 0., 0.035618176586182, 0.209745549505199, 0.031102013112007,
                     0.129821490911880, 0.024595787993302, 0., 0.011102003341613, 0.046882693498788, 0.350799388077512,
                     0.032942804719868, 0.006274407870875, 0.067048040470598]

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-01-24', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-02-18', format='%Y-%m-%d'),
                         window_size=7, correlation='kendall', cumulative_data=False,
                         progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.mst_dnm()], [round(x, 10) for x in mst_dnm_s])

    def test_mst_dnm_3(self):
        """
        Tests that the method mst_dnm() from the EWarningDNM returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and windows size is set to zero. Rest of parameters are being changed to assure
        its correctness.
        """
        mst_dnm_s = [0., 0.153774955135062, 0.379532124885078, 0.008428922769769, 0.040547222728089, 0.070290910922934,
                     0.008143919242808, 0.002445317399995, 0.000655069998889, 0.007135320641056, 0.006400457953679,
                     0.002511946818106, 0.001554370116456, 0.000733201602980, 0.000738918745483, 0.000767574447598,
                     0.000287889268119, 0.000149248225176, 0.000278348828202, 0.000134841951530, 0.000083471653601,
                     0.000063399677175, 0.000134980439481, 0.000052352448395, 0.002983244055570, 0.003639433052130,
                     0.030732732962041]

        countries = ['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'UA']

        static_adjacency = np.ones(shape=(len(countries), len(countries)))
        np.fill_diagonal(static_adjacency, 0)

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-01-30', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-02-25', format='%Y-%m-%d'),
                         countries=countries, window_size=0, correlation='pearson', cumulative_data=True,
                         static_adjacency=static_adjacency, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.mst_dnm()], [round(x, 10) for x in mst_dnm_s])

    def test_mst_dnm_4(self):
        """
        Tests that the method mst_dnm() from the EWarningDNM returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and windows size is set to a positive value. Rest of parameters are being changed to
        assure its correctness.
        """
        mst_dnm_s = [0.006580081736144, 0.001718433421641, 0.001472432157785, 0.003165327436239, 0.003107189705678,
                     0.004043997344271, 0.030147817551941, 0.003400256152265, 0.010475631462965, 0.040208207718277,
                     0.002858962304332, 0.008272405115297, 0.047796689690430, 0.008663079668945, 0.022656624888362,
                     0.118461848870236, 0.233235197800657, 0.169985795013954, 0.313405560982561, 0.140364957911194,
                     2.077691521834900, 1.972065313330318, 1.713097773371550, 1.335124042345958, 1.049010009371482,
                     1.517853665451508]

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-02-01', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                         window_size=14, correlation='spearman', cumulative_data=True,
                         progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.mst_dnm()], [round(x, 10) for x in mst_dnm_s])

    def test_sp_dnm_1(self):
        """
        Tests that the method sp_dnm() from the EWarningDNM returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and windows size is set to zero. Rest of parameters are being changed to assure
        its correctness.
        """
        sp_dnm_s = [
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0.017238722118873, 0.023759499584101, 0.023655781787522, 0.017583634216697,
             0.522768936212701],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.007032618126159],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ]

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                         window_size=0, correlation='spearman', cumulative_data=False,
                         progress_bar=False)
        ew.check_windows()

        self.assertEqual([[round(i, 10) for i in path] for path in ew.sp_dnm()],
                         [[round(i, 10) for i in path] for path in sp_dnm_s])

    def test_sp_dnm_2(self):
        """
        Tests that the method sp_dnm() from the EWarningDNM returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and windows size is set to a positive value. Rest of parameters are being changed to
        assure its correctness.
        """
        sp_dnm_s = [
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0.090949860124370, 0.675773259978965, 0.117675207896619],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0.]
        ]

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-01-24', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-02-28', format='%Y-%m-%d'),
                         window_size=7, correlation='kendall', cumulative_data=False,
                         progress_bar=False)
        ew.check_windows()

        self.assertEqual([[round(i, 10) for i in path] for path in ew.sp_dnm()],
                         [[round(i, 10) for i in path] for path in sp_dnm_s])

    def test_sp_dnm_3(self):
        """
        Tests that the method sp_dnm() from the EWarningDNM returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and windows size is set to zero. Rest of parameters are being changed to assure
        its correctness.
        """
        sp_dnm_s = [
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.042702121524257, 0.003846034546913, 0.001119337823319,
             0.000402880702502, 0.000369036770413, 0.002188239661469, 0.001029565834577, 0.000339814472014,
             0.000115711839843, 0.000045142157285, 0.000025450370381, 0.000018786386760, 0.000016652459141,
             0.000016987798976, 0.000009460436224, 0.000002734270503, 0.000001471795560, 0.000004063821179,
             0.000005610189028, 0.000006474217550, 0.000001800368954, 0.018936014972885, 0.003128235937432,
             0.022229110032786, 0.066580263845915, 0.043595819100425, 0.089345538447386],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ]

        countries = ['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'UA']

        static_adjacency = np.ones(shape=(len(countries), len(countries)))
        np.fill_diagonal(static_adjacency, 0)

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-01-25', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                         countries=countries, window_size=0, correlation='pearson', cumulative_data=True,
                         static_adjacency=static_adjacency, progress_bar=False)
        ew.check_windows()

        self.assertEqual([[round(i, 10) for i in path] for path in ew.sp_dnm([('ES', 'BE'), ('GB', 'AL')])],
                         [[round(i, 10) for i in path] for path in sp_dnm_s])

    def test_sp_dnm_4(self):
        """
        Tests that the method sp_dnm() from the EWarningDNM returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and windows size is set to a positive value. Rest of parameters are being changed to
        assure its correctness.
        """
        sp_dnm_s = [
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.617001443359845,
             0.523786167497991, 0.299032365160358, 0.043992817973976, 0.072603395363769],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
             0.168380192825989],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ]

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                         window_size=14, correlation='pearson', cumulative_data=True,
                         progress_bar=False)
        ew.check_windows()

        self.assertEqual([[round(i, 10) for i in path] for path in ew.sp_dnm()],
                         [[round(i, 10) for i in path] for path in sp_dnm_s])

    def test_sp_dnm_5(self):
        """
        Tests that the method sp_dnm() from the EWarningDNM properly throws Exception when the input is wrong.
        This test will check that the paths passed as argument are real ISO-3166-Alpha2 references.
        """
        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                         window_size=14, correlation='pearson', cumulative_data=True,
                         progress_bar=False)
        ew.check_windows()

        with self.assertRaises(CountryUndefinedException) as context:
            ew.sp_dnm([('ES', 'XX')])
        self.assertEqual(str(context.exception), 'Some ISO-3166-Alpha2 references for the paths are '
                                                 'incorrect or not established in the Class.')

    def test_sp_dnm_6(self):
        """
        Tests that the method sp_dnm() from the EWarningDNM properly throws Exception when the input is wrong.
        This test will check that the paths passed as argument are real ISO-3166-Alpha2 and contained in the list of
        countries to be studied.
        """
        countries = ['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'UA']

        static_adjacency = np.ones(shape=(len(countries), len(countries)))
        np.fill_diagonal(static_adjacency, 0)

        ew = EWarningDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                         start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                         end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                         countries=countries, window_size=0, correlation='spearman', cumulative_data=False,
                         static_adjacency=static_adjacency, progress_bar=False)
        ew.check_windows()

        with self.assertRaises(CountryUndefinedException) as context:
            ew.sp_dnm([('ES', 'GB'), ('FR', 'IT')])
        self.assertEqual(str(context.exception), 'Some ISO-3166-Alpha2 references for the paths are '
                                                 'incorrect or not established in the Class.')


if __name__ == '__main__':
    unittest.main()
