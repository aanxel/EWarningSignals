import unittest
import pandas as pd

from earlywarningsignals import COVID_CRIDA_CUMULATIVE
from earlywarningsignals.signals import EWarningLDNM
from earlywarningsignals.signals import general


class MyTestCase(unittest.TestCase):
    """
    Unittest Class used to test the class EWarningLDNM.
    """

    def test_landscape_dnm_1(self):
        """
        Tests that the method landscape_dnm() from the EWarningLDNM returns the correct List with 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and windows size is set to zero. Rest of parameters are being changed to assure
        its correctness.
        """
        landscape_dnm_s = ([[0.000011863036681, 0., 0., 0.000007642530867, 0.000006034859571, 0.000009953238075,
                             0.000000221066513, 0.000102550508965, 0.000088403409273, 0.000001129690545,
                             0.000023687043113, 0.000013444386873, 0.000001570611236, 0.000000682750741,
                             0.00000187341205, 0.000016846456327, 0.000004822556097, 0.000000528549688,
                             0.000000818424048, 0.000000562115942, 0.000000455110661, 0.000000375205925,
                             0.000000472706489, 0.000000245752487, 0.000000265087771, 0.000000193461975,
                             0.000000206419309, 0.000000148622063, 0.000019367030426, 0.00003227135604,
                             0.000066569803422, 0.000016939376628, 0.001318208722786, 0.007778593570201,
                             0.013559930606921, 0.007742223144324, 0.016110445922347, 0.061287925180331]]
                           * len(general.COUNTRIES_DEFAULT))

        ew = EWarningLDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                          start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                          end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                          window_size=0, correlation='kendall', cumulative_data=False,
                          progress_bar=False)
        ew.check_windows()

        self.assertEqual([[round(i, 10) for i in path] for path in ew.landscape_dnm()],
                         [[round(i, 10) for i in path] for path in landscape_dnm_s])

    def test_landscape_dnm_2(self):
        """
        Tests that the method landscape_dnm() from the EWarningLDNM returns the correct List with 10 decimal precision.
        This test checks cumulativeData = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and windows size is set to a positive value. Rest of parameters are being changed to
        assure its correctness.
        """
        landscape_dnm_s = ([[0.000000013161801, 0., 0.000005912536171, 0.000057556783082, 0.000002826621263,
                             0.000000105682256, 0.000001620965031, 0.0000099122146, 0.000000063806636,
                             0.000003496999266, 0.000038837435061]]
                           * len(general.COUNTRIES_DEFAULT))

        ew = EWarningLDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                          start_date=pd.to_datetime('2020-01-28', format='%Y-%m-%d'),
                          end_date=pd.to_datetime('2020-02-15', format='%Y-%m-%d'),
                          window_size=14, correlation='pearson', cumulative_data=False,
                          progress_bar=False)
        ew.check_windows()

        self.assertEqual([[round(i, 10) for i in path] for path in ew.landscape_dnm()],
                         [[round(i, 10) for i in path] for path in landscape_dnm_s])

    def test_landscape_dnm_3(self):
        """
        Tests that the method landscape_dnm() from the EWarningLDNM returns the correct List with 10 decimal precision.
        This test checks cumulativeData = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and windows size is set to zero. Rest of parameters are being changed to assure
        its correctness.
        """
        landscape_dnm_s = ([[0.000266918325322, 0.000609081192421, 0.000018601351863, 0.000029240395202,
                             0.000076492312918, 0.000010137063512, 0.000003121619645, 0.000003473536302,
                             0.000031572406594, 0.00001717419956, 0.000006368192246, 0.000004576305073,
                             0.000002625915345, 0.000001641195599, 0.000001139032885, 0.000001013287448,
                             0.000000611780599, 0.000000629225697, 0.000000446130553, 0.000000464875362,
                             0.000000613468751, 0.000010278246686]]
                           * len(general.COUNTRIES_DEFAULT))

        ew = EWarningLDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                          start_date=pd.to_datetime('2020-01-31', format='%Y-%m-%d'),
                          end_date=pd.to_datetime('2020-02-21', format='%Y-%m-%d'),
                          window_size=0, correlation='spearman', cumulative_data=True,
                          progress_bar=False)
        ew.check_windows()

        self.assertEqual([[round(i, 10) for i in path] for path in ew.landscape_dnm()],
                         [[round(i, 10) for i in path] for path in landscape_dnm_s])

    def test_landscape_dnm_4(self):
        """
        Tests that the method landscape_dnm() from the EWarningLDNM returns the correct List with 10 decimal precision.
        This test checks cumulativeData = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and windows size is set to a positive value. Rest of parameters are being changed to
        assure its correctness.
        """
        countries = ['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'UA']
        landscape_dnm_s = [[0.004480301562216, 0.011752350148134, 0.000860539608296, 0.001583483616029,
                            0.037615215671743, 0.011256133196882, 0.019398643416164, 0.005384043864648,
                            0.000029331272521, 0.000064827691768, 0.00071055911702, 0.02426952489304,
                            0.006655255433282, 0.000082002717571, 0.000119059685632, 0.000048550162268,
                            0.000054577086739, 0.000714100504822] for _ in range(len(countries))]

        ew = EWarningLDNM(covid_file=COVID_CRIDA_CUMULATIVE,
                          start_date=pd.to_datetime('2020-02-03', format='%Y-%m-%d'),
                          end_date=pd.to_datetime('2020-02-20', format='%Y-%m-%d'),
                          countries=countries,
                          window_size=7, correlation='pearson', cumulative_data=True,
                          progress_bar=False)
        ew.check_windows()

        self.assertEqual([[round(i, 10) for i in path] for path in ew.landscape_dnm()],
                         [[round(i, 10) for i in path] for path in landscape_dnm_s])


if __name__ == '__main__':
    unittest.main()
