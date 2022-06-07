import unittest
import pandas as pd
import numpy as np

from earlywarningsignals import COVID_CRIDA_CUMULATIVE, COUNTRY_INFO_CRIDA

from earlywarningsignals.signals import EWarningSpecific


class MyTestCase(unittest.TestCase):
    """
    Unittest Class used to test the class EWarningSpecific.
    """

    def test_density_1(self):
        """
        Tests that the method density() from the EWarningSpecific returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        densities = [0.002898550724638, 0.002898550724638, 0.002898550724638, 0.002898550724638, 0.002898550724638,
                     0.001932367149758, 0.001932367149758, 0.001932367149758, 0.001932367149758, 0.001932367149758,
                     0.001932367149758, 0., 0., 0.000966183574879, 0.000966183574879, 0.000966183574879,
                     0.000966183574879, 0.000966183574879, 0., 0.000966183574879, 0.000966183574879, 0.02512077294686,
                     0.065700483091787, 0.085990338164251, 0.085024154589372, 0.114009661835749, 0.150724637681159]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-02-01', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              window_size=14, correlation='pearson', threshold=0.5,
                              cumulative_data=False, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.density()], [round(x, 10) for x in densities])

    def test_density_2(self):
        """
        Tests that the method density() from the EWarningSpecific returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        densities = [0.001932367149758, 0.001932367149758, 0.001932367149758, 0.001932367149758, 0.001932367149758,
                     0.001932367149758, 0.003864734299517, 0.003864734299517, 0.003864734299517, 0.002898550724638, 0.,
                     0., 0.000966183574879, 0.000966183574879, 0.000966183574879, 0.000966183574879, 0.000966183574879,
                     0., 0., 0., 0.014492753623188, 0.052173913043478, 0.057971014492754, 0.055072463768116,
                     0.060869565217391, 0.077294685990338]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-02-05', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              window_size=14, correlation='pearson', threshold=0.6,
                              cumulative_data=False, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.density()], [round(x, 10) for x in densities])

    def test_density_3(self):
        """
        Tests that the method density() from the EWarningSpecific returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        densities = [0.002898550724638, 0.009661835748792, 0.018357487922705, 0.018357487922705, 0.018357487922705,
                     0.016425120772947, 0.016425120772947, 0.011594202898551, 0.004830917874396, 0.007729468599034,
                     0.010628019323671, 0.009661835748792, 0.009661835748792, 0.009661835748792, 0.005797101449275,
                     0.002898550724638, 0.001932367149758, 0.001932367149758, 0.000966183574879, 0.000966183574879,
                     0.000966183574879, 0.000966183574879, 0.000966183574879, 0.000966183574879, 0.000966183574879,
                     0.000966183574879, 0.027053140096618]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-30', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-25', format='%Y-%m-%d'),
                              window_size=7, correlation='spearman', threshold=0.4,
                              cumulative_data=True, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.density()], [round(x, 10) for x in densities])

    def test_density_4(self):
        """
        Tests that the method density() from the EWarningSpecific returns the correct List with a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        densities = [0.166666666666667, 0.166666666666667, 0.166666666666667, 0.194444444444444, 0.194444444444444,
                     0.194444444444444, 0.166666666666667, 0.166666666666667, 0.111111111111111, 0.083333333333333,
                     0.083333333333333, 0.083333333333333, 0.083333333333333, 0.083333333333333, 0.027777777777778,
                     0.027777777777778, 0.027777777777778, 0.027777777777778, 0.027777777777778, 0., 0.,
                     0.027777777777778, 0.111111111111111, 0.166666666666667, 0.25, 0.277777777777778,
                     0.277777777777778]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              countries=['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'TR', 'UA'],
                              window_size=14, correlation='kendall', threshold=0.7,
                              cumulative_data=True, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.density()], [round(x, 10) for x in densities])

    def test_clustering_coefficient_1(self):
        """
        Tests that the method clustering_coefficient() from EWarningSpecific returns the correct List with
        a precision of 10 decimal.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        clustering_coefficients = [0.032608695652174, 0.032608695652174, 0.032608695652174, 0.032608695652174,
                                   0.032608695652174, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                   0.081780538302277, 0.128623188405797, 0.188255153840192, 0.192595598845599,
                                   0.208019999324347, 0.223542912401608]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              window_size=14, correlation='pearson', threshold=0.5,
                              cumulative_data=False, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.clustering_coefficient()],
                         [round(x, 10) for x in clustering_coefficients])

    def test_clustering_coefficient_2(self):
        """
        Tests that the method clustering_coefficient() from EWarningSpecific returns the correct List with
        a precision of 10 decimal.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        clustering_coefficients = [0., 0., 0., 0., 0., 0., 0., 0.02536231884058, 0.036231884057971, 0.036231884057971,
                                   0.036231884057971, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.065217391304348]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-30', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-25', format='%Y-%m-%d'),
                              window_size=14, correlation='spearman', threshold=0.4,
                              cumulative_data=False, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.clustering_coefficient()],
                         [round(x, 10) for x in clustering_coefficients])

    def test_clustering_coefficient_3(self):
        """
        Tests that the method clustering_coefficient() from EWarningSpecific returns the correct List with
        a precision of 10 decimal.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        clustering_coefficients = [0., 0., 0., 0., 0., 0., 0.053260869565217, 0.058695652173913, 0.03804347826087,
                                   0.03804347826087, 0., 0., 0., 0., 0.032608695652174, 0.032608695652174, 0., 0., 0.,
                                   0., 0.]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-15', format='%Y-%m-%d'),
                              window_size=5, correlation='pearson', threshold=0.7,
                              cumulative_data=True, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.clustering_coefficient()],
                         [round(x, 10) for x in clustering_coefficients])

    def test_clustering_coefficient_4(self):
        """
        Tests that the method clustering_coefficient() from EWarningSpecific returns the correct List with
        a precision of 10 decimal.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        clustering_coefficients = [0.277777777777778, 0.277777777777778, 0.277777777777778, 0.277777777777778,
                                   0.277777777777778, 0.277777777777778, 0.277777777777778, 0.277777777777778,
                                   0.277777777777778, 0.277777777777778, 0.222222222222222, 0.222222222222222,
                                   0.222222222222222, 0.166666666666667, 0.166666666666667, 0.166666666666667,
                                   0.166666666666667, 0., 0., 0., 0.222222222222222, 0.277777777777778,
                                   0.277777777777778]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-25', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-27', format='%Y-%m-%d'),
                              countries=['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'TR', 'UA'],
                              window_size=15, correlation='kendall', threshold=0.3,
                              cumulative_data=True, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.clustering_coefficient()],
                         [round(x, 10) for x in clustering_coefficients])

    def test_assortativity_coefficient_1(self):
        """
        Tests that the method assortativity_coefficient() from EWarningSpecific returns the correct List with
        a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        assortativity_coefficients = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), -1., -1.,
                                      -1., -1., -1., -1., float('nan'), float('nan'), float('nan'), float('nan'),
                                      float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                      float('nan'), -0.322033898305085, 0.573667711598743, 0.045115009746591,
                                      0.057075421092159, 0.26727208309247, 0.389198678232136]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-25', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              window_size=14, correlation='pearson', threshold=0.5,
                              cumulative_data=False, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertTrue(np.allclose(ew.assortativity_coefficient(), assortativity_coefficients, rtol=0, atol=1e-10,
                                    equal_nan=True))

    def test_assortativity_coefficient_2(self):
        """
        Tests that the method assortativity_coefficient() from EWarningSpecific returns the correct List with
        a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        assortativity_coefficients = [0.999999999999998, 0.999999999999998, float('nan'), float('nan'), float('nan'),
                                      -0.5, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                      float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                      float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                      -0.253501400560229, -0.0756034310792, 0.021009162472578]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-02-05', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-28', format='%Y-%m-%d'),
                              window_size=7, correlation='kendall', threshold=0.4,
                              cumulative_data=False, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertTrue(np.allclose(ew.assortativity_coefficient(), assortativity_coefficients, rtol=0, atol=1e-10,
                                    equal_nan=True))

    def test_assortativity_coefficient_3(self):
        """
        Tests that the method assortativity_coefficient() from EWarningSpecific returns the correct List with
        a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        assortativity_coefficients = [1., -0.388888888888892, -0.373493975903611, -0.373493975903611,
                                      -0.373493975903611, -0.393939393939395, -0.382113821138214, -0.382113821138214,
                                      -0.545961002785514, -0.377880184331794, -0.454545454545452, -0.447368421052636,
                                      float('nan'), float('nan'), float('nan'), float('nan'), -0.714285714285714,
                                      -0.499999999999999, -1., float('nan'), float('nan'), float('nan'), -1.,
                                      float('nan'), float('nan'), 1., -0.193877551020413, -0.21274856987196,
                                      -0.126202407695415, 0.063085571517803, 0.173472881500324]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              window_size=10, correlation='spearman', threshold=0.6,
                              cumulative_data=True, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertTrue(np.allclose(ew.assortativity_coefficient(), assortativity_coefficients, rtol=0, atol=1e-10,
                                    equal_nan=True))

    def test_assortativity_coefficient_4(self):
        """
        Tests that the method assortativity_coefficient() from EWarningSpecific returns the correct List with
        a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        assortativity_coefficients = [float('nan'), float('nan'), float('nan'), -0.666666666666679, -0.714285714285714,
                                      -0.714285714285714, -0.714285714285714, -0.499999999999999, -0.499999999999999,
                                      -0.5, -1., -1., float('nan'), float('nan'), float('nan'), -1., float('nan'),
                                      float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                      -0.666666666666679, -0.548387096774194, float('nan'), float('nan'), float('nan')]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-30', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              countries=['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'TR', 'UA'],
                              window_size=14, correlation='pearson', threshold=0.8,
                              cumulative_data=True, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertTrue(np.allclose(ew.assortativity_coefficient(), assortativity_coefficients, rtol=0, atol=1e-10,
                                    equal_nan=True))

    def test_number_edges_1(self):
        """
        Tests that the method number_edges() from EWarningSpecific returns the correct List with 8 Byte precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        number_edges = [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 26, 68, 89, 88, 118, 156]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-02-15', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              window_size=14, correlation='pearson', threshold=0.5,
                              cumulative_data=False, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.number_edges()], [round(x, 10) for x in number_edges])

    def test_number_edges_2(self):
        """
        Tests that the method number_edges() from EWarningSpecific returns the correct List with 8 Byte precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        number_edges = [2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 3, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 21, 68, 98, 95, 117, 148]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-02-01', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              window_size=14, correlation='pearson', threshold=0.4,
                              cumulative_data=False, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.number_edges()], [round(x, 10) for x in number_edges])

    def test_number_edges_3(self):
        """
        Tests that the method number_edges() from EWarningSpecific returns the correct List with 8 Byte precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        number_edges = [4, 13, 19, 19, 18, 18, 17, 17, 13, 12, 12, 10, 10, 10, 9, 6, 4, 3, 2, 1, 1, 1, 1, 1, 1, 16]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-25', format='%Y-%m-%d'),
                              window_size=10, correlation='kendall', threshold=0.6,
                              cumulative_data=True, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.number_edges()], [round(x, 10) for x in number_edges])

    def test_number_edges_4(self):
        """
        Tests that the method number_edges() from EWarningSpecific returns the correct List with 8 Byte precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        number_edges = [7, 7, 7, 7, 6, 5, 4, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 3, 4]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-30', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-28', format='%Y-%m-%d'),
                              countries=['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'TR', 'UA'],
                              window_size=20, correlation='spearman', threshold=0.8,
                              cumulative_data=True, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.number_edges()], [round(x, 10) for x in number_edges])

    def test_prs_1(self):
        """
        Tests that the method prs() from EWarningSpecific returns the correct List with 8 Byte precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        prs_s = [10471290240433882, 10471290240433882, 10471290240433882, 18616740653558996, 10471290126725780,
                 2636691134803056, 2636691134803056, 10471289597835078, 19333601809043054, 19333601678496052,
                 19333601547949050, 8862311814342000, 20237821076579988, 8862311683795000, 8862311683795000,
                 8862311553248000, 8862311292154000, 8862311161607000, 0, 8208999364395696, 8208989075424852,
                 88899106287021166, 110980913577286700, 127885207163862106, 121923308115984516]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-24', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-28', format='%Y-%m-%d'),
                              window_size=14, correlation='pearson', threshold=0.4,
                              cumulative_data=False, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.prs(COUNTRY_INFO_CRIDA)], [round(x, 10) for x in prs_s])

    def test_prs_2(self):
        """
        Tests that the method prs() from EWarningSpecific returns the correct List with 8 Byte precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        prs_s = [9153401522428020, 9153401522428020, 9153400796886108, 9876724845573712, 8366504719054328,
                 8366504719054328, 8366504719054328, 0, 0, 18993771220170130, 0, 0, 0, 0, 11375509262237988, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 33701141012817926, 51591315325569034, 49911089957789978, 27742594003219352,
                 24164383863270450, 46911650488170382]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-02-01', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              window_size=7, correlation='kendall', threshold=0.6,
                              cumulative_data=False, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.prs(COUNTRY_INFO_CRIDA)], [round(x, 10) for x in prs_s])

    def test_prs_3(self):
        """
        Tests that the method prs() from EWarningSpecific returns the correct List with 8 Byte precision.
        This test checks cumulative_data = False which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        prs_s = [93600840705710854, 93600836418678050, 93600836418678050, 98434261144606154, 99947250753086534,
                 101348708207093462, 98367954699347218, 97644629027822710, 97644629027822710, 93125783072662342,
                 88266245710900464, 88266245175173162, 83349194247685660, 83349193729912618, 83349193729912618,
                 67247062259044298, 51461844979552798, 28072418983779370, 20237819884120556, 8862311161607000,
                 17071323757411892, 17071309873267696, 17071299323202852, 77082672184402506, 131908150863053092,
                 153207735620081924]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-27', format='%Y-%m-%d'),
                              window_size=12, correlation='spearman', threshold=0.5,
                              cumulative_data=True, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.prs(COUNTRY_INFO_CRIDA)], [round(x, 10) for x in prs_s])

    def test_prs_4(self):
        """
        Tests that the method prs() from EWarningSpecific returns the correct List with 8 Byte precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        prs_s = [8663565998448308, 8663565884740206, 26521551258666608, 26521551258666608, 26521548941528812,
                 25203119097235420, 25203119097235420, 25203118829801088, 25203118562366756, 23831920606845418,
                 22887542138899446, 21313997026347132, 21313997026347132, 21313996802290570, 21313996802290570,
                 21313996578234008, 21313996130120884, 8862311161607000, 8862311161607000, 8862310508872000]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-31', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-23', format='%Y-%m-%d'),
                              countries=['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'TR', 'UA'],
                              window_size=14, correlation='pearson', threshold=0.7,
                              cumulative_data=True, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.prs(COUNTRY_INFO_CRIDA)], [round(x, 10) for x in prs_s])

    def test_forman_ricci_curvature_1(self):
        """
        Tests that the method forman_ricci_curvature() from EWarningSpecific returns the correct List with
        a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        forman_ricci_curvatures = [float('nan'), float('nan'), float('nan'), 2., 2.75, 1.714285714285714,
                                   1.714285714285714, 1.714285714285714, 3.714285714285714, 3.714285714285714, 4., 2.,
                                   2., 1.333333333333333, 2., 2., 2., 2., 2., float('nan'), float('nan'), float('nan'),
                                   float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), 2., 8.,
                                   8.865853658536585, 7.316666666666666, 4.589743589743589, 2.902255639097744,
                                   4.721893491124260]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-22', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              window_size=7, correlation='spearman', threshold=0.4,
                              cumulative_data=False, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertTrue(np.allclose(ew.forman_ricci_curvature(), forman_ricci_curvatures, rtol=0, atol=1e-10,
                                    equal_nan=True))

    def test_forman_ricci_curvature_2(self):
        """
        Tests that the method forman_ricci_curvature() from EWarningSpecific returns the correct List with
        a 10 decimal precision.
        This test checks cumulative_data = False which means that the data will be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        forman_ricci_curvatures = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., float('nan'), float('nan'), 2., 2., 2.,
                                   2., 2., float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
                                   float('nan'), float('nan'), 6.]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-27', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-25', format='%Y-%m-%d'),
                              window_size=10, correlation='pearson', threshold=0.8,
                              cumulative_data=False, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertTrue(np.allclose(ew.forman_ricci_curvature(), forman_ricci_curvatures, rtol=0, atol=1e-10,
                                    equal_nan=True))

    def test_forman_ricci_curvature_3(self):
        """
        Tests that the method forman_ricci_curvature() from EWarningSpecific returns the correct List with
        a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = False which won't apply the square root to the original data to
        smooth it. Rest of parameters are being changed to assure its correctness.
        """
        forman_ricci_curvatures = [7., 6.227272727272728, 5.75, 5.96, 5.96, 6.227272727272728, 7., 6., 3.8125,
                                   4.076923076923077, 3.75, 5., 5., 3.666666666666667, 2.875, 4., 1.75,
                                   0.666666666666667, 2., 1., 1., 4.333333333333333, 9.316455696202532,
                                   12.127659574468085, 14.283333333333333]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-02-01', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-02-28', format='%Y-%m-%d'),
                              window_size=14, correlation='kendall', threshold=0.6,
                              cumulative_data=True, square_root_data=False, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.forman_ricci_curvature()],
                         [round(x, 10) for x in forman_ricci_curvatures])

    def test_forman_ricci_curvature_4(self):
        """
        Tests that the method forman_ricci_curvature() from EWarningSpecific returns the correct List with
        a 10 decimal precision.
        This test checks cumulative_data = True which means that the data won't be converted from the cumulative covid
        cases to daily cases, and square_root_data = True which apply the square root to the original data to smooth it.
        Rest of parameters are being changed to assure its correctness.
        """
        forman_ricci_curvatures = [2.857142857142857, 3.666666666666667, 3.666666666666667, 5., 5., 5., 5., 5.,
                                   3.666666666666667, 2.857142857142857, 2.4, 1.75, 3., 3., 3., 3., 3., 2., 2., 2., 2.,
                                   4., 5., 5., 5., 5., 6.]

        ew = EWarningSpecific(covid_file=COVID_CRIDA_CUMULATIVE,
                              start_date=pd.to_datetime('2020-01-31', format='%Y-%m-%d'),
                              end_date=pd.to_datetime('2020-03-01', format='%Y-%m-%d'),
                              countries=['AL', 'BE', 'FR', 'ES', 'SE', 'CH', 'GB', 'TR', 'UA'],
                              window_size=14, correlation='pearson', threshold=0.5,
                              cumulative_data=True, square_root_data=True, progress_bar=False)
        ew.check_windows()

        self.assertEqual([round(x, 10) for x in ew.forman_ricci_curvature()],
                         [round(x, 10) for x in forman_ricci_curvatures])


if __name__ == '__main__':
    unittest.main()
