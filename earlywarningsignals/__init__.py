import os

# Absolute path of folder containing this particular file
ACTUAL_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute location of the data folder based on the actual directory
DATA_DIR = os.path.join(ACTUAL_DIR, 'data')

# Time series of covid cases form different sources
COVID_WHO_NEW = os.path.join(DATA_DIR, 'who_covid_new.csv')
COVID_WHO_CUMULATIVE = os.path.join(DATA_DIR, 'who_covid_cumulative.csv')

COVID_JHU_CUMULATIVE = os.path.join(DATA_DIR, 'jhu_covid_cumulative.csv')

COVID_CRIDA_CUMULATIVE = os.path.join(DATA_DIR, 'crida_covid_cumulative.csv')

# Country information form different sources
COUNTRY_INFO = os.path.join(DATA_DIR, 'countries_info.csv')
COUNTRY_INFO_CRIDA = os.path.join(DATA_DIR, 'countries_info_crida.csv')
