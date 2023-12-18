#######################################################################################################################
# VARIABLES FILE
# This file contains most of the variables used in the other modules. Make
#   sure to update the paths below with tha path to the repository in your
#   machine.

_path_to_repo = '/share/garg/gs665/networks_underreporting_public/'

#######################################################################################################################
# PATHS:

#Paths to data directories:
_raw_census_data_dir = f'{_path_to_repo}d01_data/d01_raw/d03_population-data/2020/'
_processed_covariates_dir = f'{_path_to_repo}d01_data/d02_processed/d02_demographics/'

#Paths to shapefiles:
_nyc_shp = f'{_path_to_repo}d01_data/d01_raw/d02_spatial-data/NYC-boroughs/NYC-boroughs.shp'
_park_shp = f'{_path_to_repo}d01_data/d01_raw/d02_spatial-data/NYC-parks/NYC-parks.shp'

#Paths to 311 data:
_path_311_floods = f'{_path_to_repo}d01_data/d01_raw/d01_311-data/street-flooding-2023_raw.csv'
_path_311_rodents = f'{_path_to_repo}d01_data/d01_raw/d01_311-data/rodent_raw.csv'

#Paths to output directories:
_path_output_raw = f'{_path_to_repo}d05_raw-outputs/'
_path_output_processed = f'{_path_to_repo}d06_processed-outputs/'

#######################################################################################################################
#DEMOGRAPHIC DATA:

#Covariates that are used in the multivariate heterogeneous reporting model:
_covariates = ['log_population',
               'income_median',
               'education_bachelors_pct',
               'race_white_nh_pct',
               'age_median',
               'households_renteroccupied_pct']

#Other covariates, considered for the univariate coefficient analysis:
_covariates_unused = ['population',
                      'population_density',
                      'population_density_land',
                      'area_m2',
                      'income_poverty_pct',
                      'education_nohighschool_pct',
                      'race_hispanic_pct',
                      'race_black_nh_pct',
                      'race_asian_nh_pct',
                      'race_other_nh_pct',
                      'households_owneroccupied_pct']

#Categorical covariates used for equity analysis or bar plots:
_demographic_categories = ['race_majority',
                           'income_qt']

#Human readable names of all possible covariates:
_covariates_names = {'population':'Population',
                     'log_population':'log(Population)',
                     
                     'population_density':'Population density',
                     'population_density_land':'Population density (land area)',
                     'area_m2':'Land area',
                     
                     'age_median':'Median age',
                     
                     'income_median': 'Median income',
                     'income_poverty_pct': 'Population living below the poverty level (%)',
                     'income_poverty': 'Population living below the poverty level',
                     
                     'education_nohighschool_pct': 'Population without a highschool degree (%)',
                     'education_nohighschool': 'Population without a highschool degree',
                     'education_bachelors_pct': "Population with a bachelor\'s degree (%)",
                     'education_bachelors': "Population with a bachelor\'s degree",
                     
                     'race_white_nh_pct': 'White population (%)',
                     'race_white_nh': 'White population',
                     'race_hispanic_pct': 'Hispanic population (%)',
                     'race_hispanic': 'Hispanic population',
                     'race_black_nh_pct': 'Black population (%)',
                     'race_black_nh': 'Black population',
                     'race_asian_nh_pct': 'Asian population (%)',
                     'race_asian_nh': 'Asian population',
                     'race_other_nh_pct': 'Population from another race (%)',
                     'race_other_nh': 'Population from another race',
                     
                     'households_owneroccupied_pct':'Households occupied by owner (%)',
                     'households_owneroccupied':'Households occupied by owner',
                     'households_renteroccupied_pct': 'Households occupied by a renter (%)',
                     'households_renteroccupied': 'Households occupied by a renter'}

_covariates_for_analysis = ['population',
                            'income_poverty_pct',
                            'race_white_nh_pct',
                            'race_black_nh_pct',
                            'race_hispanic_pct',
                            'race_asian_nh_pct',
                            'age_median',
                            'education_nohighschool_pct',
                            'households_renteroccupied_pct']

_covariates_for_analysis_renaming = {'population':'population',
                                     'income_poverty_pct':'population_poverty',
                                     'race_white_nh_pct':'population_white',
                                     'race_black_nh_pct':'population_black',
                                     'race_hispanic_pct':'population_hispanic',
                                     'race_asian_nh_pct':'population_asian',
                                     'age_median': 'population_age',
                                     'education_nohighschool_pct':'population_nohighschool',
                                     'households_renteroccupied_pct':'population_renter'}

_param_description = {'theta0':'Event incidence',
                      'theta1':'Spatial correlation',
                      'alpha0':'Intercept',
                      'alpha1':'log(Population)',
                      'alpha2':'Median income',
                      'alpha3':"Bachelor's degree population",
                      'alpha4':'White population',
                      'alpha5':'Median age',
                      'alpha6':'Households occupied by renter'}

#######################################################################################################################
#SPATIAL DATA:
_projected_crs = 'EPSG:2263'

#######################################################################################################################
#FLOOD EVENTS:
import datetime as dt
_floods_start_date = {'ophelia': dt.datetime.strptime('2023-09-29 06:00:00 AM', '%Y-%m-%d %I:%M:%S %p'),
                      'ida': dt.datetime.strptime('2021-09-01 07:00:00 PM', '%Y-%m-%d %I:%M:%S %p'),
                      'henri': dt.datetime.strptime('2021-08-21 08:00:00 PM', '%Y-%m-%d %I:%M:%S %p'),
                      'elsa': dt.datetime.strptime('2021-07-08', '%Y-%m-%d'),
                      'isaias': dt.datetime.strptime('2020-08-04', '%Y-%m-%d'),
                      'sandy': dt.datetime.strptime('2012-10-28', '%Y-%m-%d'),
                      'irene': dt.datetime.strptime('2011-08-27', '%Y-%m-%d')}

_floods_start_date_notime = {'ophelia': dt.datetime.strptime('2023-09-29', '%Y-%m-%d'),
                             'ida': dt.datetime.strptime('2021-09-01', '%Y-%m-%d'),
                             'henri': dt.datetime.strptime('2021-08-21', '%Y-%m-%d'),
                             'elsa': dt.datetime.strptime('2021-07-08', '%Y-%m-%d'),
                             'isaias': dt.datetime.strptime('2020-08-04', '%Y-%m-%d'),
                             'sandy': dt.datetime.strptime('2012-10-28', '%Y-%m-%d'),
                             'irene': dt.datetime.strptime('2011-08-27', '%Y-%m-%d')}

#######################################################################################################################
#RODENT EVENTS:
_rodents_start_date = dt.datetime.strptime('2022-09-01', '%Y-%m-%d')

#######################################################################################################################
#LATEX:
_text_width= 505.89
_column_width=239.39438