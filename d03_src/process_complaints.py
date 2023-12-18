import pandas as pd
import geopandas as gpd
import datetime as dt

import sys
sys.path.append('../d03_src/')
import vars

def process_complaints(df,
                       complaints_series,
                       column_name='complaints',
                       binary=True):

    #Match complaints by index:
    df.loc[:,column_name] = complaints_series

    #Geographies with no complaint have zero complaints:
    df.loc[:,column_name] = df[column_name].fillna(0)

    #Binarize:
    if binary: df.loc[:,column_name] = df[column_name].apply(lambda x: 1 if x > 0 else 0)

    #Make into an array to return:
    complaints_array = df[column_name].values
                           
    return complaints_array
                           
def filter_and_aggregate_complaints(aggregation_gdf,
                                    complaints_gdf,
                                    complaints_dates,
                                    start_date,
                                    end_date,
                                    projected_crs=vars._projected_crs,
                                    binary=True):
    """
    Filter 311 complaints by date and aggregates 
       into a geometry geodataframe.
    
    Parameters
    ----------
    aggregation_gdf : Geopandas.GeoDataFrame
        geodataframe with units we aggregate complaints, must be
        indexed 0,... N-1 for 100% correct results!

    complaints_gdf : Geopandas.GeoDataFrame
        geodataframe with points representing complaints

    complaints_dates : pd.Series or np.Array
        datetime objects corresponding to each of the complaints
    
    start_date : datetime.datetime
        date the event starts

    end_date : datetime.datetime
        date the event ends
        
    projected_crs : Geopandas.crs
        cartographic reference system to perform
        aggregation in
        
    binary : Bool
        whether to turn all complaints in T=0/1, to follow
        a PU learning routine
    
    Returns
    ----------
    filtered_complaints : np.Array
    """
    #Filter:
    filtered_complaints = complaints_gdf[(complaints_dates < end_date) & (complaints_dates >= start_date)]
    filtered_complaints_proj = filtered_complaints.to_crs(projected_crs).reset_index(drop=True)

    #Aggregate:
    agg_complaints = filtered_complaints_proj.sjoin(aggregation_gdf.to_crs(projected_crs),
                                                    how='left',
                                                    predicate='intersects',
                                                    rsuffix='aggregated')

    #Group by the geographic unit index:
    complaints_by_unit = agg_complaints.groupby('index_aggregated').size()

    #Process:
    complaints_array = process_complaints(aggregation_gdf.copy(),
                                          complaints_by_unit,
                                          binary=binary)
    
    return complaints_array

def filter_and_aggregate_complaints_percentage(aggregation_gdf,
                                               complaints_gdf,
                                               complaints_dates,
                                               start_date,
                                               percentage,
                                               end_date=None,
                                               projected_crs=vars._projected_crs,
                                               binary=True):
    """
    Filter 311 complaints by percentage (i.e. collect all
       complaints submitted until X% of the geographic units
       have a report to fix P(T)) and aggregates into a
       geometry geodataframe.
    
    Parameters
    ----------
    aggregation_gdf : Geopandas.GeoDataFrame
        geodataframe with units we aggregate complaints, must be
        indexed 0,... N-1 for 100% correct results!

    complaints_gdf : Geopandas.GeoDataFrame
        geodataframe with points representing complaints

    complaints_dates : pd.Series or np.Array
        datetime objects corresponding to each of the complaints
    
    start_date : datetime.datetime
        date the event starts

    percentage : float
        approximate Pr(T)

    end_date : datetime.datetime
        date the event ends (i.e. won't get complaints past this)
        
    projected_crs : Geopandas.crs
        cartographic reference system to perform
        aggregation in
        
    binary : Bool
        whether to turn all complaints in T=0/1, to follow
        a PU learning routine
    
    Returns
    ----------
    filtered_complaints : np.Array

    train_end_date : datetime.Datetime
        date of the first event in the test set
    """
    #Filter:
    filtered_complaints = complaints_gdf[(complaints_dates < end_date) & (complaints_dates >= start_date)]
    filtered_complaints_proj = filtered_complaints.to_crs(projected_crs).reset_index(drop=True)

    #Aggregate:
    agg_complaints = filtered_complaints_proj.sjoin(aggregation_gdf.to_crs(projected_crs),
                                                    how='left',
                                                    predicate='intersects',
                                                    rsuffix='aggregated')

    #Select based off percentage:
    ordered_complaints_idx = agg_complaints['Created Date'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p')).sort_values().index
    ordered_complaints = agg_complaints.loc[ordered_complaints_idx].reset_index(drop=True)
    first_complaint_per_unit = ordered_complaints.drop_duplicates(subset='index_aggregated', keep='first')
    training_complaints = first_complaint_per_unit.iloc[:int(percentage*len(aggregation_gdf))].reset_index(drop=True)
    train_end_date = dt.datetime.strptime(training_complaints.iloc[-1]['Created Date'], '%m/%d/%Y %I:%M:%S %p')
    print('Train end date is: ' + str(train_end_date) +  ' (' +  str(train_end_date-start_date) + ' hours after the start)')
                                                   
    #Re-filter (this will get some edge-case complaints submitted at the same time):
    filtered_complaints = complaints_gdf[(complaints_dates <= train_end_date) & (complaints_dates >= start_date)]
    filtered_complaints_proj = filtered_complaints.to_crs(projected_crs).reset_index(drop=True)

    #Aggregate:
    agg_complaints = filtered_complaints_proj.sjoin(aggregation_gdf.to_crs(projected_crs),
                                                    how='left',
                                                    predicate='intersects',
                                                    rsuffix='aggregated')
    #Group by the geographic unit index:
    complaints_by_unit = agg_complaints.groupby('index_aggregated').size()

    #Process:
    complaints_array = process_complaints(aggregation_gdf.copy(),
                                          complaints_by_unit,
                                          binary=binary)
    
    return complaints_array, train_end_date

def get_complaints(aggregation_gdf,
                   event_date,
                   filter_mode='duration',
                   event_duration=7,
                   train_duration=2,
                   train_percentage=0.1,
                   end_date=None,
                   train_end_date=None,
                   return_test_complaints=False,
                   binary=True,
                   return_filtered_gdf=False,
                   path_311=vars._path_311_floods,
                   projected_crs=vars._projected_crs):
    """
    Collects 311 complaints relative to a certain event per
        geographic unit.

    Filter events according to dates following two possible
        modes: fixed duration or fixed end.
    
    Parameters
    ----------
    aggregation_gdf : Geopandas.GeoDataFrame
        geodataframe with units we aggregate complaints

    event_date : datetime.datetime
        date the event happens, will filter complaints at strictly
        larger dates

    filter_mode : str, `duration` or `end` or `percentage`
        how to filter the complaints temporally
    
    event_duration : int
        number of days to filter complaints for fixed duration
    train_duration : int
        number of days to filter train complaints
    train_percentage : None or float
        if `filter_mode` is `percentage`, crops complaints when
        such amount of units have a report

    end_date : datetime.datetime
        date the event ends
    train_end_date : datetime.datetime
        date the training data should end

    return_test_complaints : Bool
        whether to return the complaints during the
        test period in addition to training complaints
    binary : Bool
        whether to turn all complaints in T=0/1, to follow
        a PU learning routine

    path_311 : str
        filepath for complaints df
    
    Returns
    ----------
    filtered_T_train : np.Array
    filtered_T_full : np.Array
    """
    
    #Collect the dataframe of complaints and georeference:
    complaints_raw_df = pd.read_csv(path_311, low_memory=False)
    complaints_nonan_df = complaints_raw_df[~(complaints_raw_df.Latitude.isna()|complaints_raw_df.Longitude.isna())]
    complaints_gdf = gpd.GeoDataFrame(complaints_nonan_df,
                                      geometry=gpd.points_from_xy(complaints_nonan_df.Longitude, complaints_nonan_df.Latitude),
                                      crs='EPSG:4326')
    
    #Get the time and date of creation:
    complaints_dates = complaints_gdf['Created Date'].apply(lambda x: dt.datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p'))

    #Figure out end and train date:
    assert filter_mode in ['duration', 'end', 'percentage']
    if filter_mode == 'duration': 
        train_end_date = event_date + dt.timedelta(days=train_duration)
        end_date = event_date + dt.timedelta(days=event_duration)
        
    if end_date is None:
        end_date = event_date + dt.timedelta(days=event_duration)
        
    if filter_mode == 'end' and train_end_date is None:
        train_end_date = event_date + dt.timedelta(days=train_duration)
    
    #Filter and aggregate:
    if filter_mode == 'percentage':
        train_complaints, train_end_date = filter_and_aggregate_complaints_percentage(aggregation_gdf,
                                                                                      complaints_gdf,
                                                                                      complaints_dates=complaints_dates,
                                                                                      start_date=event_date,
                                                                                      percentage=train_percentage,
                                                                                      end_date=end_date,
                                                                                      projected_crs=projected_crs,
                                                                                      binary=binary)
    else:
        train_complaints = filter_and_aggregate_complaints(aggregation_gdf,
                                                           complaints_gdf,
                                                           complaints_dates=complaints_dates,
                                                           start_date=event_date,
                                                           end_date=train_end_date,
                                                           projected_crs=projected_crs,
                                                           binary=binary)
    
    #Get the test complaints if wanted:
    if return_test_complaints:
        test_complaints = filter_and_aggregate_complaints(aggregation_gdf,
                                                          complaints_gdf,
                                                          complaints_dates=complaints_dates,
                                                          start_date=train_end_date,
                                                          end_date=end_date,
                                                          projected_crs=projected_crs,
                                                          binary=binary)
    #Collect the outputs:
    outputs = [train_complaints]
    if return_test_complaints: outputs.append(test_complaints)
    if return_filtered_gdf: outputs.append(aggregation_gdf)

    #Return:
    return train_complaints if len(outputs)==1 else tuple(outputs)