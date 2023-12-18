import numpy as np
import pandas as pd
import geopandas as gpd

import sys
sys.path.append('../d03_src/')
import vars
    
##########################################################################################################

def get_covariates(table='GEOID_covariates',
                   processed_covariates_dir=vars._processed_covariates_dir,
                   set_index=True,
                   index_col=None):
    """
    Loads the table of demographic features per geographic unit

    Parameters
    ----------
    table : str
        file name (without csv extesion)
    processed_covariates_dr : str
        file directory
    set_index : Bool
        set index to the geoid or geohash
    index_col : str
        column to turn into index (if None,
        inferred from table name)
    
    Returns
    ----------
    pd.DataFrame
    """
                       
    #Get the dataframe:
    df = pd.read_csv(f'{processed_covariates_dir}{table}.csv')

    #Adjust the GEOID to strings:
    if 'GEOID' in df.columns:
        df.loc[:,'GEOID'] = df['GEOID'].astype(str)

    #Adjust the index:
    if set_index:
        if index_col is None:
            index_col = 'GEOID' if 'GEOID' in table else 'geohash'
        assert index_col in df.columns
        df = df.set_index(index_col)
    
    return df   
                       
##########################################################################################################
        
def include_covariates(gdf,
                       col_to_merge_on='GEOID',
                       table_name=None,
                       processed_covariates_dir=vars._processed_covariates_dir,
                       covariates_df=None, covariates_names=vars._covariates,
                       standardize=True):
    
    """
    Include demographic features in a geodataframe of
       census or geohash geometries

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        geographic units
    col_to_merge_on : `GEOID` or `geohash`
        name of the covariates table to be read, 
        and also used to infer the index column
        
    covariates_df : None or pd.DataFrame
        dataframe of covariates indexed by `index_col` to
        merge with geographic units. If None, collected.
    processed_covariates_dr : str
        file directory if covariates_df is None
    covariates_names : list
        covariate columns to include

    standardize : bool
        whether to standardize the covariates
    
    Returns
    ----------
    gpd.GeoDataFrame
    """
    #Collect the covariates dataframe:
    if covariates_df is None:
        if table_name is None:
            if col_to_merge_on == 'geohash':
                table_name = f'geohash_{len(gdf[col_to_merge_on].iloc[0])}_covariates'
            else:
                table_name=col_to_merge_on+'_covariates'
                
        covariates_df = get_covariates(table=table_name,
                                       processed_covariates_dir=processed_covariates_dir,
                                       set_index=True)
    
    #Merge:
    covariates_gdf = gdf.merge(covariates_df,
                               left_on=col_to_merge_on, right_index=True,
                               how='left')
            
    #Log the population:
    if 'log_population' in covariates_names:
        covariates_gdf['log_population'] = np.log(covariates_gdf['population'])

    #Select covariates:
    covariates_arr = np.nan_to_num(covariates_gdf[covariates_names].values)

    #Standardize:
    if standardize: covariates_arr = (covariates_arr - covariates_arr.mean(axis=0))/(covariates_arr.std(axis=0))

    return covariates_arr
                           
##########################################################################################################

def include_park_area(gdf,
                      id_col='GEOID',
                      include_as_fraction=True,
                      projected_crs=vars._projected_crs,
                      park_shp=vars._park_shp):
    """
    Include park area in a geodataframe of
       census or geohash geometries

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        geometries
    id_col : str
        column to turn into index
    include_as_fraction : bool
        whether to include `park_area` as a
        fraction of the unit area
    
    Returns
    ----------
    np.array
    """
                          
    #Read the parks geodataframe:
    parks_gdf = gpd.read_file(park_shp).to_crs(projected_crs)

    #Project the geodataframe:
    projected_gdf = gdf.to_crs(projected_crs)
                          
    #Merge the geodataframes:
    merged = gpd.overlay(parks_gdf, projected_gdf, how='intersection')

    #Get the total park area:
    merged['park_area'] = merged.area
    park_area = merged[[id_col, 'park_area']].groupby(id_col).sum()

    #Include the park area in the original dataframe:
    gdf_with_parks = gdf.merge(park_area, left_on=id_col, right_index=True, how='left')
    gdf_with_parks.loc[:,'park_area'] = gdf_with_parks['park_area'].fillna(0)

    #In case we want the percentage:
    if include_as_fraction:
        full_area = gdf_with_parks.to_crs(projected_crs).area
        gdf_with_parks.loc[:,'park_area'] = gdf_with_parks['park_area']/full_area
    
    return gdf_with_parks['park_area'].values

def include_land_area(gdf,
                      id_col='GEOID',
                      include_as_fraction=True,
                      projected_crs=vars._projected_crs,
                      outline_shp=vars._nyc_shp):
    """
    Include land area in a geodataframe of
       census or geohash geometries

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        geometries
    id_col : str
        column to turn into index
    include_as_fraction : bool
        whether to include `water_area` as a
        fraction of the unit area
    
    Returns
    ----------
    np.array
    """
                          
    #Collect the boundary:
    polygon_gdf = gpd.read_file(outline_shp)
    boundary_gdf = gpd.GeoDataFrame(geometry=[polygon_gdf.unary_union], crs=polygon_gdf.crs).to_crs(projected_crs)

    #Project the geodataframe:
    projected_gdf = gdf.to_crs(projected_crs)
                          
    #Clip the geodataframe to the boundary:
    merged = gpd.clip(projected_gdf, boundary_gdf)

    #Get the land area:
    merged['land_area'] = merged.area
    land_area = merged[[id_col, 'land_area']].groupby(id_col).sum()

    #Include the land area in the original dataframe:
    gdf_with_land = gdf.merge(land_area, left_on=id_col, right_index=True, how='left')
    gdf_with_land.loc[:,'land_area'] = gdf_with_land['land_area'].fillna(0)

    #In case we want the percentage:
    if include_as_fraction:
        full_area = gdf_with_land.to_crs(projected_crs).area
        gdf_with_land.loc[:,'land_area'] = gdf_with_land['land_area']/full_area
    
    return gdf_with_land['land_area'].values

##########################################################################################################

def project_demographics(target_gdf,
                         demographics_gdf,
                         demographic_cols=None,
                         weighting_col=None,
                         projected_crs=vars._projected_crs):

    """
    Projected the demographics of a GeoDataFrame onto another
      geodataframe, weighting results according to the
      area of intersections or according to another column

    Parameters
    ----------
    target_gdf : Geopandas.GeoDataFrame
        gdf with final geometry for which we want to compute
        demographics

    demographics_gdf : Geopandas.GeoDataFrame
        gdf with current geometry for which we have
        demographics

    demographic_cols : list of strings
        columns containing demographics we want to agregate, which
        must be numeric only

    weighting_col : None or string or dict
        column to weight values for. if a dictionary, one can pass
        different values according to demographic columns. if
        None, use area only.
    
    Returns
    ----------
    target_gdf_with_demographics : Geopandas.GeoDataFrame
    """
    
    #Let's ensure we have all weighting columns:
    if demographic_cols is None: demographic_cols = list(demographics_gdf.columns)
    if type(weighting_col) != dict: weighting_col = {k:weighting_col for k in demographic_cols}
    for column in demographic_cols:
        if column not in weighting_col: weighting_col[column] = None
    
    #First we project both geodataframes to the same CRS:
    demographics_proj = demographics_gdf.to_crs(projected_crs).reset_index(names='demographic_idx')
    target_proj = target_gdf.to_crs(projected_crs).reset_index(names='target_idx')
    
    #Compute the areas:
    demographics_proj['demographics_area'] = demographics_proj.area
    target_proj['target_area'] = target_proj.area
    
    #Take the intersections (this creates a dataframe of all tiny intersection polygons):
    intersections = gpd.overlay(target_proj, demographics_proj)
    
    #Now we compute the fraction of the demographics unit that was covered by each intersection:
    intersections['fraction_demographics'] = intersections.area/intersections['demographics_area']
    
    #We can define a weighting functions based off this fraction and a possible weighting column:
    def agg_function(col):
        #If no column we use just the area:
        if col is None:
            def agg_lambda(x):
                #Get the area of each row:
                row_area = intersections.loc[x.index, 'fraction_demographics']
                #Aggregate by summing over the column with these weights:
                agg = sum(x*row_area)
                #Return:
                return agg
            
        #Otherwise we multiply by the corresponding column:
        else:
            def agg_lambda(x):
                #Get the area of each row:
                row_area = intersections.loc[x.index, 'fraction_demographics']
                #Get the value of the weighting column:
                w = intersections.loc[x.index, col]
                #Aggregate with a weighted average if weights don't sum to zero:
                weights = row_area*w
                agg = np.average(x, weights=row_area*w) if sum(weights) != 0 else 0
                #Return:
                return agg
                
        return agg_lambda
    
    agg_dict = {covariate: agg_function(weighting_covariate) for covariate, weighting_covariate in weighting_col.items()}
    
    #For each covariate, do this weighting:
    target_gdf_with_demographics = intersections.groupby('target_idx').agg(agg_dict)
    
    #Georeference:
    target_gdf_with_demographics = target_proj.set_index('target_idx').merge(target_gdf_with_demographics, left_index=True, right_index=True)

    return target_gdf_with_demographics.reset_index(drop=True) #remove the target_idx name

##########################################################################################################

def include_liu(GEOIDs):

    liu_df = pd.read_csv('../d01_data/d01_raw/d03_population-data/liu-coefficients.csv').set_index('census_tract')
    liu_mapping = liu_df.projected_census_coe.to_dict()
    liu_vals = [liu_mapping[g] if g in liu_df.index else np.nan for g in GEOIDs]

    return liu_vals