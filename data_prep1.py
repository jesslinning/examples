"""Data prep"""

import pandas as pd
import numpy as np
import time, re
from sklearn.neighbors import KNeighborsRegressor
from scipy.spatial.distance import cdist
from scipy.stats import linregress
import utils, eureqa_utils

excel_path = '/Users/jessicalin/Documents/SE/projects/schlumberger/data/Nutonian.xlsx'
xlsx       = pd.ExcelFile( excel_path )

def main():
    prod, gen, xyc, intervals, events, warter = prep_data()
    utils.print_log( 'data prepped' )

    merged = merge_data( prod, gen, xyc, intervals, events, warter )
    utils.print_log( 'data joined' )

    features = add_features( merged )
    utils.print_log( 'features added' )

    xlsx.close()

    return features

def prep_data():
    """Prep individual data tables"""

    prod      = get_production_data()
    gen       = get_general_data()
    xyc       = get_xycdate_data()
    intervals = get_intervals()
    events    = get_events_data()
    warter    = get_warter_data()

    return prod, gen, xyc, intervals, events, warter

def merge_data( prod, gen, xyc, intervals, events, warter ):
    """Merge tables together"""
    
    # merge well characteristics
    merged = merge_characteristics( prod, [ gen, xyc ] )

    # merge intervals data
    merged = merge_intervals( merged, intervals, xyc )

    # merge in winj/pinj values for nearest water producing well
    merged = merge_warter( merged, warter, xyc )

    # merge in event data
    merged = merge_events( merged, events )

    return merged

def add_features( df ):
    """Add features to merged data"""

    df = calc_avg_production( df )
    df = calc_production_slope( df )
    df = calc_pct_change( df )

    return df


###############################
# Prep individual data tables #
###############################

def get_production_data():
    """Get monthly production data - takes ~2min"""

    f_start = time.time()

    # read in production data
    prod_df  = pd.DataFrame()
    for sheet in [ 'Monthly_PROD1_OFM', 'Monthly_PROD2_OFM', 'Monthly_PROD3_OFM', 'Monthly_PROD4_OFM' ]:
        sheet_df = pd.read_excel( xlsx, sheetname=sheet )
        prod_df  = pd.concat( [ prod_df, sheet_df ], ignore_index=True )


    # drop missing values
    start = time.time()
    prod_df.dropna(  how='all', inplace=True ) # 2208 rows in PROD2 and 617 rows in PROD3 are completely blank
    prod_df.dropna(  subset=[ 'OIL [bbl]', 'GAS [Mcf]', 'WATER [bbl]' ], how='all', inplace=True ) # 12 rows in PROD2, 9 rows in PROD3, and 10 rows in PROD4 have no prod values for rows with an assigned UNIQUEID & date
    utils.print_log( 'drop missing: ' + utils.secs_to_hms( time.time() - start ) )

     # resample to ensure monthly increments
    start   = time.time()
    prod_df = prod_df.set_index( 'Date' ).groupby( 'UNIQUEID' ).resample( '1M' ).sum().reset_index()
    prod_df.loc[ ~prod_df[ 'DAYS [days]' ].isnull(), 'Well_Open' ] = 1
    prod_df[ 'Well_Open'   ].fillna( 0, inplace=True )
    prod_df[ 'OIL [bbl]'   ].fillna( 0, inplace=True )
    prod_df[ 'GAS [Mcf]'   ].fillna( 0, inplace=True )
    prod_df[ 'WATER [bbl]' ].fillna( 0, inplace=True )
    prod_df.drop( 'DAYS [days]', axis=1, inplace=True )
    utils.print_log( 'resample: ' + utils.secs_to_hms( time.time() - start ) )

    # add counter for months of production history
    start   = time.time()
    prod_df[ 'Well_Age' ] = prod_df.groupby( 'UNIQUEID' ).cumcount() + 1
    utils.print_log( 'age counter: ' + utils.secs_to_hms( time.time() - start ) )

    utils.print_log( 'production prepped: ' + utils.secs_to_hms( time.time() - f_start ) )
    return prod_df

def get_warter_data():
    """Get warter data"""

    f_start = time.time()

    # read in data
    warter = pd.read_excel( xlsx, sheetname='Monthly_Warter_INJ_OFM' )

    # some cleanup
    warter.drop( [ 'DAYS [days]' ], axis=1, inplace=True )

    # resample to ensure monthly increments, fill nans with 0
    warter = warter.set_index( 'Date' ).groupby( 'UNIQUEID' ).resample( 'M' ).mean()
    warter[ 'WINJ [bbl]' ]  = warter[ 'WINJ [bbl]' ].fillna( 0 )
    warter[ 'PINJ [psig]' ] = warter[ 'PINJ [psig]' ].fillna( 0 )
    warter.reset_index( inplace=True )

    utils.print_log( 'warter prepped: ' + utils.secs_to_hms( time.time() - f_start ) )

    return warter

def get_general_data():
    """Get data from general tab - takes ~25 seconds"""

    f_start = time.time()

    # read in data
    gen = pd.read_excel( xlsx, sheetname='GENERAL' )

    # some cleanup
    gen.drop( [ 'WELLBORE', 'Completation_Date' ], axis=1, inplace=True )
    gen[ 'PRODUCER' ]  = gen[ 'PRODUCER' ].replace( { 'Y': 1, 'N': 0 } )

    # drop columns identified by slb as unnecessary
    gen.drop( [ 'SUB_STATE', 'COMPLETION_PACKING', 'COLLAPSED', 'Completation_Type' ], axis=1, inplace=True )

    # clean & dummy categorical columns
    gen[ 'RESERVOIR' ] = gen[ 'RESERVOIR' ].replace( np.nan, 'RESERVOIR(DNE)' ).apply( lambda x: re.search( r'.*\((.*)\)', x ).group(1) ).replace( 'DNE', np.nan )
    # gen = pd.get_dummies( gen, columns=[ 'RESERVOIR', 'WELLTYPE', 'SUB_STATE', 'COMPLETION_PACKING', 'COLLAPSED', 'Completation_Type' ] )
    gen = pd.get_dummies( gen, columns=[ 'RESERVOIR', 'WELLTYPE' ] )

    utils.print_log( 'general prepped: ' + utils.secs_to_hms( time.time() - f_start ) )

    return gen

def get_xycdate_data():
    """Get XY coordinates, elevation, depth, and glevel - takes ~25 seconds"""

    f_start = time.time()

    # read in data
    xyc = pd.read_excel( xlsx, sheetname='XY-CDATE' )

    # some cleanup
    xyc.drop( [ 'CDATE', 'ALIAS', 'WELLBORE' ], axis=1, inplace=True )

    # interpolate missing values
    for col in [ 'KBELEVATION [ft]', 'TOTALDEPTH [ft]', 'GLEVEL' ]:
        interpolate( col, xyc )

    utils.print_log( 'xyc prepped: ' + utils.secs_to_hms( time.time() - f_start ) )

    return xyc

def get_intervals():
    """Get intervals data"""

    f_start = time.time()

    # read in data
    intervals_df = pd.DataFrame()
    for sheet in [ 'INTERVALS_1', 'INTERVALS_2' ]:
        sheet_df = pd.read_excel( xlsx, sheetname=sheet )
        intervals_df = pd.concat( [ intervals_df, sheet_df ], ignore_index=True )
    intervals_df[ 'DK_Diff' ] = intervals_df[ 'TOP [ft]' ] - intervals_df[ 'BASE [ft]' ]
    intervals_df.rename( columns={ 'STATUS': 'DK_STATUS' }, inplace=True )
    intervals_df = pd.get_dummies( intervals_df, columns=[ 'DK_STATUS' ] )

    # get aggregate level data
    aggregates = intervals_df.groupby( 'UNIQUEID' )[ 'DK' ].max().to_frame() # start off with number of intervals per well
    aggregates.rename( columns={ 'DK': 'Total_Num_DK' }, inplace=True )
    for metric in [ 'TOP [ft]', 'BASE [ft]', 'DK_Diff' ]:

        # calc average depth across all DK
        new_col = 'Avg_DK_{}'.format( metric )
        aggregates[ new_col ] = intervals_df.groupby( 'UNIQUEID' )[ metric ].mean()

        # calc min depth across all DK
        new_col = 'Min_DK_{}'.format( metric )
        aggregates[ new_col ] = intervals_df.groupby( 'UNIQUEID' )[ metric ].min()

        # calc max depth across all DK
        new_col = 'Max_DK_{}'.format( metric )
        aggregates[ new_col ] = intervals_df.groupby( 'UNIQUEID' )[ metric ].max()

    aggregates.reset_index( inplace=True )

    # get time series data
    status_cols = [ c for c in intervals_df if c.startswith( 'DK_STATUS_' ) ]
    int_ts      = intervals_df.groupby( [ 'UNIQUEID', 'Date' ] )[ status_cols ].sum()
    open_dk     = intervals_df.loc[ intervals_df[ 'DK_STATUS_OPEN' ] == 1 ]
    for metric in [ 'TOP [ft]', 'BASE [ft]', 'DK_Diff' ]:

        # calc average depth across all open DK
        new_col = 'Avg_Open_DK_{}'.format( metric )
        int_ts[ new_col ] = open_dk.groupby( [ 'UNIQUEID', 'Date' ] )[ metric ].mean()

        # calc average depth across all open DK
        new_col = 'Min_Open_DK_{}'.format( metric )
        int_ts[ new_col ] = open_dk.groupby( [ 'UNIQUEID', 'Date' ] )[ metric ].min()

        # calc average depth across all open DK
        new_col = 'Max_Open_DK_{}'.format( metric )
        int_ts[ new_col ] = open_dk.groupby( [ 'UNIQUEID', 'Date' ] )[ metric ].max()

    # resample time series data from daily to monthly
    int_ts.reset_index( inplace=True )
    int_ts[ 'Date' ] = int_ts[ 'Date' ].dt.to_period( 'M' ).dt.to_timestamp( 'M' )
    int_ts = int_ts.groupby( [ 'UNIQUEID', 'Date' ] ).last()
    int_ts.fillna( 0, inplace=True )
    int_ts.reset_index( inplace=True )

    # combine aggregate with time series data
    all_interval = int_ts.merge( aggregates, how='left', on='UNIQUEID' )

    utils.print_log( 'intervals prepped: ' + utils.secs_to_hms( time.time() - f_start ) )

    return all_interval

def get_events_data():
    """Get well events - takes ~30s"""

    f_start = time.time()

    # read in event data & fix column names
    events = pd.read_excel( xlsx, sheetname='INTERVENTIONS_OFM' )
    events.columns = [ 'UNIQUEID', 'Date', 'Year', 'Month', 'Day', 'Event' ]

    # fix dates
    events[ 'Date' ] = events[ 'Date' ].dt.to_period( 'M' ).dt.to_timestamp( 'M' )
    events.drop( [ 'Year', 'Day', 'Month' ], axis=1, inplace=True )

    # filter to shorter list of events
    keep_events =   [ 
                        'Chemical Stimulation',
                        'Cleaning',
                        'COMPLETION',
                        'Pump Replacement',
                        'Reactivation',
                        'Rod Change',
                        'SRP Installation',
                        'Workover',
                        'zone isolated with arena'
                    ]
    events = events.loc[ events[ 'Event' ].isin( keep_events ) ]

    # dummy events & filter to shorter list of events
    events[ 'Event' ] = events.Event.str.replace( ' ', '_' )
    events = pd.get_dummies( events, columns=[ 'Event' ] )

    # merge multiple events per month into single row
    events = events.groupby( [ 'UNIQUEID', 'Date' ] ).sum()
    events = events.applymap( lambda x: int(bool(x)) )
    events.reset_index( inplace=True )

    utils.print_log( 'events prepped: ' + utils.secs_to_hms( time.time() - f_start ) )

    return events


##################################
# Merge specific tables together #
##################################

def merge_characteristics( prod, df_list ):
    """Merge in well characteristic data to overall time series production data - dropping 3 production wells since they don't exist in either general/xy tables, and only have 1 month of production data each"""

    merged = None

    # for well characteristic data
    for right_df in df_list:
        if merged is None:
            merged = prod.merge( right_df, how='inner', on='UNIQUEID' )
        else:
            merged = merged.merge( right_df, how='inner', on='UNIQUEID' )

    return merged

def merge_intervals( df, intervals, xyc ):
    """Merge in intervals data - timeseries, but will be missing months"""

    # get xy coordinates for unique production wells
    prod_xy = df.drop_duplicates( subset=[ 'UNIQUEID' ] )
    prod_xy[ 'point' ] = [ (x, y) for x,y in zip( prod_xy[ 'XCOOR' ], prod_xy[ 'YCOOR' ] ) ]

    # get xy coordinates for unique interval wells
    int_xy = intervals[[ 'UNIQUEID' ]].drop_duplicates().merge( xyc[[ 'UNIQUEID', 'XCOOR', 'YCOOR' ]], how='left', on='UNIQUEID' )
    int_xy[ 'point' ] = [ (x, y) for x,y in zip( int_xy[ 'XCOOR' ], int_xy[ 'YCOOR' ] ) ]

    # find nearest interval well
    prod_xy[ 'closest' ]   = [ closest_point( x, list( int_xy[ 'point' ] ) ) for x in prod_xy[ 'point' ] ]
    prod_xy[ 'closest_well' ] = [ match_value( int_xy, 'point', x, 'UNIQUEID' ) for x in prod_xy[ 'closest' ] ]

    # merge interpolated interval data into production data
    df = df.merge( prod_xy[[ 'UNIQUEID', 'closest_well' ]], on='UNIQUEID', how='left' )

    # extend intervals data to all months in production data
    extended_intervals = pd.DataFrame()
    wells = df.closest_well.unique()
    progress = eureqa_utils.Progress( len( wells ), name='Extend intervals' )
    for well in wells:
        closest  = df.loc[ df[ 'closest_well' ] == well, 'Date' ]
        min_date = closest.min()
        max_date = closest.max()
        new_ix   = pd.DatetimeIndex( start=min_date, end=max_date, freq='M' )
        
        well_intervals = intervals.loc[ intervals.UNIQUEID == well ].set_index( 'Date' )
        well_intervals = well_intervals.reindex( index=new_ix )
        well_intervals[ 'UNIQUEID' ] = well
        well_intervals.ffill( inplace=True )
        well_intervals = well_intervals.reset_index().rename( columns={ 'index': 'Date' } )
        well_intervals.fillna( 0, inplace=True )

        extended_intervals = pd.concat( [ extended_intervals, well_intervals ], ignore_index=True )
        
        progress.increment()
    progress.finish()

    df = df.merge( extended_intervals.rename( columns={ 'UNIQUEID': 'closest_well' } ), how='left', on=[ 'closest_well', 'Date' ] )
    df.drop( 'closest_well', axis=1, inplace=True )

    return df

def merge_warter( df, warter, xyc ):
    """Merge in time series data for nearest water injector wells"""

    # get xy coordinates for unique wells in warter
    warter_xy = warter[[ 'UNIQUEID' ]].drop_duplicates().merge( xyc[[ 'UNIQUEID', 'XCOOR', 'YCOOR' ]], how='left', on='UNIQUEID' )
    warter_xy[ 'point' ] = [ (x, y) for x,y in zip( warter_xy[ 'XCOOR' ], warter_xy[ 'YCOOR' ] ) ]

    # get xy coordinates for unique wells in df
    prod_xy = df.drop_duplicates( subset=[ 'UNIQUEID' ] )
    prod_xy[ 'point' ] = [ (x, y) for x,y in zip( prod_xy[ 'XCOOR' ], prod_xy[ 'YCOOR' ] ) ]

    # get closest water injector for each production well
    prod_xy[ 'closest' ]        = [ closest_point( x, list( warter_xy[ 'point' ] ) ) for x in prod_xy[ 'point' ] ]
    prod_xy[ 'closest_warter' ] = [ match_value( warter_xy, 'point', x, 'UNIQUEID' ) for x in prod_xy[ 'closest' ] ]

    # calculate distance between well and nearest water injecting well
    prod_xy[ 'warter_distance' ] = np.sqrt((np.array(zip(*prod_xy['point'])[0]) - np.array(zip(*prod_xy['closest'])[0]))**2 + (np.array(zip(*prod_xy['point'])[1]) - np.array(zip(*prod_xy['closest'])[1]))**2)

    # merge in time series winj/pinj values 
    df = df.merge( prod_xy[[ 'UNIQUEID', 'closest_warter' ]], on='UNIQUEID', how='left' )
    df = df.merge( warter.rename( columns={ 'UNIQUEID': 'closest_warter' } ), how='left', on=[ 'closest_warter', 'Date' ] )
    df = df.merge( prod_xy[[ 'UNIQUEID', 'warter_distance' ]], how='left', on='UNIQUEID' )

    # fill in missing winj/pinj values
    df[ 'WINJ [bbl]' ]  = df[ 'WINJ [bbl]' ].fillna( 0 )
    df[ 'PINJ [psig]' ] = df[ 'PINJ [psig]' ].fillna( 0 )

    df.drop( 'closest_warter', axis=1, inplace=True )

    return df

def merge_events( df, events ):
    """Merge in event occurences"""

    merged = df.merge( events, how='left', on=[ 'UNIQUEID', 'Date' ] )
    for c in [ c for c in events.columns if c.startswith( 'Event_' ) ]:
        merged[ c ] = merged[ c ].fillna( 0 )

    return merged


################
# Add features #
################

def calc_avg_production( prod ):
    """Calculate rate of production"""

    prod_metrics = [ 'OIL [bbl]', 'GAS [Mcf]', 'WATER [bbl]' ]
    other_time   = [ 'WINJ [bbl]', 'PINJ [psig]' ]
    terms = [ 12, 6, 2 ]

    progress = eureqa_utils.Progress( len( terms ) * ( len( prod_metrics ) + len( other_time ) ), name='Calc avg' )
    for months in terms:

        for metric in prod_metrics:
            new_col = '{n:.0f}Mo Production - {m}'.format( m=metric, n=months )
            prod[ new_col ] = prod.groupby( 'UNIQUEID' )[ metric ].rolling( months ).mean().values
            
            post_col = 'Post {c}'.format( c=new_col )
            prod[ post_col ] = prod.groupby( 'UNIQUEID' )[ metric ].shift( -months )

            progress.increment()

        for metric in other_time:
            new_col = '{n:.0f}Mo Avg - {m}'.format( m=metric, n=months )
            prod[ new_col ] = prod.groupby( 'UNIQUEID' )[ metric ].rolling( months ).mean().values

            progress.increment()
    progress.finish()

    return prod

def calc_production_slope( prod ):
    """Calculate linear slope over a given time period (smoother than the mean)"""

    time_metrics = [ 'OIL [bbl]', 'GAS [Mcf]', 'WATER [bbl]', 'WINJ [bbl]', 'PINJ [psig]' ]
    terms = [ 24, 12, 6 ]

    progress = eureqa_utils.Progress( len( terms ), name='Calc slope' )
    for months in terms:
        slope_cols = [ '{n}Mo Slope - {m}'.format( n=months, m=m ) for m in time_metrics ]
        prod[ slope_cols ] = prod.groupby( 'UNIQUEID' )[ time_metrics ].apply( 
                                lambda x: x.rolling( window=months, min_periods=1 ).apply(
                                        lambda y: linregress( range( len( y ) ), y ).slope ) ).reset_index( drop=True )
        progress.increment()

        # for metric in time_metrics:

        #     # calculate linear regression of production
        #     new_col = '{n}Mo Slope - {m}'.format( n=months, m=metric )
        #     prod[ new_col ] = prod.groupby( 'UNIQUEID' )[ metric ].apply( 
        #                             lambda x: x.rolling( window=months, min_periods=1 ).apply(
        #                                     lambda y: linregress( range( len( y ) ), y ).slope ) ).values
        #     progress.increment()
    progress.finish()

    return prod

def calc_pct_change( prod ):
    """Calculate % change"""

    time_metrics = [ 'OIL [bbl]', 'GAS [Mcf]', 'WATER [bbl]', 'WINJ [bbl]', 'PINJ [psig]' ]
    terms = [ 24, 12, 6, 2 ]

    progress = eureqa_utils.Progress( len( terms ) * len( time_metrics ), name='Calc pct change' )
    for months in terms:
        for metric in time_metrics:
            new_col = '{n:.0f}Mo Pct Change - {m}'.format( m=metric, n=months )
            prod[ new_col ] = prod.groupby( 'UNIQUEID' )[ metric ].pct_change( periods=months )
            prod[ new_col ] = prod[ new_col ].fillna( prod[ metric ] )

            progress.increment()
    progress.finish()

    return prod

def calc_event_features( prod ):
    event_cols = [ c for c in prod.columns if c.startswith( 'Event_' ) ]

    # calculate cumulative historical events
    cum_count = prod.groupby( 'UNIQUEID' )[ event_cols ].cumsum()
    cum_count.rename( columns={ event: '{}_Count_Total'.format( event ) for event in event_cols }, inplace=True )
    features  = pd.concat( [ prod, cum_count ], axis=1 )

    # calculate binary flag if event has happened in past
    cum_flag = features[ cum_count.columns ].applymap( lambda x: int(bool(x)) )
    cum_flag.rename( columns=lambda c: c.replace( 'Count_Total', 'Has_Occurred' ), inplace=True )
    features = pd.concat( [ features, cum_flag ], axis=1 )

    # rolling lookup
    for months in [ 12, 6, 3 ]:

        # calculate cumulative historical events
        cum_count = features.groupby( 'UNIQUEID' )[ event_cols ].rolling( window=months, min_periods=1 ).sum().reset_index( drop=True )
        cum_count.rename( columns={ event: '{e}_Count_P{n}Mo'.format( e=event, n=months ) for event in event_cols }, inplace=True )
        features  = pd.concat( [ features, cum_count ], axis=1 )

        # calculate binary flag if event has happened
        cum_flag = features[ cum_count.columns ].applymap( lambda x: int(bool(x)) )
        cum_flag.rename( columns=lambda c: c.replace( 'Count', 'Has_Occurred' ), inplace=True )
        features = pd.concat( [ features, cum_flag ], axis=1 )

    return features


###########
# helpers #
###########

def interpolate( col, df ):
    """Interpolate column value based on 3 nearest neighbors"""

    # separate into nan / non nan sets
    non_nan = df.loc[ ~df[ col ].isnull() ]
    is_nan = df.loc[ df[ col ].isnull() ]

    # learn neighbors and run interpolation
    neigh = KNeighborsRegressor( n_neighbors=3, weights='distance' )
    neigh.fit( non_nan[['XCOOR', 'YCOOR']], non_nan[ col ] )
    df.loc[ df[ col ].isnull(), col ] = neigh.predict( is_nan[['XCOOR', 'YCOOR']] )

def closest_point(point, points):
    """ Find closest point from a list of points. """
    return points[cdist([point], points).argmin()]

def match_value(df, col1, x, col2):
    """ Match value x from col1 row to value in col2. """
    return df[df[col1] == x][col2].values[0]