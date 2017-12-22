import pandas as pd
import numpy as np
from datetime import timedelta
from isoweek import Week
import utils, eureqa_utils

def prep_data():
    """Prep individual data files"""

    folder = '../data/HF_data_for_Nutonian_04042017'

    # get target variable
    path = '{}/actual_boxes_sales.csv'.format( folder )
    actual = pd.read_csv( path )
    actual.rename( columns={ 'act_boxes': 'boxes' }, inplace=True )

    # get conversion projections
    projections = prep_projections()

    # get past conversions
    path = '{}/past_conversions.csv'.format( folder )
    conversions = pd.read_csv( path )

    # get outlooks
    path = '{}/outlooks.csv'.format( folder )
    outlooks = pd.read_csv( path )
    outlooks.rename( columns={ 'outlook_week': 'hellofresh_week' }, inplace=True )
    outlooks[ 'outlook_date' ] = pd.to_datetime( outlooks[ 'outlook_date' ], format='%Y-%m-%d' )
    outlooks.sort_values( [ 'country', 'outlook_date' ], inplace=True )
    outlooks.drop( 'outlook_date', axis=1, inplace=True )
    all_outlooks = pd.DataFrame()
    for i in range(7):
        dow = outlooks.groupby( [ 'country', 'hellofresh_week' ] ).nth(i)
        dow.rename( columns=lambda c: 'DofW{i}_projected_{c}'.format( i=i+1, c=c ), inplace=True )
        all_outlooks = pd.concat( [ all_outlooks, dow ], axis=1 )
    all_outlooks.reset_index( inplace=True )

    # get holidays
    path = '{}/holidays.csv'.format( folder )
    holidays = pd.read_csv( path )
    holidays.drop( 'Year', axis=1, inplace=True )
    holidays.rename( columns={ 'Country': 'country', 'HF_Week': 'hellofresh_week' }, inplace=True )
    holidays[ 'Holiday' ] = holidays[ 'Holiday' ].str.title().str.replace( ' ', '' )
    holidays[ 'is_holiday' ] = 1
    holidays = pd.get_dummies( holidays, columns=[ 'Holiday' ] )

    return actual, projections, conversions, all_outlooks, holidays

def prep_projections():
    """Prep conversion projections"""

    # read in data
    path = '../data/HF_data_for_Nutonian_04042017/conversion_projections.csv'
    df = pd.read_csv( path )

    # convert week to datetime
    for date_col in [ 'hellofresh_week', 'CreatedInWeek' ]:
        new_col = '{}_date'.format( date_col )
        df[ new_col ] = df[ date_col ].apply( lambda x: Week( int(x[:4]), int(x[-2:] ) ).monday() - timedelta( days=2 ) )
        df[ new_col ] = pd.to_datetime( df[ new_col ], format='%Y-%m-%d' )

    # get last projection per outlook week
    projections = pd.DataFrame()
    week10 = pd.to_datetime( '2017-03-04' )
    week16 = pd.to_datetime( '2017-04-15' )
    for country in df[ 'country' ].unique():
        if country == 'US':
            country_df = df.loc[ ( df[ 'country' ] == country ) & ( df[ 'hellofresh_week_date' ] >= week10 ) & ( df[ 'hellofresh_week_date' ] <= week16 ) ]
        else:
            country_df = df.loc[ ( df[ 'country' ] == country ) & ( df[ 'hellofresh_week_date' ] >= week10 ) & ( df[ 'hellofresh_week_date' ] <= week16 ) & ( df[ 'CreatedInWeek' ] != '2017-W11' ) ]
        last = country_df.sort_values( 'CreatedInWeek_date' ).groupby( 'hellofresh_week' ).last().reset_index()
        projections = pd.concat( [ projections, last ], ignore_index=True )

    # rename and reduce columns
    projections.rename( columns={ 'Conversions': 'conversions' }, inplace=True )
    projections = projections[[ 'hellofresh_week', 'country', 'conversions' ]]

    return projections

def merge_data( actual, conversions, outlooks, holidays, projections, is_test ):
    """Merge prepped data files together"""

    # add test weeks
    if is_test:
        predict_weeks = [ '2017-W11', '2017-W12', '2017-W13', '2017-W14', '2017-W15', '2017-W16' ]
        for country in [ 'GB', 'US', 'NL' ]:
            actual = pd.concat( [ actual, pd.DataFrame( { 'hellofresh_week': predict_weeks, 'country': [ country ]*6 } ) ], ignore_index=True )

    # merge data together
    merged = actual.merge( conversions, on=[ 'country', 'hellofresh_week' ], how='left' )
    merged = merged.merge( outlooks, on=[ 'country', 'hellofresh_week' ], how='left' )
    merged = merged.merge( holidays, on=[ 'country', 'hellofresh_week' ], how='left' )
    for col in holidays.columns:
        if 'holiday' in col.lower():
            merged[ col ] = merged[ col ].fillna( 0 )

    # add conversion projections
    if is_test:
        merged = merged.merge( projections, on=[ 'country', 'hellofresh_week' ], how='left' )
        merged[ 'conversions_x' ] = merged[ 'conversions_x' ].fillna( merged[ 'conversions_y' ] )
        merged.drop( 'conversions_y', axis=1, inplace=True )
        merged.rename( columns={ 'conversions_x': 'conversions' }, inplace=True )

    # # add row weights
    # merged[ 'weight_exp' ] = merged.groupby( 'country' )[ 'boxes' ].transform( lambda c: map( lambda x: 1./(2**x), reversed(range(c.shape[0])) ) )
    # merged[ 'weight_lin' ] = merged.groupby( 'country' )[ 'boxes' ].transform( lambda c: np.linspace( 0.5, 1, num=c.shape[0] ) )

    # convert week to datetime
    merged[ 'date' ] = merged[ 'hellofresh_week' ].apply( lambda x: Week( int(x[:4]), int(x[-2:] ) ).monday() - timedelta( days=2 ) )
    merged[ 'date' ] = pd.to_datetime( merged[ 'date' ], format='%Y-%m-%d' )
    merged.sort_values( [ 'country', 'date' ], inplace=True )

    return merged

def model_prep( merged, is_test, e=None ):
    """Prep merged data for modeling and upload to eureqa"""
    folder = '../data'

    # # create separate training set
    # last_week = merged.loc[ merged[ 'date' ] == merged[ 'date' ].max(), 'hellofresh_week' ].unique()[0]
    # train = merged.loc[ merged[ 'hellofresh_week' ] != last_week ]

    # # upload to eureqa
    # if e is not None:
    #     train_ds = utils.dataframe_to_data_source( e, train, 'Train dofw', series_id='country', time_id='date' )
    #     test_ds  = utils.dataframe_to_data_source( e, merged, 'Test dofw', series_id='country', time_id='date' )
    # else:
    #     train.to_csv(  '{}/merged_all_train.csv'.format( folder ), index=False, date_format='%m/%d/%Y' )
    #     merged.to_csv( '{}/merged_all_test.csv'.format( folder ),  index=False, date_format='%m/%d/%Y' )

    # split by country
    ds_dict = {}
    countries = merged[ 'country' ].unique()
    progress = eureqa_utils.Progress( len(countries) )
    for country in countries:
        country_data  = merged.loc[ merged[ 'country' ] == country ]

        if e is not None:
            if is_test:
                ds_name = 'Test {} dofw'.format( country )
            else:
                ds_name = 'Train {} dofw'.format( country )
            ds  = utils.dataframe_to_data_source( e, country_data, ds_name, time_id='date', add_hash=True )
            ds_dict[ country ] = ds
        else:
            country_data.to_csv(  '{f}/merged_{c}_all.csv'.format( f=folder, c=country ),  index=False, date_format='%m/%d/%Y' )
        
        progress.increment()
    progress.finish()

    return ds_dict