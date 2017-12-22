"""Prep individual data types"""

import pandas as pd
import numpy as np
import re, quandl, os, datetime
from dateutil import relativedelta
from sklearn.linear_model import ElasticNet

import utils, eureqa_utils
from eureqa_utils.modeling_utils import TimeSeriesData, TimeTrendCreator

import warnings
warnings.filterwarnings('ignore')


######################
# Create joined data #
######################

def create_data( data_start_month, data_end_month ):
    """Preps & joins files; returns dataset for modeling and an extended dataset for predicting next month (takes ~4min to run)"""

    train_file     = '/Users/jessicalin/Documents/Projects/Audi/data/monthly_sales_train_round13.csv'
    selection_file = '/Users/jessicalin/Documents/Projects/Audi/data/monthly_sales_selection_round13.csv'
    test_file      = '/Users/jessicalin/Documents/Projects/Audi/data/monthly_sales_test_round13.csv'

    if not os.path.isfile( train_file ) and not os.path.isfile( selection_file ) and not os.path.isfile( test_file ):

        # convert month strings into datetime periods
        data_start = pd.to_datetime( data_start_month ).to_period( 'M' )
        data_end   = pd.to_datetime( data_end_month   ).to_period( 'M' )

        # prep individual data files
        utils.print_log( 'prepping individual files' )
        prs_data, bpo_data, saar_data, tier1_data, econ_data, mature_by_month, paid_by_month, finance_type, leads_by_month, sale_leads, dag_spend_data, polk_data, loyalty_data, loyalty_stats, conquest_data, conquest_stats, dimension_lookup, sales_managers, incentives_data, release_data, inventory_data, tier1_adspend, tier2_adspend, comp_adspend = prep_data()

        # join VIN level data and aggregate up to monthly
        utils.print_log( 'joining on vin' )
        monthly_prs = join_vin( prs_data, sale_leads, dimension_lookup, data_end )

        # join in remaining monthly data
        utils.print_log( 'join on month' )
        monthly_sales = join_monthly( monthly_prs, bpo_data, saar_data, tier1_data, econ_data, mature_by_month, paid_by_month, leads_by_month, dag_spend_data, polk_data, loyalty_stats, conquest_stats, sales_managers, incentives_data, release_data, inventory_data, tier1_adspend, tier2_adspend, comp_adspend )

        # add non-seasonal features
        utils.print_log( 'add features' )
        monthly_sales = add_variable_features( monthly_sales )

        # split data into train/selection/test sets
        utils.print_log( 'split data into train/select/test' )
        train_set, selection_set, test_set = split_train( monthly_sales, data_start, data_end )

        # add in seasonality
        utils.print_log( 'add seasonality' )
        train_set, selection_set, test_set = add_seasonality( train_set, selection_set, test_set )

        # final data prep for modeling
        train_set, selection_set, test_set = final_prep( train_set, selection_set, test_set, data_start, data_end )

        # output to csv
        train_set.to_csv(     train_file,     index=False, date_format='%m/%d/%Y' )
        selection_set.to_csv( selection_file, index=False, date_format='%m/%d/%Y' )
        test_set.to_csv(      test_file,      index=False, date_format='%m/%d/%Y' )

    else:
        train_set           = pd.read_csv(    train_file )
        train_set.Month     = pd.to_datetime( train_set.Month,     format='%m/%d/%Y' )
        selection_set       = pd.read_csv(    selection_file )
        selection_set.Month = pd.to_datetime( selection_set.Month, format='%m/%d/%Y' )
        test_set            = pd.read_csv(    test_file )
        test_set.Month      = pd.to_datetime( test_set.Month,      format='%m/%d/%Y' )

    train_set     = train_set.reset_index(     drop=True )
    selection_set = selection_set.reset_index( drop=True )
    test_set      = test_set.reset_index(      drop=True )

    return train_set, selection_set, test_set

def prep_data():
    """Preps individual data files (takes ~2 min 30 secs to run)"""

    progress = eureqa_utils.Progress( 19, name='File prep progress' )

    prs_data = prep_prs(); progress.increment()
    bpo_data = prep_bpo(); progress.increment()
    saar_data = prep_saar(); progress.increment()
    tier1_data = prep_tier1_traffic(); progress.increment()
    econ_data = prep_econ(); progress.increment()
    mature_by_month, paid_by_month, finance_type = prep_afs( prs_data ); progress.increment()
    leads_by_month, sale_leads = prep_leads_to_sales(); progress.increment()
    dag_spend_data = prep_dag_ad_spend(); progress.increment()
    polk_data = prep_polk(); progress.increment()
    loyalty_data, loyalty_stats = prep_loyalty(); progress.increment()
    conquest_data, conquest_stats = prep_conquest(); progress.increment()
    dimension_lookup = prep_sale_dimension( finance_type, loyalty_data, conquest_data ); progress.increment()
    sales_managers = prep_sales_managers(); progress.increment()
    incentives_data = prep_incentives(); progress.increment()
    release_data = prep_model_releases(); progress.increment()
    inventory_data = prep_inventory(); progress.increment()
    tier1_adspend = prep_tier1_adspend(); progress.increment()
    tier2_adspend = prep_tier2_adspend(); progress.increment()
    comp_adspend = prep_comp_adspend(); progress.increment()

    progress.finish()

    return prs_data, bpo_data, saar_data, tier1_data, econ_data, mature_by_month, paid_by_month, finance_type, leads_by_month, sale_leads, dag_spend_data, polk_data, loyalty_data, loyalty_stats, conquest_data, conquest_stats, dimension_lookup, sales_managers, incentives_data, release_data, inventory_data, tier1_adspend, tier2_adspend, comp_adspend

def join_vin( prs_data, sale_leads, dimension_lookup, data_end ):
    """Join files at VIN level, aggregate up to monthly level by dealer, and add sales seasonality (takes ~1 min 10 seconds to run)"""

    # join in sales that came from a lead
    prs_data = prs_data.merge( sale_leads, how='left', on=[ 'VIN', 'DealerCode', 'Month' ] )

    # fill nans and multiply new columns by Sales column to get to the right final count
    for col in [ 'DaysToCloseLead', 'LeadType_3RD_PARTY', 'LeadType_NATIVE' ]:
        prs_data[ col ] = prs_data[ col ].fillna( 0 )
        prs_data[ col ] = prs_data[ col ] * prs_data.Sales

    # get dimension and loyalty award amount (if applicable)
    prs_data = prs_data.merge( dimension_lookup, how='left', on='VIN' )

    # fill dimension nans
    prs_data.AwardAmount = prs_data.AwardAmount.fillna( 0 )
    prs_data.loc[ ( prs_data.AwardAmount == 0 ) & ( pd.isnull( prs_data.ConquestSale_Cash ) ), 'ConquestSale_Cash' ] = 1
    prs_data.loc[ ( prs_data.AwardAmount > 0 ) & ( pd.isnull( prs_data.LoyaltySale_Cash ) ), 'LoyaltySale_Cash' ] = 1

    # multiply by Sales column to get to the right final count
    dimension_cols = [ c for c in prs_data if c.startswith( 'ConquestSale_' ) or c.startswith( 'LoyaltySale' ) ]
    for col in dimension_cols:
        prs_data[ col ] = prs_data[ col ].fillna( 0 )
        prs_data[ col ] = prs_data[ col ] * prs_data.Sales

    # aggregate prs up to monthly level
    monthly_prs = prs_data.groupby( [ 'DealerCode', 'Month' ] ).agg( { 'Sales': 'sum', 'SalesSegmentA': 'sum', 'SalesSegmentB': 'sum', 'SalesSegmentC': 'sum', 'SalesSegmentD': 'sum', 'DaysToCloseLead': 'mean', 'LeadType_3RD_PARTY': 'sum', 'LeadType_NATIVE': 'sum', 'AwardAmount': 'mean', 'ConquestSale_Lease': 'sum', 'ConquestSale_Retail': 'sum', 'LoyaltySale_Lease': 'sum', 'LoyaltySale_Retail': 'sum', 'ConquestSale_Cash': 'sum', 'LoyaltySale_Cash': 'sum', 'DAG': 'first', 'Region': 'first', 'Area': 'first' } ).reset_index()

    # only keep dealers that have enough history & average sales volume >= 25 & survive into end of prediction period
    min_train_start  = data_end - 9
    good_dealers     = monthly_prs.loc[ monthly_prs.Month == min_train_start, 'DealerCode' ].unique() # keep only dealers that had sales 10 months ago
    low_volume       = monthly_prs.groupby( 'DealerCode' ).Sales.mean().loc[ lambda x: x < 30 ].index # keep only dealers that average >25 monthly sales
    survivors        = monthly_prs.loc[ monthly_prs.Month == data_end, 'DealerCode' ].unique() # keep dealers that survive through the end of prediction period
    filtered_monthly = monthly_prs.loc[ ( monthly_prs.DealerCode.isin( good_dealers ) ) & ( ~monthly_prs.DealerCode.isin( low_volume ) ) & ( monthly_prs.DealerCode.isin( survivors ) ) ]

    return filtered_monthly

def join_monthly( monthly_prs, bpo_data, saar_data, tier1_data, econ_data, mature_by_month, paid_by_month, leads_by_month, dag_spend_data, polk_data, loyalty_stats, conquest_stats, sales_managers, incentives_data, release_data, inventory_data, tier1_adspend, tier2_adspend, comp_adspend ):
    """Join remaining monthly files (takes ~7 seconds to run)"""

    def extend_monthly_prs( monthly_prs, bpo_data ):
        """Return extended dataframe of all dealers extended"""

        # set up
        dealers     = monthly_prs.DealerCode.unique()
        extended_df = pd.DataFrame()

        # extend frame per series
        progress = eureqa_utils.Progress( len( dealers ), name='Extend dealer frame' )
        for dealer in dealers:

            # get data/info per series
            dealer_df     = monthly_prs.loc[ monthly_prs.DealerCode == dealer ]
            last_bpo_date = bpo_data.loc[ bpo_data.DealerCode==dealer ].sort_values( 'Month' ).iloc[-1].Month

            # extend dealer data
            extended_dealer = extend_dealer( dealer_df, last_bpo_date )
            extended_df     = pd.concat( [ extended_df, extended_dealer ], ignore_index=True )

            progress.increment()
        progress.finish()

        return extended_df

    def extend_dealer( dealer_df, last_month ):
        """Return dealer dataframe extended to the given last BPO date"""

        dealer_df = dealer_df.set_index( 'Month' )

        ix        = pd.PeriodIndex( start=dealer_df.index[0], end=last_month, freq='M' )
        dealer_df = dealer_df.reindex(ix)

        base_cols = [ 'DealerCode', 'Region', 'DAG', 'Area' ]
        dealer_df[ base_cols ] = dealer_df[ base_cols ].ffill()
        dealer_df[ list( set( dealer_df.columns ) - set( base_cols ) ) ] = dealer_df[ list( set( dealer_df.columns ) - set( base_cols ) ) ].fillna(0)

        dealer_df.Area = dealer_df.Area.astype( 'int' )

        dealer_df = dealer_df.reset_index().rename( columns={ 'index': 'Month' } )

        return dealer_df

    progress = eureqa_utils.Progress( 14, name='Join monthly' )

    # join bpo
    extend_month_prs = extend_monthly_prs( monthly_prs, bpo_data )
    monthly_sales = extend_month_prs.merge( bpo_data, how='left', on=[ 'DealerCode', 'Month' ] )
    progress.increment()

    # join saar
    monthly_sales = monthly_sales.merge( saar_data, how='left', on='Month' )
    for col in saar_data.columns:
        if col != 'Month':
            monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 )
    progress.increment()

    # join tier 1
    monthly_sales = monthly_sales.merge( tier1_data, how='left', on='Month' ) # left join - tier1 data only starts in June 2013 and goes through may 2016
    for col in tier1_data.columns:
        if col != 'Month':
            group_avg = 'g_' + col
            monthly_sales[ group_avg ] = monthly_sales.groupby( 'DealerCode' )[ col ].transform( np.mean )
            monthly_sales[ col ] = monthly_sales[ col ].fillna( monthly_sales[ group_avg ] ) # if no data, fill with group mean
            del monthly_sales[ group_avg ]
    progress.increment()

    # join econ
    monthly_sales = monthly_sales.merge( econ_data, how='inner', on='Month' )
    for col in econ_data.columns:
        if col != 'Month':
            monthly_sales[ col ] = monthly_sales[ col ].fillna(0)
    progress.increment()

    # join mature/paid off by month
    monthly_sales = monthly_sales.merge( mature_by_month, how='left', on=[ 'DealerCode', 'Month' ] )
    monthly_sales = monthly_sales.merge( paid_by_month, how='left', on=[ 'DealerCode', 'Month' ] ) # afs data only goes through apr
    for col in set( mature_by_month.columns.tolist() ).union( set( paid_by_month.columns.tolist() ) ) - { 'DealerCode', 'Month' }:
        monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 ) # fill columns
    progress.increment()

    # join leads by month
    monthly_sales = monthly_sales.merge( leads_by_month, how='left', on=[ 'DealerCode', 'Month' ] )
    for col in leads_by_month.columns:
        if col != 'DealerCode' and col != 'Month':
            monthly_sales[ col ] = monthly_sales[ col ].fillna(0)
    progress.increment()

    # join dag ad spend data
    monthly_sales = monthly_sales.merge( dag_spend_data, how='left', on=[ 'DAG', 'Month' ] )
    for col in [ c for c in dag_spend_data.columns if c.endswith( 'AdSpend' ) ]:
        monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 ) # if no data for that dag, fill ad spend with 0
    progress.increment()

    # join polk data
    monthly_sales = monthly_sales.merge( polk_data, how='left', on=[ 'DealerCode', 'Month' ] )
    for col in [ c for c in polk_data.columns if c.startswith( 'Polk' ) ]:
        monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 ) # if no data for that dealer, fill polk loyalty with 0
    progress.increment()

    # join loyalty & conquest stats
    monthly_sales = monthly_sales.merge( loyalty_stats, how='left', on=[ 'DealerCode', 'Month' ] )
    monthly_sales = monthly_sales.merge( conquest_stats, how='left', on=[ 'DealerCode', 'Month' ] )
    for col in [ c for c in loyalty_stats.columns.union( conquest_stats.columns ) if c != 'Month' and c != 'DealerCode' ]:
        monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 ) # if no data, fill with 0
    progress.increment()

    # join sales managers for eastern region
    monthly_sales = monthly_sales.merge( sales_managers, how='left', on=[ 'Region', 'Month' ] ) # only have data for eastern region as a whole
    for col in [ 'BrandSpecialists', 'SalesManagers' ]:
        monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 ) # if no data for that region, fill with 0
    progress.increment()

    # join incentives level data
    monthly_sales = monthly_sales.merge( incentives_data, how='left', on='Month' )
    for col in [ c for c in incentives_data.columns if c != 'Month' ]:
        monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 ) # if no incentive data for that month, fill with 0
    progress.increment()

    # join model release dates data
    monthly_sales = monthly_sales.merge( release_data, how='left', on='Month' )
    for col in release_data.columns:
        if col != 'Month':
            monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 )
    progress.increment()

    # join inventory data
    monthly_sales = monthly_sales.merge( inventory_data, how='left', on=[ 'DealerCode', 'Month' ] )
    for col in inventory_data.columns:
        if col != 'Month':
            monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 ) # if no inventory data for that dealer/month, fill with 0
    progress.increment()

    # join tier 1 ad spend data
    monthly_sales = monthly_sales.merge( tier1_adspend, how='left', on=[ 'Month' ] )
    for col in tier1_adspend.columns:
        if col != 'Month':
            monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 )

    # join tier 2 ad spend data
    monthly_sales = monthly_sales.merge( tier2_adspend, how='left', on=[ 'Month' ] )
    for col in tier2_adspend.columns:
        if col != 'Month':
            monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 )

    # join competitor ad spend data
    monthly_sales = monthly_sales.merge( comp_adspend, how='left', on=[ 'Month' ] )
    for col in comp_adspend.columns:
        if col != 'Month':
            monthly_sales[ col ] = monthly_sales[ col ].fillna( 0 )

    # sort data by area, then by dealer, then by month
    monthly_sales = monthly_sales.sort_values( [ 'Area', 'DealerCode', 'Month' ] )
    progress.increment()

    progress.finish()

    # force dealer code to be considered a string
    monthly_sales.DealerCode = '_' + monthly_sales.DealerCode

    return monthly_sales

def add_variable_features( monthly_sales ):
    """Add non-seasonal features"""

    # calculate sales dimension percents
    dimension_cols = [ c for c in monthly_sales.columns if 'Sale_' in c ]
    pct_data       = monthly_sales[ dimension_cols ].div( monthly_sales.Sales, axis='index' ).rename( columns=lambda c: c + '_pct' )
    monthly_sales[ pct_data.columns ] = pct_data

    # calculate historical % attainment
    monthly_sales[ 'PctAttainment' ] = ( monthly_sales[ 'Sales' ] - monthly_sales[ 'BPO' ] ) / monthly_sales[ 'BPO' ] * 100
    series = pd.Series()
    grouped = monthly_sales[[ 'DealerCode', 'Month', 'PctAttainment' ]].groupby( 'DealerCode' )
    for name, group in grouped:
        cum_avg = group.PctAttainment.expanding().mean()
        series = pd.concat( [ series, cum_avg ] )
    monthly_sales[ 'PctAttainmentHistoricalAvg' ] = series

    # mom and yoy change
    pct_change_cols = [ 'AwardAmount', 'LeadType_NATIVE', 'LeadType_3RD_PARTY', '3rdPartyLeads', 'NativeLeads', '3rdPartyLeadConversionRate', 'NativeLeadConversionRate', 'DaysToCloseLead', 'SalesSegmentD', 'SalesSegmentA', 'SalesSegmentB', 'SalesSegmentC', 'Domestic Car SAAR', 'Import Car SAAR', 'Car SAAR', 'Domestic Truck SAAR', 'Import Truck SAAR', 'Light Truck SAAR', 'Domestic Lt Vehicle SAAR', 'Import Lt Vehicle SAAR', 'Lt Vehicle SAAR', 'PPI', 'CPI', 'Fuel', 'NumMaturingLease', 'NumMaturingLoan', 'NumMaturingContractsAll', 'NumEarlyLeasePayoff', 'NumEarlyLoanPayoff', 'NumRegLeasePayoff', 'NumRegLoanPayoff', 'NumPayoffAll', 'NumEarlyPayoff', 'NumRegPayoff', 'NumLeasePayoff', 'NumLoanPayoff', 'Loy_Tot_Amnt', 'Loy_Lea_Ct', 'Loy_Lea_Amnt', 'Loy_Loa_Ct', 'Loy_Loa_Amnt', 'Loy_Cas_Ct', 'Loy_Cas_Amnt', 'Con_Tot_Amnt', 'Con_Lea_Ct', 'Con_Lea_Amnt', 'Con_Loa_Ct', 'Con_Loa_Amnt', 'Con_Cas_Ct', 'Con_Cas_Amnt', 'Cnt50', 'Cnt60', 'TurnRate', 'DealerDaySupply', 'PctAttainment', 'TotalCompetitorAdSpend', 'Tier1TotalAdSpend', 'Tier2TotalAdSpend' ]
    for change_period in [ 1, 3, 12 ]:
        pct_change = monthly_sales.groupby( 'DealerCode' )[ pct_change_cols ].pct_change( change_period )
        pct_change = pct_change.replace( np.inf, np.nan, axis=1 ).fillna( 0 )
        pct_change = pct_change.rename( columns=lambda x: '{n}MonthPctChange{v}'.format( n=change_period, v=x ) )

        monthly_sales[ pct_change.columns ] = pct_change

    # % of total
    dag_ad_cols = [ c for c in monthly_sales.columns if c.startswith( 'DAG' ) and c.endswith( 'AdSpend' ) and c != 'DAGTotalAdSpend' ]
    pct_total = monthly_sales[ dag_ad_cols ].div( monthly_sales[ 'DAGTotalAdSpend' ], axis='index' ).fillna(0).rename( columns=lambda c: c + 'AsPctOfTotal' )
    monthly_sales[ pct_total.columns ] = pct_total

    tier1_ad_cols = [ c for c in monthly_sales.columns if c.startswith( 'Tier1' ) and c.endswith( 'AdSpend' ) and c != 'Tier1TotalAdSpend' ]
    pct_total = monthly_sales[ tier1_ad_cols ].div( monthly_sales[ 'Tier1TotalAdSpend' ], axis='index' ).fillna(0).rename( columns=lambda c: c + 'AsPctOfTotal' )
    monthly_sales[ pct_total.columns ] = pct_total

    tier2_ad_cols = [ c for c in monthly_sales.columns if c.startswith( 'Tier2' ) and c.endswith( 'AdSpend' ) and c != 'Tier2TotalAdSpend' ]
    pct_total = monthly_sales[ tier2_ad_cols ].div( monthly_sales[ 'Tier2TotalAdSpend' ], axis='index' ).fillna(0).rename( columns=lambda c: c + 'AsPctOfTotal' )
    monthly_sales[ pct_total.columns ] = pct_total

    comp_ad_cols = [ c for c in monthly_sales.columns if c.startswith( 'Competitor' ) and c.endswith( 'AdSpend' ) ]
    pct_total = monthly_sales[ comp_ad_cols ].div( monthly_sales[ 'TotalCompetitorAdSpend' ], axis='index' ).fillna(0).rename( columns=lambda c: c + 'AsPctOfTotal' )
    monthly_sales[ pct_total.columns ] = pct_total

    comp_ad_cols = [ 'AcuraAdSpend', 'AudiAdSpend', 'BMWAdSpend', 'InfinitiAdSpend', 'JaguarAdSpend', 'LandRoverAdSpend', 'LexusAdSpend', 'MercedesAdSpend', 'MiniAdSpend', 'PorscheAdSpend', 'SaabAdSpend', 'VolvoAdSpend' ]
    pct_total = monthly_sales[ comp_ad_cols ].div( monthly_sales[ 'TotalCompetitorAdSpend' ], axis='index' ).fillna(0).rename( columns=lambda c: c + 'AsPctOfTotal' )
    monthly_sales[ pct_total.columns ] = pct_total

    # add in end of quarter indicator
    monthly_sales[ 'IsEndOfQuarter' ] = ( monthly_sales.Month.apply( lambda x: x.month in [ 3, 6, 9, 12 ] ) ).astype( 'int' )

    return monthly_sales

def split_train( monthly_sales, data_start, data_end, selection_size=.1 ):
    """Takes dataframe of all data, splits it into """

    # figure out how many months should be withheld for the selection set (test set should always be 3 months)
    selection_months = ( monthly_sales.loc[ ( monthly_sales.Month >= data_start ) & ( monthly_sales.Month <= data_end ) ].groupby( 'DealerCode' ).Month.count() * selection_size ).reset_index().rename( columns={ 'Month': 'selection_months' } )
    selection_months.selection_months = selection_months.selection_months.apply( lambda x: int(min([6,x])))
    monthly_sales = monthly_sales.merge( selection_months, how='left', on='DealerCode' )
    monthly_sales[ 'last_train_month' ] = monthly_sales.selection_months.apply( lambda x: data_end - 3 - x )

    # create train, selection, and test sets
    train_set = monthly_sales.loc[ monthly_sales.Month <= monthly_sales.last_train_month ].reset_index( drop=True )
    selection_set = monthly_sales.loc[ monthly_sales.Month <= data_end - 3 ].reset_index( drop=True )
    test_set = monthly_sales.loc[:].reset_index( drop=True )

    # distinguish selection/test rows
    selection_set[ 'is_selection' ] = selection_set.apply( lambda row: row.Month > row.last_train_month, axis=1 ).astype( 'int' )
    test_set[ 'is_test' ] = test_set.Month.apply( lambda x: x > ( data_end - 3 ) ).astype( 'int' )

    # drop helper columns
    train_set = train_set.drop( [ 'selection_months', 'last_train_month' ], axis=1 )
    selection_set = selection_set.drop( [ 'selection_months', 'last_train_month' ], axis=1 )
    test_set = test_set.drop( [ 'selection_months', 'last_train_month' ], axis=1 )

    return train_set, selection_set, test_set

def add_seasonality( train_set, selection_set, test_set ):
    """Calculate seasonality variables"""

    # get target variables
    target_vars = [ c for c in train_set.columns if c == 'Sales' or c.endswith( '_pct' ) ]

    # fix date column for mike's wrapper
    train_set.Month = train_set.Month.dt.to_timestamp()
    selection_set.Month = selection_set.Month.dt.to_timestamp()
    test_set.Month = test_set.Month.dt.to_timestamp()

    # make sure no existing seasonality columns
    seasonal_cols = [ c + '__seasonality' for c in target_vars ]
    train_set = train_set.drop( seasonal_cols, errors='ignore' )
    selection_set = selection_set.drop( seasonal_cols, errors='ignore' )
    test_set = test_set.drop( seasonal_cols, errors='ignore' )

    # loop through each target variable to add seasonality
    progress = eureqa_utils.Progress( len( target_vars ), name='Seasonality progress' )
    for col in target_vars:
        seasonal_col = col + '__seasonality'

        # convert dataframes to Mike's TimeSeriesData wrapper
        train_ts = TimeSeriesData( data=train_set, tcol='Month', ycol=col, gcol='DealerCode' )
        selection_ts = TimeSeriesData( data=selection_set, tcol='Month', ycol=col, gcol='DealerCode' )
        test_ts = TimeSeriesData( data=test_set, tcol='Month', ycol=col, gcol='DealerCode' )

        # learn seasonal trend on training set
        preparer = TimeTrendCreator( enable_group_models=True )
        preparer.fit( train_ts )

        # add trained seasonal trend to all data
        train_set[ seasonal_col ] = preparer.transform( train_ts ).data[ seasonal_col ]
        selection_set[ seasonal_col ] = preparer.transform( selection_ts ).data[ seasonal_col ]
        test_set[ seasonal_col ] = preparer.transform( test_ts ).data[ seasonal_col ]
        
        progress.increment()
    progress.finish()

    # turn datetime back into period
    train_set.Month = train_set.Month.dt.to_period( 'M' )
    selection_set.Month = selection_set.Month.dt.to_period( 'M' )
    test_set.Month = test_set.Month.dt.to_period( 'M' )

    return train_set, selection_set, test_set

def final_prep( train_set, selection_set, test_set, data_start, data_end ):
    """Final cleanup of data for running searches"""

    # cut off data after 2015 and before 2013
    train_set = train_set.loc[ ( train_set.Month >= data_start ) & ( train_set.Month <= data_end ) ]
    selection_set = selection_set.loc[ ( selection_set.Month >= data_start ) & ( selection_set.Month <= data_end ) ]
    test_set = test_set.loc[ ( test_set.Month >= data_start ) & ( test_set.Month <= data_end ) ]

    # convert month column to datetime object
    train_set.Month = train_set.Month.dt.to_timestamp()
    selection_set.Month = selection_set.Month.dt.to_timestamp()
    test_set.Month = test_set.Month.dt.to_timestamp()

    return train_set, selection_set, test_set


#########################
# Prep individual files #
#########################

def prep_prs():
    """
    Prep pure retail sales data (takes ~30 seconds to run)

    As of 8/24/2016
    - Date ranges: Jan 2011 - Jul 2016
    """

    # get historical prs data
    prs_old_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Pure Retail Sales 2011 to 20160609 extrac 20160609.txt'
    prs_old_data = pd.read_csv( prs_old_file, sep='\t', parse_dates=[ 'ReportedDate' ], infer_datetime_format=True, usecols=[ 'VIN', 'ReportedDate', 'KOS', 'ModelYear', 'ModelCode', 'DealerCode', 'SaleType', 'Sales' ] )

    # get June 2016 pure retail sales data
    prs_jun_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Sales Data 2016-06.xlsx'
    prs_jun_data = pd.read_excel( prs_jun_file, usecols=[ 'VIN', 'SoldDate', 'KOS', 'ModelYear', 'ModelCode', 'DealerCode', 'SaleType', 'Sales' ] )
    prs_jun_data = prs_jun_data.rename( columns={ 'SoldDate': 'ReportedDate' } )

    # get July 2016 pure retail sales data
    prs_jul_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Sales Data 2016-07.xlsx'
    prs_jul_data = pd.read_excel( prs_jul_file, usecols=[ 'VIN', 'SoldDate', 'KOS', 'ModelYear', 'ModelCode', 'DealerCode', 'SaleType', 'Sales' ] )
    prs_jul_data = prs_jul_data.rename( columns={ 'SoldDate': 'ReportedDate' } )

    # combine files together
    prs_data = pd.concat( [ prs_old_data, prs_jun_data, prs_jul_data ], ignore_index=True )

    # some data cleaning
    prs_data[ 'Month' ] = prs_data.ReportedDate.dt.to_period( 'M' )
    prs_data            = prs_data.loc[ ( prs_data.KOS.isin( [ 0, 2, 3 ] ) ) & ( prs_data.SaleType == 'Pure Retail' ) ]
    prs_data[ 'Sales' ] = prs_data.Sales.fillna( 0 )
    prs_data            = prs_data.drop( [ 'ReportedDate', 'KOS', 'SaleType' ], axis=1 )

    # get car segment
    modelcode_lookup = get_modelcode_lookup()
    prs_data         = prs_data.merge( modelcode_lookup, how='inner', on='ModelCode' ) # inner join to remove model codes that are from audi mobility (ie corporate)

    # create car segment dummies
    prs_data = pd.get_dummies( prs_data, columns=[ 'Segment' ] ).rename( columns={ 'Segment_A': 'SalesSegmentA', 'Segment_B': 'SalesSegmentB', 'Segment_C': 'SalesSegmentC', 'Segment_D': 'SalesSegmentD' } )
    for col in [ 'SalesSegmentA', 'SalesSegmentB', 'SalesSegmentC', 'SalesSegmentD' ]:
        prs_data[ col ] = prs_data[ col ] * prs_data.Sales

    # get dealer dag/area/region
    dag_lookup    = get_dag_lookup()
    dealer_lookup = get_dealer_lookup()[[ 'DealerCode', 'Region', 'Area' ]]
    prs_data      = prs_data.merge( dag_lookup, how='left', on='DealerCode' )
    prs_data      = prs_data.merge( dealer_lookup, how='inner', on='DealerCode' ) # inner join to only keep active dealers

    # only keep dealers in eastern region
    prs_data = prs_data.loc[ prs_data.Region == 'Eastern' ].reset_index( drop=True )

    return prs_data

def prep_bpo():
    """
    Prep BPO data (takes ~1 second to run)

    As of 8/24/2016:
    - Date ranges: Jan 2011 - Dec 2016
    """
    
    # get bpo data
    bpo_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/BPO 2011 to 2016 fixed.txt'
    bpo_data = pd.read_csv( bpo_file, sep='\t' )

    # some initial data cleaning
    bpo_data = bpo_data.rename( columns={ '\xef\xbb\xbfegion': 'Region', 'Code': 'DealerCode' } )
    bpo_data.drop( [ 'Region', 'Area', 'Dealer', 'State', 'Carline' ], axis=1 )

    # get overall bpo by dealership
    bpo_horiz = bpo_data.groupby( 'DealerCode' ).sum().reset_index()

    # unmelt the pivot table
    bpo_vert = pd.melt( bpo_horiz, id_vars=[ 'DealerCode' ], value_vars=bpo_horiz.columns[1:].tolist(), var_name='Month', value_name='BPO' )

    # fix date column and resort rows
    bpo_vert[ 'Month' ] = pd.to_datetime( bpo_vert.Month, format='%m/%d/%Y' ).dt.to_period( 'M' )
    bpo_vert = bpo_vert.sort_values( [ 'DealerCode', 'Month' ] ).reset_index( drop=True )

    # fill nan BPO with 0
    bpo_vert[ 'BPO' ] = bpo_vert[ 'BPO' ].fillna( 0 )

    return bpo_vert

def prep_saar():
    """
    Prep SAAR data (takes <1 second to run)

    As of 8/24/2016:
    - Date ranges: Jan 2001 - Dec 2016
    """

    data_folder = '/Users/jessicalin/Documents/Projects/Audi/data/Data'

    saar_data = pd.DataFrame()
    for year in range( 2004, 2017 ):

        # get yearly data file
        saar_file = '{folder}/{year} SAAR.XLS'.format( folder=data_folder, year=year )
        year_data = pd.read_excel( saar_file, skiprows=[ 0,1,2,4 ], skip_footer=6, names=[ 'Month', 'Domestic Car SAAR', 'Import Car SAAR', 'Car SAAR', 'Domestic Truck SAAR', 'Import Truck SAAR', 'Light Truck SAAR', 'Domestic Lt Vehicle SAAR', 'Import Lt Vehicle SAAR', 'Lt Vehicle SAAR' ] )

        # convert date column
        year_data[ 'Month' ] = year_data.apply( lambda row: pd.to_datetime( '{m}-{y}'.format( m=row.Month, y=year ), format='%B-%Y' ), axis=1 ).dt.to_period( 'M' )

        # add year data to full data
        saar_data = pd.concat( [ saar_data, year_data ], ignore_index=True )

    return saar_data

def prep_tier1_traffic():
    """
    Prep website traffic data for AudiUSA.com (takes ~4 seconds to run)

    As of 8/24/2016
    - Date ranges: Jun 2013 - May 2016
    """

    def column_rename( col ):
        hyphen_split = col.split( ' - ' )
        plus_split = col.split( ' + ' )
        if len( hyphen_split ) > 1:
            return 'Tier 1 - ' + hyphen_split[-1]
        elif len( plus_split ) > 1:
            plus_split = col.split( ' + ' )
            return 'Tier 1 - ' + plus_split[-1]
        else:
            return col

    data_folder = '/Users/jessicalin/Documents/Projects/Audi/data/Data'

    tier1_data = pd.DataFrame()
    for year in range( 2013, 2017 ):

        # get year website file
        website_file = '{folder}/Tier 1 - month_inventory_body_{year}.xlsx'.format( folder=data_folder, year=year )
        year_data    = pd.read_excel( website_file, skiprows=9 )

        # filter to only monthly total rows
        year_data = year_data.loc[ year_data[ 'Item.1' ] == 'Total' ]

        # convert date column
        year_data[ 'Month' ] = pd.to_datetime( year_data.Item, format='%b %Y' ).dt.to_period( 'M' )

        # fix column names
        year_data = year_data.rename( columns=lambda col: column_rename( col ) )

        # remove unneccessary columns
        year_data = year_data.drop( [ 'Dimension', 'Item', 'Dimension.1', 'Item.1', 'Dimension.2', 'Item.2' ], axis=1 )
        year_data = year_data.drop( [ c for c in year_data.columns if c.endswith( '(Percent)' ) ], axis=1 )

        # add year data to full tier 1 data
        tier1_data = pd.concat( [ tier1_data, year_data ], ignore_index=True )

    return tier1_data

def prep_tier2_traffic():
    """
    Prep website traffic data for tier 2 data

    Ignoring/stopping prep for now because data only goes back to 2014/2015
    """

    # get data - takes ~2min
    tier2_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Tier2 and 3 Website traffic - AoA_Data_Pull_Data_05312016.xlsx'
    tier2_web_traffic  = pd.read_excel( tier2_file, sheetname='Web Traffic' )
    tier2_web_referral = pd.read_excel( tier2_file, sheetname='Web Referral' )
    tier2_web_content  = pd.read_excel( tier2_file, sheetname='Web Content' )
    tier2_phone_leads  = pd.read_excel( tier2_file, sheetname='Daily Phone Leads' )

    # trim to only tier 2 data
    tier2_web_traffic  = tier2_web_traffic.loc[ tier2_web_traffic.Tier == 'Tier II' ]
    tier2_web_referral = tier2_web_referral.loc[ tier2_web_referral[ 'Tier II' ] == 'Tier II' ]
    tier2_web_content  = tier2_web_content.loc[ tier2_web_content.Tier == 'Tier II' ]

    # convert date column
    tier2_web_traffic[ 'Month' ]  = tier2_web_traffic[ 'Month' ].dt.to_period( 'M' )
    tier2_web_referral[ 'Month' ] = tier2_web_referral[ 'Month' ].dt.to_period( 'M' )
    tier2_web_content[ 'Month' ]  = tier2_web_content[ 'Month' ].dt.to_period( 'M' )
    tier2_phone_leads[ 'Month' ]  = tier2_phone_leads[ 'Year Month' ].dt.to_period( 'M' )

    # drop unneeded columns
    tier2_web_traffic  = tier2_web_traffic.drop(  [ 'Dealer Code', 'Tier' ],    axis=1 )
    tier2_web_referral = tier2_web_referral.drop( [ 'Dealer Code', 'Tier II' ], axis=1 )
    tier2_web_content  = tier2_web_content.drop(  [ 'Dealer Code', 'Tier' ],    axis=1 )
    tier2_phone_leads  = tier2_phone_leads.drop(  [ 'Year Month' ],             axis=1 )

    # create dag lookup from web content
    dag_lookup = tier2_web_content.drop_duplicates( subset=[ 'Account Name', 'Account ID' ] )[[ 'Account Name', 'Account ID' ]]
    dag_lookup[ 'DAG' ] = dag_lookup[ 'Account ID' ].str.split( '_' ).str.get( 1 )
    dag_lookup = dag_lookup.loc[ ~dag_lookup.DAG.isin( [ 'demotier2', 'default' ] ) ]
    dag_lookup.DAG = dag_lookup.DAG.astype( 'int' )

def prep_econ():
    """
    Prep economic data (takes <1 second to run)

    As of 8/24/2016:
    - Date ranges: Jan 2011 - Jul 2016
    """
    
    # get data
    try:
        econ_data = quandl.get(["FRED/CPIAUCSL","FRED/PPIACO","EIA/PET_RWTC_M"], authtoken="U5LMkT7WF1MEvD1zqBxG", collapse="monthly", start_date="2011-01-01").reset_index()
        econ_data = econ_data.rename( columns={ 'FRED/CPIAUCSL - VALUE': 'CPI', 'FRED/PPIACO - VALUE': 'PPI', 'EIA/PET_RWTC_M - Value': 'Fuel', 'DATE': 'Month' } )
        econ_data.to_csv( '/Users/jessicalin/Documents/Projects/Audi/data/Data/econ_data.csv', index=False )
    except:
        econ_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/econ_data.csv'
        econ_data = pd.read_csv( econ_file )
        econ_data = econ_data.rename( columns={ 'DATE': 'Month' } )
        econ_data.Month = pd.to_datetime( econ_data.Month, format='%Y-%m-%d' )

    # convert date column
    econ_data[ 'Month' ] = econ_data[ 'Month' ].dt.to_period( 'M' )
    
    return econ_data

def prep_afs( prs_data ):
    """
    Prep AFS data (takes ~1min to run)

    As of 8/24/2016:
    - Date ranges: Apr 2009 - Apr 2016
    """

    def calc_maturing( afs_data ):
        """Calculate number of contracts maturing by month by dealer"""

        # calculate contract maturity month
        afs_data[ 'ContractEndMonth' ] = ( afs_data.ContractDate + afs_data.ContractTerm.values.astype( 'timedelta64[M]' ) ).dt.to_period( 'M' )

        # count matured contracts by month by finance type
        mature_by_month = afs_data.groupby( [ 'DealerCode', 'FinanceType', 'ContractEndMonth' ] )[ 'VIN' ].count().reset_index()

        # pivot finance types into columns
        mature_by_month_pv = mature_by_month.pivot_table( index=[ 'DealerCode', 'ContractEndMonth' ], columns='FinanceType', values='VIN' ).reset_index()
        mature_by_month_pv = mature_by_month_pv.fillna( 0 )
        mature_by_month_pv[ 'NumMaturingContractsAll' ] = mature_by_month_pv.LEASE + mature_by_month_pv.RETAIL

        # rename columns
        mature_by_month_pv = mature_by_month_pv.rename( columns={ 'ContractEndMonth': 'Month', 'LEASE': 'NumMaturingLease', 'RETAIL': 'NumMaturingLoan' } )

        return mature_by_month_pv

    def calc_paid_off( afs_data ):
        """Calculate number of contracts paid off by month by dealer"""

        # filter to only paid off loans and use AccountStatusChangeDate as month of payoff
        is_paid = afs_data.loc[ afs_data.AccountStatus.isin( [ 'NPMT', 'NPER' ] ) ]
        is_paid.loc[ :, 'Month' ] = is_paid.AccountStatusChangeDate.dt.to_period( 'M' )

        # count paid off contracts by month by early/regular payment and finance type
        paid_by_month = is_paid.groupby( [ 'DealerCode', 'AccountStatus', 'FinanceType', 'Month' ] ).VIN.count().reset_index()
        paid_by_month[ 'Combined' ] = paid_by_month.AccountStatus.str.cat( paid_by_month.FinanceType, sep=' - ' )

        # pivot payment type and finance type into columns
        paid_by_month_pv = paid_by_month.pivot_table( index=[ 'DealerCode', 'Month' ], columns='Combined', values='VIN' ).reset_index()
        paid_by_month_pv = paid_by_month_pv.fillna( 0 )

        # calculate overall by payment type and finance type
        paid_by_month_pv[ 'NumPayoffAll' ]    = paid_by_month_pv[ 'NPER - LEASE' ]   + paid_by_month_pv[ 'NPER - RETAIL' ] + paid_by_month_pv[ 'NPMT - LEASE' ] + paid_by_month_pv[ 'NPMT - RETAIL' ]
        paid_by_month_pv[ 'NumEarlyPayoff' ]  = paid_by_month_pv[ 'NPER - LEASE' ]   + paid_by_month_pv[ 'NPER - RETAIL' ]
        paid_by_month_pv[ 'NumRegPayoff' ]    = paid_by_month_pv[ 'NPMT - LEASE' ]   + paid_by_month_pv[ 'NPMT - RETAIL' ]
        paid_by_month_pv[ 'NumLeasePayoff']   = paid_by_month_pv[ 'NPER - LEASE' ]    + paid_by_month_pv[ 'NPMT - LEASE' ]
        paid_by_month_pv[ 'NumLoanPayoff']    = paid_by_month_pv[ 'NPER - RETAIL' ]   + paid_by_month_pv[ 'NPMT - RETAIL' ]

        # rename columns
        paid_by_month_pv = paid_by_month_pv.rename( columns={ 'NPER - LEASE': 'NumEarlyLeasePayoff', 'NPER - RETAIL': 'NumEarlyLoanPayoff', 'NPMT - LEASE': 'NumRegLeasePayoff', 'NPMT - RETAIL': 'NumRegLoanPayoff' } )

        return paid_by_month_pv

    def finance_type_by_vin( afs_data ):
        """Break out finance type by vin"""

        # keep only vin and finance type columns
        finance_type = afs_data[[ 'VIN', 'FinanceType' ]]

        # fix strings
        finance_type.FinanceType = finance_type.FinanceType.str.capitalize()

        return finance_type

    # get data
    afs_file = '/Users/jessicalin/Documents/Projects/Audi/data/unp afs_data contract date 2008 to 04302016.csv'
    afs_data = pd.read_csv( afs_file, usecols=[ 'VIN', 'ContractDate', 'AccountStatusChangeDate', 'AccountStatus', 'FinanceTypeName', 'ContractTerm' ] )
    afs_data = afs_data.drop_duplicates( subset=[ 'VIN' ] )
    afs_data = afs_data.rename( columns={ 'FinanceTypeName': 'FinanceType' } )

    # convert date column
    afs_data.ContractDate            = pd.to_datetime( afs_data.ContractDate, format='%m/%d/%Y %H:%M')
    afs_data[ 'Month' ]              = afs_data.ContractDate.dt.to_period( 'M' )
    afs_data.AccountStatusChangeDate = pd.to_datetime( afs_data.AccountStatusChangeDate, format='%m/%d/%Y %H:%M' )

    # convert balloon --> retail loans
    afs_data.FinanceType = afs_data.FinanceType.replace( 'BALLOON', 'RETAIL' )

    # get dealer code by VIN - take last dealer code associated with each vin in prs file
    last_dealer_by_vin = prs_data.groupby( 'VIN' ).DealerCode.last().reset_index() # assumes prs is already ordered (sort_values command takes ~5 min to run)
    afs_data = afs_data.merge( last_dealer_by_vin, how='inner', on='VIN' ) # inner join because some VINs in afs file seem to be for non-pure retail sales

    # created calculated metrics
    mature_by_month = calc_maturing( afs_data )
    paid_by_month   = calc_paid_off( afs_data )
    finance_type    = finance_type_by_vin( afs_data )

    return mature_by_month, paid_by_month, finance_type

def prep_leads_to_sales():
    """Prep leads to sales data (takes ~21 seconds to run)

    As of 8/29/2016:
    - Date ranges: Jan 2013 - July 2016
    """

    def calc_leads_by_month( leads_data ):
        """Calculate leads and conversion rate by month by dealer by lead provider"""

        # calculate leads and conversions by dealer/lead provider/month
        agg_leads = leads_data.groupby( [ 'DealerCode', 'LeadType', 'Month' ] ).agg( { 'SALE': 'mean', 'Model': 'count' } ).reset_index()
        pv_leads  = agg_leads.pivot_table( index=[ 'DealerCode', 'Month' ], columns='LeadType', values=[ 'Model', 'SALE' ] )

        # fix the column multiindex
        pv_leads.columns = [ '_'.join( col ) for col in pv_leads.columns ]
        pv_leads         = pv_leads.reset_index().rename( columns={ 'Model_3RD_PARTY': '3rdPartyLeads', 'Model_NATIVE': 'NativeLeads', 'SALE_3RD_PARTY': '3rdPartyLeadConversionRate', 'SALE_NATIVE': 'NativeLeadConversionRate' } )

        # fill nan with 0
        pv_leads = pv_leads.fillna( 0 )

        return pv_leads

    def prep_sale_leads( leads_data ):
        """Prep only the leads that resulted in a sale"""

        # filter to only leads that resulted in a sale
        sale_leads = leads_data.loc[ ~pd.isnull( leads_data.VIN ), [ 'LeadType', 'DealerCode', 'DaysToCloseLead', 'VIN', 'Month' ] ]
        sale_leads = sale_leads.drop_duplicates( subset=[ 'VIN', 'DealerCode', 'Month' ] )

        # clean up & dummi-ize the LeadProvider column
        sale_leads = pd.get_dummies( sale_leads, columns=[ 'LeadType' ] )

        return sale_leads

    # get leads to sales data
    leads_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/unp January 2013 to July 2016 Leads to Sales.csv'
    leads_data = pd.read_csv( leads_file, usecols=[ 'LEAD DATE', 'LEADGROUP', 'LEADDEALER', 'Lead Requested Vehicle', 'DAYS TO CLOSE', 'Sale VIN', 'SALE' ] )

    # some clean up
    leads_data.LEADDEALER = leads_data.LEADDEALER.replace( { '4.01E+55': '401E53', '4.09E+06': '409E04', '4.19E+04': '419E02', '4.19E+09': '419E07', '4.02E+52': '402E50', '4.01E+09': '401E07', '4.09E+51': '409E49', '4.23E+101': '423E99', '4.19E+08': '419E06', '4.09E+52': '409E50', '4.23E+99': '423E97', '4.24E+04': '424E02' } )
    leads_data[ 'Lead Requested Vehicle' ] = leads_data[ 'Lead Requested Vehicle' ].replace( { 'allroad': 'A4', 'RS5': 'A5', 'S7': 'A7', 'S6': 'A6', 'S5': 'A5', 'S4': 'A4', 'S8': 'A8', 'A3 e-tron': 'A3', 'RS7': 'A7', 'RS6': 'A6', 'RS4': 'A4', 'S3': 'A3', 'm90': 'Unknown' } )
    leads_data[ 'Month' ] = pd.to_datetime(leads_data[ 'LEAD DATE' ], format='%m/%d/%Y' ).dt.to_period( 'M' )
    leads_data.SALE = leads_data.SALE.fillna( 'N' )
    leads_data.SALE = leads_data.SALE.replace( { 'N': 0, 'Y': 1 } ).astype( 'int' )
    leads_data      = leads_data.rename( columns={ 'LEADGROUP': 'LeadType', 'LEADDEALER': 'DealerCode', 'Lead Requested Vehicle': 'Model', 'Sale VIN': 'VIN', 'DAYS TO CLOSE': 'DaysToCloseLead' } )
    leads_data      = leads_data.drop( [ 'LEAD DATE' ], axis=1 )

    # concat 3rd party leads
    leads_file    = '/Users/jessicalin/Documents/Projects/Audi/data/unp native and 3rd party leads 0101013-033116.csv'
    outside_leads = pd.read_csv( leads_file, usecols=[ 'EventDate', 'LEADGROUP', 'LEADDEALER', 'REQ_MAKE', 'REQ_MODEL', 'DAYS TO CLOSE', 'VIN', 'SALE' ] )

    # some clean up
    outside_leads = outside_leads.loc[ ( outside_leads.LEADGROUP == '3RD PARTY' ) & ( outside_leads.REQ_MAKE == 'Audi' ) ]
    outside_leads[ 'Month' ] = pd.to_datetime( outside_leads.EventDate, format='%m/%d/%Y 0:00' ).dt.to_period( 'M' )
    outside_leads.LEADGROUP  = outside_leads.LEADGROUP.str.replace( ' ', '_' )
    outside_leads.LEADDEALER = outside_leads.LEADDEALER.replace( { '4.01E+55': '401E53', '4.09E+06': '409E04', '4.19E+04': '419E02', '4.19E+09': '419E07', '4.02E+52': '402E50', '4.01E+09': '401E07', '4.09E+51': '409E49', '4.23E+101': '423E99', '4.19E+08': '419E06', '4.09E+52': '409E50', '4.23E+99': '423E97', '4.24E+04': '424E02' } )
    outside_leads.REQ_MODEL  = outside_leads.REQ_MODEL.str.upper().replace( 
        { 
            'S3': 'A3',
            'A3 CABRIOLET': 'A3', 
            'A3 CONVERTIBLE': 'A3', 
            'A3 E-TRON': 'A3', 
            'A3 SPORTBACK E-TRON': 'A3',
            'A3 SPORTBACK': 'A3', 
            'ALLROAD': 'A4', 
            'A4 SEDAN': 'A4', 
            'A4 WAGON': 'A4', 
            'S4': 'A4', 
            'A5 CABRIOLET': 'A5', 
            'A5 COUPE': 'A5', 
            'S5': 'A5', 
            'RS 5': 'A5', 
            'RS5': 'A5', 
            'RS 5 CABRIOLET': 'A5', 
            'S5 CABRIOLET': 'A5', 
            'A6 HYBRID': 'A6', 
            'S6': 'A6', 
            'S7': 'A7', 
            'RS7': 'A7', 
            'RS 7': 'A7', 
            'RS 7 PERFORMANCE': 'A7',
            'S8': 'A8', 
            'A8 L': 'A8', 
            'A8 L W12': 'A8', 
            'SQ5': 'Q5', 
            'Q5 HYBIRD': 'Q5', 
            'Q5 HYBRID': 'Q5', 
            'TT RS': 'TT', 
            'TT ROADSTER': 'TT', 
            'TTS': 'TT', 
            'TTS ROADSTER': 'TT', 
            'TT COUPE': 'TT', 
            'RS SPYDER': 'R8', 
            'R8 SPYDER': 'R8'
             } )
    outside_leads.SALE = outside_leads.SALE.fillna( 'N' )
    outside_leads.SALE = outside_leads.SALE.replace( { 'N': 0, 'Y': 1 } ).astype( 'int' )
    outside_leads = outside_leads.rename( columns={ 'LEADGROUP': 'LeadType', 'LEADDEALER': 'DealerCode', 'REQ_MODEL': 'Model', 'DAYS TO CLOSE': 'DaysToCloseLead' } )

    outside_leads.drop( [ 'EventDate', 'REQ_MAKE' ], axis=1, inplace=True )

    # combine lead types together
    leads_data = pd.concat( [ leads_data, outside_leads ], ignore_index=True )

    # create calculated data frames
    leads_by_month = calc_leads_by_month( leads_data )
    sale_leads     = prep_sale_leads( leads_data )

    return leads_by_month, sale_leads

def prep_dag_ad_spend():
    """Prep DAG ad spend data (takes ~7 seconds to run)

    As of 8/29/2016:
    - Date ranges: Jan 2012 - Dec 2016
    """

    dag_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/AER Historical DAG Spend 2012-2016_R2 Monthly All Markets.xlsx'
    dag_lookup = get_dag_lookup()

    dag_data = pd.DataFrame()
    for year in range( 2012, 2017 ):

        # get yearly data
        year_data = pd.read_excel( dag_file, sheetname=str(year), skiprows=[ 0,1,2,11,19,27,35,43,51 ] )

        # clean yearly data
        year_data        = year_data.drop( [ 'Unnamed: 0', 'TOTALS' ], axis=1 )
        year_data.Market = year_data.Market.ffill()

        # convert horizontal frame into pivoted vertical frame
        melt_year = pd.melt( year_data, id_vars=[ 'Market', 'Channel' ], value_vars=year_data.columns[2:].tolist(), var_name='Month' )
        pv_year   = melt_year.pivot_table( index=[ 'Market', 'Month' ], columns='Channel', values='value' ).reset_index()

        # fix date column
        pv_year.Month = pv_year.Month.apply( lambda x: pd.to_datetime( '{month}/1/{year}'.format( month=x, year=year ) ).to_period( 'M' ) )

        # get dag id
        pv_year.Market = pv_year.Market.replace( { 'Hartford /\nNew Haven': 'Hartford/New Haven' } )
        pv_year        = pv_year.merge( dag_lookup.drop_duplicates( subset=[ 'DAG', 'DAGName' ] ), how='left', left_on='Market', right_on='DAGName' )
        pv_year        = pv_year.drop( [ 'Market', 'DealerCode', 'DAGName' ], axis=1 )

        # rename columns
        pv_year = pv_year.rename( columns={ 'Digital': 'DAGDigitalAdSpend', 'Events / Sponsorships': 'DAGEventsAdSpend', 'OOH / Stadium Signage': 'DAGStadiumSignageAdSpend', 'Print': 'DAGPrintAdSpend', 'Radio': 'DAGRadioAdSpend', 'Spot TV': 'DAGSpotTVAdSpend', 'TOTAL': 'DAGTotalAdSpend' } )

        # add year data fo full data
        dag_data = pd.concat( [ dag_data, pv_year ], ignore_index=True )

    return dag_data.sort_values( [ 'DAG', 'Month' ] ).reset_index( drop=True )

def prep_polk():
    """Prep polk loyalty data

    As of 10/13/2016:
    - Date ranges: Jan 2011 - July 2016
    """

    # # get polk data
    # polk_file = '/Users/jessicalin/Documents/Projects/Audi/data/polk_loyalty.csv'
    # polk_data = pd.read_csv( polk_file, usecols=[ 'Calendar Year', 'Month', 'DealerCode', 'Target Vehicle Finance Type', 'Target Vehicle Financial Institution Type', 'RTM', 'Dealer Loyal', 'Dealer Loyal %' ] )

    # # clean column names and values
    # polk_data = polk_data.rename( columns={ 'Calendar Year': 'Year', 'Target Vehicle Finance Type': 'PolkFinanceType', 'Target Vehicle Financial Institution Type': 'PolkFinancialInstitutionType', 'RTM': 'PolkRTM', 'Dealer Loyal': 'PolkDealerLoyal' } )
    # polk_data.DealerCode = polk_data.DealerCode.replace( { '4.19E+04': '419E02', '4.02E+52': '402E50', '4.01E+55': '401E53', '4.23E+101': '423E99', '4.01E+09': '401E07', '4.09E+51': '409E49', '4.19E+08': '419E06', '4.09E+06': '409E04', '4.19E+09': '419E07' } )

    # # convert date to month
    # polk_data[ 'Day' ]   = 1
    # polk_data[ 'Month' ] = pd.to_datetime( polk_data[[ 'Year', 'Month', 'Day' ]] ).dt.to_period( 'M' )
    # polk_data            = polk_data.drop( [ 'Year', 'Day' ], axis=1 )

    # # pivot values into columns
    # pv_polk         = polk_data.pivot_table( index=[ 'DealerCode', 'Month' ], values=[ 'PolkRTM', 'PolkDealerLoyal' ], columns=[ 'PolkFinanceType', 'PolkFinancialInstitutionType' ] ).fillna( 0 )
    # pv_polk.columns = [ '_'.join(col).strip() for col in pv_polk.columns.values ]
    # pv_polk         = pv_polk.reset_index()

    # read in data
    polk_file = '/Users/jessicalin/Downloads/Polk_Dealer_Loyalty_full.txt'
    polk_data = pd.read_csv( polk_file )

    # fix columns
    polk_data[ 'Month' ] = pd.to_datetime( polk_data.Date, format='%Y-%m' ).dt.to_period( 'M' )
    polk_data = polk_data.drop( [ 'Date', 'Region', 'Area', 'SubArea' ], axis=1 )
    polk_data = polk_data.rename( columns={ 'Dealer': 'DealerCode', 'Dealer Loyal': 'PolkDealerLoyal', 'Dealer Loyal %': 'PolkDealerLoyalPct', 'Finance Type Loyal': 'PolkFinanceTypeLoyal', 'Finance Type Loyal %': 'PolkFinanceTypeLoyalPct', 'Return To Market': 'PolkReturnToMarket' } )

    return polk_data

def prep_loyalty():
    """Prep loyalty data - join against prs & afs to get IsLoyalty vs IsConquest

    As of 10/6/2016:
    - Date ranges: Jan 2011 - Sep 2016
    """

    # get loyalty payment data
    loyalty_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Loyalty_Data_full.txt'
    loyalty_data = pd.read_csv( loyalty_file, thousands=',', usecols=[ 'VIN', 'AWARD_AMNT' ] )

    # rename column
    loyalty_data = loyalty_data.rename( columns={ 'AWARD_AMNT': 'AwardAmount' } )

    # get loyalty payment stats
    stats_file    = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Loyalty_Data_Dealer_statistics.txt'
    loyalty_stats = pd.read_csv( stats_file )

    # fix columns
    loyalty_stats = loyalty_stats.rename( columns={ 'Dealer_Code': 'DealerCode' } )
    unnamed_cols  = [ c for c in loyalty_stats.columns if 'Unnamed' in c ]
    loyalty_stats = loyalty_stats.drop( unnamed_cols, axis=1 )
    loyalty_stats.Month = pd.to_datetime( loyalty_stats.Month, format='%Y-%m' ).dt.to_period( 'M' )

    return loyalty_data, loyalty_stats

def prep_conquest():
    """Prep conquest data - join against prs & afs to get IsConquest

    As of 10/6/2016:
    - Date ranges: Jan 2011 - Sep 2016
    """

    # get conquest payment data
    conquest_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Conquest_Data_full.txt'
    conquest_data = pd.read_csv( conquest_file, thousands=',', usecols=[ 'VIN', 'AWARD_AMNT' ] )

    # rename column
    conquest_data = conquest_data.rename( columns={ 'AWARD_AMNT': 'AwardAmount' } )

    # get conquest payment stats
    stats_file  = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Conquest_Data_Dealer_statistics.txt'
    conquest_stats = pd.read_csv( stats_file )

    conquest_stats = conquest_stats.rename( columns={ 'Dealer_Code': 'DealerCode' } )
    unnamed_cols = [ c for c in conquest_stats.columns if 'Unnamed' in c ]
    conquest_stats = conquest_stats.drop( unnamed_cols, axis=1 )
    conquest_stats.Month = pd.to_datetime( conquest_stats.Month, format='%Y-%m' ).dt.to_period( 'M' )

    return conquest_data, conquest_stats

def prep_sale_dimension( finance_type, loyalty_data, conquest_data ):
    """For VINs in afs & loyalty & conquest files, combine the sale and finance type"""

    def merge_loyalty_conquest( loyalty_data, conquest_data ):
        """Merge data from loyalty and conquest files together. Needs to deal with VINs which appear in both files"""

        # set sale types
        loyalty_data[ 'SaleType' ] = loyalty_data.AwardAmount.apply( lambda x: 'LoyaltySale' if x > 0 else 'ConquestSale' )
        conquest_data[ 'SaleType' ] = 'ConquestSale'

        # merge files together
        sale_type = loyalty_data.merge( conquest_data, how='outer', on='VIN' )

        # if only in loyalty file
        sale_type.loc[ ( pd.isnull( sale_type.AwardAmount_y ) ) & ( pd.isnull( sale_type.SaleType_y ) ), 'SaleType' ]    = sale_type.SaleType_x
        sale_type.loc[ ( pd.isnull( sale_type.AwardAmount_y ) ) & ( pd.isnull( sale_type.SaleType_y ) ), 'AwardAmount' ] = sale_type.AwardAmount_x

        # if only in conquest file
        sale_type.loc[ ( pd.isnull( sale_type.AwardAmount_x ) ) & ( pd.isnull( sale_type.SaleType_x ) ), 'SaleType' ]    = sale_type.SaleType_y
        sale_type.loc[ ( pd.isnull( sale_type.AwardAmount_x ) ) & ( pd.isnull( sale_type.SaleType_x ) ), 'AwardAmount' ] = sale_type.AwardAmount_y

        # if in both but conquest award amount == 0
        sale_type.loc[ ( sale_type.AwardAmount_x > 0 ) & ( sale_type.AwardAmount_y == 0 ), 'SaleType' ]    = 'LoyaltySale'
        sale_type.loc[ ( sale_type.AwardAmount_x > 0 ) & ( sale_type.AwardAmount_y == 0 ), 'AwardAmount' ] = sale_type.AwardAmount_x

        # if in both but loyalty award amount == 0
        sale_type.loc[ ( sale_type.AwardAmount_y >= 0 ) & ( sale_type.AwardAmount_x == 0 ), 'SaleType' ]    = 'ConquestSale'
        sale_type.loc[ ( sale_type.AwardAmount_y >= 0 ) & ( sale_type.AwardAmount_x == 0 ), 'AwardAmount' ] = sale_type.AwardAmount_y

        # otherwise, values from loyalty file supersede
        sale_type.loc[ pd.isnull( sale_type.SaleType ), 'AwardAmount' ] = sale_type.AwardAmount_x
        sale_type.loc[ pd.isnull( sale_type.SaleType ), 'SaleType' ]    = sale_type.SaleType_x

        # drop _x and _y merged columns
        sale_type = sale_type.drop( [ 'SaleType_x', 'SaleType_y', 'AwardAmount_x', 'AwardAmount_y' ], axis=1 )

        return sale_type

    # merge loyalty and conquest data together
    sale_type = merge_loyalty_conquest( loyalty_data, conquest_data )

    # merge finance type with loyalty type
    dimension_data             = sale_type.merge( finance_type, how='outer', on='VIN' )
    dimension_data.AwardAmount = dimension_data.AwardAmount.fillna( 0 )
    dimension_data.SaleType    = dimension_data.SaleType.fillna( 'ConquestSale' )
    dimension_data.FinanceType = dimension_data.FinanceType.fillna( 'Cash' )

    # combine loyalty & finance types
    dimension_data[ 'SaleDimension' ] = dimension_data.SaleType.str.cat( dimension_data.FinanceType, sep='_' )

    # create dummies from loyalty & sale types
    dimension_lookup = pd.get_dummies( dimension_data.drop( [ 'SaleType', 'FinanceType' ], axis=1 ), columns=[ 'SaleDimension' ] )
    dimension_lookup = dimension_lookup.rename( columns=lambda x: x.replace( 'SaleDimension_', '' ) )

    return dimension_lookup

def prep_sales_managers():
    """Prep number of sales managers for eastern region"""

    def fix_file( dataframe, value ):
        """Fix the excel file and turn it into rectangular format"""

        # remove the "national" column
        dataframe = dataframe[[ c for c in dataframe.columns if c.startswith( 'Eastern' ) or c == 'Month' ]]

        # unmelt to get years & months as a column
        pv_frame            = pd.melt( dataframe, id_vars='Month', value_vars=[ 'Eastern - 2012', 'Eastern - 2013', 'Eastern - 2014', 'Eastern - 2015', 'Eastern - 2016' ], var_name='Year', value_name=value )
        pv_frame.Year       = pv_frame.Year.str.replace( 'Eastern - ', '' )
        pv_frame[ 'Month' ] = pd.to_datetime( pv_frame.Month.str.cat( pv_frame.Year, sep='-' ), format='%B-%Y' ).dt.to_period( 'M' )

        # final cleanup
        pv_frame = pv_frame.dropna( subset=[ value ] )
        pv_frame = pv_frame.drop( 'Year', axis=1 )

        return pv_frame

    # get excel data
    managers_file     = '/Users/jessicalin/Documents/Projects/Audi/data/Data/ABS_Sales Managers analysis Eastern.xlsx'
    brand_specialists = pd.read_excel( managers_file, skiprows=range(3), skip_footer=16, names=[ 'Month', 'Eastern - 2012', 'National - 2012', 'Eastern - 2013', 'National - 2013', 'Eastern - 2014', 'National - 2014', 'Eastern - 2015', 'National - 2015', 'Eastern - 2016', 'National - 2016' ] )
    sales_managers    = pd.read_excel( managers_file, skiprows=range(19), names=[ 'Month', 'Eastern - 2012', 'National - 2012', 'Eastern - 2013', 'National - 2013', 'Eastern - 2014', 'National - 2014', 'Eastern - 2015', 'National - 2015', 'Eastern - 2016', 'National - 2016' ] )

    # convert excel format to usable format
    brand_specialists = fix_file( brand_specialists, 'BrandSpecialists' )
    sales_managers    = fix_file( sales_managers,    'SalesManagers'    )

    # merge into one frame
    all_sales_managers = brand_specialists.merge( sales_managers, how='outer', on='Month' )
    all_sales_managers[ 'Region' ] = 'Eastern'

    return all_sales_managers

def prep_incentives():
    """Prep incentives file for only the following years: 2013, 2014, 2015, 2016

    As of 9/29/2016:
    - Date ranges: Jan 2010 - Sep 2016
    """

    def fix_column_names( year_data ):
        """Fix the multiindex column headers"""

        col_names = []
        col_names.append( 'Model' )
        col_names.append( 'MY' )

        months = []
        top_names = []

        for c in year_data.columns[2:]:
                
            # get top column name - deals with names merged across columns
            if isinstance( year_data.iloc[0][c], ( str, unicode ) ):
                top_name = year_data.iloc[0][c]
                
            # get labels for names that were NOT merged down rows (ie loyalty - afs)
            if isinstance( year_data.iloc[1][c], ( str, unicode ) ):
                bottom_name = year_data.iloc[1][c]
                bottom_name = bottom_name.replace( '(', '' ).replace( ')', '' )
                col_name    = '{top}_{bottom}'.format( top=top_name, bottom=bottom_name )
                top_names.append( top_name )
            else:
                col_name = top_name
                
            # get rid of spaces in column names
            col_name = col_name.replace( ' ', '' )
                
            # get month in top row
            if isinstance(c, datetime.datetime):
                curr_month = pd.to_datetime( c, format='%Y-%m-%d 00:00:00' ).to_period( 'M' )
                months.append( curr_month )
                
            col_name = '{m} - {c}'.format( c=col_name, m=curr_month )

            col_names.append( col_name )

        # replace column names and drop first 2 rows
        year_data.columns = col_names
        year_data         = year_data.drop( [0,1] ).reset_index( drop=True )
        year_data         = year_data.drop( 'MY', axis=1 )

        # force into floats
        for col in year_data.columns[1:]:
            year_data[ col ] = year_data[ col ].astype( 'float' )

        # combine all grouped columns into one
        for month in months:
            for group_type in top_names:
                group_cols = [ c for c in year_data.columns if c.startswith( '{month} - {type}'.format( month=month, type=group_type ) ) ]
                year_data[ '{month} - {type}'.format( month=month, type=group_type ) ] = year_data[ group_cols ].sum( axis=1 )
                drop_cols  = [ c for c in year_data.columns if c.startswith( '{month} - {type}'.format( month=month, type=group_type ) ) and c != '{month} - {type}'.format( month=month, type=group_type ) ]
                year_data  = year_data.drop( drop_cols, axis=1 )

        return year_data

    def fix_models( year_data ):
        """Collapse all the model names down into A3/A4/A5/etc"""

        # deal with model names that were merged down rows (ie across multiple model years)
        year_data.Model = year_data.Model.ffill()

        # convert model names into model (ie "A3 & S3 Sedan" --> "A3")
        year_data.Model = year_data.Model.replace( { ' Q3': 'Q3', 'Q 7': 'Q7', 'RS7': 'A7', 'S8': 'A8', 'allroad': 'A4', 'TTS': 'TT' }, regex=True )
        year_data.Model = year_data.Model.str.split( ' ' ).str.get( 0 )
        year_data.Model = year_data.Model.str.split( ',' ).str.get( 0 )
        year_data.Model = year_data.Model.str.split( '/' ).str.get( 0 )

        # sum all incentives per model
        year_data = year_data.groupby( 'Model' ).sum().reset_index()

        # sum incentives per segment
        model_lookup = get_modelcode_lookup()
        year_data    = year_data.merge( model_lookup[[ 'Model', 'Segment' ]], how='left', on='Model' )
        segment_incentives         = year_data.groupby( 'Segment' )[ year_data.columns[1:-1] ].sum().reset_index()
        segment_incentives.Segment = segment_incentives.Segment.apply( lambda x: 'Segment' + x )
        segment_incentives         = segment_incentives.rename( columns={ 'Segment': 'Model' } )
        # year_data    = pd.concat( [ year_data, segment_incentives ], ignore_index=True )

        return segment_incentives

    def fix_pivot( year_data ):
        """Pivot so that months and models are rows, and different incentive types are columns"""

        # unpivot so everything is in rows
        pv_incentives = pd.melt( year_data, id_vars='Model', value_vars=year_data.columns[1:].tolist() )

        # ungroup month-incentive column
        pv_incentives[ 'Month' ]     = pv_incentives.variable.str.split( ' - ' ).str.get( 0 )
        pv_incentives[ 'Incentive' ] = pv_incentives.variable.str.split( ' - ' ).str.get( 1 )
        pv_incentives[ 'Month' ]     = pd.to_datetime( pv_incentives.Month, format='%Y-%m' ).dt.to_period( 'M' )

        # get final pivot
        pv_incentives         = pv_incentives.pivot_table( values='value', index='Month', columns=[ 'Model', 'Incentive' ] )
        pv_incentives.columns = [ '_'.join( col ) for col in pv_incentives.columns ]
        pv_incentives         = pv_incentives.rename( columns=lambda x: 'Incentive' + x )
        pv_incentives         = pv_incentives.reset_index().fillna( 0 )

        return pv_incentives

    incentives_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Retail Incentives Forcasting Tool Data_update.xlsx'

    incentives_data = pd.DataFrame()
    progress = eureqa_utils.Progress( len( range( 2010, 2017 ) ), name='Incentives progress' )
    for year in range( 2010, 2017 ):
        # read in data
        year_data = pd.read_excel( incentives_file, sheetname=str( year ) ).reset_index()

        # fix column and rows
        year_data = fix_column_names( year_data )
        year_data = fix_models( year_data )

        # pivot into incentive columns / model & month rows
        year_pv = fix_pivot( year_data )

        incentives_data = pd.concat( [ incentives_data, year_pv ], ignore_index=True )

        progress.increment()
    progress.finish()

    incentives_data = incentives_data.fillna( 0 )

    return incentives_data

def prep_model_releases():
    """
    Prep Audi & competitor releases dates (takes ~10 seconds to run)

    As of 10/4/2016:
    - Date ranges: 10/1/2010 - 12/1/2020
    """

    def aggregate_releases( release_data ):
        """Aggregate releases by brand & segment / brand / all competitor"""

        # set up aggregation types
        agg_cols     = [ 'minor_change', 'major_change', 'change', 'generation_age', 'model_age' ]
        aggregations = { 'minor_change': sum, 'major_change': sum, 'change': sum, 'generation_age': np.mean, 'model_age': np.mean }

        # create separate aggregations
        brand_segment = release_data.groupby( [ 'Brand', 'Segment', 'Month' ] )[ agg_cols ].agg( aggregations ).reset_index()
        brand         = release_data.groupby( [ 'Brand', 'Month' ] )[ agg_cols ].agg( aggregations ).reset_index()
        comp_segment  = release_data.loc[ release_data.Brand != 'Audi' ].groupby( [ 'Segment', 'Month' ] )[ agg_cols ].agg( aggregations ).reset_index()
        comp          = release_data.loc[ release_data.Brand != 'Audi' ].groupby( 'Month' )[ agg_cols ].agg( aggregations ).reset_index()
        brand[ 'Segment' ]      = 'All'
        comp_segment[ 'Brand' ] = 'Competitor' 
        comp[ 'Brand' ]         = 'Competitor'
        comp[ 'Segment' ]       = 'All'

        # combine aggregations together
        all_agg = pd.concat( [ brand_segment, brand, comp_segment, comp ], ignore_index=True )

        # convert model release signals back into binary flags
        change_cols = [ 'minor_change', 'major_change', 'change' ]
        for col in change_cols:
            all_agg[ col ] = all_agg[ col ].apply( lambda x: 1 if x >= 1 else 0 )

        return all_agg

    def calculate_months_until_change( dataframe ):
        """Calculate # months until next model release"""

        change_cols = [ 'change', 'major_change', 'minor_change' ]
        for col in change_cols:
            
            # calculate month of next change
            month_of_col              = 'month_of_' + col
            dataframe[ month_of_col ] = np.nan
            dataframe.loc[ dataframe[ col ] == 1, month_of_col ] = dataframe.loc[ dataframe[ col ] == 1, 'Month' ]
            dataframe[ month_of_col ] = dataframe.groupby( [ 'Brand', 'Segment' ] )[ month_of_col ].bfill().ffill()
            
            # calculate months until next change
            months_until_col              = 'months_until_' + col
            dataframe[ months_until_col ] = dataframe.apply( lambda row: relativedelta.relativedelta( row[ month_of_col ].to_timestamp(), row.Month.to_timestamp() ).years*12 + relativedelta.relativedelta( row[ month_of_col ].to_timestamp(), row.Month.to_timestamp() ).months, axis=1 )

        return dataframe

    def convert():
        pass

    # read in data
    release_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/output_model_releases_full.txt'
    release_data = pd.read_csv( release_file )

    # some data cleaning
    release_data[ 'Month' ]  = pd.to_datetime( release_data[ 'Date' ], format='%Y-%m' ).dt.to_period( 'M' ) # fix dates
    release_data             = release_data.loc[ release_data.Generation != '0' ] # get rid of rows before there was a generation for that car
    release_data[ 'change' ] = release_data[ 'minor_change' ] + release_data[ 'major_change' ] # calculate if there was any change that month

    # get months until next change
    # release_data[ 'months_until_change' ] = release_data.groupby( [ 'Brand', 'Model', 'Segment', 'Bodystyle', 'Generation' ] ).model_age.transform( lambda x: x.iloc[::-1] )

    # calculate overall model age
    release_data                = release_data.rename( columns={ 'model_age': 'generation_age' } ) # rename "model age" to "generation age"
    release_data[ 'group_row' ] = release_data.groupby( [ 'Brand', 'Model', 'Segment', 'Bodystyle' ] ).Brand.transform( lambda x: x.index[0] )
    release_data[ 'group_row' ] = release_data[ 'group_row' ].astype( 'int' )
    release_data[ 'model_age' ] = release_data.index.tolist() - release_data.group_row + 1

    # aggregate and prep
    agg_release = aggregate_releases( release_data )
    agg_release = calculate_months_until_change( agg_release )

    # pivot into final output
    pv_release         = agg_release.pivot_table( values=[ 'model_age', 'months_until_change', 'months_until_major_change', 'months_until_minor_change' ], index='Month', columns=[ 'Brand', 'Segment' ] )
    pv_release.columns = [ '_'.join( c ) for c in pv_release.columns ]

    # final data cleaning
    pv_release = pv_release.fillna( 0 ) # fill missing values with 0
    pv_release = pv_release.applymap( lambda x: 0 if x < 0 else x ) # if negative (no upcoming changes), set to 0
    pv_release = pv_release.reset_index()

    # get rid of too-specific variables
    drop_release_cols = [ c for c in pv_release.columns if c != 'Month' and not 'Audi' in c and not 'Competitor' in c and not c.endswith( '_All' ) ]
    pv_release = pv_release.drop( drop_release_cols, axis=1 )

    return pv_release

def prep_inventory():
    """
    Prep Audi turn rate, dealer day supply, count of vehicles in status 50, and count of vehicles in status 60 - inventory metrics

    As of 10/10/2016:
    - Date ranges: 1/1/2011 - 10/1/2016
    """

    # read in data
    inventory_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/vehicleCounts.txt'
    inventory_data = pd.read_csv( inventory_file )

    # fix date format
    inventory_data.Month = pd.to_datetime( inventory_data.Month, format='%Y-%m-%d' ).dt.to_period( 'M' )

    return inventory_data

def prep_tier1_adspend():
    """
    Prep tier 1 audi ad spend data

    As of 10/31/2016:
    - Date ranges: 1/1/2012 - 12/1/2016
    """

    # read in data    
    adspend_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/2012-2016 Audi Actual Media Spend by Month 10.14.16 T1 prepared.txt'
    adspend_data = pd.read_csv( adspend_file )

    # pivot medium to columns
    adspend_data = adspend_data.pivot( index='Month', columns='Medium', values='Dollar_Spend' )
    adspend_data = adspend_data.fillna( 0 )
    adspend_data = adspend_data.rename( columns=lambda c: 'Tier1' + c.replace( ' ', '' ) + 'AdSpend' )

    # calculate total tier1 ad spend
    adspend_data[ 'Tier1TotalAdSpend' ] = adspend_data.sum( axis=1 )

    # fix month column
    adspend_data = adspend_data.reset_index()
    adspend_data[ 'Month' ] = pd.to_datetime( adspend_data.Month, format='%Y-%m' ).dt.to_period( 'M' )

    return adspend_data

def prep_tier2_adspend():
    """
    Prep tier 2 audi ad spend data

    As of 10/31/2016:
    - Date ranges: 1/1/2012 - 12/1/2016
    """

    # read in data    
    adspend_file = '/Users/jessicalin/Documents/Projects/Audi/data/Data/2012-2016 Audi Actual Media Spend by Month 10.14.16 T2 prepared.txt'
    adspend_data = pd.read_csv( adspend_file )

    # fix medium
    adspend_data.Medium = adspend_data.Medium.str.replace( ' ', '' )

    # pivot medium to columns
    adspend_data = adspend_data.pivot( index='Month', columns='Medium', values='Dollar_Spend' )
    adspend_data = adspend_data.fillna( 0 )
    adspend_data = adspend_data.rename( columns=lambda c: 'Tier2' + c.replace( ' ', '' ) + 'AdSpend' )

    # calculate total tier1 ad spend
    adspend_data[ 'Tier2TotalAdSpend' ] = adspend_data.sum( axis=1 )

    # fix month column
    adspend_data = adspend_data.reset_index()
    adspend_data[ 'Month' ] = pd.to_datetime( adspend_data.Month, format='%Y-%m' ).dt.to_period( 'M' )

    return adspend_data

def prep_comp_adspend():
    """
    Prep competitor ad spend

    As of 10/31/2016:
    - Date ranges: 1/1/2012 - 6/1/2016
    """

    # read in data
    comp_file  = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Monthly Comp Spend Full.txt'
    comp_spend = pd.read_csv( comp_file )
    comp_spend = comp_spend.fillna( 0 )

    # get ad spend for overall brand
    comp_spend.Brand = comp_spend.Brand.str.replace( ' ', '' )
    by_brand         = comp_spend.groupby( [ 'Brand', 'Month' ] ).Dollar_Spend.sum().reset_index()
    by_brand.Brand   = by_brand.Brand + 'AdSpend'
    brand_pv         = by_brand.pivot( index='Month', columns='Brand', values='Dollar_Spend' )
    brand_pv[ 'TotalCompetitorAdSpend' ] = brand_pv.sum( axis=1 )

    # get ad spend for overall medium
    by_medium        = comp_spend.groupby( [ 'Medium', 'Month' ] ).Dollar_Spend.sum().reset_index()
    by_medium.Medium = 'Competitor' + by_medium.Medium.str.replace( ' ', '' ) + 'AdSpend'
    medium_pv        = by_medium.pivot( index='Month', columns='Medium', values='Dollar_Spend' )

    # combine together
    combined = brand_pv.merge( medium_pv, how='outer', left_index=True, right_index=True )
    combined = combined.fillna( 0 )

    # fix month column
    combined = combined.reset_index()
    combined.Month = pd.to_datetime( combined.Month, format='%Y-%m' ).dt.to_period( 'M' )

    return combined


#######################
# Create lookup files #
#######################

def get_dag_lookup():
    """Get dag lookup file"""

    dag_file   = '/Users/jessicalin/Documents/Projects/Audi/data/Data/DAG Wholesale Projections  - August 2016.xlsx'
    dag_lookup = pd.read_excel( dag_file, sheetname='DAG Participation' )[[ 'Dealer_Code', 'DAG', 'DAG Name' ]]
    dag_lookup = dag_lookup.rename( columns={ 'Dealer_Code': 'DealerCode', 'DAG Name': 'DAGName' } )

    # fix city name
    dag_lookup[ 'DAGName' ] = dag_lookup[ 'DAGName' ].apply( lambda x: re.sub( ' Area$', '', x ) )
    
    return dag_lookup

def get_dealer_lookup():
    """Get dealer lookup file"""

    # get active dealers
    dealer_file   = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Audi_US_DealerList_080516.xlsx'
    dealer_lookup = pd.read_excel( dealer_file, sheetname='Audi_Dealer', skiprows=2 )[[ 'Code', 'Region', 'Area No.', 'Area Name', 'DBA Name' ]]
    dealer_lookup = dealer_lookup.rename( columns={ 'Code': 'DealerCode', 'Area No.': 'Area', 'Area Name': 'AreaName', 'DBA Name': 'DealerName' } )

    # fix area name
    dealer_lookup.AreaName = dealer_lookup.AreaName.apply( lambda x: re.sub( '^[0-9]{2} ', '', x ) )

    # add in inactive dealers
    dealer_file2   = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Dealers - not active any more.xls'
    dealer_lookup2 = pd.read_excel( dealer_file2, sheetname='Sheet1' )

    # fix columns
    dealer_lookup2.Region = dealer_lookup2.Region.str.split( ' ' ).str.get(1)
    dealer_lookup2        = dealer_lookup2.rename( columns={ 'Code': 'DealerCode', 'Dealer': 'DealerName' } )
    dealer_lookup2        = dealer_lookup2.merge( dealer_lookup[[ 'Area', 'AreaName' ]].drop_duplicates(), how='left', on='Area' )
    dealer_lookup2        = dealer_lookup2.drop( 'State', axis=1 )

    # add in misc missing dealers
    dealer_file3   = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Missing Dealercodes from BPO File Nutonian.xls'
    dealer_lookup3 = pd.read_excel( dealer_file3, sheetname='New BPO' )

    # fix columns
    dealer_lookup3.Region = dealer_lookup3.Region.str.split( ' ' ).str.get(1)
    dealer_lookup3        = dealer_lookup3.rename( columns={ 'Code': 'DealerCode', 'Dealer': 'DealerName' } )
    dealer_lookup3        = dealer_lookup3.merge( dealer_lookup[[ 'Area', 'AreaName' ]].drop_duplicates(), how='left', on='Area' )
    dealer_lookup3        = dealer_lookup3.drop( 'State', axis=1 )

    # stitch all together
    dealer_lookup = pd.concat( [ dealer_lookup, dealer_lookup2, dealer_lookup3 ], ignore_index=True )

    # get rid of dupes
    dealer_lookup = dealer_lookup.drop_duplicates()

    return dealer_lookup

def get_modelcode_lookup():
    """Get lookup from modelcode to segment"""

    def get_carline_lookup():
        """Get lookup between modelcode -> carline"""
        
        # get data
        carline_file1 = '/Users/jessicalin/Documents/Projects/Audi/data/carline.csv'
        carline_data1 = pd.read_csv( carline_file1 )
        carline_data1 = carline_data1.rename( columns={ 'CarLine2': 'Model' } )

        # convert carlines into base line
        carline_data1.Model = carline_data1.Model.replace( { 'S5': 'A5', 'A8L': 'A8', 'RS5': 'A5', 'TTC': 'TT', 'TTRS': 'TT', 'TTR': 'TT', 'TTS': 'TT', 'S6': 'A6', 'S8': 'A8', 'S4': 'A4', 'SQ5': 'Q5', 'RS7': 'A7', 'S3': 'A3' } )

        # add in missing model codes
        carline_file2 = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Missing ModelCodes incl Mobility Nutonian Update 20160825.xls'
        carline_data2 = pd.read_excel( carline_file2 )
        carline_data2 = carline_data2.rename( columns={ 'Sale Model Description': 'ModelCode2', 'Vehicle Carline': 'Model' } )[[ 'ModelCode2', 'Model', 'ModelCode' ]]
        carline_data2.Model = carline_data2.Model.replace( { 'A6 Sedan': 'A6', 'R8 Spyder': 'R8', 'Audi Q3': 'Q3', 'TT Roadster': 'TT', 'TT Coupe': 'TT', 'Audi Q7': 'Q7', 'A3 Sportback': 'A3', 'A4 Sedan': 'A4', 'R8 Coupe': 'R8' } )

        # combine carline files together
        carline_lookup = pd.concat( [ carline_data1, carline_data2 ], ignore_index=True )
        carline_lookup = carline_lookup.drop_duplicates( subset=[ 'Model', 'ModelCode' ] ).drop( 'ModelCode2', axis=1 )

        return carline_lookup

    def get_segment_lookup():
        """Get lookup between carline -> segment"""
        
        # get data
        brands_file    = '/Users/jessicalin/Documents/Projects/Audi/data/brands_.csv'
        segment_lookup = pd.read_csv( brands_file ).loc[ lambda df: df.Brand == 'Audi', [ 'Model', 'Segment' ] ]

        # clean up
        segment_lookup.Model = segment_lookup.Model.replace( { 'A4 allroad': 'A4' } )
        segment_lookup       = segment_lookup.drop_duplicates( subset=[ 'Model', 'Segment' ] )
        
        return segment_lookup

    # get prelim lookup files
    carline_lookup = get_carline_lookup()
    segment_lookup = get_segment_lookup()

    # create final lookup
    modelcode_lookup = carline_lookup.merge( segment_lookup, how='inner', on='Model' ) # inner join removes the audi mobility model codes

    return modelcode_lookup

def get_aggregation_lookup( wrapped_vars=True ):
    """Get list of how variables should be aggregated

    If wrapped_vars is True, return keys wrapped in parentheses; otherwise return original dataset names
    """

    agg_file = '/Users/jessicalin/Documents/Projects/Audi/data/var_aggs.csv'

    if wrapped_vars:
        agg_types = pd.read_csv( agg_file ).groupby( 'ds_var_name' )['agg'].first().to_dict()
    else:
        agg_types = pd.read_csv( agg_file ).groupby( 'var_name' )['agg'].first().to_dict()

    return agg_types

def get_drivers_lookup():
    """Get list of whether a variable is a controllable or uncontrollable lever"""

    # read in data
    drivers_file   = '/Users/jessicalin/Documents/Projects/Audi/data/Data/Copy of Variables classification edit Patrick_Treffen with levels.xlsx'
    drivers_lookup = pd.read_excel( drivers_file )
    drivers_lookup = drivers_lookup.rename( columns={ 'variable name': 'ds_name', 'dashboard label': 'display_name', 'type': 'lever_type', 'dealer': 'is_dealer_lever', 'dealer group': 'is_dag_lever', 'area': 'is_area_lever', 'national': 'is_national_lever' } )

    # fix types
    drivers_lookup[ 'lever_type' ] = drivers_lookup[ 'lever_type' ].replace( 'uncontrollable', 'Uncontrollable' )

    # get variable name in dataset as well as datasource name after eureqa conversion
    drivers_lookup[ 'var_name' ] = drivers_lookup[ 'ds_name' ].str.replace( '(', '' ).str.replace( ')', '' )

    # add column with all lever types
    drivers_lookup.loc[ drivers_lookup.is_dealer_lever == 1, 'lever_level' ]   = 'dealer'
    drivers_lookup.loc[ drivers_lookup.is_dag_lever == 1, 'lever_level' ]      = 'dag'
    drivers_lookup.loc[ drivers_lookup.is_area_lever == 1, 'lever_level' ]     = 'area'
    drivers_lookup.loc[ drivers_lookup.is_national_lever == 1, 'lever_level' ] = 'national'

    # re-order columns
    drivers_lookup = drivers_lookup[[ 'var_name', 'ds_name', 'display_name', 'lever_type', 'is_dealer_lever', 'is_dag_lever', 'is_area_lever', 'is_national_lever', 'lever_level' ]]

    # get rid of pct of total variables
    drivers_lookup = drivers_lookup.loc[ ( ~drivers_lookup.var_name.str.endswith( 'AsPctOfTotal' ) ) & ( drivers_lookup.var_name != 'Accuracy' ) ]

    return drivers_lookup