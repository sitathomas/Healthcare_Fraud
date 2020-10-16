import pandas as pd
import numpy as np


def add_count_per_col(df, groupby_col, count_col, newcol):
    '''
    Adds a new column to a df that lists a column count per grouped-by
    column.
    
    Arguments:
        df: A single dataframe.
        groupby_col: A single column to group by.
        count_col: A single column to aggregate the grouped counts for.
        newcol: The desired name of the new column.
    
    Output: None.
    
    Returns: A dataframe.
    '''
    df_per_col = df.groupby(groupby_col).count()[[count_col]]
    df_per_col.columns = [newcol]
    expanded_df = pd.merge(df, df_per_col, how='left',
                           left_on=groupby_col, right_on=df_per_col.index)
    return expanded_df


def add_mean_per_col(df, groupby_col, mean_col, newcol):
    '''
    Adds a new column to a df that lists a column mean per grouped-by
    column.
    
    Arguments:
        df: A single dataframe.
        groupby_col: A single column to group by.
        mean_col: A single column to aggregate the grouped means for.
        newcol: The desired name of the new column.
    
    Output: None.
    
    Returns: A dataframe.
    '''
    df_per_col = df.groupby(groupby_col).mean()[[mean_col]]
    df_per_col.columns = [newcol]
    expanded_df = pd.merge(df, df_per_col, how='left',
                           left_on=groupby_col, right_on=df_per_col.index)
    return expanded_df


def add_sum_per_col(df, groupby_col, sum_col, newcol):
    '''
    Adds a new column to a df that lists a column sum per grouped-by
    column.
    
    Arguments:
        df: A single dataframe.
        groupby_col: A single column to group by.
        sum_col: A single column to aggregate the grouped sums for.
        newcol: The desired name of the new column.
    
    Output: None.
    
    Returns: A dataframe.
    '''
    df_per_col = df.groupby(groupby_col).sum()[[sum_col]]
    df_per_col.columns = [newcol]
    expanded_df = pd.merge(df, df_per_col, how='left',
                           left_on=groupby_col, right_on=df_per_col.index)
    return expanded_df


def check_obj_dtypes(*dfs):
    '''
    Prints the type and count of dtypes in any columns of 'object' dtype. Highlighted columns
    have multiple dtypes (i.e. 'str' and 'float' for missing values).
    
    Arguments: One or more dataframes.
    
    Output: A printout.
    
    Returns: None.
    '''
    for df in dfs:
        object_cols = df.select_dtypes('object').columns.to_list()

        for col in object_cols:
            val_counts = df[col].apply(type).value_counts()
            if len(val_counts) > 1:
                print(f'{"-" * 40}\n', val_counts, f'\n{"-" * 40}\n')
            else:
                print(val_counts, '\n')


def cols_by_dtype(df):
    '''
    Saves numeric and categorical columns of a df to three new variables: numeric_cols, 
    categorical_cols, and date_cols.
    
    Arguments: A single dataframe.
    
    Output: None.
    
    Returns: A tuple containing three lists of column names.
    '''
    numeric_cols = df.select_dtypes(np.number).columns.to_list()
    categorical_cols = df.select_dtypes(['object', 'category']).columns.to_list()
    date_cols = df.select_dtypes('datetime64[ns]').columns.to_list()
    return numeric_cols, categorical_cols, date_cols


def date_parser(df, cols):
    '''
    Converts passed columns in a df to datetime.
    
    Arguments: A single dataframe.
    
    Output: None.
    
    Returns: Dataframe columns altered in place.
    '''
    for col in cols:
        df[col] = pd.to_datetime(df[col])


def explore_df(df):
    '''
    Prints a description of a df including shape, columns and dtypes, and % missingness.
    
    Arguments: A single dataframe.
    
    Output: A printout.
    
    Returns: None.
    '''
    print('Shape:', df.shape, '\n')
    print('Columns and dtypes:\n', df.dtypes, '\n')

    percent_missing = df.isna().mean().round(4) * 100
    print('Columns with Missingness:\n',
          percent_missing[percent_missing > 0.00
                         ].sort_values(ascending=False))


def drop_chronic_prefix(*dfs):
    '''
    Replaces the ChronicCondition_ prefix from any columns in any dfs passed with a _Chronic
    suffix (useful for readability on graphs).
    
    Arguments: One or more dataframes.
    
    Output: None.
    
    Returns: Dataframe columns altered in place.
    '''
    for df in dfs:
        df.rename(columns={'ChronicCond_Alzheimer': 'ChronicCond_Alzheimers',
                           'ChronicCond_Heartfailure': 'ChronicCond_HeartFailure',
                           'ChronicCond_Osteoporasis': 'ChronicCond_Osteoporosis',
                  'ChronicCond_rheumatoidarthritis': 'ChronicCond_RheumatoidArthritis',
                           'ChronicCond_stroke': 'ChronicCond_Stroke',
                           'RenalDiseaseIndicator': 'RenalDisease'},
                  inplace=True)
        
        chronic_cols = df.columns[df.columns.str.contains('Chronic')].to_list()
        
        df.columns = \
            [f'{col}_Chronic' if col in chronic_cols else col for col in df.columns]
        
        df.columns = df.columns.str.replace('ChronicCond_', '')


def re_encode(*dfs):
    '''
    Re-encodes Gender, ChronicCond columns from 1/2 to 0/1 for any dfs passed.
    
    Arguments: One or more dataframes.
    
    Output: None.
    
    Returns: Dataframe columns altered in place.
    '''
    for df in dfs:
        cols = df.columns[df.columns.str.contains('Chronic')].to_list() + ['Gender']  

        for col in cols:
            df.loc[df[col] == 2, col] = 0
            df.loc[df[col] == 1, col] = 1


def re_encode_bool(df, cols):
    '''
    Re-encodes passed columns from Boolean type to 0/1 for any dfs passed.
    
    Arguments:
        df: A single dataframe.
        cols: a list of one or more columns.
    
    Output: None.
    
    Returns: Dataframe columns altered in place.
    '''
    for col in cols:
        df.loc[df[col] == False, col] = 0
        df.loc[df[col] == True, col]  = 1
        df[col] == df[col].astype(str).astype(int)


def split_date(df, cols):
    '''
    Splits any datetime cols specified for a df into three additional columns containing
    the date's week, month, and year, respectively.
    
    Arguments:
        df: A single dataframe.
        cols: A list of one or more columns.
    
    Output: None.
    
    Returns: Dataframe columns altered in place.
    '''
    for col in cols:
        df[f'{str(col)}_Week'] = df[col].dt.week
#         df[f'{str(col)}_Month'] = df[col].dt.month
#         df[f'{str(col)}_Year'] = df[col].dt.year


def to_category_dtype(*dfs):
    '''
    Changes the dtype of a column for all dfs passed to 'category'.
    
    Arguments: One or more dataframes.
    
    Output: None.
    
    Returns: Dataframe columns altered in place.
    '''       
    for df in dfs:
        cols = ['County', 'Gender', 'HasDied', 'HasAnyPhysician', 'HasAllPhysicians',
                'IsOutpatient', 'PotentialFraud', 'Race', 'RenalDisease', 'State'
               ] + ( df.columns[df.columns.str.contains('Chronic')].to_list()
                 + df.columns[df.columns.str.contains('_Week')].to_list() )
        
        df[cols] = df[cols].apply(lambda x: x.astype('category'))









