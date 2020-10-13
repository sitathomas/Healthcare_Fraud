import pandas as pd
import numpy as np


def check_obj_dtypes(*dfs):
    '''
    Prints the type and count of dtypes in any object columns of all dfs passed.
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
    Saves numeric and categorical columns of a df to three new variables, numeric_cols, 
    categorical_cols, and date_cols.
    '''
    numeric_cols = df.select_dtypes(np.number).columns.to_list()
    categorical_cols = df.select_dtypes(['object', 'category']).columns.to_list()
    date_cols = df.select_dtypes('datetime64[ns]').columns.to_list()
    return numeric_cols, categorical_cols, date_cols


def date_parser(df, cols):
    '''
    Converts passed columns in a df to datetime.
    '''
    for col in cols:
        df[col] = pd.to_datetime(df[col])


def explore_df(df):
    '''
    Prints a description of a df including shape, columns and dtypes, and % missingness.
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
    Re-encodes Gender, Race, ChronicCond columns from 1/2 to 0/1 for any dfs passed.
    '''
    for df in dfs:
        cols = df.columns[df.columns.str.contains('Gender')
                          | df.columns.str.contains('Race')
                          | df.columns.str.contains('Chronic')].to_list()    

        for col in cols:
            df.loc[df[col] == 2, col] = 0
            df.loc[df[col] == 1, col] = 1


def re_encode_bool(df, cols):
    '''
    Re-encodes passed columns from Boolean type to 0/1 for any dfs passed.
    '''
    for col in cols:
        df.loc[df[col] == False, col] = 0
        df.loc[df[col] == True, col]  = 1


def split_date(df, cols):
    '''
    Splits any datetime cols specified for a df into three additional columns containing
    the date's week, month, and year, respectively.
    '''
    for col in cols:
        df[f'{str(col)}_Week'] = df[col].dt.week
#         df[f'{str(col)}_Month'] = df[col].dt.month
#         df[f'{str(col)}_Year'] = df[col].dt.year


def to_category_dtype(*dfs):
    '''
    Changes the dtype of a column for all dfs passed to 'category'.
    '''
    for df in dfs:
        cols = (df.columns[df.columns.str.contains('Gender')
                    | df.columns.str.contains('Race')
                    | df.columns.str.contains('State')
                    | df.columns.str.contains('County')
                    | df.columns.str.contains('Chronic')
                    | df.columns.str.contains('IsOutpatient')
                    | df.columns.str.contains('Dt_Week')
                    | df.columns.str.contains('Dt_Week')].to_list()
            + df.select_dtypes('object').columns.to_list())
        df[cols] = \
            df[cols].apply(lambda x: x.astype('category'))









