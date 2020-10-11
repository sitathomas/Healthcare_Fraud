import pandas as pd
import numpy as np


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


def date_parser(df, cols):
    '''
    Converts passed columns in a df to datetime.
    '''
    for col in cols:
        df[col] = pd.to_datetime(df[col])


def check_obj_dtypes(*dfs):
    '''
    Prints the type and count of dtypes in any object columns of all dfs passed.
    '''
    for df in dfs:
        object_cols = df.select_dtypes('object').columns.tolist()

        for col in object_cols:
            val_counts = df[col].apply(type).value_counts()
            if len(val_counts) > 1:
                print(f'{"-" * 40}\n', val_counts, f'\n{"-" * 40}\n')
            else:
                print(val_counts, '\n')


def re_encode(*dfs):
    for df in dfs:
        cols = df.columns[df.columns.str.contains('Gender')
                          | df.columns.str.contains('Race')
                          | df.columns.str.contains('Chronic')].to_list()    

        for col in cols:
            df.loc[df[col] == 2, col] = 0
            df.loc[df[col] == 1, col] = 1


def to_category_dtype(*dfs):
    '''
    Changes the dtype of a column for all dfs passed to 'category'.
    '''
    for df in dfs:
        cols = df.columns[df.columns.str.contains('Gender')
                    | df.columns.str.contains('Race')
                    | df.columns.str.contains('RenalDiseaseIndicator')
                    | df.columns.str.contains('State')
                    | df.columns.str.contains('County')
                    | df.columns.str.contains('Chronic')
                    | df.columns.str.contains('Diagnosis')
                    | df.columns.str.contains('Procedure')].to_list()
        df[cols] = \
            df[cols].apply(lambda x: x.astype('category'))


def dummify(*dfs):
    '''
    Converts procedure and diagnosis code values to 1 and NaNs to zero for all dfs passed.
    '''
    for df in dfs:
        procedure_cols = df.columns[df.columns.str.contains('Procedure')].to_list()
        diagnosis_cols = df.columns[df.columns.str.contains('ClmDiagnosis')].to_list()

        df[procedure_cols] = df[procedure_cols].fillna(0).astype(int)
        for col in procedure_cols:
            df.loc[df[col] > 0, [col]] = 1
        
        df[diagnosis_cols] = df[diagnosis_cols].fillna(0)
        for col in diagnosis_cols:
            df.loc[df[col] != 0, [col]] = 1


def consolidate(*dfs):
    '''
    For each claim (row) in all dfs passed, sums all columns that contain a procedure or
    diagnosis code and stores those sums in two new columns, NumProcedureCodes and
    NumDiagnosisCodes, dropping the single-code procedure and diagnosis columns.
    '''
    for df in dfs:
        procedure_cols = df.columns[df.columns.str.contains('Procedure')].to_list()
        diagnosis_cols = df.columns[df.columns.str.contains('ClmDiagnosis')].to_list()
#         physician_cols = ['OperatingPhysician', 'OtherPhysician']
        
        df['NumProcedureCodes'] = df[procedure_cols].sum(axis=1)
        df['NumDiagnosisCodes'] = df[diagnosis_cols].sum(axis=1)
#         df['NumDoctors']        = df[physician_cols].count(axis=1) + 1 # +1 includes Attending

        df.drop(procedure_cols, axis=1, inplace=True)
        df.drop(diagnosis_cols, axis=1, inplace=True)
#         df.drop(physician_cols, axis=1, inplace=True)


def cols_by_dtype(df):
    '''
    Saves numeric and categorical columns of a df to three new variables, numeric_cols, 
    categorical_cols, and date_cols.
    '''
    numeric_cols = df.select_dtypes(np.number).columns.to_list()
    categorical_cols = df.select_dtypes(['object', 'category']).columns.to_list()
    date_cols = df.select_dtypes('datetime64[ns]').columns.to_list()
    return numeric_cols, categorical_cols, date_cols

def split_date(df, cols):
    '''
    Splits any datetime cols specified for a df into three additional columns containing
    the date's week, month, and year, respectively.
    '''
    for col in cols:
        df[f'{str(col)}Week'] = df[col].dt.week
        df[f'{str(col)}Month'] = df[col].dt.month
        df[f'{str(col)}Year'] = df[col].dt.year