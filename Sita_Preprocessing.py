import pandas as pd
import numpy as np

beneficiary = pd.read_csv(
    '../data/Train_Beneficiarydata-1542865627584.csv')
inpatient =  pd.read_csv(
    '../data/Train_Inpatientdata-1542865627584.csv')
outpatient =  pd.read_csv(
    '../data/Train_Outpatientdata-1542865627584.csv')
target = pd.read_csv('../data/Train-1542865627584.csv')

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
        physician_cols = df.columns[df.columns.str.contains('OperatingPhysician')
                                   | df.columns.str.contains('OtherPhysician')].to_list()
        
        df['NumProcedureCodes'] = df[procedure_cols].sum(axis=1)
        df['NumDiagnosisCodes'] = df[diagnosis_cols].sum(axis=1)
        df['NumDoctors']        = df[physician_cols].sum(axis=1) + 1 # +1 includes Attending

        df.drop(procedure_cols, axis=1, inplace=True)
        df.drop(diagnosis_cols, axis=1, inplace=True)
        df.drop(physician_cols, axis=1, inplace=True)


def col_types(df):
    '''
    Saves numeric and categorical columns of a df to two new variables, numeric_cols and 
    categorical_cols.
    '''
    numeric_cols = df.select_dtypes(np.number)
    categorical_cols = df.select_dtypes(['object', 'category'])
    numeric_cols = numeric_cols.columns.to_list()
    categorical_cols = categorical_cols.columns.to_list()
    return numeric_cols, categorical_cols        
        

def preprocessing(): 
    date_parser(beneficiary, ['DOB', 'DOD'])
    date_parser(inpatient, ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt'])
    date_parser(outpatient, ['ClaimStartDt', 'ClaimEndDt'])
    dummify(inpatient, outpatient)
    
#     ONLY CONSOLIDATE IF THERE IS NO RELATIONSHIP BETWEEN THE SINGLE COLS AND THE TARGET VARIABLE
#     consolidate(inpatient, outpatient)
    
    # numerically encode RenalDiseaseIndicator
    beneficiary.loc[beneficiary.RenalDiseaseIndicator == '0', 'RenalDiseaseIndicator'] = 0
    beneficiary.loc[beneficiary.RenalDiseaseIndicator == 'Y', 'RenalDiseaseIndicator'] = 1
    
    # change appropriate cols to 'category' dtype
    cols = beneficiary.columns[beneficiary.columns.str.contains('Gender')
                | beneficiary.columns.str.contains('Race')
                | beneficiary.columns.str.contains('RenalDiseaseIndicator')
                | beneficiary.columns.str.contains('State')
                | beneficiary.columns.str.contains('County')
                | beneficiary.columns.str.contains('Chronic')].to_list()
    beneficiary[cols] = \
        beneficiary[cols].apply(lambda x: x.astype('category'))


beneficiary_num_cols, beneficiary_cat_cols = col_types(beneficiary)[0], col_types(beneficiary)[1]
inpatient_num_cols,   inpatient_cat_cols   = col_types(inpatient)[0], col_types(inpatient)[1]
outpatient_num_cols,  outpatient_cat_cols  = col_types(outpatient)[0], col_types(outpatient)[1]

preprocessing()