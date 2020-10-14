import sys
sys.path.insert(0, '.')

import pandas as pd
import Functions as fxns
import numpy as np
from joblib import dump

# load original CSV files
beneficiary = pd.read_csv('./data/Train_Beneficiarydata-1542865627584.csv')
inpatient =  pd.read_csv('./data/Train_Inpatientdata-1542865627584.csv')
outpatient =  pd.read_csv('./data/Train_Outpatientdata-1542865627584.csv')
target = pd.read_csv('./data/Train-1542865627584.csv')

# change numeric encoding from 1/2 to 0/1
fxns.re_encode(beneficiary)

# numerically encode RenalDiseaseIndicator
beneficiary.loc[beneficiary.RenalDiseaseIndicator == '0', 'RenalDiseaseIndicator'] = 0
beneficiary.loc[beneficiary.RenalDiseaseIndicator == 'Y', 'RenalDiseaseIndicator'] = 1

# convert procedure code cols to str
for df in [inpatient, outpatient]:
    procedure_cols = df.columns[df.columns
                                    .str.contains('Procedure')].to_list()
    df[procedure_cols] = \
        df[procedure_cols].apply(lambda x: x.astype('str'))
    
    for col in procedure_cols:
        df.loc[df[col] == 'nan', [col]] = np.nan

# # encode patient type in prep for merge
inpatient['IsOutpatient'] = 0
outpatient['IsOutpatient'] = 1

# numerically encode PotentialFraud
target.loc[target.PotentialFraud == 'No', 'PotentialFraud'] = 0
target.loc[target.PotentialFraud == 'Yes', 'PotentialFraud'] = 1

# merge dfs
claims = pd.concat([inpatient, outpatient])
claims = pd.merge(claims, beneficiary, on='BeneID')
claims = pd.merge(claims, target, on='Provider')

# parse dates
fxns.date_parser(claims,
                 ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD'])

# pickle pre-processed file
dump(claims, '../claims.pkl')









