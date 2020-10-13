import pandas as pd
import Sita_Functions as fxns
import numpy as np
from joblib import dump


beneficiary = pd.read_csv('./data/Train_Beneficiarydata-1542865627584.csv')
inpatient =  pd.read_csv('./data/Train_Inpatientdata-1542865627584.csv')
outpatient =  pd.read_csv('./data/Train_Outpatientdata-1542865627584.csv')
target = pd.read_csv('./data/Train-1542865627584.csv')


fxns.date_parser(beneficiary, ['DOB', 'DOD'])
fxns.date_parser(inpatient, ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt'])
fxns.date_parser(outpatient, ['ClaimStartDt', 'ClaimEndDt'])

# change numeric encoding from 1/2 to 0/1
fxns.re_encode(beneficiary, inpatient, outpatient)

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

# # merge dfs
claims = pd.concat([inpatient, outpatient])
claims = pd.merge(claims, beneficiary, on='BeneID')
claims = pd.merge(claims, target, on='Provider')

# add date cols containing only day, week, year
fxns.split_date(claims, ['ClaimStartDt', 'ClaimEndDt', 'AdmissionDt', 'DischargeDt'])

# change various cols to dtype category
fxns.to_category_dtype(claims)

# drop ChronicCond_ prefix from applicable cols
fxns.drop_chronic_prefix(claims)


dump(claims, 'claims.pkl')








