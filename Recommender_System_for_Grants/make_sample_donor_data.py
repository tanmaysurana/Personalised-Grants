import pandas as pd
import numpy as np
import random

lender = pd.read_csv("kiva_loans.csv//recep_data.csv").sample(n=1000)

lender.loc[:, lender.columns != 'loan_amount'].applymap(lambda x: 1 if random.choice([1,2,3,4,5]) == 1 else x)

lender.to_csv("kiva_loans.csv//lender_data.csv", index=False)
