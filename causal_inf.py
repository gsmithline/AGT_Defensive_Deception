import statsmodels.api as sm
import numpy as np
import math
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
import pandas as pd
import sys
from patsy import dmatrices

from sklearn.linear_model import LinearRegression
from linearmodels.panel import PanelOLS


np.random.seed(42)


df = pd.read_csv('results_40_games.csv') 
# Aggregating metrics for each Lambda range
aggregate_metrics = df.groupby('Lambda Range').agg({
    'Price of Anarchy': 'mean',
    'Defender Utility': 'mean',
    'Percent System Working Optimally': 'mean'
}).reset_index()
df_renamed = df.rename(columns={
    'Percent System Working Optimally': 'Percent_System_Working_Optimally',
    'Lambda Range': 'Lambda_Range',
    'Game Round': 'Game_Round',
    'Defender Utility': 'Defender_Utility',
    'Price of Anarchy': 'Price_of_Anarchy',
    'Lambda Value': 'Lambda_Value'
})

y, X = dmatrices('Price_of_Anarchy ~ C(Defender_Utility) + + C(Lambda_Value) + C(Lambda_Range) + C(Game_Round)', data=df_renamed, return_type='dataframe')

model = sm.OLS(y, X).fit()
print(model.summary())
