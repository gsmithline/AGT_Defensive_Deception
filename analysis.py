import numpy as np
import math
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
import pandas as pd

np.random.seed(42)


df = pd.read_csv('results_1000_games.csv') 
columns = df.columns

#summary statistics
print(df.describe())
#price of anarchy vs lambda
sns.lineplot(data=df, x='Lambda Value', y='Price of Anarchy')
plt.show()
#price of anarchy vs lambda for each lambda range dot plot
'''
for lamba_range in df['Lambda Range'].unique():
    #sns.lineplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Price of Anarchy')
    sns.scatterplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Price of Anarchy') 
    plt.title(f'Price of Anarchy vs Lambda for {lamba_range}')
    plt.ylabel('Price of Anarchy')
    plt.xlabel('Lambda Value')
    plt.show()
'''
#potential function value vs lambda
sns.lineplot(data=df, x='Lambda Value', y='Potential Function Value')
plt.show()

