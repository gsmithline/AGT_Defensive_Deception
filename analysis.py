import numpy as np
import math
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
import pandas as pd
import sys

np.random.seed(42)


df = pd.read_csv('results_1000_games.csv') 
columns = df.columns
#send print output to txt file
original_stdout = sys.stdout
with open('analysis.txt', 'w') as f:
    sys.stdout = f
    
    '''
    #summary statistics
    print(df.describe())
    #price of anarchy vs lambda
    sns.lineplot(data=df, x='Lambda Value', y='Price of Anarchy')
    plt.show()
    #price of anarchy vs lambda for each lambda range dot plot
    '''
    '''
    for lamba_range in df['Lambda Range'].unique():
        #sns.lineplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Price of Anarchy')
        sns.scatterplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Price of Anarchy', hue='Game Round') 
        #color each game round differently in the same plot
        plt.title(f'Price of Anarchy vs Lambda for {lamba_range}')
        plt.ylabel('Price of Anarchy')
        plt.xlabel('Lambda Value')
        plt.show()
    '''
    '''
    #price of anarchy vs lambda for each lambda range line plot
    for lamba_range in df['Lambda Range'].unique():
        sns.lineplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Price of Anarchy')
        plt.title(f'Price of Anarchy vs Lambda for {lamba_range}')
        plt.ylabel('Price of Anarchy')
        plt.xlabel('Lambda Value')
        plt.show()
    '''
    '''
    #box plot of price of anarchy for each lambda range
    for lamba_range in df['Lambda Range'].unique():
        sns.boxplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Range', y='Price of Anarchy')
        plt.title(f'Price of Anarchy for {lamba_range}')
        plt.ylabel('Price of Anarchy')
        plt.xlabel('Lambda Range')
        plt.show()

    #correlation matrix for each lambda range
    for lamba_range in df['Lambda Range'].unique():
        sns.heatmap(df[df['Lambda Range'] == lamba_range].corr(), annot=True)
        plt.title(f'Correlation Matrix for {lamba_range}')
        plt.show()
    '''
    #t-test for each combination of values
    print('T-test for each combination of values:')
    for lamba_range in df['Lambda Range'].unique():
        for column in columns:
            for column2 in columns:
                if column != column2 and df[column].dtype == 'float64' and df[column2].dtype == 'float64':
                    print(f'T-test for {column} and {column2} in {lamba_range}:')
                    print(stats.ttest_ind(df[df['Lambda Range'] == lamba_range][column], df[df['Lambda Range'] == lamba_range][column2]))
    

    # ANOVA test for each combination of values
    print('ANOVA test for each combination of values:')
    for lamba_range in df['Lambda Range'].unique():
        for column in columns:
            for column2 in columns:
                if column != column2 and df[column].dtype == 'float64' and df[column2].dtype == 'float64':
                    print(f'ANOVA test for {column} and {column2} in {lamba_range}:')
                    print(stats.f_oneway(df[df['Lambda Range'] == lamba_range][column], df[df['Lambda Range'] == lamba_range][column2]))

    #kruskal wallis test for each combination of values
    print('Kruskal Wallis test for each combination of values:')
    for lamba_range in df['Lambda Range'].unique():
        for column in columns:
            for column2 in columns:
                if column != column2 and df[column].dtype == 'float64' and df[column2].dtype == 'float64':
                    print(f'Kruskal Wallis test for {column} and {column2} in {lamba_range}:')
                    print(stats.kruskal(df[df['Lambda Range'] == lamba_range][column], df[df['Lambda Range'] == lamba_range][column2]))


    #kenall tau correlation for each combination of values
    print('Kendall Tau correlation for each combination of values:')
    for lamba_range in df['Lambda Range'].unique():
        for column in columns:
            for column2 in columns:
                if column != column2 and df[column].dtype == 'float64' and df[column2].dtype == 'float64':
                    print(f'Kendall Tau correlation for {column} and {column2} in {lamba_range}:')
                    print(stats.kendalltau(df[df['Lambda Range'] == lamba_range][column], df[df['Lambda Range'] == lamba_range][column2]))


    #spearman rank correlation for each lambda range and each combination of values
    print('Spearman Rank correlation for each combination of values:')
    for lamba_range in df['Lambda Range'].unique():
        for column in columns:
            for column2 in columns:
                if column != column2 and df[column].dtype == 'float64' and df[column2].dtype == 'float64':
                    print(f'Spearman Rank correlation for {column} and {column2} in {lamba_range}:')
                    print(stats.spearmanr(df[df['Lambda Range'] == lamba_range][column], df[df['Lambda Range'] == lamba_range][column2]))


    #pearson correlation for all lambda ranges and all combinations of values
    print('Pearson correlation for for each combination of values:')
    for lamba_range in df['Lambda Range'].unique():
        for column in columns:
            for column2 in columns:
                if column != column2 and df[column].dtype == 'float64' and df[column2].dtype == 'float64':
                    print(f'Pearson correlation for {column} and {column2} in {lamba_range}:')
                    print(stats.pearsonr(df[df['Lambda Range'] == lamba_range][column], df[df['Lambda Range'] == lamba_range][column2]))
   

    #send print output to txt file
    sys.stdout = original_stdout
    print('Analysis complete')

    #causal inference for each lambda range
    for lamba_range in df['Lambda Range'].unique():
        print(f'Causal inference for {lamba_range}:')
        for column in columns:
            for column2 in columns:
                if column != column2 and df[column].dtype == 'float64' and df[column2].dtype == 'float64':
                    print(f'Causal inference for {column} and {column2} in {lamba_range}:')
                    print(stats.pearsonr(df[df['Lambda Range'] == lamba_range][column], df[df['Lambda Range'] == lamba_range][column2]))

