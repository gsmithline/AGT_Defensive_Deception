import statsmodels.api as sm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from patsy import dmatrices
from linearmodels.panel import PanelOLS
from scipy.stats import ttest_ind, kendalltau, pearsonr, spearmanr, kruskal, f_oneway
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
# Set a random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.drop(columns=['Defender Best Response Utility', 'Defender Best Response Mixed Strategy', 
                     'Distance Between Defender and Actual', 'Optimal Potential Function Value', 'Attacker Optimal Mixed Strategy Profile'], inplace=True)
    return df

def aggregate_metrics_by_lambda(df):
    return df.groupby('Lambda Range').agg({
        'Price of Anarchy': 'mean',
        'Defender Utility': 'mean',
        'Percent System Working Optimally': 'mean'
    }).reset_index()

def ols_regression(df):
    df_renamed = df.rename(columns={
        'Percent System Working Optimally': 'Percent_System_Working_Optimally',
        'Lambda Range': 'Lambda_Range',
        'Game Round': 'Game_Round',
        'Defender Utility': 'Defender_Utility',
        'Price of Anarchy': 'Price_of_Anarchy',
        'Lambda Value': 'Lambda_Value',
        'Game Number': 'Game_Number'
    })
    y, X = dmatrices('Price_of_Anarchy ~ C(Lambda_Value) + C(Lambda_Range) + C(Game_Round)', data=df_renamed, return_type='dataframe')
    model = sm.OLS(y, X).fit()
    return model

def panel_data_analysis(df, lambda_range_on=False):
    if lambda_range_on:
        for lambda_range in df['Lambda Range'].unique():
            panel_data = df[df['Lambda Range'] == lambda_range].set_index(['Game Number', 'Game Round'])
            panel_data['Lambda Range'] = panel_data['Lambda Range'].astype('category')
            mod = PanelOLS.from_formula('`Price of Anarchy` ~ 1 + `Defender Utility` +  EntityEffects', data=panel_data)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            print(f'Lambda Range: {lambda_range}')
            print(res)
    else:
        df['Game ID'] = df['Game Number'].astype(str) + '-' + df['Game Round'].astype(str)
        panel_data = df.set_index(['Game ID', 'Game Round'])
        panel_data['Lambda Range'] = panel_data['Lambda Range'].astype('category')
        mod = PanelOLS.from_formula('`Price of Anarchy` ~ 1 + `Defender Utility` + EntityEffects', data=panel_data)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
    return res

def visualize_aggregate_metrics(aggregate_metrics):
    sns.lineplot(data=aggregate_metrics, x='Lambda Range', y='Price of Anarchy', label='Price of Anarchy')
    sns.lineplot(data=aggregate_metrics, x='Lambda Range', y='Defender Utility', label='Defender Utility')
    plt.xlabel('Lambda Range')
    plt.ylabel('Metrics')
    plt.title('System Performance Metrics by Lambda Range')
    plt.legend()
    plt.show()

def visualize_price_of_anarchy(df):
    sns.lineplot(data=df, x='Lambda Value', y='Price of Anarchy', hue='Game Round')
    plt.title('Price of Anarchy by Lambda Value Across Rounds')
    plt.show()

def perform_t_tests(df):
    columns = df.select_dtypes(include=['float64']).columns
    for lamba_range in df['Lambda Range'].unique():
        for i, column1 in enumerate(columns):
            for column2 in columns[i + 1:]:
                data1 = df[df['Lambda Range'] == lamba_range][column1]
                data2 = df[df['Lambda Range'] == lamba_range][column2]
                t_stat, p_val = ttest_ind(data1, data2)
                print(f'T-test between {column1} and {column2} for Lambda Range {lamba_range}: T-stat={t_stat}, P-value={p_val}')

def visualize_correlation_matrix(df, general_correlation=False):
    if general_correlation == True:
        sns.heatmap(df.corr(), annot=True)
        plt.title('General Correlation Matrix')
        plt.show()
    else:
        for lamba_range in df['Lambda Range'].unique():
            sns.heatmap(df[df['Lambda Range'] == lamba_range].corr(), annot=True)
            plt.title(f'Correlation Matrix for {lamba_range}')
            plt.show()

def visualize_box_plot(df):
    for lamba_range in df['Lambda Range'].unique():
        sns.boxplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Range', y='Price of Anarchy')
        plt.title(f'Price of Anarchy for {lamba_range}')
        plt.ylabel('Price of Anarchy')
        plt.xlabel('Lambda Range')
        plt.show()

def visualize_price_of_anarchy_by_lambda_range(df):
    for lamba_range in df['Lambda Range'].unique():
        sns.lineplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Price of Anarchy')
        plt.title(f'Price of Anarchy vs Lambda for {lamba_range}')
        plt.ylabel('Price of Anarchy')
        plt.xlabel('Lambda Value')
        plt.show()  

def changes_in_defender_utility(df):
    for lamba_range in df['Lambda Range'].unique():
        sns.lineplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Defender Utility')
        plt.title(f'Defender Utility vs Lambda for {lamba_range}')
        plt.ylabel('Defender Utility')
        plt.xlabel('Lambda Value')
        plt.show()
def changes_in_percent_system_working_optimally(df):
    for lamba_range in df['Lambda Range'].unique():
        sns.lineplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Percent System Working Optimally')
        plt.title(f'Percent System Working Optimally vs Lambda for {lamba_range}')
        plt.ylabel('Percent System Working Optimally')
        plt.xlabel('Lambda Value')
        plt.show()
def changes_in_price_of_anarchy(df):
    for lamba_range in df['Lambda Range'].unique():
        sns.lineplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Price of Anarchy')
        plt.title(f'Price of Anarchy vs Lambda for {lamba_range}')
        plt.ylabel('Price of Anarchy')
        plt.xlabel('Lambda Value')
        plt.show()
def changes_in_attacker_potential_function_value(df):
    for lamba_range in df['Lambda Range'].unique():
        sns.lineplot(data=df[df['Lambda Range'] == lamba_range], x='Lambda Value', y='Potential Function Value')
        plt.title(f'Potential Function Value vs Lambda for {lamba_range}')
        plt.ylabel('Potential Function Value')
        plt.xlabel('Lambda Value')
        plt.show()

#attacker adaption rate
def distribution_each_variable(df):
    for column in df.columns:
        for lamba_range in df['Lambda Range'].unique():
            if df[column].dtype == 'float64':
                sns.kdeplot(df[df['Lambda Range'] == lamba_range][column], fill=True)
                plt.title(f'Distribution of {column} for {lamba_range}')
                plt.show()


def cluster_and_visualize(df, features, color_by='Game Round', n_clusters=2):
    """
    Perform K-Means clustering on specified features and visualize using PCA for dimensionality reduction.
    Points in the plot are colored based on a separate categorical variable.

    :param df: DataFrame containing the data.
    :param features: List of column names to use for clustering.
    :param color_by: Column name to use for coloring the points.
    :param n_clusters: Number of clusters to form.
    """
    for lambda_range in df['Lambda Range'].unique():
        subset_df = df[df['Lambda Range'] == lambda_range]
        
        # Extract features for clustering
        X = subset_df[features]
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_scaled)
        
        # Prepare a color mapping based on 'color_by' column
        unique_vals = subset_df[color_by].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_vals)))
        color_dict = dict(zip(unique_vals, colors))
        point_colors = [color_dict[val] for val in subset_df[color_by]]
        
        # Visualize the clusters with PCA components and color by 'Game Round'
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=point_colors, label=f'{lambda_range}', alpha=0.6)
        
        # Labeling for clarity
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'K-Means Clustering for Lambda Range {lambda_range}')
        
        # Create a custom legend for game rounds
        custom_lines = [plt.Line2D([0], [0], color=color_dict[val], lw=4) for val in unique_vals]
        plt.legend(custom_lines, unique_vals, title=color_by)
        
        plt.show()


def plot_4d_space_by_lambda_range(df, x='Lambda Value', y='Percent System Working Optimally', z='Potential Function Value', color_by='Game Round'):
    lambda_ranges = df['Lambda Range'].unique()
    
    for lambda_range in lambda_ranges:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        subset_df = df[df['Lambda Range'] == lambda_range]

        rounds = subset_df[color_by].unique()
        colors = plt.cm.jet(np.linspace(0, 1, len(rounds)))

        for i, round in enumerate(rounds):
            round_subset = subset_df[subset_df[color_by] == round]
            ax.scatter(round_subset[x], round_subset[y], round_subset[z], color=colors[i], label=f'Round {round}', s=50)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        ax.set_title(f'4D Data Visualization for Lambda Range {lambda_range}')
        ax.legend(title=color_by)

        plt.show()

def plot_metric_trend(df, metric_name, game_round_col='Game Round'):
    # Plotting the trend of the metric over game rounds
    plt.figure(figsize=(10, 6))
    for lambda_range in df['Lambda Range'].unique():
        lambda_df = df[df['Lambda Range'] == lambda_range]
        lambda_df.groupby(game_round_col)[metric_name].mean().plot(label=f'Lambda Range {lambda_range}')
    plt.xlabel(game_round_col)
    plt.ylabel(metric_name)
    plt.title(f'Trend of {metric_name} Over Game Rounds')
    plt.legend()
    plt.show()

def plot_autocorrelation(df, metric_name, game_round_col='Game Round', general_correlation=False):
    plt.figure(figsize=(10, 6))
    if general_correlation == False:
        for lambda_range in df['Lambda Range'].unique():
            lambda_df = df[df['Lambda Range'] == lambda_range]
            sm.graphics.tsa.plot_acf(lambda_df[metric_name], lags=20, title=f'Autocorrelation for {metric_name} - Lambda Range {lambda_range}')
    else:
        sm.graphics.tsa.plot_acf(df[metric_name], lags=20, title=f'Autocorrelation for {metric_name}')
    plt.show()

# Kendall Tau test
def kendall_tau_test(df, column1, column2):
    tau, p_value = kendalltau(df[column1], df[column2])
    print(f"Kendall's tau between {column1} and {column2}: Tau={tau}, P-value={p_value}")

# Pearson correlation test
def pearson_correlation_test(df, column1, column2):
    corr, p_value = pearsonr(df[column1], df[column2])
    print(f"Pearson correlation between {column1} and {column2}: Correlation={corr}, P-value={p_value}")

# Spearman rank test
def spearman_rank_test(df, column1, column2):
    coef, p_value = spearmanr(df[column1], df[column2])
    print(f"Spearman rank correlation between {column1} and {column2}: Coefficient={coef}, P-value={p_value}")

# Kruskal-Wallis test
def kruskal_wallis_test(df, *columns):
    data = [df[column] for column in columns]
    stat, p_value = kruskal(*data)
    print(f"Kruskal-Wallis test across {', '.join(columns)}: Statistic={stat}, P-value={p_value}")

# ANOVA test
def anova_test(df, *columns):
    data = [df[column] for column in columns]
    stat, p_value = f_oneway(*data)
    print(f"ANOVA test across {', '.join(columns)}: Statistic={stat}, P-value={p_value}")

#results_40_bottom_start_games_4_round_games.csv
# results_40_bottom_start_games.csv
    
# results_1000_bottom_start_games_4_round_games.csv
# linear_increasing_alpha.csv
# Load and preprocess data
# analysis/rationality_bias_factor_stationairy_lambda.csv
# analysis/linear_increasing_alpha.csv
# analysis/shift_distribution_simple_increase.csv
# analysis/change_strat.csv
# analysis/using_poa_as_feature.csv
# analysis/using_composite_of_PSWO_Potential.csv
filepath = 'analysis/using_composite_of_PSWO_Potential.csv'  # Adjust this to your file path
df = load_and_preprocess_data(filepath)
aggregate_metrics = aggregate_metrics_by_lambda(df)
    
'''
Statisitcal Summary Tests
'''
#First general correlation
'''
NOTE:
- In the general case we can see that there is not a super strong correlation between lambda, PoA
and % system working optimally.
- When we look at the range level we see a stronger correlation.  What's interesting is that 
lambda has a strong negative correlation with game rounds, meaning as game rounds increase, lambda decreases (more bounded rationality).
- We also see that as lambda increases, the price of anarchy increases, and the % system working optimally decreases.
- Lambda is also strongly correlated with defender utility, meaning as lambda increases, defender utility increases.
- We see defender utility is negatively correlated with game rounds, meaning it drops as game rounds increase.
'''
visualize_correlation_matrix(df, general_correlation=False)
'''
NOTE: 
- High p-values, one cna lower by increasing covariates 
- Strong relationship between lambda and price of anarchy, and % system working optimally 
'''
res = panel_data_analysis(df, True)
print(res)

'''
ANOVA
'''
anova_test(df, 'Price of Anarchy', 'Defender Utility', 'Percent System Working Optimally')

'''
Kruskal-Wallis
Really just significant difference in means not much info, but it is statistically significant. 
'''
kruskal_wallis_test(df, 'Price of Anarchy', 'Lambda Value', 'Percent System Working Optimally')

'''
Kendall Tau Test
- Strong correlation and statistically significant at individual lambda range level
- Again High lambda values we get NaN since it does not very much
'''

for lamba_range in df['Lambda Range'].unique():
    print(f'Lambda Range: {lamba_range}')
    kendall_tau_test(df[df['Lambda Range'] == lamba_range], 'Price of Anarchy', 'Lambda Value')

'''
Pearson Correlation Test
- Strong correlations at Lambda level but not a higher up, all statistically significant
- Negatively correlated with game round and labda together. 
'''

for lamba_range in df['Lambda Range'].unique():
    print(f'Lambda Range: {lamba_range}')
    pearson_correlation_test(df[df['Lambda Range'] == lamba_range], 'Price of Anarchy', 'Lambda Value')
    pearson_correlation_test(df[df['Lambda Range'] == lamba_range], 'Lambda Value', 'Game Round')

'''
Spearman Rank Test
- again high and statistically significant speearman rank correlation
'''

for lamba_range in df['Lambda Range'].unique():
    print(f'Lambda Range: {lamba_range}')
    spearman_rank_test(df[df['Lambda Range'] == lamba_range], 'Price of Anarchy', 'Lambda Value')


'''
Aggregate Metrics
'''
visualize_aggregate_metrics(aggregate_metrics)

'''
Correlation Matrix
again strong correlation between lambda and price of anarchy, and % system working optimally
- low correlation BUT HIGH IN OTHER correlations, migh not be a good fit. 
'''

for lamba_range in df['Lambda Range'].unique():
    visualize_correlation_matrix(df, True)


''' 
Box Plot
Mean PoA across the board is at 2.0, which is about 50% of the system working optimally.
'''
visualize_box_plot(df)
'''
Plot 4d Space by Lambda Range
'''
plot_4d_space_by_lambda_range(df)

'''
Price of Anarchy by Lambda Range
Basic linear relationships
'''

changes_in_defender_utility(df)
changes_in_percent_system_working_optimally(df)
changes_in_price_of_anarchy(df)
changes_in_attacker_potential_function_value(df)

'''
PDF
mostly follow bimodal distribution
'''
distribution_each_variable(df)
'''
Trend Line
'''
for feature in ['Price of Anarchy', 'Defender Utility', 'Percent System Working Optimally', 'Potential Function Value']:
    plot_metric_trend(df, feature) # clearly defender does worse over time, the higher lambda is the better defender does.  
    plot_autocorrelation(df, feature)








