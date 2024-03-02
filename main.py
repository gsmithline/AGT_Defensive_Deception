from target import Target
from defender.defender import Defender
from game.game import Game
from attackers.attacker import Attacker
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Globals
'''
Inititalization of game 
_____________________________________________________________________________
'''
np.random.seed(42)
num_targets = 12
#random game set up 
epsilon = 9
game_rounds = 3
num_attackers = 12
num_targets = 10
num_games = 100
poa_results_avg = []
lambda_results_avg = []
potent_function_results_avg = []
defender_utility_results_avg = []
percent_system_working_optimally = []
round_game_poa_average = {x: 0 for x in range(1, game_rounds + 1)}
totals_for_rounds = {
    'poa': {i: 0 for i in range(1, game_rounds + 1)},
    'potential_function': {i: 0 for i in range(1, game_rounds + 1)},
    'defender_utility': {i: 0 for i in range(1, game_rounds + 1)},
    'percent_optimal': {i: 0 for i in range(1, game_rounds + 1)},
    'lambda': {i: 0 for i in range(1, game_rounds + 1)}
}
lambda_ranges = [(.2, .8)] # (0, 1) and (0, float('inf')) (3, 3) ENTIRELY RATIONA
congestion_costs = [random.randint(1, 10) for i in range(num_targets)]
print(f"congestion costs: {congestion_costs}")
rewards = [random.randint(1, 10) for i in range(num_targets)]
print(f"rewards: {rewards}")
penalties = [random.randint(1, 10) for i in range(num_targets)]
print(f"penalties: {penalties}")
initial_beliefs = np.ones(num_targets) / num_targets
print(f"initial beliefs: {initial_beliefs}")
for lambda_range in lambda_ranges:
    print(f"lambda range: {lambda_range}")
    for i in range(1, num_games + 1):
        targets = {j: Target(j, congestion_costs[j], rewards[j], penalties[j], 0, 0, 0) for j in range(num_targets)}
        
        defender = Defender(num_targets, initial_beliefs, lambda_range)
        attackers = {k: Attacker(num_targets, k) for k in range(1, num_attackers + 1)}  # Corrected loop var

        game = Game(targets, rewards, congestion_costs, penalties, defender, attackers)
        '''
        Inititalization of game 
        _____________________________________________________________________________
        '''
        percent_system_working_optimally_inner = []
        for i in range(1, game_rounds + 1):
            #update lambda
            defender.update_lambda_value(list(game.past_potential_function_values.values()))
            print(f"lambda value updated: {defender.lambda_value}")
            #test qr defender 
            defender.quantal_response(defender.lambda_value, game)
            print(f"expected congestion updated: {defender.expected_congestion}")
            #test strategy computation defender
            defender.optimize_strategy(targets, defender.expected_congestion)
            print(f"new defender mixed strategy updated: {defender.mixed_strategy}")
            #test defender expected utility
            defender.calculate_utility(game)
            print(f"new past utilities updated: {defender.past_utilities}")
            game.ibr_attackers(1000, epsilon)  
            game.calculate_potential_function_value(i)
            print(f"actual potential function value: {game.actual_potential_function_value}")
            game.price_of_anarchy()
            print(game.best_potential_function_value)
            print(game.current_poa)
            percent_system_working_optimally_inner.append(1/game.current_poa)
            totals_for_rounds['poa'][i] += game.current_poa
            totals_for_rounds['potential_function'][i] += game.actual_potential_function_value
            totals_for_rounds['defender_utility'][i] += defender.past_utilities[-1]
            totals_for_rounds['percent_optimal'][i] += 1 / game.current_poa
            totals_for_rounds['lambda'][i] += defender.lambda_value
        
        average_defender_utility = sum(defender.past_utilities)/len(defender.past_utilities)
        defender_utility_results_avg.append(average_defender_utility)
        averge_poa = sum(game.past_poa)/len(game.past_poa) 
        poa_results_avg.append(averge_poa)
        avgerage_lambda = sum(defender.past_lambda_values)/len(defender.past_lambda_values)
        lambda_results_avg.append(avgerage_lambda)
        avgerage_potential_function = sum(game.past_potential_function_values.values())/len(game.past_potential_function_values.values())
        potent_function_results_avg.append(avgerage_potential_function)
        average_percent_system_working_optimally = sum(percent_system_working_optimally_inner)/len(percent_system_working_optimally_inner)
        percent_system_working_optimally.append(average_percent_system_working_optimally)

    averages_for_rounds = {}
    for key, round_totals in totals_for_rounds.items():
        averages_for_rounds[key] = {round_index: total / num_games for round_index, total in round_totals.items()}
    '''
    df = pd.DataFrame({'POA': poa_results_avg, 'Lambda': lambda_results_avg, 'Potential Function': potent_function_results_avg, 
                    'Defender Utility': defender_utility_results_avg,
                    'Percent System Working Optimally': percent_system_working_optimally})
    print(df)
    print("_"*50)
    print("Descriptive Statistics:")
    print(df.describe())
    print("_"*50)
    print("Correlation:")
    print(df.corr())
    print("_"*50)
    print("Covariance:")
    print(df.cov())
    print("_"*50)
    print("Variances:")
    print(df.var()) 
    print("_"*50)
    print("Standard Deviation:")
    print(df.std())
    print("_"*50)
    print("Mean:")
    print(df.mean())
    print("_"*50)
    print("Median:")
    print(df.median())
    print("_"*50)
    print("Mode:")
    print(df.mode())
    print("_"*50)
    print("Skewness:")
    print(df.skew())
    print("_"*50)
    print("Kurtosis:")
    print(df.kurt())
    print("_"*50)
    average_avg_poa = sum(poa_results_avg)/len(poa_results_avg)
    print("_"*50) 
    print(f"Average Percent System Working Optimally: {1/average_avg_poa}")
    print("_"*50)
    #graph POA
    plt.plot(poa_results_avg)
    plt.xlabel('Game Rounds')
    plt.ylabel('Price of Anarchy')
    plt.title(f'{lambda_range}: Price of Anarchy Value Over Time')
    plt.show()

    #graph potential function value
    plt.plot(potent_function_results_avg)
    plt.xlabel('Game Rounds')
    plt.ylabel('Potential Function Value')
    plt.title(f'{lambda_range}: Potential Function Value Over Time')
    plt.show()

    #defender lambda
    plt.plot(lambda_results_avg)
    plt.xlabel('Game Rounds')
    plt.ylabel('Lambda Value')
    plt.title(f'{lambda_range}: Lambda Value Over Time')
    plt.show()


    #defender utility
    plt.plot(defender_utility_results_avg)
    plt.xlabel('Game Rounds')
    plt.ylabel('Defender Utility')
    plt.title(f'{lambda_range}: Defender Utility Over Time')
    plt.show()

    #graph percent system working optimally
    plt.plot(percent_system_working_optimally)
    plt.xlabel('Game Rounds')
    plt.ylabel('Average Percent System Working Optimally')
    plt.title(f'{lambda_range}: Averge Percent System Working Optimally Over Time')
    plt.show()

    #lambda vs PoA
    plt.scatter(lambda_results_avg, poa_results_avg)
    a, b = np.polyfit(lambda_results_avg, poa_results_avg, 1)
    plt.plot(lambda_results_avg, a*np.array(lambda_results_avg) + b, color='red')
    plt.xlabel('Lambda Value')
    plt.ylabel('Price of Anarchy')
    plt.title(f'{lambda_range}: Lambda Value vs Price of Anarchy')
    plt.show()

    #correlation heat map
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title(f'{lambda_range}: Correlation Matrix Heatmap')
    plt.show()

    # Example for PoA across different ranges of Lambda
    sns.boxplot(data=df, x='Lambda', y='POA')
    plt.title(f'{lambda_range}: Price of Anarchy Across Lambda Ranges')
    plt.show()
    print("_"*50)
    print("Congestion Costs")
    for i in congestion_costs:
        print(i)
    print("_"*50)
    print("Rewards")
    for i in rewards:
        print(i)
    print("_"*50)
    print("Penalties")
    for i in penalties:
        print(i)
    print("_"*50)
    print("Initial Beliefs")
    for i in initial_beliefs:
        print(i)
    print("_"*50)
    print("Lambda Range")
    print(lambda_range) 

    sns.pairplot(df)
    plt.show()

    defender.graph_distribution_lambda()
    '''

    #summary stats game rounds 
    averages = averages_for_rounds.copy()
    df = pd.DataFrame(averages)
    print(df)
    print("_"*50)
    print("Descriptive Statistics:")
    print(df.describe())
    print("_"*50)
    print("Correlation:")
    print(df.corr())
    print("_"*50)
    print("Covariance:")
    print(df.cov())
    print("_"*50)
    print("Variances:")
    print(df.var()) 
    print("_"*50)
    print("Standard Deviation:")
    print(df.std())
    print("_"*50)
    print("Mean:")
    print(df.mean())
    print("_"*50)
    print("Median:")
    print(df.median())
    print("_"*50)
    print("Mode:")
    print(df.mode())
    print("_"*50)
    print("Skewness:")
    print(df.skew())
    print("_"*50)
    print("Kurtosis:")
    print(df.kurt())
    print("_"*50)
    average_avg_poa = sum(poa_results_avg)/len(poa_results_avg)
    print("_"*50) 
    print(f"Average Percent System Working Optimally: {1/average_avg_poa}")
    print("_"*50)

    #graph POA
    plt.plot(averages['poa'].keys(), averages['poa'].values())
    plt.xlabel('Game Rounds')
    plt.ylabel('Price of Anarchy')
    plt.title('Price of Anarchy Value Over Time')

    #graph potential function value
    plt.plot(averages['potential_function'].keys(), averages['potential_function'].values())
    plt.xlabel('Game Rounds')
    plt.ylabel('Potential Function Value')
    plt.title('Potential Function Value Over Time')

    #defender lambda
    plt.plot(averages['lambda'].keys(), averages['lambda'].values())
    plt.xlabel('Game Rounds')
    plt.ylabel('Lambda Value')
    plt.title('Lambda Value Over Time')

    #defender utility
    plt.plot(averages['defender_utility'].keys(), averages['defender_utility'].values())
    plt.xlabel('Game Rounds')
    plt.ylabel('Defender Utility')
    plt.title('Defender Utility Over Time')

    #graph percent system working optimally
    plt.plot(averages['percent_optimal'].keys(), averages['percent_optimal'].values())
    plt.xlabel('Game Rounds')
    plt.ylabel('Average Percent System Working Optimally')
    plt.title('Averge Percent System Working Optimally Over Time')

    #lambda vs PoA
    lambda_values = list(averages['lambda'].values())
    lambda_values = np.array(lambda_values)
    poa_values = list(averages['poa'].values())
    plt.scatter(lambda_values, poa_values)
    a, b = np.polyfit(lambda_values, poa_values, 1)
    plt.plot(lambda_values, a*lambda_values + b, color='red')
    plt.xlabel('Lambda Value')
    plt.ylabel('Price of Anarchy')
    plt.title('Lambda Value vs Price of Anarchy')

    #correlation heat map
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

    sns.pairplot(df)
    plt.show()

