import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from itertools import product
from target import Target
from defender.defender import Defender
from game.game import Game
from attackers.attacker import Attacker
game_rounds = 10    
def run_simulation(lambda_range, epsilon, num_targets, num_attackers, seed=42):
    np.random.seed(seed)
    congestion_costs = [random.randint(1, 10) for _ in range(num_targets)]
    rewards = [random.uniform(1, 10) for _ in range(num_targets)]
    penalties = [random.uniform(1, 10) for _ in range(num_targets)]
    initial_beliefs = [random.uniform(1, num_attackers + 1) for _ in range(num_targets)]

    targets = {i: Target(i, congestion_costs[i], rewards[i], penalties[i], 0, 0, 0) for i in range(num_targets)}
    defender = Defender(num_targets, initial_beliefs, lambda_range)
    attackers = {i: Attacker(num_targets, i) for i in range(1, num_attackers + 1)}


    game = Game(targets, rewards, congestion_costs, penalties, defender, attackers)

    poa_results_avg = []
    lambda_results_avg = []
    potent_function_results_avg = []
    defender_utility_results_avg = []
    percent_system_working_optimally_inner = []

    for round_number in range(1, game_rounds + 1):
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
        game.calculate_potential_function_value(round_number)
        print(f"actual potential function value: {game.actual_potential_function_value}")
        game.price_of_anarchy()
        print(game.best_potential_function_value)
        print(game.current_poa)
        percent_system_working_optimally_inner.append(1/game.current_poa)
    

    avg_poa = np.mean(poa_results_avg)
    avg_lambda = np.mean(lambda_results_avg)
    avg_potent_function = np.mean(potent_function_results_avg)
    avg_defender_utility = np.mean(defender_utility_results_avg)

    return avg_poa, avg_lambda, avg_potent_function, avg_defender_utility
'''
lambda_ranges = [(1, 3), (0, 2), (3, 3), (0, float('inf'))]
epsilons = [1, 5, 9]
num_targets_list = [10, 12]
num_attackers_list = [11, 12]

results = []
for lambda_range, epsilon, num_targets, num_attackers in product(lambda_ranges, epsilons, num_targets_list, num_attackers_list):
    result = run_simulation(lambda_range, epsilon, num_targets, num_attackers)
    results.append((lambda_range, epsilon, num_targets, num_attackers) + result)

columns = ['Lambda Range', 'Epsilon', 'Num Targets', 'Num Attackers', 'Avg POA', 'Avg Lambda', 'Avg Potential Function', 'Avg Defender Utility']
results_df = pd.DataFrame(results, columns=columns)

sns.heatmap(results_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

corr = results_df.corr()
strong_corrs = (corr > 0.5) | (corr < -0.5)
sns.heatmap(strong_corrs, annot=True, cmap='coolwarm')
plt.title('Strong Correlations')
plt.show()



sns.pairplot(results_df.describe().T[['mean', 'std']])
plt.show()

sns.boxplot(data=results_df, x='Lambda Range', y='Avg POA')
plt.title('Avg POA Across Different Lambda Ranges')
plt.show()

print(results_df.describe())
'''