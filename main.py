from target import Target
from defender.defender import Defender
from game.game import Game
from attackers.attacker import Attacker
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats 


'''
Inititalization of game 
_____________________________________________________________________________
'''
np.random.seed(42)
num_targets = 12
#random game set up 
epsilon = 10
game_rounds = 4
num_attackers = 12
num_targets = 10
num_games = 100

#lambda_ranges = [(i/10, (i+1)/10) for i in range(10)]
lambda_ranges = [(i/10, (i/10) + 0.2) for i in range(0, 10, 2)]
lambda_ranges.append((0, float('inf')) ) #no bounds
congestion_costs = [random.randint(1, 10) for i in range(num_targets)]
print(f"congestion costs: {congestion_costs}")
rewards = [random.randint(1, 10) for i in range(num_targets)]
print(f"rewards: {rewards}")
penalties = [random.randint(1, 10) for i in range(num_targets)]
print(f"penalties: {penalties}")
initial_beliefs = np.ones(num_targets) / num_targets
print(f"initial beliefs: {initial_beliefs}")
columns = ['Lambda Range', 'Game Number', 'Game Round', 'Price of Anarchy', 'Potential Function Value', 'Optimal Potential Function Value', 'Attacker Mixed Strategy Profile', 
           'Attacker Optimal Mixed Strategy Profile', 'Defender Utility', 'Defender Mixed Strategy', 'Percent System Working Optimally', 
           'Defender Best Response Utility', 'Defender Best Response Mixed Strategy', 'Distance Between Defender and Actual', 'Lambda Value']
results = pd.DataFrame(columns=columns)
new_rows = []
for lambda_range in lambda_ranges:
   
    print(f"lambda range: {lambda_range}")
    for l in range(1, num_games + 1):
        targets = {j: Target(j, congestion_costs[j], rewards[j], penalties[j], 0, 0, 0) for j in range(num_targets)}
        
        defender = Defender(num_targets, initial_beliefs, lambda_range, gamma_distribution=True, probability_distribution=True)
        attackers = {k: Attacker(num_targets, k) for k in range(1, num_attackers + 1)}  # Corrected loop var

        game = Game(targets, rewards, congestion_costs, penalties, defender, attackers)
        '''
        Inititalization of game 
        _____________________________________________________________________________
        '''
        percent_system_working_optimally_inner = []
        
        for i in range(1, game_rounds + 1):
            #update lambda
            if i > 1:
                defender.update_lambda_value(list(game.past_potential_function_values.values()))
                #defender.update_lambda_value(game.percent_system_working_optimally, current_round=i, total_rounds=game_rounds)
            print(f"lambda value updated: {defender.lambda_value}")
            
            #test qr defender 
            defender.quantal_response(defender.lambda_value, game)
            print(f"expected congestion updated: {defender.expected_congestion}")
            #test strategy computation defender
            defender.optimize_strategy(targets, defender.expected_congestion)
            print(f"new defender mixed strategy updated: {defender.mixed_strategy}")
            #test defender expected utility
            defender.calculate_utility(game)
            print(f"new past utilities updated: {defender.past_utilities[-1]}")
            #test best response
            defender.best_response(game)
            print(f"best response utility: {defender.best_response_utilities[-1]}")
            game.difference_defender_utilities()
            game.ibr_attackers(1000, epsilon)  
            game.calculate_potential_function_value(i)
            print(f"actual potential function value: {game.actual_potential_function_value}")
            game.price_of_anarchy()
            print(game.best_potential_function_value)
            print(game.current_poa)
            
            new_row = {'Lambda Range': lambda_range, 'Game Number': l, 'Game Round': i, 
                       'Price of Anarchy': game.current_poa, 'Potential Function Value': game.actual_potential_function_value, 
                       'Optimal Potential Function Value': game.best_potential_function_value, 
                       'Attacker Mixed Strategy Profile': game.attacker_strategy_profile, "Attacker Optimal Mixed Strategy Profile": game.social_optimum_strategy,
                       'Defender Utility': defender.past_utilities[-1], 'Defender Mixed Strategy': defender.mixed_strategy, 
                       'Percent System Working Optimally': 1/game.current_poa, 'Defender Best Response Utility': defender.best_response_utilities[-1],
                       'Defender Best Response Mixed Strategy': defender.best_response_mixed_strategy, 
                       'Distance Between Defender and Actual': game.diff_in_utilities_defender[-1],
                       'Lambda Value': defender.lambda_value, 'Composite Score': game.current_composite_score}
            new_rows.append(new_row)
            

if new_rows: 
    new_rows_df = pd.DataFrame(new_rows)
    results = pd.concat([results, new_rows_df], ignore_index=True)

results.to_csv('using_composite_of_PSWO_Potential.csv')