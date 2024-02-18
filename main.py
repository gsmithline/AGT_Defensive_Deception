from target import Target
from defender.defender import Defender
from game.game import Game
from attackers.attacker import Attacker
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#this file will be the main file that will run the game
#Globals
'''
Inititalization of game 
_____________________________________________________________________________
'''
np.random.seed(42)
num_targets = 10
#random game set up 
epsilon = 9
game_rounds = 10
num_attackers = 12
poa_results_avg = []
lambda_results_avg = []
potent_function_results_avg = []
defender_utility_results_avg = []
lambda_range=(1000, float('inf'))
cogestion_costs = [random.randint(1, 10) for i in range(num_targets)]
rewards = [random.uniform(1, 10) for i in range(num_targets)]
penalties = [random.uniform(1, 10) for i in range(num_targets)]
initial_beliefs = [random.uniform(1, num_attackers+1) for i in range(num_targets)]

for i in range(1, 11):
    #set up game and fill targets, this is what will be updated 
    targets = {}
    for i in range(num_targets):
        #filler 0 for now
        target = Target(i, cogestion_costs[i], rewards[i], penalties[i], 0, 0, 0)
        targets[target.name] = target  #add target to dictionary of targets
    #set up game and fill targets, this is what will be updated
    #set up defender
    defender = Defender(num_targets, initial_beliefs, lambda_range)
    attackers = {}
    #set up attackers
    for i in range(1, num_attackers + 1):
        attacker = Attacker()
        attacker.attack_id = i
        attackers[i] = attacker
    #set up gam
    game = Game(targets, rewards, cogestion_costs, penalties, defender, attackers) #fill with attackers as well
    #uodate congestion of each target from game attacker stratwgy profile
    for id in game.attacker_strategy_profile:
        attacker = game.attacker_strategy_profile[id]
        for i in range(0, len(attacker)):
            game.game_state[i].congestion += attacker[i]
            #print(game.game_state[i].congestion)
        #update attacker strategy to initial
        game.attackers[id].update_strategy(attacker) 
    for id, attacker  in attackers.items():
        attacker.attack_id = id
        attacker.actually_calc_utility(game)
    '''
    Inititalization of game 
    _____________________________________________________________________________
    '''

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

    average_defender_utility = sum(defender.past_utilities)/len(defender.past_utilities)
    defender_utility_results_avg.append(average_defender_utility)
    averge_poa = sum(game.past_poa)/len(game.past_poa) 
    poa_results_avg.append(averge_poa)
    avgerage_lambda = sum(defender.past_lambda_values)/len(defender.past_lambda_values)
    lambda_results_avg.append(avgerage_lambda)
    avgerage_potential_function = sum(game.past_potential_function_values.values())/len(game.past_potential_function_values.values())
    potent_function_results_avg.append(avgerage_potential_function)

df = pd.DataFrame({'POA': poa_results_avg, 'Lambda': lambda_results_avg, 'Potential Function': potent_function_results_avg, 'Defender Utility': defender_utility_results_avg})
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
plt.title('Price of Anarchy Value Over Time')
plt.show()

#graph potential function value
plt.plot(potent_function_results_avg)
plt.xlabel('Game Rounds')
plt.ylabel('Potential Function Value')
plt.title('Potential Function Value Over Time')
plt.show()

#defender lambda
plt.plot(lambda_results_avg)
plt.xlabel('Game Rounds')
plt.ylabel('Lambda Value')
plt.title('Lambda Value Over Time')
plt.show()


#defender utility
plt.plot(defender_utility_results_avg)
plt.xlabel('Game Rounds')
plt.ylabel('Defender Utility')
plt.title('Defender Utility Over Time')
plt.show()




