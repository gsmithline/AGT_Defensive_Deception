from target import Target
from defender.defender import Defender
from game.game import Game
from attackers.attacker import Attacker
import random
import numpy as np
import matplotlib.pyplot as plt

#this file will be the main file that will run the game
#Globals
'''
Inititalization of game 
_____________________________________________________________________________
'''
num_targets = 12
#random game set up 
game_rounds = 13
cogestion_costs = [random.randint(1, 5) for i in range(num_targets)]
rewards = [random.uniform(1, 5) for i in range(num_targets)]
penalties = [random.uniform(1, 5) for i in range(num_targets)]
#set up game and fill targets, this is what will be updated 
targets = {}
for i in range(num_targets):
    #filler 0 for now
    target = Target(i, cogestion_costs[i], rewards[i], penalties[i], 0, 0, 0)
    targets[target.name] = target  #add target to dictionary of targets
#set up game and fill targets, this is what will be updated
#set up defender
initial_beliefs = [random.uniform(1, 10) for i in range(num_targets)]

lambda_bound = .5
defender = Defender(num_targets, initial_beliefs, lambda_bound)
attackers = {}
#set up attackers
for i in range(1, 12):
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


#update lambda
#test ibr
game.ibr_attackers(1000)  
game.calculate_potential_function_value(1)
print(f"actual potential function value: {game.actual_potential_function_value}")
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
game.calculate_potential_function_value(2)
print(f"actual potential function value: {game.actual_potential_function_value}")
game.price_of_anarchy()
print(game.current_poa)
#calculate price of anarchy
#game.price_of_anarchy()
#print(game.current_poa)
for i in range(3, game_rounds):
    game.ibr_attackers(1000)  
    game.calculate_potential_function_value(1)
    print(f"actual potential function value: {game.actual_potential_function_value}")
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
    game.calculate_potential_function_value(i)
    print(f"actual potential function value: {game.actual_potential_function_value}")
    game.price_of_anarchy()
    print(game.best_potential_function_value)
    print(game.current_poa)

#graph POA
plt.plot(game.past_poa)
plt.xlabel('Game Rounds')
plt.ylabel('Price of Anarchy')
plt.title('Price of Anarchy Value Over Time')
plt.show()

#graph potential function value
plt.plot(game.past_potential_function_values.values())
plt.xlabel('Game Rounds')
plt.ylabel('Potential Function Value')
plt.title('Potential Function Value Over Time')
plt.show()



