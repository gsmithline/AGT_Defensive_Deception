from target import Target
from defender.defender import Defender
from game.game import Game
from attackers.attacker import Attacker
import random
import numpy
#this file will be the main file that will run the game
#Globals
num_targets = 50
#random game set up 
game_rounds = 200
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
#set up attackers
attacker1 = Attacker()
attacker2 = Attacker()
attacker3 = Attacker()
attacker4 = Attacker()
attacker5 = Attacker()
attackers = {1: attacker1, 2: attacker2, 3: attacker3, 4: attacker4, 5: attacker5}
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
    for target in game.game_state.values():
        attacker.calculate_expected_utility(target, defender.mixed_strategy, game.attacker_strategy_profile)
        print(attacker.expected_utilities)

print("stop here")

#test ibr
game.ibr_attackers(.0001, 1000)  
#calculate potential function
game.calculate_potential_function_value(1)
print(game.actual_potential_function_value)
#update lambda
defender.update_lambda_value(list(game.past_potential_function_values.values()))
print(defender.lambda_value)
#test qr defender 
defender.quantal_response(defender.lambda_value, game)
print(defender.expected_congestion)

#test strategy computation defender
defender.optimize_strategy(targets, defender.expected_congestion)
print(defender.mixed_strategy)
#test defender expected utility
defender.calculate_utility(game)
print(defender.past_utilities)

'''

#test bayesian
test_observed_potentuals = [random.uniform(1, 10) for i in range(num_targets)] 
defender.update_lambda_value(test_observed_potentuals)
print(defender.lambda_value)
for i in range(0, 1000):
    test_observed_potentuals.append(random.uniform(1, 10))
    defender.update_lambda_value(test_observed_potentuals)
    print("lambda value: ", defender.lambda_value)

#test strategy computation
test_qr = [random.uniform(-200, 100) for i in range(num_targets)]
defender.optimize_strategy(targets, test_qr)
print(defender.mixed_strategy)
#test update game state
for target in game.game_state.values():
    target.defender_strategy = defender.mixed_strategy[target.name]

#test defender utility calculation 
defender.calculate_utility(game)
print(defender.past_utilities)
    
#test attacker 
#test utility
for attacker in game.attackers.values():
    attacker.calculate_expected_utility(target, game.defender.mixed_strategy, game.attacker_strategy_profile)
    print(attacker.expected_utilities)
#test optimize strategy
for attacker in game.attackers.values():
    attacker.optimize_mixed_strategy(game)
    print(attacker.current_strategy)

#test potential function
game.calculate_potential_function_value(1)
print(game.actual_potential_function_value)

defender.update_lambda_value(list(game.past_potential_function_values.values()))
print(defender.lambda_value)
#test strategy computation
test_qr = [random.uniform(-200, 10000) for i in range(num_targets)]
defender.optimize_strategy(targets, test_qr)
print(defender.mixed_strategy)
#test update game state
for target in game.game_state.values():
    target.defender_strategy = defender.mixed_strategy[target.name]

#test defender utility calculation 
defender.calculate_utility(game)
print(defender.past_utilities)







#game loop 

#for i in range(game_rounds):



 
'''


