import scipy.optimize as opt
import numpy as np
import math
import random
from scipy.stats import beta, norm
from scipy.integrate import quad, IntegrationWarning
import warnings
import random
import copy
class Game:
    def __init__(self, targets, rewards, congestions, penalties, defender, attackers):
        self.defender = defender
        self.attackers = attackers #dict of attackers
        self.game_state = targets #array of filled out targets
        self.expected_potential_function_value = None #some scalar value for the potential function
        self.actual_potential_function_value = None #some scalar value for the potential function
        self.past_potential_function_values = {} #dictionary of past potential function values, key is game state ie. iteration number in game
        self.attacker_strategy_profile = {attacker_id: np.ones(len(targets)) / len(targets) 
                                          for attacker_id in attackers}
        self.average_potential_for_attacker = None #some scalar value for the potential function
        self.past_poa = []
        self.current_poa = None


    def update_game_state(self, new_game_state):
        self.game_state = new_game_state
    
    def update_defender(self, new_defender):
        self.defender = new_defender
    
    def update_attackers(self, new_attackers):
        self.attackers = new_attackers
    
    def run_congestion_game(self):
        #this runs the inner attacker congestion game

        pass
    def ibr_attackers(self, max_iterations, epsilon = 1e-100):
        #computes epsilon nash for attackers
        #this is the attackers actual congestion game
        none_can_deviate = False
        attackers_strategies = set()
        for i in range(max_iterations):
            # Update attacker strategies
            #add all attacker to the game set
            game_set = set()
            for attacker in self.attackers.values():
                if attacker not in attackers_strategies:
                    game_set.add(attacker) 
            
            for i in range(len(game_set)):
                attacker = random.choice(list(game_set)) #randomly select attacker
                if len(attackers_strategies) == len(self.attackers) and none_can_deviate == True:    
                    break
                if attacker not in attackers_strategies:
                    old_game_state = copy.deepcopy(self.game_state) #save old game state
                    print(f"Attacker {attacker.attack_id} optimizing strategy")
                    current_utility = copy.deepcopy(attacker.expected_utilities) #save old utility
                    current_strategy = copy.deepcopy(attacker.current_strategy) #save old strategy
                    attacker.optimize_mixed_strategy(self) #optimize attacker strategy
                    self.update_game_state_new() #update game state after attacker strategy update
                    for target in self.game_state.values():
                        attacker.calculate_expected_utility(target, self.defender.mixed_strategy, self.attacker_strategy_profile)
                    attacker.update_actual_utility(attacker.expected_utilities)
                    attacker.update_expected_utilities(0) #update expected utility after
                    new_utility = attacker.actual_utility #get new utility
                    print(f"Checking if Attacker {attacker.attack_id} can deviate from old utility: {current_utility} to new utility: {new_utility}")
                    if new_utility < current_utility or new_utility - current_utility < 0: # dont update strategy best choice is less than current, revert back
                        attacker.update_strategy(current_strategy)
                        attacker.update_expected_utilities(current_utility)
                        self.update_game_state(old_game_state) #revert back to old game state
                        game_set.remove(attacker) #remove attacker from game set
                        print(f"Attacker {attacker.attack_id} could not deviate, removing from game set in this round")
                    else:
                        game_set.remove(attacker)
                        print(f"Attacker {attacker.attack_id} did deviate from old utility: {current_utility}, to new utility: {new_utility}")

                    print(f"Attacker {attacker.attack_id} old utility: {current_utility}, new utility: {new_utility}")
                    if new_utility - current_utility < epsilon:
                        attackers_strategies.add(attacker)
                        print(f"Attacker {attacker.attack_id} has converged, removing from game set")
                        if attacker in game_set:
                            game_set.remove(attacker)
                        
            if len(attackers_strategies) == len(self.attackers):
                print(f"Checking none can deviate to better strategy")
                counter = 0
                for attacker in  attackers_strategies:
                        current_utility = attacker.expected_utilities
                        current_strategy = attacker.current_strategy
                        attacker.optimize_mixed_strategy(self)
                        new_utility = attacker.expected_utilities
                        if new_utility <= current_utility or (new_utility - current_utility < epsilon and new_utility - current_utility > 0):
                            #attacker.update_strategy(current_strategy)
                            counter += 1
                        else:
                            print(f"Attacker {attacker.attack_id} could deviate from old utility: {new_utility}, to new utility: {current_utility}")
                            attackers_strategies.remove(attacker)
                            break
                if counter == len(attackers_strategies):
                    none_can_deviate = True
                    print(f"None can deviate from their strategy")
                    break
                else:
                    print(f"Some attackers can deviate from their strategy")
                   
        print(f"Converged after {i} iterations")

    '''
    def price_of_anarchy(self):
        #computes price of anarchy for the attackers for the game
        #this happens after each round and we have perfect information of the game
        computed_poa = self.actual_potential_function_value
        #optimal potential function value
        #now we have defender strategy, compute optimal potential function value
        optimal_potential_function_value = 0
        for target in self.game_state.values():
            # For each target, iterate over each attacker
            for attacker_id, attacker in self.attackers.items():
                # Probability of this attacker targeting this target
                y_ij = attacker.current_strategy[target.name]
                # Utility for attacker i when choosing target j
                # Assuming calculate_expected_utility function calculates U_{ij} for a given target
                U_ij = attacker.calculate_expected_utility(target, self.defender.mixed_strategy, self.attacker_strategy_profile)
                # Add to the potential function value
                optimal_potential_function_value += y_ij * U_ij

        pass     
   
    def calculate_social_optimum(self):
    # Setup and solve the optimization problem to find the social optimum
    # This is highly dependent on your game's specifics
        defender_strategy = self.defender.mixed_strategy


    pass

    def price_of_anarchy(self):
        optimal_potential_function_value = self.calculate_social_optimum()
        poa = optimal_potential_function_value / self.actual_potential_function_value if self.actual_potential_function_value else float('inf')
        return poa
    '''
    #updateing game state after algorithm finishes
    def update_game_state_new(self):
        # Example method to update the game state based on current strategies
        for target_id, target in self.game_state.items():
            # Update defender strategy for each target
            #defender mixed strategy is array of probabilities for each target

            target.update_defender_strategy(self.defender.mixed_strategy[target_id-1])
            
            # Update attacker strategies for each target
            attacker_strategies = {attacker_id: attacker.current_strategy[target_id] for attacker_id, attacker in self.attackers.items()}
            target.update_attacker_strategies(attacker_strategies)
            # Calculate new congestion based on attacker strategies
            target.congestion = sum(attacker_strategies.values())
    
    #should be called after attackers play their strategies
    def calculate_potential_function_value(self, game_state):
        potential_function_value = 0
        # Iterate over each target in the game state
        for target in self.game_state.values():
            # For each target, iterate over each attacker
            for attacker_id, attacker in self.attackers.items():
                # Probability of this attacker targeting this target
                y_ij = attacker.current_strategy[target.name]
                # Utility for attacker i when choosing target j
                # Assuming calculate_expected_utility function calculates U_{ij} for a given target
                U_ij = attacker.calculate_expected_utility(target, self.defender.mixed_strategy, self.attacker_strategy_profile)
                # Add to the potential function value
                potential_function_value += y_ij * U_ij
        self.actual_potential_function_value = potential_function_value
        self.past_potential_function_values[game_state] = potential_function_value
        self.average_potential_for_attacker = potential_function_value / len(self.attackers)
        return potential_function_value





        