import scipy.optimize as opt
import numpy as np
import math
import random
from scipy.stats import beta, norm
from scipy.integrate import quad, IntegrationWarning
import warnings

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


    def update_game_state(self, new_game_state):
        self.game_state = new_game_state
    
    def update_defender(self, new_defender):
        self.defender = new_defender
    
    def update_attackers(self, new_attackers):
        self.attackers = new_attackers
    
    def run_congestion_game(self):
        #this runs the inner attacker congestion game
        pass

    def ibr_attackers_decayed_learning_rate(self, learning_rate):
        #computes epsilon nash for attackers
        pass

    #updateing game state after algorithm finishes
    def update_game_state(self):
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





        