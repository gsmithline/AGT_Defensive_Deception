import scipy.optimize as opt
import numpy as np
import math
import random
from scipy.stats import beta, norm, lognorm
from scipy.integrate import quad, IntegrationWarning
import warnings
from scipy.optimize import minimize
from scipy.optimize import Bounds
np.random.seed(42)
class Attacker:
    def __init__(self, num_targets, attack_id):
        #value = np.random.rand(num_targets)
        self.current_strategy = np.ones(num_targets) / num_targets # value / np.sum(value) dict with mixed strategy at each target labeld for house keeping
        self.expected_utilities = 0 #some float from utility calculation
        self.actual_utility = 0 #some float from utility calculation
        self.past_expected_utilities = []
        self.past_actual_utilities = []
        self.attack_id = attack_id #some int randomly assigned
    
    def update_strategy(self, new_strategy):
        self.current_strategy = new_strategy
    
    def update_expected_utilities(self, new_expected_utilities):
        self.expected_utilities = new_expected_utilities
    
    def update_actual_utility(self, new_actual_utility):
        self.actual_utility = new_actual_utility
    
    def update_past_expected_utilities(self, new_past_expected_utilities):
        self.past_expected_utilities = new_past_expected_utilities
    
    def update_past_actual_utilities(self, new_past_actual_utilities): 
        self.past_actual_utilities = new_past_actual_utilities
    
    def actually_calc_utility(self, game):
        utilility = 0
        for target in game.game_state.values():
            utilility += self.calculate_expected_utility(target, game.defender.mixed_strategy, game.attacker_strategy_profile)
        self.actual_utility = utilility
        self.past_actual_utilities.append(utilility)
        

    def calculate_expected_utility(self, target, defender_strategy, all_strategies): 
        R_j = target.reward
        P_j = target.penalty
        c_j = target.congestion_cost
        hat_x_j = target.defender_strategy
        
        # Calculate n_j as the sum of probabilities targeting j from all attackers' strategies
        attackers_at_target = [attacker_strategy[target.name] for attacker_strategy in all_strategies.values()]
        n_j = sum(attackers_at_target)
        
        utility = (1 - hat_x_j) * R_j**2 - hat_x_j * P_j - c_j * n_j**2
        return utility

    def optimize_mixed_strategy(self, game, POA = False):
        # Define the objective function (negative of expected utility to minimize)
        def objective(strategy):
            utilities = np.array([
                self.calculate_expected_utility(target, game.defender.mixed_strategy, game.attacker_strategy_profile)
                for target in game.game_state.values()
            ])
            return -np.dot(strategy, utilities)  # Maximize utility by minimizing its negative
        
        # Constraints: Probabilities must sum to 1
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds for each decision variable (probability between 0 and 1), normalize 
        bounds = [(0, 1) for _ in range(len(game.game_state))]
        
        # Solve the optimization problem
        result = minimize(objective, self.current_strategy, method='SLSQP', bounds=bounds, constraints=cons)
        
        if result.success:
            #normalize strategy
            if POA != True:
                self.current_strategy = result.x / np.sum(result.x)
                game.attacker_strategy_profile[self.attack_id] = self.current_strategy

        else:
            print("Optimization failed:", result.message)

    def get_mixed_strategy(self):
        return self.current_strategy
    
