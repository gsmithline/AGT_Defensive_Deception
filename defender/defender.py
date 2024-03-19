import scipy.optimize as opt
import numpy as np
import math
import random
from scipy.stats import beta, norm, gamma, gumbel_l, gumbel_r
from scipy.integrate import quad, IntegrationWarning
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog

np.random.seed(42)

class Defender:
    def __init__(self, num_targets, initial_beliefs, lambda_range=(0, 10), gamma_distribution=True, probability_distribution=True):
        self.num_targets = num_targets
        self.beliefs_congestion = initial_beliefs
        self.mixed_strategy = [1 / num_targets] * num_targets
        self.gamma_distribution = gamma_distribution    
        self.lambda_shape = 1  # Shape parameter (a) for the gamma distribution
        self.lambda_scale = 1
        self.learning_rate = 0.9
        self.probability_distribution = probability_distribution  # Scale parameter for the gamma distribution
        if gamma_distribution:
            self.lambda_bayes = gamma(a=self.lambda_shape, scale=self.lambda_scale)
        else:
            self.lambda_bayes = beta(a=1, b=1)

        self.past_lambda_values = [] #fix this later
        self.past_utilities = []
        self.best_response_utilities = []
        self.best_response_mixed_strategy = []
        self.lambda_min, self.lambda_max = lambda_range  # Set bounds for lambda
        #self.lambda_value = random.uniform(self.lambda_min, self.lambda_max) 
        self.lambda_value = self.lambda_min
        
    
    def update_lambda_value(self, observed_potentials, current_round, total_rounds):
        # Bayesian updating of lambda based on observed potentials
        ''''
        def likelihood_function(lambda_value):
            log_likelihood = np.sum([-lambda_value * potential for potential in observed_potentials])
            return np.exp(log_likelihood)
        '''
        def likelihood_function(lambda_value):
            likelihood = np.prod([np.exp(-lambda_value * potential) for potential in observed_potentials])
            return likelihood
        

        def full_bayesian_fraction(lambda_value):
            return likelihood_function(lambda_value) * self.lambda_bayes.pdf(lambda_value)
        
        
        normalization_factor, _ = quad(full_bayesian_fraction, 0, 1) #ensures posterior sums to 1
           
        expected_lambda, _ = quad(lambda x: x * full_bayesian_fraction(x), 0, 1)
        new_lambda = (expected_lambda / normalization_factor) if expected_lambda > 0 else self.lambda_bayes.mean()
        updated_shape = 0
        adjustment_factor = 1 + (current_round / total_rounds) * self.learning_rate
        if self.gamma_distribution:
            updated_shape = self.lambda_shape + len(observed_potentials)
        else: #keep shape at 1 or below
            updated_shape = max(0, min(1, self.lambda_shape + len(observed_potentials)))
        updated_scale = 1 / (1 / self.lambda_scale + np.sum(observed_potentials)) 
        if self.probability_distribution:
            if self.gamma_distribution:
                self.lambda_bayes = gamma(a=updated_shape, scale=updated_scale)
            else:
                self.lambda_bayes = beta(a=updated_shape, b=1)
        else:
            self.lambda_bayes = 1
        self.lambda_shape = updated_shape
        self.lambda_scale = updated_scale

        if expected_lambda > 0:
            new_lambda = max(self.lambda_bayes.ppf(0.99), expected_lambda / normalization_factor)  # ppf(0.99) approximates the maximum
        else:
            new_lambda = self.lambda_bayes.mean()
        self.lambda_value = new_lambda
        self.lambda_value = min(max(self.lambda_value * adjustment_factor, self.lambda_min), self.lambda_max)


        self.past_lambda_values.append(self.lambda_value)
        return self.lambda_value


    def calculate_utility(self, game):
        # Reset total utility at the start ofcalculation
        total_utility = 0

        # Iterate over each target in the game state
        for target in game.game_state.values():
            # Extract target-specific parameters
            c = target.congestion_cost
            P = target.reward
            R = target.penalty
            x = target.defender_strategy  # Defender's probability of defending this target
            n = target.congestion  # Congestion on this target

           
            utility = x * P - (1 - x) * R - c * (n ** 2)

            
            total_utility += utility

        self.past_utilities.append(total_utility)

        return total_utility

    def quantal_response(self, lambda_value, game):
        # Dictionary to store the exponentiated utilities for each attacker
        exp_utilities = {attacker_id: [] for attacker_id in game.attackers}

        # Iterate over each target to calculate utilities for each attacker
        for target in game.game_state.values():
            for attacker_id, attacker in game.attackers.items():
                # Calculate the attacker's utility for this target
                utility = attacker.calculate_expected_utility(target, self.mixed_strategy, game.attacker_strategy_profile)

                # Exponentiate the utility and append it to the list
                exp_utility = np.exp(lambda_value * utility)
                exp_utilities[attacker_id].append(exp_utility)

        # Normalize the exponentiated utilities to form a probability distribution for each attacker
        probabilities = {attacker_id: np.array(utilities) / np.sum(utilities) 
                        for attacker_id, utilities in exp_utilities.items()}

        # Aggregate probabilities across attackers for each target
        num_targets = len(game.game_state)
        aggregated_probabilities = np.zeros(num_targets)
        for attacker_id in probabilities:
            aggregated_probabilities += probabilities[attacker_id]

        # Update expected congestion
        self.expected_congestion = aggregated_probabilities

        return aggregated_probabilities


    

    #Using sequential least squares programming
    def optimize_strategy(self, targets, expected_congestion):
        c = [targets[j].congestion_cost for j in range(self.num_targets)]
        P = [targets[j].reward for j in range(self.num_targets)]
        R = [targets[j].penalty for j in range(self.num_targets)]

        def utility(x):
            utility_val = sum([x[j] * P[j] - (1 - x[j]) * R[j]**2 + c[j] * self.expected_congestion[j]**2
                               for j in range(self.num_targets)])
            return -utility_val  # Minimizing the negative utility to maximize utility

        constraints = [{'type': 'eq', 'fun': lambda x: sum(x) - 1}]
        bounds = [(0, 1) for _ in range(self.num_targets)]

        #normalize strategy
        initial_guess = [1 / self.num_targets] * self.num_targets

        #run optimization to find mixed strategy
        result = opt.minimize(utility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            self.mixed_strategy = {}
            for target in targets.values():
                target.defender_strategy = result.x[target.name] 
                self.mixed_strategy[target.name] = target.defender_strategy
        else:
            raise Exception("Optimization failed: " + result.message)
    
    def best_response(self, game):
        # Objective function: Minimize the negative of the defender's utility
        def objective(x):
            total_utility = 0
            for i, target in enumerate(game.game_state.values()):
                c = target.congestion_cost
                P = target.reward
                R = target.penalty
                n = target.congestion  # Assuming this can be calculated/updated beforehand
                utility = x[i] * P - (1 - x[i]) * R - c * (n ** 2)
                total_utility += utility
            return -total_utility  # Negate because we want to maximize utility
        
        # Constraints
        # Sum of probabilities (defender's strategy for each target) must be 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds for each decision variable (probability of defending each target)
        bounds = [(0, 1) for _ in range(self.num_targets)]
        
        # Initial guess for the strategy
        initial_guess = np.array([1 / self.num_targets] * self.num_targets)
        
        # Run optimization
        result = opt.minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            self.best_response_mixed_strategy = result.x
            print("Best response strategy found:", self.best_response_mixed_strategy)
            self.best_response_utilities.append(-result.fun)
            return self.best_response_mixed_strategy
        else:
            raise ValueError("Optimization failed: " + result.message)




    def get_mixed_strategy(self):
        return self.mixed_strategy
    
    def graph_distribution_lambda(self):
        past_lambdas = self.past_lambda_values
        sns.kdeplot(past_lambdas, fill=True)
        plt.show()