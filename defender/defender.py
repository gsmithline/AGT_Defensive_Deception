import scipy.optimize as opt
import numpy as np
import math
import random
from scipy.stats import beta, norm, gamma
from scipy.integrate import quad, IntegrationWarning
import warnings


class Defender:
    def __init__(self, num_targets, initial_beliefs, lambda_bound):
        self.num_targets = num_targets
        self.beliefs_congestion = initial_beliefs
        self.lambda_bound = lambda_bound
        self.mixed_strategy = [1 / num_targets] * num_targets
        self.lambda_bayes = beta(a=1, b=1) #between 0 and 1 to start

        #self.lambda_bayes = gamma(1) #between 0 and 1 to start
        self.lambda_value = self.lambda_bound #fix this later
        self.past_utilities = []
        
    
    def update_lambda_value(self, observed_potentials):
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
        new_lambda = expected_lambda / normalization_factor if expected_lambda > 0 else self.lambda_bayes.mean()
        
        updated_a = np.log(self.lambda_bayes.a + np.sum(observed_potentials))
        #updated_a = updated_a if updated_a > 0 else random.uniform(0, 1)
        print(f'updated_a: {updated_a}')
        updated_b = np.log(abs(self.lambda_bayes.b + len(observed_potentials) - np.sum(observed_potentials)))
        #updated_b = updated_b if updated_b > 0 else random.uniform(0, 1)
        print(f'updated_b: {updated_b}') 
        self.lambda_bayes = beta(updated_a, updated_b)
        #self.lambda_bayes = gamma(updated_a, updated_b)
        self.lambda_value = new_lambda
        
        return self.lambda_value


    def calculate_utility(self, game):
        # Reset total utility at the start of calculation
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

        self.expected_congestion = probabilities

        return probabilities

    

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
        


        return self.mixed_strategy


    def get_mixed_strategy(self):
        return self.mixed_strategy
