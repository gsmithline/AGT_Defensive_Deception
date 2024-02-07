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
    
    def ibr_attackers(self, max_iterations, epsilon=10):
        # Computes epsilon Nash for attackers
        # This is the attackers' actual congestion game
        none_can_deviate = False
        attackers_strategies = set()
        iteration = 0

        while iteration < max_iterations and not none_can_deviate:
            game_set = {attacker for attacker in self.attackers.values() if attacker not in attackers_strategies}

            if len(game_set) == 0:
                none_can_deviate = True
                break

            while game_set:
                attacker = random.choice(list(game_set))
                print(f"Attacker {attacker.attack_id} optimizing strategy")
                old_game_state = copy.deepcopy(self.game_state)
                current_utility = copy.deepcopy(attacker.actual_utility)
                current_strategy = copy.deepcopy(attacker.current_strategy)
                attacker.optimize_mixed_strategy(self)
                self.update_game_state_new()
                attacker.actually_calc_utility(self)
                new_utility = attacker.actual_utility
                print(f"Checking if Attacker {attacker.attack_id} can deviate from old utility: {current_utility} to new utility: {new_utility}")
                if (new_utility - current_utility <= epsilon and new_utility > current_utility): #came to epsilon strat 
                    print(f"Attacker {attacker.attack_id} did deviate from old utility: {current_utility}, to new utility: {new_utility}")
                    print(f"Attacker {attacker.attack_id} has converged at epsilon-strat, removing from game set")
                     # Allow small deviations
                    attackers_strategies.add(attacker)
                    game_set.remove(attacker)
                elif new_utility > current_utility and new_utility - current_utility > epsilon: # big deviation, keep new strategy, but still in game set
                    print(f"Attacker {attacker.attack_id} could deviate from old utility: {current_utility}, to new utility: {new_utility}")
                    print(f"Attacker {attacker.attack_id} could still deviate more, staying in game set")
                elif current_utility - new_utility <= epsilon and current_utility >= new_utility: #current is actually a nash we remove from game set and switch back to old strategy
                    print(f"Attacker {attacker.attack_id} did not deviate from old utility: {current_utility}, to new utility: {new_utility}")
                    print(f"Attacker {attacker.attack_id} already converged at epsilon-strat, removing from game set")
                    attacker.update_strategy(current_strategy)
                    attacker.update_actual_utility(current_utility)
                    self.update_game_state(old_game_state)
                    attackers_strategies.add(attacker)
                    game_set.remove(attacker)
                    # Revert changes if no improvement or outside ε range
                elif new_utility <= current_utility and current_utility - new_utility > epsilon:  # Revert changes if no improvement or outside ε range
                    print(f"Attacker {attacker.attack_id} did not deviate from old utility: {current_utility}, to new utility: {new_utility}")
                    print(f"Attacker {attacker.attack_id} did not converge, new strat MUCH WORSE, staying in game set")
                    attacker.update_strategy(current_strategy)
                    attacker.update_actual_utility(current_utility)
                    self.update_game_state(old_game_state)
                    game_set.remove(attacker)
                    attackers_strategies.add(attacker)

            iteration += 1

            # After updating all, check if convergence criteria met for all
            if len(attackers_strategies) == len(self.attackers):
                temp_attackers_strategies = attackers_strategies.copy()
                counter = 0  # Work with a copy to avoid modification issues during iteration
                while temp_attackers_strategies and counter < len(temp_attackers_strategies):
                    attacker = random.choice(list(temp_attackers_strategies))
                    old_game_state = copy.deepcopy(self.game_state)
                    current_utility = copy.deepcopy(attacker.actual_utility)
                    current_strategy = copy.deepcopy(attacker.current_strategy)
                    attacker.optimize_mixed_strategy(self)
                    attacker.actually_calc_utility(self)
                    new_utility = attacker.actual_utility
            
                    if new_utility - current_utility <= epsilon and new_utility > current_utility:
                        self.update_game_state_new()
                        none_can_deviate = True
                        counter += 1
                    elif current_utility - new_utility <= epsilon and current_utility > new_utility :
                        attacker.update_strategy(current_strategy)
                        attacker.update_actual_utility(current_utility)
                        self.update_game_state(old_game_state)
                        counter += 1
                        none_can_deviate = True
                    else:
                        none_can_deviate = False
                        print(f"Attacker {attacker.attack_id} can deviate from old utility: {current_utility}, to new utility: {new_utility}")
                        print(f"Attacker {attacker.attack_id} can still deviate, adding back to game set")
                        attacker.update_strategy(current_strategy)  
                        attacker.update_actual_utility(new_utility)
                        self.update_game_state_new()
                        game_set.add(attacker)
                        attackers_strategies.remove(attacker)
                        #temp_attackers_strategies.remove(attacker)
                        break
                if none_can_deviate:
                    # If none_can_deviate is still True after this check, then no attacker can improve by more than epsilon
                    print(f"None can deviate from their strategy")
                    print(f"Converged after {iteration} iterations") 
                    break  # Exit the loop as we've reached the desired convergence criterion

        if iteration == max_iterations and not none_can_deviate:
            print(f"Did not converge to epsilon-nash after {iteration} iterations, max iterations reached")
        else:
            print(f"Converged after {iteration} iterations, converged to epsilon-nash") 


    
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





        