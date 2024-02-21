import optuna
import random
import numpy as np
from target import Target
from defender.defender import Defender
from game.game import Game
from attackers.attacker import Attacker
from parameter_searching import run_simulation
from optuna.visualization import plot_optimization_history, plot_param_importances
np.random.seed(42)
results = {}
def objective(trial):
    # Define the hyperparameter space using the trial object
    lambda_start = trial.suggest_float('lambda_start', 0.5, 2.0)
    lambda_end = trial.suggest_float('lambda_end', lambda_start, 2, log=True)
    epsilon = trial.suggest_categorical('epsilon', [9])
    num_targets = trial.suggest_categorical('num_targets', [10])
    num_attackers = trial.suggest_categorical('num_attackers', [12])
    
    lambda_range = (lambda_start, lambda_end)
    avg_poa, avg_lambda, avg_potent_function, avg_defender_utility = run_simulation(lambda_range, epsilon, num_targets, num_attackers)
    
    '''
    try:
        avg_poa, avg_lambda, avg_potent_function, avg_defender_utility = run_simulation(lambda_range, epsilon, num_targets, num_attackers)
        results[(lambda_range, epsilon, num_targets, num_attackers)] = (avg_poa, avg_lambda, avg_potent_function, avg_defender_utility)
        if np.isnan(avg_poa) or np.isnan(avg_potent_function) or np.isnan(avg_defender_utility):
            return float('inf'), float('inf'), float('inf') 
    except Exception as e:
        print(f"Encountered an error during simulation: {e}")
        return float('inf'), float('inf'), float('inf') 
    
    '''
    return -avg_poa, avg_potent_function, -avg_defender_utility

sampler = optuna.samplers.TPESampler(seed=42)  # Make the optimization reproducible
study = optuna.create_study(directions=['minimize', 'minimize', 'minimize'], sampler=sampler)
study.optimize(objective, n_trials=5)  # Adjust n_trials to your computational budget

# Print the found hyperparameters
# Analyzing the results
study.trials_dataframe().dropna()
pareto_front_trials = study.best_trials
print("Pareto Front:")
for trial in pareto_front_trials:
    print(f"PoA: {-trial.values[0]}, Potential Function: {trial.values[1]}")
    print(f" Params: {trial.params}") 

# Visualize the optimization history

plot_optimization_history(study, target=objective)
plot_param_importances(study)

#print results
for key, value in results.items():
    print(f"Results for {key}: {value}")
