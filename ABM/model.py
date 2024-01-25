from mesa import Model
from mesa.time import RandomActivation
from agents import AttackerAgent, DefenderAgent, TargetAgent

class SecurityGameModel(Model):
    def __init__(self, num_targets, num_attackers):
        super().__init__()
        self.schedule = RandomActivation(self)
        # Initialize targets, attackers, and defenders here

    def step(self):
        self.schedule.step()
        # Additional steps per simulation tick (e.g., updating game state)
