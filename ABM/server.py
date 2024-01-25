from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization import ChartModule, CanvasGrid, Slider
from model import SecurityGameModel  # Adjust the import as necessary for your project structure
# Import your agents if they are used in the portrayal or elsewhere
from agents import AttackerAgent, DefenderAgent, TargetAgent  
import solara


# Define how to visually portray agents in the visualization
def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}
    if isinstance(agent, AttackerAgent):
        portrayal["Color"] = "red"
        portrayal["Layer"] = 0
    elif isinstance(agent, DefenderAgent):
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 1
    elif isinstance(agent, TargetAgent):
        portrayal["Color"] = "green"
        portrayal["Layer"] = 2
    return portrayal

# Example of a chart element
chart = ChartModule([{"Label": "StrategySuccess", "Color": "Black"}])

# Set up the server, using the new Slider for user-settable parameters
server = ModularServer(SecurityGameModel,
                       [CanvasGrid(agent_portrayal, 20, 20, 500, 500), chart],
                       "Security Game Simulation",
                       {"num_targets": Slider("Number of Targets", 50, 10, 100, 1),
                        "num_attackers": Slider("Number of Attackers", 5, 1, 10, 1)})

if __name__ == '__main__':
    server.launch()

