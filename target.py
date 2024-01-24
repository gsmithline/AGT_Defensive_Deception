
class Target:
    def __init__(self, name, congestion_cost, reward, penalty, congestion, defender_strategy, attacker_strategies):
        self.name = name
        self.congestion_cost = congestion_cost #constant congestion cost
        self.reward = reward #constant reward for being defended
        self.penalty = penalty #constant penalty for being attacked
        self.congestion = congestion #sum of attacker strategies
        self.defender_strategy = defender_strategy #defender strategy
        #dictionary of attacker strategies and their respective probabilities
        
    def request(self):
        self.__adaptee.specific_request()

    def update_defender_strategy(self, new_defender_strategy):
        self.defender_strategy = new_defender_strategy
    
    
    #should be a dict {attacker: probability}
    def update_attacker_strategies(self, new_attacker_strategies):
        self.attacker_strategies = new_attacker_strategies
    
    
    

    
    
    
    