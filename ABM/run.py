import argparse
from model import SecurityGameModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the security game simulation.")
    parser.add_argument('--num_targets', type=int, default=50, help='Number of targets')
    parser.add_argument('--num_attackers', type=int, default=5, help='Number of attackers')
    
    args = parser.parse_args()
    
    model = SecurityGameModel(num_targets=args.num_targets, num_attackers=args.num_attackers)
    model.run()  # Assuming your model class has a run method
