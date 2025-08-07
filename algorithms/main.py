from algorithms.ddpg import DDPG
from algorithms.reinforce import Reinforce
import gymnasium as gym
import argparse
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

if __name__ == "__main__":
    environment = gym.make("CarRacing-v3", render_mode="human")

    # Args Parse to choose algorithm
    parser = argparse.ArgumentParser(
        description="run samrat's custom implementations of different deep reinforcement learning algorithms!"
    )

    parser.add_argument(
        "--algorithm",
        help="choose an algorithm to run",
        default="ddpg",
        choices=["ddpg", "reinforce"],
    )

    args = parser.parse_args()

    if args.algorithm == "ddpg":
        agent = DDPG(environment)
        agent.train()
    elif args.algorithm == "reinforce":
        agent = Reinforce(environment)
        agent.train()
