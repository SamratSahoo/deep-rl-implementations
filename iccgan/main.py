from isaaclab.app import AppLauncher
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Train ICCGAN Agents")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=500, help="Interval between video recordings (in steps).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import env
import gymnasium as gym
from isaaclab.sim import SimulationCfg, SimulationContext
from env import ICCGANHumanoidEnv, ICCGANHumanoidEnvCfg
import torch
import time

gym.register(
        id="Isaac-ICCGAN-v0",
        disable_env_checker=True,
        entry_point=f"{env.__name__}:ICCGANHumanoidEnv",
        kwargs={
            "cfg": ICCGANHumanoidEnvCfg(),
        },
    )

def main():
    env = gym.make("Isaac-ICCGAN-v0", disable_env_checker=True, 
        render_mode="rgb_array" if args_cli.video else None
    )

    log_root_path = os.path.join("logs", "iccgan")
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, time.strftime("%Y-%m-%d_%H-%M-%S"), "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env.reset()
    i = 0
    while simulation_app.is_running() and i < 1000:
        env.step(torch.randn(env.action_space.shape))
        if i % 100 ==0:
            env.reset()
        
        i +=1 
    
    env.close()

if __name__ == "__main__":
    main()
