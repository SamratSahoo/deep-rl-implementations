from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import env
import gymnasium as gym
from isaaclab.sim import SimulationCfg, SimulationContext
from env import ICCGANHumanoidEnv, ICCGANHumanoidEnvCfg
import torch

gym.register(
        id="Isaac-ICCGAN-v0",
        disable_env_checker=True,
        entry_point=f"{env.__name__}:ICCGANHumanoidEnv",
        kwargs={
            "cfg": ICCGANHumanoidEnvCfg(),
        },
    )

def main():
    env = gym.make("Isaac-ICCGAN-v0", disable_env_checker=True, render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, f"videos/")
    # env.recorded_frames = []
    env.reset()
    i = 0
    while simulation_app.is_running() and i < 1000:
        env.step(torch.randn(2, env.action_space.shape[0]))
        if i % 100 ==0:
            env.reset()
    
    env.close()

if __name__ == "__main__":
    main()
