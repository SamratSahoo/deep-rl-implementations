from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_

import gym

# gym.register(
#         id="Isaac-ICCGAN-v0",
#         entry_point=f"{__name__}.env:ICCGANHumanoidEnv",
#         disable_env_checker=True,
#         kwargs={
#             "env_cfg_entry_point": f"{__name__}.env:ICCGANHumanoidEnvCfg",
#         },
#     )

from isaaclab.sim import SimulationCfg, SimulationContext
from env import ICCGANHumanoidEnv, ICCGANHumanoidEnvCfg
import torch

def main():
    # env = gym.make("Isaac-ICCGAN-v0")
    env = ICCGANHumanoidEnv(ICCGANHumanoidEnvCfg())
    while simulation_app.is_running():
        env.step(torch.randn(env.num_envs, env.action_space.shape[0]))
    
    env.close()

if __name__ == "__main__":
    main()
