from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg
import os
from utils import HUMANOID_CONFIG

@configclass
class ICCGANHumanoidEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length = 1000
    action_scale = 1

    # TODO: change action and observation spaces
    action_space = 1
    observation_space = 1
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)

    robot_cfg: ArticulationCfg = HUMANOID_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")



class ICCGANHumanoidEnv(DirectRLEnv):
    pass
