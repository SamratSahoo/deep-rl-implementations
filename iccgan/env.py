from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg
import os
from utils import HUMANOID_CONFIG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from gymnasium import spaces
import numpy as np
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.lights import DomeLightCfg
from isaaclab.utils import configclass
import torch
import isaacsim.core.utils.articulations as articulations_utils
import isaacsim.core.utils.prims as prims_utils


@configclass
class ICCGANHumanoidEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 1000
    action_scale = 1

    # TODO: change action and observation spaces
    action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,))
    """
    15 Links x 13 Observations per link = 195 observation space
    [ 
        x, y, z,         # Position (3)
        qx, qy, qz, qw,  # Orientation (4)
        vx, vy, vz,      # Linear velocity (3)
        wx, wy, wz       # Angular velocity (3)
    ]
    """
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(195,))
    sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)
    robot_cfg: ArticulationCfg = HUMANOID_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2,
        env_spacing=4.0,
        replicate_physics=True
    )


class ICCGANHumanoidEnv(DirectRLEnv):
    cfg: ICCGANHumanoidEnvCfg
    
    def __init__(self, cfg: ICCGANHumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        self.current_setup_num = 0
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
    
    def _setup_scene(self):
        self.humanoid = Articulation(self.cfg.robot_cfg)
        humanoid_usd_prim = prims_utils.get_prim_at_path(
            prim_path=f"/World/envs/env_{self.current_setup_num}/Robot"
        )
        articulations_utils.add_articulation_root(prim=humanoid_usd_prim)
        self.current_setup_num += 1

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["humanoid"] = self.humanoid

        light_cfg = DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = self.action_scale * actions.clone()