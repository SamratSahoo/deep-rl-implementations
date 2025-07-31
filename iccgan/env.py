from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg
import os
from utils import HUMANOID_CONFIG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from gymnasium import spaces
import numpy as np
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.lights import DomeLightCfg, spawn_light
import omni.kit.commands
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.utils import configclass
import torch
import isaacsim.core.utils.articulations as articulations_utils
import isaacsim.core.utils.prims as prims_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import UsdFileCfg
import omni.physics.tensors as tensors
import omni.kit.commands
import omni.usd
from pxr import UsdPhysics

@configclass
class ICCGANHumanoidEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 15
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
    robot_cfg: ArticulationCfg = HUMANOID_CONFIG.replace(prim_path="/World/envs/env_.*/humanoid/pelvis")

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2,
        env_spacing=4.0,
        replicate_physics=True
    )


class ICCGANHumanoidEnv(DirectRLEnv):
    cfg: ICCGANHumanoidEnvCfg
    
    def __init__(self, cfg: ICCGANHumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
    
    def _setup_scene(self):
        # Create ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Create humanoid in each environment using MJCF import
        enable_extension("isaacsim.asset.importer.mjcf")
        mjcf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.xml")
        
        # Build import config
        status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
        import_config.set_fix_base(False)
        import_config.set_make_default_prim(False)
        
        # Import humanoid in each environment
        for i in range(self.cfg.scene.num_envs):
            omni.kit.commands.execute(
                "MJCFCreateAsset",
                mjcf_path=mjcf_path,
                import_config=import_config,
                prim_path=f"/World/envs/env_{i}/humanoid"
            )
        
        # Remove nested articulation roots and apply to pelvis only
        stage = omni.usd.get_context().get_stage()
        for i in range(self.cfg.scene.num_envs):
            # Remove articulation root API from humanoid and worldBody
            for prim_path in [f"/World/envs/env_{i}/humanoid", f"/World/envs/env_{i}/humanoid/worldBody"]:
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid() and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            # Apply to pelvis
            pelvis_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/humanoid/pelvis")
            if pelvis_prim.IsValid():
                UsdPhysics.ArticulationRootAPI.Apply(pelvis_prim)
        
        # Create articulation object
        self.humanoid = Articulation(self.cfg.robot_cfg)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["humanoid"] = self.humanoid

        light_cfg = DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        spawn_light("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = self.action_scale * actions.clone()
    
    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        # TODO: Implement proper observations
        # For now, return dummy observations
        obs = torch.zeros((self.num_envs, 195), device=self.device)
        return {"policy": obs}
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments with the given indices."""
        # TODO: Implement proper reset
        pass
    
    def _get_rewards(self) -> torch.Tensor:
        """Get rewards from the environment."""
        # TODO: Implement proper rewards
        # For now, return zero rewards
        return torch.zeros(self.num_envs, device=self.device)

    def _apply_action(self):
        pass

    def _get_dones(self) -> dict:
        """Get termination flags from the environment."""
        # TODO: Implement proper terminations
        # For now, return no terminations
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)