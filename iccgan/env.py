from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg
import os
from utils import HUMANOID_CONFIG, quaternion_to_euler
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
import matplotlib.pyplot as plt
from isaaclab.sensors import TiledCameraCfg, TiledCamera

@configclass
class ICCGANHumanoidEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 15
    action_scale = 1

    """
    Action Space: 36 dimensions
    - 8 spherical joints × 4D (axis-angle quaternion) = 32 dimensions
      * Abdomen (abdomen_x, abdomen_y, abdomen_z)
      * Neck (neck_x, neck_y, neck_z)
      * Right Shoulder (right_shoulder_x, right_shoulder_y, right_shoulder_z)
      * Left Shoulder (left_shoulder_x, left_shoulder_y, left_shoulder_z)
      * Right Hip (right_hip_x, right_hip_y, right_hip_z)
      * Left Hip (left_hip_x, left_hip_y, left_hip_z)
      * Right Ankle (right_ankle_x, right_ankle_y, right_ankle_z)
      * Left Ankle (left_ankle_x, left_ankle_y, left_ankle_z)
      
    - 4 revolute joints × 1D (angle in radians) = 4 dimensions
      * right_elbow, left_elbow, right_knee, left_knee
    """
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

    tiled_camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-7.0, 0.0, 3.0), rot=(0.9945, 0.0, 0.1045, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=80,
        height=80,
    )


class ICCGANHumanoidEnv(DirectRLEnv):
    cfg: ICCGANHumanoidEnvCfg
    
    def __init__(self, cfg: ICCGANHumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
    
    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        self.scene.clone_environments(copy_from_source=False)
        
        enable_extension("isaacsim.asset.importer.mjcf")
        mjcf_path = os.path.join(os.path.dirname(__file__), "assets", "humanoid.xml")
        
        status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
        import_config.set_fix_base(False)
        import_config.set_make_default_prim(False)
        
        for i in range(self.cfg.scene.num_envs):
            omni.kit.commands.execute(
                "MJCFCreateAsset",
                mjcf_path=mjcf_path,
                import_config=import_config,
                prim_path=f"/World/envs/env_{i}/humanoid"
            )
        
        stage = omni.usd.get_context().get_stage()
        for i in range(self.cfg.scene.num_envs):
            for prim_path in [f"/World/envs/env_{i}/humanoid", f"/World/envs/env_{i}/humanoid/worldBody"]:
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid() and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

            pelvis_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/humanoid/pelvis")
            if pelvis_prim.IsValid():
                UsdPhysics.ArticulationRootAPI.Apply(pelvis_prim)
        
        self.humanoid = Articulation(self.cfg.robot_cfg)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["humanoid"] = self.humanoid

        light_cfg = DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        spawn_light("/World/Light", light_cfg)

        self.tiled_camera = TiledCamera(self.cfg.tiled_camera_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = self.action_scale * actions.clone()

    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        body_states = self.humanoid.data.body_state_w  # Shape: (num_envs, 15, 13)
        
        # Reshape to (num_envs, 15 * 13) = (num_envs, 195)
        obs = body_states.reshape(self.num_envs, -1)        
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
        """Apply actions to the humanoid joints."""
        plt.imsave("test.png", self.tiled_camera.data.output["rgb"])
        spherical_joint_groups = [
            ['abdomen_x', 'abdomen_y', 'abdomen_z'],
            ['neck_x', 'neck_y', 'neck_z'],
            ['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z'],
            ['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z'],
            ['right_hip_x', 'right_hip_y', 'right_hip_z'],
            ['left_hip_x', 'left_hip_y', 'left_hip_z'],
            ['right_ankle_x', 'right_ankle_y', 'right_ankle_z'],
            ['left_ankle_x', 'left_ankle_y', 'left_ankle_z']
        ]
        
        revolute_joints = [
            'right_elbow',
            'left_elbow', 
            'right_knee',
            'left_knee'
        ]
        
        num_joints = self.humanoid.data.joint_pos_target.shape[1]
        
        joint_targets = torch.zeros((self.num_envs, num_joints), device=self.device)
        
        action_idx = 0
        joint_idx = 0
        
        for spherical_group in spherical_joint_groups:
            quat = self.actions[:, action_idx:action_idx + 4]  # (num_envs, 4)
            euler_angles = quaternion_to_euler(quat)
            
            for i in range(3):
                joint_targets[:, joint_idx] = euler_angles[:, i]
                joint_idx += 1
            
            action_idx += 4
        
        for revolute_joint in revolute_joints:
            joint_targets[:, joint_idx] = self.actions[:, action_idx]
            joint_idx += 1
            action_idx += 1
        
        # Apply joint position targets
        self.humanoid.data.joint_pos_target = joint_targets

    def _get_dones(self) -> dict:
        """Get termination flags from the environment."""
        # TODO: Implement proper terminations
        # For now, return no terminations
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)