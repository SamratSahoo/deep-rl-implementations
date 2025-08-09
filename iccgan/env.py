from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg
import os
from typing import List
from rl_utils import HUMANOID_CONFIG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from gymnasium import spaces
import numpy as np
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.lights import DomeLightCfg, spawn_light
import omni.kit.commands
from isaacsim.core.utils.extensions import enable_extension
from isaaclab.utils import configclass
import torch
import torch.nn.functional as F
import omni.kit.commands
import omni.usd
from pxr import UsdPhysics
from reference_motion import ReferenceMotion
from isaaclab.sensors import TiledCameraCfg, TiledCamera
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from isaaclab.sim.spawners.sensors import PinholeCameraCfg
import isaacsim.core.utils.numpy.rotations as rot_utils
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
    sim: SimulationCfg = SimulationCfg(dt=1 / 30, render_interval=decimation)
    robot_cfg: ArticulationCfg = HUMANOID_CONFIG.replace(prim_path="/World/envs/env_.*/humanoid/pelvis")

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2,
        env_spacing=4.0,
        replicate_physics=True
    )

    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-12.0, 0.0, 5.5), rot=(0.98007, 0.0, 0.19867, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=PinholeCameraCfg(
            focal_length=12.0, focus_distance=800.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=1920,
        height=1080,
    )


class ICCGANHumanoidEnv(DirectRLEnv):
    cfg: ICCGANHumanoidEnvCfg
    
    def __init__(self, cfg: ICCGANHumanoidEnvCfg, render_mode: str | None = None, motion_file: str = "assets/motions/run.json", num_envs: int = 2, **kwargs):
        cfg.scene.num_envs = num_envs
        super().__init__(cfg, render_mode, **kwargs)
        self.action_scale = self.cfg.action_scale
        self.motion_file = motion_file

        self.episode_step = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.reference_motion = None

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(200,200)))
        
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

        self.tiled_camera = TiledCamera(self.cfg.tiled_camera)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = self.action_scale * actions.clone()
        self.episode_step += 1

    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        body_states = self.humanoid.data.body_state_w  # (num_envs, 15, 13)
        
        # Reshape to (num_envs, 15 * 13) = (num_envs, 195)
        obs = body_states.reshape(self.num_envs, -1)        
        return {"policy": obs}
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments with the given indices."""
        if self.reference_motion is None:
            self.body_mapping = self._create_body_mapping()
            self.reference_motion = ReferenceMotion(self.motion_file, self.body_mapping)
            # Parse reference frames once so sampling returns fixed-length sequences
            self.reference_motion.parse_frames()
            self.contactable_links = self.reference_motion.contactable_links
            self.clip_length = torch.full((self.num_envs,), self.reference_motion.clip_length, device=self.device)
            self.is_cyclic = torch.full((self.num_envs,), 1 if self.reference_motion.is_cyclic else 0, dtype=torch.bool, device=self.device)

        self.episode_step[env_ids] = 0
        
        super()._reset_idx(env_ids)
        
        joint_pos = self.humanoid.data.default_joint_pos[env_ids]
        joint_vel = self.humanoid.data.default_joint_vel[env_ids]
        
        default_root_state = self.humanoid.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids].to(self.device)
        
        self.humanoid.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.humanoid.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.humanoid.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    
    def _get_rewards(self) -> torch.Tensor:
        # We handle rewards in the PPO class
        return torch.zeros(self.num_envs, device=self.device)

    def _apply_action(self):
        """Apply actions to the humanoid joints."""

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
        dof_idx = 0

        # Spherical joints: convert 4D quaternion to 3D expmap
        for _ in spherical_joint_groups:
            q = self.actions[:, action_idx:action_idx+4]   # (num_envs, 4)
            action_idx += 4
            q = F.normalize(q, p=2, dim=1)

            # Unpack quaternion
            qw = q[:, 0]
            qv = q[:, 1:]  # (num_envs, 3)

            sign = torch.where(qw < 0.0, -torch.ones_like(qw), torch.ones_like(qw))
            qw = qw * sign
            qv = qv * sign.unsqueeze(-1)

            # Compute exponential map
            angle = 2.0 * torch.acos(qw.clamp(-1.0, 1.0))            # [0, π]
            sin_half = torch.sqrt((1.0 - qw * qw).clamp(min=1e-12))  # (num_envs,)
            axis = qv / sin_half.unsqueeze(-1)                       # (num_envs,3)

            expmap = axis * angle.unsqueeze(-1) # (num_envs,3)

            # Wrap to [-π, π]
            expmap = ((expmap + torch.pi) % (2.0 * torch.pi)) - torch.pi
            expmap = torch.where(torch.isfinite(expmap), expmap, torch.zeros_like(expmap))

            joint_targets[:, dof_idx:dof_idx+3] = expmap
            dof_idx += 3

        # Revolute joints: 1D angle target directly
        for _ in revolute_joints:
            # Wrap to [-π, π]
            angle = self.actions[:, action_idx]
            wrapped = ((angle + torch.pi) % (2.0 * torch.pi)) - torch.pi
            joint_targets[:, dof_idx] = torch.where(torch.isfinite(wrapped), wrapped, torch.zeros_like(wrapped))
            dof_idx += 1
            action_idx += 1

        self.humanoid.data.joint_pos_target = joint_targets

    def _get_dones(self) -> dict:
        """Get termination flags from the environment."""
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        ref_end = (~self.is_cyclic) & (self.episode_step >= self.clip_length)
        dones |= ref_end
        
        early_term = self._check_invalid_ground_contact()
        dones |= early_term

        timeout = self.is_cyclic & (self.episode_step >= 500)

        return dones, timeout
    
    def _check_invalid_ground_contact(self) -> torch.Tensor:
        """Check for invalid ground contact (e.g., torso hits ground while walking)."""

        body_states = self.humanoid.data.body_state_w  # (num_envs, 15, 13)
        positions = body_states[:, :, :3]  # (num_envs, 15, 3)        
        ground_height = 0.0
        
        invalid_contact_indices = self._get_link_indices(["torso", "head", "pelvis"])        
        invalid_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for idx in invalid_contact_indices:
            if idx < positions.shape[1]:  # Ensure index is valid
                # Check if this link is too close to ground
                link_too_low = positions[:, idx, 2] < ground_height
                invalid_contact |= link_too_low
        
        invalid_contact &= (self.episode_step > 10)
        return invalid_contact
    
    def _create_body_mapping(self) -> list:
        """Create body mapping based on actual articulation body names."""
        body_names = self.humanoid.body_names        
        body_mapping = [0] * len(body_names)
        
        for body_name in body_names:
            body_idx = self._get_link_indices([body_name])[0]
            body_mapping[body_idx] = body_name
        
        return body_mapping

    def _get_link_indices(self, link_names: List[str]) -> List[int]:
        """Get body state indices for given link names."""
        # Use the articulation's find_bodies method to get actual indices
        indices, _ = self.humanoid.find_bodies(link_names, preserve_order=True)
        return indices

    # def render(self, recompute: bool = True):
    #     if self.render_mode == "rgb_array":
    #         self.sim.render()
    #         self.tiled_camera.update(dt=1 / 30, force_recompute=recompute) 
    #         return self.get_rgb_frame().cpu().numpy()
    #     else:
    #         return super().render(recompute=recompute)
        
    def get_rgb_frame(self):
        rgb_data = self.tiled_camera.data.output["rgb"]
        return rgb_data[0]