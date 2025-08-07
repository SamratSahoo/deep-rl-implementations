from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg
import os
import json
from typing import Dict, List, Optional
from quat_utils import quaternion_to_euler
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
import isaacsim.core.utils.articulations as articulations_utils
import isaacsim.core.utils.prims as prims_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import UsdFileCfg
import omni.physics.tensors as tensors
import omni.kit.commands
import omni.usd
from pxr import UsdPhysics
from reference_motion import ReferenceMotion
from isaacsim.core.simulation_manager import SimulationManager

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
            self.contactable_links = self.reference_motion.contactable_links
            self.clip_length = torch.full((self.num_envs,), self.reference_motion.clip_length, device=self.device)
            self.is_cyclic = torch.full((self.num_envs,), 1 if self.reference_motion.is_cyclic else 0, dtype=torch.bool, device=self.device)

        # Reset episode step counter for the specified environments
        self.episode_step[env_ids] = 0
        
        # Call parent reset method first
        super()._reset_idx(env_ids)
        
        # Get default joint states from the articulation
        joint_pos = self.humanoid.data.default_joint_pos[env_ids]
        joint_vel = self.humanoid.data.default_joint_vel[env_ids]
        
        # Get default root state
        default_root_state = self.humanoid.data.default_root_state[env_ids].clone()
        # Apply environment origins to root position
        default_root_state[:, :3] += self.scene.env_origins[env_ids].to(self.device)
        
        # Write the reset states to simulation
        self.humanoid.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.humanoid.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.humanoid.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    
    def _get_rewards(self) -> torch.Tensor:
        """Get rewards from the environment."""
        # TODO: Implement proper rewards
        # For now, return zero rewards
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
        
        num_joints = self.humanoid.data.joint_pos_target.shape[1]  # total DOFs
        joint_targets = torch.zeros((self.num_envs, num_joints), device=self.device)

        action_idx = 0
        dof_idx = 0

        # --- Spherical joints: convert 4D quaternion → 3D expmap ---
        for _ in spherical_joint_groups:
            q = self.actions[:, action_idx:action_idx+4]   # (num_envs, 4)
            action_idx += 4

            # Unpack quaternion; adjust ordering if yours is (x,y,z,w)
            qw = q[:, 0]
            qv = q[:, 1:]  # (num_envs, 3)

            # Compute exponential map (axis-angle vector)
            angle = 2.0 * torch.acos(qw.clamp(-1,1))             # (num_envs,)
            sin_half = torch.sqrt((1 - qw*qw).clamp(min=1e-6))   # (num_envs,)
            axis = qv / sin_half.unsqueeze(-1)                   # (num_envs,3)

            expmap = axis * angle.unsqueeze(-1)                  # (num_envs,3)

            # Fill in those 3 DOF slots
            joint_targets[:, dof_idx:dof_idx+3] = expmap
            dof_idx += 3

        # --- Revolute joints: 1D angle target directly ---
        for _ in revolute_joints:
            joint_targets[:, dof_idx] = self.actions[:, action_idx]
            dof_idx += 1
            action_idx += 1

        # Finally assign to the articulation buffer
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
                link_too_low = positions[:, idx, 2] < ground_height + 0.15  # 15cm threshold
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