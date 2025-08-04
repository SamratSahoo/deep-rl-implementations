from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg
import os
import torch


HUMANOID_CONFIG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{os.path.dirname(os.path.abspath(__file__))}/assets/humanoid.usd",        
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0)),
    prim_path="/World/envs/env_.*/humanoid",
    actuators={
        "abdomen_x": ImplicitActuatorCfg(
            joint_names_expr=["abdomen_x"],
            effort_limit_sim=125,
            stiffness=600,
            damping=60,
            armature=0.025,
        ),
        "abdomen_y": ImplicitActuatorCfg(
            joint_names_expr=["abdomen_y"],
            effort_limit_sim=125,
            stiffness=600,
            damping=60,
            armature=0.025,
        ),
        "abdomen_z": ImplicitActuatorCfg(
            joint_names_expr=["abdomen_z"],
            effort_limit_sim=125,
            stiffness=600,
            damping=60,
            armature=0.025,
        ),
        "neck_x": ImplicitActuatorCfg(
            joint_names_expr=["neck_x"],
            effort_limit_sim=20,
            stiffness=50,
            damping=5,
            armature=0.017,
        ),
        "neck_y": ImplicitActuatorCfg(
            joint_names_expr=["neck_y"],
            effort_limit_sim=20,
            stiffness=50,
            damping=5,
            armature=0.017,
        ),
        "neck_z": ImplicitActuatorCfg(
            joint_names_expr=["neck_z"],
            effort_limit_sim=20,
            stiffness=50,
            damping=5,
            armature=0.017,
        ),
        "right_shoulder_x": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_x"],
            effort_limit_sim=70,
            stiffness=200,
            damping=20,
            armature=0.02,
        ),
        "right_shoulder_y": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_y"],
            effort_limit_sim=70,
            stiffness=200,
            damping=20,
            armature=0.02,
        ),
        "right_shoulder_z": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder_z"],
            effort_limit_sim=70,
            stiffness=200,
            damping=20,
            armature=0.02,
        ),
        "right_elbow": ImplicitActuatorCfg(
            joint_names_expr=["right_elbow"],
            effort_limit_sim=60,
            stiffness=150,
            damping=15,
            armature=0.015,
        ),
        "left_shoulder_x": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_x"],
            effort_limit_sim=70,
            stiffness=200,
            damping=20,
            armature=0.02,
        ),
        "left_shoulder_y": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_y"],
            effort_limit_sim=70,
            stiffness=200,
            damping=20,
            armature=0.02,
        ),
        "left_shoulder_z": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder_z"],
            effort_limit_sim=70,
            stiffness=200,
            damping=20,
            armature=0.02,
        ),
        "left_elbow": ImplicitActuatorCfg(
            joint_names_expr=["left_elbow"],
            effort_limit_sim=60,
            stiffness=150,
            damping=15,
            armature=0.015,
        ),
        "right_hip_x": ImplicitActuatorCfg(
            joint_names_expr=["right_hip_x"],
            effort_limit_sim=125,
            stiffness=300,
            damping=30,
            armature=0.02,
        ),
        "right_hip_z": ImplicitActuatorCfg(
            joint_names_expr=["right_hip_z"],
            effort_limit_sim=125,
            stiffness=300,
            damping=30,
            armature=0.02,
        ),
        "right_hip_y": ImplicitActuatorCfg(
            joint_names_expr=["right_hip_y"],
            effort_limit_sim=125,
            stiffness=300,
            damping=30,
            armature=0.02,
        ),
        "right_knee": ImplicitActuatorCfg(
            joint_names_expr=["right_knee"],
            effort_limit_sim=100,
            stiffness=300,
            damping=30,
            armature=0.02,
        ),
        "right_ankle_x": ImplicitActuatorCfg(
            joint_names_expr=["right_ankle_x"],
            effort_limit_sim=50,
            stiffness=200,
            damping=20,
            armature=0.01,
        ),
        "right_ankle_y": ImplicitActuatorCfg(
            joint_names_expr=["right_ankle_y"],
            effort_limit_sim=50,
            stiffness=200,
            damping=20,
            armature=0.01,
        ),
        "right_ankle_z": ImplicitActuatorCfg(
            joint_names_expr=["right_ankle_z"],
            effort_limit_sim=50,
            stiffness=200,
            damping=20,
            armature=0.01,
        ),
        "left_hip_x": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_x"],
            effort_limit_sim=125,
            stiffness=300,
            damping=30,
            armature=0.02,
        ),
        "left_hip_z": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_z"],
            effort_limit_sim=125,
            stiffness=300,
            damping=30,
            armature=0.02,
        ),
        "left_hip_y": ImplicitActuatorCfg(
            joint_names_expr=["left_hip_y"],
            effort_limit_sim=125,
            stiffness=300,
            damping=30,
            armature=0.02,
        ),
        "left_knee": ImplicitActuatorCfg(
            joint_names_expr=["left_knee"],
            effort_limit_sim=100,
            stiffness=300,
            damping=30,
            armature=0.02,
        ),
        "left_ankle_x": ImplicitActuatorCfg(
            joint_names_expr=["left_ankle_x"],
            effort_limit_sim=50,
            stiffness=200,
            damping=20,
            armature=0.01,
        ),
        "left_ankle_y": ImplicitActuatorCfg(
            joint_names_expr=["left_ankle_y"],
            effort_limit_sim=50,
            stiffness=200,
            damping=20,
            armature=0.01,
        ),
        "left_ankle_z": ImplicitActuatorCfg(
            joint_names_expr=["left_ankle_z"],
            effort_limit_sim=50,
            stiffness=200,
            damping=20,
            armature=0.01,
        ),
    },
)

def quaternion_to_euler(quat):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Extract quaternion components
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Convert to Euler angles (roll, pitch, yaw)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        # Check for gimbal lock
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.pi / 2,
            torch.asin(sinp)
        )
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=1)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)