from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg
import os
import numpy as np
from collections import deque
import random
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

class DiscriminatorBuffer:
    def __init__(self, capacity, sequence_length=5):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        self.minibuffer = []
        self.sequence_length = sequence_length

    def push(self, state, action, value, advantage, log_prob):
        if len(self.minibuffer) < self.sequence_length:
            self.minibuffer.append((state, action, value, advantage, log_prob))
        else:
            self.buffer.append(self.minibuffer)
            self.minibuffer = []

    def clear_minibuffer(self):
        self.minibuffer = []
    
    def sample(self, k):
        return random.sample(self.buffer, k)

class DiscriminatorBufferGroup:
    def __init__(self, count=2, capacity=10000, sequence_length=5):
        self.buffers = [DiscriminatorBuffer(capacity=capacity, sequence_length=sequence_length) for _ in range(count)]
        self.count = count
    
    def push(self, state_group, action_group, value_group, advantage_group, log_prob_group):
        states = np.array(state_group)
        actions = np.array(action_group)
        values = np.array(value_group)
        advantages = np.array(advantage_group)
        log_probs = np.array(log_prob_group)
        
        buffer_indices = np.arange(self.count)
        np.frompyfunc(lambda idx: self.buffers[idx].push(
            states[idx], actions[idx], values[idx], advantages[idx], log_probs[idx]
        ), 1, 0)(buffer_indices)
    
    def clear_minibuffer(self, done_group):
        if torch.is_tensor(done_group):
            done_array = done_group.cpu().numpy()
        else:
            done_array = np.array(done_group, dtype=bool)
        done_indices = np.where(done_array)[0]
        
        if len(done_indices) > 0:
            np.frompyfunc(lambda idx: self.buffers[idx].clear_minibuffer(), 1, 0)(done_indices)
    
    def sample(self, k, done_group):
        done_array = np.array(done_group.cpu().numpy(), dtype=bool)
        done_indices = np.where(done_array)[0]

        if len(done_indices) > 0:
            samples_per_buffer = max(1, k // len(done_indices))            
            sample_func = np.frompyfunc(lambda idx: self.buffers[idx].sample(samples_per_buffer), 1, 1)
            sampled_arrays = sample_func(done_indices)
            
            sampled_data = np.concatenate(sampled_arrays)
            
            data_array = np.array(sampled_data, dtype=object)
            structured_data = np.array(data_array, dtype=[
                ('state', object), ('action', object), ('value', object), 
                ('advantage', object), ('log_prob', object)
            ])
            
            # Use numpy's field access for vectorized extraction
            states = np.array(structured_data['state'])
            actions = np.array(structured_data['action'])
            values = np.array(structured_data['value'])
            advantages = np.array(structured_data['advantage'])
            log_probs = np.array(structured_data['log_prob'])
            
            return (torch.tensor(states), torch.tensor(actions), 
                    torch.tensor(values), torch.tensor(advantages), 
                    torch.tensor(log_probs))            
        else:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)