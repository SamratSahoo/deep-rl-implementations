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
        pos=(0.0, 0.0, 1.5)),
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
        def to_1d_array(x):
            arr = np.asarray(x)
            if arr.ndim == 0:
                return arr.reshape(1)
            if arr.ndim > 1:
                return arr.reshape(-1)
            return arr

        s = to_1d_array(state)
        a = to_1d_array(action)
        v = to_1d_array(value)
        adv = to_1d_array(advantage)
        lp = to_1d_array(log_prob)

        # Append until we reach the desired sequence length
        if len(self.minibuffer) < self.sequence_length:
            self.minibuffer.append((s, a, v, adv, lp))
        else:
            # Flush the completed sequence (as stacked arrays) and start a new one with current timestep
            # Transpose steps -> fields and stack per field to shape (T, feat)
            fields = list(zip(*self.minibuffer))  # 5 tuples, each length T
            seq_states = np.stack(fields[0], axis=0)
            seq_actions = np.stack(fields[1], axis=0)
            seq_values = np.stack(fields[2], axis=0)
            seq_advantages = np.stack(fields[3], axis=0)
            seq_log_probs = np.stack(fields[4], axis=0)
            self.buffer.append((seq_states, seq_actions, seq_values, seq_advantages, seq_log_probs))
            self.minibuffer = [(s, a, v, adv, lp)]

    def clear_minibuffer(self):
        self.minibuffer = []
    
    def sample(self, k):
        """Return stacked arrays per field with shapes (k, T, feat)."""
        choices = random.choices(self.buffer, k=k)
        # Fast field-wise stacking without explicit for-loops over k
        fields = list(zip(*choices))  # 5 tuples, each length k
        states = np.stack(fields[0], axis=0)
        actions = np.stack(fields[1], axis=0)
        values = np.stack(fields[2], axis=0)
        advantages = np.stack(fields[3], axis=0)
        log_probs = np.stack(fields[4], axis=0)
        return states, actions, values, advantages, log_probs

class DiscriminatorBufferGroup:
    def __init__(self, count=2, capacity=10000, sequence_length=5):
        self.buffers = [DiscriminatorBuffer(capacity=capacity, sequence_length=sequence_length) for _ in range(count)]
        self.count = count
        self.sequence_length = sequence_length
    
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
        """
        Samples k sequences from buffers indicated by done_group. Returns tensors of shape:
        - states: (B, sequence_length, obs_dim)
        - actions: (B, sequence_length, action_dim)
        - values: (B, sequence_length, 1)
        - advantages: (B, sequence_length, 1)
        - log_probs: (B, sequence_length, 1)
        """

        if torch.is_tensor(done_group):
            done_array = done_group.detach().cpu().numpy().astype(bool)
        else:
            done_array = np.array(done_group, dtype=bool)

        candidate_indices = np.where(done_array)[0]
        if candidate_indices.size == 0:
            return (
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )

        buffer_lengths = np.frompyfunc(lambda idx: len(self.buffers[int(idx)].buffer), 1, 1)(candidate_indices)
        buffer_lengths = buffer_lengths.astype(int)
        selected_mask = buffer_lengths > 0
        selected_indices = candidate_indices[selected_mask]
        num_selected = int(selected_indices.shape[0])

        if num_selected == 0:
            return (
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.float32),
            )

        samples_per_buffer = k // num_selected
        # (k_i, T, F)
        states_list, actions_list, values_list, advantages_list, log_probs_list = np.frompyfunc(
            lambda idx: self.buffers[int(idx)].sample(samples_per_buffer), 1, 5
        )(selected_indices)

        states = np.concatenate(states_list.tolist(), axis=0)
        actions = np.concatenate(actions_list.tolist(), axis=0)
        values = np.concatenate(values_list.tolist(), axis=0)
        advantages = np.concatenate(advantages_list.tolist(), axis=0)
        log_probs = np.concatenate(log_probs_list.tolist(), axis=0)

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(log_probs, dtype=torch.float32),
        )

class FeatureNormalizer:
    """Normalizes features using running mean and standard deviation."""
    
    def __init__(self, feature_dim, epsilon=1e-8, decay=0.999):
        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.decay = decay
        
        self.running_mean = torch.zeros(feature_dim, dtype=torch.float32)
        self.running_var = torch.ones(feature_dim, dtype=torch.float32)
        self.running_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def update(self, features):
        """Update running statistics with new features."""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.device)
        
        if features.dim() == 1:
            features = features.unsqueeze(0) # (batch_size, feature_dim)
        elif features.dim() > 2:
            features = features.reshape(-1, features.shape[-1]) # Flatten
        
        batch_size = features.shape[0]
        
        if self.running_count == 0: # First batch
            self.running_mean = features.mean(dim=0).to(self.device)
            self.running_var = features.var(dim=0, unbiased=False).to(self.device)
        else: # Running batches
            batch_mean = features.mean(dim=0).to(self.device)
            batch_var = features.var(dim=0, unbiased=False).to(self.device)
            
            self.running_mean = self.decay * self.running_mean + (1 - self.decay) * batch_mean
            self.running_var = self.decay * self.running_var + (1 - self.decay) * batch_var
        
        self.running_count += batch_size
    
    def normalize(self, features):
        """Normalize features using current running statistics."""
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.device)
        
        original_shape = features.shape
        
        if features.dim() == 1: 
            features = features.unsqueeze(0) # (batch_size, feature_dim)
        elif features.dim() > 2:
            features = features.reshape(-1, features.shape[-1]) # Flatten
        
        normalized = (features - self.running_mean) / (torch.sqrt(self.running_var) + self.epsilon)
        normalized = normalized.reshape(original_shape)
        
        return normalized
    
    def to(self, device):
        """Move normalizer to specified device."""
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        return self