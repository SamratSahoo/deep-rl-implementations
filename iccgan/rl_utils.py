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
        return random.choices(self.buffer, k=k)

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
        """
        Samples k samples from the buffers. The buffers are sampled based on the done_group.
        The done_group is a tensor of boolean values, where True indicates which buffers we want to sample from.
        Each buffer is sampled with a number of samples equal to (k // number) of True values in done_group.
        This method should only use vectorized operations. No for loops or list comprehensions allowed
        """
        # Convert mask to numpy boolean array
        if torch.is_tensor(done_group):
            done_array = done_group.detach().cpu().numpy().astype(bool)
        else:
            done_array = np.array(done_group, dtype=bool)

        selected_indices = np.where(done_array)[0]
        # Filter out buffers with zero stored sequences
        non_empty_mask = np.frompyfunc(lambda idx: len(self.buffers[int(idx)].buffer) > 0, 1, 1)(selected_indices)
        if isinstance(non_empty_mask, np.ndarray):
            non_empty_mask = non_empty_mask.astype(bool)
        else:
            non_empty_mask = np.array(non_empty_mask, dtype=bool)
        selected_indices = selected_indices[non_empty_mask]
        num_selected = int(selected_indices.shape[0])

        if num_selected == 0:
            raise ValueError("No buffers selected by done_group to sample from.")

        samples_per_buffer = k // num_selected
        if samples_per_buffer <= 0:
            raise ValueError("Requested k is too small relative to number of selected buffers.")

        # Sample sequences from each selected buffer using vectorized ufunc
        sampled_per_buffer = np.frompyfunc(
            lambda idx: self.buffers[int(idx)].sample(samples_per_buffer), 1, 1
        )(selected_indices)

        # Convert each per-buffer list to a 1D object array and stack into a 2D matrix, then flatten
        sampled_arrays = np.frompyfunc(lambda lst: np.array(lst, dtype=object), 1, 1)(sampled_per_buffer)
        if sampled_arrays.size == 0:
            raise ValueError("Sampling produced no sequences. Check buffer contents and k.")
        samples_matrix = np.stack(sampled_arrays.tolist(), axis=0)  # shape: (num_selected, samples_per_buffer)
        flat_samples = samples_matrix.reshape(-1)  # shape: (num_selected * samples_per_buffer,)
        print(type(flat_samples[0]), flat_samples[0])
        if flat_samples.size == 0:
            raise ValueError("Sampling produced no sequences. Check buffer contents and k.")

        # Helper to extract a stacked tensor for a specific field index from each sequence
        def stack_field_from_sequence(sequence, field_index):
            # Normalize to a list of timesteps
            if isinstance(sequence, tuple) and len(sequence) == 5:
                timesteps = [sequence]
            elif isinstance(sequence, list):
                timesteps = sequence
            else:
                arr = np.array(sequence, dtype=object)
                if arr.ndim == 0:
                    timesteps = [arr.item()]
                else:
                    timesteps = arr.tolist()

            ts_arr = np.array(timesteps, dtype=object)
            # Extract the specified field for each timestep; if an element isn't indexable, treat it as already the field
            extractor = np.frompyfunc(
                lambda ts: ts[field_index] if isinstance(ts, (tuple, list)) else ts,
                1,
                1,
            )
            field_values = extractor(ts_arr)
            return np.stack(field_values.tolist(), axis=0)

        # Vectorized extraction for all sequences
        states_per_seq = np.frompyfunc(lambda s: stack_field_from_sequence(s, 0), 1, 1)(flat_samples)
        actions_per_seq = np.frompyfunc(lambda s: stack_field_from_sequence(s, 1), 1, 1)(flat_samples)
        values_per_seq = np.frompyfunc(lambda s: stack_field_from_sequence(s, 2), 1, 1)(flat_samples)
        advantages_per_seq = np.frompyfunc(lambda s: stack_field_from_sequence(s, 3), 1, 1)(flat_samples)
        log_probs_per_seq = np.frompyfunc(lambda s: stack_field_from_sequence(s, 4), 1, 1)(flat_samples)

        # Stack across sequences to form final arrays
        states = np.stack(states_per_seq.tolist(), axis=0)
        actions = np.stack(actions_per_seq.tolist(), axis=0)
        values = np.stack(values_per_seq.tolist(), axis=0)
        advantages = np.stack(advantages_per_seq.tolist(), axis=0)
        log_probs = np.stack(log_probs_per_seq.tolist(), axis=0)

        # Convert to torch tensors (CPU, float32)
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        log_probs_t = torch.tensor(log_probs, dtype=torch.float32)

        return states_t, actions_t, values_t, advantages_t, log_probs_t

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