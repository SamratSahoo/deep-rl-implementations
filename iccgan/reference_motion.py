import json
import torch
from quat_utils import quat2axis_angle, quatdiff, quatdiff_rel
import math
import random

class ReferenceMotion:
    def __init__(self, motion_file: str, body_mapping: list = None):
        self.motion_file = motion_file
        self.motion_data = None
        self.body_mapping = body_mapping

        with open(motion_file, "r") as f:
            self.motion_data = json.load(f)
        
        self.frame_states = []

    @property
    def raw_frames(self):
        return self.motion_data["frames"]

    @property
    def contactable_links(self):
        return self.motion_data["contactable_links"]

    @property
    def is_cyclic(self):
        return self.motion_data["loopable"] > 0

    @property
    def dt(self):
        return self.motion_data["sampling_interval"]

    @property
    def clip_length(self):
        return 500 if self.is_cyclic else len(self.raw_frames)

    def parse_frames(self):
        """Parse raw frames and compute velocities for each joint."""
        for frame_index in range(len(self.raw_frames)-1):
            current_frame = self.raw_frames[frame_index]
            next_frame = self.raw_frames[frame_index+1]
            processed_frame = {}
            
            for joint in current_frame:
                joint_data = current_frame[joint]
                next_joint_data = next_frame[joint]
                
                processed_frame[joint] = joint_data
                
                # Compute velocities based on data type
                if len(joint_data) == 3 or len(joint_data) == 1:  # Position data (x, y, z)
                    processed_frame[joint + "_linvel"] = ReferenceMotion.linear_vel(joint_data, next_joint_data, self.dt)
                elif len(joint_data) == 4:  # Orientation data (qx, qy, qz, qw)
                    processed_frame[joint + "_angvel"] = ReferenceMotion.angular_vel(joint_data, next_joint_data, self.dt)
            
            self.frame_states.append(self.get_state_from_frame(processed_frame))
    
    def get_state_from_frame(self, processed_frame):
        """
        Convert processed frame data to observation vector.
        
        15 Links x 13 Observations per link = 195 observation space
        Each link has: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        """
        state_vector = []
        link_mapping = {
            "torso": ("torso", "base_position", "base_orientation"),
            "pelvis": ("pelvis", None, "pelvis"),
            "head": ("head", None, "head"),
            "right_upper_arm": ("right_upper_arm", None, "right_shoulder"),
            "left_upper_arm": ("left_upper_arm", None, "left_shoulder"),
            "right_thigh": ("right_thigh", None, "right_hip"),
            "left_thigh": ("left_thigh", None, "left_hip"),
            "right_lower_arm": ("right_lower_arm", None, "right_elbow"),
            "left_lower_arm": ("left_lower_arm", None, "left_elbow"),
            "right_shin": ("right_shin", None, "right_knee"),
            "left_shin": ("left_shin", None, "left_knee"),
            "right_hand": ("right_hand", None, "right_wrist"),
            "left_hand": ("left_hand", None, "left_wrist"),
            "right_foot": ("right_foot", None, "right_ankle"),
            "left_foot": ("left_foot", None, "left_ankle"),
        }
        
        for body in self.body_mapping:
            link_name, pos_key, orient_key = link_mapping[body]
            # Initialize with zeros
            pos = [0.0, 0.0, 0.0]
            orient = [0.0, 0.0, 0.0, 1.0]  # Identity quaternion
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            
            # Get position data (only for pelvis/base)
            if pos_key and pos_key in processed_frame:
                pos = processed_frame[pos_key]
                if pos_key + "_linvel" in processed_frame:
                    linvel = processed_frame[pos_key + "_linvel"]
            
            # Get orientation/angle data
            if orient_key and orient_key in processed_frame:
                orient_data = processed_frame[orient_key]
                if len(orient_data) == 4:  # Quaternion
                    orient = orient_data
                    if orient_key + "_angvel" in processed_frame:
                        angvel = processed_frame[orient_key + "_angvel"]
                elif len(orient_data) == 1:  # Single angle
                    # Convert single angle to quaternion (assuming rotation around z-axis)
                    angle = orient_data[0]
                    orient = [0.0, 0.0, math.sin(angle/2), math.cos(angle/2)]
                    if orient_key + "_angvel" in processed_frame:
                        angvel = processed_frame[orient_key + "_angvel"]
            
            # Combine into 13-dimensional observation: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
            link_obs = pos + orient + linvel + angvel
            state_vector.extend(link_obs)
        
        return state_vector

    def sample(self, num_samples, sample_length=5):
        if len(self.frame_states) < sample_length:
            return []
        
        max_start_idx = len(self.frame_states) - sample_length
        samples = []
        
        for _ in range(num_samples):
            start_idx = random.randint(0, max_start_idx)
            sample = self.frame_states[start_idx:start_idx + sample_length]
            samples.append(sample)
        
        return samples

    @staticmethod
    def linear_vel(p0, p1, delta_t):
        if hasattr(p0, "__len__"):
            return [(v1-v0)/delta_t for v0, v1 in zip(p0, p1)]
        return (p1-p0)/delta_t
    
    @staticmethod
    def angular_vel(q0, q1, delta_t):
        axis, angle = quat2axis_angle(quatdiff(q0, q1))
        angle /= delta_t
        return [angle*a for a in axis]
    
    @staticmethod
    def angular_vel_rel(q0, q1, delta_t):
        axis, angle = quat2axis_angle(quatdiff_rel(q0, q1))
        angle /= delta_t
        return [angle*a for a in axis]

if __name__ == "__main__":
    motion_file = "assets/motions/run.json"
    reference_motion = ReferenceMotion(motion_file)
    reference_motion.parse_frames()
    print(torch.tensor(reference_motion.frame_states).shape)