import json
import torch
from utils import quat2axis_angle, quatdiff, quatdiff_rel
import math

class ReferenceMotion:
    def __init__(self, motion_file: str):
        self.motion_file = motion_file
        self.motion_data = None
        self.contactable_links = []
        self.is_cyclic = False
        self.clip_length = 0

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
        return (
            torch.full(
                (self.num_envs,),
                500,
                dtype=torch.int64,
                device=self.device,
            )
            if self.is_cyclic
            else torch.full(
                (self.num_envs,),
                len(self.frames),
                dtype=torch.int64,
                device=self.device,
            )
        )

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
        
        Links mapping:
        0: pelvis
        1: abdomen
        2: chest
        3: neck
        4: head
        5: right_hip
        6: right_knee
        7: right_ankle
        8: right_foot
        9: right_shoulder
        10: left_hip
        11: left_knee
        12: left_ankle
        13: left_foot
        14: left_shoulder
        """
        state_vector = []
        
        # Define the 15 links and their corresponding motion data
        link_mapping = [
            ("pelvis", "base_position", "base_orientation"),
            ("abdomen", None, "abdomen"),
            ("chest", None, "abdomen"), # Use abdomen data for chest
            ("neck", None, "neck"),
            ("head", None, "head"),
            ("right_hip", None, "right_hip"),
            ("right_knee", None, "right_knee"),
            ("right_ankle", None, "right_ankle"),
            ("right_foot", None, "right_ankle"), # Use ankle data for foot
            ("right_shoulder", None, "right_shoulder"),
            ("left_hip", None, "left_hip"),
            ("left_knee", None, "left_knee"),
            ("left_ankle", None, "left_ankle"),
            ("left_foot", None, "left_ankle"),    # Use ankle data for foot
            ("left_shoulder", None, "left_shoulder"),
        ]
        
        for link_name, pos_key, orient_key in link_mapping:
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

    def sample(self, num_samples, sample_length):
        pass

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
